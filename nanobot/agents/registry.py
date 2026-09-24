"""Resolve the agents declared in one config into independent runtime settings.

The single agent nanobot has always run is the agent named ``default``: it is
configured by ``agents.defaults`` plus the top-level ``tools`` block. Agents
declared under ``agents.named`` are full peers of it, each with its own
workspace, sessions, model settings and tool set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic.alias_generators import to_camel

from nanobot.config.loader import merge_missing_defaults
from nanobot.config.schema import (
    DEFAULT_AGENT_NAME,
    AgentDefaults,
    AgentsConfig,
    Config,
    NamedAgentConfig,
    ToolsConfig,
)

__all__ = [
    "DEFAULT_AGENT_NAME",
    "NAMED_AGENT_WORKSPACE_ROOT",
    "AgentSpec",
    "agent_config",
    "agent_names",
    "agent_spec",
    "named_agent_workspace",
    "resolve_agent_specs",
]

NAMED_AGENT_WORKSPACE_ROOT = "~/.nanobot/agents"


@dataclass(frozen=True)
class AgentSpec:
    """One agent's effective settings, resolved from the config file."""

    name: str
    settings: AgentDefaults
    tools: ToolsConfig
    workspace: Path
    model: str
    # None means "the ambient runtime sessions root", which only the default
    # agent of a config loaded from no file ever resolves to.
    sessions_root: Path | None = None

    @property
    def is_default(self) -> bool:
        return self.name == DEFAULT_AGENT_NAME


def named_agent_workspace(name: str) -> Path:
    """Return the workspace a named agent gets when it declares none."""
    return (Path(NAMED_AGENT_WORKSPACE_ROOT) / name).expanduser()


def agent_names(config: Config) -> list[str]:
    """Return every declared agent name, ``default`` first."""
    return [DEFAULT_AGENT_NAME, *config.agents.named]


def _camelize(values: dict[str, Any]) -> dict[str, Any]:
    """Normalize override keys so they land on the same key as the defaults dump."""
    return {to_camel(key): value for key, value in values.items()}


def _resolve_settings(
    defaults: AgentDefaults,
    name: str,
    entry: NamedAgentConfig,
) -> AgentDefaults:
    """Lay one named agent's own entries over ``agents.defaults``."""
    base = defaults.model_dump(mode="json", by_alias=True)
    overrides = _camelize(entry.setting_overrides())
    # A declared timezone is a manual choice even when the default agent is on
    # auto-detection, which would otherwise overwrite it during revalidation.
    if "timezone" in overrides and "timezoneMode" not in overrides:
        overrides["timezoneMode"] = "manual"
    merged = {**base, **overrides}
    # A named agent never inherits the default agent's workspace.
    merged["workspace"] = entry.workspace or str(named_agent_workspace(name))
    return AgentDefaults.model_validate(merged)


def _mcp_server_overrides(raw_tools: dict[str, Any]) -> dict[str, Any]:
    for key in ("mcpServers", "mcp_servers"):
        servers = raw_tools.get(key)
        if isinstance(servers, dict):
            return cast(dict[str, Any], servers)
    return {}


def _resolve_tools(top_level: ToolsConfig, raw_tools: dict[str, Any] | None) -> ToolsConfig:
    """Lay a named agent's tools block over the top-level one.

    MCP servers are the exception: they are never inherited, so an agent has
    exactly the servers listed in its own ``tools.mcpServers``.
    """
    base = top_level.model_dump(mode="json", by_alias=True)
    base.pop("mcpServers", None)
    raw = dict(raw_tools or {})
    merged = cast(dict[str, Any], merge_missing_defaults(raw, base))
    merged.pop("mcp_servers", None)
    merged["mcpServers"] = _mcp_server_overrides(raw)
    return ToolsConfig.model_validate(merged)


def _sessions_root(config: Config, name: str) -> Path | None:
    """Return where one agent's sessions live, isolated from every other agent's.

    Session storage must stay outside the agent's workspace (ADR-0001), so each
    agent gets its own subtree of the runtime data root rather than a directory
    under its workspace. ``None`` keeps the default agent on the ambient runtime
    root, exactly as before named agents existed.
    """
    data_dir = config.runtime_data_dir
    if name == DEFAULT_AGENT_NAME:
        return data_dir / "sessions" if data_dir is not None else None
    if data_dir is not None:
        return data_dir / "agents" / name / "sessions"
    from nanobot.config.paths import get_runtime_subdir

    return get_runtime_subdir("agents") / name / "sessions"


def _build_spec(config: Config, name: str) -> AgentSpec:
    if name == DEFAULT_AGENT_NAME:
        settings = config.agents.defaults
        tools = config.tools
    else:
        entry = config.agents.named[name]
        settings = _resolve_settings(config.agents.defaults, name, entry)
        tools = _resolve_tools(config.tools, entry.tools)

    workspace = Path(settings.workspace).expanduser()
    return AgentSpec(
        name=name,
        settings=settings,
        tools=tools,
        workspace=workspace,
        model=_resolved_model(config, settings),
        sessions_root=_sessions_root(config, name),
    )


def _resolved_model(config: Config, settings: AgentDefaults) -> str:
    """Return the model an agent runs, honoring its preset selection."""
    preset_name = settings.model_preset
    if preset_name and preset_name != DEFAULT_AGENT_NAME:
        preset = config.model_presets.get(preset_name)
        if preset is not None:
            return preset.model
    return settings.model


def agent_spec(config: Config, name: str = DEFAULT_AGENT_NAME) -> AgentSpec:
    """Return one agent's resolved settings.

    Raises:
        KeyError: if no agent by that name is declared.
    """
    if name != DEFAULT_AGENT_NAME and name not in config.agents.named:
        raise KeyError(f"no agent named {name!r} is declared in agents.named")
    return _build_spec(config, name)


def resolve_agent_specs(config: Config) -> list[AgentSpec]:
    """Return every declared agent's resolved settings, ``default`` first."""
    return [_build_spec(config, name) for name in agent_names(config)]


def agent_config(config: Config, name: str = DEFAULT_AGENT_NAME) -> Config:
    """Return a config view scoped to one agent.

    The view's ``agents.defaults`` and ``tools`` are that agent's own, so every
    existing config-driven constructor (``AgentLoop.from_config``,
    ``MCPProvider.from_config``, …) builds the right runtime unchanged.
    """
    spec = agent_spec(config, name)
    view = config.model_copy(deep=True)
    view.agents = AgentsConfig(defaults=spec.settings)
    view.tools = spec.tools
    source = config.source_path
    if source is not None:
        view.bind_source_path(source)
    return view

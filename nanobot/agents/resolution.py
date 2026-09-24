"""Effective per-agent settings resolved from `agents.defaults` and `agents.named`."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nanobot.config.schema import (
    RESERVED_AGENT_NAME,
    AgentDefaults,
    Config,
    MCPServerConfig,
    NamedAgentConfig,
    ToolsConfig,
)

NAMED_AGENT_WORKSPACE_ROOT = "~/.nanobot/agents"


def named_agent_workspace(name: str) -> str:
    """Return the workspace a named agent owns when it configures none."""
    return f"{NAMED_AGENT_WORKSPACE_ROOT}/{name}"


def named_agent_entry(config: Config, name: str) -> NamedAgentConfig:
    """Return the ``agents.named`` entry for *name*.

    Raises:
        KeyError: if no agent is configured under that name, naming the ones
            that are.
    """
    entry = config.agents.named.get(name)
    if entry is None:
        known = ", ".join(sorted({RESERVED_AGENT_NAME, *config.agents.named}))
        raise KeyError(f"unknown agent {name!r}; configured agents: {known}")
    return entry


def _overlay(base: BaseModel, override: BaseModel) -> Any:
    """Return a copy of *base* with the fields *override* actually states laid over it.

    Only ``model_fields_set`` entries win, so a named agent that never mentioned
    a field keeps the configured default rather than the schema default.  Fields
    the override declares but the base does not (a `tools` block on a
    ``NamedAgentConfig``) are left to their own resolution step.
    """
    update: dict[str, Any] = {}
    for name in override.model_fields_set:
        if name not in type(base).model_fields:
            continue
        value = getattr(override, name)
        current = getattr(base, name, None)
        if isinstance(value, BaseModel) and isinstance(current, BaseModel):
            update[name] = _overlay(current, value)
        else:
            update[name] = deepcopy(value)
    # model_copy applies `update` after the deep copy without re-validating, so
    # the overlay keeps already-validated objects and no before-validator (e.g.
    # AgentDefaults.resolve_timezone) re-fires and rewrites a configured value.
    return base.model_copy(update=update, deep=True)


def _resolve_defaults(name: str, defaults: AgentDefaults, entry: NamedAgentConfig) -> AgentDefaults:
    """Lay one named agent's entry over `agents.defaults`."""
    merged: AgentDefaults = _overlay(defaults, entry)
    if "workspace" not in entry.model_fields_set:
        # Workspace is never inherited: two agents sharing a workspace would
        # share SOUL.md, USER.md and memory/, which is exactly what isolation
        # forbids.  An agent that configures none gets one of its own.
        merged.workspace = named_agent_workspace(name)
    if "timezone" in entry.model_fields_set and "timezone_mode" not in entry.model_fields_set:
        # AgentDefaults.resolve_timezone reads a stated timezone as manual mode;
        # apply the same rule to the overlay so an inherited "auto" cannot
        # re-detect over the agent's own timezone on the next load.
        merged.timezone_mode = "manual"
    if "model" in entry.model_fields_set and "model_preset" not in entry.model_fields_set:
        # An explicit model deselects an inherited preset, the same rule
        # Nanobot.from_config applies to a `--model` override.
        merged.model_preset = None
    return merged


def _resolve_tools(tools: ToolsConfig, entry: NamedAgentConfig) -> ToolsConfig:
    """Lay one named agent's tools block over the top-level tools block."""
    merged: ToolsConfig = _overlay(tools, entry.tools)
    # MCP servers are never inherited: an agent has exactly the servers in its
    # own tools.mcpServers, and the top-level block belongs to `default` alone.
    merged.mcp_servers = deepcopy(entry.tools.mcp_servers)
    return merged


@dataclass(frozen=True)
class ResolvedAgentConfig:
    """One agent's effective settings.

    ``config`` is a standalone :class:`Config` carrying those settings in the
    places the runtime already reads them — ``agents.defaults`` and ``tools`` —
    so an agent can be composed with the existing ``AgentLoop.from_config`` /
    ``MCPProvider.from_config`` entry points without any of them learning about
    named agents.  It is always a copy: mutating it cannot affect the source
    config or another agent.
    """

    name: str
    config: Config

    @property
    def is_default(self) -> bool:
        """Whether this is the default agent, which owns the top-level blocks."""
        return self.name == RESERVED_AGENT_NAME

    @property
    def agent(self) -> AgentDefaults:
        """The effective agent settings."""
        return self.config.agents.defaults

    @property
    def tools(self) -> ToolsConfig:
        """The effective tools block."""
        return self.config.tools

    @property
    def mcp_servers(self) -> dict[str, MCPServerConfig]:
        """The MCP servers this agent owns — never another agent's."""
        return self.config.tools.mcp_servers

    @property
    def workspace(self) -> Path:
        """The expanded workspace path this agent owns."""
        return self.config.workspace_path

    @property
    def model(self) -> str:
        """The effective model name, honouring a selected preset."""
        return self.config.resolve_preset().model


def resolve_agent_config(config: Config, name: str = RESERVED_AGENT_NAME) -> ResolvedAgentConfig:
    """Return the effective settings of the agent called *name*.

    ``default`` resolves to the config as it stands today: the top-level
    ``agents.defaults`` and ``tools`` blocks, unchanged.  A named agent starts
    from ``agents.defaults`` and lays its own entries over it, except that
    ``workspace`` and ``tools.mcpServers`` are not inherited — see
    :func:`_resolve_defaults` and :func:`_resolve_tools`.

    Raises:
        KeyError: if no agent is configured under that name.
    """
    resolved = config.model_copy(deep=True)
    if name == RESERVED_AGENT_NAME:
        return ResolvedAgentConfig(name=name, config=resolved)

    entry = named_agent_entry(config, name)
    resolved.agents.defaults = _resolve_defaults(name, config.agents.defaults, entry)
    resolved.tools = _resolve_tools(config.tools, entry)
    return ResolvedAgentConfig(name=name, config=resolved)

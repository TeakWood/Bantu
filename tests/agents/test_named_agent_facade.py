"""The programmatic facade builds and reports one named agent."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.nanobot import Nanobot


def write_config(tmp_path: Path, **overrides: object) -> Path:
    """Write a config with a default agent and two named agents."""
    data: dict[str, object] = {
        "providers": {"openrouter": {"apiKey": "sk-test-key"}},
        "agents": {
            "defaults": {
                "model": "openai/gpt-4.1",
                "workspace": str(tmp_path / "default-workspace"),
            },
            "named": {
                "research": {"model": "openai/gpt-4.1-mini"},
                "ops": {},
            },
        },
    }
    data.update(overrides)
    config_dir = tmp_path / "instance"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


@pytest.fixture(autouse=True)
def _named_agents_under_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the default named-agent workspace root out of the user's home."""
    monkeypatch.setattr(
        "nanobot.agents.resolution.NAMED_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "named-agents"),
    )


# --- fake MCP -----------------------------------------------------------------


class _StubMCPTool(Tool):
    """Stand-in for a tool an MCP server contributes at connect time."""

    _plugin_discoverable = False

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "stub mcp tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class _StubMCPConnection:
    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register one tool per configured server instead of launching anything.

    MCP tools reach the agent's registry at connect time rather than by
    discovery, so this is what makes ``tool_names()`` observable without a real
    server.
    """

    async def connect(
        servers: Mapping[str, Any],
        registry: ToolRegistry,
    ) -> dict[str, _StubMCPConnection]:
        for name in servers:
            registry.register(_StubMCPTool(f"mcp_{name}_ping"))
        return {name: _StubMCPConnection() for name in servers}

    monkeypatch.setattr("nanobot.agent.tools.mcp.connect_mcp_servers", connect)


# --- which agent gets built ---------------------------------------------------


def test_omitting_agent_builds_default(tmp_path: Path) -> None:
    bot = Nanobot.from_config(write_config(tmp_path))

    assert bot.agent_name == "default"
    assert bot.workspace == tmp_path / "default-workspace"


def test_naming_default_explicitly_builds_the_same_agent(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)

    omitted = Nanobot.from_config(config_path)
    named = Nanobot.from_config(config_path, agent="default")

    assert named.agent_name == omitted.agent_name == "default"
    assert named.workspace == omitted.workspace
    assert named.runtime.model == omitted.runtime.model


def test_from_config_builds_the_named_agent(tmp_path: Path) -> None:
    bot = Nanobot.from_config(write_config(tmp_path), agent="research")

    assert bot.agent_name == "research"
    assert bot.workspace == tmp_path / "named-agents" / "research"
    assert bot.workspace != tmp_path / "default-workspace"


def test_a_named_agent_workspace_is_seeded(tmp_path: Path) -> None:
    bot = Nanobot.from_config(write_config(tmp_path), agent="ops")

    assert (bot.workspace / "SOUL.md").exists()
    assert (bot.workspace / "memory").is_dir()


def test_an_agent_reports_its_own_model_and_inherits_when_it_states_none(
    tmp_path: Path,
) -> None:
    config_path = write_config(tmp_path)

    assert Nanobot.from_config(config_path, agent="research").runtime.model == (
        "openai/gpt-4.1-mini"
    )
    assert Nanobot.from_config(config_path, agent="ops").runtime.model == "openai/gpt-4.1"


def test_an_unknown_agent_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown agent 'nobody'"):
        Nanobot.from_config(write_config(tmp_path), agent="nobody")


# --- overrides reach the requested agent --------------------------------------


def test_workspace_override_applies_to_the_named_agent(tmp_path: Path) -> None:
    override = tmp_path / "override-workspace"

    bot = Nanobot.from_config(write_config(tmp_path), workspace=override, agent="research")

    assert bot.workspace == override.resolve()


def test_model_override_applies_to_the_named_agent(tmp_path: Path) -> None:
    bot = Nanobot.from_config(
        write_config(tmp_path),
        model="openai/gpt-4o-mini",
        agent="research",
    )

    assert bot.runtime.model == "openai/gpt-4o-mini"


def test_an_override_for_a_named_agent_leaves_default_alone(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)

    Nanobot.from_config(config_path, model="openai/gpt-4o-mini", agent="research")
    default = Nanobot.from_config(config_path)

    assert default.runtime.model == "openai/gpt-4.1"


# --- tool isolation -----------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_names_include_only_that_agents_mcp_tools(
    tmp_path: Path,
    fake_mcp: None,
) -> None:
    config_path = write_config(
        tmp_path,
        tools={"mcpServers": {"shared": {"command": "shared-mcp"}}},
        agents={
            "defaults": {
                "model": "openai/gpt-4.1",
                "workspace": str(tmp_path / "default-workspace"),
            },
            "named": {
                "research": {"tools": {"mcpServers": {"notes": {"command": "notes-mcp"}}}},
                "ops": {},
            },
        },
    )

    default = set(await Nanobot.from_config(config_path).tool_names())
    research = set(await Nanobot.from_config(config_path, agent="research").tool_names())
    ops = set(await Nanobot.from_config(config_path, agent="ops").tool_names())

    assert "mcp_shared_ping" in default
    assert "mcp_notes_ping" not in default
    assert "mcp_notes_ping" in research
    assert "mcp_shared_ping" not in research
    assert {"mcp_shared_ping", "mcp_notes_ping"}.isdisjoint(ops)


@pytest.mark.asyncio
async def test_tool_names_without_mcp_still_report_the_built_in_tools(
    tmp_path: Path,
) -> None:
    names = await Nanobot.from_config(write_config(tmp_path), agent="research").tool_names()

    assert "read_file" in names
    assert "spawn" in names


@pytest.mark.asyncio
async def test_subagent_tool_names_are_the_subagent_registry(tmp_path: Path) -> None:
    bot = Nanobot.from_config(write_config(tmp_path), agent="research")

    names = set(await bot.subagent_tool_names())

    assert "read_file" in names
    # `spawn` is core-scoped: a subagent may not spawn further subagents.
    assert "spawn" not in names


@pytest.mark.asyncio
async def test_a_disabled_tool_is_absent_from_the_subagent_registry(
    tmp_path: Path,
) -> None:
    enabled = Nanobot.from_config(write_config(tmp_path), agent="ops")
    disabled = Nanobot.from_config(
        write_config(tmp_path / "off", tools={"exec": {"enable": False}}),
        agent="ops",
    )

    assert "exec" in set(await enabled.subagent_tool_names())
    assert "exec" not in set(await disabled.subagent_tool_names())


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", [None, "research", "ops"])
async def test_subagent_tool_names_are_a_subset_of_tool_names(
    tmp_path: Path,
    fake_mcp: None,
    agent: str | None,
) -> None:
    config_path = write_config(
        tmp_path,
        tools={"mcpServers": {"shared": {"command": "shared-mcp"}}},
        agents={
            "defaults": {
                "model": "openai/gpt-4.1",
                "workspace": str(tmp_path / "default-workspace"),
            },
            "named": {
                "research": {"tools": {"mcpServers": {"notes": {"command": "notes-mcp"}}}},
                "ops": {},
            },
        },
    )
    kwargs = {} if agent is None else {"agent": agent}

    bot = Nanobot.from_config(config_path, **kwargs)
    tool_names = set(await bot.tool_names())
    subagent_names = set(await bot.subagent_tool_names())

    assert subagent_names
    assert subagent_names <= tool_names
    # MCP tools are never offered to a subagent, so no agent's servers can
    # reach another agent's background work either.
    assert {"mcp_shared_ping", "mcp_notes_ping"}.isdisjoint(subagent_names)

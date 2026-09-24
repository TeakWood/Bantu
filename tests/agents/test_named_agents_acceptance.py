"""Acceptance for the first six named-agent criteria, checked from outside.

Every assertion here goes through a contact point — ``nanobot agents list
--json``, ``Nanobot.from_config(agent=...)``, the config file or the files on
disk.  Nothing reaches into a private attribute, so these tests keep holding if
the composition behind the facade is rearranged.

The scaffolding is deliberately self-contained rather than shared with
``test_named_agent_facade.py``: an acceptance check that reused the unit tests'
fixtures would be corroborating them with their own assumptions.

Two seams are faked, and only these two.  ``make_provider`` returns a provider
that answers without a network call, so a turn can actually be run through an
agent; ``connect_mcp_servers`` registers one tool per configured server, since
MCP tools reach a registry at connect time rather than by discovery and no real
server is ever spawned in this repo.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.cli.commands import app
from nanobot.nanobot import Nanobot
from nanobot.providers.base import GenerationSettings, LLMResponse

runner = CliRunner()


# --- configs ------------------------------------------------------------------


def write_config(directory: Path, payload: Mapping[str, Any]) -> Path:
    """Write *payload* as a config file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def live_payload(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """A config whose default workspace is under *tmp_path*, safe to build from."""
    payload: dict[str, Any] = {
        "providers": {"openrouter": {"apiKey": "sk-test-key"}},
        "agents": {
            "defaults": {
                "model": "base/model",
                "workspace": str(tmp_path / "default-workspace"),
            },
            "named": {"research": {"model": "research/model"}, "ops": {}},
        },
    }
    payload.update(overrides)
    return payload


def inspection_payload(**overrides: Any) -> dict[str, Any]:
    """A config for the CLI, which resolves without touching the filesystem."""
    payload: dict[str, Any] = {
        "agents": {"defaults": {"workspace": "~/default-workspace", "model": "base/model"}},
    }
    payload.update(overrides)
    return payload


def list_agents(config_path: Path) -> list[dict[str, Any]]:
    """Run ``nanobot agents list --json`` and return the parsed document."""
    result = runner.invoke(app, ["agents", "list", "--json", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


# --- fakes --------------------------------------------------------------------


class _FakeProvider:
    """Answers every request with a fixed reply instead of calling a model."""

    provider_name = "fake"

    def __init__(self, reply: str = "acknowledged") -> None:
        self._reply = reply
        self.generation = GenerationSettings()

    def get_default_model(self) -> str:
        return "fake/model"

    async def chat_stream_with_retry(self, **_kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self._reply, tool_calls=[], usage=None)


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a turn run end to end without a network call."""
    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider",
        lambda _config: _FakeProvider(),
    )


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
    """Register ``mcp_<server>_ping`` per configured server instead of launching one."""

    async def connect(
        servers: Mapping[str, Any],
        registry: ToolRegistry,
    ) -> dict[str, _StubMCPConnection]:
        for name in servers:
            registry.register(_StubMCPTool(f"mcp_{name}_ping"))
        return {name: _StubMCPConnection() for name in servers}

    monkeypatch.setattr("nanobot.agent.tools.mcp.connect_mcp_servers", connect)


@pytest.fixture
def agent_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Move the default named-agent workspace root out of the user's home.

    Only the tests that actually build an agent need this; the CLI resolves
    workspaces without creating them, so it is checked against the real
    ``~/.nanobot/agents`` default.
    """
    root = tmp_path / "named-agents"
    monkeypatch.setattr("nanobot.agents.resolution.NAMED_AGENT_WORKSPACE_ROOT", str(root))
    return root


# --- filesystem ---------------------------------------------------------------


def workspace_digest(workspace: Path) -> dict[str, str]:
    """Hash every file under *workspace*, keyed by its path relative to it."""
    return {
        str(path.relative_to(workspace)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(workspace.rglob("*"))
        if path.is_file()
    }


# ==============================================================================
# Criterion 1: no named agents, no change
# ==============================================================================
#
# The other half of this criterion — "the existing suite passes" — is the test
# gate itself, which runs unchanged against this branch.


def test_a_config_without_named_agents_lists_only_default(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, inspection_payload())

    entries = list_agents(config_path)

    assert [entry["name"] for entry in entries] == ["default"]
    assert entries[0]["workspace"] == str(Path.home() / "default-workspace")


def test_a_config_without_named_agents_builds_the_same_default_agent(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "instance",
        live_payload(
            tmp_path,
            agents={
                "defaults": {
                    "model": "base/model",
                    "workspace": str(tmp_path / "default-workspace"),
                }
            },
        ),
    )

    bot = Nanobot.from_config(config_path)

    assert bot.agent_name == "default"
    assert bot.workspace == tmp_path / "default-workspace"
    assert bot.runtime.model == "base/model"


# ==============================================================================
# Criterion 2: registry and resolution
# ==============================================================================


def test_two_named_agents_list_after_default(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        inspection_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"research": {"model": "research/model"}, "ops": {}},
            }
        ),
    )

    entries = list_agents(config_path)

    assert [entry["name"] for entry in entries] == ["default", "research", "ops"]


def test_the_listing_reports_each_agents_own_or_inherited_model(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        inspection_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"research": {"model": "research/model"}, "ops": {}},
            }
        ),
    )

    by_name = {entry["name"]: entry["model"] for entry in list_agents(config_path)}

    assert by_name == {
        "default": "base/model",
        "research": "research/model",
        "ops": "base/model",
    }


def test_an_agent_without_a_workspace_lists_under_the_named_agent_root(
    tmp_path: Path,
) -> None:
    config_path = write_config(
        tmp_path,
        inspection_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"research": {"model": "research/model"}, "ops": {}},
            }
        ),
    )

    by_name = {entry["name"]: entry["workspace"] for entry in list_agents(config_path)}

    for name in ("research", "ops"):
        assert by_name[name] == str(Path.home() / ".nanobot" / "agents" / name)
    assert by_name["default"] == str(Path.home() / "default-workspace")


def test_the_facade_resolves_the_same_model_and_workspace_as_the_listing(
    tmp_path: Path,
    agent_workspace_root: Path,
) -> None:
    config_path = write_config(tmp_path / "instance", live_payload(tmp_path))

    default = Nanobot.from_config(config_path)
    research = Nanobot.from_config(config_path, agent="research")
    ops = Nanobot.from_config(config_path, agent="ops")

    assert (default.agent_name, research.agent_name, ops.agent_name) == (
        "default",
        "research",
        "ops",
    )
    assert research.runtime.model == "research/model"
    assert ops.runtime.model == "base/model"
    assert default.workspace == tmp_path / "default-workspace"
    assert ops.workspace == agent_workspace_root / "ops"


# ==============================================================================
# Criterion 3: memory isolation
# ==============================================================================


def test_a_memory_written_through_one_facade_reaches_no_other_agent(
    tmp_path: Path,
    agent_workspace_root: Path,
) -> None:
    config_path = write_config(tmp_path / "instance", live_payload(tmp_path))
    research = Nanobot.from_config(config_path, agent="research")
    ops = Nanobot.from_config(config_path, agent="ops")
    default = Nanobot.from_config(config_path)

    research.memory.write("The lab prefers decaf after 3pm.")
    research.memory.append_history("Recorded the coffee preference.")

    assert "decaf" in research.memory.read()
    # Without this the "no other agent has it" assertions below would hold
    # vacuously for a facade that wrote nowhere at all.
    assert [entry["content"] for entry in research.memory.read_history()] == [
        "Recorded the coffee preference."
    ]
    assert "decaf" not in ops.memory.read()
    assert "decaf" not in default.memory.read()
    for other in (ops, default):
        assert other.memory.read_history() == []


def test_writing_one_agents_memory_leaves_every_other_workspace_byte_identical(
    tmp_path: Path,
    agent_workspace_root: Path,
) -> None:
    config_path = write_config(tmp_path / "instance", live_payload(tmp_path))
    research = Nanobot.from_config(config_path, agent="research")
    ops = Nanobot.from_config(config_path, agent="ops")
    default = Nanobot.from_config(config_path)

    before = {
        "ops": workspace_digest(ops.workspace),
        "default": workspace_digest(default.workspace),
    }
    assert before["ops"] and before["default"]

    research.memory.write("The lab prefers decaf after 3pm.")
    research.memory.append_history("Recorded the coffee preference.")

    assert workspace_digest(ops.workspace) == before["ops"]
    assert workspace_digest(default.workspace) == before["default"]


# ==============================================================================
# Criterion 4: session isolation
# ==============================================================================


@pytest.mark.asyncio
async def test_a_turn_run_through_one_agent_is_listed_by_it_alone(
    tmp_path: Path,
    agent_workspace_root: Path,
    fake_llm: None,
) -> None:
    config_path = write_config(tmp_path / "instance", live_payload(tmp_path))
    research = Nanobot.from_config(config_path, agent="research")
    ops = Nanobot.from_config(config_path, agent="ops")
    default = Nanobot.from_config(config_path)

    await research.run(
        "remember the badge number",
        session_key="telegram:42",
        channel="telegram",
        chat_id="42",
    )

    assert [info.key for info in research.sessions.list()] == ["telegram:42"]
    assert ops.sessions.list() == []
    assert default.sessions.list() == []


@pytest.mark.asyncio
async def test_another_agent_on_the_same_chat_starts_with_empty_history(
    tmp_path: Path,
    agent_workspace_root: Path,
    fake_llm: None,
) -> None:
    config_path = write_config(tmp_path / "instance", live_payload(tmp_path))
    research = Nanobot.from_config(config_path, agent="research")
    ops = Nanobot.from_config(config_path, agent="ops")

    await research.run(
        "the badge number is 77",
        session_key="telegram:42",
        channel="telegram",
        chat_id="42",
    )
    assert ops.sessions.get("telegram:42") is None

    await ops.run(
        "what is the badge number?",
        session_key="telegram:42",
        channel="telegram",
        chat_id="42",
    )

    ops_history = ops.sessions.get("telegram:42")
    assert ops_history is not None
    contents = [str(message.get("content", "")) for message in ops_history.messages]
    assert any("what is the badge number?" in content for content in contents)
    assert not any("the badge number is 77" in content for content in contents)


# ==============================================================================
# Criterion 5: tool isolation
# ==============================================================================


def mcp_payload(tmp_path: Path) -> dict[str, Any]:
    """A config with one MCP server at top level and one under ``research``."""
    return live_payload(
        tmp_path,
        tools={"mcpServers": {"shared": {"command": "shared-mcp"}}},
        agents={
            "defaults": {
                "model": "base/model",
                "workspace": str(tmp_path / "default-workspace"),
            },
            "named": {
                "research": {
                    "model": "research/model",
                    "tools": {"mcpServers": {"notes": {"command": "notes-mcp"}}},
                },
                "ops": {},
            },
        },
    )


@pytest.mark.asyncio
async def test_an_mcp_server_declared_under_one_agent_reaches_that_agent_alone(
    tmp_path: Path,
    agent_workspace_root: Path,
    fake_mcp: None,
) -> None:
    config_path = write_config(tmp_path / "instance", mcp_payload(tmp_path))

    default = set(await Nanobot.from_config(config_path).tool_names())
    research = set(await Nanobot.from_config(config_path, agent="research").tool_names())
    ops = set(await Nanobot.from_config(config_path, agent="ops").tool_names())

    assert "mcp_notes_ping" in research
    assert "mcp_notes_ping" not in default
    assert "mcp_notes_ping" not in ops


@pytest.mark.asyncio
async def test_a_top_level_mcp_server_reaches_default_and_no_named_agent(
    tmp_path: Path,
    agent_workspace_root: Path,
    fake_mcp: None,
) -> None:
    config_path = write_config(tmp_path / "instance", mcp_payload(tmp_path))

    default = set(await Nanobot.from_config(config_path).tool_names())
    research = set(await Nanobot.from_config(config_path, agent="research").tool_names())
    ops = set(await Nanobot.from_config(config_path, agent="ops").tool_names())

    assert "mcp_shared_ping" in default
    assert "mcp_shared_ping" not in research
    assert "mcp_shared_ping" not in ops


# ==============================================================================
# Criterion 6: subagent tool inheritance
# ==============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", [None, "research", "ops"])
async def test_every_agents_subagent_tools_are_a_subset_of_its_own(
    tmp_path: Path,
    agent_workspace_root: Path,
    fake_mcp: None,
    agent: str | None,
) -> None:
    config_path = write_config(tmp_path / "instance", mcp_payload(tmp_path))
    kwargs = {} if agent is None else {"agent": agent}

    bot = Nanobot.from_config(config_path, **kwargs)
    tool_names = set(await bot.tool_names())
    subagent_names = set(await bot.subagent_tool_names())

    assert subagent_names
    assert subagent_names <= tool_names


@pytest.mark.asyncio
async def test_no_agents_mcp_tools_reach_another_agents_subagents(
    tmp_path: Path,
    agent_workspace_root: Path,
    fake_mcp: None,
) -> None:
    config_path = write_config(tmp_path / "instance", mcp_payload(tmp_path))
    servers = {"mcp_shared_ping", "mcp_notes_ping"}

    for agent in (None, "research", "ops"):
        kwargs = {} if agent is None else {"agent": agent}
        bot = Nanobot.from_config(config_path, **kwargs)

        assert servers.isdisjoint(set(await bot.subagent_tool_names()))

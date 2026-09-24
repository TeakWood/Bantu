"""Workspace, memory, session and tool isolation between agents.

Covers acceptance criteria 3 (memory isolation), 4 (session isolation),
5 (tool isolation) and 6 (subagent tool inheritance).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from nanobot.nanobot import Nanobot
from nanobot.sdk.types import SessionSnapshot

from .conftest import write_config, write_mcp_server
from .scripted_provider import ScriptedProvider, reply


def _workspace_fingerprint(root: Path) -> dict[str, str]:
    """Return a path -> content-hash map for every file under *root*."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def three_agent_config(instance_dir: Path) -> Path:
    return write_config(
        instance_dir,
        {
            "agents": {
                "defaults": {
                    "model": "openai/gpt-4.1",
                    "workspace": str(instance_dir / "default-ws"),
                },
                "named": {
                    "research": {"workspace": str(instance_dir / "research-ws")},
                    "health": {"workspace": str(instance_dir / "health-ws")},
                },
            }
        },
    )


@pytest.fixture
def agents(
    three_agent_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, Nanobot]]:
    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider",
        lambda *_args, **_kwargs: ScriptedProvider(lambda _m: reply("Noted.")),
    )
    bots = {
        "default": Nanobot.from_config(three_agent_config),
        "research": Nanobot.from_config(three_agent_config, agent="research"),
        "health": Nanobot.from_config(three_agent_config, agent="health"),
    }
    yield bots


def test_each_agent_reports_its_own_identity_and_workspace(
    agents: dict[str, Nanobot],
    instance_dir: Path,
) -> None:
    assert agents["default"].agent_name == "default"
    assert agents["research"].agent_name == "research"
    assert agents["research"].workspace == instance_dir / "research-ws"
    assert agents["health"].workspace == instance_dir / "health-ws"
    assert len({bot.workspace for bot in agents.values()}) == 3


def test_a_fact_written_to_one_agents_memory_reaches_no_other(
    agents: dict[str, Nanobot],
) -> None:
    secret = "Patient reports migraines after 3pm."
    others = [agents["default"], agents["research"]]
    before = [_workspace_fingerprint(bot.workspace) for bot in others]

    agents["health"].memory.write(secret)

    assert secret in agents["health"].memory.read()
    for bot in others:
        assert secret not in bot.memory.read()
    # The other agents' workspaces are byte-identical before and after.
    assert [_workspace_fingerprint(bot.workspace) for bot in others] == before


def test_memory_history_stays_in_the_agent_that_recorded_it(
    agents: dict[str, Nanobot],
) -> None:
    agents["research"].memory.append_history("Semis rallied on new fab guidance.")

    research_history = agents["research"].memory.read_history()
    assert any("Semis rallied" in str(entry) for entry in research_history)
    assert agents["health"].memory.read_history() == []
    assert agents["default"].memory.read_history() == []


async def test_a_conversation_creates_a_session_listed_by_that_agent_alone(
    agents: dict[str, Nanobot],
) -> None:
    await agents["research"].run(
        "what moved today?",
        session_key="telegram.research:42",
        channel="telegram.research",
        chat_id="42",
    )
    for bot in agents.values():
        bot.sessions.flush()

    assert [info.key for info in agents["research"].sessions.list()] == ["telegram.research:42"]
    assert agents["health"].sessions.list() == []
    assert agents["default"].sessions.list() == []


async def test_another_agent_on_the_same_channel_and_chat_starts_empty(
    agents: dict[str, Nanobot],
) -> None:
    key = "telegram:9001"
    await agents["research"].sessions.restore(
        SessionSnapshot(
            key=key,
            messages=[{"role": "user", "content": "remember this"}],
            metadata={},
        ),
    )

    assert agents["research"].sessions.get(key) is not None
    assert agents["default"].sessions.get(key) is None
    assert agents["health"].sessions.get(key) is None
    # An agent handed the same channel and chat id starts with empty history.
    assert agents["default"]._loop.sessions.get_or_create(key).messages == []


def test_agents_never_share_a_session_store_directory(agents: dict[str, Nanobot]) -> None:
    roots = {bot._loop.sessions.sessions_dir for bot in agents.values()}

    assert len(roots) == 3


async def test_an_mcp_server_contributes_tools_to_only_the_agent_that_declares_it(
    instance_dir: Path,
) -> None:
    servers = instance_dir / "servers"
    config_path = write_config(
        instance_dir,
        {
            "tools": {"mcpServers": {"shared": write_mcp_server(servers, "shared", "shared_echo")}},
            "agents": {
                "defaults": {"workspace": str(instance_dir / "default-ws")},
                "named": {
                    "trader": {
                        "workspace": str(instance_dir / "trader-ws"),
                        "tools": {
                            "mcpServers": {
                                "broker": write_mcp_server(servers, "broker", "place_trade"),
                            }
                        },
                    },
                    "health": {"workspace": str(instance_dir / "health-ws")},
                },
            },
        },
    )

    bots = {
        name: Nanobot.from_config(config_path, agent=name)
        for name in ("default", "trader", "health")
    }
    try:
        for bot in bots.values():
            await bot.connect_mcp()

        tools = {name: set(bot.tool_names()) for name, bot in bots.items()}
        broker_tools = {tool for tool in tools["trader"] if "place_trade" in tool}
        shared_tools = {tool for tool in tools["default"] if "shared_echo" in tool}

        # The trader alone holds the brokerage tools.
        assert broker_tools
        assert not broker_tools & tools["default"]
        assert not broker_tools & tools["health"]
        # A top-level server belongs to `default` and to no named agent.
        assert shared_tools
        assert not shared_tools & tools["trader"]
        assert not shared_tools & tools["health"]
    finally:
        for bot in bots.values():
            await bot.aclose()


async def test_subagent_tools_are_a_subset_that_excludes_every_agents_mcp_tools(
    instance_dir: Path,
) -> None:
    servers = instance_dir / "servers"
    config_path = write_config(
        instance_dir,
        {
            "tools": {"mcpServers": {"shared": write_mcp_server(servers, "shared", "shared_echo")}},
            "agents": {
                "defaults": {"workspace": str(instance_dir / "default-ws")},
                "named": {
                    "trader": {
                        "workspace": str(instance_dir / "trader-ws"),
                        "tools": {
                            "mcpServers": {
                                "broker": write_mcp_server(servers, "broker", "place_trade"),
                            }
                        },
                    },
                    "health": {"workspace": str(instance_dir / "health-ws")},
                },
            },
        },
    )

    bots = {
        name: Nanobot.from_config(config_path, agent=name)
        for name in ("default", "trader", "health")
    }
    try:
        for bot in bots.values():
            await bot.connect_mcp()

        for name, bot in bots.items():
            subagent_tools = set(bot.subagent_tool_names())
            assert subagent_tools, f"{name} subagent has no tools"
            # For every agent, the subagent tool set is a subset of its own.
            assert subagent_tools <= set(bot.tool_names())

        # No agent's MCP-only tools reach any agent's subagents.
        mcp_only = {
            tool
            for bot in bots.values()
            for tool in bot.tool_names()
            if "place_trade" in tool or "shared_echo" in tool
        }
        assert mcp_only
        for bot in bots.values():
            assert not mcp_only & set(bot.subagent_tool_names())
    finally:
        for bot in bots.values():
            await bot.aclose()

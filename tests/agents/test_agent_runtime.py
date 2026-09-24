"""One agent's fully-isolated runtime bundle: bus, tools, MCP, sessions, loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.agents import AgentRuntime, build_agent_runtime
from nanobot.config.loader import load_config
from nanobot.cron.service import CronService


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
                "research": {},
                "ops": {},
            },
        },
    }
    data.update(overrides)
    config_dir = tmp_path / "instance"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


def load(tmp_path: Path, **overrides: object):
    return load_config(write_config(tmp_path, **overrides))


@pytest.fixture(autouse=True)
def _named_agents_under_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the default named-agent workspace root out of the user's home."""
    monkeypatch.setattr(
        "nanobot.agents.resolution.NAMED_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "named-agents"),
    )


# --- two agents share nothing -------------------------------------------------


def test_two_agents_share_no_runtime_component(tmp_path: Path) -> None:
    config = load(tmp_path)

    default = build_agent_runtime(config, "default")
    research = build_agent_runtime(config, "research")

    assert isinstance(default, AgentRuntime)
    assert default.bus is not research.bus
    assert default.tools is not research.tools
    assert default.mcp_provider is not research.mcp_provider
    assert default.sessions is not research.sessions
    assert default.workspace != research.workspace


def test_each_agents_loop_uses_its_own_bus_and_registry(tmp_path: Path) -> None:
    config = load(tmp_path)

    default = build_agent_runtime(config, "default")
    research = build_agent_runtime(config, "research")

    assert default.loop.bus is default.bus
    assert research.loop.bus is research.bus
    assert default.loop.tools is default.tools
    assert research.loop.tools is research.tools
    # The MCP provider must share the loop's registry so connected servers
    # register into the tools that agent's model is actually offered.
    assert default.mcp_provider._registry is default.loop.tools
    assert research.mcp_provider._registry is research.loop.tools


def test_sessions_are_isolated_per_agent(tmp_path: Path) -> None:
    config = load(tmp_path)

    default = build_agent_runtime(config, "default")
    research = build_agent_runtime(config, "research")

    assert default.sessions.sessions_dir != research.sessions.sessions_dir
    assert default.sessions.workspace != research.sessions.workspace


def test_a_message_on_one_agents_bus_is_invisible_to_another(tmp_path: Path) -> None:
    config = load(tmp_path)

    research = build_agent_runtime(config, "research")
    ops = build_agent_runtime(config, "ops")

    research.bus.inbound.put_nowait(object())  # type: ignore[arg-type]

    assert research.bus.inbound.qsize() == 1
    assert ops.bus.inbound.qsize() == 0


# --- workspace bootstrap ------------------------------------------------------


def test_each_agents_workspace_is_created_and_seeded(tmp_path: Path) -> None:
    config = load(tmp_path)

    for name in ("default", "research", "ops"):
        runtime = build_agent_runtime(config, name)
        workspace = runtime.workspace
        assert workspace.is_dir()
        assert (workspace / "SOUL.md").is_file()
        assert (workspace / "USER.md").is_file()
        assert (workspace / "memory").is_dir()
        assert (workspace / "memory" / "MEMORY.md").is_file()


def test_a_named_agent_gets_its_own_workspace_not_the_defaults(tmp_path: Path) -> None:
    config = load(tmp_path)

    default = build_agent_runtime(config, "default")
    research = build_agent_runtime(config, "research")

    assert default.workspace == (tmp_path / "default-workspace")
    assert research.workspace == (tmp_path / "named-agents" / "research")


def test_seeded_files_are_not_overwritten_on_a_second_build(tmp_path: Path) -> None:
    config = load(tmp_path)

    first = build_agent_runtime(config, "research")
    (first.workspace / "SOUL.md").write_text("mine", encoding="utf-8")

    second = build_agent_runtime(config, "research")

    assert (second.workspace / "SOUL.md").read_text(encoding="utf-8") == "mine"


# --- cron is the default agent's alone ----------------------------------------


def test_named_agent_has_no_cron_tool(tmp_path: Path) -> None:
    config = load(tmp_path)
    cron = CronService(tmp_path / "cron" / "jobs.json")

    research = build_agent_runtime(config, "research", cron_service=cron)

    assert "cron" not in research.loop.tool_names
    assert research.loop.tools.get("cron") is None


def test_default_agent_keeps_its_cron_tool(tmp_path: Path) -> None:
    config = load(tmp_path)
    cron = CronService(tmp_path / "cron" / "jobs.json")

    default = build_agent_runtime(config, "default", cron_service=cron)

    assert "cron" in default.loop.tool_names


def test_no_cron_service_means_no_cron_tool_for_default_either(tmp_path: Path) -> None:
    config = load(tmp_path)

    default = build_agent_runtime(config, "default")

    assert "cron" not in default.loop.tool_names


# --- MCP servers are per-agent ------------------------------------------------


def test_each_agent_gets_only_its_own_mcp_servers(tmp_path: Path) -> None:
    config = load(
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

    default = build_agent_runtime(config, "default")
    research = build_agent_runtime(config, "research")
    ops = build_agent_runtime(config, "ops")

    assert default.mcp_provider.configured_server_names == {"shared"}
    assert research.mcp_provider.configured_server_names == {"notes"}
    assert ops.mcp_provider.configured_server_names == set()


@pytest.mark.asyncio
async def test_a_failing_mcp_server_still_yields_a_working_agent(tmp_path: Path) -> None:
    config = load(
        tmp_path,
        agents={
            "defaults": {
                "model": "openai/gpt-4.1",
                "workspace": str(tmp_path / "default-workspace"),
            },
            "named": {
                "research": {
                    "tools": {
                        "mcpServers": {
                            "broken": {"command": "nanobot-no-such-mcp-binary"}
                        }
                    }
                }
            },
        },
    )

    research = build_agent_runtime(config, "research")
    builtin = list(research.loop.tool_names)

    await research.connect_mcp()

    assert research.mcp_provider.configured_server_names == {"broken"}
    assert research.mcp_provider.connected_server_names == set()
    assert research.mcp_provider.runtime_status()["broken"] == "failed"
    assert research.loop.tool_names == builtin
    assert "read_file" in builtin

    await research.aclose()


# --- unknown agents -----------------------------------------------------------


def test_unknown_agent_is_rejected(tmp_path: Path) -> None:
    config = load(tmp_path)

    with pytest.raises(KeyError):
        build_agent_runtime(config, "nope")

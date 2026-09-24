"""The gateway composition root, rewired from one AgentLoop onto MultiAgentRuntime.

Scheduled work stays bound to ``default`` and this is where that is enforced end
to end: cron, Dream and heartbeat all hang off the one ``CronService`` the
gateway builds on ``agents.defaults.workspace``, and neither ``CronJob`` nor
``CronPayload`` carries an agent identity.  These tests therefore drive the real
``_run_gateway`` -- with only the network-facing edges faked -- so the binding is
checked where it is enforced rather than where it is configured.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nanobot.agent.tools.message import MessageTool
from nanobot.agents.multi import MultiAgentRuntime
from nanobot.cli.gateway_runtime import _run_gateway
from nanobot.config.loader import load_config
from nanobot.config.schema import Config
from nanobot.cron.service import CronService
from nanobot.cron.types import CronJob
from nanobot.gateway.runtime import GatewayInstance
from nanobot.providers.factory import ProviderSnapshot

TELEGRAM_INSTANCES = [
    {"id": "default", "token": "default-token"},
    {"id": "research", "token": "research-token", "agent": "research"},
]


class _StopGatewayError(RuntimeError):
    """Raised from a runtime task to end the gateway once startup has finished."""


class _FakeChannelManager:
    """Enough ChannelManager surface for assembly, with no network listeners."""

    def __init__(self, _config: Config, bus: Any, **kwargs: Any) -> None:
        self.enabled_channels = ["telegram"]
        self.bus = bus
        self.kwargs = kwargs

    def get_channel(self, _name: str) -> object | None:
        return None

    def get_status(self) -> dict[str, Any]:
        return {}

    async def start_all(self) -> None:
        await asyncio.Event().wait()

    async def stop_all(self) -> None:
        return None


def _snapshot(config: Config, **_kwargs: Any) -> ProviderSnapshot:
    provider = MagicMock()
    provider.generation.max_tokens = 4096
    return ProviderSnapshot(
        provider=provider,
        model=config.agents.defaults.model,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        signature=("test",),
    )


def write_config(tmp_path: Path, *, named: dict[str, Any] | None = None) -> Path:
    """Write a config on disk so real agent runtimes can be built from it."""
    data: dict[str, Any] = {
        "providers": {"openrouter": {"apiKey": "sk-test-key"}},
        "agents": {
            "defaults": {
                "model": "openai/gpt-4.1",
                "workspace": str(tmp_path / "default-workspace"),
                "dream": {"enabled": False},
            },
            "named": (
                named
                if named is not None
                else {"research": {"workspace": str(tmp_path / "research-workspace")}}
            ),
        },
        "gateway": {"heartbeat": {"enabled": True, "intervalS": 3600}},
        "channels": {"telegram": {"instances": TELEGRAM_INSTANCES}},
    }
    config_dir = tmp_path / "instance"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


def seed_history(workspace: Path, cursor: int) -> None:
    """Give a workspace unprocessed history, so a Dream cursor repair is visible."""
    memory_dir = workspace / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "history.jsonl").write_text(
        json.dumps({"cursor": cursor, "timestamp": "2026-01-01T00:00:00", "content": "hi"})
        + "\n",
        encoding="utf-8",
    )


def run_gateway(
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
    *,
    stop: str = "local_triggers",
    channel_manager: type[_FakeChannelManager] = _FakeChannelManager,
) -> dict[str, Any]:
    """Start the real gateway, let it reach a running state, then stop it.

    Returns whatever the composition root built, so a test can assert against the
    same objects the gateway itself wired together.
    """
    seen: dict[str, Any] = {}

    class _RecordingRuntime(MultiAgentRuntime):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            seen["runtime"] = self

    class _RecordingCron(CronService):
        def __init__(self, store_path: Path) -> None:
            super().__init__(store_path)
            seen["cron"] = self
            seen["cron_store_path"] = store_path

    async def _stop_from_local_triggers(**kwargs: Any) -> None:
        seen["local_trigger_queue_kwargs"] = kwargs
        raise _StopGatewayError("stop")

    async def _idle_local_triggers(**kwargs: Any) -> None:
        seen["local_trigger_queue_kwargs"] = kwargs
        await asyncio.Event().wait()

    async def _watch_config_file(path: Path, on_change: Any) -> None:
        seen["config_watch_path"] = path
        seen["config_watch_callback"] = on_change
        if stop == "config_watcher":
            raise _StopGatewayError("stop")
        await asyncio.Event().wait()

    monkeypatch.setattr("nanobot.agents.multi.MultiAgentRuntime", _RecordingRuntime)
    monkeypatch.setattr("nanobot.cron.service.CronService", _RecordingCron)
    monkeypatch.setattr("nanobot.channels.manager.ChannelManager", channel_manager)
    monkeypatch.setattr(
        "nanobot.triggers.local_runner.run_local_trigger_queue",
        _stop_from_local_triggers if stop == "local_triggers" else _idle_local_triggers,
    )
    monkeypatch.setattr("nanobot.config.watcher.watch_config_file", _watch_config_file)
    monkeypatch.setattr("nanobot.providers.factory.build_provider_snapshot", _snapshot)
    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider",
        lambda config, **_kwargs: _snapshot(config).provider,
    )
    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime._prepare_webui_bundle_for_gateway",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime._tcp_endpoint_reachable", lambda *_a, **_kw: False
    )
    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime._webui_endpoint_reachable", lambda *_a, **_kw: False
    )
    monkeypatch.setattr("nanobot.cli.gateway_runtime.read_webui_sidebar_state", lambda: {})

    _run_gateway(
        load_config(config_path),
        health_server_enabled=False,
        gateway_instance=GatewayInstance.resolve(config_path=config_path),
    )
    return seen


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """A gateway that ran with one named agent beside the default one."""
    config_path = write_config(tmp_path)
    return run_gateway(monkeypatch, config_path)


# --- what the composition root builds ----------------------------------------


def test_the_gateway_starts_a_multi_agent_runtime(gateway: dict[str, Any]) -> None:
    runtime = gateway["runtime"]

    assert isinstance(runtime, MultiAgentRuntime)
    assert runtime.names == ("default", "research")


def test_a_config_with_no_named_agents_yields_exactly_one_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = write_config(tmp_path, named={})

    runtime = run_gateway(monkeypatch, config_path)["runtime"]

    assert runtime.names == ("default",)
    assert len(runtime) == 1


def test_each_agent_owns_its_bus_sessions_and_workspace(gateway: dict[str, Any]) -> None:
    runtime = gateway["runtime"]
    default, research = runtime.default, runtime.get("research")

    assert default.bus is not research.bus
    # The channel-facing bus is nobody's agent bus: the demux sits between them.
    assert runtime.bus is not default.bus
    assert runtime.bus is not research.bus
    assert default.sessions is not research.sessions
    assert default.sessions.sessions_dir != research.sessions.sessions_dir
    assert default.workspace != research.workspace
    assert default.loop.bus is default.bus
    assert research.loop.bus is research.bus


def test_a_bot_bound_to_a_named_agent_is_demuxed_to_it(gateway: dict[str, Any]) -> None:
    runtime = gateway["runtime"]

    assert runtime.agent_for("telegram.research") is runtime.get("research")
    assert runtime.agent_for("telegram") is runtime.default
    assert runtime.agent_for("cli") is runtime.default


# --- scheduled work is the default agent's alone ------------------------------


def test_only_the_default_workspace_acquires_a_cron_store(
    gateway: dict[str, Any], tmp_path: Path
) -> None:
    runtime = gateway["runtime"]

    assert gateway["cron_store_path"] == runtime.default.workspace / "cron" / "jobs.json"
    assert (runtime.default.workspace / "cron" / "jobs.json").exists()
    assert not (runtime.get("research").workspace / "cron").exists()


def test_only_the_default_workspace_acquires_a_dream_cursor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dream's cursor repair runs against default's memory and no other agent's."""
    config_path = write_config(tmp_path)
    default_workspace = tmp_path / "default-workspace"
    research_workspace = tmp_path / "research-workspace"
    # Both agents have unprocessed history, so either cursor *could* advance.
    seed_history(default_workspace, 7)
    seed_history(research_workspace, 7)

    run_gateway(monkeypatch, config_path)

    assert (default_workspace / "memory" / ".dream_cursor").read_text().strip() == "7"
    assert not (research_workspace / "memory" / ".dream_cursor").exists()


def test_a_named_agent_gets_no_cron_tool(gateway: dict[str, Any]) -> None:
    """The gateway hands its one CronService to every agent; only default keeps it."""
    runtime = gateway["runtime"]

    assert "cron" in runtime.default.loop.tool_names
    assert "cron" not in runtime.get("research").loop.tool_names


def test_the_heartbeat_target_comes_only_from_the_default_agents_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = write_config(tmp_path)
    picked: dict[str, Any] = {}

    def _pick(**kwargs: Any) -> tuple[str, str]:
        picked.update(kwargs)
        return "cli", "direct"

    monkeypatch.setattr(
        "nanobot.cli.gateway_runtime._pick_heartbeat_target_from_sessions", _pick
    )
    seen = run_gateway(monkeypatch, config_path)
    runtime = seen["runtime"]

    # One chat per agent, in each agent's own store.
    runtime.default.sessions.save(
        runtime.default.sessions.get_or_create("telegram:default-chat")
    )
    research = runtime.get("research")
    research.sessions.save(research.sessions.get_or_create("telegram.research:other-chat"))
    (runtime.default.workspace / "HEARTBEAT.md").write_text(
        "## Active Tasks\n- check the build\n", encoding="utf-8"
    )

    asyncio.run(seen["cron"].on_job(CronJob(id="heartbeat", name="heartbeat")))

    keys = {item.get("key") for item in picked["sessions"]}
    assert "telegram:default-chat" in keys
    assert "telegram.research:other-chat" not in keys


# --- everything else the gateway starts keeps working -------------------------


def test_local_triggers_stay_bound_to_the_default_agent(gateway: dict[str, Any]) -> None:
    runtime = gateway["runtime"]
    kwargs = gateway["local_trigger_queue_kwargs"]

    assert kwargs["submit_turn"] == runtime.default.loop.submit_local_trigger_turn
    assert runtime.default.loop.local_trigger_store is kwargs["store"]
    assert runtime.get("research").loop.local_trigger_store is None


def test_the_channel_manager_speaks_the_channel_bus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = write_config(tmp_path)
    channels: dict[str, Any] = {}

    class _CapturingChannelManager(_FakeChannelManager):
        def __init__(self, config: Config, bus: Any, **kwargs: Any) -> None:
            super().__init__(config, bus, **kwargs)
            channels["manager"] = self

    runtime = run_gateway(
        monkeypatch, config_path, channel_manager=_CapturingChannelManager
    )["runtime"]

    assert channels["manager"].bus is runtime.bus
    assert channels["manager"].kwargs["session_manager"] is runtime.default.sessions


def test_the_config_watcher_invalidates_every_agent_and_the_routing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = write_config(tmp_path)

    seen = run_gateway(monkeypatch, config_path)
    runtime = seen["runtime"]
    assert seen["config_watch_path"] == config_path
    # Warm the memoised routing decision, then let the watcher fire.
    runtime.agent_for("telegram.research")
    for entry in runtime:
        entry.loop.runtime_resolver.invalidate = MagicMock()  # noqa: SLF001

    seen["config_watch_callback"]()

    for entry in runtime:
        entry.loop.runtime_resolver.invalidate.assert_called_once_with()
    assert runtime._routes == {}  # noqa: SLF001


def test_every_agent_can_send_proactively_on_its_own_bus(
    gateway: dict[str, Any]
) -> None:
    """A named agent's message tool delivers, and never onto another agent's bus."""
    runtime = gateway["runtime"]
    research = runtime.get("research")
    tool = research.loop.tools.get("message")
    assert isinstance(tool, MessageTool)

    result = asyncio.run(
        tool.execute("hello", channel="telegram.research", chat_id="chat-1")
    )

    assert "Error" not in result
    assert research.bus.outbound_size == 1
    assert runtime.default.bus.outbound_size == 0
    # The channel bus is fed by the outbound pump, not written to directly.
    assert runtime.bus.outbound_size == 0
    msg = research.bus.outbound.get_nowait()
    assert (msg.channel, msg.chat_id, msg.content) == ("telegram.research", "chat-1", "hello")

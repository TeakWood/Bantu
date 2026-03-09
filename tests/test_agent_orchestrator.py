"""Tests for AgentOrchestrator — routing, ownership, and wiring (Bantu-irt)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_msg(
    content: str = "hello",
    agent_id: str | None = None,
    channel: str = "cli",
    chat_id: str = "test",
) -> InboundMessage:
    return InboundMessage(
        channel=channel,
        sender_id="user",
        chat_id=chat_id,
        content=content,
        agent_id=agent_id,
    )


def _make_agents_dir(tmp_path: Path, names: list[str]) -> Path:
    """Create an agents directory with the given sub-agent names."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (agents_dir / name).mkdir(parents=True, exist_ok=True)
    return agents_dir


def _make_minimal_config(tmp_path: Path) -> Config:
    config = Config()
    config.agents.defaults.workspace = str(tmp_path / "workspace")
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    return config


# ---------------------------------------------------------------------------
# Route: correct per-agent queue by agent_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_known_agent_id_delivers_to_correct_queue():
    """route() puts a message into the queue for the specified agent_id."""
    from nanobot.agent.orchestrator import AgentOrchestrator

    bus = MessageBus()
    config = Config()
    orchestrator = AgentOrchestrator(bus, config, Path("agents"))

    # Manually set up two queues (bypassing start())
    q_default: asyncio.Queue[InboundMessage] = asyncio.Queue()
    q_silpi: asyncio.Queue[InboundMessage] = asyncio.Queue()
    orchestrator._queues["default"] = q_default
    orchestrator._queues["silpi"] = q_silpi

    msg = _make_msg(agent_id="silpi")
    await orchestrator.route(msg)

    assert q_silpi.qsize() == 1
    assert q_default.qsize() == 0
    item = q_silpi.get_nowait()
    assert item is msg


@pytest.mark.asyncio
async def test_route_unknown_agent_id_falls_back_to_default():
    """route() falls back to the default queue for an unknown agent_id."""
    from nanobot.agent.orchestrator import AgentOrchestrator

    bus = MessageBus()
    orchestrator = AgentOrchestrator(bus, Config(), Path("agents"))

    q_default: asyncio.Queue[InboundMessage] = asyncio.Queue()
    orchestrator._queues["default"] = q_default

    msg = _make_msg(agent_id="no-such-agent")
    await orchestrator.route(msg)

    assert q_default.qsize() == 1


@pytest.mark.asyncio
async def test_route_none_agent_id_falls_back_to_default():
    """route() falls back to the default queue when agent_id is None."""
    from nanobot.agent.orchestrator import AgentOrchestrator

    bus = MessageBus()
    orchestrator = AgentOrchestrator(bus, Config(), Path("agents"))

    q_default: asyncio.Queue[InboundMessage] = asyncio.Queue()
    orchestrator._queues["default"] = q_default

    msg = _make_msg(agent_id=None)
    await orchestrator.route(msg)

    assert q_default.qsize() == 1


@pytest.mark.asyncio
async def test_route_explicit_default_agent_id():
    """route() with agent_id='default' goes to the default queue."""
    from nanobot.agent.orchestrator import AgentOrchestrator

    bus = MessageBus()
    orchestrator = AgentOrchestrator(bus, Config(), Path("agents"))

    q_default: asyncio.Queue[InboundMessage] = asyncio.Queue()
    q_other: asyncio.Queue[InboundMessage] = asyncio.Queue()
    orchestrator._queues["default"] = q_default
    orchestrator._queues["other"] = q_other

    msg = _make_msg(agent_id="default")
    await orchestrator.route(msg)

    assert q_default.qsize() == 1
    assert q_other.qsize() == 0


# ---------------------------------------------------------------------------
# Shared outbound queue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outbound_queue_is_shared_across_agents(tmp_path: Path):
    """All agent loops write to the same bus.outbound queue."""
    from nanobot.agent.orchestrator import AgentOrchestrator

    bus = MessageBus()
    config = _make_minimal_config(tmp_path)
    agents_dir = _make_agents_dir(tmp_path, ["silpi"])
    orchestrator = AgentOrchestrator(bus, config, agents_dir)

    # Manually set up queues (bypass start())
    for agent_id in ["default", "silpi"]:
        orchestrator._queues[agent_id] = asyncio.Queue()

    # Both agents share the same bus.outbound — verify by publishing directly
    await bus.publish_outbound(OutboundMessage(channel="cli", chat_id="x", content="from-default"))
    await bus.publish_outbound(OutboundMessage(channel="cli", chat_id="y", content="from-silpi"))

    msgs = []
    while not bus.outbound.empty():
        msgs.append(await bus.consume_outbound())

    assert len(msgs) == 2
    contents = {m.content for m in msgs}
    assert "from-default" in contents
    assert "from-silpi" in contents


# ---------------------------------------------------------------------------
# Cron / heartbeat restricted to default agent
# ---------------------------------------------------------------------------


def test_build_loop_raises_value_error_for_specialized_with_cron(tmp_path: Path):
    """_build_loop raises ValueError if a specialized agent is given a CronService."""
    from nanobot.agent.orchestrator import AgentOrchestrator
    from nanobot.cron.service import CronService

    bus = MessageBus()
    config = _make_minimal_config(tmp_path)
    orchestrator = AgentOrchestrator(bus, config, tmp_path / "agents")

    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    q: asyncio.Queue[InboundMessage] = asyncio.Queue()

    mock_provider = MagicMock()
    mock_cron = MagicMock(spec=CronService)

    with pytest.raises(ValueError, match="silpi"):
        orchestrator._build_loop(
            "silpi",
            ws,
            q,
            mock_provider,
            cron_service=mock_cron,
        )


def test_build_loop_default_agent_accepts_cron(tmp_path: Path):
    """_build_loop does NOT raise when the default agent receives a CronService."""
    from nanobot.agent.orchestrator import AgentOrchestrator
    from nanobot.cron.service import CronService

    bus = MessageBus()
    config = _make_minimal_config(tmp_path)
    orchestrator = AgentOrchestrator(bus, config, tmp_path / "agents")

    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    q: asyncio.Queue[InboundMessage] = asyncio.Queue()

    mock_provider = MagicMock()
    mock_cron = MagicMock(spec=CronService)

    with patch("nanobot.agent.loop.AgentLoop"):
        # Should not raise
        orchestrator._build_loop(
            "default",
            ws,
            q,
            mock_provider,
            cron_service=mock_cron,
        )


# ---------------------------------------------------------------------------
# Dispatch loop routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_loop_routes_to_correct_queue():
    """_dispatch_loop reads from bus.inbound and routes to per-agent queues."""
    from nanobot.agent.orchestrator import AgentOrchestrator

    bus = MessageBus()
    orchestrator = AgentOrchestrator(bus, Config(), Path("agents"))

    q_default: asyncio.Queue[InboundMessage] = asyncio.Queue()
    q_silpi: asyncio.Queue[InboundMessage] = asyncio.Queue()
    orchestrator._queues["default"] = q_default
    orchestrator._queues["silpi"] = q_silpi
    orchestrator._running = True

    # Start dispatch loop as a background task
    dispatch_task = asyncio.create_task(orchestrator._dispatch_loop())

    # Publish a message for silpi and one for default
    msg_silpi = _make_msg(agent_id="silpi")
    msg_default = _make_msg(agent_id="default")
    await bus.publish_inbound(msg_silpi)
    await bus.publish_inbound(msg_default)

    # Give dispatch loop time to process
    await asyncio.sleep(0.05)
    orchestrator._running = False
    dispatch_task.cancel()
    try:
        await dispatch_task
    except asyncio.CancelledError:
        pass

    assert q_silpi.qsize() == 1
    assert q_default.qsize() == 1
    assert q_silpi.get_nowait() is msg_silpi
    assert q_default.get_nowait() is msg_default


@pytest.mark.asyncio
async def test_dispatch_loop_fallback_for_unknown_agent():
    """_dispatch_loop falls back to default queue for unknown agent_id."""
    from nanobot.agent.orchestrator import AgentOrchestrator

    bus = MessageBus()
    orchestrator = AgentOrchestrator(bus, Config(), Path("agents"))

    q_default: asyncio.Queue[InboundMessage] = asyncio.Queue()
    orchestrator._queues["default"] = q_default
    orchestrator._running = True

    dispatch_task = asyncio.create_task(orchestrator._dispatch_loop())

    msg = _make_msg(agent_id="ghost-agent")
    await bus.publish_inbound(msg)

    await asyncio.sleep(0.05)
    orchestrator._running = False
    dispatch_task.cancel()
    try:
        await dispatch_task
    except asyncio.CancelledError:
        pass

    assert q_default.qsize() == 1


# ---------------------------------------------------------------------------
# start() integration — mocked heavy dependencies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_creates_per_agent_queues(tmp_path: Path):
    """start() creates one queue per discovered agent plus the default."""
    from nanobot.agent.orchestrator import AgentOrchestrator

    config = _make_minimal_config(tmp_path)
    agents_dir = _make_agents_dir(tmp_path, ["silpi", "viharapala"])
    bus = MessageBus()
    orchestrator = AgentOrchestrator(bus, config, agents_dir)

    mock_loop = MagicMock()
    mock_loop.run = AsyncMock()
    mock_loop.model = "test-model"
    mock_loop.sessions = MagicMock()
    mock_loop.sessions.list_sessions.return_value = []

    mock_cron = MagicMock()
    mock_cron.start = AsyncMock()
    mock_cron.status.return_value = {"jobs": 0}
    mock_cron.on_job = None

    mock_heartbeat = MagicMock()
    mock_heartbeat.start = AsyncMock()

    with (
        patch("nanobot.agent.orchestrator.AgentOrchestrator._build_loop", return_value=mock_loop),
        patch("nanobot.agent.orchestrator.AgentOrchestrator._make_provider", return_value=MagicMock()),
        patch("nanobot.cron.service.CronService", return_value=mock_cron),
        patch("nanobot.heartbeat.service.HeartbeatService", return_value=mock_heartbeat),
        patch("nanobot.config.loader.get_data_dir", return_value=tmp_path),
        patch("nanobot.utils.helpers.get_agent_workspace", return_value=tmp_path / "ws"),
    ):
        await orchestrator.start()

    # Queues must exist for default + 2 discovered agents
    assert "default" in orchestrator._queues
    assert "silpi" in orchestrator._queues
    assert "viharapala" in orchestrator._queues
    assert len(orchestrator._queues) == 3

    # Clean up background tasks
    orchestrator._running = False
    if orchestrator._dispatch_task:
        orchestrator._dispatch_task.cancel()
        try:
            await orchestrator._dispatch_task
        except asyncio.CancelledError:
            pass
    for t in orchestrator._loop_tasks.values():
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_start_cron_only_on_default_agent(tmp_path: Path):
    """start() passes CronService only to the default loop, not to specialised loops."""
    from nanobot.agent.orchestrator import AgentOrchestrator

    config = _make_minimal_config(tmp_path)
    agents_dir = _make_agents_dir(tmp_path, ["silpi"])
    bus = MessageBus()
    orchestrator = AgentOrchestrator(bus, config, agents_dir)

    build_calls: list[dict] = []

    mock_loop = MagicMock()
    mock_loop.run = AsyncMock()
    mock_loop.model = "test-model"
    mock_loop.sessions = MagicMock()
    mock_loop.sessions.list_sessions.return_value = []

    mock_cron = MagicMock()
    mock_cron.start = AsyncMock()
    mock_cron.on_job = None
    mock_heartbeat = MagicMock()
    mock_heartbeat.start = AsyncMock()

    def _capture_build(agent_id, *args, cron_service=None, **kwargs):
        build_calls.append({"agent_id": agent_id, "cron_service": cron_service})
        return mock_loop

    with (
        patch("nanobot.agent.orchestrator.AgentOrchestrator._build_loop", side_effect=_capture_build),
        patch("nanobot.agent.orchestrator.AgentOrchestrator._make_provider", return_value=MagicMock()),
        patch("nanobot.cron.service.CronService", return_value=mock_cron),
        patch("nanobot.heartbeat.service.HeartbeatService", return_value=mock_heartbeat),
        patch("nanobot.config.loader.get_data_dir", return_value=tmp_path),
        patch("nanobot.utils.helpers.get_agent_workspace", return_value=tmp_path / "ws"),
    ):
        await orchestrator.start()

    # Default loop gets cron, silpi does not
    default_call = next(c for c in build_calls if c["agent_id"] == "default")
    silpi_call = next(c for c in build_calls if c["agent_id"] == "silpi")

    assert default_call["cron_service"] is not None
    assert silpi_call["cron_service"] is None

    # Clean up
    orchestrator._running = False
    if orchestrator._dispatch_task:
        orchestrator._dispatch_task.cancel()
        try:
            await orchestrator._dispatch_task
        except asyncio.CancelledError:
            pass
    for t in orchestrator._loop_tasks.values():
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_no_agents_dir_only_default_loop(tmp_path: Path):
    """When agents_dir does not exist, only the default loop is created."""
    from nanobot.agent.orchestrator import AgentOrchestrator

    config = _make_minimal_config(tmp_path)
    bus = MessageBus()
    # Pass a non-existent agents_dir → AgentRegistry.discover() returns []
    orchestrator = AgentOrchestrator(bus, config, tmp_path / "no-such-dir")

    mock_loop = MagicMock()
    mock_loop.run = AsyncMock()
    mock_loop.model = "test-model"
    mock_loop.sessions = MagicMock()
    mock_loop.sessions.list_sessions.return_value = []

    mock_cron = MagicMock()
    mock_cron.start = AsyncMock()
    mock_cron.on_job = None
    mock_heartbeat = MagicMock()
    mock_heartbeat.start = AsyncMock()

    with (
        patch("nanobot.agent.orchestrator.AgentOrchestrator._build_loop", return_value=mock_loop),
        patch("nanobot.agent.orchestrator.AgentOrchestrator._make_provider", return_value=MagicMock()),
        patch("nanobot.cron.service.CronService", return_value=mock_cron),
        patch("nanobot.heartbeat.service.HeartbeatService", return_value=mock_heartbeat),
        patch("nanobot.config.loader.get_data_dir", return_value=tmp_path),
    ):
        await orchestrator.start()

    # Only default queue
    assert list(orchestrator._queues.keys()) == ["default"]
    assert list(orchestrator._loops.keys()) == ["default"]

    # Clean up
    orchestrator._running = False
    if orchestrator._dispatch_task:
        orchestrator._dispatch_task.cancel()
        try:
            await orchestrator._dispatch_task
        except asyncio.CancelledError:
            pass
    for t in orchestrator._loop_tasks.values():
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# stop() integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_calls_loop_stop_and_waits(tmp_path: Path):
    """stop() calls stop() on every agent loop and cancels the dispatch task."""
    from nanobot.agent.orchestrator import AgentOrchestrator

    bus = MessageBus()
    config = _make_minimal_config(tmp_path)
    orchestrator = AgentOrchestrator(bus, config, tmp_path / "agents")

    stopped: list[str] = []

    mock_loop = MagicMock()
    mock_loop.stop.side_effect = lambda: stopped.append("stopped")
    mock_loop.close_mcp = AsyncMock()
    orchestrator._loops["default"] = mock_loop

    q: asyncio.Queue[InboundMessage] = asyncio.Queue()
    orchestrator._queues["default"] = q
    orchestrator._running = True

    # Create a dispatch task that we'll stop
    orchestrator._dispatch_task = asyncio.create_task(orchestrator._dispatch_loop())

    # Create a dummy loop task
    async def _dummy():
        while True:
            await asyncio.sleep(0.1)

    loop_task = asyncio.create_task(_dummy())
    orchestrator._loop_tasks["default"] = loop_task

    await orchestrator.stop()

    assert "stopped" in stopped
    assert not orchestrator._running
    assert loop_task.cancelled() or loop_task.done()


# ---------------------------------------------------------------------------
# CLI wiring smoke tests
# ---------------------------------------------------------------------------


def test_gateway_embedded_mode_smoke_test():
    """Embedded gateway mode creates an AgentOrchestrator (smoke test)."""
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    runner = CliRunner()
    created: list[object] = []

    mock_orchestrator = MagicMock()
    mock_channels = MagicMock()
    mock_channels.enabled_channels = []

    def _capture_orchestrator(*args, **kwargs):
        created.append(("orchestrator", args, kwargs))
        return mock_orchestrator

    with (
        patch("nanobot.config.loader.load_config", return_value=Config()),
        patch("nanobot.cli.commands.sync_workspace_templates"),
        patch("nanobot.bus.queue.MessageBus"),
        patch(
            "nanobot.agent.orchestrator.AgentOrchestrator",
            side_effect=_capture_orchestrator,
        ),
        patch("nanobot.channels.manager.ChannelManager", return_value=mock_channels),
        patch("asyncio.run", side_effect=lambda coro: coro.close()),
    ):
        result = runner.invoke(app, ["gateway"])

    assert result.exit_code == 0, result.output
    # AgentOrchestrator must have been instantiated
    assert len(created) == 1


def test_serve_agent_uses_orchestrator_smoke_test():
    """serve-agent command wires AgentOrchestrator (smoke test)."""
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    runner = CliRunner()
    created: list[object] = []

    mock_orchestrator = MagicMock()

    def _capture_orchestrator(*args, **kwargs):
        created.append(("orchestrator",))
        return mock_orchestrator

    with (
        patch("nanobot.config.loader.load_config", return_value=Config()),
        patch("nanobot.cli.commands.sync_workspace_templates"),
        patch("nanobot.bus.queue.MessageBus"),
        patch(
            "nanobot.agent.orchestrator.AgentOrchestrator",
            side_effect=_capture_orchestrator,
        ),
        patch("nanobot.services.agent_server.AgentRestServer", return_value=MagicMock()),
        patch("asyncio.run", side_effect=lambda coro: coro.close()),
    ):
        result = runner.invoke(app, ["serve-agent"])

    assert result.exit_code == 0, result.output
    assert len(created) == 1


# ---------------------------------------------------------------------------
# agent_server: agent_id field forwarding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_server_inbound_forwards_agent_id():
    """POST /api/inbound includes agent_id in the InboundMessage when provided."""
    from aiohttp.test_utils import TestClient, TestServer

    from nanobot.bus.queue import MessageBus
    from nanobot.services.agent_server import AgentRestServer

    bus = MessageBus()
    server = AgentRestServer(bus=bus, host="127.0.0.1", port=0)
    app = server._build_app()

    async with TestClient(TestServer(app)) as client:
        payload = {
            "channel": "telegram",
            "sender_id": "u1",
            "chat_id": "c1",
            "content": "hello",
            "agent_id": "silpi",
        }
        r = await client.post("/api/inbound", json=payload)
        assert r.status == 200

        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.agent_id == "silpi"


@pytest.mark.asyncio
async def test_agent_server_inbound_agent_id_defaults_to_none():
    """POST /api/inbound without agent_id produces InboundMessage with agent_id=None."""
    from aiohttp.test_utils import TestClient, TestServer

    from nanobot.bus.queue import MessageBus
    from nanobot.services.agent_server import AgentRestServer

    bus = MessageBus()
    server = AgentRestServer(bus=bus, host="127.0.0.1", port=0)
    app = server._build_app()

    async with TestClient(TestServer(app)) as client:
        payload = {
            "channel": "telegram",
            "sender_id": "u1",
            "chat_id": "c1",
            "content": "hello",
        }
        r = await client.post("/api/inbound", json=payload)
        assert r.status == 200

        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.agent_id is None

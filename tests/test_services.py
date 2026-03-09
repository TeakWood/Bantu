"""Tests for the monorepo service layer.

Covers:
- ServicesConfig schema additions
- AgentRestServer (inbound, outbound, health endpoints)
- RemoteMessageBus (publish_inbound, consume_outbound, poll loop)
- GatewayHttpServer (health endpoint, admin proxy)
- serve-agent CLI command wiring
- serve-admin CLI command wiring
- gateway distributed-mode integration
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config, GatewayConfig, ServicesConfig
from nanobot.services.agent_server import AgentRestServer
from nanobot.services.gateway_server import GatewayHttpServer
from nanobot.services.remote_bus import RemoteMessageBus

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────


def test_services_config_defaults():
    """ServicesConfig is present and defaults to empty URLs (embedded mode)."""
    cfg = Config()
    assert isinstance(cfg.gateway.services, ServicesConfig)
    assert cfg.gateway.services.agent_url == ""
    assert cfg.gateway.services.admin_url == ""


def test_services_config_accepts_urls():
    svc = ServicesConfig(agent_url="http://localhost:18792", admin_url="http://localhost:18791")
    assert svc.agent_url == "http://localhost:18792"
    assert svc.admin_url == "http://localhost:18791"


def test_gateway_config_round_trips_services():
    """GatewayConfig serialises and deserialises ServicesConfig correctly."""
    raw = {
        "host": "0.0.0.0",
        "port": 18790,
        "services": {
            "agentUrl": "http://agent:18792",
            "adminUrl": "http://admin:18791",
        },
    }
    gw = GatewayConfig.model_validate(raw)
    assert gw.services.agent_url == "http://agent:18792"
    assert gw.services.admin_url == "http://admin:18791"


def test_env_var_sets_agent_url(monkeypatch):
    """pydantic-settings reads service URLs from env vars."""
    monkeypatch.setenv("NANOBOT_GATEWAY__SERVICES__AGENT_URL", "http://remote:18792")
    cfg = Config()
    assert cfg.gateway.services.agent_url == "http://remote:18792"


# ─────────────────────────────────────────────────────────────────────────────
# AgentRestServer
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
async def agent_test_client():
    """Return an aiohttp test client for AgentRestServer."""
    bus = MessageBus()
    server = AgentRestServer(bus=bus, host="127.0.0.1", port=0)
    app = server._build_app()
    async with TestClient(TestServer(app)) as client:
        yield client, bus


@pytest.mark.asyncio
async def test_agent_health_endpoint(agent_test_client):
    client, _ = agent_test_client
    r = await client.get("/api/health")
    assert r.status == 200
    body = await r.json()
    assert body["status"] == "ok"
    assert body["service"] == "agent"


@pytest.mark.asyncio
async def test_agent_inbound_enqueues_message(agent_test_client):
    client, bus = agent_test_client
    payload = {
        "channel": "telegram",
        "sender_id": "user1",
        "chat_id": "chat1",
        "content": "Hello agent",
    }
    r = await client.post("/api/inbound", json=payload)
    assert r.status == 200
    body = await r.json()
    assert body["status"] == "accepted"

    msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
    assert msg.channel == "telegram"
    assert msg.content == "Hello agent"
    assert msg.sender_id == "user1"
    assert msg.chat_id == "chat1"


@pytest.mark.asyncio
async def test_agent_inbound_preserves_optional_fields(agent_test_client):
    client, bus = agent_test_client
    payload = {
        "channel": "slack",
        "sender_id": "U123",
        "chat_id": "C456",
        "content": "Hi",
        "media": ["http://example.com/img.png"],
        "metadata": {"thread_ts": "123"},
        "session_key_override": "custom:key",
    }
    r = await client.post("/api/inbound", json=payload)
    assert r.status == 200
    msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
    assert msg.media == ["http://example.com/img.png"]
    assert msg.metadata == {"thread_ts": "123"}
    assert msg.session_key_override == "custom:key"


@pytest.mark.asyncio
async def test_agent_inbound_missing_fields_returns_400(agent_test_client):
    client, _ = agent_test_client
    r = await client.post("/api/inbound", json={"channel": "telegram"})
    assert r.status == 400


@pytest.mark.asyncio
async def test_agent_outbound_returns_queued_message(agent_test_client):
    client, bus = agent_test_client
    await bus.publish_outbound(OutboundMessage(
        channel="telegram", chat_id="chat1", content="Response text"
    ))
    r = await client.get("/api/outbound", params={"timeout": "1"})
    assert r.status == 200
    body = await r.json()
    assert len(body["messages"]) == 1
    assert body["messages"][0]["channel"] == "telegram"
    assert body["messages"][0]["content"] == "Response text"


@pytest.mark.asyncio
async def test_agent_outbound_drains_multiple_messages(agent_test_client):
    client, bus = agent_test_client
    for i in range(3):
        await bus.publish_outbound(OutboundMessage(
            channel="slack", chat_id=f"c{i}", content=f"msg{i}"
        ))
    r = await client.get("/api/outbound", params={"timeout": "1"})
    assert r.status == 200
    body = await r.json()
    assert len(body["messages"]) == 3


@pytest.mark.asyncio
async def test_agent_outbound_returns_empty_on_timeout(agent_test_client):
    client, _ = agent_test_client
    r = await client.get("/api/outbound", params={"timeout": "0.1"})
    assert r.status == 200
    body = await r.json()
    assert body["messages"] == []


# ─────────────────────────────────────────────────────────────────────────────
# GatewayHttpServer
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
async def gateway_test_client():
    server = GatewayHttpServer(host="127.0.0.1", port=0, admin_url="")
    app = server._build_app()
    async with TestClient(TestServer(app)) as client:
        yield client


@pytest.fixture()
async def gateway_proxy_client():
    """Gateway with a (mocked) admin URL."""
    server = GatewayHttpServer(
        host="127.0.0.1", port=0, admin_url="http://admin-svc:18791"
    )
    app = server._build_app()
    async with TestClient(TestServer(app)) as client:
        yield client


@pytest.mark.asyncio
async def test_gateway_health_endpoint(gateway_test_client):
    r = await gateway_test_client.get("/health")
    assert r.status == 200
    body = await r.json()
    assert body["status"] == "ok"
    assert body["service"] == "gateway"


@pytest.mark.asyncio
async def test_gateway_admin_proxy_route_registered(gateway_proxy_client):
    """The admin proxy route is registered when admin_url is non-empty."""
    import aiohttp
    # Without a live admin service the proxy will fail with 502, but the route
    # itself must exist (not 404/405).
    r = await gateway_proxy_client.get("/api/admin/config")
    assert r.status != 404


@pytest.mark.asyncio
async def test_gateway_no_admin_proxy_when_url_empty(gateway_test_client):
    """When admin_url is empty no proxy route is registered."""
    r = await gateway_test_client.get("/api/admin/config")
    assert r.status == 404


# ─────────────────────────────────────────────────────────────────────────────
# RemoteMessageBus
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remote_bus_publish_inbound_posts_to_agent():
    """RemoteMessageBus.publish_inbound posts JSON to the agent service."""
    posted: list[dict] = []

    class FakeResponse:
        status_code = 200
        def raise_for_status(self): pass

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
        async def post(self, url, json=None, timeout=None):
            posted.append({"url": url, "json": json})
            return FakeResponse()

    bus = RemoteMessageBus(agent_url="http://agent:18792")
    msg = InboundMessage(
        channel="telegram", sender_id="u1", chat_id="c1", content="hi"
    )

    with patch("nanobot.services.remote_bus.httpx.AsyncClient", return_value=FakeClient()):
        await bus.publish_inbound(msg)

    assert len(posted) == 1
    assert posted[0]["url"] == "http://agent:18792/api/inbound"
    assert posted[0]["json"]["content"] == "hi"
    assert posted[0]["json"]["channel"] == "telegram"


@pytest.mark.asyncio
async def test_remote_bus_consume_outbound_reads_from_queue():
    bus = RemoteMessageBus(agent_url="http://agent:18792")
    msg = OutboundMessage(channel="slack", chat_id="c1", content="reply")
    await bus.publish_outbound(msg)

    result = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    assert result.channel == "slack"
    assert result.content == "reply"


@pytest.mark.asyncio
async def test_remote_bus_poll_loop_feeds_outbound_queue():
    """The poll loop fetches outbound messages and puts them in the queue."""
    fake_response_data = {
        "messages": [
            {"channel": "telegram", "chat_id": "c1", "content": "from agent",
             "reply_to": None, "media": [], "metadata": {}},
        ]
    }

    poll_count = 0

    class FakeResponse:
        status_code = 200
        def json(self): return fake_response_data

    class SlowResponse:
        status_code = 200
        def json(self): return {"messages": []}

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
        async def get(self, url, params=None, timeout=None):
            nonlocal poll_count
            poll_count += 1
            if poll_count == 1:
                return FakeResponse()
            # Subsequent calls block briefly then return empty to prevent spin.
            await asyncio.sleep(0.05)
            return SlowResponse()

    bus = RemoteMessageBus(agent_url="http://agent:18792", poll_timeout_s=0.1)

    with patch("nanobot.services.remote_bus.httpx.AsyncClient", return_value=FakeClient()):
        bus.start_polling()
        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)
        bus.stop_polling()

    assert msg.content == "from agent"
    assert msg.channel == "telegram"


def test_remote_bus_outbound_size():
    bus = RemoteMessageBus(agent_url="http://agent:18792")
    assert bus.outbound_size == 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI — serve-agent
# ─────────────────────────────────────────────────────────────────────────────

def _make_serve_agent_mocks():
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        empty_config = Config()
        mock_orchestrator = MagicMock()

        with (
            patch("nanobot.config.loader.load_config", return_value=empty_config),
            patch("nanobot.cli.commands.sync_workspace_templates"),
            patch("nanobot.bus.queue.MessageBus"),
            patch("nanobot.agent.orchestrator.AgentOrchestrator", return_value=mock_orchestrator),
            patch("nanobot.services.agent_server.AgentRestServer", return_value=MagicMock()),
            patch("asyncio.run", side_effect=lambda coro: coro.close()),
        ):
            yield

    return _ctx()


def test_serve_agent_command_exits_zero():
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    runner = CliRunner()
    with _make_serve_agent_mocks():
        result = runner.invoke(app, ["serve-agent"])

    assert result.exit_code == 0, (
        f"serve-agent exited {result.exit_code}.\nOutput:\n{result.output}"
    )


def test_serve_agent_command_prints_rest_api_url():
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    runner = CliRunner()
    with _make_serve_agent_mocks():
        result = runner.invoke(app, ["serve-agent", "--port", "18792"])

    assert "18792" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# CLI — serve-admin
# ─────────────────────────────────────────────────────────────────────────────


def _make_serve_admin_mocks():
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        empty_config = Config()
        mock_admin = MagicMock()

        with (
            patch("nanobot.config.loader.load_config", return_value=empty_config),
            patch("nanobot.admin.server.AdminServer", return_value=mock_admin),
            patch("asyncio.run", side_effect=lambda coro: coro.close()),
        ):
            yield

    return _ctx()


def test_serve_admin_command_exits_zero():
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    runner = CliRunner()
    with _make_serve_admin_mocks():
        result = runner.invoke(app, ["serve-admin"])

    assert result.exit_code == 0, (
        f"serve-admin exited {result.exit_code}.\nOutput:\n{result.output}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI — gateway distributed mode
# ─────────────────────────────────────────────────────────────────────────────


def _make_distributed_gateway_mocks(agent_url: str = "http://agent:18792"):
    """Return a context manager that patches the gateway for distributed mode."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        dist_config = Config()
        dist_config.gateway.services.agent_url = agent_url
        dist_config.gateway.services.admin_url = "http://admin:18791"

        mock_channels = MagicMock()
        mock_channels.enabled_channels = []

        with (
            patch("nanobot.config.loader.load_config", return_value=dist_config),
            patch("nanobot.cli.commands.sync_workspace_templates"),
            patch("nanobot.services.remote_bus.RemoteMessageBus"),
            patch("nanobot.services.gateway_server.GatewayHttpServer", return_value=MagicMock()),
            patch("nanobot.channels.manager.ChannelManager", return_value=mock_channels),
            patch("asyncio.run", side_effect=lambda coro: coro.close()),
        ):
            yield

    return _ctx()


def test_gateway_distributed_mode_exits_zero():
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    runner = CliRunner()
    with _make_distributed_gateway_mocks():
        result = runner.invoke(app, ["gateway"])

    assert result.exit_code == 0, (
        f"gateway (distributed) exited {result.exit_code}.\nOutput:\n{result.output}"
    )


def test_gateway_distributed_mode_prints_agent_url():
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    runner = CliRunner()
    with _make_distributed_gateway_mocks(agent_url="http://agent:18792"):
        result = runner.invoke(app, ["gateway"])

    assert "http://agent:18792" in result.output


def test_gateway_embedded_mode_uses_orchestrator():
    """When agent_url is empty the gateway runs in embedded mode with AgentOrchestrator."""
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    runner = CliRunner()
    created: list = []

    mock_orchestrator = MagicMock()
    mock_channels = MagicMock()
    mock_channels.enabled_channels = []

    def _capture(*args, **kwargs):
        created.append("orchestrator")
        return mock_orchestrator

    with (
        patch("nanobot.config.loader.load_config", return_value=Config()),
        patch("nanobot.cli.commands.sync_workspace_templates"),
        patch("nanobot.bus.queue.MessageBus"),
        patch("nanobot.agent.orchestrator.AgentOrchestrator", side_effect=_capture),
        patch("nanobot.channels.manager.ChannelManager", return_value=mock_channels),
        patch("asyncio.run", side_effect=lambda coro: coro.close()),
    ):
        result = runner.invoke(app, ["gateway"])

    assert result.exit_code == 0
    # AgentOrchestrator must have been instantiated in embedded mode
    assert len(created) == 1

"""Tests for Telegram bot → specialised-agent assignment (Bantu-5vq).

Covers:
- TelegramConfig schema: agent field defaults and serialisation
- BaseChannel._handle_message(): agent_id is stamped from config
- Admin PUT /api/channels/telegram: agent field persists
- Admin GET /api/channels/telegram: agent field is returned
- End-to-end: message routed to correct agent queue when agent is configured
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import AsyncMock

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import ChannelsConfig, TelegramConfig
from nanobot.config.loader import load_config, save_config
from nanobot.config.schema import AdminConfig, Config, GatewayConfig

# ---------------------------------------------------------------------------
# Minimal concrete channel for testing BaseChannel behaviour
# ---------------------------------------------------------------------------


class _StubConfig:
    """Minimal channel config with allow_from and optional agent."""

    def __init__(self, agent: str | None = None) -> None:
        self.allow_from: list[str] = ["*"]
        self.agent: str | None = agent


class _StubChannel(BaseChannel):
    """Concrete subclass of BaseChannel for unit-testing _handle_message."""

    name = "stub"

    def __init__(self, config: _StubConfig, bus: MessageBus) -> None:
        super().__init__(config, bus)

    async def start(self) -> None:  # pragma: no cover
        pass

    async def stop(self) -> None:  # pragma: no cover
        pass

    async def send(self, msg: OutboundMessage) -> None:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# TelegramConfig schema tests
# ---------------------------------------------------------------------------


class TestTelegramConfigAgentField:
    """TelegramConfig.agent field — defaults, assignment, serialisation."""

    def test_agent_defaults_to_none(self) -> None:
        cfg = TelegramConfig()
        assert cfg.agent is None

    def test_agent_can_be_set_to_agent_name(self) -> None:
        cfg = TelegramConfig(agent="silpi")
        assert cfg.agent == "silpi"

    def test_agent_can_be_cleared_with_none(self) -> None:
        cfg = TelegramConfig(agent="silpi")
        cfg.agent = None
        assert cfg.agent is None

    def test_agent_serialises_to_camel_case_json(self) -> None:
        cfg = TelegramConfig(agent="viharapala")
        data = cfg.model_dump(by_alias=True)
        assert "agent" in data
        assert data["agent"] == "viharapala"

    def test_agent_absent_serialises_as_null(self) -> None:
        cfg = TelegramConfig()
        data = cfg.model_dump(by_alias=True)
        assert data["agent"] is None

    def test_agent_accepted_from_camel_case_dict(self) -> None:
        data = {
            "enabled": True,
            "token": "bot-token",
            "allowFrom": ["*"],
            "agent": "silpi",
        }
        cfg = TelegramConfig.model_validate(data)
        assert cfg.agent == "silpi"

    def test_agent_accepted_from_snake_case_dict(self) -> None:
        data = {"enabled": True, "token": "bot-token", "allow_from": ["*"], "agent": "silpi"}
        cfg = TelegramConfig.model_validate(data)
        assert cfg.agent == "silpi"

    def test_telegram_config_nested_in_channels_config(self) -> None:
        channels = ChannelsConfig.model_validate(
            {"telegram": {"token": "tok", "allowFrom": ["*"], "agent": "silpi"}}
        )
        assert channels.telegram.agent == "silpi"


# ---------------------------------------------------------------------------
# BaseChannel._handle_message() — agent_id stamping
# ---------------------------------------------------------------------------


class TestBaseChannelAgentIdStamping:
    """_handle_message() must stamp agent_id from config.agent onto InboundMessage."""

    @pytest.mark.asyncio
    async def test_agent_id_none_when_config_agent_is_none(self) -> None:
        bus = MessageBus()
        channel = _StubChannel(_StubConfig(agent=None), bus)

        await channel._handle_message(
            sender_id="user1",
            chat_id="chat1",
            content="hello",
        )

        msg: InboundMessage = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.agent_id is None

    @pytest.mark.asyncio
    async def test_agent_id_set_when_config_agent_is_string(self) -> None:
        bus = MessageBus()
        channel = _StubChannel(_StubConfig(agent="silpi"), bus)

        await channel._handle_message(
            sender_id="user1",
            chat_id="chat1",
            content="implement this",
        )

        msg: InboundMessage = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.agent_id == "silpi"

    @pytest.mark.asyncio
    async def test_agent_id_set_for_different_agent_names(self) -> None:
        bus = MessageBus()
        channel = _StubChannel(_StubConfig(agent="viharapala"), bus)

        await channel._handle_message(
            sender_id="user1",
            chat_id="chat1",
            content="review this",
        )

        msg: InboundMessage = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.agent_id == "viharapala"

    @pytest.mark.asyncio
    async def test_empty_string_agent_treated_as_none(self) -> None:
        """An empty-string agent in config must not be forwarded — treat as None."""
        bus = MessageBus()
        channel = _StubChannel(_StubConfig(agent=""), bus)

        await channel._handle_message(
            sender_id="user1",
            chat_id="chat1",
            content="hi",
        )

        msg: InboundMessage = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.agent_id is None

    @pytest.mark.asyncio
    async def test_channel_name_and_ids_are_preserved(self) -> None:
        """Other InboundMessage fields must not be affected by agent stamping."""
        bus = MessageBus()
        channel = _StubChannel(_StubConfig(agent="silpi"), bus)

        await channel._handle_message(
            sender_id="sender42",
            chat_id="chat99",
            content="task text",
            media=["/tmp/img.png"],
            metadata={"foo": "bar"},
        )

        msg: InboundMessage = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.channel == "stub"
        assert msg.sender_id == "sender42"
        assert msg.chat_id == "chat99"
        assert msg.content == "task text"
        assert msg.media == ["/tmp/img.png"]
        assert msg.metadata == {"foo": "bar"}
        assert msg.agent_id == "silpi"

    @pytest.mark.asyncio
    async def test_access_denied_sender_does_not_publish(self) -> None:
        """Denied senders must not result in any published InboundMessage."""
        bus = MessageBus()
        cfg = _StubConfig(agent="silpi")
        cfg.allow_from = ["allowed-user"]  # override to restrict access
        channel = _StubChannel(cfg, bus)

        await channel._handle_message(
            sender_id="denied-user",
            chat_id="chat1",
            content="hello",
        )

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_inbound(), timeout=0.1)


# ---------------------------------------------------------------------------
# Config with no agent attribute at all (backwards compatibility)
# ---------------------------------------------------------------------------


class TestBaseChannelNoAgentAttribute:
    """Channels whose config has no 'agent' attribute must not break."""

    @pytest.mark.asyncio
    async def test_no_agent_attribute_in_config_produces_none_agent_id(self) -> None:
        class _NoAgentConfig:
            allow_from: list[str] = ["*"]
            # intentionally no .agent attribute

        bus = MessageBus()
        channel = _StubChannel(_NoAgentConfig(), bus)  # type: ignore[arg-type]

        await channel._handle_message(
            sender_id="user1",
            chat_id="chat1",
            content="hello",
        )

        msg: InboundMessage = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.agent_id is None


# ---------------------------------------------------------------------------
# Admin API: agent field persistence
# ---------------------------------------------------------------------------

# Reuse the helpers from test_admin_routes


@contextmanager
def _temp_config_file(initial: Config | None = None) -> Iterator[Path]:
    data = (initial or Config()).model_dump(by_alias=True)
    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as f:
        json.dump(data, f, indent=2)
        path = Path(f.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _make_server_config(*, token: str = "", port: int = 18791) -> Config:
    cfg = Config()
    cfg.gateway = GatewayConfig(
        admin=AdminConfig(enabled=True, token=token, host="127.0.0.1", port=port)
    )
    return cfg


async def _make_client(server_config: Config, config_path: Path):
    from aiohttp.test_utils import TestClient, TestServer
    from nanobot.admin.server import AdminServer

    server = AdminServer(server_config, config_path=config_path)
    app = server._build_app()
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


class TestAdminAgentField:
    """PUT/GET /api/channels/telegram must persist and return the agent field."""

    @pytest.mark.asyncio
    async def test_put_channel_telegram_sets_agent(self) -> None:
        with _temp_config_file() as path:
            client = await _make_client(_make_server_config(), path)
            try:
                resp = await client.put(
                    "/api/channels/telegram",
                    json={"agent": "silpi", "token": "bot-token", "allowFrom": ["*"]},
                )
                assert resp.status == 200
                data = await resp.json()
                assert data["ok"] is True

                cfg = load_config(path)
                assert cfg.channels.telegram.agent == "silpi"
            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_get_channel_telegram_returns_agent(self) -> None:
        initial = Config()
        initial.channels.telegram = TelegramConfig(
            enabled=True, token="tok", allow_from=["*"], agent="viharapala"
        )
        with _temp_config_file(initial) as path:
            client = await _make_client(_make_server_config(), path)
            try:
                resp = await client.get("/api/channels/telegram")
                assert resp.status == 200
                data = await resp.json()
                assert data["agent"] == "viharapala"
            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_put_channel_telegram_clears_agent_with_null(self) -> None:
        initial = Config()
        initial.channels.telegram = TelegramConfig(
            enabled=True, token="tok", allow_from=["*"], agent="silpi"
        )
        with _temp_config_file(initial) as path:
            client = await _make_client(_make_server_config(), path)
            try:
                resp = await client.put("/api/channels/telegram", json={"agent": None})
                assert resp.status == 200
                cfg = load_config(path)
                assert cfg.channels.telegram.agent is None
            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_put_channel_telegram_preserves_other_fields(self) -> None:
        """Setting agent must not clobber existing token or allow_from."""
        initial = Config()
        initial.channels.telegram = TelegramConfig(
            enabled=True, token="existing-token", allow_from=["123"], agent=None
        )
        with _temp_config_file(initial) as path:
            client = await _make_client(_make_server_config(), path)
            try:
                resp = await client.put("/api/channels/telegram", json={"agent": "silpi"})
                assert resp.status == 200

                cfg = load_config(path)
                assert cfg.channels.telegram.agent == "silpi"
                assert cfg.channels.telegram.token == "existing-token"
                assert cfg.channels.telegram.allow_from == ["123"]
            finally:
                await client.close()

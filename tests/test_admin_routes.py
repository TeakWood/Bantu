"""Unit tests for nanobot.admin.routes — REST API endpoints."""

from __future__ import annotations

import asyncio
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nanobot.admin.server import AdminServer
from nanobot.config.loader import load_config, save_config
from nanobot.config.schema import (
    AdminConfig,
    Config,
    GatewayConfig,
    MCPServerConfig,
    TelegramConfig,
    ChannelsConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server_config(*, token: str = "", port: int = 18791) -> Config:
    """Config whose gateway.admin fields drive the server's auth/port settings."""
    cfg = Config()
    cfg.gateway = GatewayConfig(
        admin=AdminConfig(enabled=True, token=token, host="127.0.0.1", port=port)
    )
    return cfg


@contextmanager
def _temp_config_file(initial: Config | None = None) -> Iterator[Path]:
    """Write *initial* (or a blank Config) to a temp file; yield the path."""
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


async def _client(server_config: Config, config_path: Path) -> TestClient:
    """Build a started TestClient for AdminServer using the given config file."""
    server = AdminServer(server_config, config_path=config_path)
    app = server._build_app()
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


# ---------------------------------------------------------------------------
# GET /api/config — full config with sensitive fields masked
# ---------------------------------------------------------------------------


async def test_get_config_returns_200():
    """GET /api/config returns 200 with a JSON object."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/config")
            assert resp.status == 200
            data = await resp.json()
            assert isinstance(data, dict)
        finally:
            await client.close()


async def test_get_config_masks_sensitive_fields():
    """Sensitive fields (token, key, secret, password, credential) become '****'."""
    initial = Config()
    initial.providers.anthropic.api_key = "sk-ant-realkey"
    initial.channels.telegram.token = "bot123"

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/config")
            data = await resp.json()

            # anthropic.apiKey should be masked
            assert data["providers"]["anthropic"]["apiKey"] == "****"
            # telegram.token should be masked
            assert data["channels"]["telegram"]["token"] == "****"
        finally:
            await client.close()


async def test_get_config_empty_sensitive_fields_not_masked():
    """Empty-string sensitive fields should NOT be masked (indicates not configured)."""
    initial = Config()
    # api_key defaults to "" — should stay ""

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/config")
            data = await resp.json()
            assert data["providers"]["anthropic"]["apiKey"] == ""
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# GET /api/providers
# ---------------------------------------------------------------------------


async def test_get_providers_returns_list():
    """GET /api/providers returns a list of provider objects."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/providers")
            assert resp.status == 200
            data = await resp.json()
            assert isinstance(data, list)
            assert len(data) > 0
        finally:
            await client.close()


async def test_get_providers_fields():
    """Each provider object has the required fields."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/providers")
            data = await resp.json()
            for item in data:
                assert "name" in item
                assert "label" in item
                assert "has_key" in item
                assert "api_base" in item
                assert "is_oauth" in item
                assert "is_local" in item
        finally:
            await client.close()


async def test_get_providers_has_key_true_when_set():
    """has_key is True when the provider has a non-empty api_key."""
    initial = Config()
    initial.providers.anthropic.api_key = "sk-ant-test"

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/providers")
            data = await resp.json()
            anthropic = next(p for p in data if p["name"] == "anthropic")
            assert anthropic["has_key"] is True
        finally:
            await client.close()


async def test_get_providers_has_key_false_when_empty():
    """has_key is False when api_key is empty."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/providers")
            data = await resp.json()
            anthropic = next(p for p in data if p["name"] == "anthropic")
            assert anthropic["has_key"] is False
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# PUT /api/providers/{name}
# ---------------------------------------------------------------------------


async def test_put_provider_update_api_key():
    """PUT /api/providers/{name} updates api_key and persists it."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/providers/anthropic",
                json={"api_key": "sk-ant-updated"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True

            # Verify persisted to disk
            cfg = load_config(path)
            assert cfg.providers.anthropic.api_key == "sk-ant-updated"
        finally:
            await client.close()


async def test_put_provider_update_api_base():
    """PUT /api/providers/{name} updates api_base."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/providers/openai",
                json={"api_base": "https://custom.openai.example.com/v1"},
            )
            assert resp.status == 200
            cfg = load_config(path)
            assert cfg.providers.openai.api_base == "https://custom.openai.example.com/v1"
        finally:
            await client.close()


async def test_put_provider_update_extra_headers():
    """PUT /api/providers/{name} updates extra_headers."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/providers/aihubmix",
                json={"extra_headers": {"APP-Code": "my-code"}},
            )
            assert resp.status == 200
            cfg = load_config(path)
            assert cfg.providers.aihubmix.extra_headers == {"APP-Code": "my-code"}
        finally:
            await client.close()


async def test_put_provider_clear_api_base_with_null():
    """api_base can be cleared by setting to null."""
    initial = Config()
    initial.providers.openai.api_base = "https://old.example.com"

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/providers/openai",
                json={"api_base": None},
            )
            assert resp.status == 200
            cfg = load_config(path)
            assert cfg.providers.openai.api_base is None
        finally:
            await client.close()


async def test_put_provider_unknown_name_404():
    """PUT /api/providers/{name} with an unknown provider name returns 404."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/providers/nonexistent",
                json={"api_key": "x"},
            )
            assert resp.status == 404
        finally:
            await client.close()


async def test_put_provider_invalid_json_400():
    """PUT /api/providers/{name} with non-JSON body returns 400."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/providers/anthropic",
                data=b"not-json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
        finally:
            await client.close()


async def test_put_provider_extra_headers_not_object_400():
    """extra_headers must be an object (dict) — a string value returns 400."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/providers/anthropic",
                json={"extra_headers": "not-a-dict"},
            )
            assert resp.status == 400
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# GET /api/channels
# ---------------------------------------------------------------------------


async def test_get_channels_returns_list():
    """GET /api/channels returns a list of channel summaries."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/channels")
            assert resp.status == 200
            data = await resp.json()
            assert isinstance(data, list)
        finally:
            await client.close()


async def test_get_channels_all_channels_present():
    """GET /api/channels includes all known channel names."""
    expected = {
        "whatsapp", "telegram", "discord", "feishu", "mochat",
        "dingtalk", "email", "slack", "qq", "matrix",
    }
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/channels")
            data = await resp.json()
            names = {item["name"] for item in data}
            assert expected == names
        finally:
            await client.close()


async def test_get_channels_fields():
    """Each channel entry has name, enabled, and summary fields."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/channels")
            data = await resp.json()
            for item in data:
                assert "name" in item
                assert "enabled" in item
                assert "summary" in item
        finally:
            await client.close()


async def test_get_channels_enabled_status():
    """Enabled field reflects current channel config."""
    initial = Config()
    initial.channels.telegram = TelegramConfig(enabled=True, token="tok")

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/channels")
            data = await resp.json()
            tg = next(c for c in data if c["name"] == "telegram")
            assert tg["enabled"] is True
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# GET /api/channels/{name}
# ---------------------------------------------------------------------------


async def test_get_channel_returns_full_config():
    """GET /api/channels/{name} returns the full channel config dict."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/channels/telegram")
            assert resp.status == 200
            data = await resp.json()
            assert "enabled" in data
        finally:
            await client.close()


async def test_get_channel_masks_sensitive_fields():
    """GET /api/channels/{name} masks sensitive values."""
    initial = Config()
    initial.channels.telegram = TelegramConfig(enabled=True, token="secret-bot-token")

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/channels/telegram")
            data = await resp.json()
            assert data["token"] == "****"
        finally:
            await client.close()


async def test_get_channel_unknown_name_404():
    """GET /api/channels/{name} with unknown channel returns 404."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/channels/fakeplatform")
            assert resp.status == 404
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# PUT /api/channels/{name}
# ---------------------------------------------------------------------------


async def test_put_channel_update_enabled():
    """PUT /api/channels/{name} enables a channel and persists it."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/channels/telegram",
                json={"enabled": True, "token": "new-bot-token"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True

            cfg = load_config(path)
            assert cfg.channels.telegram.enabled is True
            assert cfg.channels.telegram.token == "new-bot-token"
        finally:
            await client.close()


async def test_put_channel_merge_update():
    """PUT /api/channels/{name} merges — untouched fields are preserved."""
    initial = Config()
    initial.channels.telegram = TelegramConfig(enabled=False, token="original-token")

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            # Only enable the channel, don't change the token
            resp = await client.put("/api/channels/telegram", json={"enabled": True})
            assert resp.status == 200

            cfg = load_config(path)
            assert cfg.channels.telegram.enabled is True
            assert cfg.channels.telegram.token == "original-token"
        finally:
            await client.close()


async def test_put_channel_unknown_name_404():
    """PUT /api/channels/{name} with unknown channel returns 404."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put("/api/channels/unknownchannel", json={"enabled": True})
            assert resp.status == 404
        finally:
            await client.close()


async def test_put_channel_invalid_json_400():
    """PUT /api/channels/{name} with non-JSON body returns 400."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/channels/telegram",
                data=b"not-json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# GET /api/mcp
# ---------------------------------------------------------------------------


async def test_get_mcp_empty():
    """GET /api/mcp returns an empty list when no servers are configured."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/mcp")
            assert resp.status == 200
            data = await resp.json()
            assert data == []
        finally:
            await client.close()


async def test_get_mcp_lists_servers():
    """GET /api/mcp returns the list of configured MCP servers."""
    initial = Config()
    initial.tools.mcp_servers["my-server"] = MCPServerConfig(
        command="npx", args=["-y", "my-mcp-pkg"], tool_timeout=45
    )

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/mcp")
            data = await resp.json()
            assert len(data) == 1
            assert data[0]["name"] == "my-server"
            assert data[0]["command"] == "npx"
            assert data[0]["tool_timeout"] == 45
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# POST /api/mcp/{name}
# ---------------------------------------------------------------------------


async def test_post_mcp_creates_server():
    """POST /api/mcp/{name} creates a new MCP server entry."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.post(
                "/api/mcp/new-server",
                json={"command": "node", "args": ["server.js"], "toolTimeout": 60},
            )
            assert resp.status == 201

            cfg = load_config(path)
            assert "new-server" in cfg.tools.mcp_servers
            assert cfg.tools.mcp_servers["new-server"].command == "node"
            assert cfg.tools.mcp_servers["new-server"].tool_timeout == 60
        finally:
            await client.close()


async def test_post_mcp_duplicate_409():
    """POST /api/mcp/{name} when server already exists returns 409."""
    initial = Config()
    initial.tools.mcp_servers["existing"] = MCPServerConfig(command="npx")

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.post("/api/mcp/existing", json={"command": "node"})
            assert resp.status == 409
        finally:
            await client.close()


async def test_post_mcp_invalid_json_400():
    """POST /api/mcp/{name} with bad JSON returns 400."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.post(
                "/api/mcp/newserver",
                data=b"{{bad",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# PUT /api/mcp/{name}
# ---------------------------------------------------------------------------


async def test_put_mcp_updates_server():
    """PUT /api/mcp/{name} updates an existing MCP server entry."""
    initial = Config()
    initial.tools.mcp_servers["srv"] = MCPServerConfig(command="npx", tool_timeout=30)

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put("/api/mcp/srv", json={"toolTimeout": 90})
            assert resp.status == 200

            cfg = load_config(path)
            assert cfg.tools.mcp_servers["srv"].tool_timeout == 90
            assert cfg.tools.mcp_servers["srv"].command == "npx"  # preserved
        finally:
            await client.close()


async def test_put_mcp_unknown_404():
    """PUT /api/mcp/{name} for an unknown server returns 404."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put("/api/mcp/ghost", json={"command": "x"})
            assert resp.status == 404
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# DELETE /api/mcp/{name}
# ---------------------------------------------------------------------------


async def test_delete_mcp_removes_server():
    """DELETE /api/mcp/{name} removes the server entry."""
    initial = Config()
    initial.tools.mcp_servers["to-delete"] = MCPServerConfig(command="node")

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.delete("/api/mcp/to-delete")
            assert resp.status == 200

            cfg = load_config(path)
            assert "to-delete" not in cfg.tools.mcp_servers
        finally:
            await client.close()


async def test_delete_mcp_unknown_404():
    """DELETE /api/mcp/{name} for an unknown server returns 404."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.delete("/api/mcp/ghost")
            assert resp.status == 404
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# MCP full CRUD round-trip
# ---------------------------------------------------------------------------


async def test_mcp_full_crud_roundtrip():
    """Create, read, update, and delete an MCP server in sequence."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            # Create
            resp = await client.post("/api/mcp/roundtrip", json={"command": "npx", "toolTimeout": 20})
            assert resp.status == 201

            # List — should appear
            resp = await client.get("/api/mcp")
            data = await resp.json()
            assert any(s["name"] == "roundtrip" for s in data)

            # Update
            resp = await client.put("/api/mcp/roundtrip", json={"toolTimeout": 99})
            assert resp.status == 200
            cfg = load_config(path)
            assert cfg.tools.mcp_servers["roundtrip"].tool_timeout == 99

            # Delete
            resp = await client.delete("/api/mcp/roundtrip")
            assert resp.status == 200

            # List — should be gone
            resp = await client.get("/api/mcp")
            data = await resp.json()
            assert not any(s["name"] == "roundtrip" for s in data)
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# GET /api/agent
# ---------------------------------------------------------------------------


async def test_get_agent_returns_defaults():
    """GET /api/agent returns the current agent defaults."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/agent")
            assert resp.status == 200
            data = await resp.json()
            assert "model" in data
            assert "temperature" in data
            assert "maxTokens" in data
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# PUT /api/agent
# ---------------------------------------------------------------------------


async def test_put_agent_updates_model():
    """PUT /api/agent can update the model field."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put("/api/agent", json={"model": "openai/gpt-4o"})
            assert resp.status == 200
            assert (await resp.json())["ok"] is True

            cfg = load_config(path)
            assert cfg.agents.defaults.model == "openai/gpt-4o"
        finally:
            await client.close()


async def test_put_agent_updates_temperature():
    """PUT /api/agent can update temperature."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put("/api/agent", json={"temperature": 0.7})
            assert resp.status == 200
            cfg = load_config(path)
            assert abs(cfg.agents.defaults.temperature - 0.7) < 1e-9
        finally:
            await client.close()


async def test_put_agent_invalid_json_400():
    """PUT /api/agent with non-JSON body returns 400."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/agent",
                data=b"not-json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
        finally:
            await client.close()


async def test_put_agent_partial_update_preserves_other_fields():
    """PUT /api/agent with a single field preserves all other defaults."""
    initial = Config()
    initial.agents.defaults.temperature = 0.5
    initial.agents.defaults.max_tokens = 4096

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            # Only update model
            resp = await client.put("/api/agent", json={"model": "openai/gpt-4o-mini"})
            assert resp.status == 200

            cfg = load_config(path)
            assert cfg.agents.defaults.model == "openai/gpt-4o-mini"
            assert abs(cfg.agents.defaults.temperature - 0.5) < 1e-9
            assert cfg.agents.defaults.max_tokens == 4096
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Concurrency — simultaneous PUTs must both be persisted
# ---------------------------------------------------------------------------


async def test_concurrent_agent_puts_both_persisted():
    """Two simultaneous PUT /api/agent requests must both be persisted.

    The _config_lock guarantees that the load→mutate→save cycle is atomic,
    so neither request overwrites the other's changes.
    """
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            # Fire two concurrent PUTs targeting different fields
            resp_a, resp_b = await asyncio.gather(
                client.put("/api/agent", json={"model": "concurrent-model-A"}),
                client.put("/api/agent", json={"temperature": 0.999}),
            )
            assert resp_a.status == 200
            assert resp_b.status == 200

            # Both changes must survive
            resp = await client.get("/api/agent")
            data = await resp.json()
            assert data["model"] == "concurrent-model-A"
            assert abs(data["temperature"] - 0.999) < 1e-9
        finally:
            await client.close()


async def test_concurrent_provider_puts_both_persisted():
    """Two simultaneous PUT /api/providers requests must both be persisted."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp_a, resp_b = await asyncio.gather(
                client.put("/api/providers/anthropic", json={"api_key": "ant-key-concurrent"}),
                client.put("/api/providers/openai", json={"api_key": "oai-key-concurrent"}),
            )
            assert resp_a.status == 200
            assert resp_b.status == 200

            cfg = load_config(path)
            assert cfg.providers.anthropic.api_key == "ant-key-concurrent"
            assert cfg.providers.openai.api_key == "oai-key-concurrent"
        finally:
            await client.close()

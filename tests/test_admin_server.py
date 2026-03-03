"""Unit tests for nanobot.admin.server — auth middleware and CORS headers."""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nanobot.admin.server import AdminServer
from nanobot.config.schema import AdminConfig, Config, GatewayConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(*, token: str = "", port: int = 18791) -> Config:
    """Return a Config with admin enabled and the given token/port."""
    cfg = Config()
    cfg.gateway = GatewayConfig(
        admin=AdminConfig(enabled=True, token=token, host="127.0.0.1", port=port)
    )
    return cfg


async def _client(config: Config) -> TestClient:
    """Create a *started* TestClient backed by AdminServer._build_app()."""
    server = AdminServer(config)
    app = server._build_app()
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


# ---------------------------------------------------------------------------
# Bearer-token auth middleware tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_valid_token_allowed():
    """A request with the correct Bearer token is accepted (200)."""
    config = _make_config(token="my-secret")
    client = await _client(config)
    try:
        resp = await client.get("/", headers={"Authorization": "Bearer my-secret"})
        assert resp.status == 200
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auth_invalid_token_rejected():
    """A request with a wrong Bearer token is rejected (401)."""
    config = _make_config(token="my-secret")
    client = await _client(config)
    try:
        resp = await client.get("/", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status == 401
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auth_missing_token_header_rejected():
    """A request with no Authorization header is rejected (401) when token is configured."""
    config = _make_config(token="my-secret")
    client = await _client(config)
    try:
        resp = await client.get("/")
        assert resp.status == 401
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auth_no_token_configured_bypass():
    """When no token is configured, all requests pass auth regardless of header."""
    config = _make_config(token="")
    client = await _client(config)
    try:
        # No Authorization header — should still be allowed
        resp = await client.get("/")
        assert resp.status == 200
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auth_no_token_configured_with_arbitrary_header():
    """When no token is configured, even a bogus Authorization header is accepted."""
    config = _make_config(token="")
    client = await _client(config)
    try:
        resp = await client.get("/", headers={"Authorization": "Bearer anything"})
        assert resp.status == 200
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# CORS middleware tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cors_allowed_origin_127_0_0_1():
    """Requests from http://127.0.0.1:<port> receive correct CORS response headers."""
    port = 18791
    config = _make_config(port=port)
    origin = f"http://127.0.0.1:{port}"
    client = await _client(config)
    try:
        resp = await client.get("/", headers={"Origin": origin})
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == origin
        assert "GET" in resp.headers.get("Access-Control-Allow-Methods", "")
        assert "Authorization" in resp.headers.get("Access-Control-Allow-Headers", "")
        assert resp.headers.get("Access-Control-Allow-Credentials") == "false"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cors_allowed_origin_localhost():
    """Requests from http://localhost:<port> receive correct CORS response headers."""
    port = 18791
    config = _make_config(port=port)
    origin = f"http://localhost:{port}"
    client = await _client(config)
    try:
        resp = await client.get("/", headers={"Origin": origin})
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == origin
        assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")
        assert "Content-Type" in resp.headers.get("Access-Control-Allow-Headers", "")
        assert resp.headers.get("Access-Control-Allow-Credentials") == "false"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cors_disallowed_origin_omitted():
    """Requests from an external origin do NOT receive Access-Control-Allow-Origin."""
    config = _make_config()
    client = await _client(config)
    try:
        resp = await client.get("/", headers={"Origin": "http://evil.example.com"})
        # Server should respond (no crash), but without CORS headers
        assert resp.status == 200
        assert "Access-Control-Allow-Origin" not in resp.headers
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cors_no_origin_header_no_cors_response():
    """Requests without an Origin header do not receive any CORS headers."""
    config = _make_config()
    client = await _client(config)
    try:
        resp = await client.get("/")
        assert resp.status == 200
        assert "Access-Control-Allow-Origin" not in resp.headers
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# CORS preflight (OPTIONS) tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cors_preflight_allowed_origin():
    """OPTIONS preflight from an allowed origin returns 200 with CORS headers."""
    port = 18791
    config = _make_config(port=port, token="secret")  # auth should NOT block preflight
    origin = f"http://127.0.0.1:{port}"
    client = await _client(config)
    try:
        resp = await client.options(
            "/",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == origin
        assert "GET" in resp.headers.get("Access-Control-Allow-Methods", "")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cors_preflight_disallowed_origin():
    """OPTIONS preflight from a disallowed origin returns 200 but without CORS headers."""
    config = _make_config(token="secret")
    client = await _client(config)
    try:
        resp = await client.options(
            "/",
            headers={
                "Origin": "http://attacker.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status == 200
        assert "Access-Control-Allow-Origin" not in resp.headers
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cors_preflight_bypasses_auth():
    """OPTIONS preflight must succeed even when a valid token is required (browsers cannot
    include Authorization in the preflight request)."""
    port = 18791
    config = _make_config(port=port, token="top-secret")
    origin = f"http://localhost:{port}"
    client = await _client(config)
    try:
        # No Authorization header on preflight
        resp = await client.options(
            "/",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization, Content-Type",
            },
        )
        # Must not be 401 — auth middleware must not run for OPTIONS
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == origin
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# CORS — Access-Control-Allow-Origin must echo origin, not be wildcard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cors_origin_is_exact_not_wildcard():
    """Access-Control-Allow-Origin must be the exact request Origin, never '*'."""
    port = 18791
    config = _make_config(port=port)
    origin = f"http://127.0.0.1:{port}"
    client = await _client(config)
    try:
        resp = await client.get("/", headers={"Origin": origin})
        acao = resp.headers.get("Access-Control-Allow-Origin", "")
        assert acao != "*", "Wildcard CORS origin is forbidden"
        assert acao == origin
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Root endpoint — static page served
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_returns_html():
    """GET / must return an HTML response."""
    config = _make_config()
    client = await _client(config)
    try:
        resp = await client.get("/")
        assert resp.status == 200
        content_type = resp.headers.get("Content-Type", "")
        assert "text/html" in content_type
    finally:
        await client.close()

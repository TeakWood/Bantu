"""REST API route handlers for the Bantu admin console."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from aiohttp import web

from nanobot.config.loader import load_config, save_config
from nanobot.config.schema import AgentDefaults, MCPServerConfig
from nanobot.providers.registry import PROVIDERS, find_by_name

# Module-level concurrency lock declaration (as required by spec).
# In practice, each AdminServer app creates its own lock instance stored in
# app[APP_KEY_CONFIG_LOCK] to support multiple server instances and clean test isolation.
_config_lock: asyncio.Lock = asyncio.Lock()

# Typed app-state keys — use web.AppKey to avoid string-key warnings and collisions.
APP_KEY_CONFIG_LOCK: web.AppKey[asyncio.Lock] = web.AppKey("config_lock", asyncio.Lock)
APP_KEY_CONFIG_PATH: web.AppKey[Path | None] = web.AppKey("config_path", Path)

# Sensitive field name patterns — case-insensitive substring match on key name.
_SENSITIVE_PATTERNS: tuple[str, ...] = ("token", "key", "secret", "password", "credential")

# Ordered channel field names on ChannelsConfig.
_CHANNEL_NAMES: tuple[str, ...] = (
    "whatsapp",
    "telegram",
    "discord",
    "feishu",
    "mochat",
    "dingtalk",
    "email",
    "slack",
    "qq",
    "matrix",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_sensitive(key: str) -> bool:
    """Return True if *key* (camelCase or snake_case) contains a sensitive pattern."""
    key_lower = key.lower()
    return any(pat in key_lower for pat in _SENSITIVE_PATTERNS)


def _mask_dict(data: Any) -> Any:
    """Recursively mask non-empty sensitive string values in a dict tree."""
    if isinstance(data, dict):
        return {
            k: ("****" if _is_sensitive(k) and isinstance(v, str) and v else _mask_dict(v))
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_mask_dict(item) for item in data]
    return data


def _get_lock(request: web.Request) -> asyncio.Lock:
    """Return the per-app config lock (falls back to module-level lock)."""
    return request.app.get(APP_KEY_CONFIG_LOCK, _config_lock)


def _get_config_path(request: web.Request) -> Path | None:
    """Return the config file path stored in the app (None = use default path)."""
    return request.app.get(APP_KEY_CONFIG_PATH)


async def _parse_json_body(request: web.Request) -> dict:
    """Parse and return request body as a JSON object; raises HTTP 400 on failure."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        raise web.HTTPBadRequest(reason="Invalid JSON body")
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(reason="Request body must be a JSON object")
    return body


# ---------------------------------------------------------------------------
# GET /api/config
# ---------------------------------------------------------------------------


async def handle_get_config(request: web.Request) -> web.Response:
    """Return the full sanitized config with all sensitive fields masked."""
    cfg = load_config(_get_config_path(request))
    data = cfg.model_dump(by_alias=True)
    return web.json_response(_mask_dict(data))


# ---------------------------------------------------------------------------
# GET /api/providers
# ---------------------------------------------------------------------------


async def handle_get_providers(request: web.Request) -> web.Response:
    """List all known providers with metadata."""
    cfg = load_config(_get_config_path(request))
    result = []
    for spec in PROVIDERS:
        provider_cfg = getattr(cfg.providers, spec.name, None)
        if provider_cfg is None:
            continue
        result.append(
            {
                "name": spec.name,
                "label": spec.label,
                "has_key": bool(provider_cfg.api_key),
                "api_base": provider_cfg.api_base,
                "is_oauth": spec.is_oauth,
                "is_local": spec.is_local,
            }
        )
    return web.json_response(result)


# ---------------------------------------------------------------------------
# PUT /api/providers/{name}
# ---------------------------------------------------------------------------


async def handle_put_provider(request: web.Request) -> web.Response:
    """Update a provider's api_key, api_base, and/or extra_headers."""
    name = request.match_info["name"]
    if find_by_name(name) is None:
        raise web.HTTPNotFound(reason=f"Unknown provider: {name}")

    body = await _parse_json_body(request)

    async with _get_lock(request):
        cfg = load_config(_get_config_path(request))
        provider_cfg = getattr(cfg.providers, name, None)
        if provider_cfg is None:
            raise web.HTTPNotFound(reason=f"Unknown provider: {name}")

        if "api_key" in body:
            provider_cfg.api_key = str(body["api_key"])
        if "api_base" in body:
            val = body["api_base"]
            provider_cfg.api_base = str(val) if val is not None else None
        if "extra_headers" in body:
            val = body["extra_headers"]
            if val is not None and not isinstance(val, dict):
                raise web.HTTPBadRequest(reason="extra_headers must be an object or null")
            provider_cfg.extra_headers = val

        save_config(cfg, _get_config_path(request))

    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# GET /api/channels
# ---------------------------------------------------------------------------


async def handle_get_channels(request: web.Request) -> web.Response:
    """List all channels with their enabled status and a short summary."""
    cfg = load_config(_get_config_path(request))
    result = []
    for name in _CHANNEL_NAMES:
        ch = getattr(cfg.channels, name, None)
        if ch is None:
            continue
        enabled = getattr(ch, "enabled", False)
        result.append(
            {
                "name": name,
                "enabled": enabled,
                "summary": f"{name} ({'enabled' if enabled else 'disabled'})",
            }
        )
    return web.json_response(result)


# ---------------------------------------------------------------------------
# GET /api/channels/{name}
# ---------------------------------------------------------------------------


async def handle_get_channel(request: web.Request) -> web.Response:
    """Return the full config dict for a single channel with sensitive fields masked."""
    name = request.match_info["name"]
    if name not in _CHANNEL_NAMES:
        raise web.HTTPNotFound(reason=f"Unknown channel: {name}")

    cfg = load_config(_get_config_path(request))
    ch = getattr(cfg.channels, name, None)
    if ch is None:
        raise web.HTTPNotFound(reason=f"Unknown channel: {name}")

    data = ch.model_dump(by_alias=True)
    return web.json_response(_mask_dict(data))


# ---------------------------------------------------------------------------
# PUT /api/channels/{name}
# ---------------------------------------------------------------------------


async def handle_put_channel(request: web.Request) -> web.Response:
    """Merge-update a channel's config and persist to disk."""
    name = request.match_info["name"]
    if name not in _CHANNEL_NAMES:
        raise web.HTTPNotFound(reason=f"Unknown channel: {name}")

    body = await _parse_json_body(request)

    async with _get_lock(request):
        cfg = load_config(_get_config_path(request))
        ch = getattr(cfg.channels, name, None)
        if ch is None:
            raise web.HTTPNotFound(reason=f"Unknown channel: {name}")

        # Merge body on top of the current camelCase representation, then revalidate.
        current = ch.model_dump(by_alias=True)
        current.update(body)

        try:
            updated = type(ch).model_validate(current)
        except Exception as exc:
            raise web.HTTPBadRequest(reason=f"Invalid channel config: {exc}") from exc

        setattr(cfg.channels, name, updated)
        save_config(cfg, _get_config_path(request))

    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# GET /api/mcp
# ---------------------------------------------------------------------------


async def handle_get_mcp(request: web.Request) -> web.Response:
    """List MCP servers with their name and key connection parameters."""
    cfg = load_config(_get_config_path(request))
    result = [
        {
            "name": server_name,
            "command": server_cfg.command,
            "url": server_cfg.url,
            "tool_timeout": server_cfg.tool_timeout,
        }
        for server_name, server_cfg in cfg.tools.mcp_servers.items()
    ]
    return web.json_response(result)


# ---------------------------------------------------------------------------
# POST /api/mcp/{name}
# ---------------------------------------------------------------------------


async def handle_post_mcp(request: web.Request) -> web.Response:
    """Create a new MCP server entry."""
    name = request.match_info["name"]
    body = await _parse_json_body(request)

    async with _get_lock(request):
        cfg = load_config(_get_config_path(request))
        if name in cfg.tools.mcp_servers:
            raise web.HTTPConflict(reason=f"MCP server already exists: {name}")

        try:
            new_server = MCPServerConfig.model_validate(body)
        except Exception as exc:
            raise web.HTTPBadRequest(reason=f"Invalid MCP server config: {exc}") from exc

        cfg.tools.mcp_servers[name] = new_server
        save_config(cfg, _get_config_path(request))

    return web.json_response({"ok": True}, status=201)


# ---------------------------------------------------------------------------
# PUT /api/mcp/{name}
# ---------------------------------------------------------------------------


async def handle_put_mcp(request: web.Request) -> web.Response:
    """Update an existing MCP server entry."""
    name = request.match_info["name"]
    body = await _parse_json_body(request)

    async with _get_lock(request):
        cfg = load_config(_get_config_path(request))
        if name not in cfg.tools.mcp_servers:
            raise web.HTTPNotFound(reason=f"Unknown MCP server: {name}")

        current = cfg.tools.mcp_servers[name].model_dump(by_alias=True)
        current.update(body)

        try:
            updated = MCPServerConfig.model_validate(current)
        except Exception as exc:
            raise web.HTTPBadRequest(reason=f"Invalid MCP server config: {exc}") from exc

        cfg.tools.mcp_servers[name] = updated
        save_config(cfg, _get_config_path(request))

    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# DELETE /api/mcp/{name}
# ---------------------------------------------------------------------------


async def handle_delete_mcp(request: web.Request) -> web.Response:
    """Remove an MCP server entry."""
    name = request.match_info["name"]

    async with _get_lock(request):
        cfg = load_config(_get_config_path(request))
        if name not in cfg.tools.mcp_servers:
            raise web.HTTPNotFound(reason=f"Unknown MCP server: {name}")

        del cfg.tools.mcp_servers[name]
        save_config(cfg, _get_config_path(request))

    return web.json_response({"ok": True})


# ---------------------------------------------------------------------------
# GET /api/agent
# ---------------------------------------------------------------------------


async def handle_get_agent(request: web.Request) -> web.Response:
    """Return the current agent defaults."""
    cfg = load_config(_get_config_path(request))
    data = cfg.agents.defaults.model_dump(by_alias=True)
    return web.json_response(data)


# ---------------------------------------------------------------------------
# PUT /api/agent
# ---------------------------------------------------------------------------


async def handle_put_agent(request: web.Request) -> web.Response:
    """Update agent defaults and persist to disk."""
    body = await _parse_json_body(request)

    async with _get_lock(request):
        cfg = load_config(_get_config_path(request))
        current = cfg.agents.defaults.model_dump(by_alias=True)
        current.update(body)

        try:
            updated = AgentDefaults.model_validate(current)
        except Exception as exc:
            raise web.HTTPBadRequest(reason=f"Invalid agent config: {exc}") from exc

        cfg.agents.defaults = updated
        save_config(cfg, _get_config_path(request))

    return web.json_response({"ok": True})

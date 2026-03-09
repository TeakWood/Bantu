"""Tests for per-agent config overrides: schema, resolve_agent_config, and admin endpoints."""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nanobot.admin import routes as _routes
from nanobot.admin.server import AdminServer
from nanobot.agent.registry import AgentRegistry
from nanobot.config.loader import load_config
from nanobot.config.schema import (
    AdminConfig,
    AgentDefaults,
    AgentOverride,
    AgentsConfig,
    Config,
    GatewayConfig,
    resolve_agent_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server_config(*, port: int = 18791) -> Config:
    cfg = Config()
    cfg.gateway = GatewayConfig(
        admin=AdminConfig(enabled=True, token="", host="127.0.0.1", port=port)
    )
    return cfg


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


async def _client(
    server_config: Config,
    config_path: Path,
    registry: AgentRegistry | None = None,
) -> TestClient:
    """Build a started TestClient, optionally injecting a custom AgentRegistry."""
    server = AdminServer(server_config, config_path=config_path)
    app = server._build_app()
    if registry is not None:
        app[_routes.APP_KEY_AGENT_REGISTRY] = registry
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


# ---------------------------------------------------------------------------
# resolve_agent_config — no override
# ---------------------------------------------------------------------------


def test_resolve_no_override_returns_defaults():
    """With no override the defaults are returned unchanged."""
    defaults = AgentDefaults(model="openai/gpt-4o", temperature=0.2)
    agents_cfg = AgentsConfig(defaults=defaults)

    resolved = resolve_agent_config("unknown-agent", agents_cfg)

    assert resolved.model == "openai/gpt-4o"
    assert abs(resolved.temperature - 0.2) < 1e-9
    assert resolved.max_tokens == defaults.max_tokens


# ---------------------------------------------------------------------------
# resolve_agent_config — full override
# ---------------------------------------------------------------------------


def test_resolve_full_override_all_fields_win():
    """When every override field is set, all of them win over defaults."""
    defaults = AgentDefaults(
        model="anthropic/claude-opus-4-5",
        provider="auto",
        max_tokens=8192,
        temperature=0.1,
        max_tool_iterations=40,
        memory_window=100,
        reasoning_effort=None,
    )
    override = AgentOverride(
        model="openai/gpt-4o-mini",
        provider="openai",
        max_tokens=4096,
        temperature=0.9,
        max_tool_iterations=10,
        memory_window=50,
        reasoning_effort="high",
    )
    agents_cfg = AgentsConfig(defaults=defaults, overrides={"code-agent": override})

    resolved = resolve_agent_config("code-agent", agents_cfg)

    assert resolved.model == "openai/gpt-4o-mini"
    assert resolved.provider == "openai"
    assert resolved.max_tokens == 4096
    assert abs(resolved.temperature - 0.9) < 1e-9
    assert resolved.max_tool_iterations == 10
    assert resolved.memory_window == 50
    assert resolved.reasoning_effort == "high"


# ---------------------------------------------------------------------------
# resolve_agent_config — partial override
# ---------------------------------------------------------------------------


def test_resolve_partial_override_only_set_fields_win():
    """Only non-None override fields replace defaults; others stay as defaults."""
    defaults = AgentDefaults(
        model="anthropic/claude-opus-4-5",
        temperature=0.1,
        max_tokens=8192,
    )
    # Only model is overridden
    override = AgentOverride(model="openai/gpt-4o-mini")
    agents_cfg = AgentsConfig(defaults=defaults, overrides={"research-agent": override})

    resolved = resolve_agent_config("research-agent", agents_cfg)

    assert resolved.model == "openai/gpt-4o-mini"
    assert abs(resolved.temperature - 0.1) < 1e-9  # from defaults
    assert resolved.max_tokens == 8192  # from defaults


def test_resolve_partial_override_temperature_only():
    """Partial override touching only temperature leaves other fields at defaults."""
    defaults = AgentDefaults(model="anthropic/claude-opus-4-5", temperature=0.1, max_tokens=8192)
    override = AgentOverride(temperature=0.7)
    agents_cfg = AgentsConfig(defaults=defaults, overrides={"warm-agent": override})

    resolved = resolve_agent_config("warm-agent", agents_cfg)

    assert resolved.model == "anthropic/claude-opus-4-5"
    assert abs(resolved.temperature - 0.7) < 1e-9
    assert resolved.max_tokens == 8192


# ---------------------------------------------------------------------------
# resolve_agent_config — defaults not mutated
# ---------------------------------------------------------------------------


def test_resolve_does_not_mutate_defaults():
    """resolve_agent_config must not modify the original AgentDefaults object."""
    defaults = AgentDefaults(model="original-model", temperature=0.1)
    override = AgentOverride(model="override-model")
    agents_cfg = AgentsConfig(defaults=defaults, overrides={"x": override})

    resolve_agent_config("x", agents_cfg)

    assert agents_cfg.defaults.model == "original-model"


# ---------------------------------------------------------------------------
# AgentOverride Pydantic model
# ---------------------------------------------------------------------------


def test_agent_override_all_fields_optional():
    """AgentOverride can be constructed with no arguments (all-None)."""
    ov = AgentOverride()
    assert ov.model is None
    assert ov.provider is None
    assert ov.max_tokens is None
    assert ov.temperature is None
    assert ov.max_tool_iterations is None
    assert ov.memory_window is None
    assert ov.reasoning_effort is None


def test_agents_config_overrides_default_empty():
    """AgentsConfig.overrides defaults to an empty dict."""
    cfg = AgentsConfig()
    assert cfg.overrides == {}


# ---------------------------------------------------------------------------
# GET /api/agents — empty registry and no overrides
# ---------------------------------------------------------------------------


async def test_get_agents_empty():
    """GET /api/agents returns an empty list when registry is empty and no overrides."""
    registry = AgentRegistry()
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path, registry=registry)
        try:
            resp = await client.get("/api/agents")
            assert resp.status == 200
            data = await resp.json()
            assert data == []
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# GET /api/agents — agents from registry
# ---------------------------------------------------------------------------


async def test_get_agents_from_registry():
    """GET /api/agents lists agents registered in the AgentRegistry."""
    registry = AgentRegistry()
    registry.register("code-agent")
    registry.register("research-agent")

    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path, registry=registry)
        try:
            resp = await client.get("/api/agents")
            assert resp.status == 200
            data = await resp.json()
            names = [item["name"] for item in data]
            assert "code-agent" in names
            assert "research-agent" in names
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# GET /api/agents — agents from config overrides
# ---------------------------------------------------------------------------


async def test_get_agents_from_config_overrides():
    """GET /api/agents includes agents that have persisted overrides."""
    initial = Config()
    initial.agents.overrides["fast-agent"] = AgentOverride(model="openai/gpt-4o-mini")

    registry = AgentRegistry()  # empty
    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path, registry=registry)
        try:
            resp = await client.get("/api/agents")
            assert resp.status == 200
            data = await resp.json()
            names = [item["name"] for item in data]
            assert "fast-agent" in names
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# GET /api/agents — resolved config is included
# ---------------------------------------------------------------------------


async def test_get_agents_includes_resolved_config():
    """Each entry in GET /api/agents contains the effective resolved config."""
    initial = Config()
    initial.agents.overrides["my-agent"] = AgentOverride(model="openai/gpt-4o-mini")

    registry = AgentRegistry()
    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path, registry=registry)
        try:
            resp = await client.get("/api/agents")
            data = await resp.json()
            agent = next(a for a in data if a["name"] == "my-agent")
            assert agent["config"]["model"] == "openai/gpt-4o-mini"
            # defaults still fill the rest
            assert "temperature" in agent["config"]
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# GET /api/agents/{name}/config — returns defaults when no override
# ---------------------------------------------------------------------------


async def test_get_agent_config_no_override_returns_defaults():
    """GET /api/agents/{name}/config returns global defaults when no override exists."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/agents/any-agent/config")
            assert resp.status == 200
            data = await resp.json()
            assert "model" in data
            assert "temperature" in data
            assert "maxTokens" in data
        finally:
            await client.close()


async def test_get_agent_config_with_override_returns_merged():
    """GET /api/agents/{name}/config returns merged config when override exists."""
    initial = Config()
    initial.agents.overrides["code-agent"] = AgentOverride(
        model="openai/gpt-4o-mini", temperature=0.0
    )

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.get("/api/agents/code-agent/config")
            assert resp.status == 200
            data = await resp.json()
            assert data["model"] == "openai/gpt-4o-mini"
            assert abs(data["temperature"] - 0.0) < 1e-9
            # Non-overridden fields use defaults
            assert data["maxTokens"] == Config().agents.defaults.max_tokens
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# PUT /api/agents/{name}/config — create new override
# ---------------------------------------------------------------------------


async def test_put_agent_config_creates_override():
    """PUT /api/agents/{name}/config creates a new per-agent override."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/agents/code-agent/config",
                json={"model": "openai/gpt-4o-mini", "temperature": 0.0},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True

            # Verify persisted
            cfg = load_config(path)
            assert "code-agent" in cfg.agents.overrides
            assert cfg.agents.overrides["code-agent"].model == "openai/gpt-4o-mini"
            assert abs(cfg.agents.overrides["code-agent"].temperature - 0.0) < 1e-9
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# PUT /api/agents/{name}/config — merge update (partial)
# ---------------------------------------------------------------------------


async def test_put_agent_config_merge_update():
    """PUT /api/agents/{name}/config merges into an existing override."""
    initial = Config()
    initial.agents.overrides["my-agent"] = AgentOverride(
        model="openai/gpt-4o-mini", temperature=0.5
    )

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path)
        try:
            # Only update temperature
            resp = await client.put(
                "/api/agents/my-agent/config",
                json={"temperature": 0.9},
            )
            assert resp.status == 200

            cfg = load_config(path)
            # model preserved, temperature updated
            assert cfg.agents.overrides["my-agent"].model == "openai/gpt-4o-mini"
            assert abs(cfg.agents.overrides["my-agent"].temperature - 0.9) < 1e-9
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# PUT /api/agents/{name}/config — invalid JSON
# ---------------------------------------------------------------------------


async def test_put_agent_config_invalid_json_400():
    """PUT /api/agents/{name}/config with non-JSON body returns 400."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            resp = await client.put(
                "/api/agents/my-agent/config",
                data=b"not-json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Round-trip: PUT then GET
# ---------------------------------------------------------------------------


async def test_put_then_get_agent_config_roundtrip():
    """PUT followed by GET returns the effective merged config."""
    with _temp_config_file() as path:
        client = await _client(_make_server_config(), path)
        try:
            await client.put(
                "/api/agents/rt-agent/config",
                json={"model": "openai/gpt-4o", "maxTokens": 2048},
            )

            resp = await client.get("/api/agents/rt-agent/config")
            assert resp.status == 200
            data = await resp.json()
            assert data["model"] == "openai/gpt-4o"
            assert data["maxTokens"] == 2048
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# GET /api/agents — union of registry + overrides (no duplicates)
# ---------------------------------------------------------------------------


async def test_get_agents_union_no_duplicates():
    """Agents present in both registry and overrides appear only once."""
    initial = Config()
    initial.agents.overrides["shared-agent"] = AgentOverride(model="openai/gpt-4o-mini")

    registry = AgentRegistry()
    registry.register("shared-agent")
    registry.register("registry-only-agent")

    with _temp_config_file(initial) as path:
        client = await _client(_make_server_config(), path, registry=registry)
        try:
            resp = await client.get("/api/agents")
            data = await resp.json()
            names = [item["name"] for item in data]
            assert names.count("shared-agent") == 1
            assert "registry-only-agent" in names
        finally:
            await client.close()

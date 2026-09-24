"""Registry and resolution for agents declared in one config.

Covers acceptance criteria 1 (no named agents, no change), 2 (registry and
resolution) and the config-level half of 5 (tool isolation).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nanobot.agents import (
    DEFAULT_AGENT_NAME,
    agent_config,
    agent_names,
    agent_spec,
    named_agent_workspace,
    resolve_agent_specs,
)
from nanobot.config.errors import ConfigLoadError
from nanobot.config.loader import load_config, resolve_config_env_vars
from nanobot.config.schema import Config

from .conftest import write_config


def _load(path: Path) -> Config:
    return resolve_config_env_vars(load_config(path), config_path=path)


def test_config_without_named_agents_resolves_only_default(instance_dir: Path) -> None:
    path = write_config(
        instance_dir,
        {"agents": {"defaults": {"model": "anthropic/claude-sonnet-5", "workspace": "~/ws"}}},
    )

    specs = resolve_agent_specs(_load(path))

    assert [spec.name for spec in specs] == [DEFAULT_AGENT_NAME]
    assert specs[0].workspace == Path("~/ws").expanduser()
    assert specs[0].model == "anthropic/claude-sonnet-5"


def test_named_agents_are_listed_after_default_in_declaration_order(
    instance_dir: Path,
) -> None:
    path = write_config(
        instance_dir,
        {
            "agents": {
                "defaults": {"model": "anthropic/claude-sonnet-5"},
                "named": {
                    "research": {
                        "model": "anthropic/claude-opus-5-5",
                        "workspace": str(instance_dir / "research-ws"),
                    },
                    "trader": {},
                },
            }
        },
    )

    specs = resolve_agent_specs(_load(path))

    assert [spec.name for spec in specs] == [DEFAULT_AGENT_NAME, "research", "trader"]
    # A named agent that sets its own model reports it.
    assert specs[1].model == "anthropic/claude-opus-5-5"
    assert specs[1].workspace == instance_dir / "research-ws"
    # One that sets none inherits agents.defaults.model...
    assert specs[2].model == "anthropic/claude-sonnet-5"
    # ...but never the default agent's workspace.
    assert specs[2].workspace == named_agent_workspace("trader")
    assert specs[2].workspace != specs[0].workspace


def test_named_agent_inherits_unset_settings_and_overrides_set_ones(
    instance_dir: Path,
) -> None:
    path = write_config(
        instance_dir,
        {
            "agents": {
                "defaults": {"maxTokens": 1234, "temperature": 0.9, "botName": "base"},
                "named": {"research": {"maxTokens": 4321}},
            }
        },
    )

    spec = agent_spec(_load(path), "research")

    assert spec.settings.max_tokens == 4321
    assert spec.settings.temperature == 0.9
    assert spec.settings.bot_name == "base"


def test_named_agent_resolves_model_through_its_own_preset(instance_dir: Path) -> None:
    path = write_config(
        instance_dir,
        {
            "modelPresets": {"deep": {"model": "anthropic/claude-opus-5-5"}},
            "agents": {
                "defaults": {"model": "anthropic/claude-sonnet-5"},
                "named": {"research": {"modelPreset": "deep"}},
            },
        },
    )

    assert agent_spec(_load(path), "research").model == "anthropic/claude-opus-5-5"


def test_named_agent_keeps_a_declared_timezone_against_auto_detection(
    instance_dir: Path,
) -> None:
    path = write_config(
        instance_dir,
        {"agents": {"named": {"research": {"timezone": "Asia/Tokyo"}}}},
    )

    assert agent_spec(_load(path), "research").settings.timezone == "Asia/Tokyo"


def test_mcp_servers_are_never_inherited_in_either_direction(instance_dir: Path) -> None:
    path = write_config(
        instance_dir,
        {
            "tools": {"mcpServers": {"shared": {"command": "shared-mcp"}}},
            "agents": {
                "named": {
                    "trader": {"tools": {"mcpServers": {"broker": {"command": "broker-mcp"}}}},
                    "health": {},
                }
            },
        },
    )
    config = _load(path)

    # Top-level servers belong to the default agent only.
    assert set(agent_spec(config, DEFAULT_AGENT_NAME).tools.mcp_servers) == {"shared"}
    # A named agent has exactly the servers it declares...
    assert set(agent_spec(config, "trader").tools.mcp_servers) == {"broker"}
    # ...and one that declares none has none at all.
    assert agent_spec(config, "health").tools.mcp_servers == {}


def test_non_mcp_tool_settings_merge_over_the_top_level_block(instance_dir: Path) -> None:
    path = write_config(
        instance_dir,
        {
            "tools": {
                "restrictToWorkspace": True,
                "maxSessionMessagesPerMinute": 9,
                "mcpServers": {"shared": {"command": "shared-mcp"}},
            },
            "agents": {"named": {"research": {"tools": {"maxSessionMessagesPerMinute": 2}}}},
        },
    )

    tools = agent_spec(_load(path), "research").tools

    assert tools.max_session_messages_per_minute == 2  # own entry wins
    assert tools.restrict_to_workspace is True  # the rest merges over
    assert tools.mcp_servers == {}  # except MCP servers


def test_agent_config_view_carries_that_agents_settings_and_tools(
    instance_dir: Path,
) -> None:
    path = write_config(
        instance_dir,
        {
            "tools": {"mcpServers": {"shared": {"command": "shared-mcp"}}},
            "agents": {
                "defaults": {"model": "anthropic/claude-sonnet-5"},
                "named": {
                    "research": {
                        "model": "anthropic/claude-opus-5-5",
                        "tools": {"mcpServers": {"papers": {"command": "papers-mcp"}}},
                    }
                },
            },
        },
    )
    config = _load(path)

    view = agent_config(config, "research")

    assert view.agents.defaults.model == "anthropic/claude-opus-5-5"
    assert set(view.tools.mcp_servers) == {"papers"}
    assert view.agents.named == {}
    assert view.source_path == path
    # The view is a copy: resolving it does not disturb the loaded config.
    assert config.agents.defaults.model == "anthropic/claude-sonnet-5"
    assert set(config.tools.mcp_servers) == {"shared"}


def test_each_agent_gets_its_own_session_store(instance_dir: Path) -> None:
    path = write_config(
        instance_dir,
        {"agents": {"named": {"research": {}, "trader": {}}}},
    )

    roots = {spec.name: spec.sessions_root for spec in resolve_agent_specs(_load(path))}

    assert len(set(roots.values())) == 3
    assert roots[DEFAULT_AGENT_NAME] == instance_dir / "sessions"
    # Session storage stays outside every agent workspace (ADR-0001).
    for spec in resolve_agent_specs(_load(path)):
        assert spec.sessions_root is not None
        assert not spec.sessions_root.is_relative_to(spec.workspace)


def test_unknown_agent_name_is_rejected(instance_dir: Path) -> None:
    config = _load(write_config(instance_dir, {}))

    assert agent_names(config) == [DEFAULT_AGENT_NAME]
    with pytest.raises(KeyError):
        agent_spec(config, "nope")


@pytest.mark.parametrize("name", ["Research", "-lead", "has space", "dots.here", ""])
def test_invalid_agent_names_are_rejected(name: str) -> None:
    with pytest.raises(ValidationError, match="invalid agent name"):
        Config.model_validate({"agents": {"named": {name: {}}}})


@pytest.mark.parametrize("name", ["research", "trader2", "health-notes", "a_b", "x"])
def test_valid_agent_names_are_accepted(name: str) -> None:
    config = Config.model_validate({"agents": {"named": {name: {}}}})

    assert agent_names(config) == [DEFAULT_AGENT_NAME, name]


def test_default_is_a_reserved_agent_name() -> None:
    with pytest.raises(ValidationError, match="reserved"):
        Config.model_validate({"agents": {"named": {"default": {}}}})


def test_an_invalid_agent_name_fails_the_whole_config_load(instance_dir: Path) -> None:
    path = write_config(instance_dir, {"agents": {"named": {"Bad Name": {}}}})

    with pytest.raises(ConfigLoadError):
        _load(path)


def test_a_malformed_tools_block_is_rejected_at_load(instance_dir: Path) -> None:
    path = write_config(
        instance_dir,
        {"agents": {"named": {"research": {"tools": {"maxSessionMessagesPerMinute": 0}}}}},
    )

    with pytest.raises(Exception):
        _load(path)

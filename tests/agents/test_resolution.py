"""Effective settings of one agent, resolved from agents.defaults and agents.named."""

from pathlib import Path

import pytest

from nanobot.agents import (
    ResolvedAgentConfig,
    named_agent_workspace,
    resolve_agent_config,
)
from nanobot.config.schema import Config


def make_config(**overrides: object) -> Config:
    """Build a config whose every relevant block is explicitly configured."""
    raw: dict[str, object] = {
        "agents": {
            "defaults": {
                "workspace": "~/default-workspace",
                "model": "base/model",
                "maxTokens": 4242,
                "timezone": "Asia/Shanghai",
                "dream": {"intervalH": 7},
            },
            "named": {},
        },
        "tools": {
            "maxSessionMessagesPerMinute": 9,
            "restrictToWorkspace": True,
            "exec": {"timeout": 120, "sandbox": "bwrap"},
            "mcpServers": {"shared": {"command": "shared-mcp"}},
        },
    }
    raw.update(overrides)
    return Config.model_validate(raw)


def with_named(named: dict[str, object]) -> Config:
    config = make_config()
    return Config.model_validate(
        {
            **config.model_dump(mode="json", by_alias=True),
            "agents": {
                **config.model_dump(mode="json", by_alias=True)["agents"],
                "named": named,
            },
        }
    )


# --- default reproduces today's effective settings ---------------------------


def test_default_reproduces_todays_effective_settings() -> None:
    config = with_named({"research": {"model": "r/model", "workspace": "~/research"}})
    resolved = resolve_agent_config(config, "default")

    assert isinstance(resolved, ResolvedAgentConfig)
    assert resolved.is_default
    assert resolved.config.model_dump(mode="json", by_alias=True) == config.model_dump(
        mode="json", by_alias=True
    )
    assert resolved.agent.model == config.agents.defaults.model
    assert resolved.workspace == Path("~/default-workspace").expanduser()
    assert resolved.tools.model_dump() == config.tools.model_dump()


def test_default_is_the_implicit_agent() -> None:
    config = make_config()
    assert resolve_agent_config(config).name == "default"


def test_default_resolution_keeps_the_configured_source_path(tmp_path: Path) -> None:
    config = make_config()
    config.bind_source_path(tmp_path / "config.json")

    assert resolve_agent_config(config, "default").config.runtime_data_dir == tmp_path
    config = with_named({"research": {}})
    config.bind_source_path(tmp_path / "config.json")
    assert resolve_agent_config(config, "research").config.runtime_data_dir == tmp_path


# --- inheriting agents.defaults ----------------------------------------------


def test_named_agent_setting_model_reports_it() -> None:
    config = with_named({"research": {"model": "research/model"}})
    resolved = resolve_agent_config(config, "research")

    assert resolved.name == "research"
    assert not resolved.is_default
    assert resolved.agent.model == "research/model"
    assert resolved.model == "research/model"


def test_named_agent_without_a_model_reports_the_default_agents_model() -> None:
    config = with_named({"research": {"maxTokens": 10}})
    resolved = resolve_agent_config(config, "research")

    assert resolved.agent.model == config.agents.defaults.model == "base/model"
    assert resolved.model == "base/model"


def test_a_field_the_agent_never_states_keeps_the_configured_default() -> None:
    """`maxTokens: 4242` must survive, not fall back to the schema default."""
    config = with_named({"research": {"model": "research/model"}})
    resolved = resolve_agent_config(config, "research")

    assert resolved.agent.max_tokens == 4242
    assert resolved.agent.timezone == "Asia/Shanghai"


def test_a_field_set_to_the_schema_default_still_wins() -> None:
    config = with_named({"research": {"maxTokens": 8192}})
    assert resolve_agent_config(config, "research").agent.max_tokens == 8192


def test_nested_agent_blocks_merge_field_by_field() -> None:
    config = with_named({"research": {"dream": {"enabled": False}}})
    resolved = resolve_agent_config(config, "research")

    assert resolved.agent.dream.enabled is False
    assert resolved.agent.dream.interval_h == 7  # inherited from agents.defaults


def test_an_agents_own_timezone_survives_an_inherited_auto_mode() -> None:
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {"model": "base/model"},  # timezoneMode: auto
                "named": {"research": {"timezone": "Europe/Berlin"}, "ops": {}},
            }
        }
    )
    assert config.agents.defaults.timezone_mode == "auto"

    research = resolve_agent_config(config, "research")
    assert research.agent.timezone == "Europe/Berlin"
    assert research.agent.timezone_mode == "manual"
    # A reload of the resolved settings must not re-detect over it.
    assert (
        type(research.agent)
        .model_validate(research.agent.model_dump(mode="json", by_alias=True))
        .timezone
        == "Europe/Berlin"
    )

    ops = resolve_agent_config(config, "ops")
    assert ops.agent.timezone == config.agents.defaults.timezone
    assert ops.agent.timezone_mode == "auto"


def test_an_explicit_model_deselects_an_inherited_preset() -> None:
    config = Config.model_validate(
        {
            "modelPresets": {"fast": {"model": "preset/model"}},
            "agents": {
                "defaults": {"modelPreset": "fast", "model": "base/model"},
                "named": {"research": {"model": "research/model"}, "ops": {}},
            },
        }
    )

    research = resolve_agent_config(config, "research")
    assert research.agent.model_preset is None
    assert research.model == "research/model"

    ops = resolve_agent_config(config, "ops")
    assert ops.agent.model_preset == "fast"
    assert ops.model == "preset/model"


# --- workspace is not inherited ----------------------------------------------


def test_named_agent_without_a_workspace_gets_its_own() -> None:
    config = with_named({"research": {"model": "research/model"}})
    resolved = resolve_agent_config(config, "research")

    assert config.agents.defaults.workspace == "~/default-workspace"
    assert resolved.agent.workspace == "~/.nanobot/agents/research"
    assert resolved.workspace == Path("~/.nanobot/agents/research").expanduser()
    assert named_agent_workspace("research") == "~/.nanobot/agents/research"


def test_named_agent_with_a_workspace_keeps_it() -> None:
    config = with_named({"research": {"workspace": "~/research-lab"}})
    resolved = resolve_agent_config(config, "research")

    assert resolved.agent.workspace == "~/research-lab"
    assert resolved.workspace == Path("~/research-lab").expanduser()


def test_every_agent_owns_a_distinct_workspace() -> None:
    config = with_named({"research": {}, "ops": {}})
    workspaces = {
        name: resolve_agent_config(config, name).workspace
        for name in ("default", "research", "ops")
    }
    assert len(set(workspaces.values())) == 3


# --- tools.mcpServers is not inherited ---------------------------------------


def test_top_level_mcp_servers_are_absent_from_every_named_agent() -> None:
    config = with_named({"research": {"tools": {"restrictToWorkspace": False}}, "ops": {}})

    assert set(resolve_agent_config(config, "default").mcp_servers) == {"shared"}
    assert resolve_agent_config(config, "research").mcp_servers == {}
    assert resolve_agent_config(config, "ops").mcp_servers == {}


def test_an_agents_mcp_servers_reach_no_other_agent() -> None:
    config = with_named(
        {
            "research": {"tools": {"mcpServers": {"notes": {"command": "notes-mcp"}}}},
            "ops": {"tools": {"mcpServers": {"pager": {"command": "pager-mcp"}}}},
        }
    )

    assert set(resolve_agent_config(config, "research").mcp_servers) == {"notes"}
    assert set(resolve_agent_config(config, "ops").mcp_servers) == {"pager"}
    assert set(resolve_agent_config(config, "default").mcp_servers) == {"shared"}
    assert resolve_agent_config(config, "research").mcp_servers["notes"].command == "notes-mcp"


# --- the rest of the tools block merges --------------------------------------


def test_non_mcp_tools_fields_merge_over_the_top_level_block() -> None:
    config = with_named(
        {
            "research": {
                "tools": {
                    "restrictToWorkspace": False,
                    "exec": {"timeout": 5},
                    "mcpServers": {"notes": {"command": "notes-mcp"}},
                }
            }
        }
    )
    resolved = resolve_agent_config(config, "research")

    assert resolved.tools.restrict_to_workspace is False  # stated by the agent
    assert resolved.tools.max_session_messages_per_minute == 9  # inherited
    assert resolved.tools.exec.timeout == 5  # stated by the agent
    assert resolved.tools.exec.sandbox == "bwrap"  # inherited field of the same block


def test_an_agent_with_no_tools_block_inherits_everything_but_mcp_servers() -> None:
    config = with_named({"research": {}})
    resolved = resolve_agent_config(config, "research")

    assert resolved.tools.model_dump(exclude={"mcp_servers"}) == config.tools.model_dump(
        exclude={"mcp_servers"}
    )
    assert resolved.mcp_servers == {}


# --- isolation of the returned config ----------------------------------------


def test_resolution_never_mutates_the_source_config() -> None:
    config = with_named({"research": {"tools": {"exec": {"timeout": 5}}}})
    before = config.model_dump(mode="json", by_alias=True)

    resolved = resolve_agent_config(config, "research")
    resolved.agent.model = "mutated"
    resolved.tools.exec.timeout = 999
    resolved.mcp_servers["late"] = config.tools.mcp_servers["shared"]

    assert config.model_dump(mode="json", by_alias=True) == before


def test_two_agents_share_no_mutable_state() -> None:
    config = with_named({"research": {}, "ops": {}})
    research = resolve_agent_config(config, "research")
    ops = resolve_agent_config(config, "ops")

    assert research.config is not ops.config
    assert research.tools is not ops.tools
    assert research.tools.exec is not ops.tools.exec
    assert research.agent.dream is not ops.agent.dream


def test_unknown_agent_is_rejected() -> None:
    config = with_named({"research": {}})
    with pytest.raises(KeyError, match="unknown agent 'nope'"):
        resolve_agent_config(config, "nope")


def test_a_config_with_no_named_agents_still_resolves_default() -> None:
    config = Config()
    resolved = resolve_agent_config(config, "default")

    assert resolved.name == "default"
    assert resolved.agent.workspace == config.agents.defaults.workspace
    with pytest.raises(KeyError, match="configured agents: default"):
        resolve_agent_config(config, "research")

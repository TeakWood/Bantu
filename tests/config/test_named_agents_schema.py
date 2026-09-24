"""Schema-level contract for `agents.named` entries and their names."""

import pytest
from pydantic import ValidationError

from nanobot.config.schema import (
    AgentDefaults,
    AgentsConfig,
    Config,
    NamedAgentConfig,
    ToolsConfig,
    validate_agent_name,
)


def test_named_agent_accepts_every_agent_defaults_field() -> None:
    defaults_fields = set(AgentDefaults.model_fields)
    entry_fields = set(NamedAgentConfig.model_fields)
    assert defaults_fields <= entry_fields
    assert entry_fields - defaults_fields == {"tools"}


def test_named_agent_tools_block_has_the_top_level_tools_shape() -> None:
    assert NamedAgentConfig.model_fields["tools"].annotation is ToolsConfig
    assert set(Config.model_fields["tools"].annotation.model_fields) == set(ToolsConfig.model_fields)


def test_named_agent_entry_parses_every_kind_of_field() -> None:
    entry = NamedAgentConfig.model_validate(
        {
            "workspace": "~/research",
            "model": "anthropic/claude-sonnet-5",
            "maxTokens": 4096,
            "timezone": "Asia/Shanghai",
            "dream": {"intervalH": 6},
            "tools": {
                "restrictToWorkspace": True,
                "mcpServers": {"notes": {"command": "notes-mcp"}},
            },
        }
    )
    assert entry.workspace == "~/research"
    assert entry.model == "anthropic/claude-sonnet-5"
    assert entry.max_tokens == 4096
    assert entry.timezone == "Asia/Shanghai"
    assert entry.dream.interval_h == 6
    assert entry.tools.restrict_to_workspace is True
    assert entry.tools.mcp_servers["notes"].command == "notes-mcp"


@pytest.mark.parametrize("field_name", sorted(NamedAgentConfig.model_fields))
def test_set_is_distinguishable_from_unset_for_every_field(field_name: str) -> None:
    """A field carrying AgentDefaults' own default must still read as configured."""
    default_value = getattr(NamedAgentConfig(), field_name)
    configured = NamedAgentConfig.model_validate(
        {field_name: default_value.model_dump() if hasattr(default_value, "model_dump") else default_value}
    )

    assert field_name in configured.model_fields_set
    assert field_name not in NamedAgentConfig().model_fields_set


def test_timezone_stays_unset_unlike_agent_defaults() -> None:
    """AgentDefaults injects a detected timezone; an overlay entry must not."""
    assert "timezone" in AgentDefaults().model_fields_set
    assert "timezone" not in NamedAgentConfig().model_fields_set
    assert "timezone_mode" not in NamedAgentConfig().model_fields_set

    explicit = NamedAgentConfig.model_validate({"timezone": "Asia/Shanghai"})
    assert explicit.model_fields_set == {"timezone"}


def test_invalid_timezone_is_still_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown timezone"):
        NamedAgentConfig.model_validate({"timezone": "Mars/Olympus"})


def test_serialization_keeps_only_configured_fields() -> None:
    """A save/load round-trip must not freeze unset fields into the overlay."""
    entry = NamedAgentConfig.model_validate({"model": "x", "tools": {"restrictToWorkspace": True}})
    dumped = entry.model_dump(mode="json", by_alias=True)

    assert dumped == {"model": "x", "tools": {"restrictToWorkspace": True}}
    assert NamedAgentConfig.model_validate(dumped).model_fields_set == entry.model_fields_set


def test_serialization_uses_the_configured_alias() -> None:
    entry = NamedAgentConfig.model_validate({"idleCompactAfterMinutes": 30})
    assert entry.model_dump(mode="json", by_alias=True) == {"idleCompactAfterMinutes": 30}
    assert entry.model_dump(mode="json", by_alias=False) == {"session_ttl_minutes": 30}


@pytest.mark.parametrize("name", ["research", "a", "0", "trading-bot", "health_agent", "a1-b_2"])
def test_valid_agent_names_are_accepted(name: str) -> None:
    config = Config.model_validate({"agents": {"named": {name: {}}}})
    assert set(config.agents.named) == {name}


@pytest.mark.parametrize(
    "name",
    ["Research", "-lead", "_lead", "has space", "dots.here", "trailing!", "", "ÜBER"],
)
def test_invalid_agent_names_are_rejected(name: str) -> None:
    with pytest.raises(ValidationError, match="agent names must match"):
        Config.model_validate({"agents": {"named": {name: {}}}})


def test_default_is_reserved() -> None:
    with pytest.raises(ValidationError, match="reserved for agents.defaults"):
        Config.model_validate({"agents": {"named": {"default": {}}}})

    with pytest.raises(ValueError, match="reserved for agents.defaults"):
        validate_agent_name("default")


def test_config_without_named_is_unchanged() -> None:
    assert Config().agents.named == {}
    assert AgentsConfig().named == {}
    assert "named" not in Config().model_dump(mode="json", by_alias=True)["agents"]


def test_named_round_trips_through_the_root_config() -> None:
    raw = {"agents": {"named": {"research": {"model": "x"}}}}
    config = Config.model_validate(raw)
    dumped = config.model_dump(mode="json", by_alias=True)

    assert dumped["agents"]["named"] == {"research": {"model": "x"}}
    assert Config.model_validate(dumped).agents.named["research"].model_fields_set == {"model"}


def test_unknown_keys_under_an_entry_are_ignored() -> None:
    entry = NamedAgentConfig.model_validate({"model": "x", "nonsense": 1})
    assert entry.model_fields_set == {"model"}

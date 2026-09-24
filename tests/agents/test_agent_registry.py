"""The agent registry and the inbound routing decision, both resolved from config."""

from pathlib import Path
from typing import Any

import pytest

from nanobot.agents import (
    AgentRegistryEntry,
    agent_names,
    agent_registry,
    named_agent_workspace,
    route,
)
from nanobot.config.schema import Config


def make_config(
    *,
    named: dict[str, Any] | None = None,
    telegram: dict[str, Any] | None = None,
) -> Config:
    raw: dict[str, Any] = {
        "agents": {
            "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
            "named": named or {},
        },
    }
    if telegram is not None:
        raw["channels"] = {"telegram": telegram}
    return Config.model_validate(raw)


def bound_config() -> Config:
    """Three bots: the default one unbound, two bound to declared agents."""
    return make_config(
        named={"research": {"model": "r/model"}, "ops": {"workspace": "~/ops-ws"}},
        telegram={
            "instances": [
                {"id": "default", "token": "default-token"},
                {"id": "research", "token": "research-token", "agent": "research"},
                {"id": "ops", "token": "ops-token", "agent": "ops"},
            ]
        },
    )


# --- route ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        ("telegram.research", "research"),
        ("telegram.ops", "ops"),
    ],
)
def test_a_bound_bot_routes_to_its_agent(channel: str, expected: str) -> None:
    assert route(bound_config(), channel, "chat-1") == expected


def test_the_plain_telegram_channel_routes_to_default_when_unbound() -> None:
    assert route(bound_config(), "telegram", "chat-1") == "default"


def test_a_named_bot_with_no_agent_field_routes_to_default() -> None:
    config = make_config(
        named={"research": {}},
        telegram={
            "instances": [
                {"id": "default", "token": "default-token"},
                {"id": "marketing", "token": "marketing-token"},
            ]
        },
    )

    assert route(config, "telegram.marketing", "chat-1") == "default"


def test_a_bot_with_a_blank_agent_field_routes_to_default() -> None:
    config = make_config(
        named={"research": {}},
        telegram={"instances": [{"id": "spare", "token": "t", "agent": "   "}]},
    )

    assert route(config, "telegram.spare", "chat-1") == "default"


@pytest.mark.parametrize(
    "channel",
    ["feishu", "feishu.product", "discord", "slack", "cli", "system", "websocket"],
)
def test_every_non_telegram_channel_routes_to_default(channel: str) -> None:
    assert route(bound_config(), channel, "chat-1") == "default"


def test_a_config_with_no_telegram_section_routes_everything_to_default() -> None:
    config = make_config(named={"research": {"model": "r/model"}})

    assert route(config, "telegram", "chat-1") == "default"
    assert route(config, "telegram.research", "chat-1") == "default"


def test_a_legacy_flat_telegram_section_routes_to_default() -> None:
    config = make_config(
        named={"research": {}},
        telegram={"enabled": True, "token": "legacy-token"},
    )

    assert route(config, "telegram", "chat-1") == "default"


@pytest.mark.parametrize("chat_id", ["1", "-100200300", "", None, "another-chat"])
def test_chat_id_does_not_affect_the_result(chat_id: str | None) -> None:
    config = bound_config()

    assert route(config, "telegram.research", chat_id) == "research"
    assert route(config, "telegram", chat_id) == "default"


def test_routing_ignores_whether_the_bot_is_enabled() -> None:
    config = make_config(
        named={"research": {}},
        telegram={
            "instances": [
                {"id": "research", "token": "t", "agent": "research", "enabled": False},
            ]
        },
    )

    assert route(config, "telegram.research", "chat-1") == "research"


def test_an_undeclared_agent_name_is_reported_rather_than_silently_defaulted() -> None:
    # Detecting this is out of scope; returning the name verbatim keeps the
    # mismatch visible instead of sending that bot's traffic to `default`.
    config = make_config(
        named={"research": {}},
        telegram={"instances": [{"id": "ghost", "token": "t", "agent": "typo"}]},
    )

    assert route(config, "telegram.ghost", "chat-1") == "typo"


# --- registry ---------------------------------------------------------------


def test_a_config_with_no_named_agents_has_exactly_one_entry() -> None:
    entries = agent_registry(make_config())

    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, AgentRegistryEntry)
    assert entry.name == "default"
    assert entry.is_default
    assert entry.workspace == Path("~/default-workspace").expanduser()
    assert entry.model == "base/model"
    assert entry.channels == ()


def test_the_registry_lists_default_first_then_named_agents_in_order() -> None:
    entries = agent_registry(bound_config())

    assert [entry.name for entry in entries] == ["default", "research", "ops"]
    assert [entry.is_default for entry in entries] == [True, False, False]


def test_each_entry_carries_its_resolved_workspace_and_model() -> None:
    entries = {entry.name: entry for entry in agent_registry(bound_config())}

    assert entries["default"].workspace == Path("~/default-workspace").expanduser()
    assert entries["default"].model == "base/model"
    # Stated by the agent itself.
    assert entries["research"].model == "r/model"
    # Inherited from agents.defaults, not the raw (absent) entry value.
    assert entries["ops"].model == "base/model"
    # Never inherited: an agent that states none owns one under the agents root.
    assert entries["research"].workspace == Path(named_agent_workspace("research")).expanduser()
    assert entries["ops"].workspace == Path("~/ops-ws").expanduser()


def test_each_entry_carries_the_runtime_channel_names_bound_to_it() -> None:
    entries = {entry.name: entry for entry in agent_registry(bound_config())}

    assert entries["default"].channels == ("telegram",)
    assert entries["research"].channels == ("telegram.research",)
    assert entries["ops"].channels == ("telegram.ops",)


def test_several_bots_can_share_one_agent() -> None:
    config = make_config(
        named={"research": {}},
        telegram={
            "instances": [
                {"id": "default", "token": "a"},
                {"id": "papers", "token": "b", "agent": "research"},
                {"id": "labs", "token": "c", "agent": "research"},
                {"id": "spare", "token": "d"},
            ]
        },
    )
    entries = {entry.name: entry for entry in agent_registry(config)}

    assert entries["default"].channels == ("telegram", "telegram.spare")
    assert entries["research"].channels == ("telegram.papers", "telegram.labs")


def test_a_bot_bound_to_an_undeclared_agent_adds_no_registry_entry() -> None:
    config = make_config(
        named={"research": {}},
        telegram={"instances": [{"id": "ghost", "token": "t", "agent": "typo"}]},
    )
    entries = agent_registry(config)

    assert [entry.name for entry in entries] == ["default", "research"]
    assert all(entry.channels == () for entry in entries)


def test_every_registry_entry_agrees_with_route() -> None:
    config = bound_config()

    for entry in agent_registry(config):
        for channel in entry.channels:
            assert route(config, channel, "chat-1") == entry.name


def test_to_dict_is_json_ready() -> None:
    entry = agent_registry(bound_config())[1]

    assert entry.to_dict() == {
        "name": "research",
        "workspace": str(Path(named_agent_workspace("research")).expanduser()),
        "model": "r/model",
        "channels": ["telegram.research"],
    }


def test_agent_names_lists_default_first() -> None:
    assert agent_names(bound_config()) == ["default", "research", "ops"]
    assert agent_names(make_config()) == ["default"]


def test_the_registry_does_not_start_a_gateway_or_mutate_the_config() -> None:
    config = bound_config()
    before = config.model_dump(mode="json", by_alias=True)

    agent_registry(config)

    assert config.model_dump(mode="json", by_alias=True) == before

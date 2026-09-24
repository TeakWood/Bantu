"""Multi-instance contract for the Telegram channel package."""

from __future__ import annotations

from typing import Any

import pytest

from nanobot.channels.contracts import (
    channel_instance_config,
    channel_instance_specs,
    channel_runtime_name,
    channel_set_config_enabled,
)
from nanobot.channels.plugin import load_channel_package
from nanobot.channels.telegram.config import TelegramConfig, telegram_default_config
from nanobot.channels.telegram.instances import (
    TELEGRAM_MANAGEMENT,
    canonical_telegram_section,
    runtime_channel_name,
    telegram_instance_specs,
    upsert_telegram_instance,
    validate_instance_id,
)

MULTI_BOT_SECTION: dict[str, Any] = {
    "instances": [
        {"enabled": True, "token": "111:default"},
        {"id": "research", "enabled": True, "token": "222:research", "agent": "scholar"},
        {"id": "ops", "enabled": True, "token": "333:ops", "agent": "operator"},
    ]
}


def _plugin():
    plugin = load_channel_package("telegram")
    assert plugin is not None
    return plugin


def _specs(section: Any, *, enabled_only: bool = False):
    return telegram_instance_specs(section, telegram_default_config(), enabled_only=enabled_only)


def test_management_declares_multi_instance_support() -> None:
    assert TELEGRAM_MANAGEMENT.multi_instance is True
    assert _plugin().management.multi_instance is True


def test_default_bot_plus_named_bots_load() -> None:
    specs = _specs(MULTI_BOT_SECTION, enabled_only=True)

    assert [spec.instance_id for spec in specs] == ["default", "research", "ops"]
    assert [spec.config["token"] for spec in specs] == [
        "111:default",
        "222:research",
        "333:ops",
    ]


def test_runtime_channel_name_namespaces_named_bots_only() -> None:
    assert runtime_channel_name("telegram", "default") == "telegram"
    assert runtime_channel_name("telegram", "research") == "telegram.research"


def test_plugin_runtime_names_follow_instance_ids() -> None:
    plugin = _plugin()
    names = [
        channel_runtime_name(plugin, spec.instance_id)
        for spec in channel_instance_specs(plugin, MULTI_BOT_SECTION, enabled_only=False)
    ]

    assert names == ["telegram", "telegram.research", "telegram.ops"]


def test_each_bot_exposes_its_bound_agent() -> None:
    agents = {spec.instance_id: spec.config["agent"] for spec in _specs(MULTI_BOT_SECTION)}

    assert agents == {"default": "", "research": "scholar", "ops": "operator"}


def test_bot_without_agent_field_falls_back_to_default_agent() -> None:
    (spec,) = _specs({"enabled": True, "token": "111:default"})

    assert spec.config["agent"] == ""


def test_blank_agent_field_is_normalized() -> None:
    (spec,) = _specs({"enabled": True, "token": "111:default", "agent": "  scholar  "})

    assert spec.config["agent"] == "scholar"


def test_legacy_flat_section_loads_as_the_default_instance() -> None:
    plugin = _plugin()
    section = {"enabled": True, "token": "111:legacy", "groupPolicy": "open"}

    specs = channel_instance_specs(plugin, section, enabled_only=True)

    assert len(specs) == 1
    assert specs[0].instance_id == "default"
    assert channel_runtime_name(plugin, specs[0].instance_id) == "telegram"
    assert specs[0].config["token"] == "111:legacy"
    assert specs[0].config["groupPolicy"] == "open"


def test_legacy_flat_section_is_not_rewritten_by_reading_it() -> None:
    section = {"enabled": True, "token": "111:legacy"}

    _specs(section)

    assert section == {"enabled": True, "token": "111:legacy"}


def test_sibling_keys_become_inherited_defaults() -> None:
    section = {
        "groupPolicy": "open",
        "streaming": False,
        "instances": [
            {"enabled": True, "token": "111:default"},
            {"id": "research", "enabled": True, "token": "222:research", "groupPolicy": "mention"},
        ],
    }

    specs = _specs(section)

    assert specs[0].config["groupPolicy"] == "open"
    assert specs[1].config["groupPolicy"] == "mention"
    assert all(spec.config["streaming"] is False for spec in specs)


def test_disabled_instances_are_filtered_when_enabled_only() -> None:
    section = {
        "instances": [
            {"enabled": False, "token": "111:default"},
            {"id": "research", "enabled": True, "token": "222:research"},
        ]
    }

    assert [spec.instance_id for spec in _specs(section, enabled_only=True)] == ["research"]
    assert [spec.instance_id for spec in _specs(section)] == ["default", "research"]


def test_duplicate_instance_ids_are_dropped_from_runtime_specs() -> None:
    section = {
        "instances": [
            {"id": "research", "enabled": True, "token": "222:research"},
            {"id": "research", "enabled": True, "token": "333:clash"},
        ]
    }

    specs = _specs(section, enabled_only=True)

    assert [spec.instance_id for spec in specs] == ["research"]
    assert specs[0].config["token"] == "222:research"


def test_duplicate_instance_ids_are_rejected_when_writing() -> None:
    section = {
        "instances": [
            {"id": "research", "token": "222:research"},
            {"id": "research", "token": "333:clash"},
        ]
    }

    with pytest.raises(ValueError, match="duplicate Telegram instance id 'research'"):
        canonical_telegram_section(section, telegram_default_config())


@pytest.mark.parametrize("value", ["", "   ", "has space", "dot.ted", "slash/ed"])
def test_invalid_instance_ids_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match=r"\[A-Za-z0-9_-\]\+"):
        validate_instance_id(value)


def test_upsert_migrates_a_legacy_flat_section_and_adds_a_bot() -> None:
    defaults = telegram_default_config()
    section = {"enabled": True, "token": "111:legacy"}

    migrated = upsert_telegram_instance(
        section,
        defaults,
        "research",
        {"enabled": True, "token": "222:research", "agent": "scholar"},
    )

    instances = migrated["instances"]
    assert [instance["id"] for instance in instances] == ["default", "research"]
    assert instances[0]["token"] == "111:legacy"
    assert instances[1]["agent"] == "scholar"


def test_upsert_updates_an_existing_bot_in_place() -> None:
    defaults = telegram_default_config()

    updated = upsert_telegram_instance(
        MULTI_BOT_SECTION,
        defaults,
        "research",
        {"agent": "librarian"},
    )

    research = next(item for item in updated["instances"] if item["id"] == "research")
    assert research["agent"] == "librarian"
    assert research["token"] == "222:research"


def test_toggling_one_bot_leaves_its_siblings_alone() -> None:
    plugin = _plugin()

    disabled = channel_set_config_enabled(plugin, MULTI_BOT_SECTION, False, instance_id="research")

    states = {item["id"]: item["enabled"] for item in disabled["instances"]}
    assert states == {"default": True, "research": False, "ops": True}
    assert channel_instance_config(plugin, disabled, instance_id="research")["agent"] == "scholar"


def test_instance_config_round_trips_through_the_runtime_model() -> None:
    (spec,) = _specs({"instances": [{"id": "research", "enabled": True, "token": "222:x", "agent": "scholar"}]})

    config = TelegramConfig.model_validate(spec.config)

    assert config.instance_id == "research"
    assert config.agent == "scholar"
    assert config.token == "222:x"


def test_default_config_carries_instance_identity_fields() -> None:
    defaults = telegram_default_config()

    assert defaults["instanceId"] == "default"
    assert defaults["agent"] == ""

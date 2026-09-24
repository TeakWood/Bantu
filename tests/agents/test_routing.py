"""Telegram bot-to-agent binding and inbound routing.

Covers acceptance criterion 7.
"""

from __future__ import annotations

from pathlib import Path

from nanobot.agents import DEFAULT_AGENT_NAME, agent_channels, channel_agent_bindings, route
from nanobot.channels.contracts import channel_instance_specs, channel_runtime_name
from nanobot.channels.registry import load_channel_plugin
from nanobot.config.loader import load_config, resolve_config_env_vars
from nanobot.config.schema import Config

from .conftest import write_config

THREE_BOTS = {
    "channels": {
        "telegram": {
            "enabled": True,
            "instances": [
                {"id": "default", "token": "111:aaa"},
                {"id": "research", "token": "222:bbb", "agent": "research"},
                {"id": "trader", "token": "333:ccc", "agent": "trader"},
            ],
        }
    },
    "agents": {"named": {"research": {}, "trader": {}}},
}


def _load(path: Path) -> Config:
    return resolve_config_env_vars(load_config(path), config_path=path)


def test_a_config_with_the_default_bot_plus_two_bots_with_ids_loads(
    instance_dir: Path,
) -> None:
    config = _load(write_config(instance_dir, THREE_BOTS))

    plugin = load_channel_plugin("telegram")
    specs = channel_instance_specs(plugin, config.channels.telegram, enabled_only=False)
    runtime_names = [channel_runtime_name(plugin, spec.instance_id) for spec in specs]

    assert runtime_names == ["telegram", "telegram.research", "telegram.trader"]


def test_route_returns_each_bots_bound_agent(instance_dir: Path) -> None:
    config = _load(write_config(instance_dir, THREE_BOTS))

    assert route(config, "telegram.research", "42") == "research"
    assert route(config, "telegram.trader", "42") == "trader"


def test_route_returns_default_for_the_unbound_bot_and_other_channels(
    instance_dir: Path,
) -> None:
    config = _load(write_config(instance_dir, THREE_BOTS))

    assert route(config, "telegram", "42") == DEFAULT_AGENT_NAME
    for channel in ("cli", "websocket", "slack", "feishu.sales", "telegram.unknown"):
        assert route(config, channel, "42") == DEFAULT_AGENT_NAME


def test_chat_id_does_not_change_the_result(instance_dir: Path) -> None:
    config = _load(write_config(instance_dir, THREE_BOTS))

    assert {route(config, "telegram.research", chat) for chat in ("1", "2", "")} == {"research"}


def test_each_bot_is_listed_under_its_agent(instance_dir: Path) -> None:
    config = _load(write_config(instance_dir, THREE_BOTS))

    assert agent_channels(config) == {
        DEFAULT_AGENT_NAME: ("telegram",),
        "research": ("telegram.research",),
        "trader": ("telegram.trader",),
    }


def test_a_bot_bound_to_an_undeclared_agent_falls_back_to_default(
    instance_dir: Path,
) -> None:
    config = _load(
        write_config(
            instance_dir,
            {
                "channels": {
                    "telegram": {
                        "enabled": True,
                        "instances": [
                            {"id": "default", "token": "111:aaa"},
                            {"id": "ghost", "token": "222:bbb", "agent": "not-declared"},
                        ],
                    }
                }
            },
        )
    )

    assert channel_agent_bindings(config) == {}
    assert route(config, "telegram.ghost", "1") == DEFAULT_AGENT_NAME
    assert agent_channels(config) == {
        DEFAULT_AGENT_NAME: ("telegram", "telegram.ghost"),
    }


def test_a_pre_existing_single_bot_config_loads_unchanged(instance_dir: Path) -> None:
    config = _load(
        write_config(
            instance_dir,
            {
                "channels": {
                    "telegram": {
                        "enabled": True,
                        "token": "111:aaa",
                        "groupPolicy": "open",
                    }
                }
            },
        )
    )

    plugin = load_channel_plugin("telegram")
    specs = channel_instance_specs(plugin, config.channels.telegram, enabled_only=True)

    assert len(specs) == 1
    assert specs[0].instance_id == "default"
    assert channel_runtime_name(plugin, specs[0].instance_id) == "telegram"
    assert specs[0].config["token"] == "111:aaa"
    assert specs[0].config["groupPolicy"] == "open"
    assert route(config, "telegram", "42") == DEFAULT_AGENT_NAME
    assert agent_channels(config) == {DEFAULT_AGENT_NAME: ("telegram",)}


def test_a_config_with_no_telegram_section_routes_everything_to_default(
    instance_dir: Path,
) -> None:
    config = _load(write_config(instance_dir, {"agents": {"named": {"research": {}}}}))

    assert route(config, "telegram", "1") == DEFAULT_AGENT_NAME
    assert agent_channels(config) == {DEFAULT_AGENT_NAME: (), "research": ()}


def test_sibling_keys_are_inherited_by_every_instance(instance_dir: Path) -> None:
    config = _load(
        write_config(
            instance_dir,
            {
                "channels": {
                    "telegram": {
                        "enabled": True,
                        "groupPolicy": "open",
                        "instances": [
                            {"id": "default", "token": "111:aaa"},
                            {"id": "research", "token": "222:bbb", "groupPolicy": "mention"},
                        ],
                    }
                }
            },
        )
    )

    plugin = load_channel_plugin("telegram")
    specs = {
        spec.instance_id: spec.config
        for spec in channel_instance_specs(plugin, config.channels.telegram, enabled_only=True)
    }

    assert specs["default"]["groupPolicy"] == "open"  # inherited from the section
    assert specs["research"]["groupPolicy"] == "mention"  # own entry wins


def test_two_instances_sharing_one_bot_token_are_deduped(instance_dir: Path) -> None:
    config = _load(
        write_config(
            instance_dir,
            {
                "channels": {
                    "telegram": {
                        "enabled": True,
                        "instances": [
                            {"id": "default", "token": "111:aaa"},
                            {"id": "clone", "token": "111:aaa"},
                        ],
                    }
                }
            },
        )
    )

    plugin = load_channel_plugin("telegram")
    specs = channel_instance_specs(plugin, config.channels.telegram, enabled_only=True)

    assert [spec.instance_id for spec in specs] == ["default"]


def test_a_disabled_instance_is_not_started(instance_dir: Path) -> None:
    config = _load(
        write_config(
            instance_dir,
            {
                "channels": {
                    "telegram": {
                        "instances": [
                            {"id": "default", "token": "111:aaa", "enabled": True},
                            {"id": "research", "token": "222:bbb", "enabled": False},
                        ],
                    }
                }
            },
        )
    )

    plugin = load_channel_plugin("telegram")
    enabled = channel_instance_specs(plugin, config.channels.telegram, enabled_only=True)
    declared = channel_instance_specs(plugin, config.channels.telegram, enabled_only=False)

    assert [spec.instance_id for spec in enabled] == ["default"]
    assert [spec.instance_id for spec in declared] == ["default", "research"]


def test_updating_an_instance_migrates_a_flat_section_to_the_canonical_shape() -> None:
    from nanobot.channels.telegram.instances import update_managed_telegram_instance

    section = {"enabled": True, "token": "111:aaa", "groupPolicy": "open"}

    updated = update_managed_telegram_instance(section, {"agent": "research"})

    assert list(updated) == ["instances"]
    assert len(updated["instances"]) == 1
    entry = updated["instances"][0]
    assert entry["id"] == "default"
    assert entry["instanceId"] == "default"
    assert entry["agent"] == "research"
    # Values the caller did not touch survive the migration.
    assert entry["token"] == "111:aaa"
    assert entry["groupPolicy"] == "open"


def test_updating_an_unknown_instance_appends_a_new_bot() -> None:
    from nanobot.channels.telegram.instances import update_managed_telegram_instance

    section = {"instances": [{"id": "default", "token": "111:aaa"}]}

    updated = update_managed_telegram_instance(
        section,
        {"token": "222:bbb", "agent": "research"},
        instance_id="research",
    )

    assert [entry["id"] for entry in updated["instances"]] == ["default", "research"]
    research = updated["instances"][1]
    assert research["token"] == "222:bbb"
    assert research["agent"] == "research"
    # A non-default instance gets a distinguishing display name.
    assert research["name"] == "nanobot research"


def test_an_existing_instance_is_updated_in_place() -> None:
    from nanobot.channels.telegram.instances import update_managed_telegram_instance

    section = {
        "instances": [
            {"id": "default", "token": "111:aaa"},
            {"id": "research", "token": "222:bbb", "agent": "research"},
        ]
    }

    updated = update_managed_telegram_instance(
        section,
        {"agent": "health"},
        instance_id="research",
    )

    assert [entry["id"] for entry in updated["instances"]] == ["default", "research"]
    assert updated["instances"][1]["agent"] == "health"
    assert updated["instances"][1]["token"] == "222:bbb"


def test_the_write_path_rejects_a_duplicate_instance_id() -> None:
    import pytest as _pytest

    from nanobot.channels.telegram.config import telegram_default_config
    from nanobot.channels.telegram.instances import canonical_telegram_section

    section = {"instances": [{"id": "a", "token": "1:a"}, {"id": "a", "token": "2:b"}]}

    with _pytest.raises(ValueError, match="duplicate"):
        canonical_telegram_section(section, telegram_default_config())


def test_the_write_path_rejects_a_malformed_instance_id() -> None:
    import pytest as _pytest

    from nanobot.channels.telegram.instances import update_managed_telegram_instance

    with _pytest.raises(ValueError, match="instance id"):
        update_managed_telegram_instance({}, {}, instance_id="has space")


def test_a_bot_token_identity_never_exposes_the_secret() -> None:
    from nanobot.channels.telegram.instances import telegram_bot_identity_key

    assert telegram_bot_identity_key("123456:SUPER-SECRET") == "telegram:123456"
    assert telegram_bot_identity_key("") == ""
    assert telegram_bot_identity_key(None) == ""


def test_a_non_object_instance_entry_is_skipped_not_fatal() -> None:
    from nanobot.channels.telegram.config import telegram_default_config
    from nanobot.channels.telegram.instances import telegram_instance_specs

    section = {"instances": ["nonsense", {"id": "ok", "token": "1:a"}]}

    specs = telegram_instance_specs(section, telegram_default_config())

    assert [spec.instance_id for spec in specs] == ["ok"]


def test_each_bot_gets_a_distinguishing_display_name(instance_dir: Path) -> None:
    config = _load(
        write_config(
            instance_dir,
            {
                "channels": {
                    "telegram": {
                        "enabled": True,
                        "instances": [
                            {"id": "default", "token": "111:aaa"},
                            {"id": "research", "token": "222:bbb"},
                            {"id": "trader", "token": "333:ccc", "name": "Portfolio desk"},
                        ],
                    }
                }
            },
        )
    )

    plugin = load_channel_plugin("telegram")
    names = {
        spec.instance_id: spec.config["name"]
        for spec in channel_instance_specs(plugin, config.channels.telegram, enabled_only=True)
    }

    assert names == {
        "default": "nanobot",
        "research": "nanobot research",
        "trader": "Portfolio desk",
    }

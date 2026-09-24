"""Telegram-owned helpers for its persisted multi-bot configuration.

Follows the Feishu convention: the `telegram` section may hold either one flat
bot config or an `instances` list, each entry carrying an `id`. A bot's runtime
channel name is `telegram.<id>`, except for the `default` instance which keeps
the bare `telegram` name so existing single-bot configs are unchanged.

Each entry may also carry an `agent` field binding that bot to a named agent;
every chat arriving through the bot then goes to that agent.
"""

from __future__ import annotations

import re
from typing import Any, cast

from loguru import logger

from nanobot.channels.contracts import ChannelInstanceSpec, ChannelManagementSpec
from nanobot.channels.telegram.config import telegram_default_config
from nanobot.config.loader import merge_missing_defaults

DEFAULT_INSTANCE_ID = "default"
_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_instance_id(value: str) -> str:
    """Return a normalized instance id or raise ValueError."""
    instance_id = value.strip()
    if not instance_id or not _INSTANCE_ID_RE.fullmatch(instance_id):
        raise ValueError("instance id must match [A-Za-z0-9_-]+")
    return instance_id


def runtime_channel_name(base_name: str, instance_id: str) -> str:
    """Return the channel key used for routing messages at runtime."""
    return base_name if instance_id == DEFAULT_INSTANCE_ID else f"{base_name}.{instance_id}"


def telegram_bot_identity_key(token: Any) -> str:
    """Return the stable identity of one bot account.

    A Telegram token is `<bot_id>:<secret>`; only the numeric id is used so the
    secret never lands in a comparison key or a log line.
    """
    token = str(token or "").strip()
    if not token:
        return ""
    return f"telegram:{token.split(':', 1)[0]}"


def _base_telegram_instance_config(defaults: dict[str, Any]) -> dict[str, Any]:
    config = dict(defaults)
    config["instanceId"] = DEFAULT_INSTANCE_ID
    config["name"] = "nanobot"
    return config


def _telegram_instance_inputs(
    section: Any,
    defaults: dict[str, Any],
) -> tuple[list[Any], dict[str, Any] | None]:
    if hasattr(section, "model_dump"):
        section = section.model_dump(mode="json", by_alias=True)
    if not isinstance(section, dict):
        section = {}
    section_data = cast(dict[str, Any], section)

    instances = section_data.get("instances")
    if isinstance(instances, list):
        inherited = {key: value for key, value in section_data.items() if key != "instances"}
        return list(cast(list[Any], instances)), inherited
    return (
        [section_data] if section_data else [_base_telegram_instance_config(defaults)],
        None,
    )


def _normalize_telegram_instance(
    raw: dict[str, Any],
    defaults: dict[str, Any],
    *,
    inherited: dict[str, Any] | None = None,
    fallback_id: str = DEFAULT_INSTANCE_ID,
) -> dict[str, Any]:
    config = cast(dict[str, Any], merge_missing_defaults(inherited or {}, defaults))
    config = cast(dict[str, Any], merge_missing_defaults(raw, config))

    raw_id = raw.get("id") or raw.get("instanceId") or raw.get("instance_id") or fallback_id
    instance_id = validate_instance_id(str(raw_id))
    config["id"] = instance_id
    config["instanceId"] = instance_id
    # The channel default carries one name for every bot, so derive a
    # distinguishing one unless this entry (or the section) states its own.
    stated_name = raw.get("name") or (inherited or {}).get("name")
    config["name"] = (
        str(stated_name)
        if stated_name
        else ("nanobot" if instance_id == DEFAULT_INSTANCE_ID else f"nanobot {instance_id}")
    )
    return config


def telegram_instance_specs(
    section: Any,
    defaults: dict[str, Any],
    *,
    enabled_only: bool = False,
) -> list[ChannelInstanceSpec]:
    """Expand legacy or canonical Telegram config into runtime instance specs."""
    raw_specs, inherited = _telegram_instance_inputs(section, defaults)

    specs: list[ChannelInstanceSpec] = []
    instance_ids: set[str] = set()
    identity_owners: dict[str, str] = {}
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            logger.warning(
                "Skipping invalid Telegram instance at index {}: expected an object", index,
            )
            continue
        fallback_id = DEFAULT_INSTANCE_ID if index == 0 else f"bot-{index + 1}"
        try:
            config = _normalize_telegram_instance(
                cast(dict[str, Any], raw),
                defaults,
                inherited=inherited,
                fallback_id=fallback_id,
            )
        except ValueError as exc:
            logger.warning("Skipping invalid Telegram instance config: {}", exc)
            continue

        instance_id = str(config["instanceId"])
        if instance_id in instance_ids:
            logger.warning("Skipping duplicate Telegram instance id '{}'", instance_id)
            continue

        instance_ids.add(instance_id)
        enabled = bool(config.get("enabled", defaults.get("enabled", False)))
        if enabled_only and not enabled:
            continue

        identity = telegram_bot_identity_key(config.get("token"))
        if enabled_only and identity:
            if identity in identity_owners:
                logger.warning(
                    "Skipping Telegram instance '{}' because it uses the same bot as instance '{}'",
                    instance_id,
                    identity_owners[identity],
                )
                continue
            identity_owners[identity] = instance_id

        specs.append(ChannelInstanceSpec(instance_id=instance_id, config=config))

    return specs


def canonical_telegram_section(section: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    """Return the section in canonical ``{"instances": [...]}`` shape.

    Unlike spec expansion this raises rather than skipping, because it is a
    write path that must not silently drop a user's bot.
    """
    raw_specs, inherited = _telegram_instance_inputs(section, defaults)
    instances: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            raise ValueError(f"Telegram instance at index {index} must be an object")
        fallback_id = DEFAULT_INSTANCE_ID if index == 0 else f"bot-{index + 1}"
        config = _normalize_telegram_instance(
            cast(dict[str, Any], raw),
            defaults,
            inherited=inherited,
            fallback_id=fallback_id,
        )
        instance_id = str(config["instanceId"])
        if instance_id in seen:
            raise ValueError(f"duplicate Telegram instance id '{instance_id}'")
        seen.add(instance_id)
        instances.append(config)
    return {"instances": instances}


def upsert_telegram_instance(
    section: Any,
    defaults: dict[str, Any],
    instance_id: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Merge *values* into one instance, returning the canonical section."""
    instance_id = validate_instance_id(instance_id)
    canonical = canonical_telegram_section(section, defaults)
    instances = cast(list[dict[str, Any]], canonical["instances"])
    for index, entry in enumerate(instances):
        if str(entry.get("id") or entry.get("instanceId")) == instance_id:
            instances[index] = _normalize_telegram_instance(
                {**entry, **values},
                defaults,
                fallback_id=instance_id,
            )
            return canonical
    instances.append(
        _normalize_telegram_instance(
            {**values, "id": instance_id},
            defaults,
            fallback_id=instance_id,
        )
    )
    return canonical


def managed_telegram_instance_specs(
    section: Any,
    *,
    enabled_only: bool = True,
) -> list[ChannelInstanceSpec]:
    return telegram_instance_specs(
        section,
        telegram_default_config(),
        enabled_only=enabled_only,
    )


def update_managed_telegram_instance(
    section: Any,
    values: dict[str, Any],
    *,
    instance_id: str = DEFAULT_INSTANCE_ID,
) -> dict[str, Any]:
    existing = cast(dict[str, Any], section) if isinstance(section, dict) else {}
    return upsert_telegram_instance(
        existing,
        telegram_default_config(),
        instance_id,
        values,
    )


def telegram_agent_bindings(section: Any) -> dict[str, str]:
    """Map each Telegram runtime channel name to the agent name it is bound to.

    Bots with no ``agent`` field are omitted; the caller routes them to the
    default agent.
    """
    bindings: dict[str, str] = {}
    for spec in telegram_instance_specs(section, telegram_default_config()):
        config = cast(dict[str, Any], spec.config)
        agent = str(config.get("agent") or "").strip()
        if not agent:
            continue
        bindings[runtime_channel_name("telegram", spec.instance_id)] = agent
    return bindings


TELEGRAM_MANAGEMENT = ChannelManagementSpec(
    multi_instance=True,
    default_config=telegram_default_config,
    instance_specs=managed_telegram_instance_specs,
    update_instance_config=update_managed_telegram_instance,
    runtime_name=runtime_channel_name,
)

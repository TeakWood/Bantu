"""Telegram-owned helpers for its persisted multi-instance configuration."""

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


def _base_telegram_instance_config(defaults: dict[str, Any]) -> dict[str, Any]:
    config = dict(defaults)
    config["instanceId"] = DEFAULT_INSTANCE_ID
    return config


def telegram_bound_agent(config: Any) -> str:
    """Return the agent a bot is bound to, empty when it falls back to default."""
    if hasattr(config, "model_dump"):
        config = config.model_dump(mode="json", by_alias=True)
    if not isinstance(config, dict):
        return ""
    values = cast(dict[str, Any], config)
    return str(values.get("agent") or "").strip()


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
    # A bot with no agent field belongs to the default agent.
    config["agent"] = telegram_bound_agent(config)
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
    return ([section_data] if section_data else [_base_telegram_instance_config(defaults)]), None


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
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            logger.warning("Skipping invalid Telegram instance at index {}: expected an object", index)
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

        specs.append(
            ChannelInstanceSpec(
                instance_id=instance_id,
                config=config,
            )
        )

    return specs


def canonical_telegram_section(section: Any, defaults: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical section, rejecting input that cannot be preserved safely."""
    raw_specs, inherited = _telegram_instance_inputs(section, defaults)
    instances: list[dict[str, Any]] = []
    instance_ids: set[str] = set()

    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            raise ValueError(f"Telegram instance at index {index} must be an object")
        fallback_id = DEFAULT_INSTANCE_ID if index == 0 else f"bot-{index + 1}"
        try:
            config = _normalize_telegram_instance(
                cast(dict[str, Any], raw),
                defaults,
                inherited=inherited,
                fallback_id=fallback_id,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid Telegram instance at index {index}: {exc}") from exc

        instance_id = str(config["instanceId"])
        if instance_id in instance_ids:
            raise ValueError(f"duplicate Telegram instance id '{instance_id}'")
        instance_ids.add(instance_id)
        instances.append(config)

    return {"instances": instances}


def upsert_telegram_instance(
    section: Any,
    defaults: dict[str, Any],
    instance_id: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Return canonical Telegram section with one instance created or updated."""
    instance_id = validate_instance_id(instance_id)
    canonical = canonical_telegram_section(section, defaults)
    instances = canonical.setdefault("instances", [])

    for instance in instances:
        if instance.get("id") == instance_id or instance.get("instanceId") == instance_id:
            instance.update(values)
            instance["id"] = instance_id
            instance["instanceId"] = instance_id
            instance["agent"] = telegram_bound_agent(instance)
            return canonical

    config = _normalize_telegram_instance(
        {**values, "id": instance_id},
        defaults,
        fallback_id=instance_id,
    )
    instances.append(config)
    return canonical


TELEGRAM_MANAGEMENT = ChannelManagementSpec(
    multi_instance=True,
    default_config=telegram_default_config,
    instance_specs=managed_telegram_instance_specs,
    update_instance_config=update_managed_telegram_instance,
    runtime_name=runtime_channel_name,
)


__all__ = [
    "DEFAULT_INSTANCE_ID",
    "TELEGRAM_MANAGEMENT",
    "canonical_telegram_section",
    "runtime_channel_name",
    "telegram_bound_agent",
    "telegram_instance_specs",
    "upsert_telegram_instance",
    "validate_instance_id",
]

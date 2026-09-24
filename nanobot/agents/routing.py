"""Decide which agent serves an inbound message.

Each agent talks to the user through its own Telegram bot, so the binding is
declared per bot in the channel config. Every other channel, and any bot with no
``agent`` field, goes to the default agent.
"""

from __future__ import annotations

from nanobot.agents.registry import DEFAULT_AGENT_NAME, agent_names
from nanobot.config.schema import Config

__all__ = [
    "agent_channels",
    "channel_agent_bindings",
    "declared_agent_bindings",
    "route",
]


def declared_agent_bindings(config: Config) -> dict[str, str]:
    """Map each bound runtime channel name to the agent name its config names.

    Unlike :func:`channel_agent_bindings` a binding is kept even when no such
    agent is declared, so a caller can tell "bound to an agent that does not
    exist" apart from "not bound at all". The gateway needs that distinction:
    serving the first from ``default`` would put another agent's conversation in
    the default agent's memory, which is the leak named agents exist to prevent.
    """
    from nanobot.channels.telegram.instances import telegram_agent_bindings

    section = getattr(config.channels, "telegram", None)
    if section is None:
        return {}
    return telegram_agent_bindings(section)


def channel_agent_bindings(config: Config) -> dict[str, str]:
    """Map each bound runtime channel name to the agent that serves it.

    A binding naming an undeclared agent is dropped rather than raising, so one
    stale entry cannot take down the gateway.
    """
    declared = set(agent_names(config))
    return {
        channel: agent
        for channel, agent in declared_agent_bindings(config).items()
        if agent in declared
    }


def route(config: Config, channel: str, chat_id: str) -> str:
    """Return the name of the agent an inbound message is routed to.

    ``chat_id`` does not affect the result today; it is in the signature so
    per-chat binding can be added later without changing every caller.
    """
    del chat_id  # reserved for per-chat binding
    return channel_agent_bindings(config).get(channel, DEFAULT_AGENT_NAME)


def agent_channels(config: Config) -> dict[str, tuple[str, ...]]:
    """Map each agent name to the runtime channel names bound to it."""
    from nanobot.channels.telegram.config import telegram_default_config
    from nanobot.channels.telegram.instances import (
        runtime_channel_name,
        telegram_instance_specs,
    )

    channels: dict[str, list[str]] = {name: [] for name in agent_names(config)}
    section = getattr(config.channels, "telegram", None)
    if section is not None:
        bindings = channel_agent_bindings(config)
        for spec in telegram_instance_specs(section, telegram_default_config()):
            name = runtime_channel_name("telegram", spec.instance_id)
            channels[bindings.get(name, DEFAULT_AGENT_NAME)].append(name)
    return {name: tuple(values) for name, values in channels.items()}

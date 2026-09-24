"""The list of agents, and the inbound routing decision, resolved from config alone.

Both surfaces are pure functions of :class:`Config`: nothing here starts a
gateway, imports a channel SDK or touches the network, so the CLI can render the
registry and a composition root can decide routing before any runtime exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.agents.resolution import resolve_agent_config
from nanobot.channels.telegram.instances import (
    managed_telegram_instance_specs,
    runtime_channel_name,
    telegram_bound_agent,
)
from nanobot.config.schema import RESERVED_AGENT_NAME, Config

__all__ = [
    "AgentRegistryEntry",
    "agent_names",
    "agent_registry",
    "route",
]

TELEGRAM_CHANNEL = "telegram"


@dataclass(frozen=True)
class AgentRegistryEntry:
    """One agent as the CLI and the composition root see it."""

    name: str
    workspace: Path
    model: str
    channels: tuple[str, ...]

    @property
    def is_default(self) -> bool:
        """Whether this is the default agent, which owns the top-level blocks."""
        return self.name == RESERVED_AGENT_NAME

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping for machine-readable output."""
        return {
            "name": self.name,
            "workspace": str(self.workspace),
            "model": self.model,
            "channels": list(self.channels),
        }


def agent_names(config: Config) -> list[str]:
    """Return every configured agent name, ``default`` first."""
    return [RESERVED_AGENT_NAME, *config.agents.named]


def _channel_bindings(config: Config) -> dict[str, str]:
    """Return runtime channel name -> bound agent name, empty when unbound.

    Telegram is the only channel that carries a per-instance ``agent`` field, so
    it is the only source of bindings; every other channel is absent from the
    mapping and therefore belongs to the default agent.  Instance enablement is
    deliberately ignored: a binding is a property of the config, and a channel
    that is switched off simply never produces a message to route.
    """
    section = getattr(config.channels, TELEGRAM_CHANNEL, None)
    if not section:
        # No telegram section at all: the channel defaults would otherwise
        # expand into a phantom 'telegram' bot the install never declared.
        return {}
    return {
        runtime_channel_name(TELEGRAM_CHANNEL, spec.instance_id): telegram_bound_agent(spec.config)
        for spec in managed_telegram_instance_specs(section, enabled_only=False)
    }


def route(config: Config, channel: str, chat_id: str | None) -> str:
    """Return the agent an inbound message on runtime channel *channel* goes to.

    A Telegram bot with an ``agent`` field routes to that agent; a bot without
    one, and every non-Telegram channel — including dotted runtime names of other
    multi-instance channels such as ``feishu.product`` and internal ones such as
    ``cli`` — routes to ``default``.

    *chat_id* does not affect the result under this spec.  It is part of the
    signature so per-chat binding can be added later without changing callers.

    The name a bot declares is returned verbatim: detecting a bot bound to an
    agent that ``agents.named`` never declares is out of scope here, and
    reporting it beats silently sending that bot's traffic to ``default``.
    """
    del chat_id  # Reserved for per-chat binding; see the docstring.
    return _channel_bindings(config).get(channel, "") or RESERVED_AGENT_NAME


def agent_registry(config: Config) -> list[AgentRegistryEntry]:
    """Return every configured agent, ``default`` first, then declaration order.

    Each entry carries the agent's resolved workspace and model — the values it
    would actually run with, not the raw ``agents.named`` entry — and the runtime
    channel names that :func:`route` sends to it.
    """
    names = agent_names(config)
    bound: dict[str, list[str]] = {name: [] for name in names}
    for channel, agent in _channel_bindings(config).items():
        target = agent or RESERVED_AGENT_NAME
        if target not in bound:
            # An undeclared agent has no entry to attach the channel to.  route()
            # still reports the name the bot declares, so the mismatch surfaces
            # there rather than being invented into the registry.
            continue
        bound[target].append(channel)

    entries: list[AgentRegistryEntry] = []
    for name in names:
        resolved = resolve_agent_config(config, name)
        entries.append(
            AgentRegistryEntry(
                name=name,
                workspace=resolved.workspace,
                model=resolved.model,
                channels=tuple(bound[name]),
            )
        )
    return entries

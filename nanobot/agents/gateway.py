"""The multi-agent gateway runtime: every agent, its sessions and its subagents.

Each agent owns a private :class:`MessageBus`, so a message published for one
agent is never visible to another and two agents can never steal each other's
inbound traffic. Outbound traffic from every agent is funnelled into one queue
tagged with the runtime channel it must leave by, which is what both
``nanobot gateway`` and :func:`open_gateway` consume.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.hooks import create_file_edit_activity_hook
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agents.registry import (
    DEFAULT_AGENT_NAME,
    AgentSpec,
    agent_config,
    resolve_agent_specs,
)
from nanobot.agents.routing import declared_agent_bindings, route
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.providers.image_generation import image_gen_provider_configs
from nanobot.session.manager import SessionManager

__all__ = [
    "AgentRuntime",
    "InboundRouter",
    "MultiAgentGateway",
    "NamedAgentFleet",
    "OutboundRecord",
    "open_gateway",
]


@dataclass(frozen=True)
class OutboundRecord:
    """One outgoing message, tagged with the channel it must leave by."""

    channel: str
    chat_id: str
    content: str
    agent: str
    media: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentRuntime:
    """One agent's private bus, tool registry, sessions and loop."""

    def __init__(self, spec: AgentSpec, config: Config) -> None:
        self.spec = spec
        self.name = spec.name
        self.config = agent_config(config, spec.name)
        self.bus = MessageBus()
        self.tools = ToolRegistry()
        self.mcp_provider = MCPProvider.from_config(self.config, self.tools)
        self.sessions = SessionManager(
            spec.workspace,
            sessions_root=spec.sessions_root,
        )
        self.loop = AgentLoop.from_config(
            self.config,
            self.bus,
            session_manager=self.sessions,
            image_generation_provider_configs=image_gen_provider_configs(self.config),
            hook_factories=[create_file_edit_activity_hook],
            tool_registry=self.tools,
        )

    def tool_names(self) -> list[str]:
        """Return the names of the tools this agent's model is offered."""
        return list(self.loop.tools.tool_names)

    def subagent_tool_names(self) -> list[str]:
        """Return the names of the tools this agent's background subagents get."""
        return self.loop.subagents.tool_names()

    async def aclose(self) -> None:
        try:
            await self.loop.aclose()
        finally:
            await self.mcp_provider.aclose()


class MultiAgentGateway:
    """Run every declared agent side by side over one config."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.runtimes: dict[str, AgentRuntime] = {
            spec.name: AgentRuntime(spec, config) for spec in resolve_agent_specs(config)
        }
        self._bindings = declared_agent_bindings(config)
        self._outbound: asyncio.Queue[OutboundRecord] = asyncio.Queue()
        self._tasks: list[asyncio.Task[Any]] = []

    @property
    def agent_names(self) -> list[str]:
        return list(self.runtimes)

    def runtime(self, name: str) -> AgentRuntime:
        """Return one agent's runtime.

        Raises:
            KeyError: if no agent by that name is running.
        """
        return self.runtimes[name]

    def agent_for(self, channel: str, chat_id: str = "") -> str:
        """Return the agent serving inbound traffic on *channel*.

        A channel bound to an agent this config never declares reports that
        name, not ``default``: :meth:`inject` then fails loudly instead of
        quietly filing the conversation in the default agent's memory.
        """
        bound = self._bindings.get(channel)
        return bound if bound is not None else route(self.config, channel, chat_id)

    async def inject(
        self,
        channel: str,
        chat_id: str,
        sender_id: str,
        text: str,
    ) -> str:
        """Deliver a message as if it had arrived on runtime channel *channel*.

        Returns the name of the agent it was routed to.

        Raises:
            KeyError: if the channel is bound to an agent that is not running.
        """
        name = self.agent_for(channel, chat_id)
        if name not in self.runtimes:
            raise KeyError(
                f"channel {channel!r} is bound to agent {name!r}, which is not declared"
            )
        await self.runtimes[name].bus.publish_inbound(
            InboundMessage(
                channel=channel,
                sender_id=sender_id,
                chat_id=chat_id,
                content=text,
            )
        )
        return name

    async def next_outbound(self, timeout: float = 10.0) -> OutboundRecord:
        """Return the next outgoing message.

        Raises:
            asyncio.TimeoutError: if none arrives within *timeout* seconds.
        """
        return await asyncio.wait_for(self._outbound.get(), timeout=timeout)

    async def _pump_outbound(self, runtime: AgentRuntime) -> None:
        """Forward one agent's replies into the shared outbound queue.

        Progress, streaming and lifecycle traffic ride ``OutboundMessage.event``;
        only a plain message is a reply the user sees as one.
        """
        while True:
            try:
                msg: OutboundMessage = await runtime.bus.consume_outbound()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Agent {} outbound pump error: {}", runtime.name, exc)
                continue
            if msg.event is not None or not msg.content:
                continue
            await self._outbound.put(
                OutboundRecord(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=msg.content,
                    agent=runtime.name,
                    media=tuple(msg.media),
                    metadata=dict(msg.metadata),
                )
            )

    async def start(self) -> None:
        for runtime in self.runtimes.values():
            await runtime.mcp_provider.connect()
            self._tasks.append(
                asyncio.create_task(
                    runtime.loop.run(),
                    name=f"nanobot-agent-loop:{runtime.name}",
                )
            )
            self._tasks.append(
                asyncio.create_task(
                    self._pump_outbound(runtime),
                    name=f"nanobot-agent-outbound:{runtime.name}",
                )
            )

    async def aclose(self) -> None:
        for runtime in self.runtimes.values():
            runtime.loop.stop()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        for runtime in self.runtimes.values():
            with suppress(Exception):
                await runtime.aclose()
            with suppress(Exception):
                await runtime.bus.drain()


class InboundRouter(MessageBus):
    """The default agent's bus, with inbound traffic diverted by channel binding.

    Channels publish everything onto one bus. A message arriving on a channel
    bound to a named agent is put on that agent's private inbound queue instead,
    so the two agent loops never compete for one queue -- ``consume_inbound`` is
    an unaddressed queue read, and whichever loop got there first would win.

    A message bound to an agent that is not running is dropped rather than
    served by ``default``. Handing it to the default agent would file another
    agent's conversation in the default agent's memory and session store, which
    is exactly the compartment named agents exist to seal.
    """

    def __init__(self, config: Config, routes: dict[str, MessageBus] | None = None) -> None:
        super().__init__()
        self._config = config
        self._routes: dict[str, MessageBus] = dict(routes or {})
        self._bindings = declared_agent_bindings(config)

    async def publish_inbound(self, msg: InboundMessage) -> None:
        bound = self._bindings.get(msg.channel)
        if bound is not None and bound != DEFAULT_AGENT_NAME and bound not in self._routes:
            logger.error(
                "Dropping message on channel '{}': it is bound to agent '{}', which is not "
                "running. Declare it under agents.named or remove the binding.",
                msg.channel,
                bound,
            )
            return
        target = self._routes.get(route(self._config, msg.channel, msg.chat_id))
        if target is None:
            await super().publish_inbound(msg)
            return
        await target.publish_inbound(msg)


class NamedAgentFleet:
    """Every named agent, running beside a host runtime that owns ``default``.

    ``nanobot gateway`` owns the default agent's runtime -- its cron jobs,
    heartbeat, Dream consolidation and WebUI surface, all of which stay with
    ``default`` under this feature -- and delegates every other agent here. The
    named agents it runs are the same :class:`AgentRuntime` objects
    :func:`open_gateway` runs, so live Telegram traffic and an injected message
    reach an agent the same way.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.runtimes: dict[str, AgentRuntime] = {}
        for spec in resolve_agent_specs(config):
            if spec.is_default:
                continue
            try:
                self.runtimes[spec.name] = AgentRuntime(spec, config)
            except Exception as exc:  # noqa: BLE001 - one bad agent must not stop the gateway
                # Leaving it out is the safe failure: its bots are then served
                # by nobody rather than by the default agent.
                logger.error("Agent '{}' could not be started: {}", spec.name, exc)
        self._tasks: list[asyncio.Task[Any]] = []

    def __bool__(self) -> bool:
        return bool(self.runtimes)

    @property
    def agent_names(self) -> list[str]:
        return list(self.runtimes)

    @property
    def routes_traffic(self) -> bool:
        """Whether this config binds any channel away from the default agent."""
        return bool(self.runtimes) or bool(declared_agent_bindings(self.config))

    @property
    def unroutable_channels(self) -> tuple[str, ...]:
        """Channels bound to an agent that is not running, so served by nobody."""
        return tuple(
            sorted(
                channel
                for channel, agent in declared_agent_bindings(self.config).items()
                if agent != DEFAULT_AGENT_NAME and agent not in self.runtimes
            )
        )

    def host_bus(self) -> InboundRouter:
        """Return the bus channels publish onto, routing by channel binding."""
        return InboundRouter(
            self.config,
            {name: runtime.bus for name, runtime in self.runtimes.items()},
        )

    async def _pump_outbound(self, runtime: AgentRuntime, host: MessageBus) -> None:
        """Forward one named agent's outbound traffic to the channel-facing bus.

        Messages and events are forwarded verbatim: each already carries the
        runtime channel and chat it must leave by, so a reply goes back out
        through the same bot the message arrived on.
        """
        while True:
            try:
                msg = await runtime.bus.consume_outbound()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Agent {} outbound pump error: {}", runtime.name, exc)
                continue
            await host.publish_outbound(msg)

    async def start(self, host: MessageBus) -> list[asyncio.Task[Any]]:
        """Run every named agent, funnelling its replies back onto *host*."""
        for runtime in self.runtimes.values():
            with suppress(Exception):
                await runtime.mcp_provider.connect()
            self._tasks.append(
                asyncio.create_task(
                    runtime.loop.run(),
                    name=f"nanobot-agent-loop:{runtime.name}",
                )
            )
            self._tasks.append(
                asyncio.create_task(
                    self._pump_outbound(runtime, host),
                    name=f"nanobot-agent-outbound:{runtime.name}",
                )
            )
        return list(self._tasks)

    async def aclose(self) -> None:
        for runtime in self.runtimes.values():
            runtime.loop.stop()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        for runtime in self.runtimes.values():
            with suppress(Exception):
                await runtime.aclose()
            with suppress(Exception):
                await runtime.bus.drain()
            runtime.sessions.flush_all()


@asynccontextmanager
async def open_gateway(config_path: str | Path | None = None) -> AsyncIterator[MultiAgentGateway]:
    """Run the same multi-agent gateway runtime as ``nanobot gateway``.

    Every agent, its sessions and its subagents are live; no external channel is
    connected. Inject messages with ``inject`` and read replies with
    ``next_outbound``.
    """
    from nanobot.config.loader import load_config, resolve_config_env_vars

    resolved: Path | None = None
    if config_path is not None:
        resolved = Path(config_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Config not found: {resolved}")

    config = resolve_config_env_vars(load_config(resolved), config_path=resolved)
    gateway = MultiAgentGateway(config)
    await gateway.start()
    try:
        yield gateway
    finally:
        await gateway.aclose()

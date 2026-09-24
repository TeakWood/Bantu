"""The gateway runtime, in-process, attached to no external channel.

``nanobot gateway`` wires ``ChannelManager`` to a :class:`MultiAgentRuntime`:
every chat platform speaks one channel bus, the runtime demuxes it across the
agents, and each agent's replies are pumped back onto that same bus.  This module
keeps the runtime and replaces only the outermost edge — the channels — with two
calls::

    async with open_gateway(config_path) as gateway:
        await gateway.inject("telegram.research", "chat-1", "user-1", "hello")
        reply = await gateway.next_outbound(timeout=5.0)

:meth:`GatewayHarness.inject` publishes on the channel bus exactly where a
channel runtime would have published, and :meth:`GatewayHarness.next_outbound`
pops the queue ``ChannelManager`` would have dispatched from.  Everything between
the two — routing, sessions, tools, subagents, MCP — is the gateway's own, so a
message injected on a bound runtime channel is processed by that channel's agent
for the same reason it is in production.

Nothing here contacts a network service: no channel is constructed, so a config
carrying no Telegram token (or a placeholder one) runs exactly as a configured
one does.  The model provider is the caller's to fake, as elsewhere.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agents.multi import MultiAgentRuntime
from nanobot.agents.runtime import AgentRuntime
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.config.schema import Config

if TYPE_CHECKING:
    from nanobot.cron.service import CronService

__all__ = [
    "DEFAULT_OUTBOUND_TIMEOUT_S",
    "GatewayHarness",
    "open_gateway",
]

# How long :meth:`GatewayHarness.next_outbound` waits by default.  A turn runs a
# real ``AgentLoop``, so this is generous; it exists to fail rather than hang.
DEFAULT_OUTBOUND_TIMEOUT_S = 10.0

# Bounds on the two lifecycle waits, neither of which should ever be reached.
_READY_TIMEOUT_S = 30.0
_SHUTDOWN_TIMEOUT_S = 15.0
_READY_POLL_INTERVAL_S = 0.005


class GatewayHarness:
    """The running gateway, addressed as a channel would address it.

    Holds the same :class:`MultiAgentRuntime` ``nanobot gateway`` runs — every
    agent, its sessions and its subagents — with the channel edge replaced by
    :meth:`inject` and :meth:`next_outbound`.
    """

    def __init__(self, runtime: MultiAgentRuntime) -> None:
        self._runtime = runtime
        self._task: asyncio.Task[None] | None = None
        # Event-carrying outbound messages skipped by ``next_outbound``: progress,
        # stream deltas and turn lifecycle.  Kept rather than dropped so a caller
        # that wants them can still read them, in arrival order.
        self.events: list[OutboundMessage] = []

    # --- what is running ----------------------------------------------------

    @property
    def runtime(self) -> MultiAgentRuntime:
        """The multi-agent runtime this harness drives."""
        return self._runtime

    @property
    def config(self) -> Config:
        """The configuration every agent was resolved from."""
        return self._runtime.config

    @property
    def agents(self) -> Mapping[str, AgentRuntime]:
        """Every agent, ``default`` first, keyed by name."""
        return self._runtime.agents

    @property
    def names(self) -> tuple[str, ...]:
        """Every configured agent name, ``default`` first."""
        return self._runtime.names

    @property
    def default(self) -> AgentRuntime:
        """The agent every unbound channel routes to."""
        return self._runtime.default

    @property
    def is_running(self) -> bool:
        """Whether the runtime is live."""
        return self._runtime.is_running

    def agent_for(self, channel: str, chat_id: str | None = None) -> AgentRuntime:
        """Return the agent a message on *channel* is delivered to."""
        return self._runtime.agent_for(channel, chat_id)

    def __iter__(self) -> Iterator[AgentRuntime]:
        return iter(self._runtime)

    # --- the channel edge ---------------------------------------------------

    async def inject(
        self,
        channel: str,
        chat_id: str,
        sender_id: str,
        text: str,
        *,
        media: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        session_key_override: str | None = None,
    ) -> InboundMessage:
        """Deliver *text* as if it had arrived on runtime channel *channel*.

        The message goes onto the channel bus, which is where a channel runtime
        publishes it, so the runtime's own demux decides which agent sees it.
        Returns the message, whose ``session_key`` is the one the receiving
        agent's store will be keyed by.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id=sender_id,
            chat_id=chat_id,
            content=text,
            media=list(media or []),
            metadata=dict(metadata or {}),
            session_key_override=session_key_override,
        )
        await self._runtime.bus.publish_inbound(msg)
        return msg

    async def next_outbound(
        self,
        timeout: float = DEFAULT_OUTBOUND_TIMEOUT_S,
        *,
        include_events: bool = False,
    ) -> OutboundMessage:
        """Return the next outgoing message, with channel, chat_id and content.

        Reads the queue ``ChannelManager`` dispatches from, so the message is the
        one a real channel would have sent, on the runtime channel it arrived on.

        Event-carrying messages — progress, stream deltas, turn lifecycle — are a
        turn's internal chatter rather than its reply, and would otherwise arrive
        first; they are collected into :attr:`events` and skipped unless
        *include_events* is set.

        Raises:
            TimeoutError: if nothing arrives within *timeout* seconds.  The wait
                is always bounded, including across skipped events.
        """
        expired = TimeoutError(f"no outbound message within {timeout}s")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            # One deadline for the whole call, not one per skipped event.
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise expired
            try:
                msg = await asyncio.wait_for(
                    self._runtime.bus.consume_outbound(), timeout=remaining
                )
            except asyncio.TimeoutError:
                raise expired from None
            if include_events or msg.event is None:
                return msg
            self.events.append(msg)

    def pending_outbound(self) -> int:
        """How many outgoing messages are queued and not yet read."""
        return self._runtime.bus.outbound_size

    # --- lifecycle ----------------------------------------------------------

    async def start(self, *, timeout: float = _READY_TIMEOUT_S) -> None:
        """Start every agent loop, the inbound demux and every outbound pump.

        Returns once the runtime has accepted the start.  Injecting before an
        agent loop has taken its first step is safe either way: the buses queue.
        """
        if self._task is not None:
            raise RuntimeError("gateway harness is already started")
        task = asyncio.create_task(self._runtime.run(), name="nanobot-gateway-harness")
        self._task = task
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self._runtime.is_running:
            if task.done():
                # Surfaces a startup failure here rather than as a mysterious
                # timeout on the first ``next_outbound``.
                await task
                raise RuntimeError("gateway runtime finished before it started")
            if loop.time() >= deadline:
                raise TimeoutError(f"gateway runtime did not start within {timeout}s")
            await asyncio.sleep(_READY_POLL_INTERVAL_S)

    async def aclose(self) -> BaseException | None:
        """Stop and close every agent, leaving no task of ours pending.

        Returns the failure that brought the runtime down, if one did; a caller
        that is not already unwinding an exception should re-raise it.
        """
        self._runtime.flush_sessions()
        await self._runtime.aclose()
        task, self._task = self._task, None
        if task is None:
            return None
        try:
            await asyncio.wait_for(task, timeout=_SHUTDOWN_TIMEOUT_S)
        except asyncio.TimeoutError:
            # Bounded like the gateway's own shutdown: a child that swallows
            # cancellation must not hold the harness open indefinitely.
            logger.warning("Gateway harness: runtime did not stop in time; cancelling")
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the caller instead
            return exc
        return None


def _load_gateway_config(config_path: str | Path | None) -> Config:
    """Load *config_path* the way every other entry point loads it."""
    from nanobot.config.loader import load_config, resolve_config_env_vars

    resolved: Path | None = None
    if config_path is not None:
        resolved = Path(config_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Config not found: {resolved}")
    return resolve_config_env_vars(load_config(resolved), config_path=resolved)


@asynccontextmanager
async def open_gateway(
    config_path: str | Path | None = None,
    *,
    cron_service: CronService | None = None,
    ready_timeout: float = _READY_TIMEOUT_S,
    **loop_kwargs: Any,
) -> AsyncGenerator[GatewayHarness, None]:
    """Run the multi-agent gateway in-process and yield its channel edge.

    Every agent in *config_path* is built exactly as ``nanobot gateway`` builds
    it — its own workspace, sessions, tools, MCP servers and subagents — and the
    same :class:`MultiAgentRuntime` routes between them.  No channel is
    constructed and no network service is contacted, so this runs without a real
    Telegram token configured.

    Args:
        config_path: Path to ``config.json``.  Defaults to the usual location.
        cron_service: Optional scheduler, honoured for the default agent alone,
            as in the gateway.  Omitted by default: scheduled work is out of
            scope for an in-process harness.
        ready_timeout: How long to wait for the runtime to accept the start.
        loop_kwargs: Forwarded to every agent's ``AgentLoop``, e.g. a fake
            provider.  Anything that must not be shared between agents — a
            session manager — is refused by the runtime.

    Exiting the context manager stops every agent loop, closes every MCP provider
    and awaits every task the harness started, so nothing is left pending.
    """
    config = _load_gateway_config(config_path)
    runtime = MultiAgentRuntime.from_config(
        config, cron_service=cron_service, **loop_kwargs
    )
    # The ``message`` tool needs no wiring here: ``MessageTool.create`` binds it
    # to the bus of the agent that owns it, which the outbound pump already
    # carries.  The gateway re-wires it only to mirror proactive sends into that
    # agent's session, which is a WebUI concern rather than a routing one.
    harness = GatewayHarness(runtime)
    try:
        await harness.start(timeout=ready_timeout)
    except BaseException:
        await harness.aclose()
        raise
    try:
        yield harness
    except BaseException:
        # Already unwinding: a runtime failure must not replace the real cause.
        await harness.aclose()
        raise
    else:
        failure = await harness.aclose()
        if failure is not None:
            raise failure

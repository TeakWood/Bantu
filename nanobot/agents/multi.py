"""The channel-facing edge: one channel bus in front of N isolated agents.

The shape this runtime owns::

    ChannelManager -> channel bus -> route(config, channel, chat_id)
                   -> that agent's bus -> its AgentLoop
    each agent's outbound bus -> pumped back onto the channel bus -> ChannelManager

Agent selection deliberately lives here rather than inside ``AgentLoop``.
``.agent/design.md`` is normative that ``agent/loop.py`` and ``agent/runner.py``
are the critical core path; routing inside ``AgentLoop._effective_session_key``
would have been the shorter diff but couples routing to the thing being routed
to.  Nothing under ``nanobot/agent/`` changes for named agents.

Outbound needs no rewriting.  ``OutboundMessage.channel`` is already the runtime
channel name, copied from the inbound message by ``TurnRoute``, and
``ChannelManager`` dispatches on ``msg.channel``, so the pump forwards the
message unchanged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agents.registry import agent_names, route
from nanobot.agents.runtime import AgentRuntime, build_agent_runtime
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import RESERVED_AGENT_NAME, Config

if TYPE_CHECKING:
    from nanobot.cron.service import CronService

__all__ = ["PUMP_POLL_INTERVAL_S", "MultiAgentRuntime"]

# How long a demux/pump task waits on an idle queue before re-checking whether
# the runtime is still running.  This bounds ``stop()``; it does not delay a
# message, since the wait returns as soon as its queue has one.  Matches the
# interval ``AgentLoop.run`` and ``ChannelManager`` already poll on.
PUMP_POLL_INTERVAL_S = 1.0

# Loop keyword arguments that must never be shared between agents: sessions are
# the isolation every other bead in this feature establishes by construction.
# ``bus`` needs no guard — ``from_config`` consumes it as the channel bus, so it
# can never reach ``build_agent_runtime`` as an agent's private bus.
_UNSHAREABLE_LOOP_KWARGS = ("session_manager",)


class MultiAgentRuntime:
    """One channel-facing bus demuxed across N fully-isolated agent runtimes.

    Every agent owns its own :class:`MessageBus` because ``MessageBus.inbound``
    is a single queue that ``consume_inbound`` pops: N ``AgentLoop``s sharing
    one bus would steal each other's messages.  This runtime is what reconnects
    those private buses to the single bus ``ChannelManager`` speaks.
    """

    def __init__(
        self,
        config: Config,
        agents: Mapping[str, AgentRuntime],
        *,
        bus: MessageBus | None = None,
    ) -> None:
        if RESERVED_AGENT_NAME not in agents:
            raise KeyError(
                f"a multi-agent runtime needs a '{RESERVED_AGENT_NAME}' agent; "
                f"got {sorted(agents)}"
            )
        self.config = config
        self.bus = bus if bus is not None else MessageBus()
        self._agents: dict[str, AgentRuntime] = dict(agents)
        self._routes: dict[str, str] = {}
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        # Set by every ``stop`` and cleared only by a ``run`` that has finished.
        # ``_running`` alone cannot carry a shutdown across the gap between
        # scheduling ``run`` and its first step, because ``run`` sets
        # ``_running = True`` itself and would erase the flag; this one is
        # sticky, so a stop requested before ``run`` ever starts still lands.
        self._stop_pending = False

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        bus: MessageBus | None = None,
        cron_service: CronService | None = None,
        **loop_kwargs: Any,
    ) -> MultiAgentRuntime:
        """Build one isolated :class:`AgentRuntime` per configured agent.

        ``cron_service`` is forwarded to every agent but honoured for ``default``
        alone — :func:`build_agent_runtime` zeroes it for named agents, so a
        composition root can pass its single gateway service uniformly.

        A config with no ``agents.named`` yields exactly one agent, built exactly
        as the single-agent composition builds it today.
        """
        for key in _UNSHAREABLE_LOOP_KWARGS:
            if key in loop_kwargs:
                raise TypeError(
                    f"'{key}' cannot be shared across agents; "
                    "each agent builds its own"
                )
        agents = {
            name: build_agent_runtime(
                config, name, cron_service=cron_service, **loop_kwargs
            )
            for name in agent_names(config)
        }
        return cls(config, agents, bus=bus)

    # --- the agents ---------------------------------------------------------

    @property
    def agents(self) -> Mapping[str, AgentRuntime]:
        """Every agent, ``default`` first, keyed by name."""
        return self._agents

    @property
    def names(self) -> tuple[str, ...]:
        """Every configured agent name, ``default`` first."""
        return tuple(self._agents)

    @property
    def default(self) -> AgentRuntime:
        """The agent every unbound channel routes to."""
        return self._agents[RESERVED_AGENT_NAME]

    @property
    def is_running(self) -> bool:
        """Whether the demux, the pumps and the agent loops are live."""
        return self._running

    def get(self, name: str) -> AgentRuntime | None:
        """Return the agent called *name*, or ``None`` if it is not configured."""
        return self._agents.get(name)

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self) -> Iterator[AgentRuntime]:
        return iter(self._agents.values())

    def __contains__(self, name: object) -> bool:
        return name in self._agents

    # --- routing ------------------------------------------------------------

    def invalidate_routing(self) -> None:
        """Drop the memoised routing decisions after the config file changes."""
        self._routes.clear()

    def agent_for(self, channel: str, chat_id: str | None = None) -> AgentRuntime:
        """Return the agent an inbound message on *channel* is delivered to.

        The decision itself is :func:`nanobot.agents.route`, a pure function of
        config; it is memoised per runtime channel name because this sits on the
        per-message path while ``route`` re-reads the channel section each call.

        A channel bound to an agent ``agents.named`` never declares falls back to
        ``default`` with a warning: detecting that mismatch is out of scope for
        this feature, and dropping the traffic would be worse than answering it.
        """
        name = self._routes.get(channel)
        if name is None:
            name = route(self.config, channel, chat_id)
            if name not in self._agents:
                logger.warning(
                    "Channel '{}' is bound to unconfigured agent '{}'; "
                    "routing it to '{}'",
                    channel,
                    name,
                    RESERVED_AGENT_NAME,
                )
                name = RESERVED_AGENT_NAME
            self._routes[channel] = name
        return self._agents[name]

    # --- the demux and the pumps -------------------------------------------

    async def deliver_inbound(self, msg: InboundMessage) -> AgentRuntime:
        """Hand *msg* to its agent's own bus, and to no other agent's."""
        agent = self.agent_for(msg.channel, msg.chat_id)
        await agent.bus.publish_inbound(msg)
        return agent

    async def deliver_outbound(self, msg: OutboundMessage) -> None:
        """Forward one agent's reply onto the channel bus, unchanged.

        ``msg.channel`` is already the runtime channel name the message arrived
        on, so ``ChannelManager`` dispatches it exactly as it dispatches a
        single-agent reply.
        """
        await self.bus.publish_outbound(msg)

    async def _demux_inbound(self) -> None:
        """Pop the channel bus and push each message onto its agent's bus."""
        logger.info("Agent inbound demux started for {} agent(s)", len(self._agents))
        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(), timeout=PUMP_POLL_INTERVAL_S
                )
            except asyncio.TimeoutError:
                continue
            try:
                await self.deliver_inbound(msg)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Failed to route inbound message on channel '{}'", msg.channel
                )
        logger.info("Agent inbound demux stopped")

    async def _pump_outbound(self, agent: AgentRuntime) -> None:
        """Pop one agent's outbound bus and republish onto the channel bus."""
        while self._running:
            try:
                msg = await asyncio.wait_for(
                    agent.bus.consume_outbound(), timeout=PUMP_POLL_INTERVAL_S
                )
            except asyncio.TimeoutError:
                continue
            try:
                await self.deliver_outbound(msg)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Failed to forward outbound message from agent '{}'", agent.name
                )

    # --- lifecycle ----------------------------------------------------------

    async def connect_mcp(self) -> None:
        """Connect every agent's MCP servers concurrently, degrading per agent.

        ``MCPProvider.connect`` already absorbs a server that cannot launch, so a
        failure here is logged rather than allowed to abort another agent's
        startup.
        """
        results = await asyncio.gather(
            *(agent.connect_mcp() for agent in self._agents.values()),
            return_exceptions=True,
        )
        for name, result in zip(self._agents, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("Agent '{}': MCP startup incomplete: {}", name, result)

    async def _run_agent(self, agent: AgentRuntime) -> None:
        """Run one agent's loop, unless shutdown arrived before this task ran.

        ``asyncio.create_task`` only schedules a coroutine; its body runs on a
        later tick.  A ``stop`` landing in that gap would otherwise be lost,
        because ``AgentLoop.run`` sets its own running flag on entry and so is
        not stopped by a ``stop`` that preceded it.  This guard is the first
        statement of the task body and nothing awaits between it and
        ``agent.run()``, whose own first statement is that flag — so the loop
        cannot start after a shutdown was requested.
        """
        if not self._running or self._stop_pending:
            return
        await agent.run()

    async def run(self) -> None:
        """Run every agent loop, the inbound demux and every outbound pump.

        Returns when :meth:`stop` is called.  A genuine failure in any child is
        re-raised; cancellation caused by our own shutdown is not.
        """
        if self._running:
            raise RuntimeError("MultiAgentRuntime is already running")
        self._running = True
        try:
            await self.connect_mcp()
            # Two windows close here.  MCP startup is the only await between
            # accepting ``run`` and owning tasks, and for stdio servers it lasts
            # seconds — the likeliest moment for an operator Ctrl-C.  And a
            # ``stop`` that arrived before this coroutine's first step was
            # erased from ``_running`` by the assignment above, so only the
            # sticky ``_stop_pending`` still carries it.  Either must not be
            # lost: ``AgentLoop.run`` re-sets its own running flag on entry, so
            # starting the loops here would resurrect every agent we were just
            # asked to shut down — against already closed MCP providers, in the
            # ``aclose`` case.  The gap after ``create_task`` is closed by
            # ``_run_agent``; this check alone does not reach it.
            if not self._running or self._stop_pending:
                logger.info(
                    "MultiAgentRuntime was stopped during startup; "
                    "no agent loop started"
                )
                return
            self._tasks = [
                asyncio.create_task(self._demux_inbound(), name="nanobot-agent-demux"),
                *(
                    asyncio.create_task(
                        self._pump_outbound(agent),
                        name=f"nanobot-agent-outbound-{agent.name}",
                    )
                    for agent in self._agents.values()
                ),
                *(
                    asyncio.create_task(
                        self._run_agent(agent), name=f"nanobot-agent-loop-{agent.name}"
                    )
                    for agent in self._agents.values()
                ),
            ]
            # FIRST_EXCEPTION, not gather(return_exceptions=True): a crashed
            # agent loop must bring the runtime down now rather than leave the
            # demux feeding an agent that is no longer consuming.  With no
            # failure this waits for all of them, which is what ``stop`` ends.
            done, _pending = await asyncio.wait(
                self._tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                if task.cancelled():
                    # Our own shutdown; not a failure to report.
                    continue
                failure = task.exception()
                if failure is not None:
                    raise failure
        finally:
            self._running = False
            # Only a ``run`` that has finished clears the sticky flag, so a
            # stopped runtime can be started again without inheriting it.
            self._stop_pending = False
            await self._cancel_tasks()

    def stop(self) -> None:
        """Ask every agent loop, the demux and every pump to finish.

        Flag-based like :meth:`AgentLoop.stop`, so an in-flight turn is left to
        unwind rather than cancelled mid-tool-call.

        Safe to call before :meth:`run` has taken its first step and while it is
        still connecting MCP servers: the request is sticky, so ``run`` observes
        it and starts no loop at all.
        """
        self._running = False
        self._stop_pending = True
        for agent in self._agents.values():
            agent.stop()

    async def _cancel_tasks(self) -> None:
        """Cancel and await whatever this runtime started.  Idempotent."""
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def drain(self) -> None:
        """Finish scheduled local event dispatch on every bus this runtime owns."""
        await self.bus.drain()
        for agent in self._agents.values():
            await agent.bus.drain()

    def flush_sessions(self) -> int:
        """Flush every agent's cached sessions to disk; return how many were written."""
        return sum(agent.sessions.flush_all() for agent in self._agents.values())

    async def aclose(self) -> None:
        """Stop everything, then close every agent loop and MCP provider.

        Ordered like the single-agent gateway shutdown: runtime tasks first, then
        each agent's loop-owned resources and its MCP provider, then the buses.
        One agent failing to close never skips another's cleanup.
        """
        self.stop()
        await self._cancel_tasks()
        results = await asyncio.gather(
            *(agent.aclose() for agent in self._agents.values()),
            return_exceptions=True,
        )
        for name, result in zip(self._agents, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("Agent '{}': cleanup incomplete: {}", name, result)
        await self.drain()

"""Multi-agent orchestrator — routes inbound messages to per-agent loops."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.config.schema import Config
    from nanobot.cron.service import CronService
    from nanobot.heartbeat.service import HeartbeatService
    from nanobot.providers.base import LLMProvider


class AgentOrchestrator:
    """Manages one :class:`~nanobot.agent.loop.AgentLoop` per active agent.

    ``AgentOrchestrator`` owns the shared :class:`~nanobot.bus.queue.MessageBus`
    outbound queue (used by all loops for fan-out) and creates one
    :class:`asyncio.Queue` per agent for isolated inbound delivery.

    A dedicated ``_dispatch_loop`` coroutine reads from ``bus.inbound`` and
    routes each :class:`~nanobot.bus.events.InboundMessage` to the correct
    per-agent queue by matching ``msg.agent_id``.  Unknown or ``None``
    agent IDs fall back to the default agent's queue.

    :class:`~nanobot.cron.service.CronService` and
    :class:`~nanobot.heartbeat.service.HeartbeatService` are exclusively
    owned by the default agent.  Attempting to pass either service to a
    specialized loop raises :exc:`ValueError`.
    """

    DEFAULT_AGENT_ID = "default"

    def __init__(
        self,
        bus: MessageBus,
        config: Config,
        agents_dir: Path,
    ) -> None:
        self._bus = bus
        self._config = config
        self._agents_dir = agents_dir

        self._loops: dict[str, AgentLoop] = {}
        self._queues: dict[str, asyncio.Queue[InboundMessage]] = {}
        self._loop_tasks: dict[str, asyncio.Task] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._running = False

        self._cron: CronService | None = None
        self._heartbeat: HeartbeatService | None = None

    # ------------------------------------------------------------------
    # Provider construction
    # ------------------------------------------------------------------

    def _make_provider(self) -> LLMProvider:
        """Create the LLM provider from config (falls back to NullProvider)."""
        from nanobot.providers.custom_provider import CustomProvider
        from nanobot.providers.litellm_provider import LiteLLMProvider
        from nanobot.providers.null_provider import NullProvider
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider

        model = self._config.agents.defaults.model
        provider_name = self._config.get_provider_name(model)
        p = self._config.get_provider(model)

        if provider_name == "openai_codex" or model.startswith("openai-codex/"):
            return OpenAICodexProvider(default_model=model)

        if provider_name == "custom":
            return CustomProvider(
                api_key=p.api_key if p else "no-key",
                api_base=self._config.get_api_base(model) or "http://localhost:8000/v1",
                default_model=model,
            )

        from nanobot.providers.registry import find_by_name

        spec = find_by_name(provider_name)
        if (
            not model.startswith("bedrock/")
            and not (p and p.api_key)
            and not (spec and spec.is_oauth)
        ):
            logger.warning(
                "AgentOrchestrator: no API key configured; using NullProvider"
            )
            return NullProvider()

        return LiteLLMProvider(
            api_key=p.api_key if p else None,
            api_base=self._config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            provider_name=provider_name,
        )

    # ------------------------------------------------------------------
    # Loop factory
    # ------------------------------------------------------------------

    def _build_loop(
        self,
        agent_id: str,
        workspace: Path,
        inbound_queue: asyncio.Queue[InboundMessage],
        provider: LLMProvider,
        *,
        cron_service: CronService | None = None,
        agent_assets_dir: Path | None = None,
    ) -> AgentLoop:
        """Instantiate an :class:`~nanobot.agent.loop.AgentLoop` for one agent.

        Raises :exc:`ValueError` if a *cron_service* is passed for a
        specialized agent (CronService is restricted to the default agent).
        """
        if agent_id != self.DEFAULT_AGENT_ID and cron_service is not None:
            raise ValueError(
                f"Specialized agent '{agent_id}' must not receive a CronService. "
                "CronService is exclusive to the default agent."
            )

        from nanobot.agent.loop import AgentLoop
        from nanobot.config.schema import resolve_agent_config
        from nanobot.session.manager import SessionManager

        cfg = resolve_agent_config(agent_id, self._config.agents)
        session_manager = SessionManager(workspace)

        return AgentLoop(
            bus=self._bus,
            provider=provider,
            workspace=workspace,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_iterations=cfg.max_tool_iterations,
            memory_window=cfg.memory_window,
            reasoning_effort=cfg.reasoning_effort,
            brave_api_key=self._config.tools.web.search.api_key or None,
            web_proxy=self._config.tools.web.proxy or None,
            exec_config=self._config.tools.exec,
            cron_service=cron_service,
            restrict_to_workspace=self._config.tools.restrict_to_workspace,
            session_manager=session_manager,
            mcp_servers=self._config.tools.mcp_servers,
            channels_config=self._config.channels,
            inbound_queue=inbound_queue,
            agent_assets_dir=agent_assets_dir,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Instantiate and start all agent loops, services, and the dispatch loop.

        For each discovered agent a dedicated
        :class:`asyncio.Queue[InboundMessage]` is created.  The default agent
        additionally receives :class:`~nanobot.cron.service.CronService` and
        :class:`~nanobot.heartbeat.service.HeartbeatService` ownership.
        """
        from nanobot.agent.registry import AgentRegistry
        from nanobot.config.loader import get_data_dir
        from nanobot.cron.service import CronService
        from nanobot.cron.types import CronJob
        from nanobot.heartbeat.service import HeartbeatService
        from nanobot.utils.helpers import get_agent_workspace

        registry = AgentRegistry(self._agents_dir)
        discovered = registry.discover()

        # Ordered list: default first, then specialised agents
        agent_ids = [self.DEFAULT_AGENT_ID] + [m.name for m in discovered]
        meta_by_name = {m.name: m for m in discovered}

        # Per-agent inbound queues
        for agent_id in agent_ids:
            self._queues[agent_id] = asyncio.Queue()

        # Shared provider (all loops share one LLM provider instance)
        provider = self._make_provider()

        # ------------------------------------------------------------------
        # Default agent: gets CronService
        # ------------------------------------------------------------------
        cron_store_path = get_data_dir() / "cron" / "jobs.json"
        self._cron = CronService(cron_store_path)

        default_ws = self._config.workspace_path
        default_loop = self._build_loop(
            self.DEFAULT_AGENT_ID,
            default_ws,
            self._queues[self.DEFAULT_AGENT_ID],
            provider,
            cron_service=self._cron,
        )
        self._loops[self.DEFAULT_AGENT_ID] = default_loop

        # Wire CronService → default loop
        async def on_cron_job(job: CronJob) -> str | None:
            from nanobot.agent.tools.cron import CronTool
            from nanobot.agent.tools.message import MessageTool

            reminder_note = (
                "[Scheduled Task] Timer finished.\n\n"
                f"Task '{job.name}' has been triggered.\n"
                f"Scheduled instruction: {job.payload.message}"
            )
            cron_tool = default_loop.tools.get("cron")
            cron_token = None
            if isinstance(cron_tool, CronTool):
                cron_token = cron_tool.set_cron_context(True)
            try:
                response = await default_loop.process_direct(
                    reminder_note,
                    session_key=f"cron:{job.id}",
                    channel=job.payload.channel or "cli",
                    chat_id=job.payload.to or "direct",
                )
            finally:
                if isinstance(cron_tool, CronTool) and cron_token is not None:
                    cron_tool.reset_cron_context(cron_token)

            message_tool = default_loop.tools.get("message")
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return response

            if job.payload.deliver and job.payload.to and response:
                await self._bus.publish_outbound(
                    OutboundMessage(
                        channel=job.payload.channel or "cli",
                        chat_id=job.payload.to,
                        content=response,
                    )
                )
            return response

        self._cron.on_job = on_cron_job

        # ------------------------------------------------------------------
        # Specialised agent loops (no CronService / HeartbeatService)
        # ------------------------------------------------------------------
        for agent_id in agent_ids[1:]:
            meta = meta_by_name[agent_id]
            agent_ws = get_agent_workspace(agent_id)
            loop = self._build_loop(
                agent_id,
                agent_ws,
                self._queues[agent_id],
                provider,
                cron_service=None,
                agent_assets_dir=meta.path,
            )
            self._loops[agent_id] = loop

        # ------------------------------------------------------------------
        # HeartbeatService → default loop
        # ------------------------------------------------------------------
        async def on_heartbeat_execute(tasks: str) -> str:
            async def _silent(*_a: object, **_kw: object) -> None:
                pass

            return await default_loop.process_direct(
                tasks,
                session_key="heartbeat",
                channel="cli",
                chat_id="direct",
                on_progress=_silent,
            )

        async def on_heartbeat_notify(response: str) -> None:
            # Deliver to first known external session found in default loop's history
            for item in default_loop.sessions.list_sessions():
                key = item.get("key", "")
                if ":" not in key:
                    continue
                channel, chat_id = key.split(":", 1)
                if channel not in {"cli", "system"} and chat_id:
                    await self._bus.publish_outbound(
                        OutboundMessage(
                            channel=channel,
                            chat_id=chat_id,
                            content=response,
                        )
                    )
                    return

        hb_cfg = self._config.gateway.heartbeat
        self._heartbeat = HeartbeatService(
            workspace=default_ws,
            provider=provider,
            model=default_loop.model,
            on_execute=on_heartbeat_execute,
            on_notify=on_heartbeat_notify,
            interval_s=hb_cfg.interval_s,
            enabled=hb_cfg.enabled,
        )

        # ------------------------------------------------------------------
        # Start services
        # ------------------------------------------------------------------
        await self._cron.start()
        await self._heartbeat.start()

        # Start dispatch loop
        self._running = True
        self._dispatch_task = asyncio.create_task(
            self._dispatch_loop(), name="orchestrator-dispatch"
        )

        # Start one task per agent loop
        for agent_id, loop in self._loops.items():
            task = asyncio.create_task(loop.run(), name=f"agent-loop-{agent_id}")
            self._loop_tasks[agent_id] = task

        logger.info(
            "AgentOrchestrator started with {} agent loop(s): {}",
            len(self._loops),
            list(self._loops),
        )

    async def run(self) -> None:
        """Start the orchestrator and block until all agent loops complete.

        This is the top-level blocking entry point that replaces a bare
        :meth:`AgentLoop.run() <nanobot.agent.loop.AgentLoop.run>` call.
        Call :meth:`stop` to initiate a clean shutdown.
        """
        await self.start()
        try:
            if self._loop_tasks:
                await asyncio.gather(
                    *self._loop_tasks.values(), return_exceptions=True
                )
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Cleanly shut down the dispatch loop and all agent loops in parallel."""
        self._running = False

        # Cancel the dispatch coroutine
        if self._dispatch_task and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        # Signal every loop to exit its run() loop
        for loop in self._loops.values():
            loop.stop()

        # Cancel loop tasks as a safety net (handles loops that do not exit naturally)
        for task in self._loop_tasks.values():
            if not task.done():
                task.cancel()

        # Wait for all loop tasks to finish
        if self._loop_tasks:
            await asyncio.gather(*self._loop_tasks.values(), return_exceptions=True)

        # Close MCP connections
        close_fns = [loop.close_mcp() for loop in self._loops.values()]
        if close_fns:
            await asyncio.gather(*close_fns, return_exceptions=True)

        # Stop background services
        if self._heartbeat is not None:
            self._heartbeat.stop()
        if self._cron is not None:
            self._cron.stop()

        logger.info("AgentOrchestrator stopped")

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        """Read from ``bus.inbound`` and route each message to the correct agent queue."""
        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self._bus.consume_inbound(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            await self.route(msg)

    async def route(self, msg: InboundMessage) -> None:
        """Put *msg* into the correct per-agent inbound queue.

        The target queue is determined by ``msg.agent_id``; an unknown or
        ``None`` agent ID falls back to the default agent's queue.
        """
        queue: asyncio.Queue[InboundMessage] | None = None
        if msg.agent_id:
            queue = self._queues.get(msg.agent_id)
        if queue is None:
            queue = self._queues.get(self.DEFAULT_AGENT_ID)
        if queue is not None:
            await queue.put(msg)
        else:
            logger.warning(
                "AgentOrchestrator: no queue for agent_id={!r}; message dropped",
                msg.agent_id,
            )

"""One fully-isolated agent: its own bus, tools, MCP servers, sessions and loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentRunHookContext
from nanobot.agent.hooks import create_file_edit_activity_hook
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agents.resolution import ResolvedAgentConfig, resolve_agent_config
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import RESERVED_AGENT_NAME, Config
from nanobot.providers.image_generation import image_gen_provider_configs
from nanobot.session.manager import SessionManager
from nanobot.utils.helpers import sync_workspace_templates

if TYPE_CHECKING:
    from nanobot.cron.service import CronService

__all__ = [
    "AgentRuntime",
    "agent_session_manager",
    "bootstrap_agent_workspace",
    "build_agent_runtime",
]


class MCPReadinessHook(AgentHook):
    """Retry this agent's own MCP connections before the runner reads tools."""

    def __init__(self, provider: MCPProvider) -> None:
        super().__init__()
        self._provider = provider

    async def before_run(self, context: AgentRunHookContext) -> None:
        await self._provider.connect()


def bootstrap_agent_workspace(workspace: Path) -> Path:
    """Create *workspace* and seed it with the bundled templates.

    Every agent gets the treatment the default agent's workspace already gets at
    gateway startup, so a named agent owns its own SOUL.md, USER.md and memory/
    rather than reading another agent's.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    sync_workspace_templates(workspace, silent=True)
    return workspace


def agent_session_manager(agent_config: Config) -> SessionManager:
    """Build the session store one agent owns, keyed by its own workspace.

    Sessions live under the shared runtime data directory rather than inside the
    workspace, in a subdirectory :class:`SessionManager` derives from the
    workspace id — so two agents cannot collide even though the root is shared.

    Exposed separately from :func:`build_agent_runtime` because a composition
    root often needs the store before the loop exists: a turn delivery factory
    and a recovery coordinator both take it and are themselves loop arguments.
    """
    data_dir = agent_config.runtime_data_dir
    if data_dir is None:
        # Keep the call byte-identical to a plain single-agent composition, which
        # lets SessionManager apply its own runtime-subdirectory default.
        return SessionManager(agent_config.workspace_path)
    return SessionManager(agent_config.workspace_path, sessions_root=data_dir / "sessions")


@dataclass(frozen=True)
class AgentRuntime:
    """Everything one agent owns, shared with no other agent.

    The bus is per-agent because ``MessageBus.inbound`` is a single queue and
    ``consume_inbound`` pops it: two ``AgentLoop``s sharing one bus would steal
    each other's messages.  It also makes subagent result routing correct for
    free, since ``SubagentManager`` republishes onto the spawning agent's bus.
    """

    name: str
    resolved: ResolvedAgentConfig
    bus: MessageBus
    tools: ToolRegistry
    mcp_provider: MCPProvider
    sessions: SessionManager
    loop: AgentLoop

    @property
    def is_default(self) -> bool:
        """Whether this is the default agent, which owns the top-level blocks."""
        return self.name == RESERVED_AGENT_NAME

    @property
    def config(self) -> Config:
        """This agent's effective settings as a standalone config."""
        return self.resolved.config

    @property
    def workspace(self) -> Path:
        """The workspace this agent owns — never another agent's."""
        return self.loop.workspace

    async def connect_mcp(self) -> None:
        """Connect this agent's MCP servers, degrading to built-in tools."""
        await self.mcp_provider.connect()

    async def run(self) -> None:
        """Run this agent's loop until it is stopped."""
        await self.loop.run()

    def stop(self) -> None:
        """Ask this agent's loop to stop consuming its bus."""
        self.loop.stop()

    async def aclose(self) -> None:
        """Release this agent's loop and MCP connections."""
        try:
            await self.loop.aclose()
        finally:
            await self.mcp_provider.aclose()


def build_agent_runtime(
    config: Config,
    name: str = RESERVED_AGENT_NAME,
    *,
    cron_service: CronService | None = None,
    bus: MessageBus | None = None,
    **loop_kwargs: Any,
) -> AgentRuntime:
    """Build one fully-isolated agent from its resolved config.

    Nothing built here is shared with another agent: the bus, tool registry, MCP
    provider, session manager and workspace all belong to *name* alone.

    ``cron_service`` is honoured for the default agent only.  Scheduled work is
    out of scope for named agents, and ``CronTool`` gates purely on
    ``ctx.cron_service is not None``, so passing ``None`` is the enforcement —
    the tool is never constructed for them.

    Extra keyword arguments are forwarded to :meth:`AgentLoop.from_config`, so a
    composition root can still supply a provider snapshot, delivery factory or
    extra hooks.

    Raises:
        KeyError: if no agent is configured under that name.
    """
    resolved = resolve_agent_config(config, name)
    agent_config = resolved.config
    bootstrap_agent_workspace(agent_config.workspace_path)

    agent_bus = bus if bus is not None else MessageBus()
    tools = ToolRegistry()
    mcp_provider = MCPProvider.from_config(agent_config, tools)

    if not resolved.is_default and cron_service is not None:
        logger.debug(
            "Agent '{}' is not the default agent; scheduled work is not available to it",
            name,
        )
    effective_cron = cron_service if resolved.is_default else None

    sessions = loop_kwargs.pop("session_manager", None)
    if sessions is None:
        sessions = agent_session_manager(agent_config)

    hooks = [MCPReadinessHook(mcp_provider), *(loop_kwargs.pop("hooks", None) or [])]
    hook_factories = loop_kwargs.pop("hook_factories", None) or [
        create_file_edit_activity_hook
    ]
    loop_kwargs.setdefault(
        "image_generation_provider_configs",
        image_gen_provider_configs(agent_config),
    )

    loop = AgentLoop.from_config(
        agent_config,
        agent_bus,
        cron_service=effective_cron,
        session_manager=sessions,
        hooks=hooks,
        hook_factories=hook_factories,
        tool_registry=tools,
        **loop_kwargs,
    )
    return AgentRuntime(
        name=name,
        resolved=resolved,
        bus=agent_bus,
        tools=tools,
        mcp_provider=mcp_provider,
        sessions=sessions,
        loop=loop,
    )

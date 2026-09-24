"""High-level programmatic interface to nanobot."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from nanobot.agent.hook import AgentHook, SDKCaptureHook
from nanobot.agent.hooks import create_file_edit_activity_hook
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agents.resolution import named_agent_entry
from nanobot.agents.runtime import build_agent_runtime
from nanobot.config.schema import RESERVED_AGENT_NAME, AgentDefaults, Config, NamedAgentConfig
from nanobot.providers.base import LLMUsage
from nanobot.providers.image_generation import image_gen_provider_configs
from nanobot.sdk.clients import MemoryClient, RuntimeClient, SessionClient
from nanobot.sdk.runtime import (
    build_process_direct_kwargs,
    ensure_single_model_selector,
)
from nanobot.sdk.streaming import RunStream, SDKStreamEmitter, SDKStreamingHook
from nanobot.sdk.types import (
    STREAM_EVENT_REASONING_COMPLETED,
    STREAM_EVENT_REASONING_DELTA,
    STREAM_EVENT_RUN_COMPLETED,
    STREAM_EVENT_RUN_FAILED,
    STREAM_EVENT_RUN_STARTED,
    STREAM_EVENT_TEXT_COMPLETED,
    STREAM_EVENT_TEXT_DELTA,
    STREAM_EVENT_TOOL_COMPLETED,
    STREAM_EVENT_TOOL_FAILED,
    STREAM_EVENT_TOOL_STARTED,
    STREAM_EVENT_TYPES,
    RunResult,
    SessionInfo,
    SessionSnapshot,
    StreamEvent,
    StreamEventType,
    result_from_response,
)
from nanobot.utils.llm_runtime import LLMRuntime

__all__ = [
    "Nanobot",
    "LLMUsage",
    "RunResult",
    "RunStream",
    "SessionInfo",
    "SessionSnapshot",
    "STREAM_EVENT_REASONING_COMPLETED",
    "STREAM_EVENT_REASONING_DELTA",
    "STREAM_EVENT_RUN_COMPLETED",
    "STREAM_EVENT_RUN_FAILED",
    "STREAM_EVENT_RUN_STARTED",
    "STREAM_EVENT_TEXT_COMPLETED",
    "STREAM_EVENT_TEXT_DELTA",
    "STREAM_EVENT_TOOL_COMPLETED",
    "STREAM_EVENT_TOOL_FAILED",
    "STREAM_EVENT_TOOL_STARTED",
    "STREAM_EVENT_TYPES",
    "StreamEvent",
    "StreamEventType",
]


def _override_target(config: Config, agent: str) -> AgentDefaults | NamedAgentConfig:
    """Return the config block the agent called *agent* resolves its settings from."""
    if agent == RESERVED_AGENT_NAME:
        return config.agents.defaults
    return named_agent_entry(config, agent)


def _apply_agent_overrides(
    target: AgentDefaults | NamedAgentConfig,
    *,
    workspace: str | Path | None,
    model: str | None,
    model_preset: str | None,
) -> None:
    """Write the caller's overrides onto one agent's config block."""
    if workspace is not None:
        target.workspace = str(Path(workspace).expanduser().resolve())
    if model is not None:
        target.model_preset = None
        target.model = model
        target.provider = "auto"
    elif model_preset is not None:
        target.model_preset = model_preset


class Nanobot:
    """Programmatic facade for running the nanobot agent.

    Usage::

        bot = Nanobot.from_config()
        result = await bot.run("Summarize this repo", hooks=[MyHook()])
        print(result.content)
    """

    def __init__(
        self,
        loop: AgentLoop,
        *,
        config: Config | None = None,
        mcp_provider: MCPProvider | None = None,
        agent_name: str = RESERVED_AGENT_NAME,
    ) -> None:
        self._loop = loop
        self._config = config
        self._mcp_provider = mcp_provider
        self._agent_name = agent_name
        self.sessions = SessionClient(loop)
        self.memory = MemoryClient(loop)
        self.runtime = RuntimeClient(loop)

    @property
    def agent_name(self) -> str:
        """The name of the agent this instance drives (``default`` unless named)."""
        return self._agent_name

    @property
    def workspace(self) -> Path:
        """The workspace this agent owns — never another agent's."""
        return self._loop.workspace

    async def tool_names(self) -> list[str]:
        """Return the names of the tools this agent's model is offered.

        MCP tools are registered into the shared registry when the provider
        connects rather than at discovery time, so the provider is connected
        first — the same lazy connect :meth:`run` performs.
        """
        if self._mcp_provider is not None:
            await self._mcp_provider.connect()
        return list(self._loop.tool_names)

    async def subagent_tool_names(self) -> list[str]:
        """Return the names of the tools a background subagent is offered.

        This is the registry ``SubagentManager`` builds for a spawned subagent,
        read without running one.  MCP tools are absent by construction: the
        subagent registry is loaded with the ``subagent`` scope and MCP tools
        are only ever registered into the agent's own shared registry.
        """
        return list(self._loop.subagents.build_tool_registry().tool_names)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        workspace: str | Path | None = None,
        model: str | None = None,
        model_preset: str | None = None,
        agent: str = RESERVED_AGENT_NAME,
    ) -> Nanobot:
        """Create a Nanobot instance from a config file.

        Args:
            config_path: Path to ``config.json``.  Defaults to
                ``~/.nanobot/config.json``.
            workspace: Override the workspace directory from config.
            model: Override the instance default model.
            model_preset: Override the instance default model preset.
            agent: Which configured agent to build.  ``default`` — the value
                used when it is omitted — builds the agent that owns the
                top-level blocks, exactly as before.  Any other name must
                appear under ``agents.named``.

        Raises:
            KeyError: if *agent* names an agent that is not configured.
        """
        from nanobot.config.loader import load_config, resolve_config_env_vars

        ensure_single_model_selector(model=model, model_preset=model_preset)
        resolved: Path | None = None
        if config_path is not None:
            resolved = Path(config_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Config not found: {resolved}")

        config: Config = resolve_config_env_vars(
            load_config(resolved),
            config_path=resolved,
        )
        # Overrides are written onto the block the requested agent resolves
        # from, so `--workspace`/`--model` mean the same thing for a named
        # agent as they do for default rather than being silently discarded by
        # the overlay (a named agent never inherits `workspace`).
        _apply_agent_overrides(
            _override_target(config, agent),
            workspace=workspace,
            model=model,
            model_preset=model_preset,
        )

        if agent != RESERVED_AGENT_NAME:
            runtime = build_agent_runtime(config, agent)
            return cls(
                runtime.loop,
                config=runtime.config,
                mcp_provider=runtime.mcp_provider,
                agent_name=agent,
            )

        tools = ToolRegistry()
        mcp_provider = MCPProvider.from_config(config, tools)
        loop = AgentLoop.from_config(
            config,
            image_generation_provider_configs=image_gen_provider_configs(config),
            hook_factories=[create_file_edit_activity_hook],
            tool_registry=tools,
        )
        return cls(loop, config=config, mcp_provider=mcp_provider)

    async def run(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        ephemeral: bool = False,
        attributes: Mapping[str, Any] | None = None,
        hooks: list[AgentHook] | None = None,
        model: str | None = None,
        model_preset: str | None = None,
    ) -> RunResult:
        """Run the agent once and return the result.

        Args:
            message: The user message to process.
            session_key: Session identifier for conversation isolation.
                Different keys get independent history.
            channel: Logical channel label for runtime context.
            chat_id: Logical chat identifier for runtime context.
            sender_id: Logical sender identifier for runtime context.
            media: Optional local media paths attached to the message.
            ephemeral: If true, do not persist the turn or compact session history.
            attributes: Optional caller-owned request data exposed to context
                providers and turn-hook factories. Attributes are kept separate
                from nanobot's trusted internal message metadata.
            hooks: Optional lifecycle hooks for this run.
            model: Override the model for this run only.
            model_preset: Override the model preset for this run only.
        """
        capture = SDKCaptureHook()
        per_run_hooks = [capture, *(hooks or [])]
        runtime = self._loop.runtime_resolver.resolve_override(
            model=model,
            model_preset=model_preset,
            config=self._config,
        )
        kwargs = build_process_direct_kwargs(
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            sender_id=sender_id,
            media=media,
            ephemeral=ephemeral,
            attributes=attributes,
        )
        if runtime is not None:
            kwargs["runtime"] = runtime
        if self._mcp_provider is not None:
            await self._mcp_provider.connect()
        response = await self._loop.process_direct(
            message,
            **kwargs,
            hooks=per_run_hooks,
        )

        return result_from_response(response, capture)

    async def run_streamed(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        ephemeral: bool = False,
        attributes: Mapping[str, Any] | None = None,
        hooks: list[AgentHook] | None = None,
        model: str | None = None,
        model_preset: str | None = None,
    ) -> RunStream:
        """Start a streamed run and return a handle for events and final result."""
        override_runtime = self._loop.runtime_resolver.resolve_override(
            model=model,
            model_preset=model_preset,
            config=self._config,
        )
        queue: asyncio.Queue[StreamEvent | object] = asyncio.Queue(maxsize=256)
        emitter = SDKStreamEmitter(queue)
        stream_hook = SDKStreamingHook(emitter)
        capture = SDKCaptureHook()
        per_run_hooks = [capture, stream_hook, *(hooks or [])]
        run_started = False

        async def _emit_run_started(runtime: LLMRuntime | None = None) -> None:
            nonlocal run_started
            if run_started:
                return
            if runtime is None:
                runtime = override_runtime
            metadata: dict[str, Any] = {
                "session_key": session_key,
                "channel": channel,
                "chat_id": chat_id,
                "sender_id": sender_id,
            }
            if runtime is not None:
                metadata.update({
                    "model": runtime.model,
                    "model_preset": runtime.model_preset,
                })
            await emitter.emit(StreamEvent(
                type=STREAM_EVENT_RUN_STARTED,
                metadata=metadata,
            ))
            run_started = True

        async def _on_stream(delta: str) -> None:
            await emitter.text_delta(delta)

        async def _on_stream_end(*_args: Any, resuming: bool = False, **_kwargs: Any) -> None:
            await emitter.text_completed(resuming=resuming)

        async def _run() -> RunResult:
            kwargs = build_process_direct_kwargs(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                sender_id=sender_id,
                media=media,
                ephemeral=ephemeral,
                attributes=attributes,
                on_stream=_on_stream,
                on_stream_end=_on_stream_end,
            )
            kwargs["on_runtime_admitted"] = _emit_run_started
            if override_runtime is not None:
                kwargs["runtime"] = override_runtime
            try:
                if self._mcp_provider is not None:
                    await self._mcp_provider.connect()
                response = await self._loop.process_direct(
                    message,
                    **kwargs,
                    hooks=per_run_hooks,
                )
                await _emit_run_started()
                await emitter.text_completed(resuming=False, force=False)
                result = result_from_response(response, capture)
                await emitter.emit(StreamEvent(
                    type=STREAM_EVENT_RUN_COMPLETED,
                    content=result.content,
                    result=result,
                    usage=result.usage,
                    metadata=dict(result.metadata),
                ))
                return result
            except Exception as exc:
                await _emit_run_started()
                await emitter.emit(StreamEvent(
                    type=STREAM_EVENT_RUN_FAILED,
                    error=str(exc),
                    metadata={"exception_type": type(exc).__name__},
                ))
                raise
            finally:
                await emitter.close()

        task = asyncio.create_task(_run())
        return RunStream(task, queue)

    async def stream(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        ephemeral: bool = False,
        attributes: Mapping[str, Any] | None = None,
        hooks: list[AgentHook] | None = None,
        model: str | None = None,
        model_preset: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream events for one agent turn."""
        run = await self.run_streamed(
            message,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            sender_id=sender_id,
            media=media,
            ephemeral=ephemeral,
            attributes=attributes,
            hooks=hooks,
            model=model,
            model_preset=model_preset,
        )
        try:
            async for event in run.stream_events():
                yield event
            await run.wait()
        finally:
            if not run.done:
                await run.aclose()

    async def aclose(self) -> None:
        """Release resources held by this instance."""
        try:
            await self._loop.aclose()
        finally:
            if self._mcp_provider is not None:
                await self._mcp_provider.aclose()

    async def __aenter__(self) -> Nanobot:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

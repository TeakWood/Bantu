"""The in-process gateway: the real runtime, addressed as a channel addresses it.

Every assertion goes through ``open_gateway``'s own surface — ``inject``,
``next_outbound`` and the agents it exposes — plus the session files on disk.
Only the model provider is faked: a turn has to produce a reply without a
network call.  No channel is ever constructed, which is what the harness is for,
and one test proves that by making construction fatal.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.tools.message import MessageTool
from nanobot.agents import MultiAgentRuntime, open_gateway, route
from nanobot.agents.runtime import AgentRuntime
from nanobot.bus.events import OutboundMessage
from nanobot.bus.outbound_events import ProgressEvent
from nanobot.providers.base import GenerationSettings, LLMResponse

TELEGRAM_INSTANCES = [
    {"id": "default", "token": "000:placeholder"},
    {"id": "research", "token": "111:placeholder", "agent": "research"},
    {"id": "ops", "token": "222:placeholder", "agent": "ops"},
]


# --- fixtures -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fast_pumps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the idle poll so a flag-based stop does not cost a second a test."""
    monkeypatch.setattr("nanobot.agents.multi.PUMP_POLL_INTERVAL_S", 0.02)


@pytest.fixture(autouse=True)
def _named_agents_under_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the default named-agent workspace root out of the user's home."""
    monkeypatch.setattr(
        "nanobot.agents.resolution.NAMED_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "named-agents"),
    )


class _FakeProvider:
    """Answers every request with a fixed reply instead of calling a model."""

    provider_name = "fake"

    def __init__(self, reply: str = "acknowledged") -> None:
        self._reply = reply
        self.generation = GenerationSettings()

    def get_default_model(self) -> str:
        return "fake/model"

    async def chat_stream_with_retry(self, **_kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self._reply, tool_calls=[], usage=None)


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a turn run end to end without a network call."""
    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider",
        lambda _config: _FakeProvider(),
    )


def write_config(tmp_path: Path, **overrides: Any) -> Path:
    """A config on disk with a default agent, two named agents and three bots."""
    data: dict[str, Any] = {
        "providers": {"openrouter": {"apiKey": "sk-test-key"}},
        "agents": {
            "defaults": {
                "model": "openai/gpt-4.1",
                "workspace": str(tmp_path / "default-workspace"),
            },
            "named": {"research": {}, "ops": {}},
        },
        "channels": {"telegram": {"instances": TELEGRAM_INSTANCES}},
    }
    data.update(overrides)
    config_dir = tmp_path / "instance"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


def session_keys(agent: AgentRuntime) -> set[str]:
    """Every session *agent* has persisted, read back from its own store."""
    return {str(info["key"]) for info in agent.sessions.list_sessions()}


# --- the contract ---------------------------------------------------------------


async def test_open_gateway_yields_a_harness_with_inject_and_next_outbound(
    tmp_path: Path,
) -> None:
    async with open_gateway(write_config(tmp_path)) as gateway:
        assert callable(gateway.inject)
        assert callable(gateway.next_outbound)
        assert isinstance(gateway.runtime, MultiAgentRuntime)
        assert gateway.names == ("default", "research", "ops")
        assert gateway.is_running


async def test_it_connects_to_no_external_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whole turn runs while constructing any channel at all would fail."""
    from nanobot.channels.manager import ChannelManager

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the harness must not construct a channel")

    monkeypatch.setattr(ChannelManager, "__init__", explode)

    async with open_gateway(write_config(tmp_path)) as gateway:
        await gateway.inject("telegram.research", "chat-1", "user-1", "hello")
        reply = await gateway.next_outbound(timeout=10.0)

    assert reply.content


async def test_it_runs_without_a_real_telegram_token(tmp_path: Path) -> None:
    """Both shapes: placeholder tokens, and no Telegram section at all."""
    async with open_gateway(write_config(tmp_path)) as gateway:
        assert gateway.agent_for("telegram.research").name == "research"

    channelless = write_config(tmp_path / "bare", channels={})
    async with open_gateway(channelless) as gateway:
        assert gateway.names == ("default", "research", "ops")
        assert gateway.agent_for("telegram").name == "default"


# --- routing a real message -----------------------------------------------------


async def test_a_message_on_a_bound_channel_is_processed_by_that_channels_agent(
    tmp_path: Path,
) -> None:
    async with open_gateway(write_config(tmp_path)) as gateway:
        msg = await gateway.inject(
            "telegram.research", "chat-1", "user-1", "who is on call?"
        )
        await gateway.next_outbound(timeout=10.0)
        gateway.runtime.flush_sessions()

        assert session_keys(gateway.agents["research"]) == {msg.session_key}
        assert session_keys(gateway.agents["ops"]) == set()
        assert session_keys(gateway.default) == set()


async def test_the_reply_carries_the_channel_and_chat_it_arrived_on(
    tmp_path: Path,
) -> None:
    async with open_gateway(write_config(tmp_path)) as gateway:
        await gateway.inject("telegram.ops", "chat-77", "user-1", "ship it")

        reply = await gateway.next_outbound(timeout=10.0)

        assert reply.channel == "telegram.ops"
        assert reply.chat_id == "chat-77"
        assert reply.content == "acknowledged"


async def test_an_unbound_channel_is_answered_by_the_default_agent(
    tmp_path: Path,
) -> None:
    async with open_gateway(write_config(tmp_path)) as gateway:
        msg = await gateway.inject("telegram", "chat-3", "user-1", "hello")

        reply = await gateway.next_outbound(timeout=10.0)
        gateway.runtime.flush_sessions()

        assert reply.channel == "telegram"
        assert session_keys(gateway.default) == {msg.session_key}
        assert session_keys(gateway.agents["research"]) == set()


async def test_two_agents_on_the_same_chat_id_keep_separate_sessions(
    tmp_path: Path,
) -> None:
    async with open_gateway(write_config(tmp_path)) as gateway:
        await gateway.inject("telegram.research", "42", "user-1", "the badge is 77")
        await gateway.next_outbound(timeout=10.0)
        await gateway.inject("telegram.ops", "42", "user-1", "what is the badge?")
        await gateway.next_outbound(timeout=10.0)
        gateway.runtime.flush_sessions()

        research = session_keys(gateway.agents["research"])
        ops = session_keys(gateway.agents["ops"])

    assert research == {"telegram.research:42"}
    assert ops == {"telegram.ops:42"}


async def test_agent_for_agrees_with_the_routing_decision(tmp_path: Path) -> None:
    async with open_gateway(write_config(tmp_path)) as gateway:
        for channel in ("telegram", "telegram.research", "telegram.ops", "cli"):
            expected = route(gateway.config, channel, "chat-1")
            assert gateway.agent_for(channel, "chat-1").name == expected


# --- next_outbound --------------------------------------------------------------


async def test_next_outbound_honours_its_timeout_rather_than_hanging(
    tmp_path: Path,
) -> None:
    async with open_gateway(write_config(tmp_path)) as gateway:
        loop = asyncio.get_running_loop()
        started = loop.time()

        with pytest.raises(TimeoutError):
            await gateway.next_outbound(timeout=0.1)

        assert loop.time() - started < 2.0


async def test_events_are_skipped_but_kept(tmp_path: Path) -> None:
    async with open_gateway(write_config(tmp_path)) as gateway:
        research = gateway.agents["research"]
        await research.bus.publish_outbound(
            OutboundMessage(
                channel="telegram.research",
                chat_id="chat-1",
                content="thinking",
                event=ProgressEvent(content="thinking"),
            )
        )
        await research.bus.publish_outbound(
            OutboundMessage(
                channel="telegram.research", chat_id="chat-1", content="the answer"
            )
        )

        msg = await gateway.next_outbound(timeout=5.0)

        assert msg.content == "the answer"
        assert [event.content for event in gateway.events] == ["thinking"]


async def test_events_are_returned_when_asked_for(tmp_path: Path) -> None:
    async with open_gateway(write_config(tmp_path)) as gateway:
        await gateway.default.bus.publish_outbound(
            OutboundMessage(
                channel="telegram",
                chat_id="chat-1",
                content="thinking",
                event=ProgressEvent(content="thinking"),
            )
        )

        msg = await gateway.next_outbound(timeout=5.0, include_events=True)

        assert msg.event is not None
        assert gateway.events == []


async def test_a_timeout_spent_on_events_alone_is_still_bounded(
    tmp_path: Path,
) -> None:
    """The deadline spans skipped events; it does not restart for each one."""
    async with open_gateway(write_config(tmp_path)) as gateway:
        for index in range(3):
            await gateway.default.bus.publish_outbound(
                OutboundMessage(
                    channel="telegram",
                    chat_id="chat-1",
                    content=str(index),
                    event=ProgressEvent(content=str(index)),
                )
            )

        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(TimeoutError):
            await gateway.next_outbound(timeout=0.2)

        assert loop.time() - started < 2.0
        assert len(gateway.events) == 3


# --- the agents behind the edge --------------------------------------------------


async def test_every_agent_can_send_proactively_on_its_own_bus(
    tmp_path: Path,
) -> None:
    """The ``message`` tool is wired per agent, as it is in the gateway."""
    async with open_gateway(write_config(tmp_path)) as gateway:
        for name in gateway.names:
            tool = gateway.agents[name].loop.tools.get("message")
            assert isinstance(tool, MessageTool)
            await tool.execute(
                content=f"from {name}", channel=f"telegram.{name}", chat_id="chat-5"
            )
            sent = await gateway.next_outbound(timeout=5.0)
            assert sent.content == f"from {name}"
            assert sent.chat_id == "chat-5"


async def test_each_agent_owns_its_bus_workspace_and_sessions(tmp_path: Path) -> None:
    async with open_gateway(write_config(tmp_path)) as gateway:
        agents = list(gateway)

        assert len({id(agent.bus) for agent in agents}) == 3
        assert all(agent.bus is not gateway.runtime.bus for agent in agents)
        assert len({agent.workspace for agent in agents}) == 3
        assert len({id(agent.sessions) for agent in agents}) == 3


async def test_inject_carries_metadata_and_an_explicit_session_key(
    tmp_path: Path,
) -> None:
    """A channel's thread-scoped key reaches the agent's store as it would live."""
    async with open_gateway(write_config(tmp_path)) as gateway:
        msg = await gateway.inject(
            "telegram.research",
            "chat-1",
            "user-1",
            "hello",
            metadata={"message_id": "7"},
            session_key_override="telegram.research:chat-1:thread-9",
        )
        assert msg.metadata == {"message_id": "7"}
        assert msg.session_key == "telegram.research:chat-1:thread-9"

        await gateway.next_outbound(timeout=10.0)
        gateway.runtime.flush_sessions()

        assert session_keys(gateway.agents["research"]) == {msg.session_key}


# --- lifecycle -------------------------------------------------------------------


def leaked_tasks(before: set[asyncio.Task[Any]]) -> set[str]:
    """Tasks started since *before* that are still pending, by name."""
    return {
        task.get_name()
        for task in asyncio.all_tasks()
        if task not in before and not task.done()
    }


async def test_exiting_shuts_every_agent_down_cleanly(tmp_path: Path) -> None:
    before = {task for task in asyncio.all_tasks()}

    async with open_gateway(write_config(tmp_path)) as gateway:
        await gateway.inject("telegram.research", "chat-1", "user-1", "hello")
        await gateway.next_outbound(timeout=10.0)
        runtime = gateway.runtime

    # Checked at the instant of exit, with no settling sleep: a task the harness
    # started but never awaited is exactly what raises "Task was destroyed but it
    # is pending", and it would have finished on its own during a sleep.
    assert leaked_tasks(before) == set()
    assert not runtime.is_running
    assert not any(agent.loop._running for agent in runtime)  # noqa: SLF001

    await asyncio.sleep(0.05)  # nothing may resurrect afterwards either

    assert leaked_tasks(before) == set()
    assert not any(agent.loop._running for agent in runtime)  # noqa: SLF001


async def test_the_harness_closes_even_if_the_body_raises(tmp_path: Path) -> None:
    runtime: MultiAgentRuntime | None = None

    with pytest.raises(ValueError, match="from the body"):
        async with open_gateway(write_config(tmp_path)) as gateway:
            runtime = gateway.runtime
            raise ValueError("from the body")

    assert runtime is not None
    assert not runtime.is_running


async def test_a_crashing_agent_loop_surfaces_rather_than_hanging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def crash(_self: Any) -> None:
        raise RuntimeError("loop exploded")

    monkeypatch.setattr("nanobot.agent.loop.AgentLoop.run", crash)

    # Surfaces from the start if the loops crash before the harness is ready, and
    # from the exit otherwise; either way the caller is told rather than left
    # waiting on a reply that can never arrive.
    with pytest.raises(RuntimeError, match="loop exploded"):
        async with open_gateway(write_config(tmp_path)) as gateway:
            await asyncio.sleep(0.05)
            assert gateway.names


async def test_a_loop_that_crashes_after_startup_surfaces_on_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure is retrieved from the run task rather than left unhandled."""
    crash_now = asyncio.Event()

    async def crash(_self: Any) -> None:
        await crash_now.wait()
        raise RuntimeError("loop exploded late")

    monkeypatch.setattr("nanobot.agent.loop.AgentLoop.run", crash)

    with pytest.raises(RuntimeError, match="loop exploded late"):
        async with open_gateway(write_config(tmp_path)) as gateway:
            assert gateway.is_running
            crash_now.set()
            await asyncio.sleep(0.05)


async def test_starting_twice_is_refused(tmp_path: Path) -> None:
    async with open_gateway(write_config(tmp_path)) as gateway:
        with pytest.raises(RuntimeError, match="already started"):
            await gateway.start()


async def test_a_missing_config_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        async with open_gateway(tmp_path / "nope.json"):
            pass

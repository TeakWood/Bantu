"""The channel-facing demux: one channel bus in front of N isolated agents."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nanobot.agents import (
    MultiAgentRuntime,
    agent_registry,
    build_agent_runtime,
    route,
)
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config
from nanobot.config.schema import Config
from nanobot.session.manager import SessionManager


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


# --- fixtures -----------------------------------------------------------------


TELEGRAM_INSTANCES = [
    {"id": "default", "token": "default-token"},
    {"id": "research", "token": "research-token", "agent": "research"},
    {"id": "ops", "token": "ops-token", "agent": "ops"},
]


def make_config(
    *,
    named: dict[str, Any] | None = None,
    telegram: dict[str, Any] | None = None,
) -> Config:
    """A config that needs no filesystem: enough for routing decisions alone."""
    raw: dict[str, Any] = {
        "agents": {
            "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
            "named": named if named is not None else {"research": {}, "ops": {}},
        },
    }
    if telegram is not None:
        raw["channels"] = {"telegram": telegram}
    return Config.model_validate(raw)


def bound_config(**kwargs: Any) -> Config:
    """Three bots: the default one unbound, two bound to declared agents."""
    return make_config(telegram={"instances": TELEGRAM_INSTANCES}, **kwargs)


def write_config(tmp_path: Path, **overrides: Any) -> Path:
    """Write a config on disk so real agent runtimes can be built from it."""
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
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


def load(tmp_path: Path, **overrides: Any) -> Config:
    return load_config(write_config(tmp_path, **overrides))


class FakeAgent:
    """Stands in for an :class:`AgentRuntime`, minus the LLM and the filesystem.

    The demux only ever touches ``name``, ``bus`` and the four lifecycle calls,
    so a substitute here lets the routing and shutdown behaviour be exercised
    without running a real agent loop.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.bus = MessageBus()
        self.sessions = None
        self.starts = 0
        self.live = False
        self.stopped = False
        self.closed = False
        self.mcp_connected = False

    async def connect_mcp(self) -> None:
        self.mcp_connected = True

    async def run(self) -> None:
        self.starts += 1
        self.live = True
        while self.live:
            await asyncio.sleep(0.005)

    def stop(self) -> None:
        self.stopped = True
        self.live = False

    async def aclose(self) -> None:
        self.closed = True


def fake_runtime(config: Config, *names: str) -> tuple[MultiAgentRuntime, dict[str, FakeAgent]]:
    """A runtime over substitute agents, ``default`` plus *names*."""
    agents = {name: FakeAgent(name) for name in ("default", *names)}
    runtime = MultiAgentRuntime(config, agents)  # pyright: ignore[reportArgumentType]
    return runtime, agents


def gate_mcp(agent: FakeAgent) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold *agent*'s MCP startup open, the way a stdio server takes seconds.

    Returns ``(entered, release)``: ``entered`` is set once startup is in
    flight, ``release`` lets it finish.
    """
    entered = asyncio.Event()
    release = asyncio.Event()

    async def connect() -> None:
        entered.set()
        await release.wait()
        agent.mcp_connected = True

    agent.connect_mcp = connect  # pyright: ignore[reportAttributeAccessIssue]
    return entered, release


async def until(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    """Poll *predicate* until it holds or *timeout* elapses."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


async def started(runtime: MultiAgentRuntime, agents: dict[str, FakeAgent]) -> asyncio.Task[None]:
    """Start *runtime* and wait until it and every agent loop are live."""
    task = asyncio.create_task(runtime.run())
    assert await until(
        lambda: runtime.is_running and all(agent.live for agent in agents.values())
    )
    return task


async def shut_down(runtime: MultiAgentRuntime, task: asyncio.Task[None]) -> None:
    """Stop *runtime* and await its run task, failing rather than hanging."""
    runtime.stop()
    await asyncio.wait_for(task, timeout=2.0)


def inbound(channel: str, chat_id: str = "chat-1", content: str = "hello") -> InboundMessage:
    return InboundMessage(
        channel=channel, sender_id="user-1", chat_id=chat_id, content=content
    )


# --- construction -------------------------------------------------------------


def test_a_config_with_no_named_agents_yields_exactly_one_agent(tmp_path: Path) -> None:
    config = load(tmp_path, agents={"defaults": {"model": "openai/gpt-4.1"}})

    runtime = MultiAgentRuntime.from_config(config)

    assert runtime.names == ("default",)
    assert len(runtime) == 1
    assert runtime.default is runtime.get("default")
    assert runtime.default.workspace == config.workspace_path


def test_from_config_builds_one_isolated_agent_per_configured_name(tmp_path: Path) -> None:
    runtime = MultiAgentRuntime.from_config(load(tmp_path))

    assert runtime.names == ("default", "research", "ops")
    buses = [agent.bus for agent in runtime]
    assert len({id(bus) for bus in buses}) == 3
    assert all(bus is not runtime.bus for bus in buses)
    assert len({id(agent.sessions) for agent in runtime}) == 3
    assert len({agent.workspace for agent in runtime}) == 3


def test_the_runtime_owns_a_channel_bus_unless_one_is_supplied(tmp_path: Path) -> None:
    config = load(tmp_path)
    channel_bus = MessageBus()

    assert MultiAgentRuntime.from_config(config).bus is not channel_bus
    assert MultiAgentRuntime.from_config(config, bus=channel_bus).bus is channel_bus


def test_a_runtime_without_a_default_agent_is_rejected() -> None:
    with pytest.raises(KeyError, match="default"):
        MultiAgentRuntime(make_config(), {})


def test_a_shared_session_manager_is_refused(tmp_path: Path) -> None:
    config = load(tmp_path)

    with pytest.raises(TypeError, match="session_manager"):
        MultiAgentRuntime.from_config(config, session_manager=SessionManager(tmp_path))


def test_a_supplied_bus_is_the_channel_bus_not_an_agent_bus(tmp_path: Path) -> None:
    channel_bus = MessageBus()

    runtime = MultiAgentRuntime.from_config(load(tmp_path), bus=channel_bus)

    assert runtime.bus is channel_bus
    assert all(agent.bus is not channel_bus for agent in runtime)


def test_membership_and_iteration_expose_the_agents(tmp_path: Path) -> None:
    runtime = MultiAgentRuntime.from_config(load(tmp_path))

    assert "research" in runtime
    assert "marketing" not in runtime
    assert runtime.get("marketing") is None
    assert [agent.name for agent in runtime] == ["default", "research", "ops"]


# --- routing ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        ("telegram.research", "research"),
        ("telegram.ops", "ops"),
        ("telegram", "default"),
        ("feishu.product", "default"),
        ("cli", "default"),
        ("system", "default"),
        ("websocket", "default"),
    ],
)
def test_agent_for_follows_the_configured_binding(channel: str, expected: str) -> None:
    runtime, _agents = fake_runtime(bound_config(), "research", "ops")

    assert runtime.agent_for(channel, "chat-1").name == expected


def test_a_channel_bound_to_an_unconfigured_agent_falls_back_to_default() -> None:
    config = make_config(
        named={"research": {}},
        telegram={"instances": [{"id": "ghost", "token": "t", "agent": "marketing"}]},
    )
    runtime, _agents = fake_runtime(config, "research")

    assert runtime.agent_for("telegram.ghost", "chat-1").name == "default"


def test_chat_id_does_not_change_the_agent() -> None:
    runtime, _agents = fake_runtime(bound_config(), "research", "ops")

    chosen = {runtime.agent_for("telegram.research", chat).name for chat in ("a", "b", None)}
    assert chosen == {"research"}


def test_routing_is_memoised_and_invalidated_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _agents = fake_runtime(bound_config(), "research", "ops")
    calls: list[str] = []

    def counting_route(config: Config, channel: str, chat_id: str | None) -> str:
        calls.append(channel)
        return route(config, channel, chat_id)

    monkeypatch.setattr("nanobot.agents.multi.route", counting_route)

    for _ in range(3):
        assert runtime.agent_for("telegram.research", "chat-1").name == "research"
    assert calls == ["telegram.research"]

    runtime.invalidate_routing()
    assert runtime.agent_for("telegram.research", "chat-1").name == "research"
    assert calls == ["telegram.research", "telegram.research"]


# --- inbound demux ------------------------------------------------------------


async def test_a_bound_channel_reaches_its_agent_and_no_other() -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    task = await started(runtime, agents)
    try:
        await runtime.bus.publish_inbound(inbound("telegram.research", content="hi"))

        delivered = await asyncio.wait_for(
            agents["research"].bus.consume_inbound(), timeout=2.0
        )
        assert delivered.content == "hi"
        assert await until(lambda: runtime.bus.inbound_size == 0)
        assert agents["default"].bus.inbound_size == 0
        assert agents["ops"].bus.inbound_size == 0
    finally:
        await shut_down(runtime, task)


@pytest.mark.parametrize("channel", ["telegram", "feishu.product", "cli"])
async def test_an_unbound_or_non_telegram_channel_reaches_default(channel: str) -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    task = await started(runtime, agents)
    try:
        await runtime.bus.publish_inbound(inbound(channel))

        delivered = await asyncio.wait_for(
            agents["default"].bus.consume_inbound(), timeout=2.0
        )
        assert delivered.channel == channel
        assert agents["research"].bus.inbound_size == 0
        assert agents["ops"].bus.inbound_size == 0
    finally:
        await shut_down(runtime, task)


async def test_the_demux_preserves_the_message_it_forwards() -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    task = await started(runtime, agents)
    try:
        msg = inbound("telegram.ops", chat_id="chat-9", content="ship it")
        await runtime.bus.publish_inbound(msg)

        delivered = await asyncio.wait_for(
            agents["ops"].bus.consume_inbound(), timeout=2.0
        )
        assert delivered is msg
    finally:
        await shut_down(runtime, task)


async def test_a_failing_delivery_does_not_kill_the_demux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    calls: list[str] = []
    original = MultiAgentRuntime.deliver_inbound

    async def flaky(self: MultiAgentRuntime, msg: InboundMessage) -> Any:
        calls.append(msg.channel)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return await original(self, msg)

    monkeypatch.setattr(MultiAgentRuntime, "deliver_inbound", flaky)
    task = await started(runtime, agents)
    try:
        await runtime.bus.publish_inbound(inbound("telegram.research"))
        await runtime.bus.publish_inbound(inbound("telegram.ops"))

        delivered = await asyncio.wait_for(
            agents["ops"].bus.consume_inbound(), timeout=2.0
        )
        assert delivered.channel == "telegram.ops"
        assert agents["research"].bus.inbound_size == 0
    finally:
        await shut_down(runtime, task)


# --- outbound pump ------------------------------------------------------------


@pytest.mark.parametrize("name", ["default", "research", "ops"])
async def test_an_outbound_message_reaches_the_channel_bus_unchanged(name: str) -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    task = await started(runtime, agents)
    try:
        reply = OutboundMessage(
            channel=f"telegram.{name}", chat_id="chat-7", content="done"
        )
        await agents[name].bus.publish_outbound(reply)

        forwarded = await asyncio.wait_for(runtime.bus.consume_outbound(), timeout=2.0)
        assert forwarded is reply
        assert forwarded.channel == f"telegram.{name}"
        assert forwarded.chat_id == "chat-7"
        assert forwarded.content == "done"
    finally:
        await shut_down(runtime, task)


async def test_every_agent_has_its_own_outbound_pump() -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    task = await started(runtime, agents)
    try:
        for name, agent in agents.items():
            await agent.bus.publish_outbound(
                OutboundMessage(channel="telegram", chat_id=name, content=name)
            )

        seen: set[str] = set()
        for _ in range(3):
            msg = await asyncio.wait_for(runtime.bus.consume_outbound(), timeout=2.0)
            seen.add(msg.content)
        assert seen == {"default", "research", "ops"}
    finally:
        await shut_down(runtime, task)


# --- lifecycle ----------------------------------------------------------------


async def test_starting_the_runtime_starts_every_agent_loop_and_connects_mcp() -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    task = await started(runtime, agents)
    try:
        assert runtime.is_running
        assert all(agent.mcp_connected for agent in agents.values())
        assert all(agent.starts == 1 for agent in agents.values())
    finally:
        await shut_down(runtime, task)


async def test_stopping_the_runtime_stops_every_agent_loop() -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    task = await started(runtime, agents)

    await shut_down(runtime, task)

    assert not runtime.is_running
    assert all(agent.stopped for agent in agents.values())
    assert task.done() and not task.cancelled()


async def test_closing_the_runtime_closes_every_agent() -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    task = await started(runtime, agents)

    await runtime.aclose()
    await asyncio.wait_for(task, timeout=2.0)

    assert all(agent.stopped for agent in agents.values())
    assert all(agent.closed for agent in agents.values())


async def test_one_agent_failing_to_close_does_not_skip_the_others() -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")

    async def boom() -> None:
        raise RuntimeError("cleanup failed")

    agents["research"].aclose = boom  # pyright: ignore[reportAttributeAccessIssue]

    await runtime.aclose()

    assert agents["default"].closed
    assert agents["ops"].closed


async def test_a_failing_mcp_startup_does_not_abort_the_other_agents() -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")

    async def boom() -> None:
        raise RuntimeError("no such server")

    agents["ops"].connect_mcp = boom  # pyright: ignore[reportAttributeAccessIssue]
    task = await started(runtime, agents)
    try:
        assert agents["default"].mcp_connected
        assert agents["research"].mcp_connected
        assert agents["ops"].live
    finally:
        await shut_down(runtime, task)


async def test_a_crashing_agent_loop_surfaces_from_run() -> None:
    runtime, agents = fake_runtime(bound_config(), "research")

    async def crash() -> None:
        raise RuntimeError("loop exploded")

    agents["research"].run = crash  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(RuntimeError, match="loop exploded"):
        await asyncio.wait_for(runtime.run(), timeout=2.0)
    assert not runtime.is_running


async def test_running_twice_concurrently_is_refused() -> None:
    runtime, agents = fake_runtime(bound_config(), "research")
    task = await started(runtime, agents)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await runtime.run()
    finally:
        await shut_down(runtime, task)


async def test_a_stopped_runtime_can_be_started_again() -> None:
    runtime, agents = fake_runtime(bound_config(), "research")
    await shut_down(runtime, await started(runtime, agents))

    task = await started(runtime, agents)
    try:
        assert runtime.is_running
        assert all(agent.starts == 2 for agent in agents.values())
    finally:
        await shut_down(runtime, task)


async def test_aclose_is_safe_before_the_runtime_ever_ran() -> None:
    runtime, agents = fake_runtime(bound_config(), "research")

    await runtime.aclose()

    assert all(agent.closed for agent in agents.values())


# --- shutdown racing startup ---------------------------------------------------
#
# MCP startup is the one await ``run`` performs before it owns any task, and for
# stdio servers it lasts seconds — the likeliest moment for an operator Ctrl-C.
# ``AgentLoop.run`` re-sets its own running flag on entry, so a shutdown lost in
# that window would start every loop it had just been asked to stop.
#
# Startup is not one window but three: before ``run`` takes its first step at
# all, during MCP startup, and between ``create_task`` and the task bodies'
# first step.  Each of the three is covered below.


async def test_a_stop_before_run_takes_its_first_step_starts_no_agent_loop() -> None:
    """``create_task`` only schedules ``run``; its body has not executed yet."""
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    task = asyncio.create_task(runtime.run())

    runtime.stop()  # run() has not executed a single statement
    await asyncio.wait_for(task, timeout=2.0)
    await asyncio.sleep(0.05)  # a resurrected loop would become live here

    assert not runtime.is_running
    assert all(agent.stopped for agent in agents.values())
    assert all(agent.starts == 0 for agent in agents.values())
    assert not any(agent.live for agent in agents.values())


async def test_an_aclose_before_run_takes_its_first_step_starts_no_agent_loop() -> None:
    """The dangerous direction: loops would run against already-closed agents."""
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    task = asyncio.create_task(runtime.run())

    await runtime.aclose()
    await asyncio.wait_for(task, timeout=2.0)
    await asyncio.sleep(0.05)

    assert not runtime.is_running
    assert all(agent.closed for agent in agents.values())
    assert all(agent.starts == 0 for agent in agents.values())
    assert not any(agent.live for agent in agents.values())


@pytest.mark.parametrize("ticks", range(12))
async def test_a_stop_at_any_startup_tick_leaves_no_loop_live(ticks: int) -> None:
    """Sweep the event-loop ticks ``run`` takes to reach its first await.

    One of these lands between ``create_task(agent.run())`` and those tasks'
    first step — the window the post-MCP check cannot see, since ``AgentLoop.run``
    sets its own running flag when the task body finally executes.
    """
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    task = asyncio.create_task(runtime.run())
    for _ in range(ticks):
        await asyncio.sleep(0)

    runtime.stop()
    await asyncio.wait_for(task, timeout=2.0)
    await asyncio.sleep(0.05)

    assert not runtime.is_running
    assert not any(agent.live for agent in agents.values())


async def test_stopping_during_mcp_startup_starts_no_agent_loop() -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    entered, release = gate_mcp(agents["research"])
    task = asyncio.create_task(runtime.run())
    await asyncio.wait_for(entered.wait(), timeout=2.0)

    runtime.stop()
    release.set()
    await asyncio.wait_for(task, timeout=2.0)
    await asyncio.sleep(0.05)  # a resurrected loop would become live here

    assert not runtime.is_running
    assert all(agent.stopped for agent in agents.values())
    assert all(agent.starts == 0 for agent in agents.values())
    assert not any(agent.live for agent in agents.values())


async def test_closing_during_mcp_startup_starts_no_agent_loop() -> None:
    runtime, agents = fake_runtime(bound_config(), "research", "ops")
    entered, release = gate_mcp(agents["research"])
    task = asyncio.create_task(runtime.run())
    await asyncio.wait_for(entered.wait(), timeout=2.0)

    await runtime.aclose()
    release.set()
    await asyncio.wait_for(task, timeout=2.0)
    await asyncio.sleep(0.05)  # a loop running against closed resources shows here

    assert not runtime.is_running
    assert all(agent.closed for agent in agents.values())
    assert all(agent.starts == 0 for agent in agents.values())
    assert not any(agent.live for agent in agents.values())


async def test_a_stop_during_startup_does_not_latch_the_runtime_shut() -> None:
    runtime, agents = fake_runtime(bound_config(), "research")
    entered, release = gate_mcp(agents["research"])
    first = asyncio.create_task(runtime.run())
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    runtime.stop()
    release.set()
    await asyncio.wait_for(first, timeout=2.0)

    task = await started(runtime, agents)
    try:
        assert all(agent.starts == 1 for agent in agents.values())
    finally:
        await shut_down(runtime, task)


# --- real agent runtimes ------------------------------------------------------


async def test_real_agents_route_and_reply_through_their_own_buses(tmp_path: Path) -> None:
    """End-to-end over real AgentRuntimes, driving the buses rather than an LLM."""
    runtime = MultiAgentRuntime.from_config(load(tmp_path))
    try:
        research = runtime.agent_for("telegram.research", "chat-1")
        assert research.name == "research"
        assert research is runtime.get("research")

        await runtime.deliver_inbound(inbound("telegram.research"))
        assert research.bus.inbound_size == 1
        assert runtime.default.bus.inbound_size == 0

        reply = OutboundMessage(
            channel="telegram.research", chat_id="chat-1", content="answered"
        )
        await research.bus.publish_outbound(reply)
        assert research.bus.outbound_size == 1
        await runtime.deliver_outbound(await research.bus.consume_outbound())
        assert (await runtime.bus.consume_outbound()) is reply
    finally:
        await runtime.aclose()


async def test_closing_real_agents_racing_startup_leaves_no_loop_running(
    tmp_path: Path,
) -> None:
    """The same race against real ``AgentLoop``s, whose flag the fakes only mimic."""
    runtime = MultiAgentRuntime.from_config(load(tmp_path))

    task = asyncio.create_task(runtime.run())
    await runtime.aclose()
    await asyncio.wait_for(task, timeout=3.0)
    await asyncio.sleep(0.05)

    assert not runtime.is_running
    assert not any(agent.loop._running for agent in runtime)  # noqa: SLF001


async def test_the_bus_the_demux_targets_is_the_bus_the_loop_pops(tmp_path: Path) -> None:
    """The link between 'delivered to A's bus' and 'processed by A's loop'."""
    runtime = MultiAgentRuntime.from_config(load(tmp_path))
    try:
        assert all(agent.loop.bus is agent.bus for agent in runtime)
    finally:
        await runtime.aclose()


async def test_agent_for_agrees_with_the_registrys_channel_attribution(
    tmp_path: Path,
) -> None:
    config = load(tmp_path)
    runtime = MultiAgentRuntime.from_config(config)
    try:
        attributed = [
            (channel, entry.name)
            for entry in agent_registry(config)
            for channel in entry.channels
        ]
        assert attributed  # the config binds bots; otherwise this proves nothing
        for channel, name in attributed:
            assert runtime.agent_for(channel, "chat-1").name == name
    finally:
        await runtime.aclose()


async def test_flush_sessions_covers_every_agent(tmp_path: Path) -> None:
    runtime = MultiAgentRuntime.from_config(load(tmp_path))
    try:
        assert runtime.flush_sessions() == 0
    finally:
        await runtime.aclose()


def test_a_supplied_agent_mapping_is_copied_not_aliased(tmp_path: Path) -> None:
    config = load(tmp_path)
    agents = {"default": build_agent_runtime(config, "default")}
    runtime = MultiAgentRuntime(config, agents)

    agents["research"] = build_agent_runtime(config, "research")

    assert runtime.names == ("default",)

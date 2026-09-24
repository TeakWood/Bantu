"""The fleet and bus `nanobot gateway` runs its channels on.

`open_gateway` drives the same per-agent runtimes with no channel attached; this
covers the half that only the live gateway exercises — the single bus every
channel publishes onto, and what it does with a message bound to a named agent.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from nanobot.agents import DEFAULT_AGENT_NAME, InboundRouter, NamedAgentFleet
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.cli import gateway_runtime
from nanobot.config.loader import load_config, resolve_config_env_vars

from .conftest import write_config
from .scripted_provider import ScriptedProvider, reply

BOTS = [
    {"id": "default", "token": "111:aaa"},
    {"id": "research", "token": "222:bbb", "agent": "research"},
]


def _config_path(instance_dir: Path, *, named: dict[str, object] | None = None) -> Path:
    return write_config(
        instance_dir,
        {
            "channels": {"telegram": {"enabled": True, "instances": BOTS}},
            "agents": {
                "defaults": {"workspace": str(instance_dir / "default-ws")},
                "named": (
                    {"research": {"workspace": str(instance_dir / "research-ws")}}
                    if named is None
                    else named
                ),
            },
        },
    )


def _load(path: Path):
    return resolve_config_env_vars(load_config(path), config_path=path)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    """Install a scripted provider for every agent built during the test."""
    state: dict[str, object] = {"script": lambda _m: reply("ok")}

    def factory(*_args: object, **_kwargs: object) -> ScriptedProvider:
        return ScriptedProvider(lambda messages: state["script"](messages))  # type: ignore[operator]

    monkeypatch.setattr("nanobot.providers.factory.make_provider", factory)
    return state


def _inbound(channel: str) -> InboundMessage:
    return InboundMessage(channel=channel, sender_id="user-1", chat_id="42", content="hi")


async def test_the_gateway_bus_hands_a_bound_bot_to_its_own_agent(
    instance_dir: Path,
    scripted: dict[str, object],
) -> None:
    """The bus a channel publishes onto diverts a bound bot to its agent's queue."""
    config = _load(_config_path(instance_dir))
    fleet = NamedAgentFleet(config)
    bus = fleet.host_bus()

    await bus.publish_inbound(_inbound("telegram.research"))

    # It never reaches the default agent's queue, which this bus *is*.
    assert bus.inbound_size == 0
    research = fleet.runtimes["research"]
    assert research.bus.inbound_size == 1
    assert research.bus.inbound.get_nowait().channel == "telegram.research"


async def test_the_gateway_bus_leaves_the_default_bot_to_the_default_agent(
    instance_dir: Path,
    scripted: dict[str, object],
) -> None:
    config = _load(_config_path(instance_dir))
    fleet = NamedAgentFleet(config)
    bus = fleet.host_bus()

    await bus.publish_inbound(_inbound("telegram"))
    await bus.publish_inbound(_inbound("websocket"))

    assert bus.inbound_size == 2
    assert fleet.runtimes["research"].bus.inbound_size == 0


async def test_a_bot_bound_to_an_undeclared_agent_is_served_by_nobody(
    instance_dir: Path,
    scripted: dict[str, object],
) -> None:
    """A sealed compartment stays sealed even when the config is wrong.

    `research` is bound but never declared. Handing its chat to `default` would
    file it in the default agent's memory and session store, so it is dropped.
    """
    config = _load(_config_path(instance_dir, named={}))
    fleet = NamedAgentFleet(config)
    bus = fleet.host_bus()

    assert fleet.agent_names == []
    assert fleet.routes_traffic  # a binding exists, so the gateway must honour it
    assert fleet.unroutable_channels == ("telegram.research",)

    await bus.publish_inbound(_inbound("telegram.research"))
    assert bus.inbound_size == 0

    # The unbound bot is unaffected.
    await bus.publish_inbound(_inbound("telegram"))
    assert bus.inbound_size == 1


async def test_a_config_with_no_named_agents_routes_everything_to_default(
    instance_dir: Path,
    scripted: dict[str, object],
) -> None:
    config = _load(
        write_config(
            instance_dir,
            {
                "channels": {"telegram": {"enabled": True, "token": "111:aaa"}},
                "agents": {"defaults": {"workspace": str(instance_dir / "default-ws")}},
            },
        )
    )
    fleet = NamedAgentFleet(config)

    # Nothing is bound, so the gateway keeps the plain bus it has always run on.
    assert not fleet
    assert not fleet.routes_traffic
    assert fleet.unroutable_channels == ()

    bus = fleet.host_bus()
    await bus.publish_inbound(_inbound("telegram"))
    assert bus.inbound_size == 1


async def test_a_named_agent_replies_on_the_bus_the_channels_consume(
    instance_dir: Path,
    scripted: dict[str, object],
) -> None:
    """End to end over the live-gateway wiring: in on the host bus, out on it too."""
    scripted["script"] = lambda _m: reply("Semis led the tape.")
    config = _load(_config_path(instance_dir))
    fleet = NamedAgentFleet(config)
    bus = fleet.host_bus()

    await fleet.start(bus)
    try:
        await bus.publish_inbound(_inbound("telegram.research"))
        out: OutboundMessage = await asyncio.wait_for(bus.consume_outbound(), timeout=30)
        while out.event is not None or not out.content:
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=30)

        # The reply leaves through the same bot and chat it arrived on.
        assert (out.channel, out.chat_id, out.content) == (
            "telegram.research",
            "42",
            "Semis led the tape.",
        )
    finally:
        await fleet.aclose()

    # The conversation is in research's store and in no other.
    research = fleet.runtimes["research"]
    assert [row["key"] for row in research.sessions.list_sessions()] == ["telegram.research:42"]
    default_sessions = instance_dir / "default-ws"
    assert not list(default_sessions.rglob("telegram.research*"))


def test_the_live_gateway_builds_its_bus_from_the_agent_fleet() -> None:
    """Regression guard: `nanobot gateway` must not run on one unrouted bus.

    Serving every declared bot from a single default-agent bus is the leak this
    feature exists to close, and it is invisible from outside until a named
    agent is configured.
    """
    source = inspect.getsource(gateway_runtime._run_gateway)

    assert "NamedAgentFleet(config)" in source
    assert "named_agents.host_bus() if named_agents.routes_traffic" in source
    assert "await named_agents.start(bus)" in source
    assert "await named_agents.aclose()" in source


def test_the_routing_bus_is_a_message_bus() -> None:
    """Everything downstream — channels, the WebUI, cron — takes the bus as given."""
    from nanobot.bus.queue import MessageBus

    assert issubclass(InboundRouter, MessageBus)
    assert DEFAULT_AGENT_NAME == "default"

"""Live routing, replies and subagent result delivery through `open_gateway`.

Covers acceptance criteria 8 (live routing and replies) and 9 (subagent result
routing).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agents import DEFAULT_AGENT_NAME, open_gateway
from nanobot.providers.base import LLMResponse

from .conftest import write_config
from .scripted_provider import ScriptedProvider, reply, tool_call, transcript

TWO_BOTS = {
    "channels": {
        "telegram": {
            "enabled": True,
            "instances": [
                {"id": "default", "token": "111:aaa"},
                {"id": "research", "token": "222:bbb", "agent": "research"},
            ],
        }
    },
}


def _config(instance_dir: Path) -> Path:
    return write_config(
        instance_dir,
        {
            **TWO_BOTS,
            "agents": {
                "defaults": {"workspace": str(instance_dir / "default-ws")},
                "named": {"research": {"workspace": str(instance_dir / "research-ws")}},
            },
        },
    )


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    """Install a scripted provider for every agent and expose the script hook."""
    state: dict[str, object] = {"script": lambda _m: reply("ok")}

    def factory(*_args: object, **_kwargs: object) -> ScriptedProvider:
        return ScriptedProvider(lambda messages: state["script"](messages))  # type: ignore[operator]

    monkeypatch.setattr("nanobot.providers.factory.make_provider", factory)
    return state


async def test_a_message_is_processed_by_the_agent_its_bot_is_bound_to(
    instance_dir: Path,
    scripted: dict[str, object],
) -> None:
    scripted["script"] = lambda _m: reply("Semis led the tape.")

    async with open_gateway(_config(instance_dir)) as gateway:
        assert gateway.agent_for("telegram.research") == "research"
        assert gateway.agent_for("telegram") == DEFAULT_AGENT_NAME

        assert await gateway.inject("telegram.research", "42", "user-1", "what moved?") == (
            "research"
        )
        out = await gateway.next_outbound(timeout=20)

        # The reply leaves through the same bot and chat the message arrived on.
        assert out.channel == "telegram.research"
        assert out.chat_id == "42"
        assert out.content == "Semis led the tape."
        assert out.agent == "research"

        research = gateway.runtime("research")
        default = gateway.runtime(DEFAULT_AGENT_NAME)
        research.sessions.flush_all()
        default.sessions.flush_all()

        # The session lands in research's store and in no other store.
        assert [row["key"] for row in research.sessions.list_sessions()] == [
            "telegram.research:42"
        ]
        assert default.sessions.list_sessions() == []


async def test_an_unbound_bot_is_served_by_the_default_agent(
    instance_dir: Path,
    scripted: dict[str, object],
) -> None:
    scripted["script"] = lambda _m: reply("Default here.")

    async with open_gateway(_config(instance_dir)) as gateway:
        assert await gateway.inject("telegram", "77", "user-1", "hello") == DEFAULT_AGENT_NAME
        out = await gateway.next_outbound(timeout=20)

        assert (out.channel, out.chat_id, out.agent) == ("telegram", "77", DEFAULT_AGENT_NAME)

        gateway.runtime("research").sessions.flush_all()
        assert gateway.runtime("research").sessions.list_sessions() == []


async def test_traffic_for_one_agent_never_reaches_another(
    instance_dir: Path,
    scripted: dict[str, object],
) -> None:
    def script(messages: list[dict[str, object]]) -> LLMResponse:
        return reply("research-secret" if "sector notes" in transcript(messages) else "plain")

    scripted["script"] = script

    async with open_gateway(_config(instance_dir)) as gateway:
        await gateway.inject("telegram.research", "42", "user-1", "sector notes please")
        first = await gateway.next_outbound(timeout=20)
        assert first.agent == "research"

        await gateway.inject("telegram", "42", "user-1", "anything about sectors?")
        second = await gateway.next_outbound(timeout=20)

        assert second.agent == DEFAULT_AGENT_NAME
        assert second.channel == "telegram"
        # The default agent has no sight of the research conversation.
        assert second.content == "plain"
        default_session = gateway.runtime(DEFAULT_AGENT_NAME).sessions.get_or_create(
            "telegram.research:42"
        )
        assert default_session.messages == []


SUBAGENT_TASK = "read the sector wires"
SUBAGENT_ANSWER = "FABS_BOOKED_OUT_THROUGH_Q3"


def _spawning_script(messages: list[dict[str, object]]) -> LLMResponse:
    """Spawn once, acknowledge, then answer from the subagent's finding."""
    text = transcript(messages)
    if SUBAGENT_ANSWER in text:
        # The parent's follow-up turn, driven by the subagent's delivered result.
        return reply("Wires say: fabs are booked out.")
    if any(message.get("role") == "tool" for message in messages):
        return reply("Working on it.")
    if SUBAGENT_TASK in text:
        # This is the subagent's own run.
        return reply(SUBAGENT_ANSWER)
    return tool_call("spawn", {"task": SUBAGENT_TASK, "label": "wires"})


async def test_a_subagent_result_returns_to_the_session_it_was_spawned_from(
    instance_dir: Path,
    scripted: dict[str, object],
) -> None:
    scripted["script"] = _spawning_script

    async with open_gateway(_config(instance_dir)) as gateway:
        research = gateway.runtime("research")
        await gateway.inject("telegram.research", "42", "user-1", "dig into the sector")

        # The subagent delivers its own result back into the originating
        # session; the agent may acknowledge first or fold the finding straight
        # into one reply, so read until the finding surfaces.
        replies = []
        while True:
            replies.append(await gateway.next_outbound(timeout=60))
            if "fabs are booked out" in replies[-1].content:
                break

        # Every reply goes out on the channel and chat the message came from.
        for record in replies:
            assert (record.channel, record.chat_id, record.agent) == (
                "telegram.research",
                "42",
                "research",
            )

        research.sessions.flush_all()
        session = research.sessions.get_or_create("telegram.research:42")
        assert any(SUBAGENT_ANSWER in str(message) for message in session.messages)

        # No other agent's session received anything from it.
        default = gateway.runtime(DEFAULT_AGENT_NAME)
        default.sessions.flush_all()
        assert default.sessions.list_sessions() == []


async def test_every_agent_runs_side_by_side_under_one_gateway(
    instance_dir: Path,
    scripted: dict[str, object],
) -> None:
    async with open_gateway(_config(instance_dir)) as gateway:
        assert gateway.agent_names == [DEFAULT_AGENT_NAME, "research"]
        # Each agent owns a private bus, so no two agents can consume each
        # other's inbound traffic.
        buses = {id(runtime.bus) for runtime in gateway.runtimes.values()}
        assert len(buses) == 2


async def test_next_outbound_times_out_when_nothing_is_sent(
    instance_dir: Path,
    scripted: dict[str, object],
) -> None:
    async with open_gateway(_config(instance_dir)) as gateway:
        with pytest.raises(TimeoutError):
            await gateway.next_outbound(timeout=0.2)


async def test_open_gateway_rejects_a_missing_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        async with open_gateway(tmp_path / "nope.json"):
            pass

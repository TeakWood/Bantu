"""Acceptance for named-agent criteria 7-9, checked from outside the gateway.

Criterion 7 goes through the config file, :func:`nanobot.agents.route` and
``nanobot agents list --json``.  Criteria 8 and 9 go through
:func:`nanobot.agents.open_gateway` — ``inject`` and ``next_outbound`` are the
whole surface — plus the session files each agent persisted on disk.

Reading the stores on disk is what lets the negatives be stated as negatives: a
criterion that only asserted "the result arrived" would pass just as well for a
runtime that broadcast it to every agent.  ``sessions_dir`` is a public
attribute of ``SessionManager``, and every path here is built with ``pathlib``,
so the same assertions hold on Windows.

Nothing contacts a network service.  No channel is constructed — ``open_gateway``
replaces the channel edge outright — so the Telegram tokens below are
placeholders, and the model provider is faked, as it is elsewhere in this suite.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from nanobot.agents import open_gateway, route
from nanobot.agents.runtime import AgentRuntime
from nanobot.cli.commands import app
from nanobot.config.loader import load_config
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest

runner = CliRunner()

# Three bots: the default one claims no agent, the other two claim one each.
TELEGRAM_INSTANCES: list[dict[str, Any]] = [
    {"id": "default", "token": "000:placeholder", "enabled": True},
    {"id": "research", "token": "111:placeholder", "enabled": True, "agent": "research"},
    {"id": "ops", "token": "222:placeholder", "enabled": True, "agent": "ops"},
]

# The single-bot shape an install written before named agents existed carries.
LEGACY_TELEGRAM_SECTION: dict[str, Any] = {"enabled": True, "token": "999:placeholder"}


# --- configs ------------------------------------------------------------------


def write_config(directory: Path, payload: Mapping[str, Any]) -> Path:
    """Write *payload* as a config file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def payload(tmp_path: Path, telegram: Any) -> dict[str, Any]:
    """A config with two named agents and *telegram* as its Telegram section."""
    return {
        "providers": {"openrouter": {"apiKey": "sk-test-key"}},
        "agents": {
            "defaults": {
                "model": "base/model",
                "workspace": str(tmp_path / "default-workspace"),
            },
            "named": {"research": {}, "ops": {}},
        },
        "channels": {"telegram": telegram},
    }


def multi_bot_config(tmp_path: Path) -> Path:
    """The default bot plus two bots with ids, each bound to a named agent."""
    return write_config(
        tmp_path / "instance",
        payload(tmp_path, {"instances": [dict(spec) for spec in TELEGRAM_INSTANCES]}),
    )


def single_bot_config(tmp_path: Path) -> Path:
    """A pre-existing single-bot Telegram config, written the old flat way."""
    return write_config(
        tmp_path / "legacy",
        payload(tmp_path, dict(LEGACY_TELEGRAM_SECTION)),
    )


def list_agents(config_path: Path) -> list[dict[str, Any]]:
    """Run ``nanobot agents list --json`` and return the parsed document."""
    result = runner.invoke(app, ["agents", "list", "--json", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def channels_by_agent(config_path: Path) -> dict[str, list[str]]:
    """The runtime channels the CLI attributes to each agent."""
    return {entry["name"]: entry["channels"] for entry in list_agents(config_path)}


# --- fakes --------------------------------------------------------------------

REPLY = "acknowledged"

# What the spawning agent asks its subagent for, what the subagent answers, and
# the two replies the agent itself produces.
SPAWN_TASK = "look up the badge number"
SUBAGENT_FINDING = "Badge number 77 belongs to the night shift."
SPAWN_ACK = "Started looking into it."
FOLLOW_UP = "The badge belongs to the night shift."

# Markers the scripted provider reads off the transcript to decide where in the
# subagent round trip it is.  Each is produced by production code: the subagent's
# own system prompt, the announce template, and ``SubagentManager.spawn``'s
# return value respectively.
SUBAGENT_SYSTEM_PROMPT_MARKER = "# Subagent"
ANNOUNCE_MARKER = "[Subagent '"
SPAWN_STARTED_MARKER = "] started (id:"


def message_text(message: Mapping[str, Any]) -> str:
    """Flatten one request message to text, whatever content shape it carries."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in cast(list[Any], content):
            parts.append(str(part.get("text", "")) if isinstance(part, Mapping) else str(part))
        return " ".join(parts)
    return "" if content is None else str(content)


class _FakeProvider:
    """Answers every request with a fixed reply instead of calling a model."""

    provider_name = "fake"

    def __init__(self, reply: str = REPLY) -> None:
        self._reply = reply
        self.generation = GenerationSettings()

    def get_default_model(self) -> str:
        return "fake/model"

    async def chat_stream_with_retry(self, **_kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self._reply, tool_calls=[])


class _SpawningProvider(_FakeProvider):
    """Drives one background-subagent round trip, reading the transcript.

    The agent spawns on its first request, acknowledges once the spawn returns,
    and summarises once the subagent's result reaches it.  Which branch to take
    is decided from the transcript rather than from a call counter, because the
    subagent's own requests are interleaved with the agent's and a counter would
    be a race.
    """

    async def chat_stream_with_retry(self, **kwargs: Any) -> LLMResponse:
        messages: Sequence[Mapping[str, Any]] = kwargs.get("messages") or []
        system = message_text(messages[0]) if messages else ""
        transcript = "\n".join(message_text(message) for message in messages)

        if system.lstrip().startswith(SUBAGENT_SYSTEM_PROMPT_MARKER):
            return LLMResponse(content=SUBAGENT_FINDING, tool_calls=[])
        if ANNOUNCE_MARKER in transcript:
            return LLMResponse(content=FOLLOW_UP, tool_calls=[])
        if SPAWN_STARTED_MARKER in transcript:
            return LLMResponse(content=SPAWN_ACK, tool_calls=[])
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call-spawn-1",
                    name="spawn",
                    arguments={"task": SPAWN_TASK, "label": "badge"},
                )
            ],
        )


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let a turn run end to end without a network call."""
    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider",
        lambda _config: _FakeProvider(),
    )


@pytest.fixture
def spawning_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every agent answers its first message by spawning a background subagent."""
    monkeypatch.setattr(
        "nanobot.providers.factory.make_provider",
        lambda _config: _SpawningProvider(),
    )


@pytest.fixture(autouse=True)
def _named_agents_under_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the default named-agent workspace root out of the user's home."""
    monkeypatch.setattr(
        "nanobot.agents.resolution.NAMED_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "named-agents"),
    )


@pytest.fixture(autouse=True)
def _fast_pumps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the idle poll so a flag-based stop does not cost a second a test."""
    monkeypatch.setattr("nanobot.agents.multi.PUMP_POLL_INTERVAL_S", 0.02)


# --- the session stores, on disk ----------------------------------------------


def session_files(agent: AgentRuntime) -> dict[str, str]:
    """Every session file *agent* has written, keyed by file name.

    Read straight off ``sessions_dir`` rather than through the manager's cache,
    so "no other agent received anything" is a statement about the filesystem.
    """
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(agent.sessions.sessions_dir.glob("*.jsonl"))
    }


def persisted_text(agent: AgentRuntime) -> str:
    """Everything *agent* has persisted, as one blob to search."""
    return "\n".join(session_files(agent).values())


def session_keys(agent: AgentRuntime) -> set[str]:
    """Every session key *agent* has persisted, read back from its own store."""
    return {str(info["key"]) for info in agent.sessions.list_sessions()}


# ==============================================================================
# Criterion 7: Telegram binding
# ==============================================================================


def test_a_config_with_three_bots_loads_with_all_three_agents(tmp_path: Path) -> None:
    config = load_config(multi_bot_config(tmp_path))

    assert list(config.agents.named) == ["research", "ops"]
    assert [entry["name"] for entry in list_agents(multi_bot_config(tmp_path))] == [
        "default",
        "research",
        "ops",
    ]


def test_route_returns_each_bots_bound_agent(tmp_path: Path) -> None:
    config = load_config(multi_bot_config(tmp_path))

    assert route(config, "telegram.research", "chat-1") == "research"
    assert route(config, "telegram.ops", "chat-1") == "ops"


def test_route_returns_default_for_the_unbound_bot(tmp_path: Path) -> None:
    config = load_config(multi_bot_config(tmp_path))

    assert route(config, "telegram", "chat-1") == "default"


@pytest.mark.parametrize(
    "channel",
    ["cli", "system", "webui", "discord", "feishu", "feishu.product", "websocket"],
)
def test_route_returns_default_for_any_non_telegram_channel(
    tmp_path: Path,
    channel: str,
) -> None:
    config = load_config(multi_bot_config(tmp_path))

    assert route(config, channel, "chat-1") == "default"


def test_the_listing_shows_each_bot_under_its_agent(tmp_path: Path) -> None:
    by_agent = channels_by_agent(multi_bot_config(tmp_path))

    assert by_agent == {
        "default": ["telegram"],
        "research": ["telegram.research"],
        "ops": ["telegram.ops"],
    }


def test_a_single_bot_config_still_loads_as_telegram_under_default(
    tmp_path: Path,
) -> None:
    """The shape an install written before named agents existed still carries."""
    config_path = single_bot_config(tmp_path)
    config = load_config(config_path)

    assert route(config, "telegram", "chat-1") == "default"
    assert channels_by_agent(config_path) == {
        "default": ["telegram"],
        "research": [],
        "ops": [],
    }


def test_reading_a_single_bot_config_leaves_its_telegram_section_unchanged(
    tmp_path: Path,
) -> None:
    """Loading is a read: the flat section is neither migrated nor rewritten."""
    config_path = single_bot_config(tmp_path)

    config = load_config(config_path)
    route(config, "telegram", "chat-1")
    list_agents(config_path)

    section = getattr(config.channels, "telegram", None)
    assert section == LEGACY_TELEGRAM_SECTION
    on_disk = json.loads(config_path.read_text(encoding="utf-8"))
    assert on_disk["channels"]["telegram"] == LEGACY_TELEGRAM_SECTION


async def test_the_running_gateway_routes_the_way_route_says_it_will(
    tmp_path: Path,
    fake_llm: None,
) -> None:
    """The live demux and the pure routing decision agree, bot by bot."""
    config_path = multi_bot_config(tmp_path)

    async with open_gateway(config_path) as gateway:
        for channel in ("telegram", "telegram.research", "telegram.ops", "cli"):
            expected = route(gateway.config, channel, "chat-1")
            assert gateway.agent_for(channel, "chat-1").name == expected


# ==============================================================================
# Criterion 8: live routing and replies
# ==============================================================================


async def test_a_message_on_a_bound_bot_is_processed_by_that_bots_agent(
    tmp_path: Path,
    fake_llm: None,
) -> None:
    async with open_gateway(multi_bot_config(tmp_path)) as gateway:
        msg = await gateway.inject("telegram.research", "chat-1", "user-1", "who is on call?")
        await gateway.next_outbound(timeout=10.0)
        gateway.runtime.flush_sessions()

        research, ops, default = (
            gateway.agents["research"],
            gateway.agents["ops"],
            gateway.default,
        )

        assert session_keys(research) == {msg.session_key}
        # The negative, stated against the filesystem: the other two agents wrote
        # no session file at all, not merely no session with this key.
        assert session_files(ops) == {}
        assert session_files(default) == {}
        assert "who is on call?" in persisted_text(research)


async def test_the_whole_round_trip_runs_with_no_channel_constructed(
    tmp_path: Path,
    fake_llm: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing can reach api.telegram.org: building any channel at all is fatal."""
    from nanobot.channels.manager import ChannelManager

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("acceptance must not construct a channel")

    monkeypatch.setattr(ChannelManager, "__init__", explode)

    async with open_gateway(multi_bot_config(tmp_path)) as gateway:
        await gateway.inject("telegram.research", "chat-1", "user-1", "who is on call?")
        reply = await gateway.next_outbound(timeout=10.0)

    assert reply.channel == "telegram.research"
    assert reply.content == REPLY


async def test_the_reply_comes_back_on_the_bot_and_chat_it_arrived_on(
    tmp_path: Path,
    fake_llm: None,
) -> None:
    async with open_gateway(multi_bot_config(tmp_path)) as gateway:
        await gateway.inject("telegram.research", "chat-1", "user-1", "who is on call?")

        reply = await gateway.next_outbound(timeout=10.0)

        assert reply.channel == "telegram.research"
        assert reply.chat_id == "chat-1"
        assert reply.content == REPLY


async def test_two_bots_on_the_same_chat_id_write_to_different_stores(
    tmp_path: Path,
    fake_llm: None,
) -> None:
    """Same chat id, different bots: neither session file can be the other's."""
    async with open_gateway(multi_bot_config(tmp_path)) as gateway:
        await gateway.inject("telegram.research", "42", "user-1", "the badge is 77")
        await gateway.next_outbound(timeout=10.0)
        await gateway.inject("telegram.ops", "42", "user-1", "what is the badge?")
        await gateway.next_outbound(timeout=10.0)
        gateway.runtime.flush_sessions()

        research, ops = gateway.agents["research"], gateway.agents["ops"]

        assert session_keys(research) == {"telegram.research:42"}
        assert session_keys(ops) == {"telegram.ops:42"}
        assert research.sessions.sessions_dir != ops.sessions.sessions_dir
        assert "the badge is 77" in persisted_text(research)
        assert "the badge is 77" not in persisted_text(ops)
        assert session_files(gateway.default) == {}


# ==============================================================================
# Criterion 9: subagent result routing
# ==============================================================================


async def test_a_subagents_result_lands_in_its_spawners_session_and_nowhere_else(
    tmp_path: Path,
    spawning_llm: None,
) -> None:
    """The test that proves the per-agent bus, stated as a negative.

    A subagent announces its result as ``InboundMessage(channel='system')``
    republished onto the spawning agent's own bus.  Were that bus shared, the
    result would reach whichever agent popped it first — so the assertions that
    matter are the ones about the stores that must stay empty.

    The announcement reaches the agent while the spawning turn is still open, so
    the agent folds it into that turn and the follow-up reply is the turn's one
    outgoing message.  It still travels the agent's own bus to get there.
    """
    async with open_gateway(multi_bot_config(tmp_path)) as gateway:
        research, ops, default = (
            gateway.agents["research"],
            gateway.agents["ops"],
            gateway.default,
        )

        msg = await gateway.inject(
            "telegram.research", "chat-1", "user-1", "who is on call?"
        )
        follow_up = await gateway.next_outbound(timeout=30.0)
        gateway.runtime.flush_sessions()

        # The result reached the originating session, and only that session...
        assert session_keys(research) == {msg.session_key}
        assert len(session_files(research)) == 1
        persisted = persisted_text(research)
        assert SUBAGENT_FINDING in persisted
        assert FOLLOW_UP in persisted

        # ...the follow-up went out on the channel and chat the message came from...
        assert follow_up.content == FOLLOW_UP
        assert follow_up.channel == "telegram.research"
        assert follow_up.chat_id == "chat-1"

        # ...and no other agent's store was touched by any of it.
        assert session_files(ops) == {}
        assert session_files(default) == {}


async def test_two_agents_spawning_on_the_same_chat_id_never_cross(
    tmp_path: Path,
    spawning_llm: None,
) -> None:
    """Both agents run the same round trip at once; neither sees the other's."""
    async with open_gateway(multi_bot_config(tmp_path)) as gateway:
        research, ops, default = (
            gateway.agents["research"],
            gateway.agents["ops"],
            gateway.default,
        )

        await gateway.inject("telegram.research", "42", "user-1", "research asks")
        await gateway.inject("telegram.ops", "42", "user-1", "ops asks")
        replies = [await gateway.next_outbound(timeout=30.0) for _ in range(2)]
        gateway.runtime.flush_sessions()

        assert {reply.channel for reply in replies} == {
            "telegram.research",
            "telegram.ops",
        }
        assert {reply.chat_id for reply in replies} == {"42"}
        assert {reply.content for reply in replies} == {FOLLOW_UP}

        assert session_keys(research) == {"telegram.research:42"}
        assert session_keys(ops) == {"telegram.ops:42"}
        assert "research asks" in persisted_text(research)
        assert "research asks" not in persisted_text(ops)
        assert "ops asks" in persisted_text(ops)
        assert "ops asks" not in persisted_text(research)
        assert SUBAGENT_FINDING in persisted_text(research)
        assert SUBAGENT_FINDING in persisted_text(ops)
        assert session_files(default) == {}

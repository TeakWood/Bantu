"""Tests for session ownership by agent (Bantu-kx9)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.session.manager import Session, SessionManager


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _manager(tmp_path: Path) -> SessionManager:
    return SessionManager(workspace=tmp_path)


# ---------------------------------------------------------------------------
# Session.agent_id field
# ---------------------------------------------------------------------------

class TestSessionAgentIdField:
    def test_default_agent_id(self) -> None:
        s = Session(key="telegram:123")
        assert s.agent_id == "default"

    def test_explicit_agent_id(self) -> None:
        s = Session(key="telegram:123", agent_id="silpi")
        assert s.agent_id == "silpi"


# ---------------------------------------------------------------------------
# Separate sessions per agent for the same key
# ---------------------------------------------------------------------------

class TestSeparateSessionsPerAgent:
    def test_different_agents_get_different_sessions(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        key = "telegram:123"

        s_default = mgr.get_or_create(key, agent_id="default")
        s_silpi = mgr.get_or_create(key, agent_id="silpi")

        assert s_default is not s_silpi

    def test_same_agent_same_key_returns_same_object(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        key = "telegram:123"

        s1 = mgr.get_or_create(key, agent_id="silpi")
        s2 = mgr.get_or_create(key, agent_id="silpi")

        assert s1 is s2

    def test_different_agents_stored_in_separate_files(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        key = "telegram:123"

        s_default = mgr.get_or_create(key, agent_id="default")
        s_silpi = mgr.get_or_create(key, agent_id="silpi")

        s_default.add_message("user", "hello from default")
        s_silpi.add_message("user", "hello from silpi")

        mgr.save(s_default)
        mgr.save(s_silpi)

        path_default = mgr._get_session_path(key, "default")
        path_silpi = mgr._get_session_path(key, "silpi")

        assert path_default.exists()
        assert path_silpi.exists()
        assert path_default != path_silpi

    def test_default_agent_uses_bare_filename(self, tmp_path: Path) -> None:
        """Backward-compat: the default agent's file must not have an agent prefix."""
        mgr = _manager(tmp_path)
        path = mgr._get_session_path("telegram:123", "default")
        # Must not start with "default__"
        assert not path.name.startswith("default__")

    def test_non_default_agent_uses_prefixed_filename(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        path = mgr._get_session_path("telegram:123", "silpi")
        assert path.name.startswith("silpi__")

    def test_messages_are_isolated_between_agents(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        key = "discord:456"

        s_default = mgr.get_or_create(key, agent_id="default")
        s_silpi = mgr.get_or_create(key, agent_id="silpi")

        s_default.add_message("user", "message for default")
        mgr.save(s_default)
        mgr.save(s_silpi)

        # Reload both
        mgr2 = _manager(tmp_path)
        r_default = mgr2.get_or_create(key, agent_id="default")
        r_silpi = mgr2.get_or_create(key, agent_id="silpi")

        assert len(r_default.messages) == 1
        assert len(r_silpi.messages) == 0


# ---------------------------------------------------------------------------
# Session persistence with agent_id
# ---------------------------------------------------------------------------

class TestSessionPersistenceWithAgentId:
    def test_agent_id_written_to_jsonl(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        s = mgr.get_or_create("telegram:99", agent_id="silpi")
        mgr.save(s)

        path = mgr._get_session_path("telegram:99", "silpi")
        with open(path, encoding="utf-8") as fh:
            first_line = json.loads(fh.readline())

        assert first_line.get("_type") == "metadata"
        assert first_line.get("agent_id") == "silpi"

    def test_agent_id_round_trips_through_disk(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        s = mgr.get_or_create("telegram:99", agent_id="silpi")
        s.add_message("user", "hi")
        mgr.save(s)

        mgr2 = _manager(tmp_path)
        loaded = mgr2.get_or_create("telegram:99", agent_id="silpi")

        assert loaded.agent_id == "silpi"
        assert len(loaded.messages) == 1

    def test_loading_session_without_agent_id_defaults_to_default(self, tmp_path: Path) -> None:
        """Old JSONL files that lack 'agent_id' in metadata must load as 'default'."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir(parents=True)

        # Write a legacy-style JSONL file without agent_id in metadata
        legacy_path = sessions_dir / "telegram_legacy.jsonl"
        metadata = {
            "_type": "metadata",
            "key": "telegram:legacy",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "metadata": {},
            "last_consolidated": 0,
            # Note: no "agent_id" key
        }
        with open(legacy_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(metadata) + "\n")

        mgr = _manager(tmp_path)
        session = mgr.get_or_create("telegram:legacy", agent_id="default")

        assert session.agent_id == "default"

    def test_default_agent_session_path_unchanged_from_legacy(self, tmp_path: Path) -> None:
        """The file produced for agent_id='default' must be loadable as a plain key."""
        mgr = _manager(tmp_path)
        s = mgr.get_or_create("cli:user", agent_id="default")
        s.add_message("user", "legacy content")
        mgr.save(s)

        # File should live at sessions/cli_user.jsonl, not sessions/default__cli_user.jsonl
        expected = tmp_path / "sessions" / "cli_user.jsonl"
        assert expected.exists()


# ---------------------------------------------------------------------------
# InboundMessage.agent_id routing hint
# ---------------------------------------------------------------------------

class TestInboundMessageAgentId:
    def test_agent_id_defaults_to_none(self) -> None:
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="123", content="hi")
        assert msg.agent_id is None

    def test_agent_id_can_be_set(self) -> None:
        msg = InboundMessage(
            channel="telegram", sender_id="u1", chat_id="123",
            content="hi", agent_id="silpi",
        )
        assert msg.agent_id == "silpi"

    def test_session_key_is_unaffected_by_agent_id(self) -> None:
        msg_no_agent = InboundMessage(
            channel="telegram", sender_id="u1", chat_id="123", content="hi",
        )
        msg_with_agent = InboundMessage(
            channel="telegram", sender_id="u1", chat_id="123",
            content="hi", agent_id="silpi",
        )
        assert msg_no_agent.session_key == msg_with_agent.session_key == "telegram:123"


# ---------------------------------------------------------------------------
# AgentLoop._process_message routing
# ---------------------------------------------------------------------------

def _make_loop(tmp_path: Path):
    """Create a minimal AgentLoop for testing session routing."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.session.manager import SessionManager

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    session_manager = SessionManager(workspace=tmp_path)

    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SubagentManager"):
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=workspace,
            session_manager=session_manager,
        )
    return loop, bus, session_manager


class TestAgentLoopRoutingByAgentId:
    @pytest.mark.asyncio
    async def test_message_without_agent_id_uses_default(self, tmp_path: Path) -> None:
        loop, bus, session_manager = _make_loop(tmp_path)

        msg = InboundMessage(
            channel="telegram", sender_id="u1", chat_id="42", content="hello",
        )

        captured: list[tuple[str, str]] = []

        original_get_or_create = session_manager.get_or_create

        def spy_get_or_create(key: str, agent_id: str = "default") -> Session:
            captured.append((key, agent_id))
            return original_get_or_create(key, agent_id=agent_id)

        session_manager.get_or_create = spy_get_or_create  # type: ignore[method-assign]

        loop._run_agent_loop = AsyncMock(  # type: ignore[method-assign]
            return_value=("hi there", [], [])
        )
        loop.context.build_messages = MagicMock(return_value=[])  # type: ignore[method-assign]

        await loop._process_message(msg)

        assert any(agent_id == "default" for _, agent_id in captured)

    @pytest.mark.asyncio
    async def test_message_with_agent_id_uses_that_agent(self, tmp_path: Path) -> None:
        loop, bus, session_manager = _make_loop(tmp_path)

        msg = InboundMessage(
            channel="telegram", sender_id="u1", chat_id="42",
            content="hello", agent_id="silpi",
        )

        captured: list[tuple[str, str]] = []

        original_get_or_create = session_manager.get_or_create

        def spy_get_or_create(key: str, agent_id: str = "default") -> Session:
            captured.append((key, agent_id))
            return original_get_or_create(key, agent_id=agent_id)

        session_manager.get_or_create = spy_get_or_create  # type: ignore[method-assign]

        loop._run_agent_loop = AsyncMock(  # type: ignore[method-assign]
            return_value=("hi there", [], [])
        )
        loop.context.build_messages = MagicMock(return_value=[])  # type: ignore[method-assign]

        await loop._process_message(msg)

        assert any(agent_id == "silpi" for _, agent_id in captured)

    @pytest.mark.asyncio
    async def test_different_agents_same_key_get_separate_sessions(self, tmp_path: Path) -> None:
        """Two messages for the same channel:chat_id but different agents must
        resolve to distinct Session objects with no shared history."""
        loop, bus, session_manager = _make_loop(tmp_path)

        for agent in ("default", "silpi"):
            msg = InboundMessage(
                channel="telegram", sender_id="u1", chat_id="99",
                content="hello", agent_id=agent if agent != "default" else None,
            )
            loop._run_agent_loop = AsyncMock(  # type: ignore[method-assign]
                return_value=(f"reply from {agent}", [], [])
            )
            loop.context.build_messages = MagicMock(return_value=[])  # type: ignore[method-assign]
            await loop._process_message(msg)

        key = "telegram:99"
        s_default = session_manager.get_or_create(key, agent_id="default")
        s_silpi = session_manager.get_or_create(key, agent_id="silpi")

        assert s_default is not s_silpi


# ---------------------------------------------------------------------------
# SessionManager.invalidate respects agent_id
# ---------------------------------------------------------------------------

class TestInvalidate:
    def test_invalidate_removes_only_the_specified_agent_entry(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        key = "telegram:7"

        s_default = mgr.get_or_create(key, agent_id="default")
        s_silpi = mgr.get_or_create(key, agent_id="silpi")

        mgr.invalidate(key, agent_id="default")

        # default entry gone from cache; silpi entry still present
        assert ("default", key) not in mgr._cache
        assert ("silpi", key) in mgr._cache
        assert mgr._cache[("silpi", key)] is s_silpi

    def test_invalidate_default_agent_when_no_agent_id_given(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        key = "telegram:8"

        mgr.get_or_create(key)  # default
        mgr.invalidate(key)  # should evict ("default", key)

        assert ("default", key) not in mgr._cache

"""Tests for per-agent workspace isolation and context loading (Bantu-dez)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.utils.helpers import _get_writable_workspace, get_agent_workspace


# ---------------------------------------------------------------------------
# get_agent_workspace
# ---------------------------------------------------------------------------


class TestGetAgentWorkspace:
    def test_default_agent_returns_bantu_workspace(self, tmp_path: Path) -> None:
        """The default agent's workspace is ~/.bantu/workspace (via get_workspace_path)."""
        expected = tmp_path / ".bantu" / "workspace"
        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            result = get_agent_workspace("default")
        assert result == expected
        assert result.exists()

    def test_specialized_agent_returns_agents_subdir(self, tmp_path: Path) -> None:
        """A specialized agent gets ~/.bantu/agents/<name>/."""
        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            result = get_agent_workspace("silpi")
        assert result == tmp_path / ".bantu" / "agents" / "silpi"
        assert result.exists()

    def test_different_agents_return_different_paths(self, tmp_path: Path) -> None:
        """Two different specialized agents get separate workspace directories."""
        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            path_a = get_agent_workspace("agent-a")
            path_b = get_agent_workspace("agent-b")
        assert path_a != path_b
        assert path_a == tmp_path / ".bantu" / "agents" / "agent-a"
        assert path_b == tmp_path / ".bantu" / "agents" / "agent-b"

    def test_specialized_workspace_is_created(self, tmp_path: Path) -> None:
        """get_agent_workspace creates the directory if it doesn't exist."""
        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            result = get_agent_workspace("new-agent")
        assert result.is_dir()

    def test_specialized_workspace_not_under_default_workspace(self, tmp_path: Path) -> None:
        """Specialized workspace is a sibling of workspace/, not nested inside it."""
        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            default_ws = get_agent_workspace("default")
            specialized_ws = get_agent_workspace("silpi")
        # default is ~/.bantu/workspace; specialized is ~/.bantu/agents/silpi
        assert not str(specialized_ws).startswith(str(default_ws))


# ---------------------------------------------------------------------------
# _get_writable_workspace — default agent (no restriction)
# ---------------------------------------------------------------------------


class TestWriteGuardDefaultAgent:
    def test_default_agent_any_path_allowed(self, tmp_path: Path) -> None:
        """The default agent is never blocked by the write guard."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        # Should not raise
        result = _get_writable_workspace(workspace, "default")
        assert result == workspace

    def test_default_agent_with_nonexistent_path_allowed(self, tmp_path: Path) -> None:
        """Even a non-existent path is accepted for the default agent."""
        workspace = tmp_path / "some" / "arbitrary" / "path"
        result = _get_writable_workspace(workspace, "default")
        assert result == workspace


# ---------------------------------------------------------------------------
# _get_writable_workspace — specialized agent within correct boundary
# ---------------------------------------------------------------------------


class TestWriteGuardWithinBoundary:
    def test_correct_workspace_allowed(self, tmp_path: Path) -> None:
        """Workspace inside ~/.bantu/agents/<name>/ is permitted."""
        workspace = tmp_path / ".bantu" / "agents" / "silpi"
        workspace.mkdir(parents=True)

        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            result = _get_writable_workspace(workspace, "silpi")

        assert result == workspace

    def test_subdirectory_of_agent_workspace_allowed(self, tmp_path: Path) -> None:
        """A subdirectory (e.g. sessions/) inside the agent workspace is permitted."""
        workspace = tmp_path / ".bantu" / "agents" / "silpi" / "sessions"
        workspace.mkdir(parents=True)

        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            result = _get_writable_workspace(workspace, "silpi")

        assert result == workspace


# ---------------------------------------------------------------------------
# _get_writable_workspace — write guard raises for default workspace
# ---------------------------------------------------------------------------


class TestWriteGuardRaisesForDefaultWorkspace:
    def test_specialized_agent_cannot_use_default_workspace(self, tmp_path: Path) -> None:
        """A specialized agent must not write to the default agent's ~/.bantu/workspace."""
        default_workspace = tmp_path / ".bantu" / "workspace"
        default_workspace.mkdir(parents=True)

        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            with pytest.raises(PermissionError, match="silpi"):
                _get_writable_workspace(default_workspace, "silpi")

    def test_error_message_mentions_agent_name(self, tmp_path: Path) -> None:
        """The PermissionError message includes the agent name for clarity."""
        bad_workspace = tmp_path / ".bantu" / "workspace"
        bad_workspace.mkdir(parents=True)

        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            with pytest.raises(PermissionError) as exc_info:
                _get_writable_workspace(bad_workspace, "viharapala")

        assert "viharapala" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _get_writable_workspace — write guard raises for sibling agent directory
# ---------------------------------------------------------------------------


class TestWriteGuardRaisesForSiblingAgent:
    def test_specialized_agent_cannot_write_to_sibling(self, tmp_path: Path) -> None:
        """Agent 'foo' must not write into agent 'bar's workspace directory."""
        sibling_workspace = tmp_path / ".bantu" / "agents" / "bar"
        sibling_workspace.mkdir(parents=True)

        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            with pytest.raises(PermissionError, match="foo"):
                _get_writable_workspace(sibling_workspace, "foo")

    def test_sibling_error_includes_both_names(self, tmp_path: Path) -> None:
        """The PermissionError message should mention the requesting agent."""
        sibling_workspace = tmp_path / ".bantu" / "agents" / "viharapala"
        sibling_workspace.mkdir(parents=True)

        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            with pytest.raises(PermissionError) as exc_info:
                _get_writable_workspace(sibling_workspace, "silpi")

        assert "silpi" in str(exc_info.value)

    def test_parent_of_agent_dir_is_rejected(self, tmp_path: Path) -> None:
        """The ~/.bantu/agents/ parent itself is outside any single agent's boundary."""
        agents_parent = tmp_path / ".bantu" / "agents"
        agents_parent.mkdir(parents=True)

        with patch("nanobot.utils.helpers.Path.home", return_value=tmp_path):
            with pytest.raises(PermissionError):
                _get_writable_workspace(agents_parent, "silpi")


# ---------------------------------------------------------------------------
# ContextBuilder — agent_assets_dir loads bootstrap files from correct source
# ---------------------------------------------------------------------------


class TestContextBuilderAgentAssetsDir:
    def test_no_agent_assets_dir_loads_from_workspace(self, tmp_path: Path) -> None:
        """Without agent_assets_dir, SOUL.md is loaded from workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "SOUL.md").write_text("workspace soul", encoding="utf-8")

        cb = ContextBuilder(workspace=workspace)
        result = cb._load_bootstrap_files()

        assert "workspace soul" in result

    def test_agent_assets_dir_overrides_workspace_for_bootstrap(self, tmp_path: Path) -> None:
        """When agent_assets_dir is set, SOUL.md is loaded from there, not workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "SOUL.md").write_text("workspace soul", encoding="utf-8")

        assets_dir = tmp_path / "agents" / "silpi"
        assets_dir.mkdir(parents=True)
        (assets_dir / "SOUL.md").write_text("silpi soul", encoding="utf-8")

        cb = ContextBuilder(workspace=workspace, agent_assets_dir=assets_dir)
        result = cb._load_bootstrap_files()

        assert "silpi soul" in result
        assert "workspace soul" not in result

    def test_agent_assets_dir_loads_agents_md(self, tmp_path: Path) -> None:
        """When agent_assets_dir is set, AGENTS.md is loaded from agent_assets_dir."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "AGENTS.md").write_text("workspace agents", encoding="utf-8")

        assets_dir = tmp_path / "agents" / "silpi"
        assets_dir.mkdir(parents=True)
        (assets_dir / "AGENTS.md").write_text("silpi agents doc", encoding="utf-8")

        cb = ContextBuilder(workspace=workspace, agent_assets_dir=assets_dir)
        result = cb._load_bootstrap_files()

        assert "silpi agents doc" in result
        assert "workspace agents" not in result

    def test_agent_assets_dir_only_loads_present_files(self, tmp_path: Path) -> None:
        """Files absent from agent_assets_dir are simply skipped (no error)."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)

        assets_dir = tmp_path / "agents" / "silpi"
        assets_dir.mkdir(parents=True)
        # Only SOUL.md present; AGENTS.md, USER.md etc. are absent
        (assets_dir / "SOUL.md").write_text("silpi soul only", encoding="utf-8")

        cb = ContextBuilder(workspace=workspace, agent_assets_dir=assets_dir)
        result = cb._load_bootstrap_files()

        assert "silpi soul only" in result

    def test_agent_assets_dir_none_is_same_as_not_provided(self, tmp_path: Path) -> None:
        """Explicitly passing agent_assets_dir=None behaves like the default (workspace)."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "SOUL.md").write_text("default soul", encoding="utf-8")

        cb_default = ContextBuilder(workspace=workspace)
        cb_explicit_none = ContextBuilder(workspace=workspace, agent_assets_dir=None)

        assert cb_default._load_bootstrap_files() == cb_explicit_none._load_bootstrap_files()

    def test_memory_still_uses_workspace_not_assets_dir(self, tmp_path: Path) -> None:
        """MemoryStore inside ContextBuilder always targets the workspace, not agent_assets_dir."""
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True)

        assets_dir = tmp_path / "agents" / "silpi"
        assets_dir.mkdir(parents=True)

        cb = ContextBuilder(workspace=workspace, agent_assets_dir=assets_dir)

        # Memory files should be in workspace/memory/, not assets_dir/memory/
        assert cb.memory.memory_dir == workspace / "memory"


# ---------------------------------------------------------------------------
# SessionManager and MemoryStore workspace isolation (integration-style)
# ---------------------------------------------------------------------------


class TestSessionManagerIsolation:
    def test_sessions_stored_in_agent_workspace(self, tmp_path: Path) -> None:
        """SessionManager with specialized workspace stores files under agents/<name>/sessions/."""
        from nanobot.session.manager import SessionManager

        agent_ws = tmp_path / ".bantu" / "agents" / "silpi"
        agent_ws.mkdir(parents=True)

        mgr = SessionManager(workspace=agent_ws)
        session = mgr.get_or_create("telegram:42", agent_id="silpi")
        session.add_message("user", "hello")
        mgr.save(session)

        session_path = mgr._get_session_path("telegram:42", "silpi")
        assert session_path.is_relative_to(agent_ws / "sessions")
        assert session_path.exists()

    def test_specialized_agent_sessions_not_in_default_workspace(self, tmp_path: Path) -> None:
        """Sessions stored via agent workspace cannot end up in the default workspace."""
        from nanobot.session.manager import SessionManager

        default_ws = tmp_path / ".bantu" / "workspace"
        default_ws.mkdir(parents=True)
        agent_ws = tmp_path / ".bantu" / "agents" / "silpi"
        agent_ws.mkdir(parents=True)

        # Default manager
        mgr_default = SessionManager(workspace=default_ws)
        # Specialized manager
        mgr_silpi = SessionManager(workspace=agent_ws)

        s_default = mgr_default.get_or_create("telegram:1")
        s_silpi = mgr_silpi.get_or_create("telegram:1", agent_id="silpi")

        s_default.add_message("user", "default message")
        s_silpi.add_message("user", "silpi message")

        mgr_default.save(s_default)
        mgr_silpi.save(s_silpi)

        # The two session files must be in different directories
        path_default = mgr_default._get_session_path("telegram:1", "default")
        path_silpi = mgr_silpi._get_session_path("telegram:1", "silpi")

        assert path_default.is_relative_to(default_ws / "sessions")
        assert path_silpi.is_relative_to(agent_ws / "sessions")


class TestMemoryStoreIsolation:
    def test_memory_stored_in_agent_workspace(self, tmp_path: Path) -> None:
        """MemoryStore with specialized workspace writes MEMORY.md to agents/<name>/memory/."""
        from nanobot.agent.memory import MemoryStore

        agent_ws = tmp_path / ".bantu" / "agents" / "silpi"
        agent_ws.mkdir(parents=True)

        store = MemoryStore(workspace=agent_ws)
        store.write_long_term("test memory content")

        expected = agent_ws / "memory" / "MEMORY.md"
        assert expected.exists()
        assert expected.read_text(encoding="utf-8") == "test memory content"

    def test_specialized_memory_isolated_from_default(self, tmp_path: Path) -> None:
        """Writing to a specialized agent's MemoryStore does not touch the default workspace."""
        from nanobot.agent.memory import MemoryStore

        default_ws = tmp_path / ".bantu" / "workspace"
        default_ws.mkdir(parents=True)
        agent_ws = tmp_path / ".bantu" / "agents" / "silpi"
        agent_ws.mkdir(parents=True)

        # Write to silpi's memory
        silpi_store = MemoryStore(workspace=agent_ws)
        silpi_store.write_long_term("silpi memory")

        # Default workspace memory file must not exist
        default_memory = default_ws / "memory" / "MEMORY.md"
        assert not default_memory.exists()

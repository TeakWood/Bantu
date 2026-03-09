"""Tests for AgentRegistry — filesystem discovery of the agents/ folder (Bantu-02b)."""

from __future__ import annotations

from pathlib import Path

from nanobot.agent.registry import AgentMeta, AgentRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(agents_dir: Path, name: str, files: list[str] | None = None) -> Path:
    """Create an agent sub-directory inside *agents_dir* with optional files."""
    agent_dir = agents_dir / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    for filename in files or []:
        (agent_dir / filename).write_text(f"# {filename}\n")
    return agent_dir


# ---------------------------------------------------------------------------
# AgentMeta
# ---------------------------------------------------------------------------


class TestAgentMeta:
    def test_fields(self, tmp_path: Path) -> None:
        agent_dir = tmp_path / "silpi"
        agent_dir.mkdir()
        meta = AgentMeta(
            name="silpi",
            path=agent_dir,
            identity_files=["SOUL.md", "AGENTS.md"],
        )
        assert meta.name == "silpi"
        assert meta.path == agent_dir
        assert meta.identity_files == ["SOUL.md", "AGENTS.md"]

    def test_default_identity_files_empty(self, tmp_path: Path) -> None:
        meta = AgentMeta(name="silpi", path=tmp_path)
        assert meta.identity_files == []


# ---------------------------------------------------------------------------
# AgentRegistry.discover()
# ---------------------------------------------------------------------------


class TestDiscover:
    def test_discover_multiple_agents(self, tmp_path: Path) -> None:
        """discover() returns one AgentMeta per immediate subdirectory."""
        _make_agent(tmp_path, "silpi", ["SOUL.md", "AGENTS.md"])
        _make_agent(tmp_path, "viharapala", ["SOUL.md"])

        registry = AgentRegistry(tmp_path)
        agents = registry.discover()

        names = {a.name for a in agents}
        assert names == {"silpi", "viharapala"}

    def test_discover_identity_files_populated(self, tmp_path: Path) -> None:
        """identity_files lists the files present in each agent folder."""
        _make_agent(tmp_path, "silpi", ["SOUL.md", "AGENTS.md"])

        registry = AgentRegistry(tmp_path)
        (agent,) = registry.discover()

        assert sorted(agent.identity_files) == ["AGENTS.md", "SOUL.md"]

    def test_discover_agent_with_no_files(self, tmp_path: Path) -> None:
        """An agent directory with no files is still included."""
        _make_agent(tmp_path, "empty-agent")

        registry = AgentRegistry(tmp_path)
        (agent,) = registry.discover()

        assert agent.name == "empty-agent"
        assert agent.identity_files == []

    def test_discover_path_is_absolute(self, tmp_path: Path) -> None:
        """AgentMeta.path is always an absolute path."""
        _make_agent(tmp_path, "silpi")

        registry = AgentRegistry(tmp_path)
        (agent,) = registry.discover()

        assert agent.path.is_absolute()

    def test_discover_ignores_files_in_agents_dir(self, tmp_path: Path) -> None:
        """Non-directory entries at the top level of agents_dir are skipped."""
        _make_agent(tmp_path, "silpi")
        (tmp_path / "README.md").write_text("docs\n")
        (tmp_path / "run.py").write_text("# script\n")

        registry = AgentRegistry(tmp_path)
        agents = registry.discover()

        assert [a.name for a in agents] == ["silpi"]

    def test_discover_empty_agents_dir(self, tmp_path: Path) -> None:
        """discover() returns [] when agents_dir exists but contains no subdirs."""
        registry = AgentRegistry(tmp_path)
        assert registry.discover() == []

    def test_discover_agents_dir_does_not_exist(self, tmp_path: Path) -> None:
        """discover() returns [] when agents_dir does not exist at all."""
        registry = AgentRegistry(tmp_path / "no-such-dir")
        assert registry.discover() == []

    def test_discover_rebuilds_index_on_each_call(self, tmp_path: Path) -> None:
        """Calling discover() a second time reflects the current filesystem state."""
        _make_agent(tmp_path, "silpi")
        registry = AgentRegistry(tmp_path)
        first = registry.discover()
        assert len(first) == 1

        _make_agent(tmp_path, "viharapala")
        second = registry.discover()
        assert len(second) == 2

    def test_discover_only_scans_immediate_subdirs(self, tmp_path: Path) -> None:
        """Nested subdirectories are not included as separate agents."""
        agent_dir = _make_agent(tmp_path, "silpi")
        # nested sub-subdirectory — should not appear as a separate agent
        (agent_dir / "skills").mkdir()

        registry = AgentRegistry(tmp_path)
        agents = registry.discover()

        assert len(agents) == 1
        assert agents[0].name == "silpi"


# ---------------------------------------------------------------------------
# AgentRegistry.get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_existing_agent(self, tmp_path: Path) -> None:
        """get() returns the correct AgentMeta for a known agent name."""
        _make_agent(tmp_path, "silpi", ["SOUL.md"])
        registry = AgentRegistry(tmp_path)
        registry.discover()

        meta = registry.get("silpi")

        assert meta is not None
        assert meta.name == "silpi"

    def test_get_missing_agent_returns_none(self, tmp_path: Path) -> None:
        """get() returns None for a name that does not exist."""
        _make_agent(tmp_path, "silpi")
        registry = AgentRegistry(tmp_path)
        registry.discover()

        assert registry.get("unknown-agent") is None

    def test_get_triggers_discover_if_not_yet_called(self, tmp_path: Path) -> None:
        """get() works without an explicit prior call to discover()."""
        _make_agent(tmp_path, "silpi")
        registry = AgentRegistry(tmp_path)

        meta = registry.get("silpi")

        assert meta is not None
        assert meta.name == "silpi"

    def test_get_on_empty_dir_returns_none(self, tmp_path: Path) -> None:
        """get() returns None when agents_dir is empty."""
        registry = AgentRegistry(tmp_path)
        assert registry.get("silpi") is None

    def test_get_on_nonexistent_dir_returns_none(self, tmp_path: Path) -> None:
        """get() returns None when agents_dir does not exist."""
        registry = AgentRegistry(tmp_path / "no-such-dir")
        assert registry.get("silpi") is None


# ---------------------------------------------------------------------------
# AgentRegistry.list()
# ---------------------------------------------------------------------------


class TestList:
    def test_list_sorted_by_name(self, tmp_path: Path) -> None:
        """list() returns agents in ascending alphabetical order by name."""
        _make_agent(tmp_path, "zebra")
        _make_agent(tmp_path, "alpha")
        _make_agent(tmp_path, "mango")

        registry = AgentRegistry(tmp_path)
        registry.discover()

        names = [a.name for a in registry.list()]
        assert names == ["alpha", "mango", "zebra"]

    def test_list_triggers_discover_if_not_yet_called(self, tmp_path: Path) -> None:
        """list() works without an explicit prior call to discover()."""
        _make_agent(tmp_path, "silpi")
        registry = AgentRegistry(tmp_path)

        agents = registry.list()

        assert len(agents) == 1
        assert agents[0].name == "silpi"

    def test_list_empty_agents_dir(self, tmp_path: Path) -> None:
        """list() returns [] for an empty agents_dir."""
        registry = AgentRegistry(tmp_path)
        assert registry.list() == []

    def test_list_agents_dir_does_not_exist(self, tmp_path: Path) -> None:
        """list() returns [] when agents_dir does not exist."""
        registry = AgentRegistry(tmp_path / "no-such-dir")
        assert registry.list() == []

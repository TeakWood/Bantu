"""Agent registry — runtime name tracking and filesystem discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# AgentRuntimeRegistry — in-memory registry of named agents
# ---------------------------------------------------------------------------


class AgentRuntimeRegistry:
    """In-memory registry of named agents.

    Agents can be registered at startup (e.g. when an AgentLoop is created
    for a named specialist agent).  The admin API uses this registry to
    enumerate known agents; the ``overrides`` dict in ``AgentsConfig`` is the
    persistent counterpart that survives restarts.
    """

    def __init__(self) -> None:
        self._names: set[str] = set()

    def register(self, name: str) -> None:
        """Register *name* as a known agent."""
        self._names.add(name)

    def unregister(self, name: str) -> None:
        """Remove *name* from the registry (no-op if absent)."""
        self._names.discard(name)

    def names(self) -> list[str]:
        """Return sorted list of registered agent names."""
        return sorted(self._names)

    def has(self, name: str) -> bool:
        """Return ``True`` if *name* is registered."""
        return name in self._names


# Module-level default instance — used by admin routes when no test override is injected.
AGENT_REGISTRY: AgentRuntimeRegistry = AgentRuntimeRegistry()


# ---------------------------------------------------------------------------
# AgentMeta / AgentRegistry — filesystem discovery of agents/ folder
# ---------------------------------------------------------------------------


@dataclass
class AgentMeta:
    """Metadata for a single discovered agent.

    Attributes:
        name:           Folder name (= agent identifier).
        path:           Absolute path to the agent's directory.
        identity_files: Names of files found in the directory
                        (e.g. ``["SOUL.md", "AGENTS.md"]``).
    """

    name: str
    path: Path
    identity_files: list[str] = field(default_factory=list)


class AgentRegistry:
    """Scans the ``agents/`` directory and builds a searchable index.

    Usage::

        registry = AgentRegistry(Path("agents"))
        agents = registry.discover()
        silpi = registry.get("silpi")

    The default/personal-assistant agent lives at the project root and is
    **not** represented in this registry — only subdirectories of
    *agents_dir* are indexed.
    """

    def __init__(self, agents_dir: Path) -> None:
        self._agents_dir = agents_dir
        self._index: dict[str, AgentMeta] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> list[AgentMeta]:
        """Scan *agents_dir* for immediate subdirectories and return their metadata.

        Each immediate subdirectory becomes one :class:`AgentMeta` entry.
        An entry is always included if its directory exists, even when no
        identity files are present yet.

        Returns an empty list when *agents_dir* does not exist.
        """
        self._index = {}

        if not self._agents_dir.exists():
            return []

        for entry in sorted(self._agents_dir.iterdir()):
            if not entry.is_dir():
                continue
            identity_files = sorted(f.name for f in entry.iterdir() if f.is_file())
            meta = AgentMeta(
                name=entry.name,
                path=entry.resolve(),
                identity_files=identity_files,
            )
            self._index[meta.name] = meta

        return list(self._index.values())

    def get(self, name: str) -> AgentMeta | None:
        """Return the :class:`AgentMeta` for *name*, or ``None`` if not found.

        Triggers :meth:`discover` on first call if the index has not yet
        been built.
        """
        if self._index is None:
            self.discover()
        return self._index.get(name)  # type: ignore[union-attr]

    def list(self) -> list[AgentMeta]:
        """Return all discovered agents sorted by name.

        Triggers :meth:`discover` on first call if the index has not yet
        been built.
        """
        if self._index is None:
            self.discover()
        return sorted(self._index.values(), key=lambda m: m.name)  # type: ignore[union-attr]

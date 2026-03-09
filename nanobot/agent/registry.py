"""Agent registry — tracks named agents known to this process."""

from __future__ import annotations


class AgentRegistry:
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
AGENT_REGISTRY: AgentRegistry = AgentRegistry()

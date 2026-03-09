"""Utility functions for nanobot."""

import re
from datetime import datetime
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path() -> Path:
    """~/.bantu data directory."""
    return ensure_dir(Path.home() / ".bantu")


def get_workspace_path(workspace: str | None = None) -> Path:
    """Resolve and ensure workspace path. Defaults to ~/.bantu/workspace."""
    path = Path(workspace).expanduser() if workspace else Path.home() / ".bantu" / "workspace"
    return ensure_dir(path)


def timestamp() -> str:
    """Current ISO timestamp."""
    return datetime.now().isoformat()


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')

#: Sentinel name for the built-in default agent.
DEFAULT_AGENT_NAME: str = "default"


def safe_filename(name: str) -> str:
    """Replace unsafe path characters with underscores."""
    return _UNSAFE_CHARS.sub("_", name).strip()


def get_agent_workspace(agent_name: str) -> Path:
    """Return the runtime workspace for an agent.

    The default agent uses the shared ``~/.bantu/workspace`` path (same as
    :func:`get_workspace_path`).  Every other agent gets its own isolated
    directory at ``~/.bantu/agents/<agent_name>/``.

    Raises :exc:`ValueError` if *agent_name* contains path-traversal
    characters (``/``, ``\\``, or ``..`` segments).
    """
    if agent_name != DEFAULT_AGENT_NAME:
        if safe_filename(agent_name) != agent_name or ".." in agent_name.split("/"):
            raise ValueError(
                f"agent_name contains unsafe characters: {agent_name!r}"
            )
        return ensure_dir(Path.home() / ".bantu" / "agents" / agent_name)
    return get_workspace_path()


def _get_writable_workspace(workspace: Path, agent_name: str) -> Path:
    """Validate that *workspace* is within the boundary for *agent_name*.

    For specialized agents (``agent_name != DEFAULT_AGENT_NAME``) the workspace
    must resolve inside ``~/.bantu/agents/<agent_name>/``.  Writing to the
    default agent's workspace or to a sibling agent's directory is forbidden.

    The default agent has no such restriction.

    Returns *workspace* unchanged on success; raises :exc:`PermissionError`
    on violation.
    """
    if agent_name == DEFAULT_AGENT_NAME:
        return workspace

    expected = (Path.home() / ".bantu" / "agents" / agent_name).resolve()
    resolved = workspace.resolve()

    try:
        resolved.relative_to(expected)
    except ValueError:
        raise PermissionError(
            f"Specialized agent '{agent_name}' cannot write to '{workspace}': "
            f"path resolves outside its allowed workspace '{expected}'."
        )

    return workspace


def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """Sync bundled templates to workspace. Only creates missing files."""
    from importlib.resources import files as pkg_files
    try:
        tpl = pkg_files("nanobot") / "templates"
    except Exception:
        return []
    if not tpl.is_dir():
        return []

    added: list[str] = []

    def _write(src, dest: Path):
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8") if src else "", encoding="utf-8")
        added.append(str(dest.relative_to(workspace)))

    for item in tpl.iterdir():
        if item.name.endswith(".md"):
            _write(item, workspace / item.name)
    _write(tpl / "memory" / "MEMORY.md", workspace / "memory" / "MEMORY.md")
    _write(None, workspace / "memory" / "HISTORY.md")
    (workspace / "skills").mkdir(exist_ok=True)

    if added and not silent:
        from rich.console import Console
        for name in added:
            Console().print(f"  [dim]Created {name}[/dim]")
    return added

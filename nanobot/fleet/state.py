"""Durable supervisor state, shared between the three fleet commands.

`fleet status` and `fleet stop` run in a different shell from the supervisor,
so the supervisor publishes what it knows to a file.  That file lives under the
user's nanobot directory rather than beside the fleet file: instances are
denied the fleet file, and the supervisor's own bookkeeping has no reason to
sit in a directory the contract protects.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

STATE_DIR_ENV = "NANOBOT_FLEET_STATE_DIR"
STATE_FILE_NAME = "supervisor.json"


def fleet_state_root() -> Path:
    """Root directory holding one run directory per fleet file."""
    override = os.environ.get(STATE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".nanobot" / "fleet"


def fleet_run_dir(fleet_path: Path) -> Path:
    """Run directory for the fleet declared at *fleet_path*."""
    digest = hashlib.sha256(str(fleet_path).encode("utf-8")).hexdigest()[:16]
    return fleet_state_root() / digest


def state_file(fleet_path: Path) -> Path:
    return fleet_run_dir(fleet_path) / STATE_FILE_NAME


def write_state(fleet_path: Path, payload: dict[str, Any]) -> None:
    """Publish *payload* atomically so a concurrent reader never sees a partial file."""
    path = state_file(fleet_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_state(fleet_path: Path) -> dict[str, Any] | None:
    """Read the published state, or ``None`` when no supervisor has run."""
    path = state_file(fleet_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def clear_state(fleet_path: Path) -> None:
    with suppress(OSError):
        state_file(fleet_path).unlink()

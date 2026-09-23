"""Shared fixtures for the fleet supervisor tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from nanobot.fleet.state import STATE_DIR_ENV


@pytest.fixture(autouse=True)
def _isolate_fleet_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep supervisor state out of the user's real ~/.nanobot/fleet."""
    monkeypatch.setenv(STATE_DIR_ENV, str(tmp_path / "fleet-state"))
    yield


@pytest.fixture
def make_entry(tmp_path: Path) -> Callable[..., dict[str, Any]]:
    """Build a fleet entry backed by a real config file on disk."""

    def _make(
        name: str,
        *,
        workspace: str | Path | None = None,
        config_data: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        config_dir = tmp_path / f".nanobot-{name}"
        config_dir.mkdir(parents=True, exist_ok=True)
        resolved_workspace = Path(workspace) if workspace else config_dir / "workspace"
        data = (
            config_data
            if config_data is not None
            else {"agents": {"defaults": {"workspace": str(resolved_workspace)}}}
        )
        config_path = config_dir / "config.json"
        config_path.write_text(json.dumps(data), encoding="utf-8")
        entry: dict[str, Any] = {
            "config": str(config_path),
            "mode": "gateway",
            "memoryLimitMb": 256,
        }
        entry.update(overrides)
        return entry

    return _make


@pytest.fixture
def write_fleet(tmp_path: Path) -> Callable[..., Path]:
    """Write a fleet file and return its path."""

    def _write(instances: dict[str, Any], *, file_name: str = "fleet.json") -> Path:
        path = tmp_path / file_name
        path.write_text(json.dumps({"instances": instances}), encoding="utf-8")
        return path

    return _write

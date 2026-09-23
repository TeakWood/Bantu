"""Fleet file parsing and validation.

A fleet file declares the instances a supervisor runs.  Everything the
supervisor needs to confine an instance is derived here: the instance's own
config directory (its config file's parent, where sessions and runtime data
live) and its workspace (whatever that config resolves).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INSTANCE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
INSTANCE_MODES = ("gateway", "serve")
DEFAULT_WORKSPACE = "~/.nanobot/workspace"


class FleetConfigError(ValueError):
    """Raised when a fleet file cannot be used as written."""


@dataclass(frozen=True)
class FleetInstance:
    """One supervised instance, with every path already resolved."""

    name: str
    config_path: Path
    config_dir: Path
    workspace: Path
    mode: str
    memory_limit_mb: int
    env: tuple[str, ...]

    @property
    def owned_paths(self) -> tuple[Path, Path]:
        """The paths this instance may read and write."""
        return (self.workspace, self.config_dir)


@dataclass(frozen=True)
class Fleet:
    """A validated fleet file."""

    path: Path
    instances: tuple[FleetInstance, ...]

    def instance(self, name: str) -> FleetInstance | None:
        return next((entry for entry in self.instances if entry.name == name), None)

    def others(self, name: str) -> tuple[FleetInstance, ...]:
        """Every instance except *name*."""
        return tuple(entry for entry in self.instances if entry.name != name)


def expand_path(value: str) -> Path:
    """Expand ``~`` and environment references, then resolve symlinks.

    Seatbelt matches real paths, so the supervisor stores resolved paths and
    compares resolved paths when it looks for overlapping instances.
    """
    return Path(os.path.expandvars(str(value))).expanduser().resolve(strict=False)


def load_fleet(path: str | Path) -> Fleet:
    """Read and validate the fleet file at *path*."""
    fleet_path = expand_path(str(path))
    try:
        raw = fleet_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FleetConfigError(f"fleet file cannot be read: {fleet_path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FleetConfigError(f"fleet file is not valid JSON: {fleet_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FleetConfigError(f"fleet file must contain a JSON object: {fleet_path}")
    entries = data.get("instances")
    if not isinstance(entries, dict) or not entries:
        raise FleetConfigError(
            f"fleet file must declare a non-empty 'instances' object: {fleet_path}"
        )
    instances = tuple(
        _parse_instance(str(name), entry) for name, entry in entries.items()
    )
    _reject_overlaps(instances)
    return Fleet(path=fleet_path, instances=instances)


def _parse_instance(name: str, entry: Any) -> FleetInstance:
    if not INSTANCE_NAME_PATTERN.match(name):
        raise FleetConfigError(
            f"instance name {name!r} must match {INSTANCE_NAME_PATTERN.pattern}"
        )
    if not isinstance(entry, dict):
        raise FleetConfigError(f"instance {name!r} must be a JSON object")

    config_value = entry.get("config")
    if not isinstance(config_value, str) or not config_value.strip():
        raise FleetConfigError(f"instance {name!r} must set 'config' to a config file path")
    config_path = expand_path(config_value)
    if not config_path.is_file():
        raise FleetConfigError(f"instance {name!r}: config file not found: {config_path}")

    mode = entry.get("mode", "gateway")
    if mode not in INSTANCE_MODES:
        raise FleetConfigError(
            f"instance {name!r}: 'mode' must be one of {', '.join(INSTANCE_MODES)}, got {mode!r}"
        )

    memory_limit = entry.get("memoryLimitMb")
    # bool is an int subclass, and `true` in JSON is never a memory cap.
    if isinstance(memory_limit, bool) or not isinstance(memory_limit, int) or memory_limit <= 0:
        raise FleetConfigError(
            f"instance {name!r}: 'memoryLimitMb' must be a positive integer, got {memory_limit!r}"
        )

    return FleetInstance(
        name=name,
        config_path=config_path,
        config_dir=config_path.parent,
        workspace=_workspace_for(name, config_path),
        mode=str(mode),
        memory_limit_mb=memory_limit,
        env=_parse_env(name, entry.get("env")),
    )


def _parse_env(name: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise FleetConfigError(f"instance {name!r}: 'env' must be a list of variable names")
    out: list[str] = []
    for item in value:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(item, str) or not ENV_NAME_PATTERN.match(item):
            raise FleetConfigError(
                f"instance {name!r}: {item!r} is not a valid environment variable name"
            )
        if item not in out:
            out.append(item)
    return tuple(out)


def _workspace_for(name: str, config_path: Path) -> Path:
    """Resolve the workspace the instance's own config selects."""
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FleetConfigError(
            f"instance {name!r}: config file cannot be read: {config_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FleetConfigError(
            f"instance {name!r}: config file is not valid JSON: {config_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise FleetConfigError(
            f"instance {name!r}: config file must contain a JSON object: {config_path}"
        )
    agents = data.get("agents")
    defaults = agents.get("defaults") if isinstance(agents, dict) else None
    configured = defaults.get("workspace") if isinstance(defaults, dict) else None
    if configured is not None and not isinstance(configured, str):
        raise FleetConfigError(
            f"instance {name!r}: agents.defaults.workspace must be a string: {config_path}"
        )
    return expand_path(configured or DEFAULT_WORKSPACE)


def _reject_overlaps(instances: tuple[FleetInstance, ...]) -> None:
    """Refuse a fleet where one instance could reach another's files by path.

    Nesting *within* one instance is normal — the default layout puts the
    workspace inside the config directory — so only cross-instance pairs are
    compared.
    """
    owned = [
        (entry, label, path)
        for entry in instances
        for label, path in (("workspace", entry.workspace), ("config directory", entry.config_dir))
    ]
    for index, (first, first_label, first_path) in enumerate(owned):
        for second, second_label, second_path in owned[index + 1:]:
            if first.name == second.name:
                continue
            if first_path == second_path:
                raise FleetConfigError(
                    f"instances {first.name!r} and {second.name!r} overlap: "
                    f"{first.name}'s {first_label} {str(first_path)!r} is also "
                    f"{second.name}'s {second_label}"
                )
            inner, outer = None, None
            if first_path.is_relative_to(second_path):
                inner, outer = (first, first_label, first_path), (second, second_label, second_path)
            elif second_path.is_relative_to(first_path):
                inner, outer = (second, second_label, second_path), (first, first_label, first_path)
            if inner is not None and outer is not None:
                raise FleetConfigError(
                    f"instances {first.name!r} and {second.name!r} overlap: "
                    f"{inner[0].name}'s {inner[1]} {str(inner[2])!r} is inside "
                    f"{outer[0].name}'s {outer[1]} {str(outer[2])!r}"
                )

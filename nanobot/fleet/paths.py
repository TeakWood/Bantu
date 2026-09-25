"""Resolve one instance's config file to the pair of directories it owns.

The fleet supervisor has to know, for every instance, exactly two directories:
the instance's config dir (where nanobot keeps that instance's sessions, media,
cron and logs) and its agent workspace. Those two paths become Seatbelt allow
rules for the instance and deny rules for every one of its peers, so resolving
them is a security operation, not a convenience.

Three properties follow from that, and none of them hold for nanobot's ordinary
path helpers:

*No side effects.* ``nanobot.config.paths.get_data_dir`` and
``get_workspace_path`` both call ``ensure_dir``, so merely asking either of them
for a path creates the directory. The supervisor asks about every instance in the
fleet, including instances it is about to refuse to start, and it must be able to
validate a fleet document without laying down a single directory. Neither helper
may be used here.

*Canonical paths.* ``Config.workspace_path`` (``config/schema.py``) calls only
``.expanduser()``, never ``.resolve()``. A path that traverses a symlink
therefore stays non-canonical, and a Seatbelt rule written from a non-canonical
path silently matches nothing — the kernel evaluates the real path, so the deny
rule that was supposed to wall an instance off from its peers just never fires
and the OS reports no error. Everything returned here is resolved.

*Fail closed on a missing file.* ``load_config`` on a path that does not exist
returns ``Config()`` defaults (``config/loader.py``), which means a typo in one
instance's config path would hand back ``~/.nanobot/workspace`` — the *same*
workspace every other typo would produce, and the one the unsupervised install
already uses. A missing file raises here instead.

Reading the JSON directly rather than through ``load_config`` is what buys the
last property, and it also keeps this module free of the config schema: nothing
here imports pydantic, the agent tools whose configs the schema references, or
the SSRF whitelist that ``load_config`` reconfigures globally as a side effect.
The one cost is the duplicated workspace default below.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple, cast

# Mirrors ``AgentDefaults.workspace`` in nanobot/config/schema.py. Duplicated
# rather than imported so this module stays off the config schema; a test asserts
# the two stay equal.
DEFAULT_WORKSPACE = "~/.nanobot/workspace"

# Distinguishes an absent key from one explicitly set to JSON ``null``. Only the
# absent key may fall back to a default; a null is a declaration that the schema
# would reject, and treating it as "unset" would confine the instance somewhere
# its config never named.
_MISSING: object = object()


class FleetPathError(Exception):
    """An instance config file cannot be turned into a pair of paths.

    Carries the offending path so the fleet's validation layer can name the
    instance it belongs to when it refuses to start the fleet.
    """

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


class InstancePaths(NamedTuple):
    """The two canonical, absolute directories one instance owns."""

    config_dir: Path
    workspace: Path


class InstanceConfig(NamedTuple):
    """One instance's config file: where it is, and what it says.

    ``path`` is expanded but deliberately *not* resolved, because the config
    dir is derived from the parent as written — see :func:`instance_paths`.
    """

    path: Path
    data: Mapping[str, object]


def read_instance_config(config_path: str | Path) -> InstanceConfig:
    """Read one instance config file into memory, without interpreting it.

    Split out of :func:`resolve_instance_paths` for the same reason
    ``nanobot.fleet.config`` splits ``load_fleet_file`` from ``parse_fleet_file``:
    the fleet's validation layer needs more out of a config file than the two
    directories — the raw workspace value, so it can tell a relative declaration
    from an absolute one, and the declared ports — and reading a
    confinement-critical file twice would let the two reads disagree.

    Raises:
        FleetPathError: the file is missing, is not a file, is unreadable, is
            not valid UTF-8, is not JSON, or is not a JSON object.
    """
    path = Path(config_path).expanduser()
    if not path.exists():
        # A dangling symlink lands here too, which is right: the config it was
        # meant to point at is not readable and nothing about the instance's
        # confinement can be derived from it.
        raise FleetPathError(path, "instance config file does not exist")
    if not path.is_file():
        raise FleetPathError(path, "instance config path is not a file")
    return InstanceConfig(path=path, data=_config_object(path))


def instance_paths(config: InstanceConfig) -> InstancePaths:
    """Derive the ``(config_dir, workspace)`` pair from an already-read config.

    Creates nothing — the workspace commonly does not exist yet, and a fleet that
    fails validation must leave no directories behind — and never re-opens the
    config file; canonicalising does stat the paths, because that is what
    following a symlink means.

    ``config_dir`` is the parent of the config file, matching how nanobot itself
    derives an instance's data dir (``get_data_dir`` is ``get_config_path()``'s
    parent). The parent as *written* is what nanobot will use, so that is what
    gets canonicalised — resolving the config file itself first would relocate
    the data dir whenever the file is a symlink into a shared directory.

    Both results are run through ``expanduser()`` then ``resolve(strict=False)``,
    so a relative input — either the config path or a relative
    ``agents.defaults.workspace`` — becomes absolute against the current working
    directory. ``nanobot.fleet.validate`` refuses relative paths declared in a
    fleet document before they reach here; this function is deliberately not the
    place that decision is made.

    Raises:
        FleetPathError: the config declares a workspace that is not a non-empty
            string.
    """
    return InstancePaths(
        config_dir=_canonical(config.path.parent),
        workspace=_canonical(Path(declared_workspace(config))),
    )


def declared_workspace(config: InstanceConfig) -> str:
    """Return ``agents.defaults.workspace`` exactly as written, or the default.

    The *raw* value, before expansion or resolution, because that is the only
    form in which "the operator declared a relative workspace" is still visible;
    :func:`instance_paths` has already turned it into something absolute.

    Raises:
        FleetPathError: the key is present but is not a non-empty string, or a
            section on the way to it is not a JSON object.
    """
    return _workspace_value(config.path, config.data)


def resolve_instance_paths(config_path: str | Path) -> InstancePaths:
    """Resolve ``config_path`` to its canonical ``(config_dir, workspace)`` pair.

    Raises:
        FleetPathError: the file is missing, unreadable, not JSON, not a JSON
            object, or declares a workspace that is not a non-empty string.
    """
    return instance_paths(read_instance_config(config_path))


def _canonical(path: Path) -> Path:
    """Expand ``~`` and resolve to a real absolute path, existing or not."""
    return path.expanduser().resolve(strict=False)


def _config_object(path: Path) -> Mapping[str, object]:
    """Read one config file as a JSON object, raising on anything else."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = cast(object, json.load(handle))
    except json.JSONDecodeError as exc:
        raise FleetPathError(
            path, f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise FleetPathError(path, "file is not valid UTF-8") from exc
    except OSError as exc:
        detail = exc.strerror or type(exc).__name__
        raise FleetPathError(path, f"unable to read file: {detail}") from exc
    if not isinstance(data, dict):
        raise FleetPathError(
            path, f"top level must be a JSON object, found {type(data).__name__}"
        )
    return cast(Mapping[str, object], data)


def _workspace_value(path: Path, data: Mapping[str, object]) -> str:
    """Return the declared ``agents.defaults.workspace``, or the schema default.

    An absent key falls back to the default exactly as the schema does. A key
    that is present but unusable — wrong type, or blank — raises instead: an
    empty string would resolve to the working directory and confine nothing, and
    a fleet is better off refusing to start than starting mis-confined.
    """
    agents = _child_object(path, data, "agents")
    defaults = _child_object(path, agents, "defaults")
    value = _lookup(defaults, "workspace")
    if value is _MISSING:
        return DEFAULT_WORKSPACE
    if not isinstance(value, str) or not value.strip():
        raise FleetPathError(
            path, "agents.defaults.workspace must be a non-empty string when present"
        )
    return value


def _lookup(mapping: Mapping[str, object] | None, *names: str) -> object:
    """Return the value of the first key spelling present, else ``_MISSING``.

    Callers pass every accepted spelling of a key because the config schema's
    ``Base`` accepts both camelCase and snake_case (``nanobot/config_base.py``).
    Every key this module reads happens to be a single word today, so the two
    spellings coincide; the parameter exists so a future multi-word key cannot be
    read here in only one of the two forms the config file is allowed to use.
    """
    if mapping is None:
        return _MISSING
    for name in names:
        if name in mapping:
            return mapping[name]
    return _MISSING


def _child_object(
    path: Path,
    mapping: Mapping[str, object] | None,
    *names: str,
) -> Mapping[str, object] | None:
    """Return a nested JSON object, ``None`` if absent, raising if mistyped."""
    value = _lookup(mapping, *names)
    if value is _MISSING:
        return None
    if not isinstance(value, dict):
        raise FleetPathError(
            path, f"{names[0]} must be a JSON object, found {type(value).__name__}"
        )
    return cast(Mapping[str, object], value)

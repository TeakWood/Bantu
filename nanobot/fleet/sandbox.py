"""Operating-system confinement for supervised fleet instances.

The shell tool's Seatbelt backend confines one command to one workspace and
denies everything else.  A whole nanobot instance cannot be confined that way:
it loads the Python runtime, site-packages, model and font caches, keychain
helpers and whatever the operator's own tools reach for, and an allow-list
breaks on the first unlisted dependency.

So the fleet profile inverts the default.  What it guarantees is exactly what
the fleet contract asks for: an instance reaches its own workspace and config
directory, and cannot read, write, create or list another instance's workspace
or config directory, or the fleet file.  The rest of the user's home directory
is deliberately left alone.

The profile applies to the instance process and every process it starts, the
shell tool's commands included, because Seatbelt confinement is inherited
across ``exec`` and ``fork``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from nanobot.agent.tools.sandbox import sbpl_quote
from nanobot.fleet.config import FleetInstance

SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def instance_profile(
    instance: FleetInstance,
    *,
    others: Sequence[FleetInstance],
    fleet_path: Path,
) -> str:
    """Build the SBPL profile confining *instance*.

    The allow for the instance's own paths is stated *before* the denials
    because SBPL applies the last matching rule: the fleet file may legitimately
    live inside an instance's own config directory, and it must stay unreadable
    even there.
    """
    rules = [
        "(version 1)",
        "(allow default)",
        "(allow file-read* file-write* "
        + " ".join(
            f"(subpath {sbpl_quote(str(path))})"
            for path in _unique(instance.owned_paths)
        )
        + ")",
    ]
    denied = _unique(
        path
        for other in others
        if other.name != instance.name
        for path in other.owned_paths
    )
    if denied:
        rules.append(
            "(deny file-read* file-write* "
            + " ".join(f"(subpath {sbpl_quote(str(path))})" for path in denied)
            + ")"
        )
    rules.append(f"(deny file-read* file-write* (literal {sbpl_quote(str(fleet_path))}))")
    return "\n".join(rules)


def sandbox_command(argv: Sequence[str], *, profile: str) -> list[str]:
    """Wrap *argv* so the operating system applies *profile* to it."""
    return [SANDBOX_EXEC, "-p", profile, *argv]


def confinement_available(platform: str | None = None) -> bool:
    """Return whether this host can enforce fleet confinement."""
    return (platform or sys.platform) == "darwin" and Path(SANDBOX_EXEC).exists()


def instance_command(instance: FleetInstance, *, python_executable: str) -> list[str]:
    """Build the nanobot invocation for *instance* against its own config."""
    argv = [python_executable, "-m", "nanobot", instance.mode, "--config", str(instance.config_path)]
    if instance.mode == "gateway":
        # `nanobot gateway` can detach; the supervisor owns the process, so the
        # child must stay attached for its exit to be observable.
        argv.append("--foreground")
    return argv


def _unique(paths: Iterable[Path]) -> tuple[Path, ...]:
    out: list[Path] = []
    for path in paths:
        if path not in out:
            out.append(path)
    return tuple(out)

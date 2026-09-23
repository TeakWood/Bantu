"""Per-instance environments: a minimal base plus that instance's own names."""

from __future__ import annotations

from collections.abc import Mapping

from nanobot.fleet.config import FleetInstance

# Enough for a process to start, find its home and write temporary files.
# Everything else — API keys, broker tokens, medical-record credentials —
# reaches an instance only by being named in its own fleet entry.
BASE_ENV_VARS = ("PATH", "HOME", "LANG", "TMPDIR")


def instance_environment(
    instance: FleetInstance,
    source: Mapping[str, str],
) -> dict[str, str]:
    """Build the environment for *instance* out of the supervisor's *source*."""
    env = {name: source[name] for name in BASE_ENV_VARS if name in source}
    for name in instance.env:
        if name in source:
            env[name] = source[name]
    return env

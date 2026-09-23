"""Credential isolation between instances."""

from __future__ import annotations

from pathlib import Path

from nanobot.fleet.config import FleetInstance
from nanobot.fleet.environment import BASE_ENV_VARS, instance_environment

SUPERVISOR_ENV = {
    "PATH": "/usr/bin",
    "HOME": "/Users/operator",
    "LANG": "en_US.UTF-8",
    "TMPDIR": "/tmp/operator",
    "SECRET_A": "a-value",
    "SECRET_B": "b-value",
    "SENTINEL": "leak",
}


def _instance(name: str, env: tuple[str, ...]) -> FleetInstance:
    config_dir = Path("/tmp") / f".nanobot-{name}"
    return FleetInstance(
        name=name,
        config_path=config_dir / "config.json",
        config_dir=config_dir,
        workspace=config_dir / "workspace",
        mode="gateway",
        memory_limit_mb=256,
        env=env,
    )


class TestInstanceEnvironment:
    def test_passes_only_the_instances_own_names(self) -> None:
        env = instance_environment(_instance("a", ("SECRET_A",)), SUPERVISOR_ENV)

        assert env["SECRET_A"] == "a-value"
        assert "SECRET_B" not in env
        assert "SENTINEL" not in env

    def test_includes_the_minimal_base(self) -> None:
        env = instance_environment(_instance("a", ()), SUPERVISOR_ENV)

        assert set(env) == set(BASE_ENV_VARS)

    def test_skips_names_the_supervisor_does_not_have(self) -> None:
        env = instance_environment(_instance("a", ("SECRET_A", "ABSENT")), SUPERVISOR_ENV)

        assert "ABSENT" not in env
        assert env["SECRET_A"] == "a-value"

    def test_an_empty_supervisor_environment_yields_an_empty_environment(self) -> None:
        assert instance_environment(_instance("a", ("SECRET_A",)), {}) == {}

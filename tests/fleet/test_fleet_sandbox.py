"""Filesystem confinement and the command each instance is launched with."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from nanobot.fleet.config import FleetInstance, load_fleet
from nanobot.fleet.sandbox import (
    SANDBOX_EXEC,
    confinement_available,
    instance_command,
    instance_profile,
    sandbox_command,
)


def _instance(name: str, tmp_path: Path, **overrides: Any) -> FleetInstance:
    config_dir = tmp_path / f".nanobot-{name}"
    defaults: dict[str, Any] = {
        "name": name,
        "config_path": config_dir / "config.json",
        "config_dir": config_dir,
        "workspace": config_dir / "workspace",
        "mode": "gateway",
        "memory_limit_mb": 256,
        "env": (),
    }
    defaults.update(overrides)
    return FleetInstance(**defaults)


class TestInstanceProfile:
    def test_allows_own_paths_and_denies_every_other_instance(self, tmp_path: Path) -> None:
        research = _instance("research", tmp_path)
        trader = _instance("trader", tmp_path)
        fleet_file = tmp_path / "fleet.json"

        profile = instance_profile(
            research, others=(research, trader), fleet_path=fleet_file
        )

        assert f'(allow file-read* file-write* (subpath "{research.workspace}")' in profile
        assert f'(subpath "{research.config_dir}")' in profile
        assert f'(deny file-read* file-write* (subpath "{trader.workspace}")' in profile
        assert f'(subpath "{trader.config_dir}")' in profile

    def test_never_denies_the_instance_its_own_paths(self, tmp_path: Path) -> None:
        research = _instance("research", tmp_path)

        profile = instance_profile(
            research, others=(research,), fleet_path=tmp_path / "fleet.json"
        )

        deny_lines = [line for line in profile.splitlines() if line.startswith("(deny")]
        assert not any(str(research.workspace) in line for line in deny_lines)
        assert not any(str(research.config_dir) in line for line in deny_lines)

    def test_denies_the_fleet_file_after_allowing_own_paths(self, tmp_path: Path) -> None:
        # A fleet file inside the instance's own config directory must still be
        # unreadable, which only holds if the deny comes last.
        research = _instance("research", tmp_path)
        fleet_file = research.config_dir / "fleet.json"

        rules = instance_profile(
            research, others=(research,), fleet_path=fleet_file
        ).splitlines()

        allow_index = next(i for i, rule in enumerate(rules) if rule.startswith("(allow file-"))
        deny_index = next(i for i, rule in enumerate(rules) if str(fleet_file) in rule)
        assert deny_index > allow_index
        assert rules[deny_index].startswith("(deny file-read* file-write* (literal")

    def test_starts_from_allow_default(self, tmp_path: Path) -> None:
        profile = instance_profile(
            _instance("research", tmp_path),
            others=(),
            fleet_path=tmp_path / "fleet.json",
        )

        assert profile.splitlines()[:2] == ["(version 1)", "(allow default)"]

    def test_quotes_paths_containing_sbpl_metacharacters(self, tmp_path: Path) -> None:
        odd = tmp_path / 'we"ird'
        research = _instance("research", tmp_path, workspace=odd)

        profile = instance_profile(
            research, others=(), fleet_path=tmp_path / "fleet.json"
        )

        assert r'we\"ird' in profile

    def test_real_fleet_file_denies_each_peer(
        self,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        fleet = load_fleet(
            write_fleet({"research": make_entry("research"), "trader": make_entry("trader")})
        )
        research = fleet.instance("research")
        trader = fleet.instance("trader")
        assert research is not None and trader is not None

        profile = instance_profile(
            research, others=fleet.instances, fleet_path=fleet.path
        )

        assert str(trader.workspace) in profile
        assert str(fleet.path) in profile


class TestLaunchCommand:
    def test_gateway_mode_stays_attached(self, tmp_path: Path) -> None:
        instance = _instance("research", tmp_path)

        argv = instance_command(instance, python_executable="/usr/bin/python3")

        assert argv[:5] == ["/usr/bin/python3", "-m", "nanobot", "gateway", "--config"]
        assert argv[5] == str(instance.config_path)
        assert argv[-1] == "--foreground"

    def test_serve_mode_runs_the_openai_api(self, tmp_path: Path) -> None:
        instance = _instance("research", tmp_path, mode="serve")

        argv = instance_command(instance, python_executable="/usr/bin/python3")

        assert argv == [
            "/usr/bin/python3",
            "-m",
            "nanobot",
            "serve",
            "--config",
            str(instance.config_path),
        ]

    def test_sandbox_command_prefixes_the_profile(self) -> None:
        wrapped = sandbox_command(["echo", "hi"], profile="(version 1)")

        assert wrapped == [SANDBOX_EXEC, "-p", "(version 1)", "echo", "hi"]


class TestConfinementAvailability:
    def test_requires_macos(self) -> None:
        assert confinement_available(platform="linux") is False

    def test_macos_depends_on_sandbox_exec(self) -> None:
        assert confinement_available(platform="darwin") is Path(SANDBOX_EXEC).exists()

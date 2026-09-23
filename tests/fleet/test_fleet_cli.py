"""The `nanobot fleet` contact points."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from nanobot.fleet import cli as fleet_cli
from nanobot.fleet.config import load_fleet
from nanobot.fleet.supervisor import STATE_RUNNING


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app() -> Any:
    return fleet_cli.create_fleet_app()


class TestFleetStart:
    def test_refuses_overlapping_instances_without_starting_anything(
        self,
        tmp_path: Path,
        runner: CliRunner,
        app: Any,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        started: list[Any] = []
        monkeypatch.setattr(
            fleet_cli, "Supervisor", lambda *args, **kwargs: started.append(args) or object()
        )
        outer = tmp_path / "shared"
        fleet_path = write_fleet(
            {
                "research": make_entry("research", workspace=outer),
                "trader": make_entry("trader", workspace=outer / "nested"),
            }
        )

        result = runner.invoke(app, ["start", "--fleet", str(fleet_path)])

        assert result.exit_code == 1
        assert "research" in result.output and "trader" in result.output
        assert started == []

    def test_reports_an_unreadable_fleet_file(
        self,
        tmp_path: Path,
        runner: CliRunner,
        app: Any,
    ) -> None:
        result = runner.invoke(app, ["start", "--fleet", str(tmp_path / "absent.json")])

        assert result.exit_code == 1
        assert "cannot be read" in result.output

    def test_refuses_a_host_without_seatbelt(
        self,
        runner: CliRunner,
        app: Any,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(fleet_cli, "confinement_available", lambda: False)
        fleet_path = write_fleet({"research": make_entry("research")})

        result = runner.invoke(app, ["start", "--fleet", str(fleet_path)])

        assert result.exit_code == 1
        assert "sandbox-exec" in result.output

    def test_supervises_a_validated_fleet(
        self,
        runner: CliRunner,
        app: Any,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runs: list[str] = []

        class StubSupervisor:
            def __init__(self, fleet: Any) -> None:
                self.fleet = fleet

            def run(self) -> None:
                runs.append(str(self.fleet.path))

        monkeypatch.setattr(fleet_cli, "confinement_available", lambda: True)
        monkeypatch.setattr(fleet_cli, "Supervisor", StubSupervisor)
        fleet_path = write_fleet({"research": make_entry("research")})

        result = runner.invoke(app, ["start", "--fleet", str(fleet_path)])

        assert result.exit_code == 0
        assert runs == [str(load_fleet(fleet_path).path)]


    def test_reports_an_instance_that_could_not_be_launched(
        self,
        runner: CliRunner,
        app: Any,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FailingSupervisor:
            def __init__(self, fleet: Any) -> None:
                pass

            def run(self) -> None:
                raise OSError("no such executable")

        monkeypatch.setattr(fleet_cli, "confinement_available", lambda: True)
        monkeypatch.setattr(fleet_cli, "Supervisor", FailingSupervisor)
        fleet_path = write_fleet({"research": make_entry("research")})

        result = runner.invoke(app, ["start", "--fleet", str(fleet_path)])

        assert result.exit_code == 1
        assert "no such executable" in result.output


class TestFleetStatus:
    def _publish(
        self,
        fleet_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[dict[str, Any]]:
        entries = [
            {
                "name": "research",
                "pid": 4100,
                "state": STATE_RUNNING,
                "exit_reason": None,
                "workspace": "/tmp/ws-research",
                "config_dir": "/tmp/cfg-research",
                "memory_limit_mb": 1024,
            },
            {
                "name": "trader",
                "pid": 4101,
                "state": "exited",
                "exit_reason": "memory",
                "workspace": "/tmp/ws-trader",
                "config_dir": "/tmp/cfg-trader",
                "memory_limit_mb": 512,
            },
        ]
        monkeypatch.setattr(fleet_cli, "fleet_status", lambda _path: entries)
        return entries

    def test_json_output_is_one_object_per_instance(
        self,
        tmp_path: Path,
        runner: CliRunner,
        app: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        expected = self._publish(tmp_path / "fleet.json", monkeypatch)

        result = runner.invoke(app, ["status", "--fleet", str(tmp_path / "fleet.json"), "--json"])

        assert result.exit_code == 0
        assert json.loads(result.output) == expected

    def test_table_output_lists_every_instance(
        self,
        tmp_path: Path,
        runner: CliRunner,
        app: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._publish(tmp_path / "fleet.json", monkeypatch)

        result = runner.invoke(app, ["status", "--fleet", str(tmp_path / "fleet.json")])

        assert result.exit_code == 0
        assert "research" in result.output and "trader" in result.output

    def test_errors_when_no_supervisor_has_run(
        self,
        tmp_path: Path,
        runner: CliRunner,
        app: Any,
    ) -> None:
        result = runner.invoke(app, ["status", "--fleet", str(tmp_path / "fleet.json"), "--json"])

        assert result.exit_code == 1
        assert "no fleet supervisor" in result.output


class TestFleetStop:
    def test_reports_a_clean_stop(
        self,
        tmp_path: Path,
        runner: CliRunner,
        app: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            fleet_cli,
            "stop_fleet",
            lambda _path: fleet_cli.StopResult(True, "fleet stopped"),
        )

        result = runner.invoke(app, ["stop", "--fleet", str(tmp_path / "fleet.json")])

        assert result.exit_code == 0
        assert "Fleet stopped" in result.output

    def test_is_quietly_idempotent_without_a_supervisor(
        self,
        tmp_path: Path,
        runner: CliRunner,
        app: Any,
    ) -> None:
        result = runner.invoke(app, ["stop", "--fleet", str(tmp_path / "fleet.json")])

        assert result.exit_code == 0
        assert "no fleet supervisor state" in result.output

    def test_fails_when_instance_processes_survive(
        self,
        tmp_path: Path,
        runner: CliRunner,
        app: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            fleet_cli,
            "stop_fleet",
            lambda _path: fleet_cli.StopResult(False, "instance processes still alive", (4100,)),
        )

        result = runner.invoke(app, ["stop", "--fleet", str(tmp_path / "fleet.json")])

        assert result.exit_code == 1
        assert "still alive" in result.output

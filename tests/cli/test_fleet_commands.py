"""Tests for ``nanobot fleet start``.

The command's whole contribution is an ordering — validate, prepare, prove, only
then start — so these tests are mostly about what did *not* happen. Each of the
three gates is failed in turn and the assertion is that the next stage was never
reached and that no instance process exists, which is the acceptance criterion
written out literally.

The gates are driven with real fleet documents and real instance configs wherever
the behaviour is platform-independent: validation and profile building are pure
enough to run anywhere, so those tests exercise the production code rather than a
stand-in. The two things that are not — proving confinement, which needs a kernel
with Seatbelt, and the foreground loop, which needs real children — are injected,
so the command's decision logic is tested on every platform instead of skipping
on most of them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import nanobot.cli.fleet as fleet_cli
from nanobot.cli.commands import app as root_app
from nanobot.cli.fleet import fleet_app
from nanobot.cli.process_identity import set_cli_process_identity
from nanobot.fleet.instance import InstanceLaunchError
from nanobot.fleet.probe import ProbeResult
from nanobot.fleet.profile import SeatbeltProfileError
from nanobot.fleet.state import InstanceRecord
from nanobot.fleet.supervisor import FleetPlan

runner = CliRunner()


# ----------------------------------------------------------------------------
# Fixtures on disk
# ----------------------------------------------------------------------------


def write_instance(
    root: Path,
    name: str,
    *,
    port: int,
    workspace: Path | None = None,
) -> Path:
    """Lay down one instance config the way nanobot lays one out by itself.

    ``<root>/<name>/config.json`` with ``<root>/<name>/workspace`` beneath it —
    the default layout, which the fleet must accept. The port is explicit
    because every ``serve`` instance binds ``api.port`` and validation refuses a
    fleet whose instances would collide on it, so a fixture that left them all
    on the default would be refused for a reason no test here is about.
    """
    config_dir = root / name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "api": {"port": port},
                "agents": {
                    "defaults": {
                        "workspace": str(workspace or config_dir / "workspace")
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def write_fleet(root: Path, configs: dict[str, Path]) -> Path:
    """A fleet document naming ``configs``, written beside them."""
    fleet_path = root / "fleet.json"
    fleet_path.write_text(
        json.dumps(
            {
                "instances": {
                    name: {
                        "config": str(path),
                        "mode": "serve",
                        "memoryLimitMb": 512,
                    }
                    for name, path in configs.items()
                }
            }
        ),
        encoding="utf-8",
    )
    return fleet_path


def separable_fleet(root: Path, *names: str) -> Path:
    """A fleet of mutually separable instances in the default layout."""
    return write_fleet(
        root,
        {
            name: write_instance(root, name, port=8901 + index)
            for index, name in enumerate(names)
        },
    )


def start(fleet_path: Path | str) -> object:
    """Invoke ``fleet start`` against ``fleet_path``."""
    return runner.invoke(fleet_app, ["start", "--fleet", str(fleet_path)])


# ----------------------------------------------------------------------------
# Stand-ins for the two stages that need a real kernel
# ----------------------------------------------------------------------------


class Spy:
    """A callable that records its calls and refuses to be reached, by default.

    The default is the point: every gate test asserts that a later stage was
    never entered, and a spy that quietly returned something would let a command
    that ignored a refusal pass three tests at once.
    """

    def __init__(self, result: object = None, *, reachable: bool = False) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.result = result
        self.reachable = reachable

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        if not self.reachable:
            raise AssertionError("this stage must not have been reached")
        return self.result

    @property
    def called(self) -> bool:
        return bool(self.calls)


class StubSupervisor:
    """A started fleet whose foreground loop returns immediately."""

    def __init__(self, records: Sequence[InstanceRecord]) -> None:
        self._records = tuple(records)
        self.runs = 0

    @property
    def records(self) -> tuple[InstanceRecord, ...]:
        return self._records

    def run(self, **_: object) -> tuple[InstanceRecord, ...]:
        self.runs += 1
        return tuple(
            InstanceRecord(
                name=record.name,
                pid=record.pid,
                state="exited",
                exit_reason="exit",
                workspace=record.workspace,
                config_dir=record.config_dir,
                memory_limit_mb=record.memory_limit_mb,
            )
            for record in self._records
        )


def records_for(plan: FleetPlan) -> tuple[InstanceRecord, ...]:
    """One running record per instance in ``plan``, as ``start_fleet`` would."""
    return tuple(
        InstanceRecord(
            name=instance.name,
            pid=4000 + index,
            state="running",
            exit_reason=None,
            workspace=instance.workspace,
            config_dir=instance.config_dir,
            memory_limit_mb=instance.entry.memory_limit_mb,
        )
        for index, instance in enumerate(plan.instances)
    )


def prove_confinement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every probe report a profile that was observed to bind."""

    def probe(
        instances: Sequence[object],
        profiles: object,
        *,
        state_path: Path,
        **_: object,
    ) -> tuple[ProbeResult, ...]:
        return tuple(
            ProbeResult(
                name=getattr(instance, "name"),
                path=state_path,
                confined=True,
                reason="refused under this profile and permitted without it",
            )
            for instance in instances
        )

    monkeypatch.setattr(fleet_cli, "probe_fleet_confinement", probe)


def supervise_with(monkeypatch: pytest.MonkeyPatch) -> list[StubSupervisor]:
    """Replace the launch step with a stub supervisor; return the ones created."""
    created: list[StubSupervisor] = []

    def launch(plan: FleetPlan, **_: object) -> StubSupervisor:
        supervisor = StubSupervisor(records_for(plan))
        created.append(supervisor)
        return supervisor

    monkeypatch.setattr(fleet_cli, "start_fleet", launch)
    return created


# ----------------------------------------------------------------------------
# Gate 1 — validation
# ----------------------------------------------------------------------------


def test_an_overlapping_fleet_is_refused_and_names_both_instances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Epic criterion 7, at the command: overlap refused, nothing prepared."""
    alpha = write_instance(tmp_path, "alpha", port=8901)
    beta = write_instance(
        tmp_path, "beta", port=8902, workspace=tmp_path / "alpha" / "shared"
    )
    fleet_path = write_fleet(tmp_path, {"alpha": alpha, "beta": beta})
    prepare = Spy()
    monkeypatch.setattr(fleet_cli, "prepare_fleet", prepare)

    result = start(fleet_path)

    assert result.exit_code == 1
    assert "alpha" in result.output
    assert "beta" in result.output
    assert not prepare.called


def test_a_validation_failure_creates_nothing_on_disk(tmp_path: Path) -> None:
    """A refused fleet leaves no workspace, no log directory and no state file."""
    alpha = write_instance(tmp_path, "alpha", port=8901)
    beta = write_instance(
        tmp_path, "beta", port=8902, workspace=tmp_path / "alpha" / "shared"
    )
    fleet_path = write_fleet(tmp_path, {"alpha": alpha, "beta": beta})

    assert start(fleet_path).exit_code == 1
    assert not (tmp_path / "alpha" / "workspace").exists()
    assert not (tmp_path / "beta" / "workspace").exists()
    assert list(tmp_path.glob("*fleet*state*")) == []


def test_a_relative_config_path_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fleet_path = write_fleet(tmp_path, {"alpha": Path("relative/config.json")})
    prepare = Spy()
    monkeypatch.setattr(fleet_cli, "prepare_fleet", prepare)

    result = start(fleet_path)

    assert result.exit_code == 1
    assert "alpha" in result.output
    assert not prepare.called


def test_a_malformed_fleet_document_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fleet_path = tmp_path / "fleet.json"
    fleet_path.write_text("{not json", encoding="utf-8")
    prepare = Spy()
    monkeypatch.setattr(fleet_cli, "prepare_fleet", prepare)

    result = start(fleet_path)

    assert result.exit_code == 1
    assert not prepare.called


def test_a_missing_fleet_file_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepare = Spy()
    monkeypatch.setattr(fleet_cli, "prepare_fleet", prepare)

    result = start(tmp_path / "absent.json")

    assert result.exit_code == 1
    assert not prepare.called


# ----------------------------------------------------------------------------
# Gate 2 — preparation
# ----------------------------------------------------------------------------


def test_a_profile_that_cannot_be_built_starts_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A builder refusal is the fleet declining to emit a rule that would not bind."""
    fleet_path = separable_fleet(tmp_path, "alpha", "beta")

    def refuse(*_: object, **__: object) -> FleetPlan:
        raise SeatbeltProfileError("instance alpha: /gone does not exist")

    monkeypatch.setattr(fleet_cli, "prepare_fleet", refuse)
    probe = Spy()
    monkeypatch.setattr(fleet_cli, "probe_fleet_confinement", probe)
    launch = Spy()
    monkeypatch.setattr(fleet_cli, "start_fleet", launch)

    result = start(fleet_path)

    assert result.exit_code == 1
    assert "alpha" in result.output
    assert not probe.called
    assert not launch.called


# ----------------------------------------------------------------------------
# Gate 3 — the confinement self-probe
# ----------------------------------------------------------------------------


def test_an_unproven_profile_names_the_instance_and_starts_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The gate the whole command exists for: unproven means nothing runs."""
    fleet_path = separable_fleet(tmp_path, "alpha", "beta")

    def probe(
        instances: Sequence[object],
        profiles: object,
        *,
        state_path: Path,
        **_: object,
    ) -> tuple[ProbeResult, ...]:
        return tuple(
            ProbeResult(
                name=getattr(instance, "name"),
                path=state_path,
                confined=index != 0,
                reason="sandbox-exec is missing on this host",
            )
            for index, instance in enumerate(instances)
        )

    monkeypatch.setattr(fleet_cli, "probe_fleet_confinement", probe)
    launch = Spy()
    monkeypatch.setattr(fleet_cli, "start_fleet", launch)

    result = start(fleet_path)

    assert result.exit_code == 1
    assert "instances.alpha" in result.output
    assert "sandbox-exec is missing" in result.output
    assert not launch.called


def test_every_unproven_instance_is_named_not_only_the_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An operator fixing a host must not have to rerun once per instance."""
    fleet_path = separable_fleet(tmp_path, "alpha", "beta")

    def probe(
        instances: Sequence[object],
        profiles: object,
        *,
        state_path: Path,
        **_: object,
    ) -> tuple[ProbeResult, ...]:
        return tuple(
            ProbeResult(
                name=getattr(instance, "name"),
                path=state_path,
                confined=False,
                reason="the deny did not bind",
            )
            for instance in instances
        )

    monkeypatch.setattr(fleet_cli, "probe_fleet_confinement", probe)
    monkeypatch.setattr(fleet_cli, "start_fleet", Spy())

    result = start(fleet_path)

    assert result.exit_code == 1
    assert "instances.alpha" in result.output
    assert "instances.beta" in result.output


def test_the_probe_runs_against_the_prepared_profiles_and_state_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The probe must see what the launcher will apply, not a rebuild of it."""
    fleet_path = separable_fleet(tmp_path, "alpha", "beta")
    seen: dict[str, object] = {}

    def probe(
        instances: Sequence[object],
        profiles: object,
        *,
        state_path: Path,
        **_: object,
    ) -> tuple[ProbeResult, ...]:
        seen["profiles"] = profiles
        seen["state_path"] = state_path
        seen["names"] = [getattr(one, "name") for one in instances]
        return tuple(
            ProbeResult(
                name=getattr(instance, "name"),
                path=state_path,
                confined=True,
                reason="bound",
            )
            for instance in instances
        )

    monkeypatch.setattr(fleet_cli, "probe_fleet_confinement", probe)
    supervise_with(monkeypatch)

    assert start(fleet_path).exit_code == 0
    assert seen["names"] == ["alpha", "beta"]
    profiles = seen["profiles"]
    assert isinstance(profiles, dict)
    assert sorted(profiles) == ["alpha", "beta"]
    assert Path(str(seen["state_path"])).is_file()


# ----------------------------------------------------------------------------
# The happy path
# ----------------------------------------------------------------------------


def test_a_proven_fleet_starts_every_instance_and_runs_in_the_foreground(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fleet_path = separable_fleet(tmp_path, "alpha", "beta")
    prove_confinement(monkeypatch)
    created = supervise_with(monkeypatch)

    result = start(fleet_path)

    assert result.exit_code == 0
    assert len(created) == 1
    assert created[0].runs == 1
    assert "Started alpha" in result.output
    assert "Started beta" in result.output
    assert "alpha: exited (exit)" in result.output


def test_preparation_creates_the_workspaces_and_the_state_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The real ``prepare_fleet`` runs before the probe, as the ordering requires."""
    fleet_path = separable_fleet(tmp_path, "alpha", "beta")
    prove_confinement(monkeypatch)
    supervise_with(monkeypatch)

    assert start(fleet_path).exit_code == 0
    assert (tmp_path / "alpha" / "workspace").is_dir()
    assert (tmp_path / "beta" / "workspace").is_dir()


def test_a_launch_failure_exits_non_zero_and_names_the_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fleet_path = separable_fleet(tmp_path, "alpha", "beta")
    prove_confinement(monkeypatch)

    def refuse(*_: object, **__: object) -> object:
        raise InstanceLaunchError("beta", "the interpreter could not be executed")

    monkeypatch.setattr(fleet_cli, "start_fleet", refuse)

    result = start(fleet_path)

    assert result.exit_code == 1
    assert "beta" in result.output


def test_the_fleet_path_is_expanded_before_anything_reads_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``--fleet ~/fleet.json`` must not be looked for under a literal ``~``."""
    fleet_path = separable_fleet(tmp_path, "alpha")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    prove_confinement(monkeypatch)
    supervise_with(monkeypatch)

    assert runner.invoke(fleet_app, ["start", "--fleet", "~/fleet.json"]).exit_code == 0
    assert fleet_path.is_file()


def test_the_fleet_option_is_required() -> None:
    assert runner.invoke(fleet_app, ["start"]).exit_code != 0


# ----------------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------------


def test_the_fleet_group_is_mounted_on_the_root_command() -> None:
    command = typer.main.get_command(root_app)
    assert "fleet" in getattr(command, "commands", {})


def test_the_root_command_exposes_fleet_start() -> None:
    result = runner.invoke(root_app, ["fleet", "--help"])
    assert result.exit_code == 0
    assert "start" in result.output


def test_the_supervisor_process_is_named_after_its_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreground supervisor must be findable in ``ps`` as ``nanobot-fleet``."""
    titles: list[str] = []
    monkeypatch.setattr("nanobot.cli.process_identity.os.name", "posix")
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", titles.append)

    set_cli_process_identity(["fleet", "start", "--fleet", "/tmp/fleet.json"])

    assert titles == ["nanobot-fleet"]

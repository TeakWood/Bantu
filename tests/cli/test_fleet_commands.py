"""Tests for ``nanobot fleet start`` and ``nanobot fleet stop``.

``start``'s whole contribution is an ordering — validate, prepare, prove, only
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

``stop``'s tests are the other way round: the termination policy has its own
suite in ``tests/fleet/test_fleet_stop.py``, driven over real and faked process
tables, so what is left here is the command's own three decisions — where the
state file is, that the fleet document is never consulted, and that a survivor
is never reported as a successful stop.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import suppress
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import nanobot.cli.fleet as fleet_cli
from nanobot.cli.commands import app as root_app
from nanobot.cli.fleet import fleet_app
from nanobot.cli.process_identity import set_cli_process_identity
from nanobot.fleet.instance import InstanceLaunchError, instance_identity
from nanobot.fleet.memory import process_parent_pid
from nanobot.fleet.probe import ProbeResult
from nanobot.fleet.profile import SeatbeltProfileError
from nanobot.fleet.state import (
    RECORD_FIELDS,
    InstanceRecord,
    fleet_state_path,
    write_fleet_state,
)
from nanobot.fleet.stop import FleetStopReport
from nanobot.fleet.supervisor import FleetPlan, prepare_fleet
from nanobot.fleet.validate import validate_fleet_file

#: The subprocess tests run the real ``python -m nanobot`` from here, so its
#: ``pyproject.toml`` is found and a coverage-measuring child would at least use
#: this repository's rules — see :func:`child_environment` for why one must not.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

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
# ``fleet status``
# ----------------------------------------------------------------------------


def status(fleet_path: Path | str, *extra: str, **kwargs: object) -> object:
    """Invoke ``fleet status`` against ``fleet_path``."""
    return runner.invoke(
        fleet_app,
        ["status", "--fleet", str(fleet_path), *extra],
        **kwargs,  # type: ignore[arg-type]
    )


def lay_down_state(
    tmp_path: Path,
    *records: InstanceRecord,
    fleet_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write ``records`` where a supervisor for ``fleet_path`` would put them.

    Through the production writer, so what these tests read back is a file that
    really came from :func:`~nanobot.fleet.state.write_fleet_state` — a
    hand-written document could satisfy the reader while diverging from anything
    the supervisor can produce.
    """
    path = fleet_path or (tmp_path / "fleet.json")
    state_path = fleet_state_path(path)
    write_fleet_state(records, path=state_path)
    return path, state_path


def running(
    tmp_path: Path,
    name: str,
    *,
    pid: int,
    limit: int = 512,
    identity: dict[str, str | int | None] | None = None,
) -> InstanceRecord:
    """A record the supervisor would have written when ``name`` started."""
    return InstanceRecord(
        name=name,
        pid=pid,
        state="running",
        exit_reason=None,
        workspace=tmp_path / name / "workspace",
        config_dir=tmp_path / name,
        memory_limit_mb=limit,
        identity=dict(identity or {}),
    )


def status_json(fleet_path: Path | str, **kwargs: object) -> list[dict[str, object]]:
    """The parsed ``--json`` document, insisting the command succeeded first."""
    result = status(fleet_path, "--json", **kwargs)
    assert getattr(result, "exit_code") == 0, getattr(result, "output")
    parsed = json.loads(getattr(result, "output"))
    assert isinstance(parsed, list)
    return parsed


def test_status_json_publishes_all_seven_fields_with_the_right_types(
    tmp_path: Path,
) -> None:
    """The contract, field by field: the criterion written out literally."""
    fleet_path, _ = lay_down_state(
        tmp_path, running(tmp_path, "alpha", pid=os.getpid(), limit=384)
    )

    reported = status_json(fleet_path)

    assert len(reported) == 1
    one = reported[0]
    assert one["name"] == "alpha"
    assert isinstance(one["name"], str)
    assert one["pid"] == os.getpid()
    assert isinstance(one["pid"], int) and not isinstance(one["pid"], bool)
    assert one["state"] == "running"
    assert one["exit_reason"] is None
    assert one["workspace"] == str(tmp_path / "alpha" / "workspace")
    assert one["config_dir"] == str(tmp_path / "alpha")
    assert one["memory_limit_mb"] == 384
    assert isinstance(one["memory_limit_mb"], int)


def test_status_json_publishes_exactly_those_seven_and_no_identity(
    tmp_path: Path,
) -> None:
    """The identity token is a private format, and its absence is the contract.

    It is also the only extra thing on disk, so this doubles as the assertion
    that the published object is a projection of the record rather than the
    record itself — a future field added to the state file does not silently
    become part of what ``--json`` promises.
    """
    fleet_path, _ = lay_down_state(
        tmp_path,
        running(
            tmp_path,
            "alpha",
            pid=os.getpid(),
            identity=instance_identity(os.getpid()),
        ),
    )

    reported = status_json(fleet_path)

    assert sorted(reported[0]) == sorted(RECORD_FIELDS)
    assert "identity" not in reported[0]
    assert "stable_identity" not in reported[0]


def test_status_json_reports_one_object_per_instance_in_written_order(
    tmp_path: Path,
) -> None:
    fleet_path, _ = lay_down_state(
        tmp_path,
        running(tmp_path, "alpha", pid=os.getpid()),
        running(tmp_path, "beta", pid=os.getpid()),
    )

    reported = status_json(fleet_path)

    assert [one["name"] for one in reported] == ["alpha", "beta"]
    assert all(isinstance(one, dict) for one in reported)


def test_a_recycled_pid_is_reported_as_exited_not_running(tmp_path: Path) -> None:
    """Liveness is reconciled on read, which is why a pid is not an identity.

    The pid here is unambiguously alive — it is this test process — and the
    recorded identity is one no process could hold. Reporting it as running would
    tell an operator a dead instance is serving, and would aim the next thing
    that reads this file at somebody else's process.
    """
    fleet_path, _ = lay_down_state(
        tmp_path,
        running(
            tmp_path,
            "alpha",
            pid=os.getpid(),
            identity={"stable_identity": "darwin:999999:1:2"},
        ),
    )

    reported = status_json(fleet_path)

    assert reported[0]["pid"] == os.getpid()
    assert reported[0]["state"] == "exited"
    # Nothing observed why it stopped, and status must not invent a reason.
    assert reported[0]["exit_reason"] is None


def test_an_exit_reason_the_supervisor_observed_is_published(tmp_path: Path) -> None:
    """``memory`` is a claim only the supervisor can make; status must carry it."""
    fleet_path, _ = lay_down_state(
        tmp_path,
        InstanceRecord(
            name="alpha",
            pid=4242,
            state="exited",
            exit_reason="memory",
            workspace=tmp_path / "alpha" / "workspace",
            config_dir=tmp_path / "alpha",
            memory_limit_mb=512,
        ),
    )

    reported = status_json(fleet_path)

    assert reported[0]["state"] == "exited"
    assert reported[0]["exit_reason"] == "memory"


def test_an_exited_instance_is_a_report_and_not_an_error(tmp_path: Path) -> None:
    """Status never judges: a stopped fleet is something to print, not to refuse."""
    fleet_path, _ = lay_down_state(
        tmp_path,
        InstanceRecord(
            name="alpha",
            pid=4242,
            state="exited",
            exit_reason="signal",
            workspace=tmp_path / "alpha" / "workspace",
            config_dir=tmp_path / "alpha",
            memory_limit_mb=512,
        ),
    )

    assert status(fleet_path, "--json").exit_code == 0
    assert status(fleet_path).exit_code == 0


def test_a_prepared_but_unstarted_fleet_reports_an_empty_array(
    tmp_path: Path,
) -> None:
    """``prepare_fleet`` lays the state file down empty, and that is honest."""
    fleet_path, state_path = lay_down_state(tmp_path)
    assert state_path.is_file()

    assert status_json(fleet_path) == []
    assert "No instance has been started" in status(fleet_path).output


def test_status_never_writes_the_state_file(tmp_path: Path) -> None:
    """A reader in a second shell must not race the supervisor's own writes.

    Asserted on the bytes *and* the modification time, because the dangerous
    edit is not one that changes the contents — it is one that rewrites the file
    with what it just reconciled, which for a running fleet would often produce
    the same bytes while still clobbering whatever the supervisor wrote in
    between.
    """
    fleet_path, state_path = lay_down_state(
        tmp_path,
        running(
            tmp_path,
            "alpha",
            pid=os.getpid(),
            identity={"stable_identity": "darwin:999999:1:2"},
        ),
    )
    before = state_path.read_bytes()
    before_mtime = state_path.stat().st_mtime_ns

    # Reconciliation flips this record to exited, so a writer would have
    # something new to say and would say it.
    assert status_json(fleet_path)[0]["state"] == "exited"

    assert state_path.read_bytes() == before
    assert state_path.stat().st_mtime_ns == before_mtime


def test_the_json_document_is_not_wrapped_to_the_terminal_width(
    tmp_path: Path,
) -> None:
    """The reason the array goes through ``typer.echo`` and not the rich console.

    Rich wraps at the terminal width and interprets ``[`` as markup, so a long
    workspace path in an 40-column window would come back folded across lines or
    with its brackets eaten — and the document a consumer parsed would depend on
    how wide their window happened to be.
    """
    deep = tmp_path / ("a-workspace-path-far-longer-than-forty-columns-" * 3)
    fleet_path, _ = lay_down_state(
        tmp_path,
        InstanceRecord(
            name="alpha",
            pid=os.getpid(),
            state="running",
            exit_reason=None,
            workspace=deep,
            config_dir=tmp_path / "alpha",
            memory_limit_mb=512,
        ),
    )

    reported = status_json(fleet_path, env={"COLUMNS": "40"})

    assert reported[0]["workspace"] == str(deep)


def test_a_missing_state_file_is_refused_rather_than_reported_as_stopped(
    tmp_path: Path,
) -> None:
    """Answering "no instances" here would invite a second fleet on the same ports."""
    result = status(tmp_path / "fleet.json", "--json")

    assert result.exit_code == 1
    assert "state" in result.output.lower()


def test_an_unparseable_state_file_is_refused(tmp_path: Path) -> None:
    fleet_path = tmp_path / "fleet.json"
    fleet_state_path(fleet_path).write_text("{not an array", encoding="utf-8")

    result = status(fleet_path, "--json")

    assert result.exit_code == 1


def test_a_state_file_the_supervisor_could_not_have_written_is_refused(
    tmp_path: Path,
) -> None:
    """A record the writer cannot produce is a file from somewhere else.

    A pid of ``0`` is the sharpest example: signalling it would hit the reader's
    own process group, so it is refused rather than reported, and the refusal
    names the instance it came from.
    """
    fleet_path = tmp_path / "fleet.json"
    payload = {
        "name": "alpha",
        "pid": 0,
        "state": "running",
        "exit_reason": None,
        "workspace": str(tmp_path / "alpha" / "workspace"),
        "config_dir": str(tmp_path / "alpha"),
        "memory_limit_mb": 512,
    }
    fleet_state_path(fleet_path).write_text(json.dumps([payload]), encoding="utf-8")

    result = status(fleet_path, "--json", env={"COLUMNS": "200"})

    assert result.exit_code == 1
    assert "alpha" in result.output
    assert "pid must be a positive integer" in result.output


def test_the_status_table_names_every_instance_and_its_pid(tmp_path: Path) -> None:
    """The human form carries the same facts, and says where they came from."""
    fleet_path, state_path = lay_down_state(
        tmp_path,
        running(tmp_path, "alpha", pid=os.getpid()),
        InstanceRecord(
            name="beta",
            pid=4242,
            state="exited",
            exit_reason="memory",
            workspace=tmp_path / "beta" / "workspace",
            config_dir=tmp_path / "beta",
            memory_limit_mb=512,
        ),
    )

    result = status(fleet_path, env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "running" in result.output
    assert str(os.getpid()) in result.output
    assert "beta" in result.output
    assert "exited" in result.output
    assert "memory" in result.output
    assert state_path.name in result.output


def test_the_status_fleet_option_is_required() -> None:
    assert runner.invoke(fleet_app, ["status"]).exit_code != 0


def test_the_status_fleet_path_is_expanded_before_anything_reads_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    lay_down_state(tmp_path, running(tmp_path, "alpha", pid=os.getpid()))

    result = runner.invoke(fleet_app, ["status", "--fleet", "~/fleet.json", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)[0]["name"] == "alpha"


def test_status_reads_the_state_file_the_supervisor_writes(tmp_path: Path) -> None:
    """``status`` and ``prepare_fleet`` must agree on where the file is.

    Derived through the real ``prepare_fleet`` rather than restated, so a command
    that resolved the fleet path differently would report an unstarted fleet
    instead of an error — the one failure mode of this command that looks like an
    answer.
    """
    fleet_path = separable_fleet(tmp_path, "alpha")
    plan = prepare_fleet(validate_fleet_file(fleet_path), fleet_path=fleet_path)
    write_fleet_state(
        [running(tmp_path, "alpha", pid=os.getpid())], path=plan.state_path
    )

    assert [one["name"] for one in status_json(fleet_path)] == ["alpha"]


def test_a_symlinked_fleet_document_resolves_to_the_same_state_file(
    tmp_path: Path,
) -> None:
    """The one case where the two spellings of a fleet path disagree.

    ``prepare_fleet`` canonicalises the fleet path *including the file itself*
    before deriving the state path, while
    :func:`~nanobot.fleet.state.fleet_state_path` canonicalises only the parent —
    so a command that passed the link through unresolved would look beside the
    link instead of beside its target. It would then find no file and refuse,
    which reads as "this fleet was never started" about a fleet that is running.
    Protects ``stop`` as well: both derive the path through the same helper.
    """
    real = tmp_path / "real"
    real.mkdir()
    fleet_path = separable_fleet(real, "alpha")
    link = tmp_path / "link.json"
    link.symlink_to(fleet_path)

    plan = prepare_fleet(validate_fleet_file(link), fleet_path=link)
    write_fleet_state([running(real, "alpha", pid=os.getpid())], path=plan.state_path)

    assert plan.state_path == fleet_state_path(fleet_path)
    assert [one["name"] for one in status_json(link)] == ["alpha"]


# ----------------------------------------------------------------------------
# ``fleet status`` from a second shell, against real processes
# ----------------------------------------------------------------------------


#: A stand-in for a supervisor: spawns one child in its own session, exactly as
#: ``start_fleet`` does, announces the child's pid and then waits. Its only job
#: is to be a real process that is genuinely the parent of a real "instance", so
#: that "the supervisor's pid is never reported" is a claim about a pid that
#: could have been found rather than about an arbitrary number.
SUPERVISOR_STAND_IN = """
import subprocess, sys, time
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(120)"],
    start_new_session=True,
)
print(child.pid, flush=True)
time.sleep(120)
"""

#: Long enough for a cold ``python -m nanobot`` import on a loaded machine, short
#: enough that a hang is a failure rather than a stalled suite. There is no
#: pytest-timeout in this repo, so every wait here carries its own deadline.
STATUS_TIMEOUT_SECONDS = 90.0


def child_environment(**overrides: str) -> dict[str, str]:
    """This process's environment, cleaned of what a child must not inherit.

    ``NANOBOT_*`` is dropped because :class:`nanobot.config.Config` reads those.
    ``COV_CORE_*`` and ``COVERAGE_*`` go because ``pytest-cov`` asks every child
    to start measuring, and a child that measures without this repository's
    ``omit`` rules quietly changes the whole suite's coverage denominator.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NANOBOT_", "COV_CORE_", "COVERAGE_"))
    }
    return env | overrides


@pytest.fixture
def fleet_stand_in(tmp_path: Path) -> Iterator[tuple[int, int]]:
    """A live "supervisor" and the live "instance" it is the parent of.

    Yields ``(supervisor_pid, instance_pid)``. Both are killed by pid on the way
    out and never by process group: the child is in a session of its own, and a
    group signal aimed at a pid whose ``start_new_session`` had been dropped
    would take the test runner with it.
    """
    supervisor = subprocess.Popen(
        [sys.executable, "-c", SUPERVISOR_STAND_IN],
        stdout=subprocess.PIPE,
        text=True,
        cwd=tmp_path,
        env=child_environment(),
    )
    try:
        assert supervisor.stdout is not None
        line = supervisor.stdout.readline().strip()
        assert line.isdigit(), line
        yield supervisor.pid, int(line)
    finally:
        with suppress(OSError):
            os.kill(int(line), signal.SIGKILL)
        supervisor.kill()
        supervisor.wait(timeout=30)


def run_status(fleet_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    """Run the real command in a separate process, as a second shell would."""
    home = fleet_path.parent / "home"
    home.mkdir(exist_ok=True)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "nanobot",
            "fleet",
            "status",
            "--fleet",
            str(fleet_path),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=STATUS_TIMEOUT_SECONDS,
        cwd=REPOSITORY_ROOT,
        env=child_environment(HOME=str(home), COLUMNS="200"),
    )


def test_status_answers_from_a_separate_process_while_the_fleet_runs(
    tmp_path: Path,
    fleet_stand_in: tuple[int, int],
) -> None:
    """The criterion's second half, through the real command and real processes.

    A supervisor holds the foreground of the shell it was started in, so the only
    honest test of "it works from a separate shell" is a separate process. The
    instance recorded here is alive and carries the identity the supervisor would
    have recorded for it, so a ``running`` answer is reconciled rather than
    copied.
    """
    supervisor_pid, instance_pid = fleet_stand_in
    fleet_path, _ = lay_down_state(
        tmp_path,
        running(
            tmp_path,
            "alpha",
            pid=instance_pid,
            identity=instance_identity(instance_pid),
        ),
    )

    completed = run_status(fleet_path, "--json")

    assert completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"
    reported = json.loads(completed.stdout)
    assert [one["name"] for one in reported] == ["alpha"]
    assert reported[0]["pid"] == instance_pid
    assert reported[0]["state"] == "running"
    assert instance_pid != supervisor_pid


def test_the_supervisors_pid_is_never_reported_as_an_instances(
    tmp_path: Path,
    fleet_stand_in: tuple[int, int],
) -> None:
    """The criterion's sharpest clause, and it is a structural property.

    The instance is a real child of the stand-in supervisor, so that pid *is*
    discoverable from what status reads — :func:`nanobot.fleet.stop._supervisor`
    recovers it exactly that way. Status still must not publish it: the state
    file records instances and nothing else, and the published object is a
    projection of one record. Asserted against the whole document rather than
    against the ``pid`` fields, so an edit that added the supervisor anywhere in
    the output fails here.
    """
    supervisor_pid, instance_pid = fleet_stand_in
    assert process_parent_pid(instance_pid) in (None, supervisor_pid)
    fleet_path, _ = lay_down_state(
        tmp_path,
        running(
            tmp_path,
            "alpha",
            pid=instance_pid,
            identity=instance_identity(instance_pid),
        ),
    )

    completed = run_status(fleet_path, "--json")

    assert completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"
    assert str(supervisor_pid) not in completed.stdout
    assert all(one["pid"] != supervisor_pid for one in json.loads(completed.stdout))


def test_a_separate_process_leaves_the_state_file_exactly_as_it_found_it(
    tmp_path: Path,
    fleet_stand_in: tuple[int, int],
) -> None:
    """The file-not-IPC design only holds if the second shell is a pure reader."""
    _, instance_pid = fleet_stand_in
    fleet_path, state_path = lay_down_state(
        tmp_path,
        running(
            tmp_path,
            "alpha",
            pid=instance_pid,
            identity=instance_identity(instance_pid),
        ),
    )
    before = state_path.read_bytes()
    before_mtime = state_path.stat().st_mtime_ns

    assert run_status(fleet_path, "--json").returncode == 0

    assert state_path.read_bytes() == before
    assert state_path.stat().st_mtime_ns == before_mtime


# ----------------------------------------------------------------------------
# ``fleet stop``
# ----------------------------------------------------------------------------


def stop(fleet_path: Path | str, *extra: str) -> object:
    """Invoke ``fleet stop`` against ``fleet_path``."""
    return runner.invoke(fleet_app, ["stop", "--fleet", str(fleet_path), *extra])


def stopping(
    monkeypatch: pytest.MonkeyPatch,
    report: FleetStopReport,
) -> list[tuple[Path, float]]:
    """Replace the stop policy with one that returns ``report``; record its calls."""
    calls: list[tuple[Path, float]] = []

    def stop_it(state_path: Path, *, grace: float, **_: object) -> FleetStopReport:
        calls.append((state_path, grace))
        return report

    monkeypatch.setattr(fleet_cli, "stop_fleet", stop_it)
    return calls


def report_for(state_path: Path, **overrides: object) -> FleetStopReport:
    """A complete stop of one instance, unless a test says otherwise."""
    fields: dict[str, object] = {
        "state_path": state_path,
        "targeted": ("alpha",),
        "survivors": (),
        "supervisor_pid": 4242,
        "supervisor_stopped": True,
    }
    fields.update(overrides)
    return FleetStopReport(**fields)  # type: ignore[arg-type]


def test_stop_targets_the_state_file_beside_the_fleet_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The one place the state file's location is decided is ``fleet_state_path``."""
    fleet_path = separable_fleet(tmp_path, "alpha")
    expected = fleet_state_path(fleet_path.resolve())
    calls = stopping(monkeypatch, report_for(expected))

    result = stop(fleet_path)

    assert result.exit_code == 0
    assert [path for path, _ in calls] == [expected]
    assert "Stopped alpha" in result.output


def test_stop_does_not_read_the_fleet_document_at_all(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A running fleet must stay stoppable after its declaration goes bad.

    Validating here would be the one failure mode a stop command cannot afford:
    confined processes still running, and the only command that can reach them
    refusing to run.
    """
    fleet_path = tmp_path / "fleet.json"
    fleet_path.write_text("{not json at all", encoding="utf-8")
    expected = fleet_state_path(fleet_path.resolve())
    stopping(monkeypatch, report_for(expected))
    monkeypatch.setattr(fleet_cli, "validate_fleet_file", Spy())

    assert stop(fleet_path).exit_code == 0


def test_a_missing_state_file_is_refused(tmp_path: Path) -> None:
    """Never started, or the wrong path — either way, not a fleet that stopped."""
    result = stop(tmp_path / "fleet.json")

    assert result.exit_code == 1
    assert "state" in result.output.lower()


def test_a_survivor_makes_the_command_exit_non_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``stop`` returning 0 has to mean the fleet is gone."""
    fleet_path = separable_fleet(tmp_path, "alpha", "beta")
    state_path = fleet_state_path(fleet_path.resolve())
    stopping(
        monkeypatch,
        report_for(state_path, targeted=("alpha", "beta"), survivors=("beta",)),
    )

    result = stop(fleet_path)

    assert result.exit_code == 1
    assert "Stopped alpha" in result.output
    assert "instances.beta" in result.output


def test_a_surviving_supervisor_makes_the_command_exit_non_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fleet_path = separable_fleet(tmp_path, "alpha")
    state_path = fleet_state_path(fleet_path.resolve())
    stopping(monkeypatch, report_for(state_path, supervisor_stopped=False))

    result = stop(fleet_path)

    assert result.exit_code == 1
    assert "4242" in result.output


def test_stopping_a_fleet_that_is_not_running_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Idempotent, so an operator can run it twice without reading an error."""
    fleet_path = separable_fleet(tmp_path, "alpha")
    state_path = fleet_state_path(fleet_path.resolve())
    stopping(
        monkeypatch,
        report_for(state_path, targeted=(), supervisor_pid=None),
    )

    result = stop(fleet_path)

    assert result.exit_code == 0
    assert "No instance" in result.output


def test_the_grace_option_reaches_the_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fleet_path = separable_fleet(tmp_path, "alpha")
    state_path = fleet_state_path(fleet_path.resolve())
    calls = stopping(monkeypatch, report_for(state_path))

    assert stop(fleet_path, "--grace", "1.5").exit_code == 0
    assert [grace for _, grace in calls] == [1.5]


def test_a_negative_grace_is_refused(tmp_path: Path) -> None:
    fleet_path = separable_fleet(tmp_path, "alpha")
    assert stop(fleet_path, "--grace", "-1").exit_code != 0


def test_the_stop_fleet_option_is_required() -> None:
    assert runner.invoke(fleet_app, ["stop"]).exit_code != 0


def test_the_stop_fleet_path_is_expanded_before_anything_reads_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fleet_path = separable_fleet(tmp_path, "alpha")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    expected = fleet_state_path(fleet_path.resolve())
    calls = stopping(monkeypatch, report_for(expected))

    assert runner.invoke(fleet_app, ["stop", "--fleet", "~/fleet.json"]).exit_code == 0
    assert [path for path, _ in calls] == [expected]


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


def test_the_root_command_exposes_fleet_stop() -> None:
    """The callback must keep ``fleet`` a group now that it has three subcommands."""
    result = runner.invoke(root_app, ["fleet", "--help"])
    assert result.exit_code == 0
    assert "stop" in result.output


def test_the_root_command_exposes_fleet_status() -> None:
    result = runner.invoke(root_app, ["fleet", "--help"])
    assert result.exit_code == 0
    assert "status" in result.output


def test_the_supervisor_process_is_named_after_its_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreground supervisor must be findable in ``ps`` as ``nanobot-fleet``."""
    titles: list[str] = []
    monkeypatch.setattr("nanobot.cli.process_identity.os.name", "posix")
    monkeypatch.setattr("nanobot.cli.process_identity._set_process_title", titles.append)

    set_cli_process_identity(["fleet", "start", "--fleet", "/tmp/fleet.json"])

    assert titles == ["nanobot-fleet"]

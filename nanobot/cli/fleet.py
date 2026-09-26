"""Typer commands for the confined fleet supervisor.

``nanobot fleet start`` is the only place the fleet package's four gates are run
in order, and the order is the whole point:

1. **validate** the fleet document and every instance's paths
   (:func:`~nanobot.fleet.validate.validate_fleet_file`);
2. **prepare** the directories, the supervisor state file and every instance's
   Seatbelt profile (:func:`~nanobot.fleet.supervisor.prepare_fleet`);
3. **prove** each of those profiles actually binds on this host
   (:func:`~nanobot.fleet.probe.probe_fleet_confinement`);
4. only then **start** the instances and watch them in the foreground.

Steps 1 to 3 all refuse by exiting non-zero and naming the instance, and none of
them starts an instance process. That ordering is the command's contribution: the
fleet package refuses to *emit* a rule it cannot defend, but "the builder was
satisfied" is a claim about the builder. Seatbelt accepts a deny naming a path
that does not exist, confines nothing, and exits 0 — so a fleet that skipped the
probe would report every instance as started and confined while each could read
all of its peers. Starting nothing is strictly better than starting something
that only looks confined, which is why the probe is a gate and not a warning.

``nanobot fleet stop`` is the mirror image and shares none of that machinery. It
never reads the fleet document: a fleet that is already running must be
stoppable even if its declaration has since been edited, moved, or made invalid,
and a ``stop`` that refused on a validation error would leave confined processes
running with the only command that can reach them refusing to run. All it needs
is the state file, which :mod:`nanobot.fleet.stop` turns into process trees.

``nanobot fleet status`` needs even less: it reads that same file and writes
nothing at all. Both facts are requirements rather than conveniences. A
supervisor holds the foreground of the terminal it was started in, so status has
to be answerable from a *second* shell — which is why fleet state is a file in
the first place, and not something a supervisor would have to be asked for. And
because the supervisor is the file's only writer, a reader that repaired or
rewrote what it found would race the very ``os.replace`` it was reading; so
status derives liveness on the way past (:func:`~nanobot.fleet.state.read_fleet_state`
reconciles every pid against a PID-reuse-safe identity) and leaves the bytes
exactly as it found them. Status also never judges: an exited instance is a
report, not an error, so it exits zero. Only a state file that cannot be read or
believed is a refusal, for the reason
:class:`~nanobot.fleet.state.FleetStateError` gives — answering "no instances"
over an unreadable file would tell an operator the fleet is stopped, and the next
thing that operator does is start a second fleet on the same ports and
workspaces.

Deliberately *not* here: the decision logic. This module reads as a sequence of
calls into :mod:`nanobot.fleet` because every judgement it could make has an
owner there already — what counts as a separable layout, what a profile must
deny, what proves a deny bound, when an instance is dead, and what belongs to an
instance's tree. Its own job is to turn those results into an exit code and a
message an operator can act on.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from nanobot.fleet.config import FleetConfigError
from nanobot.fleet.instance import InstanceLaunchError
from nanobot.fleet.probe import ProbeResult, probe_fleet_confinement, unproven
from nanobot.fleet.profile import SeatbeltProfileError
from nanobot.fleet.state import (
    FleetStateError,
    InstanceRecord,
    fleet_state_path,
    read_fleet_state,
    status_payload,
)
from nanobot.fleet.stop import (
    DEFAULT_STOP_GRACE_SECONDS,
    FleetStopReport,
    stop_fleet,
)
from nanobot.fleet.supervisor import FleetPlan, prepare_fleet, start_fleet
from nanobot.fleet.validate import (
    FleetValidationError,
    ResolvedInstance,
    validate_fleet_file,
)

console = Console()

fleet_app = typer.Typer(
    help="Run several confined nanobot instances under one supervisor.",
)


@fleet_app.callback()
def fleet() -> None:
    """Group the fleet subcommands.

    Present only so that ``fleet`` stays a command *group*: Typer collapses a
    single-command app into that command. It must stay even though this app now
    has two subcommands — a future edit that left only one would silently
    re-spell that one as ``nanobot fleet``.
    """


@fleet_app.command("start")
def fleet_start(
    fleet: str = typer.Option(..., "--fleet", "-f", help="Path to the fleet file"),
) -> None:
    """Validate a fleet, prove it is confined, then run it in the foreground."""
    path = Path(fleet).expanduser()
    instances = _validated(path)
    plan = _prepared(instances, path)
    _require_proven_confinement(plan)
    _announce(plan)
    _report(_supervise(plan))


@fleet_app.command("status")
def fleet_status(
    fleet: str = typer.Option(..., "--fleet", "-f", help="Path to the fleet file"),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print the machine-readable JSON array instead of a table.",
    ),
) -> None:
    """Report every instance of a fleet, reconciling liveness as it reads.

    Answerable from a second shell while the supervisor holds the first, and
    writes nothing: see this module's docstring for why both matter. Exits zero
    whatever the instances are doing — only an unreadable state file is a refusal.
    """
    state_path = _state_path(fleet)
    try:
        records = read_fleet_state(state_path)
    except FleetStateError as exc:
        _refuse(str(exc))
    if as_json:
        _print_status_json(records)
        return
    _print_status_table(records, state_path)


def _print_status_json(records: Sequence[InstanceRecord]) -> None:
    """Write the status array, and nothing else, to stdout.

    Through ``typer.echo`` rather than the rich console deliberately: the console
    wraps at the terminal width, so a long workspace path would be folded across
    lines and the document a consumer parsed would depend on how wide their
    window happened to be. What each object contains is
    :func:`~nanobot.fleet.state.status_payload`'s decision, not this module's.
    """
    typer.echo(json.dumps([status_payload(record) for record in records], indent=2))


def _print_status_table(
    records: Sequence[InstanceRecord],
    state_path: Path,
) -> None:
    """Render the same facts for a person, naming the file they came from."""
    if not records:
        console.print("[dim]No instance has been started for this fleet.[/dim]")
        console.print(f"State: {escape(str(state_path))}")
        return
    table = Table(title="Fleet Status")
    table.add_column("Instance", style="cyan")
    table.add_column("State")
    table.add_column("PID", justify="right")
    table.add_column("Exit reason")
    table.add_column("Memory", justify="right")
    # Folded rather than truncated: a workspace path an operator cannot read in
    # full is the one fact this table exists to show them.
    table.add_column("Workspace", overflow="fold")
    table.add_column("Config", overflow="fold")
    for record in records:
        running = record.state == "running"
        table.add_row(
            escape(record.name),
            "[green]running[/green]" if running else "[dim]exited[/dim]",
            str(record.pid),
            escape(record.exit_reason or "-"),
            f"{record.memory_limit_mb} MB",
            escape(str(record.workspace)),
            escape(str(record.config_dir)),
        )
    console.print(table)
    console.print(f"State: {escape(str(state_path))}")


@fleet_app.command("stop")
def fleet_stop(
    fleet: str = typer.Option(..., "--fleet", "-f", help="Path to the fleet file"),
    grace: float = typer.Option(
        DEFAULT_STOP_GRACE_SECONDS,
        "--grace",
        help="Seconds to allow after SIGTERM before escalating to SIGKILL.",
        min=0.0,
    ),
) -> None:
    """Terminate every instance's process tree, then the supervisor.

    Returns only once nothing of the fleet is left, and exits non-zero naming
    whatever would not die.
    """
    try:
        report = stop_fleet(_state_path(fleet), grace=grace)
    except FleetStateError as exc:
        _refuse(str(exc))
    _report_stop(report)


def _state_path(fleet: str) -> Path:
    """Where the supervisor for the fleet declared at ``fleet`` keeps its state.

    The one expression the two commands that never read the fleet document use,
    shared for the reason :func:`~nanobot.fleet.state.fleet_state_path` exists at
    all: ``prepare_fleet`` canonicalises the fleet path before deriving the state
    path, so a command that derived it from a different spelling would look for a
    file the supervisor does not write — and would report an unstarted fleet
    rather than an error.
    """
    return fleet_state_path(Path(fleet).expanduser().resolve(strict=False))


def _report_stop(report: FleetStopReport) -> None:
    """Say what stopped, and refuse to claim success over a survivor."""
    if not report.targeted:
        console.print("[dim]No instance of this fleet was running.[/dim]")
        return
    for name in report.stopped:
        console.print(f"Stopped {escape(name)}")
    if report.supervisor_pid is not None and report.supervisor_stopped:
        console.print(f"Stopped the supervisor (pid {report.supervisor_pid})")
    if report.complete:
        return
    _refuse(
        "\n".join(
            [
                "This fleet was not fully stopped.",
                "",
                *(f"  instances.{name} is still running" for name in report.survivors),
                *(
                    [f"  the supervisor (pid {report.supervisor_pid}) is still running"]
                    if not report.supervisor_stopped
                    else []
                ),
            ]
        )
    )


def _validated(path: Path) -> tuple[ResolvedInstance, ...]:
    """Load and validate the fleet document, or refuse.

    Creates nothing. A fleet refused here has left no directories, no state file
    and no profiles behind, which is what makes "a validation failure starts
    nothing" true of the filesystem and not only of the process table.
    """
    try:
        return validate_fleet_file(path)
    except (FleetConfigError, FleetValidationError) as exc:
        _refuse(str(exc))


def _prepared(instances: Sequence[ResolvedInstance], path: Path) -> FleetPlan:
    """Lay the fleet down on disk and build every profile, or refuse.

    The refusals here are the fleet package declining to emit a rule it cannot
    defend — a deny naming a path that does not exist would be accepted by the
    kernel, match nothing, and confine nothing.
    """
    try:
        return prepare_fleet(instances, fleet_path=path)
    except (SeatbeltProfileError, FleetStateError, InstanceLaunchError) as exc:
        _refuse(f"This fleet cannot be prepared: {path}\n\n{exc}")


def _require_proven_confinement(plan: FleetPlan) -> None:
    """Refuse unless every instance's profile was *observed* to refuse a read.

    The gate that has to come after the profiles exist and before any instance
    does. Every unproven instance is named, not just the first: an operator
    fixing a host that cannot run ``sandbox-exec`` would otherwise rerun the
    command once per instance to discover the same thing each time.
    """
    results = probe_fleet_confinement(
        plan.instances, plan.profiles, state_path=plan.state_path
    )
    failures = unproven(results)
    if not failures:
        return
    _refuse(
        "\n".join(
            [
                "Confinement could not be proven, so no instance was started.",
                "",
                *_probe_lines(failures),
            ]
        )
    )


def _probe_lines(failures: Iterable[ProbeResult]) -> list[str]:
    """One indented location/reason pair per unproven instance."""
    lines: list[str] = []
    for failure in failures:
        lines.extend((f"  instances.{failure.name}", f"    {failure.reason}", ""))
    return lines[:-1] if lines else lines


def _announce(plan: FleetPlan) -> None:
    """Say what is about to start, before the foreground loop takes the terminal."""
    console.print(
        f"[green]Confinement proven for {len(plan.instances)} instance(s).[/green]"
    )
    console.print(f"State: {escape(str(plan.state_path))}")


def _supervise(plan: FleetPlan) -> tuple[InstanceRecord, ...]:
    """Start every instance and watch it until the fleet is done, or refuse.

    ``start_fleet`` is all or nothing: if any instance fails to launch it stops
    and records the ones that already started before re-raising, so a refusal
    here still leaves nothing of this fleet running.
    """
    try:
        supervisor = start_fleet(plan)
    except (InstanceLaunchError, FleetStateError) as exc:
        _refuse(f"This fleet could not be started: {exc}")
    for record in supervisor.records:
        console.print(f"Started {escape(record.name)} (pid {record.pid})")
    return supervisor.run()


def _report(records: Sequence[InstanceRecord]) -> None:
    """Print how each instance ended, once the supervisor has returned."""
    console.print("[dim]Fleet stopped.[/dim]")
    for record in records:
        reason = record.exit_reason or "unknown"
        console.print(f"{escape(record.name)}: {record.state} ({reason})")


def _refuse(message: str) -> NoReturn:
    """Print a refusal and exit non-zero.

    Typed :data:`~typing.NoReturn` so that the callers which end in a bare call
    to it — every gate in this module — are seen to terminate rather than to
    fall through returning ``None`` where an instance or a plan was promised.
    """
    console.print(f"[red]{escape(message)}[/red]")
    raise typer.Exit(1)

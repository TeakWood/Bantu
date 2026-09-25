"""Tests for stopping a fleet from outside it.

Two halves, deliberately.

The first drives :mod:`nanobot.fleet.stop` over a *fake process table* — a
dictionary of pids, groups, parents and identities that signals are delivered
into. That is not a way of avoiding real processes; it is the only way to test
the cases that matter, because the interesting ones are a process that ignores
``SIGTERM``, a pid that gets recycled mid-escalation, and a descendant that has
been orphaned out of every walk. None of those can be arranged reliably against
a real kernel. The *production* signalling path still runs for real: nothing
patches :func:`~nanobot.fleet.supervisor.signal_process_tree`, so every test here
goes through it down to ``os.killpg``, which is where the bead's acceptance
criterion points.

The second half is three macOS tests against real processes — a three-level
supervisor/instance/descendant tree, with the descendant ignoring ``SIGTERM`` so
that the carry-forward is exercised rather than asserted. They are what stops the
first half from being a test of its own fake.

Fake pids are all above 900000. macOS pids stop at 99998, so a fake can never
collide with a live process, and the guards that compare a target against this
process's own group are never satisfied by accident.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nanobot.fleet import stop as stop_module
from nanobot.fleet import supervisor as supervisor_module
from nanobot.fleet.instance import instance_identity
from nanobot.fleet.memory import process_parent_pid
from nanobot.fleet.state import (
    FleetStateError,
    InstanceRecord,
    write_fleet_state,
)
from nanobot.fleet.stop import (
    DEFAULT_STOP_GRACE_SECONDS,
    STOP_ESCALATION,
    FleetStopReport,
    InstanceTree,
    TreeMember,
    stop_fleet,
)
from nanobot.process_runtime import process_is_running

darwin_only = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the fleet's process-tree walks are implemented for macOS only",
)

#: Well above the largest pid macOS will issue, so a fake pid is never a real one.
FAKE_PID_BASE = 900_000

REAL_TIMEOUT_SECONDS = 30.0


# ----------------------------------------------------------------------------
# A process table signals can be delivered into
# ----------------------------------------------------------------------------


@dataclass
class FakeProcess:
    """One entry in :class:`World`'s process table."""

    pid: int
    identity: str
    group: int
    parent: int | None = None
    #: Signals this process catches and survives. ``SIGKILL`` is never in here,
    #: for the reason the escalation ends with it: it cannot be caught.
    ignores: frozenset[int] = frozenset()
    #: Survives everything, including ``SIGKILL``. Not something a process can
    #: arrange, but a process wedged in an uninterruptible wait presents this way
    #: — and the report has to be honest about it rather than claim success.
    unkillable: bool = False


@dataclass
class Delivered:
    """One signal the policy sent, and how it was addressed."""

    #: ``"group"`` for ``killpg`` and ``"pid"`` for ``kill`` — the distinction the
    #: acceptance criterion is about.
    kind: str
    target: int
    signal: int


class World:
    """A process table, its topology, and a log of what was signalled."""

    def __init__(self) -> None:
        self.processes: dict[int, FakeProcess] = {}
        self.delivered: list[Delivered] = []
        self._next = FAKE_PID_BASE

    # -- construction --------------------------------------------------

    def spawn(
        self,
        *,
        parent: int | None = None,
        group: int | None = None,
        ignores: frozenset[int] = frozenset(),
        unkillable: bool = False,
    ) -> FakeProcess:
        """Add a process. Its group defaults to itself, as ``setsid`` would."""
        self._next += 1
        pid = self._next
        process = FakeProcess(
            pid=pid,
            identity=f"darwin:{pid}:1:{pid}",
            group=pid if group is None else group,
            parent=parent,
            ignores=ignores,
            unkillable=unkillable,
        )
        self.processes[pid] = process
        return process

    def recycle(self, pid: int) -> FakeProcess:
        """Hand ``pid`` to an unrelated process, as the kernel eventually does."""
        replacement = FakeProcess(
            pid=pid, identity=f"darwin:{pid}:9:99", group=pid, parent=None
        )
        self.processes[pid] = replacement
        return replacement

    # -- the probes the policy uses ------------------------------------

    def is_running(self, pid: int) -> bool:
        return pid in self.processes

    def identity_match(self, recorded: object, pid: int) -> str:
        process = self.processes.get(pid)
        if process is None:
            return "unknown"
        if recorded is None:
            return "match"
        return "match" if recorded == process.identity else "mismatch"

    def identity_record(self, pid: int) -> dict[str, str | int | None]:
        process = self.processes.get(pid)
        return {"stable_identity": None if process is None else process.identity}

    def group_pids(self, group: int) -> list[int] | None:
        return sorted(p.pid for p in self.processes.values() if p.group == group)

    def tree_pids(self, group: int) -> list[int] | None:
        """The group, plus the descendant closure of its leader — as libproc does."""
        found = set(self.group_pids(group) or ())
        pending = [group] if group in self.processes else []
        while pending:
            current = pending.pop()
            for process in self.processes.values():
                if process.parent == current and process.pid not in found:
                    found.add(process.pid)
                    pending.append(process.pid)
        return sorted(found)

    def parent_pid(self, pid: int) -> int | None:
        process = self.processes.get(pid)
        return None if process is None else process.parent

    # -- signal delivery -----------------------------------------------

    def killpg(self, group: int, sig: int) -> None:
        members = [p for p in self.processes.values() if p.group == group]
        if not members:
            raise ProcessLookupError(f"no such process group {group}")
        self.delivered.append(Delivered("group", group, sig))
        for member in members:
            self._deliver(member.pid, sig)

    def kill(self, pid: int, sig: int) -> None:
        if pid not in self.processes:
            raise ProcessLookupError(f"no such process {pid}")
        self.delivered.append(Delivered("pid", pid, sig))
        self._deliver(pid, sig)

    def _deliver(self, pid: int, sig: int) -> None:
        process = self.processes.get(pid)
        if process is None:
            return
        if process.unkillable or (sig != signal.SIGKILL and sig in process.ignores):
            return
        del self.processes[pid]

    # -- convenience ----------------------------------------------------

    def signals_to(self, target: int) -> list[Delivered]:
        return [one for one in self.delivered if one.target == target]


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    """A fake process table wired into every probe the stop policy reads.

    ``signal_process_tree`` is deliberately *not* replaced: it is the production
    code that decides between a group and a pid, so the tests reach ``os.killpg``
    through it rather than around it.
    """
    table = World()
    # The stop policy's own view of the world.
    monkeypatch.setattr(stop_module, "process_is_running", table.is_running)
    monkeypatch.setattr(stop_module, "instance_identity_match", table.identity_match)
    monkeypatch.setattr(stop_module, "instance_identity", table.identity_record)
    monkeypatch.setattr(stop_module, "process_tree_pids", table.tree_pids)
    monkeypatch.setattr(stop_module, "process_parent_pid", table.parent_pid)
    monkeypatch.setattr(stop_module.os, "kill", table.kill)
    # What ``signal_process_tree`` reads to find strays, and how it signals.
    monkeypatch.setattr(supervisor_module, "process_tree_pids", table.tree_pids)
    monkeypatch.setattr(supervisor_module, "process_group_pids", table.group_pids)
    monkeypatch.setattr(supervisor_module.os, "killpg", table.killpg)
    monkeypatch.setattr(supervisor_module.os, "kill", table.kill)
    # Reconciliation inside ``read_fleet_state`` looks at the same table, so a
    # record for a process this world does not have reads as exited.
    monkeypatch.setattr("nanobot.fleet.state.process_is_running", table.is_running)
    monkeypatch.setattr(
        "nanobot.fleet.state.instance_identity_match", table.identity_match
    )
    return table


@dataclass
class FakeTime:
    """A clock the escalation can be driven over without waiting for it."""

    now: float = 0.0
    slept: list[float] = field(default_factory=list)

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += max(seconds, 0.0)

    def monotonic(self) -> float:
        return self.now


def state_file(
    path: Path,
    world: World,
    *instances: tuple[str, FakeProcess],
    exited: tuple[str, ...] = (),
) -> Path:
    """Write a state file describing ``instances`` as the supervisor would."""
    records = [record_for(name, process) for name, process in instances]
    records.extend(
        InstanceRecord(
            name=name,
            pid=FAKE_PID_BASE - index - 1,
            state="exited",
            exit_reason="exit",
            workspace=Path("/tmp") / name / "workspace",
            config_dir=Path("/tmp") / name,
            memory_limit_mb=512,
        )
        for index, name in enumerate(exited)
    )
    write_fleet_state(records, path=path)
    return path


def record_for(name: str, process: FakeProcess) -> InstanceRecord:
    """The record the supervisor would have written for ``process``."""
    return InstanceRecord(
        name=name,
        pid=process.pid,
        state="running",
        exit_reason=None,
        workspace=Path("/tmp") / name / "workspace",
        config_dir=Path("/tmp") / name,
        memory_limit_mb=512,
        identity={"stable_identity": process.identity},
    )


def run_stop(path: Path, *, grace: float = 2.0) -> FleetStopReport:
    """Drive ``stop_fleet`` over a fake clock."""
    fake = FakeTime()
    return stop_fleet(
        path,
        grace=grace,
        poll_interval=0.05,
        sleep=fake.sleep,
        clock=fake.monotonic,
    )


# ----------------------------------------------------------------------------
# Group-directed signalling — the bead's named acceptance criterion
# ----------------------------------------------------------------------------


def test_termination_is_addressed_to_the_process_group_not_the_pid(
    world: World,
    tmp_path: Path,
) -> None:
    """``os.killpg`` on the instance's group, and the pid is never signalled alone."""
    instance = world.spawn()
    report = run_stop(state_file(tmp_path / "fleet.json.state.json", world, ("alpha", instance)))

    assert report.complete
    group_signals = [one for one in world.delivered if one.kind == "group"]
    assert [(one.target, one.signal) for one in group_signals] == [
        (instance.group, signal.SIGTERM)
    ]
    assert [one for one in world.delivered if one.kind == "pid"] == []


def test_a_signal_to_the_group_reaches_a_child_that_stayed_in_it(
    world: World,
    tmp_path: Path,
) -> None:
    """The point of addressing the group: one call covers everything still in it."""
    instance = world.spawn()
    child = world.spawn(parent=instance.pid, group=instance.group)

    report = run_stop(state_file(tmp_path / "state.json", world, ("alpha", instance)))

    assert report.complete
    assert child.pid not in world.processes
    # One group call did it; the child was never addressed individually.
    assert [one.kind for one in world.delivered] == ["group"]


def test_a_descendant_that_left_the_group_is_signalled_too(
    world: World,
    tmp_path: Path,
) -> None:
    """A ``setsid`` child is in the tree but not the group — the shell tool's shape."""
    instance = world.spawn()
    detached = world.spawn(parent=instance.pid)  # its own group, as setsid gives

    report = run_stop(state_file(tmp_path / "state.json", world, ("alpha", instance)))

    assert report.complete
    assert detached.pid not in world.processes
    assert Delivered("pid", detached.pid, signal.SIGTERM) in world.delivered


def test_escalation_is_sigterm_then_sigkill(world: World, tmp_path: Path) -> None:
    """A process that ignores ``SIGTERM`` gets ``SIGKILL``, in that order."""
    instance = world.spawn(ignores=frozenset({int(signal.SIGTERM)}))

    report = run_stop(state_file(tmp_path / "state.json", world, ("alpha", instance)))

    assert report.complete
    assert [one.signal for one in world.signals_to(instance.group)] == [
        signal.SIGTERM,
        signal.SIGKILL,
    ]
    assert STOP_ESCALATION == (signal.SIGTERM, signal.SIGKILL)


def test_sigkill_is_not_sent_to_an_instance_that_honoured_sigterm(
    world: World,
    tmp_path: Path,
) -> None:
    """The escalation stops as soon as there is nothing left to escalate against."""
    instance = world.spawn()

    run_stop(state_file(tmp_path / "state.json", world, ("alpha", instance)))

    assert signal.SIGKILL not in {one.signal for one in world.delivered}


def test_an_orphaned_descendant_is_still_killed_after_its_parent_dies(
    world: World,
    tmp_path: Path,
) -> None:
    """The case the remembered members exist for, and the only one that needs them.

    The instance honours ``SIGTERM`` and the descendant does not, so by the time
    ``SIGKILL`` is due there is no live process left to walk the tree *from*: the
    group is empty and the descendant's parent is gone. Forgetting it between the
    two rounds would leave it running, which is exactly the fleet-stop failure
    the acceptance criterion names.
    """
    instance = world.spawn()
    orphan = world.spawn(parent=instance.pid, ignores=frozenset({int(signal.SIGTERM)}))

    report = run_stop(state_file(tmp_path / "state.json", world, ("alpha", instance)))

    assert report.complete
    assert orphan.pid not in world.processes
    assert Delivered("pid", orphan.pid, signal.SIGKILL) in world.delivered


# ----------------------------------------------------------------------------
# Blocking until the fleet is gone
# ----------------------------------------------------------------------------


def test_stop_does_not_return_while_an_instance_is_still_running(
    world: World,
    tmp_path: Path,
) -> None:
    """It waits, rather than signalling and reporting success optimistically."""
    instance = world.spawn(ignores=frozenset({int(signal.SIGTERM)}))
    fake = FakeTime()

    report = stop_fleet(
        state_file(tmp_path / "state.json", world, ("alpha", instance)),
        grace=2.0,
        poll_interval=0.05,
        sleep=fake.sleep,
        clock=fake.monotonic,
    )

    assert report.complete
    # It slept through the whole SIGTERM grace before escalating, which is the
    # observable form of "did not return after signalling".
    assert sum(fake.slept) >= 2.0


def test_a_survivor_is_named_and_the_stop_is_not_reported_complete(
    world: World,
    tmp_path: Path,
) -> None:
    """Nothing kills this one; the report must say so rather than claim success."""
    stubborn = world.spawn(unkillable=True)

    report = run_stop(
        state_file(tmp_path / "state.json", world, ("alpha", stubborn)), grace=0.2
    )

    assert not report.complete
    assert report.survivors == ("alpha",)
    assert report.stopped == ()


def test_a_fleet_with_nothing_running_is_a_no_op(world: World, tmp_path: Path) -> None:
    """Idempotent: stopping a stopped fleet signals nothing and succeeds."""
    report = run_stop(
        state_file(tmp_path / "state.json", world, exited=("alpha", "beta"))
    )

    assert report.complete
    assert report.targeted == ()
    assert world.delivered == []


def test_an_instance_whose_pid_was_recycled_is_never_signalled(
    world: World,
    tmp_path: Path,
) -> None:
    """Reconciliation is what protects the stranger now holding that pid."""
    instance = world.spawn()
    path = state_file(tmp_path / "state.json", world, ("alpha", instance))
    stranger = world.recycle(instance.pid)

    report = run_stop(path)

    assert report.complete
    assert report.targeted == ()
    assert world.delivered == []
    assert stranger.pid in world.processes


def test_a_remembered_descendant_whose_pid_was_recycled_is_not_signalled(
    world: World,
    tmp_path: Path,
) -> None:
    """A carried-forward member is re-identified, not merely re-probed for life.

    Without the identity check the set of "processes we were about to kill" would
    be a list of pids the kernel is free to hand to anybody.
    """
    instance = world.spawn(ignores=frozenset({int(signal.SIGTERM)}))
    detached = world.spawn(parent=instance.pid)
    tree = InstanceTree(record_for("alpha", instance), process_group=instance.group)
    assert detached.pid in tree.signal(signal.SIGTERM)

    stranger = world.recycle(detached.pid)
    world.delivered.clear()
    tree.signal(signal.SIGKILL)

    assert stranger.pid in world.processes
    assert world.signals_to(stranger.pid) == []


def test_a_tree_recorded_in_this_processs_own_group_is_never_enumerated(
    world: World,
) -> None:
    """The guard that stops a stop from sweeping up its own unrelated siblings.

    A recorded group is only an instance's whole tree because the instance was
    spawned with ``start_new_session``. If that were ever dropped, or a state
    file survived into a shell whose group happened to match, enumerating it
    would put every process sharing this command's group into the set about to
    be killed — this command included. So the enumeration refuses, and
    :func:`~nanobot.fleet.supervisor.signal_process_tree` independently refuses
    to address the group at all.
    """
    own = os.getpgid(0)
    instance = world.spawn(group=own)
    sibling = world.spawn(group=own)
    tree = InstanceTree(record_for("alpha", instance), process_group=own)

    assert {member.pid for member in tree.survey()} == {instance.pid}

    tree.signal(signal.SIGTERM)

    assert sibling.pid in world.processes
    assert world.signals_to(sibling.pid) == []
    assert [one for one in world.delivered if one.kind == "group"] == []


# ----------------------------------------------------------------------------
# The supervisor
# ----------------------------------------------------------------------------


def test_the_supervisor_is_found_by_parentage_and_stopped_last(
    world: World,
    tmp_path: Path,
) -> None:
    """The state file names no supervisor, so parentage is the only route to it."""
    supervisor = world.spawn()
    alpha = world.spawn(parent=supervisor.pid)
    beta = world.spawn(parent=supervisor.pid)

    report = run_stop(
        state_file(tmp_path / "state.json", world, ("alpha", alpha), ("beta", beta))
    )

    assert report.complete
    assert report.supervisor_pid == supervisor.pid
    assert report.supervisor_stopped
    assert supervisor.pid not in world.processes
    # Instances first, then the supervisor: the bead's stated order.
    order = [one.target for one in world.delivered]
    assert order.index(supervisor.pid) > order.index(alpha.group)
    assert order.index(supervisor.pid) > order.index(beta.group)


def test_a_supervisor_that_ignores_sigterm_is_killed(
    world: World,
    tmp_path: Path,
) -> None:
    supervisor = world.spawn(ignores=frozenset({int(signal.SIGTERM)}))
    alpha = world.spawn(parent=supervisor.pid)

    report = run_stop(state_file(tmp_path / "state.json", world, ("alpha", alpha)))

    assert report.complete
    assert [one.signal for one in world.signals_to(supervisor.pid)] == [
        signal.SIGTERM,
        signal.SIGKILL,
    ]


def test_an_orphaned_instance_names_no_supervisor(world: World, tmp_path: Path) -> None:
    """Reparented to ``launchd`` means the supervisor is already gone, not that it is 1."""
    instance = world.spawn(parent=1)

    report = run_stop(state_file(tmp_path / "state.json", world, ("alpha", instance)))

    assert report.complete
    assert report.supervisor_pid is None
    assert world.signals_to(1) == []


def test_instances_that_disagree_about_their_parent_name_no_supervisor(
    world: World,
    tmp_path: Path,
) -> None:
    """A guess here is a signal aimed at a process nothing has identified."""
    first = world.spawn()
    second = world.spawn()
    alpha = world.spawn(parent=first.pid)
    beta = world.spawn(parent=second.pid)

    report = run_stop(
        state_file(tmp_path / "state.json", world, ("alpha", alpha), ("beta", beta))
    )

    assert report.supervisor_pid is None
    assert first.pid in world.processes
    assert second.pid in world.processes


def test_the_stopping_process_is_never_taken_for_the_supervisor(
    world: World,
    tmp_path: Path,
) -> None:
    """``fleet stop`` run from the shell that started the fleet must not kill itself."""
    for parent in (os.getpid(), os.getppid()):
        instance = world.spawn(parent=parent)
        report = run_stop(
            state_file(tmp_path / f"state-{parent}.json", world, ("alpha", instance))
        )
        assert report.supervisor_pid is None
        assert world.signals_to(parent) == []


# ----------------------------------------------------------------------------
# Refusals and reporting
# ----------------------------------------------------------------------------


def test_a_missing_state_file_is_refused(tmp_path: Path) -> None:
    """Not an empty fleet: a path that was never a fleet must not report success."""
    with pytest.raises(FleetStateError) as caught:
        stop_fleet(tmp_path / "absent.state.json")
    assert caught.value.kind == "io_error"


def test_an_unparseable_state_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("[{\"name\": \"alpha\"}]", encoding="utf-8")
    with pytest.raises(FleetStateError) as caught:
        stop_fleet(path)
    assert caught.value.kind == "invalid_record"


def test_the_report_separates_what_stopped_from_what_survived(
    world: World,
    tmp_path: Path,
) -> None:
    alpha = world.spawn()
    beta = world.spawn(unkillable=True)

    report = run_stop(
        state_file(tmp_path / "state.json", world, ("alpha", alpha), ("beta", beta)),
        grace=0.2,
    )

    assert report.targeted == ("alpha", "beta")
    assert report.stopped == ("alpha",)
    assert report.survivors == ("beta",)
    assert not report.complete


def test_the_default_grace_matches_the_supervisors_own_shutdown() -> None:
    """An instance sees the same deadline however the fleet is being stopped."""
    assert DEFAULT_STOP_GRACE_SECONDS == (
        supervisor_module.DEFAULT_SHUTDOWN_GRACE_SECONDS
    )


def test_a_member_with_no_identity_is_trusted_rather_than_abandoned() -> None:
    """``process_runtime``'s posture: an unidentifiable live pid is still ours."""
    member = TreeMember(os.getpid(), None)
    assert member.alive() is True
    assert TreeMember(0).alive() is False


# ----------------------------------------------------------------------------
# Real processes
# ----------------------------------------------------------------------------

DESCENDANT_SCRIPT = """
import pathlib, signal, sys, time

# Survives SIGTERM on purpose: by the time SIGKILL is due its parent is gone and
# it is reachable only through what the stop policy remembered about it.
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text("ready", encoding="utf-8")
time.sleep(300)
"""

INSTANCE_SCRIPT = """
import json, pathlib, subprocess, sys, time

root = pathlib.Path(sys.argv[1])
child = subprocess.Popen(
    [sys.executable, str(root / "descendant.py"), str(root / "descendant.ready")],
    # What nanobot's shell tool does to every command it runs, which is what
    # takes the child out of this instance's process group.
    start_new_session=True,
)
(root / "instance.json").write_text(json.dumps({"descendant": child.pid}), "utf-8")
time.sleep(300)
"""

SUPERVISOR_SCRIPT = """
import json, pathlib, subprocess, sys, time

root = pathlib.Path(sys.argv[1])
child = subprocess.Popen(
    [sys.executable, str(root / "instance.py"), str(root)],
    start_new_session=True,
)
(root / "supervisor.json").write_text(json.dumps({"instance": child.pid}), "utf-8")
time.sleep(300)
"""


def child_environment() -> dict[str, str]:
    """The parent environment minus anything that would disturb the test run.

    ``COV_CORE_*``/``COVERAGE_*`` in particular: a pytest-cov child started
    outside the repository finds no ``pyproject.toml``, measures without the
    configured ``omit`` rules, and silently moves the whole suite's coverage
    denominator.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COV_CORE_", "COVERAGE_"))
    }


def wait_for(path: Path, *, timeout: float = REAL_TIMEOUT_SECONDS) -> None:
    """Block until ``path`` exists. No pytest-timeout in this repo, so wait here."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"{path} never appeared")


@dataclass(frozen=True)
class RealFleet:
    """A live supervisor/instance/descendant chain and its state file."""

    supervisor: subprocess.Popen[bytes]
    instance_pid: int
    descendant_pid: int
    state_path: Path


@pytest.fixture
def real_fleet(tmp_path: Path) -> Iterator[RealFleet]:
    """Three real processes shaped like a fleet, and the state file describing it."""
    (tmp_path / "descendant.py").write_text(DESCENDANT_SCRIPT, encoding="utf-8")
    (tmp_path / "instance.py").write_text(INSTANCE_SCRIPT, encoding="utf-8")
    (tmp_path / "supervisor.py").write_text(SUPERVISOR_SCRIPT, encoding="utf-8")

    recorded: dict[str, int] = {}
    supervisor = subprocess.Popen(
        [sys.executable, str(tmp_path / "supervisor.py"), str(tmp_path)],
        env=child_environment(),
        cwd=str(tmp_path),
    )
    try:
        wait_for(tmp_path / "supervisor.json")
        wait_for(tmp_path / "instance.json")
        wait_for(tmp_path / "descendant.ready")
        recorded.update(
            json.loads((tmp_path / "supervisor.json").read_text(encoding="utf-8"))
        )
        recorded.update(
            json.loads((tmp_path / "instance.json").read_text(encoding="utf-8"))
        )
        instance_pid = recorded["instance"]
        descendant_pid = recorded["descendant"]
        state_path = tmp_path / "fleet.json.state.json"
        write_fleet_state(
            [
                InstanceRecord(
                    name="alpha",
                    pid=instance_pid,
                    state="running",
                    exit_reason=None,
                    workspace=tmp_path / "workspace",
                    config_dir=tmp_path / "alpha",
                    memory_limit_mb=512,
                    identity=instance_identity(instance_pid),
                )
            ],
            path=state_path,
        )
        yield RealFleet(supervisor, instance_pid, descendant_pid, state_path)
    finally:
        # Every pid this fixture knows about, whether or not the setup got that
        # far: a test that fails mid-way must not leave a sleeping tree behind.
        for pid in (
            recorded.get("descendant"),
            recorded.get("instance"),
            supervisor.pid,
        ):
            if pid:
                with suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
        if supervisor.poll() is None:
            supervisor.wait(timeout=5)


@darwin_only
def test_a_real_fleet_leaves_no_instance_or_descendant_behind(
    real_fleet: RealFleet,
) -> None:
    """Acceptance criterion 1, against the kernel rather than a fake.

    The descendant ignores ``SIGTERM`` and is out of the instance's process
    group, so it survives the first round and is orphaned by the second — the
    exact shape the existing ``_stop_gateway`` helper, which terminates only the
    direct child, would miss entirely.
    """
    report = stop_fleet(real_fleet.state_path, grace=2.0)

    assert report.complete, report
    assert not process_is_running(real_fleet.instance_pid)
    assert not process_is_running(real_fleet.descendant_pid)


@darwin_only
def test_a_real_stop_takes_the_supervisor_with_it(real_fleet: RealFleet) -> None:
    report = stop_fleet(real_fleet.state_path, grace=2.0)

    assert report.supervisor_pid == real_fleet.supervisor.pid
    assert report.supervisor_stopped
    assert real_fleet.supervisor.wait(timeout=5) is not None


@darwin_only
def test_a_real_stop_has_already_happened_by_the_time_it_returns(
    real_fleet: RealFleet,
) -> None:
    """The blocking contract, checked the way an operator would check it.

    ``ps`` is asked *after* the call returns, with nothing waited for in between,
    so a stop that signalled and returned optimistically would fail here.
    """
    stop_fleet(real_fleet.state_path, grace=2.0)

    listed = subprocess.run(
        ["/bin/ps", "-o", "pid=", "-p", ",".join(
            str(pid)
            for pid in (
                real_fleet.instance_pid,
                real_fleet.descendant_pid,
                real_fleet.supervisor.pid,
            )
        )],
        capture_output=True,
        text=True,
        check=False,
    )
    # The supervisor is this process's child and may linger as a zombie until it
    # is reaped, which ``ps`` still lists; nothing else may appear at all.
    remaining = {int(line) for line in listed.stdout.split()}
    assert real_fleet.instance_pid not in remaining
    assert real_fleet.descendant_pid not in remaining


@darwin_only
def test_process_parent_pid_reads_a_real_parent(real_fleet: RealFleet) -> None:
    """The lookup ``stop`` recovers the supervisor with."""
    assert process_parent_pid(real_fleet.instance_pid) == real_fleet.supervisor.pid
    assert process_parent_pid(real_fleet.descendant_pid) == real_fleet.instance_pid
    assert process_parent_pid(os.getpid()) == os.getppid()



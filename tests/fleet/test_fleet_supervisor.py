"""Tests for the supervisor's foreground loop.

Two layers, for the same reason the launcher's tests have three.

Almost everything here drives the real loop over *stub children* — objects that
answer ``poll`` and nothing else. That is not a convenience: the behaviours this
bead is about are all about what the supervisor does with an exit, and a stub
child is the only way to say "this instance died of signal 9 now, and that one is
still serving" without arranging real processes to die on cue. The loop's clock
and its sleep are injected for the same reason, so a five-second shutdown grace
costs no wall-clock time and the tests stay deterministic.

The last test starts two real confined instances through the real kernel policy
and kills one of them with ``kill``, which is the acceptance criterion written
out literally: the peer must keep running *and keep serving*, the dead one must
be reported as exited with a reason, and nothing must be restarted. Its stub
interpreter writes a heartbeat into its own workspace so "still serving" is
something the test can observe rather than infer from a pid.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

import nanobot.fleet.supervisor as supervisor_module
from nanobot.fleet.config import FleetInstance, InstanceMode
from nanobot.fleet.instance import (
    SANDBOX_EXEC,
    InstanceLaunchError,
    LaunchedInstance,
    instance_log_path,
)
from nanobot.fleet.memory import process_group_pids, process_tree_pids
from nanobot.fleet.profile import SeatbeltProfileError, build_fleet_profiles
from nanobot.fleet.state import (
    STATE_FILE_MODE,
    FleetStateError,
    fleet_state_path,
    load_fleet_state,
)
from nanobot.fleet.supervisor import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    MAX_POLL_INTERVAL_SECONDS,
    STOP_SIGNALS,
    FleetPlan,
    FleetSupervisor,
    exit_reason_for,
    prepare_fleet,
    run_fleet,
    signal_instance_tree,
    start_fleet,
)
from nanobot.fleet.validate import ResolvedInstance
from nanobot.process_runtime import process_is_running

confinement_available = pytest.mark.skipif(
    sys.platform != "darwin" or not Path(SANDBOX_EXEC).is_file(),
    reason="a real instance cannot be launched without native Seatbelt",
)

READY_TIMEOUT_SECONDS = 30.0

# Far above any pid macOS will hand out, so a stub's process group can never
# collide with a real one — including the test runner's, which a group-directed
# signal must never reach.
STUB_PGID_BASE = 1_000_000


def make_instance(
    root: Path,
    name: str,
    *,
    mode: InstanceMode = "serve",
    memory_limit_mb: int = 512,
) -> ResolvedInstance:
    """Resolve one instance laid out the way nanobot lays one out by itself.

    ``<root>/<name>/config.json`` with ``<root>/<name>/workspace`` beneath it.
    Neither the workspace nor the log directory is created: creating them is
    :func:`prepare_fleet`'s job, and that it does so is under test.
    """
    config_dir = root / name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    return ResolvedInstance(
        name=name,
        entry=FleetInstance(
            config=str(config_path),
            mode=mode,
            memory_limit_mb=memory_limit_mb,
            env=[],
        ),
        config_path=config_path,
        config_dir=config_dir,
        workspace=config_dir / "workspace",
        port=None,
        port_setting="api.port",
    )


def make_fleet(root: Path, *names: str) -> tuple[Path, tuple[ResolvedInstance, ...]]:
    """A fleet document beside ``names`` instances, all mutually separable."""
    fleet_path = root / "fleet.json"
    fleet_path.write_text("{}", encoding="utf-8")
    return fleet_path, tuple(make_instance(root, name) for name in names)


class StubChild:
    """A child process that exists only to be polled.

    ``wait`` raises rather than blocking, which pins the property the fleet's
    independence rests on: the supervisor must never wait on one instance, or a
    child that hangs would freeze its peers' lifecycles, the memory cap's
    deadline, and the supervisor's own response to ``SIGTERM``.
    """

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.polls = 0

    def poll(self) -> int | None:
        self.polls += 1
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        raise AssertionError("the supervisor must never block on one instance")

    def finish(self, returncode: int) -> None:
        """Make this child report as exited from the next poll onwards."""
        self.returncode = returncode


class StubLauncher:
    """A spawn step that hands back stub children and records how it was used."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[str] = []
        self.children: dict[str, StubChild] = {}
        self.launched: dict[str, LaunchedInstance] = {}

    def __call__(self, instance: ResolvedInstance, profile: str) -> LaunchedInstance:
        self.calls.append(instance.name)
        if instance.name == self.fail_on:
            raise InstanceLaunchError(instance.name, "stub launch failure")
        pid = STUB_PGID_BASE + len(self.launched) + 1
        child = StubChild(pid)
        launched = LaunchedInstance(
            name=instance.name,
            mode=instance.mode,
            pid=pid,
            pgid=pid,
            command=("stub", profile),
            log_path=instance_log_path(instance),
            identity={"stable_identity": f"darwin:{pid}:1:2"},
            process=child,
        )
        self.children[instance.name] = child
        self.launched[instance.name] = launched
        return launched

    def child_at(self, pgid: int) -> StubChild:
        """The child whose process group is ``pgid``."""
        name = next(one for one, live in self.launched.items() if live.pgid == pgid)
        return self.children[name]


class FakeTime:
    """A monotonic clock that only advances when the loop sleeps.

    Paired deliberately: a fake sleep against a real clock would turn every
    bounded wait in the supervisor into a busy spin for the real duration.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += max(seconds, 0.0)


class RecordingSignals:
    """Stands in for ``os.killpg``/``os.kill``, and kills the stub on cue."""

    def __init__(
        self,
        launcher: StubLauncher,
        *,
        dies_on: int | None = signal.SIGKILL,
        returncode: int = -int(signal.SIGKILL),
    ) -> None:
        self.launcher = launcher
        self.dies_on = dies_on
        self.returncode = returncode
        self.groups: list[tuple[int, int]] = []
        self.pids: list[tuple[int, int]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> RecordingSignals:
        monkeypatch.setattr(os, "killpg", self.killpg)
        monkeypatch.setattr(os, "kill", self.kill)
        return self

    def killpg(self, pgid: int, sig: int) -> None:
        self.groups.append((pgid, sig))
        if self.dies_on is not None and sig == self.dies_on:
            self.launcher.child_at(pgid).finish(self.returncode)

    def kill(self, pid: int, sig: int) -> None:
        self.pids.append((pid, sig))


def start_stubbed(
    root: Path,
    *names: str,
    launcher: StubLauncher | None = None,
    timing: FakeTime | None = None,
    **kwargs: Any,
) -> tuple[FleetSupervisor, StubLauncher, FleetPlan]:
    """Prepare and start a fleet of ``names`` over stub children."""
    fleet_path, instances = make_fleet(root, *names)
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    spawn = launcher or StubLauncher()
    beat = timing or FakeTime()
    supervisor = start_fleet(
        plan, launch=spawn, sleep=beat.sleep, clock=beat.clock, **kwargs
    )
    return supervisor, spawn, plan


def state_of(plan: FleetPlan) -> dict[str, tuple[str, str | None]]:
    """The state file as written: instance name to ``(state, exit_reason)``.

    Read through :func:`load_fleet_state` rather than ``json.loads`` so the file
    is proved to satisfy the reader's own strictness — a supervisor that wrote a
    record ``nanobot fleet status`` would refuse is a supervisor reporting
    nothing at all.
    """
    return {
        record.name: (record.state, record.exit_reason)
        for record in load_fleet_state(plan.state_path)
    }


# --------------------------------------------------------------------------
# Preparing the ground
# --------------------------------------------------------------------------


def test_preparing_a_fleet_creates_paths_and_builds_every_profile(tmp_path) -> None:
    """One call that leaves the fleet startable, and says so by its outputs."""
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a", "b")

    plan = prepare_fleet(instances, fleet_path=fleet_path)

    assert plan.instances == instances
    assert plan.fleet_path == fleet_path
    assert plan.state_path == fleet_state_path(fleet_path)
    assert plan.state_path.read_text(encoding="utf-8") == "[]\n"
    assert stat.S_IMODE(plan.state_path.stat().st_mode) == STATE_FILE_MODE
    assert sorted(plan.profiles) == ["a", "b"]
    for instance in instances:
        assert instance.workspace.is_dir()
        assert instance_log_path(instance).parent.is_dir()
    # Each profile denies the other instance and never its own directories.
    assert str(instances[1].workspace) in plan.profiles["a"]
    assert str(instances[0].workspace) not in plan.profiles["a"].split("; peer", 1)[1]


def test_profiles_cannot_be_built_before_the_paths_they_deny_exist(tmp_path) -> None:
    """Why :func:`prepare_fleet` is one ordered step rather than three calls.

    Seatbelt accepts a rule naming a path that does not resolve, matches nothing,
    confines nothing and reports no error — so the profile builder refuses such a
    rule outright. That makes "create the directories and the state file first"
    a correctness requirement, and this test proves the requirement is real
    rather than inherited from a comment: the same build that fails on a bare
    fleet succeeds once ``prepare_fleet`` has run.
    """
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a", "b")

    with pytest.raises(SeatbeltProfileError):
        build_fleet_profiles(
            instances, fleet_path=fleet_path, state_path=fleet_state_path(fleet_path)
        )

    assert sorted(prepare_fleet(instances, fleet_path=fleet_path).profiles) == ["a", "b"]


def test_preparing_a_fleet_twice_keeps_the_state_of_the_first(tmp_path) -> None:
    """``prepare_fleet`` must not truncate a running fleet's state file."""
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    plan.state_path.write_text("[]  \n", encoding="utf-8")

    prepare_fleet(instances, fleet_path=fleet_path)

    assert plan.state_path.read_text(encoding="utf-8") == "[]  \n"


# --------------------------------------------------------------------------
# Starting every instance
# --------------------------------------------------------------------------


def test_starting_a_fleet_launches_every_instance_and_publishes_it(tmp_path) -> None:
    """The fleet is visible to another shell before the loop runs, not after."""
    supervisor, launcher, plan = start_stubbed(tmp_path.resolve(), "a", "b")

    assert launcher.calls == ["a", "b"]
    assert [record.name for record in supervisor.records] == ["a", "b"]
    assert state_of(plan) == {"a": ("running", None), "b": ("running", None)}
    written = {record.name: record for record in load_fleet_state(plan.state_path)}
    assert written["a"].pid == launcher.launched["a"].pid
    assert written["a"].memory_limit_mb == 512
    assert written["a"].identity == launcher.launched["a"].identity


def test_each_instance_is_started_under_its_own_profile(tmp_path) -> None:
    """A profile mix-up would confine an instance out of its own workspace."""
    supervisor, launcher, plan = start_stubbed(tmp_path.resolve(), "a", "b")

    for name, launched in launcher.launched.items():
        assert launched.command[1] == plan.profiles[name]
    assert len(supervisor.records) == 2


def test_an_instance_with_no_profile_is_refused_and_nothing_is_started(tmp_path) -> None:
    """Refusing beats launching unconfined: the fleet would look identical."""
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a", "b")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    without_b = FleetPlan(
        instances=plan.instances,
        fleet_path=plan.fleet_path,
        state_path=plan.state_path,
        profiles={"a": plan.profiles["a"]},
    )
    launcher = StubLauncher()
    signals = RecordingSignals(launcher, dies_on=signal.SIGTERM, returncode=-15)
    timing = FakeTime()

    with pytest.MonkeyPatch.context() as patch:
        signals.install(patch)
        with pytest.raises(InstanceLaunchError, match="no Seatbelt profile"):
            start_fleet(
                without_b, launch=launcher, sleep=timing.sleep, clock=timing.clock
            )

    assert launcher.calls == ["a"]
    assert state_of(plan) == {"a": ("exited", "signal")}


def test_a_partial_start_records_an_instance_it_could_not_kill(tmp_path) -> None:
    """The one record that must never be dropped.

    The start failed, the supervisor is about to exit, and this instance refused
    both signals. It is a confined process still running with nothing watching
    it, and the state file is the only place anything could learn that it exists
    — so it is written as it is, running, rather than omitted or flattered into
    ``exited``.
    """
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a", "b")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    launcher = StubLauncher(fail_on="b")
    signals = RecordingSignals(launcher, dies_on=None)
    timing = FakeTime()

    with pytest.MonkeyPatch.context() as patch:
        signals.install(patch)
        with pytest.raises(InstanceLaunchError):
            start_fleet(plan, launch=launcher, sleep=timing.sleep, clock=timing.clock)

    pgid = launcher.launched["a"].pgid
    assert signals.groups == [(pgid, signal.SIGTERM), (pgid, signal.SIGKILL)]
    assert state_of(plan) == {"a": ("running", None)}


def test_a_failed_start_stops_what_it_already_started(tmp_path) -> None:
    """A half-started fleet is worse than none.

    The supervisor is about to exit with the launch error, so any instance it
    already started would keep running confined with nothing watching it — and
    with no state file entry, ``nanobot fleet stop`` could not find it either.
    """
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a", "b", "c")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    launcher = StubLauncher(fail_on="b")
    signals = RecordingSignals(launcher, dies_on=signal.SIGTERM)
    timing = FakeTime()

    with pytest.MonkeyPatch.context() as patch:
        signals.install(patch)
        with pytest.raises(InstanceLaunchError, match="stub launch failure"):
            start_fleet(plan, launch=launcher, sleep=timing.sleep, clock=timing.clock)

    assert launcher.calls == ["a", "b"]
    assert signals.groups == [(launcher.launched["a"].pgid, signal.SIGTERM)]
    # Only the instance that actually started is recorded, and it is not running.
    assert state_of(plan) == {"a": ("exited", "signal")}


def test_a_start_that_launches_nothing_still_replaces_stale_state(tmp_path) -> None:
    """A previous run's records would otherwise be read as this fleet's."""
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    stale = [
        {
            "name": "a",
            "pid": 4242,
            "state": "running",
            "exit_reason": None,
            "workspace": str(instances[0].workspace),
            "config_dir": str(instances[0].config_dir),
            "memory_limit_mb": 512,
        }
    ]
    plan.state_path.write_text(json.dumps(stale), encoding="utf-8")
    timing = FakeTime()

    with pytest.raises(InstanceLaunchError):
        start_fleet(
            plan,
            launch=StubLauncher(fail_on="a"),
            sleep=timing.sleep,
            clock=timing.clock,
        )

    assert state_of(plan) == {}


def test_the_default_spawn_step_is_the_confined_launcher(tmp_path, monkeypatch) -> None:
    """Without an injected launcher, instances go through ``launch_instance``.

    The argv order that makes Seatbelt outermost, the ``env -i`` link and the new
    session that makes an instance's tree addressable all live in that function.
    A supervisor that spawned children itself would have reimplemented all three,
    and could weaken any of them without a single test noticing.
    """
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    launcher = StubLauncher()
    seen: list[tuple[str, str, str | None]] = []

    def fake(
        instance: ResolvedInstance,
        *,
        profile: str,
        python_executable: str | None = None,
    ) -> LaunchedInstance:
        seen.append((instance.name, profile, python_executable))
        return launcher(instance, profile)

    monkeypatch.setattr(supervisor_module, "launch_instance", fake)
    supervisor = start_fleet(plan, python_executable="/opt/python")

    assert seen == [("a", plan.profiles["a"], "/opt/python")]
    assert [one.state for one in supervisor.records] == ["running"]


def test_the_supervisor_refuses_processes_that_are_not_the_plans(tmp_path) -> None:
    """A record without a handle could never be reaped, and would read as live."""
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a", "b")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    launcher = StubLauncher()
    only_a = launcher(instances[0], plan.profiles["a"])

    with pytest.raises(ValueError, match="do not match the plan"):
        FleetSupervisor(plan, [only_a])


# --------------------------------------------------------------------------
# One instance dies; the fleet does not
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("returncode", "reason"),
    [(-int(signal.SIGKILL), "signal"), (-int(signal.SIGTERM), "signal"), (1, "exit"), (0, "exit")],
)
def test_an_exit_is_recorded_with_a_reason_while_its_peer_keeps_running(
    tmp_path, returncode, reason
) -> None:
    """The acceptance criterion, over stub children.

    The dead instance is reported as exited with a reason an operator can act on;
    the live one is untouched. Both halves matter: a supervisor that took the
    fleet down with one instance would make every instance a single point of
    failure for the others.
    """
    supervisor, launcher, plan = start_stubbed(tmp_path.resolve(), "a", "b")
    launcher.children["a"].finish(returncode)

    exits = supervisor.tick()

    assert [(one.name, one.returncode, one.reason) for one in exits] == [
        ("a", returncode, reason)
    ]
    assert state_of(plan) == {"a": ("exited", reason), "b": ("running", None)}
    assert [one.name for one in supervisor.running] == ["b"]


def test_a_sweep_with_no_exit_does_not_rewrite_the_state_file(tmp_path) -> None:
    """State is written on transitions, which is what makes them transitions.

    Rewriting every tick would churn a file other shells read, for no new
    information — and would hide a genuinely missed transition in the noise.
    """
    supervisor, _, plan = start_stubbed(tmp_path.resolve(), "a")
    before = plan.state_path.stat().st_mtime_ns

    assert supervisor.tick() == ()

    assert plan.state_path.stat().st_mtime_ns == before


def test_a_hanging_instance_does_not_delay_observing_its_peers_exit(tmp_path) -> None:
    """Reaping is non-blocking for every instance, not just the first.

    ``a`` is swept before ``b`` and never exits. If the sweep waited on it — the
    obvious way to reap a child — ``b``'s death would go unnoticed for as long as
    ``a`` kept running, which is indefinitely.
    """
    supervisor, launcher, plan = start_stubbed(tmp_path.resolve(), "a", "b")
    launcher.children["b"].finish(3)

    exits = supervisor.tick()

    assert [one.name for one in exits] == ["b"]
    assert state_of(plan) == {"a": ("running", None), "b": ("exited", "exit")}


def test_an_exited_instance_is_never_probed_or_started_again(tmp_path) -> None:
    """No restart, and no second look at a pid that may have been recycled."""
    supervisor, launcher, plan = start_stubbed(tmp_path.resolve(), "a", "b")
    launcher.children["a"].finish(-int(signal.SIGKILL))
    supervisor.tick()
    polls = launcher.children["a"].polls

    launcher.children["b"].finish(0)
    supervisor.run(handle_signals=False)

    assert launcher.calls == ["a", "b"]
    assert launcher.children["a"].polls == polls
    assert state_of(plan) == {"a": ("exited", "signal"), "b": ("exited", "exit")}


def test_a_declared_reason_survives_the_reap_that_observes_the_signal(tmp_path) -> None:
    """How the memory cap will record ``"memory"`` rather than ``"signal"``.

    The cap kills the tree itself, so the reaper would otherwise see only that a
    signal arrived and publish the weaker answer. Declaring the reason before the
    kill is what keeps the cause attached to the effect.
    """
    supervisor, launcher, plan = start_stubbed(tmp_path.resolve(), "a", "b")

    supervisor.record_exit_reason("a", "memory")
    launcher.children["a"].finish(-int(signal.SIGKILL))
    exits = supervisor.tick()

    assert [one.reason for one in exits] == ["memory"]
    assert state_of(plan) == {"a": ("exited", "memory"), "b": ("running", None)}


def test_a_declared_reason_is_not_reused_by_a_later_instance(tmp_path) -> None:
    """It describes one death, so it is consumed by the exit it explains."""
    supervisor, launcher, plan = start_stubbed(tmp_path.resolve(), "a", "b")
    supervisor.record_exit_reason("a", "memory")
    launcher.children["a"].finish(-int(signal.SIGKILL))
    supervisor.tick()

    launcher.children["b"].finish(-int(signal.SIGKILL))
    supervisor.tick()

    assert state_of(plan) == {"a": ("exited", "memory"), "b": ("exited", "signal")}


@pytest.mark.parametrize(
    ("name", "reason", "error"),
    [("a", "boom", ValueError), ("nobody", "memory", KeyError)],
)
def test_declaring_a_reason_the_state_file_would_refuse_is_rejected(
    tmp_path, name, reason, error
) -> None:
    """The writer must not be able to produce a record the reader refuses."""
    supervisor, _, _ = start_stubbed(tmp_path.resolve(), "a")

    with pytest.raises(error):
        supervisor.record_exit_reason(name, reason)


class MuteHandle:
    """A child handle that cannot be polled at all."""

    pid = STUB_PGID_BASE + 7


class BrokenHandle:
    """A child handle whose poll fails, as a closed or inherited one would."""

    pid = STUB_PGID_BASE + 8

    def poll(self) -> int | None:
        raise OSError("this handle cannot answer")


@pytest.mark.parametrize("handle", [MuteHandle, BrokenHandle])
def test_a_reason_is_withheld_when_the_manner_of_death_is_unknown(
    tmp_path, handle
) -> None:
    """A handle that cannot report is honest silence, not a guessed reason.

    Liveness still has a second source — ``process_is_running`` — so the exit is
    detected either way. The manner of it has only one source, the handle, so
    ``exit_reason`` stays null rather than becoming a plausible ``"exit"``.
    """
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    launched = LaunchedInstance(
        name="a",
        mode="serve",
        pid=handle.pid,
        pgid=handle.pid,
        command=("stub",),
        log_path=instance_log_path(instances[0]),
        identity={},
        process=handle(),
    )
    supervisor = FleetSupervisor(plan, [launched])

    exits = supervisor.tick()

    # No such pid exists, so ``process_is_running`` is the only available answer.
    assert [(one.name, one.returncode, one.reason) for one in exits] == [("a", None, None)]
    assert state_of(plan) == {"a": ("exited", None)}


@pytest.mark.parametrize(
    ("returncode", "reason"),
    [(None, None), (0, "exit"), (2, "exit"), (-1, "signal"), (-9, "signal")],
)
def test_return_codes_map_to_the_two_reasons_the_supervisor_can_observe(
    returncode, reason
) -> None:
    """``subprocess`` reports a signalled child as a negative code; publish that."""
    assert exit_reason_for(returncode) == reason


# --------------------------------------------------------------------------
# The loop itself
# --------------------------------------------------------------------------


def test_the_loop_runs_until_every_instance_has_exited(tmp_path) -> None:
    """Finite by construction: instances only ever move running to exited."""
    timing = FakeTime()
    supervisor, launcher, plan = start_stubbed(tmp_path.resolve(), "a", "b", timing=timing)
    order = iter(["a", "b"])

    def tock(seconds: float) -> None:
        timing.now += seconds
        launcher.children[next(order, "b")].finish(0)

    supervisor._sleep = tock
    final = supervisor.run(handle_signals=False)

    assert [(one.name, one.state) for one in final] == [("a", "exited"), ("b", "exited")]
    assert state_of(plan) == {"a": ("exited", "exit"), "b": ("exited", "exit")}


def test_the_loop_sleeps_at_the_configured_interval(tmp_path) -> None:
    """Which bounds how late the memory cap can notice a tree over its limit."""
    timing = FakeTime()
    supervisor, launcher, _ = start_stubbed(tmp_path.resolve(), "a", timing=timing)

    def tock(seconds: float) -> None:
        timing.sleep(seconds)
        launcher.children["a"].finish(0)

    supervisor._sleep = tock
    supervisor.run(handle_signals=False)

    assert timing.slept == [DEFAULT_POLL_INTERVAL_SECONDS]


@pytest.mark.parametrize("interval", [0, -1.0, MAX_POLL_INTERVAL_SECONDS + 0.001, 5.0])
def test_an_interval_the_memory_cap_could_not_honour_is_refused(tmp_path, interval) -> None:
    """The cap must kill within five seconds of a crossing, so it samples often.

    Refused in the constructor rather than documented, because a supervisor built
    with a coarse interval would pass every test in this file and quietly miss
    that deadline.
    """
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    launcher = StubLauncher()

    with pytest.raises(ValueError, match="poll_interval"):
        FleetSupervisor(
            plan,
            [launcher(instances[0], plan.profiles["a"])],
            poll_interval=interval,
        )


def test_the_longest_permitted_interval_is_accepted(tmp_path) -> None:
    """The boundary is inclusive; one second is the spec's own ceiling."""
    supervisor, _, _ = start_stubbed(
        tmp_path.resolve(), "a", poll_interval=MAX_POLL_INTERVAL_SECONDS
    )

    assert supervisor.poll_interval == MAX_POLL_INTERVAL_SECONDS


def test_run_fleet_prepares_starts_and_supervises_in_one_call(tmp_path) -> None:
    """The whole of ``fleet start`` after validation, minus the self-probe."""
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a")
    launcher = StubLauncher()

    def launch(instance: ResolvedInstance, profile: str) -> LaunchedInstance:
        launched = launcher(instance, profile)
        launcher.children[instance.name].finish(0)
        return launched

    final = run_fleet(
        instances, fleet_path=fleet_path, launch=launch, handle_signals=False
    )

    assert [(one.name, one.state, one.exit_reason) for one in final] == [
        ("a", "exited", "exit")
    ]
    assert state_of(
        FleetPlan(
            instances=instances,
            fleet_path=fleet_path,
            state_path=fleet_state_path(fleet_path),
            profiles={},
        )
    ) == {"a": ("exited", "exit")}


# --------------------------------------------------------------------------
# Stopping
# --------------------------------------------------------------------------


def test_shutdown_signals_the_process_group_and_escalates(tmp_path) -> None:
    """The group, because an instance's shell tool may have left children.

    ``SIGTERM`` first so a well-behaved instance can close its session store,
    then ``SIGKILL`` for whatever ignored it — the escalation
    ``process_runtime`` already uses for a single managed process.
    """
    launcher = StubLauncher()
    timing = FakeTime()
    supervisor, _, plan = start_stubbed(
        tmp_path.resolve(), "a", launcher=launcher, timing=timing
    )
    signals = RecordingSignals(launcher, dies_on=signal.SIGKILL)

    with pytest.MonkeyPatch.context() as patch:
        signals.install(patch)
        supervisor.shutdown(grace=1.0)

    pgid = launcher.launched["a"].pgid
    assert signals.groups == [(pgid, signal.SIGTERM), (pgid, signal.SIGKILL)]
    assert signals.pids == []
    assert state_of(plan) == {"a": ("exited", "signal")}


def test_an_instance_that_leaves_on_sigterm_is_not_killed(tmp_path) -> None:
    """No escalation past what was needed, and the grace is not spent waiting."""
    launcher = StubLauncher()
    supervisor, _, plan = start_stubbed(tmp_path.resolve(), "a", launcher=launcher)
    signals = RecordingSignals(launcher, dies_on=signal.SIGTERM, returncode=0)

    with pytest.MonkeyPatch.context() as patch:
        signals.install(patch)
        supervisor.shutdown(grace=5.0)

    assert signals.groups == [(launcher.launched["a"].pgid, signal.SIGTERM)]
    assert state_of(plan) == {"a": ("exited", "exit")}


def test_shutdown_leaves_an_already_exited_instance_alone(tmp_path) -> None:
    """Signalling a reaped pid would signal whoever holds it now."""
    launcher = StubLauncher()
    supervisor, _, plan = start_stubbed(tmp_path.resolve(), "a", "b", launcher=launcher)
    launcher.children["a"].finish(0)
    signals = RecordingSignals(launcher, dies_on=signal.SIGTERM, returncode=0)

    with pytest.MonkeyPatch.context() as patch:
        signals.install(patch)
        supervisor.shutdown(grace=1.0)

    assert signals.groups == [(launcher.launched["b"].pgid, signal.SIGTERM)]
    assert state_of(plan) == {"a": ("exited", "exit"), "b": ("exited", "exit")}


def test_the_supervisors_own_process_group_is_never_signalled(tmp_path) -> None:
    """The guard that stops a mutation from presenting as the fleet vanishing.

    An instance's group is its tree only because it was spawned into a new
    session. If that were ever dropped, the child would join the supervisor's
    group and a group-directed ``SIGKILL`` would take down the supervisor and
    every other instance — silently, with no failed assertion anywhere.
    """
    fleet_path, instances = make_fleet(tmp_path.resolve(), "a")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    launcher = StubLauncher()
    launched = launcher(instances[0], plan.profiles["a"])
    own_group = LaunchedInstance(
        name=launched.name,
        mode=launched.mode,
        pid=launched.pid,
        pgid=os.getpgid(0),
        command=launched.command,
        log_path=launched.log_path,
        identity=launched.identity,
        process=launched.process,
    )
    timing = FakeTime()
    supervisor = FleetSupervisor(plan, [own_group], sleep=timing.sleep, clock=timing.clock)
    signals = RecordingSignals(launcher, dies_on=None)

    with pytest.MonkeyPatch.context() as patch:
        signals.install(patch)
        supervisor.shutdown(grace=1.0)

    assert signals.groups == []
    assert signals.pids == [
        (launched.pid, signal.SIGTERM),
        (launched.pid, signal.SIGKILL),
    ]


def test_a_group_that_refuses_the_signal_falls_back_to_the_instance_pid(
    tmp_path,
) -> None:
    """One more attempt beats none when the group cannot be addressed.

    A group-directed signal can fail for reasons the instance process itself does
    not share, and the instance is the process that matters most — it is the one
    holding the port and the session store.
    """
    launcher = StubLauncher()
    timing = FakeTime()
    supervisor, _, _ = start_stubbed(
        tmp_path.resolve(), "a", launcher=launcher, timing=timing
    )
    signals = RecordingSignals(launcher, dies_on=None)

    def refuse(pgid: int, sig: int) -> None:
        raise PermissionError("not permitted")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "killpg", refuse)
        patch.setattr(os, "kill", signals.kill)
        supervisor.shutdown(grace=1.0)

    pid = launcher.launched["a"].pid
    assert signals.pids == [(pid, signal.SIGTERM), (pid, signal.SIGKILL)]


def test_a_tree_that_cannot_be_walked_still_gets_the_group_signal(tmp_path) -> None:
    """An unreadable tree must not turn a kill into no kill at all.

    The descendant walk is a ``ctypes`` call into ``libproc`` and can fail for
    reasons the instance has nothing to do with. Signalling the group is still
    the largest correct thing to do; withholding it would leave the instance and
    the tree alive because the supervisor could not enumerate the tree.
    """
    launcher = StubLauncher()
    timing = FakeTime()
    supervisor, _, _ = start_stubbed(
        tmp_path.resolve(), "a", launcher=launcher, timing=timing
    )
    signals = RecordingSignals(launcher, dies_on=signal.SIGTERM, returncode=-15)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(supervisor_module, "process_tree_pids", lambda _pgid: None)
        signals.install(patch)
        supervisor.shutdown(grace=1.0)

    assert signals.groups == [(launcher.launched["a"].pgid, signal.SIGTERM)]
    assert signals.pids == []


def test_a_stop_request_ends_the_loop_and_takes_the_fleet_with_it(tmp_path) -> None:
    """A foreground supervisor that returned alone would orphan its instances."""
    launcher = StubLauncher()
    timing = FakeTime()
    supervisor, _, plan = start_stubbed(
        tmp_path.resolve(), "a", "b", launcher=launcher, timing=timing
    )
    signals = RecordingSignals(launcher, dies_on=signal.SIGTERM, returncode=-15)

    def tock(seconds: float) -> None:
        timing.sleep(seconds)
        supervisor.request_stop()

    supervisor._sleep = tock
    with pytest.MonkeyPatch.context() as patch:
        signals.install(patch)
        final = supervisor.run(handle_signals=False)

    assert supervisor.stop_requested is True
    assert [one.state for one in final] == ["exited", "exited"]
    assert state_of(plan) == {"a": ("exited", "signal"), "b": ("exited", "signal")}
    assert launcher.calls == ["a", "b"]


@pytest.mark.parametrize("stop_signal", STOP_SIGNALS)
def test_a_stop_signal_ends_the_foreground_loop_and_handlers_are_restored(
    tmp_path, stop_signal
) -> None:
    """Ctrl-C on ``nanobot fleet start`` must stop the fleet, not abandon it.

    Driven through the real ``signal`` machinery rather than the flag, because
    the handler install is the part that can be wrong: without it the supervisor
    would die and leave confined processes running whose pids nothing would ever
    update again.
    """
    launcher = StubLauncher()
    timing = FakeTime()
    supervisor, _, plan = start_stubbed(
        tmp_path.resolve(), "a", launcher=launcher, timing=timing
    )
    signals = RecordingSignals(launcher, dies_on=signal.SIGTERM, returncode=-15)
    before = {one: signal.getsignal(one) for one in STOP_SIGNALS}

    def tock(seconds: float) -> None:
        timing.sleep(seconds)
        # Checked before the signal is raised rather than after. If the loop ever
        # stopped installing handlers, the default disposition for ``SIGTERM``
        # would kill the test runner outright — a mutation that presents as a
        # silent death with an empty log instead of a failed assertion. This is
        # what keeps it legible.
        for one in STOP_SIGNALS:
            installed = signal.getsignal(one)
            assert installed not in (signal.SIG_DFL, signal.SIG_IGN, before[one]), (
                f"the loop installed no stop handler for {one!r}"
            )
        os.kill(os.getpid(), stop_signal)

    supervisor._sleep = tock
    with pytest.MonkeyPatch.context() as patch:
        # Only the group signal is faked; the process signal above is real, and
        # must reach the handler ``run`` installs.
        patch.setattr(os, "killpg", signals.killpg)
        final = supervisor.run()

    assert {one: signal.getsignal(one) for one in STOP_SIGNALS} == before
    assert [one.state for one in final] == ["exited"]
    assert state_of(plan) == {"a": ("exited", "signal")}


# --------------------------------------------------------------------------
# Bookkeeping that fails
# --------------------------------------------------------------------------


def test_a_state_write_failure_is_retried_rather_than_ending_the_fleet(tmp_path) -> None:
    """Reporting badly is not a reason to stop serving.

    A supervisor that died on its own write error would turn a full disk into an
    outage; one that dropped the transition would report a dead instance as
    serving forever. The transition stays pending until it lands.
    """
    root = tmp_path.resolve()
    fleet_path, instances = make_fleet(root, "a", "b")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    launcher = StubLauncher()
    unwritable = FleetPlan(
        instances=plan.instances,
        fleet_path=plan.fleet_path,
        state_path=root / "gone" / "fleet.json.state.json",
        profiles=plan.profiles,
    )
    seen: list[FleetStateError] = []
    supervisor = FleetSupervisor(
        unwritable,
        [launcher(instance, plan.profiles[instance.name]) for instance in instances],
        on_state_error=seen.append,
    )

    launcher.children["a"].finish(-int(signal.SIGKILL))
    exits = supervisor.tick()

    assert [one.name for one in exits] == ["a"]
    assert [error.kind for error in seen] == ["io_error"]
    assert supervisor.state_error is not None
    assert not unwritable.state_path.exists()

    unwritable.state_path.parent.mkdir()
    assert supervisor.tick() == ()

    assert supervisor.state_error is None
    assert state_of(unwritable) == {"a": ("exited", "signal"), "b": ("running", None)}


# --------------------------------------------------------------------------
# Real processes, the real kernel policy
# --------------------------------------------------------------------------

# A stand-in for the instance interpreter that keeps working after it starts, so
# "still serving" is observable from outside. It writes a counter into its own
# workspace — which its profile allows and every peer's denies — and never exits
# on its own, which is what makes killing it the only way it can die.
#
# One stub serves the whole fleet: it locates its own workspace from the argv the
# launcher builds (``-m nanobot <mode> --config <path>``), so the supervisor's
# real default launcher can be used unmodified rather than a per-instance
# injection. That also re-pins the argv shape from the other side — a stub that
# did not find ``--config <existing file>`` there exits 64 instead of reporting in.
#
# Free of ``ps`` deliberately: ``sandbox-exec`` refuses to exec a setgid binary
# even under ``(allow default)``, and ``/bin/ps`` is setgid ``kmem``.
HEARTBEAT_STUB = """#!/bin/sh
# argv is "-m nanobot <mode> --config <path>", so the config path is the fifth word.
config="$5"
[ "$4" = "--config" ] && [ -f "$config" ] || exit 64
heartbeat="$(dirname "$config")/workspace/heartbeat"
count=0
while : ; do
    count=$((count + 1))
    printf '%s\\n' "$count" > "$heartbeat"
    sleep 0.05
done
"""


def write_heartbeat_stub(path: Path) -> Path:
    """Write the shared interpreter stub, executable."""
    path.write_text(HEARTBEAT_STUB, encoding="utf-8")
    path.chmod(0o700)
    return path


def wait_until(predicate: Any, what: str) -> None:
    """Poll ``predicate`` until it holds, or fail naming what never happened.

    There is no ``pytest-timeout`` in this repo, so every wait carries its own
    deadline, following ``tests/webui/test_gateway_webui_smoke.py``.
    """
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail(f"{what} did not happen within {READY_TIMEOUT_SECONDS}s")


def heartbeat_of(instance: ResolvedInstance) -> int:
    """The counter the instance last wrote, or ``0`` before its first write."""
    path = instance.workspace / "heartbeat"
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8").strip()
    return int(text) if text.isdigit() else 0


@confinement_available
def test_killing_one_real_instance_leaves_its_peer_serving(tmp_path) -> None:
    """The acceptance criterion, against real confined processes and real ``kill``.

    Everything above proves the loop does the right thing with an exit it is
    handed. This proves the exits are real: two instances are started under the
    kernel's own policy, one is killed the way an operator would kill it, and the
    other must go on doing work — not merely hold a pid. The heartbeat counter is
    what makes "and serving" observable; a test that only checked liveness would
    pass for an instance wedged by its peer's death.
    """
    root = tmp_path.resolve()
    fleet_path, instances = make_fleet(root, "a", "b")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    stub = write_heartbeat_stub(root / "stub.sh")

    # No injected launcher: this is the production spawn path, so the argv, the
    # Seatbelt wrapper and the new session are all the real ones.
    supervisor = start_fleet(plan, python_executable=str(stub))
    try:
        for instance in instances:
            wait_until(lambda i=instance: heartbeat_of(i) > 0, f"{instance.name} started")

        victim = supervisor.launched("a")
        assert victim.pgid == victim.pid
        os.killpg(victim.pgid, signal.SIGKILL)
        wait_until(lambda: any(supervisor.tick()), "the supervisor observed the kill")

        assert state_of(plan) == {"a": ("exited", "signal"), "b": ("running", None)}
        # Still doing work, not merely holding a pid.
        was = heartbeat_of(instances[1])
        wait_until(lambda: heartbeat_of(instances[1]) > was, "b kept serving")
        # Nothing is replaced: further sweeps report no transition, a keeps the
        # pid it died with, and no new process is ever spawned for it.
        assert supervisor.tick() == ()
        assert supervisor.launched("a") is victim
        assert supervisor.records[0].pid == victim.pid
    finally:
        supervisor.shutdown()

    assert state_of(plan)["b"][0] == "exited"
    assert [one.state for one in supervisor.records] == ["exited", "exited"]


# A stub that starts a descendant in a session of its own, which is what nanobot's
# shell tool does to every command it runs (``agent/tools/shell.py`` spawns with
# ``start_new_session=True`` so it can kill a runaway command by group). The
# descendant is therefore out of the instance's process group from its first
# instruction, and a group-directed signal never reaches it.
STRAY_STUB = """#!/bin/sh
config="$5"
[ "$4" = "--config" ] && [ -f "$config" ] || exit 64
workspace="$(dirname "$config")/workspace"
__PYTHON__ -c 'import os, sys, time
os.setsid()
with open(sys.argv[1], "w") as handle:
    handle.write(str(os.getpid()))
time.sleep(600)' "$workspace/stray.pid" &
sleep 600
"""


def write_stray_stub(path: Path, python_executable: str) -> Path:
    """Write the stub with the interpreter baked in — ``env -i`` strips little.

    The path is absolute because the descendant must be started the same way on
    any host, not found through whatever ``PATH`` the minimal environment carries.
    """
    path.write_text(STRAY_STUB.replace("__PYTHON__", python_executable), encoding="utf-8")
    path.chmod(0o700)
    return path


@confinement_available
def test_signalling_a_tree_reaches_the_descendants_that_left_the_group(tmp_path) -> None:
    """The group is not the tree, and the difference is where the memory goes.

    An instance's shell tool starts every command in a new session, so the
    process a memory cap fires at is never in the instance's process group by the
    time it has allocated anything. A group-only ``SIGKILL`` would report the
    instance as killed and leave the allocation running, reparented and owned by
    nobody — the fleet would look tidy and the machine would not recover.
    """
    root = tmp_path.resolve()
    fleet_path, instances = make_fleet(root, "a")
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    stub = write_stray_stub(root / "stub.sh", sys.executable)
    stray_path = instances[0].workspace / "stray.pid"
    stray_pid = 0

    supervisor = start_fleet(plan, python_executable=str(stub))
    try:
        wait_until(lambda: stray_path.exists(), "the instance started a descendant")
        stray_pid = int(stray_path.read_text(encoding="utf-8").strip())
        launched = supervisor.launched("a")

        # The premise, checked rather than assumed: this descendant is genuinely
        # unreachable by a signal to the instance's group.
        assert os.getpgid(stray_pid) != launched.pgid
        assert stray_pid not in (process_group_pids(launched.pgid) or [])
        assert stray_pid in (process_tree_pids(launched.pgid) or [])

        signal_instance_tree(launched, signal.SIGKILL)

        wait_until(lambda: not process_is_running(stray_pid), "the descendant was killed")
        wait_until(lambda: any(supervisor.tick()), "the supervisor observed the kill")
        assert state_of(plan) == {"a": ("exited", "signal")}
    finally:
        supervisor.shutdown()
        if stray_pid:
            with suppress(ProcessLookupError, PermissionError):
                os.kill(stray_pid, signal.SIGKILL)

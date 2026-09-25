"""Tests for the fleet memory cap: the threshold, the deadline, the blast radius.

Two layers, deliberately separate.

The first drives :class:`~nanobot.fleet.cap.MemoryCap` with no fleet at all — a
name-to-limit mapping and a function returning numbers. That is the whole point
of the policy living in its own module: every threshold decision the fleet makes
is reachable without a process, a profile or a state file.

The second drives the real supervisor loop over stub children and an *injected*
sampler, which is how the acceptance criterion is met without arranging real
memory pressure. The sampler is a function of the fake clock, so "the tree
crossed its cap at t=1.0" is something the test states rather than approximates,
and the deadline can be asserted as a number instead of a hope.

The fake clock carries a deadline of its own and raises once it passes. That
guard is not decoration: a cap that never kills anything makes :meth:`run` spin
forever over a clock that costs nothing to advance, so the natural failure mode
of every mutation in this file is a hang with no output rather than a red test.
"""

from __future__ import annotations

import os
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nanobot.fleet.cap import (
    BYTES_PER_MB,
    KILL_DEADLINE_SECONDS,
    MAX_SAMPLE_INTERVAL_SECONDS,
    CapBreach,
    MemoryCap,
)
from nanobot.fleet.config import FleetInstance
from nanobot.fleet.instance import LaunchedInstance, instance_log_path
from nanobot.fleet.state import load_fleet_state
from nanobot.fleet.supervisor import (
    MAX_POLL_INTERVAL_SECONDS,
    FleetPlan,
    FleetSupervisor,
    prepare_fleet,
    start_fleet,
)
from nanobot.fleet.validate import ResolvedInstance

# Far above any pid macOS hands out, so a stub's process group can never collide
# with a real one — least of all the test runner's, which is the group a
# mis-aimed tree kill would reach.
STUB_PGID_BASE = 1_000_000

#: Long enough that a working cap never reaches it and short enough that a
#: broken one fails quickly instead of hanging.
FAKE_CLOCK_DEADLINE_SECONDS = 120.0


# --------------------------------------------------------------------------
# The policy on its own
# --------------------------------------------------------------------------


def cap(resident: int | None, *, limit_mb: int = 512) -> MemoryCap:
    """A one-instance cap named ``a`` whose tree always reads ``resident``."""
    return MemoryCap({"a": limit_mb}, sample=lambda _pgid: resident)


def test_a_megabyte_is_binary() -> None:
    """The conversion factor, as a literal, and its effect, as literals.

    Asserted against hard numbers rather than against ``BYTES_PER_MB`` itself:
    a test that converts through the constant it is checking re-derives from
    whatever the constant says and passes for every value it could possibly
    hold. Every fleet file's ``memoryLimitMb`` means the same thing ``ulimit``,
    container limits and ``ps`` mean by it, so a decimal conversion here would
    make every cap in every fleet quietly five percent looser than declared.
    """
    assert BYTES_PER_MB == 1_048_576
    assert cap(1_000_001, limit_mb=1).breach("a", STUB_PGID_BASE) is None
    assert cap(1_048_577, limit_mb=1).breach("a", STUB_PGID_BASE) is not None


def test_a_tree_under_its_limit_is_not_a_breach() -> None:
    assert cap(4 * BYTES_PER_MB).breach("a", STUB_PGID_BASE) is None


def test_a_tree_exactly_on_its_limit_is_not_a_breach() -> None:
    """Strictly over, because a tree that used what it was given used no more.

    The alternative makes every number in every fleet file mean one byte less
    than it says, which is the kind of off-by-one nobody discovers until an
    instance dies at precisely its declared cap.
    """
    assert cap(8 * BYTES_PER_MB, limit_mb=8).breach("a", STUB_PGID_BASE) is None


def test_a_tree_over_its_limit_reports_what_it_used() -> None:
    """The reading is carried, not just the verdict.

    An operator deciding whether the cap or the workload is wrong needs the
    number, and it is not recoverable from anywhere else once the tree is dead.
    """
    resident = 9 * BYTES_PER_MB
    breach = cap(resident, limit_mb=8).breach("a", STUB_PGID_BASE)

    assert breach == CapBreach(
        name="a",
        process_group=STUB_PGID_BASE,
        limit_mb=8,
        resident_bytes=resident,
    )
    assert breach.limit_bytes == 8 * BYTES_PER_MB
    assert breach.excess_bytes == BYTES_PER_MB
    assert "over its 8 MB limit" in breach.describe()


def test_an_unsampleable_tree_is_left_alone() -> None:
    """``None`` means "cannot tell", and the cap never kills what it cannot read.

    The sampler returns ``None`` for an unsupported platform and for a failed
    group walk. Treating either as a breach would kill a healthy confined
    service on the strength of a failed ``ctypes`` call; a tree that has really
    exited samples as ``0``, which is how "using nothing" stays distinguishable
    from "unknowable".
    """
    assert cap(None, limit_mb=1).breach("a", STUB_PGID_BASE) is None
    assert cap(0, limit_mb=1).breach("a", STUB_PGID_BASE) is None


def test_a_limit_that_is_not_positive_is_refused() -> None:
    """Re-checked here so the policy is honest standing alone.

    The fleet document already requires ``memoryLimitMb > 0``, but a zero limit
    reaching this class would make every instance breach on its first sample.
    """
    with pytest.raises(ValueError, match="b, c"):
        MemoryCap({"a": 1, "b": 0, "c": -1}, sample=lambda _pgid: 0)


def test_an_unknown_instance_is_an_error_rather_than_a_pass() -> None:
    """A caller and the fleet document disagreeing is not "no breach"."""
    with pytest.raises(KeyError):
        cap(10**12).breach("nobody", STUB_PGID_BASE)


def test_limits_are_published_read_only() -> None:
    policy = MemoryCap({"a": 64}, sample=lambda _pgid: 0)

    assert dict(policy.limits) == {"a": 64}
    assert policy.limit_bytes("a") == 64 * BYTES_PER_MB
    with pytest.raises(TypeError):
        policy.limits["a"] = 1  # type: ignore[index]


# --------------------------------------------------------------------------
# The timing rule
# --------------------------------------------------------------------------


def test_the_sampling_interval_is_at_most_one_second() -> None:
    """The acceptance criterion, as a literal.

    A threshold comparison enforces nothing about *when* a tree is killed; how
    often the number is taken is what does. The relation to the deadline is
    asserted too, since an interval at or above it could not be met even by a
    kill that took no time at all.
    """
    assert MAX_SAMPLE_INTERVAL_SECONDS == 1.0
    assert KILL_DEADLINE_SECONDS == 5.0
    assert MAX_SAMPLE_INTERVAL_SECONDS < KILL_DEADLINE_SECONDS
    # The supervisor's tick is the cap's sampling interval, not merely similar
    # to it: one loop takes both readings.
    assert MAX_POLL_INTERVAL_SECONDS == MAX_SAMPLE_INTERVAL_SECONDS


def test_a_supervisor_that_samples_too_slowly_is_refused(tmp_path) -> None:
    """Constructor-time, so the cap cannot be handed a loop that misses its own
    deadline. The boundary is inclusive: exactly one second is allowed."""
    plan, launcher = plan_for(tmp_path.resolve(), "a")

    with pytest.raises(ValueError, match="memory cap"):
        FleetSupervisor(plan, launcher.all(), poll_interval=1.01)

    assert (
        FleetSupervisor(
            plan, launcher.all(), poll_interval=MAX_SAMPLE_INTERVAL_SECONDS
        ).poll_interval
        == 1.0
    )


# --------------------------------------------------------------------------
# Wiring: stubs shared by the supervisor-level tests
# --------------------------------------------------------------------------


class StubChild:
    """A child that exists only to be polled, and to die when signalled.

    ``immortal`` makes it ignore the signal, which is how a test can hold an
    instance in the state the supervisor must keep handling: caught by the cap,
    killed, and still there on the next sweep.
    """

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.immortal = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        raise AssertionError("the supervisor must never block on one instance")

    def finish(self, returncode: int) -> None:
        if self.immortal:
            return
        self.returncode = returncode


class StubLauncher:
    """A spawn step handing back stub children in their own process groups."""

    def __init__(self) -> None:
        self.children: dict[str, StubChild] = {}
        self.launched: dict[str, LaunchedInstance] = {}

    def __call__(self, instance: ResolvedInstance, profile: str) -> LaunchedInstance:
        pid = STUB_PGID_BASE + len(self.launched) + 1
        child = StubChild(pid)
        self.children[instance.name] = child
        self.launched[instance.name] = LaunchedInstance(
            name=instance.name,
            mode=instance.mode,
            pid=pid,
            pgid=pid,
            command=("stub", profile),
            log_path=instance_log_path(instance),
            identity={"stable_identity": f"darwin:{pid}:1:2"},
            process=child,
        )
        return self.launched[instance.name]

    def all(self) -> list[LaunchedInstance]:
        return list(self.launched.values())

    def pgid(self, name: str) -> int:
        return self.launched[name].pgid

    def name_of(self, pgid: int) -> str:
        return next(one for one, live in self.launched.items() if live.pgid == pgid)


class FakeTime:
    """A monotonic clock that advances only when the loop sleeps.

    ``deadline`` is what keeps a broken cap reportable. :meth:`run` returns only
    once nothing is running, so an instance that is never killed makes the loop
    spin over a clock it costs nothing to advance — a hang with an empty log,
    not a failed assertion. Raising once the clock passes turns that into the
    sharp failure it should be, and the guard is the property under test.
    """

    def __init__(self, *, deadline: float = FAKE_CLOCK_DEADLINE_SECONDS) -> None:
        self.now = 0.0
        self.deadline = deadline

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)
        if self.now > self.deadline:
            raise AssertionError(
                f"the fleet was still running {self.now:.1f}s in; nothing was "
                f"ever killed for memory"
            )


class RecordingKills:
    """Stands in for ``os.killpg``/``os.kill``, and kills the stub on cue."""

    def __init__(self, launcher: StubLauncher, timing: FakeTime) -> None:
        self.launcher = launcher
        self.timing = timing
        self.groups: list[tuple[int, int]] = []
        self.pids: list[tuple[int, int]] = []
        #: Instance name to the clock reading of its first ``SIGKILL``.
        self.killed_at: dict[str, float] = {}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> RecordingKills:
        monkeypatch.setattr(os, "killpg", self.killpg)
        monkeypatch.setattr(os, "kill", self.kill)
        return self

    def killpg(self, pgid: int, sig: int) -> None:
        self.groups.append((pgid, sig))
        if sig != signal.SIGKILL:
            return
        name = self.launcher.name_of(pgid)
        self.killed_at.setdefault(name, self.timing.now)
        self.launcher.children[name].finish(-int(signal.SIGKILL))

    def kill(self, pid: int, sig: int) -> None:
        self.pids.append((pid, sig))


def make_instance(root: Path, name: str, *, memory_limit_mb: int) -> ResolvedInstance:
    """One instance laid out the way nanobot lays one out by itself."""
    config_dir = root / name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    return ResolvedInstance(
        name=name,
        entry=FleetInstance(
            config=str(config_path),
            mode="serve",
            memory_limit_mb=memory_limit_mb,
            env=[],
        ),
        config_path=config_path,
        config_dir=config_dir,
        workspace=config_dir / "workspace",
        port=None,
        port_setting="api.port",
    )


def plan_for(
    root: Path, *names: str, memory_limit_mb: int = 512
) -> tuple[FleetPlan, StubLauncher]:
    """A prepared fleet of ``names`` and the launcher that will stub it."""
    fleet_path = root / "fleet.json"
    fleet_path.write_text("{}", encoding="utf-8")
    instances = tuple(
        make_instance(root, name, memory_limit_mb=memory_limit_mb) for name in names
    )
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    launcher = StubLauncher()
    for instance in instances:
        launcher(instance, plan.profiles[instance.name])
    return plan, launcher


def start_capped(
    root: Path,
    *names: str,
    sample: Callable[[int], int | None],
    memory_limit_mb: int = 512,
    timing: FakeTime | None = None,
    **kwargs: Any,
) -> tuple[FleetSupervisor, StubLauncher, FleetPlan, FakeTime]:
    """Start a fleet of stub children under an injected sampler."""
    fleet_path = root / "fleet.json"
    fleet_path.write_text("{}", encoding="utf-8")
    instances = tuple(
        make_instance(root, name, memory_limit_mb=memory_limit_mb) for name in names
    )
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    launcher = StubLauncher()
    beat = timing or FakeTime()
    supervisor = start_fleet(
        plan,
        launch=launcher,
        sleep=beat.sleep,
        clock=beat.clock,
        sample_memory=sample,
        **kwargs,
    )
    return supervisor, launcher, plan, beat


def state_of(plan: FleetPlan) -> dict[str, tuple[str, str | None]]:
    """The state file as written, read through the reader's own strictness."""
    return {
        record.name: (record.state, record.exit_reason)
        for record in load_fleet_state(plan.state_path)
    }


# --------------------------------------------------------------------------
# Enforcement in the supervisor
# --------------------------------------------------------------------------


def test_an_instance_over_its_cap_has_its_whole_group_killed(
    tmp_path, monkeypatch
) -> None:
    """The acceptance criterion at the level of one sweep.

    The group and not the pid: the allocation that crosses a cap is typically in
    a child the instance's shell tool started, which is exactly why the metric
    is defined over the tree.
    """
    over = 600 * BYTES_PER_MB
    supervisor, launcher, plan, beat = start_capped(
        tmp_path.resolve(), "a", sample=lambda _pgid: over, memory_limit_mb=512
    )
    kills = RecordingKills(launcher, beat).install(monkeypatch)

    breaches = supervisor.enforce_memory_caps()

    assert [one.name for one in breaches] == ["a"]
    assert breaches[0].resident_bytes == over
    assert kills.groups == [(launcher.pgid("a"), signal.SIGKILL)]
    assert kills.pids == []
    assert supervisor.breaches == breaches


def test_the_cap_kills_rather_than_asks(tmp_path, monkeypatch) -> None:
    """``SIGKILL``, never ``SIGTERM``.

    A graceful signal can be caught, delayed or ignored by precisely the runaway
    allocation the cap exists to stop, and this is the one case where the
    instance has already proved it cannot be trusted with the machine.
    """
    supervisor, launcher, _plan, beat = start_capped(
        tmp_path.resolve(), "a", sample=lambda _pgid: 10**12, memory_limit_mb=1
    )
    kills = RecordingKills(launcher, beat).install(monkeypatch)

    supervisor.enforce_memory_caps()

    assert {sig for _pgid, sig in kills.groups} == {signal.SIGKILL}


def test_a_capped_instance_is_recorded_with_exit_reason_memory(
    tmp_path, monkeypatch
) -> None:
    """The reason survives the reaper, which would otherwise see only a signal.

    ``SIGKILL`` arrives at the child as return code ``-9``, which
    ``exit_reason_for`` classifies as ``"signal"``. Only the component that sent
    it can say it meant ``"memory"``, so the declaration has to come first — and
    the state file, which is the only thing another shell can read, has to carry
    it.
    """
    supervisor, launcher, plan, beat = start_capped(
        tmp_path.resolve(), "a", sample=lambda _pgid: 10**12, memory_limit_mb=1
    )
    RecordingKills(launcher, beat).install(monkeypatch)

    supervisor.tick()
    supervisor.tick()

    assert state_of(plan) == {"a": ("exited", "memory")}
    assert supervisor.records[0].exit_reason == "memory"


def test_a_breach_is_killed_and_reaped_on_the_same_sweep(
    tmp_path, monkeypatch
) -> None:
    """Enforcement runs before the reap, so no tick is wasted on a dead tree.

    Every tick spent not noticing comes out of the five-second budget.
    """
    supervisor, launcher, _plan, beat = start_capped(
        tmp_path.resolve(), "a", sample=lambda _pgid: 10**12, memory_limit_mb=1
    )
    RecordingKills(launcher, beat).install(monkeypatch)

    exits = supervisor.tick()

    assert [one.name for one in exits] == ["a"]
    assert [one.reason for one in exits] == ["memory"]


def test_other_instances_keep_running_when_one_is_capped(
    tmp_path, monkeypatch
) -> None:
    """One instance's cap is not the fleet's.

    The sampler answers per group, so only ``a`` is over. ``b`` must be
    untouched by every measure the supervisor has: not signalled, not recorded,
    still running.
    """
    root = tmp_path.resolve()
    fleet_path = root / "fleet.json"
    fleet_path.write_text("{}", encoding="utf-8")
    instances = (
        make_instance(root, "a", memory_limit_mb=512),
        make_instance(root, "b", memory_limit_mb=512),
    )
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    launcher = StubLauncher()
    beat = FakeTime()
    readings: dict[str, int] = {}
    supervisor = start_fleet(
        plan,
        launch=launcher,
        sleep=beat.sleep,
        clock=beat.clock,
        sample_memory=lambda pgid: readings.get(launcher.name_of(pgid), 0),
    )
    kills = RecordingKills(launcher, beat).install(monkeypatch)
    readings["a"] = 600 * BYTES_PER_MB
    readings["b"] = 8 * BYTES_PER_MB

    supervisor.tick()
    supervisor.tick()

    assert state_of(plan) == {"a": ("exited", "memory"), "b": ("running", None)}
    assert [one.name for one in supervisor.running] == ["b"]
    assert kills.groups == [(launcher.pgid("a"), signal.SIGKILL)]
    assert launcher.children["b"].returncode is None


def test_a_caught_instance_is_not_resampled_but_is_signalled_again(
    tmp_path, monkeypatch
) -> None:
    """A tree that falls back under its cap while dying is not reprieved.

    Its ``"memory"`` reason is already pending and its kill is already in
    flight, so a second reading has nothing to add and one bad one would undo
    both. The signal is repeated instead — one idempotent syscall, in case a
    first delivery raced a process joining the group.
    """
    root = tmp_path.resolve()
    sampled: list[int] = []

    def sample(process_group: int) -> int | None:
        sampled.append(process_group)
        return 10**12

    supervisor, launcher, _plan, beat = start_capped(
        root, "a", sample=sample, memory_limit_mb=1
    )
    kills = RecordingKills(launcher, beat).install(monkeypatch)
    # A child that refuses to die, so the instance stays running and eligible.
    launcher.children["a"].immortal = True

    assert supervisor.enforce_memory_caps()
    assert supervisor.enforce_memory_caps() == ()
    assert supervisor.enforce_memory_caps() == ()

    assert sampled == [launcher.pgid("a")]
    assert kills.groups == [(launcher.pgid("a"), signal.SIGKILL)] * 3


def test_an_exited_instance_is_never_sampled(tmp_path) -> None:
    """Nothing a second look could find but a recycled pid belonging elsewhere.

    Instances are never restarted, so an exited record is final; sampling one
    would measure whatever the kernel handed that group id next.
    """
    sampled: list[int] = []

    def sample(process_group: int) -> int | None:
        sampled.append(process_group)
        return 0

    supervisor, launcher, _plan, _beat = start_capped(
        tmp_path.resolve(), "a", sample=sample
    )
    launcher.children["a"].finish(0)
    supervisor.tick()
    sampled.clear()

    supervisor.tick()

    assert sampled == []


def test_an_unsampleable_fleet_is_left_running(tmp_path, monkeypatch) -> None:
    """Off macOS — and on a failed group walk — the cap does nothing at all.

    A supervisor that killed its whole fleet because it could not read a kernel
    struct would be far worse than one that never enforced a cap.
    """
    supervisor, launcher, plan, beat = start_capped(
        tmp_path.resolve(), "a", "b", sample=lambda _pgid: None
    )
    kills = RecordingKills(launcher, beat).install(monkeypatch)

    supervisor.tick()

    assert kills.groups == []
    assert state_of(plan) == {"a": ("running", None), "b": ("running", None)}


def test_an_instance_sharing_the_supervisors_group_is_never_capped(
    tmp_path, monkeypatch
) -> None:
    """The reading would be the supervisor's whole world, not one tree.

    An instance's group is its tree only because it was spawned with
    ``start_new_session``. If that were ever dropped every instance would
    measure the supervisor, itself and every peer, each would breach its own cap
    on the first tick, and the fleet would be killed off one pid at a time with
    the whole thing recorded as a memory problem. Found by the full suite rather
    than by reading the code: the same layout under a heavier test run put the
    runner's own group over a 512 MB cap.

    The refusal mirrors the one :func:`signal_instance_tree` already makes, and
    the sampler here is deliberately generous — anything it is asked about is
    over its cap — so only the skip can keep the fleet alive.
    """
    root = tmp_path.resolve()
    plan, launcher = plan_for(root, "a", memory_limit_mb=1)
    own = launcher.launched["a"]
    timing = FakeTime()
    supervisor = FleetSupervisor(
        plan,
        [
            LaunchedInstance(
                name=own.name,
                mode=own.mode,
                pid=own.pid,
                pgid=os.getpgid(0),
                command=own.command,
                log_path=own.log_path,
                identity=own.identity,
                process=own.process,
            )
        ],
        sleep=timing.sleep,
        clock=timing.clock,
        sample_memory=lambda _pgid: 10**12,
    )
    kills = RecordingKills(launcher, timing).install(monkeypatch)

    assert supervisor.enforce_memory_caps() == ()

    assert supervisor.breaches == ()
    assert kills.groups == []
    assert kills.pids == []
    assert [one.name for one in supervisor.running] == ["a"]


@pytest.mark.parametrize("ask_first", [True, False])
def test_a_fleet_being_shut_down_is_not_capped(
    tmp_path, monkeypatch, ask_first: bool
) -> None:
    """A fleet coming down by the operator's decision is not a cap breach.

    ``shutdown`` sweeps repeatedly while it waits out its grace period, and
    every one of those sweeps would otherwise re-sample instances that are
    already on their way out — recording the operator's stop as a memory kill
    and pre-empting the ``SIGTERM`` that makes the stop graceful at all.

    Run both ways round because ``shutdown`` is reachable without
    ``request_stop`` (``_abandon`` calls it directly on a fleet that failed to
    start), so it has to mark the stop itself rather than rely on its caller.
    """
    supervisor, launcher, plan, beat = start_capped(
        tmp_path.resolve(), "a", sample=lambda _pgid: 10**12, memory_limit_mb=1
    )
    kills = RecordingKills(launcher, beat).install(monkeypatch)

    if ask_first:
        supervisor.request_stop()
    supervisor.shutdown()

    assert supervisor.stop_requested is True
    assert supervisor.breaches == ()
    assert state_of(plan) == {"a": ("exited", "signal")}
    assert kills.groups[0] == (launcher.pgid("a"), signal.SIGTERM)


# --------------------------------------------------------------------------
# The deadline, through the real loop
# --------------------------------------------------------------------------


def test_a_tree_that_crosses_its_cap_dies_within_five_seconds(
    tmp_path, monkeypatch
) -> None:
    """The acceptance criterion, measured from the crossing rather than from the
    start of the run.

    The sampler is a function of the fake clock, so the crossing happens at a
    stated instant partway through the run and the deadline is the difference
    between two readings of the same clock. Nothing here spends wall-clock time,
    and the clock's own deadline means a cap that never fires fails loudly
    instead of spinning.
    """
    # Deliberately not a multiple of the poll interval: the worst case the
    # deadline has to survive is a tree crossing just *after* a sample, which
    # then goes unnoticed for a whole interval.
    crossing = 1.2
    timing = FakeTime()
    supervisor, launcher, plan, beat = start_capped(
        tmp_path.resolve(),
        "a",
        sample=lambda _pgid: 10**12 if timing.now >= crossing else 0,
        memory_limit_mb=512,
        timing=timing,
    )
    kills = RecordingKills(launcher, beat).install(monkeypatch)

    records = supervisor.run(handle_signals=False)

    assert kills.killed_at["a"] - crossing <= KILL_DEADLINE_SECONDS
    # Tighter than the criterion, and the reason the criterion is reachable:
    # detection costs at most one sampling interval.
    assert kills.killed_at["a"] - crossing <= MAX_SAMPLE_INTERVAL_SECONDS
    assert [(one.state, one.exit_reason) for one in records] == [("exited", "memory")]
    assert state_of(plan) == {"a": ("exited", "memory")}


def test_the_loop_samples_at_least_once_per_interval(tmp_path, monkeypatch) -> None:
    """What makes the deadline hold: one reading per tick, not per transition.

    Asserted as a rate over the fake clock rather than as a call count, because
    the number that matters is not how often the sampler runs but how long a
    crossed cap can go unnoticed. The run is deliberately long enough to take
    several readings before anything crosses.
    """
    timing = FakeTime()
    at: list[float] = []

    def sample(_process_group: int) -> int | None:
        at.append(timing.now)
        return 10**12 if timing.now >= 3.0 else 0

    supervisor, launcher, _plan, beat = start_capped(
        tmp_path.resolve(), "a", sample=sample, memory_limit_mb=1, timing=timing
    )
    RecordingKills(launcher, beat).install(monkeypatch)

    supervisor.run(handle_signals=False)

    assert len(at) > 3
    gaps = [later - earlier for earlier, later in zip(at, at[1:], strict=False)]
    assert max(gaps) <= MAX_SAMPLE_INTERVAL_SECONDS
    assert supervisor.breaches[0].name == "a"

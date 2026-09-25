"""Start every instance in a fleet, then stay in the foreground and watch them.

The supervisor is deliberately the least clever component in the fleet. It starts
each instance once, polls for exits, records what it saw, and stops. What makes it
worth its own module is the set of things it refuses to do.

*It never restarts anything.* An instance that exits stays exited and keeps being
reported as exited with the reason that was observed. Restarting would be the
obvious kindness and it is the wrong one here: every instance is confined by a
Seatbelt profile built from a snapshot of the fleet document, holds a credential
set derived from the supervisor's environment at start, and has a workspace whose
separability from its peers was checked once, before anything was launched. A
restart would re-spawn it from state that may no longer be true, and an operator
watching ``nanobot fleet status`` would see an instance flapping rather than a
fleet that needs attention. A crash loop that a supervisor hides is a crash loop
nobody fixes.

*One instance's death is not the fleet's.* Each instance lives and dies
independently — the fleet is a set of separate services sharing a supervisor, not
a cluster with a quorum. So the loop reaps with :meth:`~subprocess.Popen.poll`
and never ``wait``: a blocking wait on one child would stop the supervisor from
noticing any other instance's exit, from sampling memory (the cap that
:mod:`nanobot.fleet.memory` feeds is enforced on a deadline), and from responding
to its own ``SIGTERM``. Polling is what keeps every instance's lifecycle
independent of every other's.

*Every transition reaches the state file before anything else happens.* The state
file is the only thing a second shell can read, so an unrecorded transition is an
instance that ``nanobot fleet status`` reports as serving while it is gone —
which is the one lie the whole design of
:mod:`nanobot.fleet.state` exists to prevent. A write that fails is retried on
the next tick rather than taken as fatal: a supervisor that killed a healthy
fleet because its own bookkeeping hit a full disk would have turned a reporting
problem into an outage.

*Exit reasons are only the ones the supervisor is in a position to observe.*
``"signal"`` when the child was terminated by a signal and ``"exit"`` for an
ordinary exit — that is the whole vocabulary the reaper has. ``"memory"`` is a
claim about *why* a signal was sent, which only the component that sent it can
make, so :meth:`FleetSupervisor.enforce_memory_caps` declares it through
:meth:`FleetSupervisor.record_exit_reason` before it kills the tree, and the
reaper prefers that declaration over the bare fact that a signal arrived.

*The memory cap rides on the same tick.* :mod:`nanobot.fleet.cap` decides
whether a tree is over its limit and this module does the killing, because
signalling a tree means addressing its process group — which lives on the owned
handle, together with the guard that keeps a mis-spawned instance's group from
resolving to the supervisor's own. That is also why the tick may not be coarser
than :data:`~nanobot.fleet.cap.MAX_SAMPLE_INTERVAL_SECONDS`: the cap's deadline
is spent almost entirely on waiting for the next sample.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path

from nanobot.fleet.cap import (
    MAX_SAMPLE_INTERVAL_SECONDS,
    CapBreach,
    MemoryCap,
    Sampler,
)
from nanobot.fleet.instance import (
    InstanceLaunchError,
    LaunchedInstance,
    ensure_instance_directories,
    launch_instance,
)
from nanobot.fleet.memory import process_group_memory_bytes
from nanobot.fleet.profile import build_fleet_profiles
from nanobot.fleet.state import (
    EXIT_REASONS,
    ExitReason,
    FleetStateError,
    InstanceRecord,
    ensure_fleet_state_file,
    fleet_state_path,
    running_record,
    write_fleet_state,
)
from nanobot.fleet.validate import ResolvedInstance
from nanobot.process_runtime import process_is_running

#: How often the loop looks for exits. Well under a second because the memory cap
#: is enforced on this same tick and must kill a tree within five seconds of it
#: crossing its limit; a coarser interval would spend that budget on sleeping.
DEFAULT_POLL_INTERVAL_SECONDS = 0.5

#: The coarsest interval the supervisor will accept, for the reason above. Owned
#: by :mod:`nanobot.fleet.cap`, which is where the deadline it derives from
#: lives; re-exported under this module's name because the tick it bounds is
#: this module's. A constructor-time refusal rather than a comment, so the cap
#: policy cannot be handed a supervisor that samples too slowly to honour its
#: own deadline.
MAX_POLL_INTERVAL_SECONDS = MAX_SAMPLE_INTERVAL_SECONDS

#: How long a still-running instance is given to leave after ``SIGTERM`` before
#: the supervisor escalates to ``SIGKILL`` on its own way out.
DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0

#: The signals that mean "stop the fleet" when the supervisor itself receives
#: them. ``SIGINT`` is the operator's Ctrl-C on a foreground command and
#: ``SIGTERM`` is what a process manager sends.
STOP_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGINT, signal.SIGTERM)

#: Injection points. Named so a test can drive the loop deterministically without
#: the production signature loosening into ``**kwargs``.
Launch = Callable[[ResolvedInstance, str], LaunchedInstance]
Sleep = Callable[[float], None]
Clock = Callable[[], float]
StateErrorHandler = Callable[[FleetStateError], None]


@dataclass(frozen=True)
class FleetPlan:
    """Everything that must be true on disk before a fleet can be launched.

    Produced by :func:`prepare_fleet`, which exists as a separate step because
    the order of its three parts is load-bearing and because ``nanobot fleet
    start`` has to run its confinement self-probe against the finished profiles
    *before* any instance process exists.
    """

    instances: tuple[ResolvedInstance, ...]
    fleet_path: Path
    state_path: Path
    #: Instance name to SBPL profile, from :func:`build_fleet_profiles`.
    profiles: Mapping[str, str]


@dataclass(frozen=True)
class InstanceExit:
    """One instance's observed death, as the supervisor saw it.

    ``returncode`` is the raw ``Popen`` value — negative for a signal, following
    that module's convention — and is kept beside ``reason`` because the two
    answer different questions: the reason is what the state file publishes, the
    return code is what an operator needs in order to debug.

    ``returncode`` is ``None`` when the child's own handle could not report one,
    in which case the exit is known but its manner is not, and ``reason`` is
    ``None`` rather than a guess.
    """

    name: str
    pid: int
    returncode: int | None
    reason: ExitReason | None


def exit_reason_for(returncode: int | None) -> ExitReason | None:
    """Classify a ``Popen`` return code as a fleet exit reason.

    ``subprocess`` reports a child killed by signal *n* as ``-n``, and the
    distinction is worth publishing: an instance that exited non-zero has
    diagnostics in its own log, while one that was signalled was killed by
    something outside itself — the operator, the memory cap, or the OS.

    A ``None`` return code yields ``None``: the supervisor knows the instance is
    gone and cannot honestly say how it went.
    """
    if returncode is None:
        return None
    return "signal" if returncode < 0 else "exit"


def prepare_fleet(
    instances: Sequence[ResolvedInstance],
    *,
    fleet_path: Path,
) -> FleetPlan:
    """Lay down what the fleet needs on disk and build every instance's profile.

    Three steps whose order is a correctness requirement rather than a
    preference, because :func:`build_fleet_profiles` refuses to name a path that
    does not exist — Seatbelt would accept such a rule, match nothing, confine
    nothing, and exit zero:

    1. create each instance's workspace and log directory, which commonly do not
       exist before its first start;
    2. create the supervisor's state file, which never exists before a fleet's
       first start;
    3. build the profiles, now that every path they deny is real.

    ``fleet_path`` is canonicalised here, once, and the state path is derived
    from the canonical form: the profile builder requires canonical paths because
    the kernel matches rules against real ones, and deriving the state path
    twice from two different spellings of the fleet file would let the deny and
    the writer disagree silently.

    Raises:
        InstanceLaunchError: a directory could not be created.
        FleetStateError: the state file could not be created.
        SeatbeltProfileError: some path in the fleet cannot be denied
            meaningfully, or two instances are not separable after all.
    """
    canonical = Path(fleet_path).expanduser().resolve(strict=False)
    for instance in instances:
        ensure_instance_directories(instance)
    state_path = ensure_fleet_state_file(fleet_state_path(canonical))
    profiles = build_fleet_profiles(
        instances, fleet_path=canonical, state_path=state_path
    )
    return FleetPlan(
        instances=tuple(instances),
        fleet_path=canonical,
        state_path=state_path,
        profiles=profiles,
    )


def start_fleet(
    plan: FleetPlan,
    *,
    launch: Launch | None = None,
    python_executable: str | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    sleep: Sleep = time.sleep,
    clock: Clock = time.monotonic,
    on_state_error: StateErrorHandler | None = None,
    sample_memory: Sampler = process_group_memory_bytes,
) -> FleetSupervisor:
    """Launch every instance in ``plan`` and return the supervisor watching them.

    All or nothing. If any instance fails to start, the ones already started are
    signalled, reaped and recorded as exited before the failure is re-raised —
    a half-started fleet whose supervisor then exited would leave confined
    processes running with nothing watching them and no ``fleet stop`` able to
    find them, since the state file would never have been written.

    The state file is written once here, before the loop, so a fleet is visible
    to ``nanobot fleet status`` from the moment it exists rather than from its
    first transition.

    Args:
        plan: from :func:`prepare_fleet`.
        launch: the spawn step, taking an instance and its profile. Defaults to
            :func:`nanobot.fleet.instance.launch_instance`; injectable so the
            loop can be driven over stub children.
        python_executable: forwarded to the default launcher only.
        poll_interval: see :class:`FleetSupervisor`.
        sleep: see :class:`FleetSupervisor`.
        clock: see :class:`FleetSupervisor`.
        on_state_error: see :class:`FleetSupervisor`.
        sample_memory: see :class:`FleetSupervisor`.

    Raises:
        InstanceLaunchError: an instance has no profile in ``plan``, or its
            launch failed.
        FleetStateError: the initial state could not be written, which is fatal
            *before* the loop for the reason a later failure is not: nothing is
            running yet, so refusing costs nothing and starting blind would
            produce a fleet no other shell can see.
    """
    spawn = _default_launch(python_executable) if launch is None else launch
    launched: list[LaunchedInstance] = []
    try:
        for instance in plan.instances:
            profile = plan.profiles.get(instance.name)
            if profile is None:
                raise InstanceLaunchError(
                    instance.name,
                    "no Seatbelt profile was built for this instance; the fleet "
                    "refuses to start an unconfined instance",
                )
            launched.append(spawn(instance, profile))
    except BaseException:
        # Includes KeyboardInterrupt: an operator interrupting a start must not
        # be left with the instances that had already come up.
        _abandon(plan, launched, sleep=sleep, clock=clock)
        raise

    supervisor = FleetSupervisor(
        plan,
        launched,
        poll_interval=poll_interval,
        sleep=sleep,
        clock=clock,
        on_state_error=on_state_error,
        sample_memory=sample_memory,
    )
    write_fleet_state(supervisor.records, path=plan.state_path)
    return supervisor


def run_fleet(
    instances: Sequence[ResolvedInstance],
    *,
    fleet_path: Path,
    launch: Launch | None = None,
    python_executable: str | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    handle_signals: bool = True,
    sample_memory: Sampler = process_group_memory_bytes,
) -> tuple[InstanceRecord, ...]:
    """Prepare, start and supervise a validated fleet in the foreground.

    The whole of ``nanobot fleet start`` after validation, minus the confinement
    self-probe — which needs the profiles from :func:`prepare_fleet` before any
    instance exists, and so belongs to a caller that runs the three steps itself.

    Returns:
        Every instance's final record, in fleet order.
    """
    plan = prepare_fleet(instances, fleet_path=fleet_path)
    supervisor = start_fleet(
        plan,
        launch=launch,
        python_executable=python_executable,
        poll_interval=poll_interval,
        sample_memory=sample_memory,
    )
    return supervisor.run(handle_signals=handle_signals)


class FleetSupervisor:
    """A started fleet, and the foreground loop that watches it.

    Holds the owned ``Popen`` handles, which is why it has to be the thing that
    reaps: on POSIX an exited child stays visible to ``kill(pid, 0)`` until its
    parent waits for it, so no other process — not ``nanobot fleet status``, not
    the cap policy — can tell one of this fleet's live instances from a zombie.
    That asymmetry is the reason liveness is reconciled from identity records
    everywhere else and observed directly here.
    """

    def __init__(
        self,
        plan: FleetPlan,
        launched: Iterable[LaunchedInstance],
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        on_state_error: StateErrorHandler | None = None,
        sample_memory: Sampler = process_group_memory_bytes,
    ) -> None:
        """Bind a plan to the processes started from it.

        Args:
            plan: the plan the instances were launched from.
            launched: one entry per instance in ``plan``, in any order.
            poll_interval: seconds between sweeps, ``0 <
                poll_interval <= MAX_POLL_INTERVAL_SECONDS``.
            sleep: the wait primitive. Injectable together with ``clock`` so a
                test can run the loop without spending real time in it.
            clock: the monotonic clock used for shutdown deadlines. Paired with
                ``sleep`` because a fake sleep that did not advance a fake clock
                would turn every bounded wait into a busy spin.
            on_state_error: called when a state write fails. The failure is not
                raised out of the loop — see :meth:`tick` — so this is the only
                way it becomes visible; without it the error is available on
                :attr:`state_error` and nowhere else.
            sample_memory: how an instance's process group is measured against
                its cap. Injectable so the policy can be driven over known
                numbers instead of by arranging real memory pressure.

        Raises:
            ValueError: the interval is out of range, some instance declares a
                non-positive memory limit, or the launched processes do not
                correspond one-to-one with the plan's instances.
        """
        if not 0 < poll_interval <= MAX_POLL_INTERVAL_SECONDS:
            raise ValueError(
                f"poll_interval must be greater than 0 and at most "
                f"{MAX_POLL_INTERVAL_SECONDS} seconds (the memory cap is enforced "
                f"on this tick), but got {poll_interval}"
            )

        self.plan = plan
        self.poll_interval = poll_interval
        self.state_error: FleetStateError | None = None
        self._sleep = sleep
        self._clock = clock
        self._on_state_error = on_state_error
        self._stop_requested = False
        self._state_dirty = False
        self._pending_reasons: dict[str, ExitReason] = {}
        self._breaches: dict[str, CapBreach] = {}
        self.memory_cap = MemoryCap(
            {one.name: one.entry.memory_limit_mb for one in plan.instances},
            sample=sample_memory,
        )

        by_name = {one.name: one for one in launched}
        expected = [instance.name for instance in plan.instances]
        if sorted(by_name) != sorted(expected):
            raise ValueError(
                f"the launched processes do not match the plan: expected "
                f"{', '.join(sorted(expected)) or '(none)'}, got "
                f"{', '.join(sorted(by_name)) or '(none)'}"
            )
        self._launched: dict[str, LaunchedInstance] = {
            name: by_name[name] for name in expected
        }
        self._records: dict[str, InstanceRecord] = {
            instance.name: running_record(
                instance,
                pid=by_name[instance.name].pid,
                identity=by_name[instance.name].identity,
            )
            for instance in plan.instances
        }

    # ------------------------------------------------------------------
    # What the fleet looks like right now
    # ------------------------------------------------------------------

    @property
    def records(self) -> tuple[InstanceRecord, ...]:
        """Every instance's current record, in fleet order.

        Exited instances are kept, never dropped: "this instance ran and stopped
        for this reason" is the answer an operator needs, and a fleet that
        shortened its own status on every death would report a successful fleet
        and an empty one identically.
        """
        return tuple(self._records[instance.name] for instance in self.plan.instances)

    @property
    def running(self) -> tuple[InstanceRecord, ...]:
        """The records still marked running, as of the last sweep."""
        return tuple(one for one in self.records if one.state == "running")

    @property
    def stop_requested(self) -> bool:
        """Whether a stop has been asked for by a signal or by a caller."""
        return self._stop_requested

    @property
    def breaches(self) -> tuple[CapBreach, ...]:
        """Every instance caught over its memory cap, in fleet order.

        Kept after the instance has been killed and reaped, for the same reason
        exited records are kept: the reading that ended an instance is what an
        operator needs in order to decide whether the cap or the workload is
        wrong, and it is not recoverable from the state file, which publishes
        only the reason.
        """
        return tuple(
            self._breaches[one.name]
            for one in self.plan.instances
            if one.name in self._breaches
        )

    def launched(self, name: str) -> LaunchedInstance:
        """The live handle for ``name``.

        The seam the memory cap needs: measuring and killing a tree is addressed
        by process *group*, which is on the handle and not in the state file.

        Raises:
            KeyError: no such instance in this fleet.
        """
        return self._launched[name]

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def record_exit_reason(self, name: str, reason: ExitReason) -> None:
        """Declare *why* ``name`` is about to die, before it does.

        For a caller that is itself the cause — today only the memory cap, which
        kills a tree and must have that recorded as ``"memory"`` rather than as
        the ``"signal"`` the reaper would otherwise infer. Declared up front
        rather than patched afterwards because the exit may be observed on the
        very next sweep, and because a reason applied after the fact would race
        the state write that publishes it.

        A declaration for an instance that then exits on its own is still used:
        the caller knew something the return code does not carry.

        Raises:
            KeyError: no such instance in this fleet.
            ValueError: not one of the reasons the state file accepts.
        """
        if name not in self._records:
            raise KeyError(name)
        if reason not in EXIT_REASONS:
            raise ValueError(
                f"{reason!r} is not a fleet exit reason (expected one of "
                f"{', '.join(EXIT_REASONS)})"
            )
        self._pending_reasons[name] = reason

    def reap(self) -> tuple[InstanceExit, ...]:
        """Collect any instance that has exited since the last sweep.

        Non-blocking for every instance, which is the property the fleet's
        independence rests on: each child is asked whether it has finished and is
        never waited for, so one instance that hangs on shutdown cannot delay the
        supervisor's view of the rest — nor its own memory sampling, nor its
        response to ``SIGTERM``.

        An already-exited instance is not probed again. It is never restarted, so
        there is nothing a second look could discover except a recycled pid
        belonging to somebody else.

        Returns:
            One entry per instance that transitioned on this sweep, in fleet
            order. Empty when nothing changed, which is the common case.
        """
        exits: list[InstanceExit] = []
        for name, launched in self._launched.items():
            record = self._records[name]
            if record.state == "exited":
                continue
            alive, returncode = _observe(launched)
            if alive:
                continue
            reason = self._pending_reasons.pop(name, None) or exit_reason_for(returncode)
            self._records[name] = record.exited(reason)
            self._state_dirty = True
            exits.append(
                InstanceExit(
                    name=name,
                    pid=record.pid,
                    returncode=returncode,
                    reason=self._records[name].exit_reason,
                )
            )
        return tuple(exits)

    def enforce_memory_caps(self) -> tuple[CapBreach, ...]:
        """Sample every running instance's tree and kill the ones over their cap.

        The sampling side of the fleet's memory limit, run once per :meth:`tick`
        so that the interval between two samples is the poll interval — which is
        what the constructor bounds, and what makes the spec's five-second kill
        deadline reachable at all.

        ``SIGKILL`` rather than ``SIGTERM``. This is not the operator asking an
        instance to stop; it is the one case where the instance has already
        proved it cannot be trusted with the machine, and a graceful signal can
        be caught, delayed, or ignored by exactly the runaway allocation the cap
        exists to stop. To the whole process group, not the instance pid, because
        the allocation is typically in a child the instance's shell tool
        started — which is why the metric is defined over the tree in the first
        place.

        The reason is declared *before* the signal. The exit can be observed on
        this very sweep, so a reason patched on afterwards would race the write
        that publishes it and the instance would be reported as having died of
        an ordinary signal.

        An instance already caught is not sampled again, but it *is* signalled
        again: re-sampling could find a tree that had fallen back under its cap
        while it was being killed and reprieve an instance whose ``"memory"``
        reason is already pending, whereas a second ``SIGKILL`` is one idempotent
        syscall that covers a first delivery racing a process joining the group.

        An instance whose recorded group is the supervisor's own is skipped
        outright, the same refusal :func:`signal_instance_tree` makes for the
        same reason. The group is the instance's tree only because it was
        spawned with ``start_new_session``; if that were ever dropped, every
        instance would sample the supervisor's entire world — itself, the
        supervisor, and every peer — and each would breach its own cap on the
        first tick. The fleet would then be killed off one pid at a time and the
        whole thing recorded as a memory problem. Not enforcing a cap the
        supervisor cannot attribute is the only safe reading.

        Skipped entirely once a stop has been requested. The fleet is already
        coming down by the operator's decision, and attributing that to a memory
        cap would misreport why it stopped.

        Returns:
            The breaches found on this sweep — newly caught instances only, in
            fleet order. Empty in the common case.
        """
        if self._stop_requested:
            return ()
        own_group = _own_process_group()
        found: list[CapBreach] = []
        for name, launched in self._launched.items():
            if self._records[name].state == "exited":
                continue
            if launched.pgid <= 0 or launched.pgid == own_group:
                continue
            if name in self._breaches:
                signal_instance_tree(launched, signal.SIGKILL)
                continue
            breach = self.memory_cap.breach(name, launched.pgid)
            if breach is None:
                continue
            self._breaches[name] = breach
            self.record_exit_reason(name, "memory")
            signal_instance_tree(launched, signal.SIGKILL)
            found.append(breach)
        return tuple(found)

    def write_state(self) -> bool:
        """Publish the current records to the state file.

        Returns:
            Whether the write succeeded. A failure is reported rather than
            raised — see :meth:`tick` — and leaves the records marked unwritten
            so the next sweep tries again.
        """
        try:
            write_fleet_state(self.records, path=self.plan.state_path)
        except FleetStateError as exc:
            self.state_error = exc
            if self._on_state_error is not None:
                self._on_state_error(exc)
            return False
        self._state_dirty = False
        self.state_error = None
        return True

    def tick(self) -> tuple[InstanceExit, ...]:
        """One sweep: enforce the memory caps, reap, then publish any change.

        The unit :meth:`run` repeats. Enforcement runs first so that a tree
        killed on this sweep can be reaped on it too rather than waiting a whole
        interval — the cap's deadline is measured from the crossing, and every
        tick spent not noticing is spent out of it.

        A failed write keeps the records dirty and is retried here on the next
        sweep instead of propagating: the supervisor's bookkeeping failing is a
        reason to keep trying to report, not a reason to abandon a fleet that is
        serving correctly.
        """
        self.enforce_memory_caps()
        exits = self.reap()
        if self._state_dirty:
            self.write_state()
        return exits

    # ------------------------------------------------------------------
    # The foreground loop
    # ------------------------------------------------------------------

    def run(self, *, handle_signals: bool = True) -> tuple[InstanceRecord, ...]:
        """Watch the fleet until every instance has exited, or a stop is asked for.

        Returns on its own only when nothing is left running. Instances are never
        replaced, so the loop is finite by construction: each sweep can only move
        instances from running to exited.

        Args:
            handle_signals: install handlers for :data:`STOP_SIGNALS` for the
                duration of the loop, restoring the previous ones on the way
                out. On for the foreground command, because a fleet whose
                supervisor died to a Ctrl-C while its confined children kept
                running would be unstoppable by any means the fleet provides —
                the pids live in a state file nothing would ever update again.
                Off for a caller that owns its own signal disposition.

        Returns:
            Every instance's final record, in fleet order.
        """
        with self._stop_on_signals(handle_signals):
            while True:
                self.tick()
                if not self.running:
                    break
                if self._stop_requested:
                    self.shutdown()
                    break
                self._sleep(self.poll_interval)
        return self.records

    def request_stop(self) -> None:
        """Ask the loop to shut the fleet down at its next opportunity.

        Safe to call from a signal handler: it sets a flag and touches nothing
        else, so it cannot deadlock against a write or a sweep in progress.
        """
        self._stop_requested = True

    def shutdown(
        self,
        *,
        grace: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
    ) -> tuple[InstanceRecord, ...]:
        """Take the still-running instances down and record how they went.

        ``SIGTERM`` to each instance's whole process group, then ``SIGKILL`` to
        whatever is left after ``grace``. The group rather than the pid, because
        an instance's shell tool may have started children that would otherwise
        outlive it — and the group is the tree exactly because every instance is
        spawned into its own session.

        This is the supervisor taking its own children with it when it stops. The
        operator-facing ``nanobot fleet stop`` is a different problem with a
        different tool: a separate process with no handles, working from the pids
        in the state file.

        Marks the fleet as stopping before it sweeps. A shutdown *is* a stop,
        whoever asked for it, and the flag is what keeps the memory cap from
        running over instances that are already being taken down — every sweep
        in the grace period would otherwise re-sample them, and one that found a
        tree over its cap would record the operator's stop as a memory kill.

        Returns:
            Every instance's final record, in fleet order.
        """
        self._stop_requested = True
        self.tick()
        for escalation in (signal.SIGTERM, signal.SIGKILL):
            if not self.running:
                break
            for record in self.running:
                signal_instance_tree(self._launched[record.name], escalation)
            self._wait_for_exits(grace)
        self.tick()
        return self.records

    def _wait_for_exits(self, timeout: float) -> bool:
        """Sweep until nothing is running or ``timeout`` elapses."""
        deadline = self._clock() + timeout
        while True:
            self.tick()
            if not self.running:
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(min(self.poll_interval, timeout))

    @contextmanager
    def _stop_on_signals(self, enabled: bool) -> Iterator[None]:
        """Route :data:`STOP_SIGNALS` to :meth:`request_stop` for the duration.

        Best effort: ``signal.signal`` is only usable from the main thread, and a
        supervisor driven from a worker thread should still run rather than
        refuse. The previous handlers are restored even if the loop raises, so a
        caller that embeds a fleet is not left with this module's disposition.
        """
        if not enabled:
            yield
            return

        def handler(signum: int, frame: object) -> None:
            self.request_stop()

        previous: dict[signal.Signals, object] = {}
        for stop_signal in STOP_SIGNALS:
            with suppress(OSError, ValueError):
                previous[stop_signal] = signal.signal(stop_signal, handler)
        try:
            yield
        finally:
            for stop_signal, original in previous.items():
                with suppress(OSError, ValueError, TypeError):
                    signal.signal(stop_signal, original)  # type: ignore[arg-type]


def signal_instance_tree(launched: LaunchedInstance, sig: int) -> None:
    """Send ``sig`` to an instance's whole process group.

    Falls back to the instance pid alone if the recorded group is the
    supervisor's own. That guard is not defensive padding: the group is the
    instance's tree only because it was spawned with ``start_new_session``, and
    if that were ever dropped the child would join the supervisor's group and a
    group-directed signal would kill the supervisor and every other instance with
    it. The failure would present as the fleet vanishing rather than as an error.

    Already-dead targets are ignored: reaping is :meth:`FleetSupervisor.reap`'s
    job, and a race between the two is expected rather than exceptional.
    """
    killpg = getattr(os, "killpg", None)
    if killpg is not None and launched.pgid > 0 and launched.pgid != _own_process_group():
        try:
            killpg(launched.pgid, sig)
        except OSError:
            pass
        else:
            return
    with suppress(OSError):
        os.kill(launched.pid, sig)


def _own_process_group() -> int | None:
    """The supervisor's own process group, or ``None`` where there is no such thing."""
    getpgid = getattr(os, "getpgid", None)
    if getpgid is None:  # pragma: no cover - POSIX only, as is the whole fleet
        return None
    try:
        return int(getpgid(0))
    except OSError:  # pragma: no cover - reading one's own group does not fail
        return None


def _observe(launched: LaunchedInstance) -> tuple[bool, int | None]:
    """Whether ``launched`` is still alive, and its return code if it is not.

    The owned handle is asked first because it both reports *and* reaps, and only
    the parent can do either; a pid probe would see an uncollected zombie as a
    live instance. ``process_is_running`` is the fallback for a handle that
    cannot answer, and it can only report liveness — hence the ``None``.
    """
    poll = getattr(launched.process, "poll", None)
    if callable(poll):
        try:
            returncode = poll()
        except OSError:
            pass
        else:
            return returncode is None, returncode
    return process_is_running(launched.pid), None


def _default_launch(python_executable: str | None) -> Launch:
    """The production spawn step, with the interpreter choice bound in."""

    def launch(instance: ResolvedInstance, profile: str) -> LaunchedInstance:
        return launch_instance(
            instance, profile=profile, python_executable=python_executable
        )

    return launch


def _abandon(
    plan: FleetPlan,
    launched: Sequence[LaunchedInstance],
    *,
    sleep: Sleep,
    clock: Clock,
) -> None:
    """Stop and record the instances of a fleet that failed to come up.

    Reuses the supervisor's own shutdown over the subset that did start, so a
    partial start is torn down by the same escalation as a complete one and the
    state file ends up describing a stopped fleet rather than retaining whatever
    a previous run left in it. Best effort by nature: it runs while an exception
    is propagating, and the exception the caller is raising is the one that
    matters.

    The state file is written whatever the outcome, including the case where an
    instance would not die. Recording an instance the supervisor could not kill
    looks like an admission of failure and is the only useful thing to do with
    one: it is the sole record of a confined process that is still running, and
    without it nothing — no operator, no later ``fleet stop`` — could find it.
    """
    if not launched:
        # Still worth replacing the file: it may hold a previous run's records,
        # and leaving those would report a fleet that is not running as running.
        with suppress(FleetStateError):
            write_fleet_state((), path=plan.state_path)
        return
    started = {one.name for one in launched}
    partial = replace(
        plan,
        instances=tuple(one for one in plan.instances if one.name in started),
    )
    try:
        supervisor = FleetSupervisor(partial, launched, sleep=sleep, clock=clock)
    except ValueError:  # pragma: no cover - the subset is derived from ``launched``
        return
    with suppress(Exception):
        supervisor.shutdown()
    supervisor.write_state()

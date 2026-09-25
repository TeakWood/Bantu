"""Stop a running fleet from outside it, using nothing but the state file.

``nanobot fleet stop`` is a different problem from
:meth:`nanobot.fleet.supervisor.FleetSupervisor.shutdown`, and the difference is
what this module exists for. The supervisor takes its own children down: it holds
their ``Popen`` handles, so it can both signal them and *reap* them, and reaping
is the only way on POSIX to tell a live child from a zombie. A separate ``stop``
process holds nothing. It has a state file listing pids, and everything else it
needs it has to re-derive from the kernel.

That gap drives three decisions:

*The tree, not the pid, and the tree re-derived on every round.* An instance's
shell tool starts each command with ``start_new_session``, so a long-running
child is out of the instance's process group from the moment it starts and a
group signal alone would leave it holding whatever it holds. The group plus the
descendant closure of its leader is the tree —
:func:`~nanobot.fleet.supervisor.signal_process_tree` enumerates and signals
both — but the closure is only walkable while the instance is alive to be walked
*from*. So every process this module has ever seen in a tree is remembered and
re-signalled, which is what keeps a descendant that survived ``SIGTERM`` and was
then orphaned by its parent's death reachable for the ``SIGKILL``.

*A remembered pid is re-checked against an identity, never against liveness
alone.* A pid that dies is handed out again, and a set of "processes we were
about to kill" is exactly the wrong thing to consult carelessly. Every member is
recorded with the PID-reuse-safe identity
:mod:`nanobot.fleet.instance` produces, and a member whose identity no longer
matches is dropped rather than signalled. The instance process itself is checked
against the identity the *supervisor* recorded at launch, which
:func:`~nanobot.fleet.state.read_fleet_state` has already used to decide the
instance is running at all.

*The supervisor is found by parentage, before anything is killed.* The state file
records instances and says nothing about the supervisor — deliberately, since the
supervisor is not a confined process and nothing else needs to find it. But
``stop`` does: a supervisor left behind would sit in its foreground loop holding
a terminal. Every instance is a direct child of the supervisor, so
:func:`~nanobot.fleet.memory.process_parent_pid` recovers it, and it must be
asked while the instances are still alive — an orphan is reparented to ``launchd``
and the link is gone.

Instances first, then the supervisor, and the whole thing escalates ``SIGTERM``
then ``SIGKILL`` — the escalation ``nanobot/process_runtime.py`` already uses for
a background service. The command returns only once nothing is left, because a
stop that returned optimistically would hand the operator a prompt back while
confined processes were still writing to their workspaces, and the next thing
they do is start the fleet again on the same ports.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from nanobot.fleet.instance import instance_identity, instance_identity_match
from nanobot.fleet.memory import process_parent_pid, process_tree_pids
from nanobot.fleet.state import InstanceRecord, read_fleet_state
from nanobot.fleet.supervisor import signal_process_tree
from nanobot.process_runtime import process_is_running

#: How long everything still running is given to leave after ``SIGTERM`` before
#: ``SIGKILL``, and again after ``SIGKILL`` before the command gives up and says
#: so. Matches the supervisor's own shutdown grace, so an instance sees the same
#: deadline however the fleet is being stopped.
DEFAULT_STOP_GRACE_SECONDS = 5.0

#: How often the wait re-checks. Much finer than the supervisor's poll interval:
#: nothing else happens on this tick, and the command's whole job is to return as
#: soon as the fleet is actually gone.
STOP_POLL_INTERVAL_SECONDS = 0.05

#: The escalation, in order. ``SIGTERM`` so an instance can flush and exit
#: cleanly; ``SIGKILL`` because a stop that can be ignored is not a stop.
STOP_ESCALATION: tuple[signal.Signals, ...] = (signal.SIGTERM, signal.SIGKILL)

#: Injection points, named so a test can drive the escalation without spending
#: real time in it.
Sleep = Callable[[float], None]
Clock = Callable[[], float]


@dataclass(frozen=True)
class TreeMember:
    """One process belonging to an instance's tree, and how to re-identify it.

    ``identity`` is what :mod:`nanobot.fleet.instance` records beside a pid so a
    later look can tell "still the same process" from "this pid was handed to
    somebody else". ``None`` compares as a match, which is
    ``process_runtime``'s own posture: a platform that cannot read identities
    falls back to trusting the pid rather than refusing to stop the fleet.
    """

    pid: int
    identity: object = None

    def alive(self) -> bool:
        """Whether *this* process — not merely this pid — is still running."""
        if self.pid <= 0 or not process_is_running(self.pid):
            return False
        return instance_identity_match(self.identity, self.pid) != "mismatch"


@dataclass(frozen=True)
class FleetStopReport:
    """What ``stop`` found, what it signalled, and what refused to die."""

    state_path: Path
    #: Names of the instances that were running when the state file was read.
    targeted: tuple[str, ...]
    #: Names still holding at least one live process when the command gave up.
    survivors: tuple[str, ...]
    #: The supervisor, if parentage could name one. ``None`` when no instance was
    #: running, or when the instances had already been orphaned.
    supervisor_pid: int | None
    #: Whether the supervisor is gone — vacuously true when there was none.
    supervisor_stopped: bool

    @property
    def complete(self) -> bool:
        """Whether nothing is left of this fleet."""
        return not self.survivors and self.supervisor_stopped

    @property
    def stopped(self) -> tuple[str, ...]:
        """The targeted instances whose trees are gone."""
        survived = set(self.survivors)
        return tuple(name for name in self.targeted if name not in survived)


class InstanceTree:
    """One instance's process tree, tracked across the escalation.

    Mutable on purpose: the set of processes belonging to an instance changes
    while it is being killed, and this object is the only thing that remembers
    the members that have become unwalkable — a descendant whose parent has died
    is reachable by neither the group nor the closure, so if it is forgotten
    between the ``SIGTERM`` and the ``SIGKILL`` it survives the stop entirely.
    """

    def __init__(self, record: InstanceRecord, *, process_group: int) -> None:
        self.name = record.name
        self.pid = record.pid
        self.process_group = process_group
        # Seeded with the supervisor's own identity for the instance process, so
        # the one pid that has an authoritative identity is checked against it
        # rather than against one read now.
        self._members: dict[int, TreeMember] = {
            record.pid: TreeMember(record.pid, record.recorded_identity)
        }

    def survey(self) -> tuple[TreeMember, ...]:
        """Re-derive the live members, keeping the ones still identifiable.

        Remembered members that have exited — or whose pid now belongs to an
        unrelated process — are dropped here, which is what stops a recycled pid
        from being carried into the next signal.
        """
        live = {
            pid: member for pid, member in self._members.items() if member.alive()
        }
        for pid in _tree_pids(self.process_group):
            if pid in live or not process_is_running(pid):
                continue
            live[pid] = TreeMember(pid, _identity_of(pid))
        self._members = live
        return tuple(live.values())

    def alive(self) -> bool:
        """Whether any process of this instance's tree is still running."""
        return bool(self.survey())

    def signal(self, sig: int) -> tuple[int, ...]:
        """Send ``sig`` to everything still in this tree.

        The group and its walkable descendants go in
        :func:`~nanobot.fleet.supervisor.signal_process_tree`'s single pass, and
        the remembered members are signalled individually afterwards. The two
        overlap in the common case, which costs one idempotent syscall per
        process and covers the case that matters: an orphan the walk can no
        longer reach.

        Returns:
            The pids that were signalled, in ascending order.
        """
        members = self.survey()
        if not members:
            return ()
        signal_process_tree(self.pid, self.process_group, sig)
        for member in members:
            if member.pid == self.pid:
                continue
            with suppress(OSError):
                os.kill(member.pid, sig)
        return tuple(sorted(member.pid for member in members))


def stop_fleet(
    state_path: Path,
    *,
    grace: float = DEFAULT_STOP_GRACE_SECONDS,
    poll_interval: float = STOP_POLL_INTERVAL_SECONDS,
    sleep: Sleep = time.sleep,
    clock: Clock = time.monotonic,
) -> FleetStopReport:
    """Stop every instance recorded in ``state_path``, then the supervisor.

    Blocks until nothing of the fleet is left, or until the escalation has run
    out of deadline — which the report says plainly rather than hiding, because
    "stop returned" has to mean "the fleet is gone" for the operator's next
    command to be safe.

    Idempotent. A fleet whose instances have all already exited is a no-op that
    reports success: reconciliation in :func:`~nanobot.fleet.state.read_fleet_state`
    has already turned every dead pid into an ``exited`` record, so nothing is
    signalled and nothing is waited for.

    Args:
        state_path: the supervisor's state file, from
            :func:`~nanobot.fleet.state.fleet_state_path`.
        grace: seconds allowed after each signal in :data:`STOP_ESCALATION`.
        poll_interval: how often the waits re-check.
        sleep: the wait primitive, injectable with ``clock`` so a test can run
            the escalation without spending real time in it.
        clock: the monotonic clock the deadlines are measured on.

    Raises:
        FleetStateError: the state file is missing, unreadable, or was not
            written by a supervisor. Refusing is the only safe answer — a
            ``stop`` that treated an unparseable file as an empty fleet would
            report success over a fleet it never looked at.
    """
    records = read_fleet_state(state_path)
    trees = [
        InstanceTree(record, process_group=_process_group(record.pid))
        for record in records
        if record.state == "running"
    ]
    supervisor = _supervisor(trees)

    for escalation in STOP_ESCALATION:
        if not _anything_left(trees, supervisor):
            break
        for tree in trees:
            tree.signal(escalation)
        # The supervisor last, as the fleet's own ordering: its instances are
        # already down, so a SIGTERM it honours has nothing left to tear down and
        # a SIGTERM it ignores costs only the escalation's next round.
        if supervisor is not None and supervisor.alive():
            with suppress(OSError):
                os.kill(supervisor.pid, escalation)
        _wait_until_gone(
            trees,
            supervisor,
            grace=grace,
            poll_interval=poll_interval,
            sleep=sleep,
            clock=clock,
        )

    return FleetStopReport(
        state_path=state_path,
        targeted=tuple(tree.name for tree in trees),
        survivors=tuple(tree.name for tree in trees if tree.alive()),
        supervisor_pid=None if supervisor is None else supervisor.pid,
        supervisor_stopped=supervisor is None or not supervisor.alive(),
    )


def _wait_until_gone(
    trees: Sequence[InstanceTree],
    supervisor: TreeMember | None,
    *,
    grace: float,
    poll_interval: float,
    sleep: Sleep,
    clock: Clock,
) -> bool:
    """Block until nothing is left or ``grace`` elapses; whether it was gone."""
    budget = max(float(grace), 0.0)
    deadline = clock() + budget
    while True:
        if not _anything_left(trees, supervisor):
            return True
        if clock() >= deadline:
            return False
        sleep(min(max(float(poll_interval), 0.0), budget))


def _anything_left(
    trees: Sequence[InstanceTree],
    supervisor: TreeMember | None,
) -> bool:
    """Whether any instance process or the supervisor is still running."""
    if any(tree.alive() for tree in trees):
        return True
    return supervisor is not None and supervisor.alive()


def _supervisor(trees: Sequence[InstanceTree]) -> TreeMember | None:
    """The process watching these instances, named by their shared parentage.

    Asked before anything is signalled, because the answer disappears with the
    first death. ``None`` unless every running instance agrees on one parent that
    is neither ``launchd`` nor this command nor whatever started it: a fleet
    whose instances disagree is not something to guess about, and a guess here is
    a signal aimed at an unrelated process.
    """
    forbidden = {0, 1, os.getpid(), os.getppid()}
    candidates: set[int] = set()
    for tree in trees:
        parent = process_parent_pid(tree.pid)
        if parent is None or parent in forbidden:
            return None
        candidates.add(parent)
    if len(candidates) != 1:
        return None
    pid = candidates.pop()
    return TreeMember(pid, _identity_of(pid))


def _tree_pids(process_group: int) -> tuple[int, ...]:
    """The live members of a tree, or nothing if it cannot be enumerated.

    Refuses to enumerate the caller's own group, for the reason
    :func:`~nanobot.fleet.supervisor.signal_process_tree` refuses to signal it:
    an instance whose recorded group is this process's own would otherwise put
    every unrelated sibling into the set of things about to be killed.
    """
    if process_group <= 0 or process_group == _own_process_group():
        return ()
    return tuple(pid for pid in (process_tree_pids(process_group) or ()) if pid > 0)


def _process_group(pid: int) -> int:
    """``pid``'s process group, falling back to the pid itself.

    Every instance is spawned with ``start_new_session``, so its group id *is*
    its pid; the lookup observes that rather than assuming it, and the fallback
    covers a process that exits between the state file being read and this call.
    """
    getpgid = getattr(os, "getpgid", None)
    if getpgid is None:  # pragma: no cover - POSIX only, as is the whole fleet
        return pid
    try:
        return int(getpgid(pid))
    except OSError:
        return pid


def _own_process_group() -> int | None:
    """This command's own process group, or ``None`` where there is no such thing."""
    getpgid = getattr(os, "getpgid", None)
    if getpgid is None:  # pragma: no cover - POSIX only, as is the whole fleet
        return None
    try:
        return int(getpgid(0))
    except OSError:  # pragma: no cover - reading one's own group does not fail
        return None


def _identity_of(pid: int) -> object:
    """A PID-reuse-safe identity for ``pid``, read now.

    Mirrors :attr:`~nanobot.fleet.state.InstanceRecord.recorded_identity`'s
    precedence: ``stable_identity`` first, because on macOS the bare ``identity``
    is only the process group and a recycled pid leading the same group would
    match it.
    """
    record = instance_identity(pid)
    stable = record.get("stable_identity")
    return stable if stable is not None else record.get("identity")

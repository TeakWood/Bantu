"""Tests for process-tree resident memory sampling."""

from __future__ import annotations

import ctypes
import os
import selectors
import signal
import struct
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

import pytest

from nanobot.fleet import memory

darwin_only = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="libproc resident-size sampling is implemented for macOS only",
)

ALLOCATION_BYTES = 64 * 1024 * 1024
CHILD_TIMEOUT_SECONDS = 30.0

# A leader that allocates on demand and can fork children into its own process
# group. Children are how a real instance grows a tree: a shell tool forks, and
# the fork inherits the group. ``setpgid`` cannot be used from the test process
# instead, because the leader is in its own session.
TREE_SCRIPT = """
import subprocess
import sys

held = [b"a" * int(sys.argv[2])]
children = []
sys.stdout.write("ready\\n")
sys.stdout.flush()

if sys.argv[1] == "child":
    # Blocks until the leader dies and closes this pipe, so no child outlives
    # the group it was spawned into.
    sys.stdin.readline()
    sys.exit(0)

for line in sys.stdin:
    command, _, argument = line.strip().partition(" ")
    if command == "allocate":
        held.append(b"a" * int(argument))
        sys.stdout.write(f"allocated {argument}\\n")
    elif command in ("spawn", "detach"):
        child = subprocess.Popen(
            [sys.executable, sys.argv[0], "child", argument],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            # "detach" is what nanobot's own shell tool does to every command it
            # runs, and it takes the child out of this leader's process group.
            start_new_session=command == "detach",
        )
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        children.append(child)
        sys.stdout.write(f"{command}ed {child.pid}\\n")
    else:
        break
    sys.stdout.flush()
"""


@pytest.fixture(autouse=True)
def _unbind_libproc_handle() -> Iterator[None]:
    """Keep a patched ``sys.platform`` from leaking through the cached handles.

    Both binders, not just the group one: the two walks are cached separately so
    that either can fail on its own, and a platform patch that cleared only one
    would leave the other answering from a handle bound before the patch.
    """
    binders = (memory._darwin_libproc, memory._darwin_proc_listchildpids)
    for binder in binders:
        binder.cache_clear()
    yield
    for binder in binders:
        binder.cache_clear()


def _read_line(stream: IO[str] | None, what: str) -> str:
    """Read one line under an explicit deadline; this suite has no test timeout."""
    assert stream is not None
    deadline = time.monotonic() + CHILD_TIMEOUT_SECONDS
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"process tree never reported {what}")
            if selector.select(remaining):
                line = stream.readline()
                if not line:
                    raise AssertionError(f"process tree exited before reporting {what}")
                return line.strip()


def _command(leader: subprocess.Popen[str], line: str, expected: str) -> str:
    assert leader.stdin is not None
    leader.stdin.write(f"{line}\n")
    leader.stdin.flush()
    reply = _read_line(leader.stdout, expected)
    assert reply.startswith(expected), reply
    return reply


class ProcessTree:
    """A process group led by one child of this test process."""

    def __init__(self, script: Path, resident_bytes: int = 0) -> None:
        self.leader = subprocess.Popen(
            [sys.executable, str(script), "leader", str(resident_bytes)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        _read_line(self.leader.stdout, "startup")
        self.process_group = os.getpgid(self.leader.pid)

    def allocate(self, size: int) -> None:
        _command(self.leader, f"allocate {size}", f"allocated {size}")

    def spawn(self, resident_bytes: int = 0) -> int:
        return int(_command(self.leader, f"spawn {resident_bytes}", "spawned").split()[1])

    def detach(self, resident_bytes: int = 0) -> int:
        """Fork a child into a session of its own, as the shell tool does."""
        return int(_command(self.leader, f"detach {resident_bytes}", "detached").split()[1])

    def sample(self) -> int:
        total = memory.process_group_memory_bytes(self.process_group)
        assert total is not None
        return total

    def sample_tree(self) -> int:
        total = memory.process_tree_memory_bytes(self.process_group)
        assert total is not None
        return total

    def stop(self) -> None:
        if self.leader.poll() is None:
            os.killpg(self.process_group, signal.SIGKILL)
        self.leader.wait(timeout=CHILD_TIMEOUT_SECONDS)


@pytest.fixture
def tree_script(tmp_path: Path) -> Path:
    script = tmp_path / "process_tree.py"
    script.write_text(textwrap.dedent(TREE_SCRIPT), encoding="utf-8")
    return script


@pytest.fixture
def tree(tree_script: Path) -> Iterator[ProcessTree]:
    started = ProcessTree(tree_script)
    try:
        yield started
    finally:
        started.stop()


@darwin_only
def test_group_total_rises_by_at_least_a_known_allocation(tree: ProcessTree) -> None:
    assert tree.process_group == tree.leader.pid
    before = tree.sample()
    assert before > 0

    tree.allocate(ALLOCATION_BYTES)

    assert tree.sample() - before >= ALLOCATION_BYTES


@darwin_only
def test_group_total_covers_members_beyond_the_leader(tree: ProcessTree) -> None:
    before = tree.sample()

    # Only the forked child allocates, so this rise can only be found by walking
    # the whole group rather than sampling the leader alone.
    child_pid = tree.spawn(resident_bytes=ALLOCATION_BYTES)

    pids = memory.process_group_pids(tree.process_group)
    assert pids is not None
    assert child_pid in pids
    assert tree.leader.pid in pids
    assert tree.sample() - before >= ALLOCATION_BYTES


@darwin_only
def test_group_walk_grows_past_a_saturated_first_buffer(
    tree_script: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A truncated walk would silently under-report a tree, which is the one
    # failure this sampler must not have; force the grow-and-retry path.
    monkeypatch.setattr(memory, "_PID_LIST_INITIAL_CAPACITY", 1)
    started = ProcessTree(tree_script)
    try:
        expected = {started.leader.pid} | {started.spawn() for _ in range(3)}

        pids = memory.process_group_pids(started.process_group)
        assert pids is not None
        assert expected <= set(pids)
    finally:
        started.stop()


@darwin_only
def test_group_survives_one_member_exiting(tree: ProcessTree) -> None:
    child_pid = tree.spawn(resident_bytes=ALLOCATION_BYTES)
    with_child = tree.sample()
    os.kill(child_pid, signal.SIGKILL)

    # The leader never reaps, so the dead child lingers in the group as a zombie;
    # what must drop is the total, and the leader must still be counted.
    deadline = time.monotonic() + CHILD_TIMEOUT_SECONDS
    while True:
        after = tree.sample()
        if after <= with_child - ALLOCATION_BYTES:
            break
        if time.monotonic() > deadline:
            raise AssertionError("killed child's memory never left the group total")
    assert after > 0


# --------------------------------------------------------------------------
# The half of the tree the process group cannot see
# --------------------------------------------------------------------------


@darwin_only
def test_a_descendant_that_left_the_group_is_still_counted(tree: ProcessTree) -> None:
    """The case the fleet's memory cap exists for, and the group walk misses.

    nanobot's shell tool starts every command with ``start_new_session``, so the
    allocation an instance is capped for happens in a process that is no longer
    in the instance's process group by the time it allocates a byte. A group-only
    sampler reads the same number before and after.
    """
    before_group, before_tree = tree.sample(), tree.sample_tree()

    detached_pid = tree.detach(resident_bytes=ALLOCATION_BYTES)

    assert os.getpgid(detached_pid) != tree.process_group
    assert memory.process_group_pids(tree.process_group) is not None
    assert detached_pid not in (memory.process_group_pids(tree.process_group) or [])
    assert detached_pid in (memory.process_tree_pids(tree.process_group) or [])
    # The reading the cap acts on rises; the group-only reading does not.
    assert tree.sample_tree() - before_tree >= ALLOCATION_BYTES
    assert tree.sample() - before_group < ALLOCATION_BYTES


@darwin_only
def test_the_tree_is_the_union_of_both_walks(tree: ProcessTree) -> None:
    """Neither walk is a superset of the other, so the tree is both."""
    in_group = tree.spawn()
    out_of_group = tree.detach()

    pids = memory.process_tree_pids(tree.process_group)

    assert pids is not None
    assert {tree.leader.pid, in_group, out_of_group} <= set(pids)
    assert len(pids) == len(set(pids))


@darwin_only
def test_a_detached_child_is_reached_by_parentage_alone(tree: ProcessTree) -> None:
    """``setsid`` changes a process's group and session, never its parent.

    Which is the whole reason the second walk exists. This pins the kernel side
    of it against real processes; the closure built on top of it — descending
    through generations — is pinned over a declared parent table below, where the
    shape of the tree can be stated rather than arranged.
    """
    detached_pid = tree.detach()

    assert memory.process_child_pids(tree.leader.pid) is not None
    assert detached_pid in (memory.process_child_pids(tree.leader.pid) or [])
    assert detached_pid in (memory.process_descendant_pids(tree.leader.pid) or [])


@darwin_only
def test_a_tree_that_has_fully_exited_samples_as_zero(tree_script: Path) -> None:
    started = ProcessTree(tree_script)
    process_group = started.process_group
    started.stop()

    # Zero, not None, for the same reason the group walk reports zero: a cap must
    # never confuse "used nothing" with "could not be read".
    assert memory.process_tree_pids(process_group) == []
    assert memory.process_tree_memory_bytes(process_group) == 0


@darwin_only
def test_resident_bytes_of_this_process_is_plausible() -> None:
    resident = memory.process_resident_bytes(os.getpid())
    assert resident is not None
    assert resident > 1024 * 1024


@darwin_only
def test_resident_bytes_is_none_for_a_process_that_has_exited(tree_script: Path) -> None:
    started = ProcessTree(tree_script)
    pid = started.leader.pid
    started.stop()

    assert memory.process_resident_bytes(pid) is None


@darwin_only
def test_group_that_has_fully_exited_samples_as_zero(tree_script: Path) -> None:
    started = ProcessTree(tree_script)
    process_group = started.process_group
    started.stop()

    # Zero, not None: a supervisor must be able to tell "using no memory" from
    # "cannot tell", or it would kill instances it merely failed to read.
    assert memory.process_group_pids(process_group) == []
    assert memory.process_group_memory_bytes(process_group) == 0


def test_unsupported_platform_samples_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory.sys, "platform", "linux")
    memory._darwin_libproc.cache_clear()

    assert memory._darwin_libproc() is None
    assert memory.process_group_pids(os.getpgid(0)) is None
    assert memory.process_resident_bytes(os.getpid()) is None
    assert memory.process_group_memory_bytes(os.getpgid(0)) is None


def test_libproc_that_will_not_load_samples_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def unloadable(*_args: object, **_kwargs: object) -> object:
        raise OSError("no such library")

    monkeypatch.setattr(memory.sys, "platform", "darwin")
    monkeypatch.setattr(memory.ctypes, "CDLL", unloadable)
    memory._darwin_libproc.cache_clear()

    assert memory._darwin_libproc() is None
    assert memory.process_group_memory_bytes(4242) is None


def test_libproc_without_the_symbols_we_need_samples_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OlderLibproc:
        """A ``libproc`` carrying ``proc_pidinfo`` but not the group walk."""

        def __init__(self) -> None:
            self.proc_pidinfo = lambda *_args: 0

    monkeypatch.setattr(memory.sys, "platform", "darwin")
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda *_args, **_kwargs: OlderLibproc())
    memory._darwin_libproc.cache_clear()

    assert memory._darwin_libproc() is None


def test_libproc_calls_that_raise_are_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object) -> int:
        raise OSError("libproc refused the call")

    monkeypatch.setattr(memory, "_darwin_libproc", lambda: (explode, explode))

    assert memory.process_group_pids(4242) is None
    assert memory.process_resident_bytes(4242) is None
    assert memory.process_group_memory_bytes(4242) is None


def test_non_positive_identifiers_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object) -> int:  # pragma: no cover - must not be reached
        raise AssertionError("libproc was called with a non-positive identifier")

    monkeypatch.setattr(memory, "_darwin_libproc", lambda: (refuse, refuse))

    assert memory.process_group_pids(0) is None
    assert memory.process_group_pids(-1) is None
    assert memory.process_resident_bytes(0) is None
    assert memory.process_group_memory_bytes(-1) is None


def _fake_handles(
    monkeypatch: pytest.MonkeyPatch,
    list_pgrp_pids: Any,
    pid_info: Any,
) -> None:
    monkeypatch.setattr(memory, "_darwin_libproc", lambda: (pid_info, list_pgrp_pids))


def test_failed_group_walk_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_pgid: int, _buffer: object, _size: int) -> int:
        return -1

    _fake_handles(monkeypatch, fail, lambda *_args: 0)

    assert memory.process_group_pids(4242) is None
    assert memory.process_group_memory_bytes(4242) is None


def test_group_walk_that_never_stops_saturating_gives_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    def always_full(_pgid: int, buffer: Any, size: int) -> int:
        capacity = size // ctypes.sizeof(ctypes.c_int32)
        attempts.append(capacity)
        for index in range(capacity):
            buffer[index] = index + 1
        return capacity

    _fake_handles(monkeypatch, always_full, lambda *_args: 0)

    assert memory.process_group_pids(4242) is None
    assert attempts[0] == memory._PID_LIST_INITIAL_CAPACITY
    assert attempts[-1] <= memory._PID_LIST_MAX_CAPACITY


def test_taskinfo_shorter_than_the_hard_coded_struct_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def short_write(_pid: int, _flavor: int, _arg: int, buffer: Any, _size: int) -> int:
        struct.pack_into("=Q", buffer, memory._PTI_RESIDENT_SIZE_OFFSET, 123456)
        return memory._PROC_TASKINFO_SIZE - 8

    _fake_handles(monkeypatch, lambda *_args: 0, short_write)

    assert memory.process_resident_bytes(4242) is None


def test_members_that_cannot_be_read_are_skipped_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readable = {11: 4096, 13: 8192}

    def three_members(_pgid: int, buffer: Any, _size: int) -> int:
        buffer[0] = 11
        buffer[1] = 12
        buffer[2] = 13
        return 3

    def sample(pid: int, flavor: int, _arg: int, buffer: Any, size: int) -> int:
        assert flavor == memory._PROC_PIDTASKINFO
        assert size == memory._PROC_TASKINFO_SIZE
        resident = readable.get(pid)
        if resident is None:
            return 0
        struct.pack_into("=Q", buffer, memory._PTI_RESIDENT_SIZE_OFFSET, resident)
        return memory._PROC_TASKINFO_SIZE

    _fake_handles(monkeypatch, three_members, sample)

    assert memory.process_group_pids(4242) == [11, 12, 13]
    assert memory.process_group_memory_bytes(4242) == 4096 + 8192


# --------------------------------------------------------------------------
# The tree walk's own failure modes
# --------------------------------------------------------------------------


def _fake_children(monkeypatch: pytest.MonkeyPatch, children: dict[int, list[int]]) -> None:
    """Stand in for ``proc_listchildpids`` over a declared parent-child table."""

    def list_child_pids(pid: int, buffer: Any, size: int) -> int:
        found = children.get(pid, [])
        capacity = size // ctypes.sizeof(ctypes.c_int32)
        for index, child in enumerate(found[:capacity]):
            buffer[index] = child
        return min(len(found), capacity)

    monkeypatch.setattr(memory, "_darwin_proc_listchildpids", lambda: list_child_pids)


def test_a_tree_neither_walk_can_read_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "_darwin_libproc", lambda: None)
    monkeypatch.setattr(memory, "_darwin_proc_listchildpids", lambda: None)

    assert memory.process_child_pids(4242) is None
    assert memory.process_descendant_pids(4242) is None
    assert memory.process_tree_pids(4242) is None
    assert memory.process_tree_memory_bytes(4242) is None


def test_one_walk_failing_does_not_lose_what_the_other_saw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Halves, because the two walks see different things and fail separately."""
    _fake_children(monkeypatch, {4242: [77]})
    monkeypatch.setattr(memory, "_darwin_libproc", lambda: None)

    # No group walk: the leader is still part of its own tree, and so is the
    # descendant that left the group.
    assert memory.process_tree_pids(4242) == [4242, 77]

    _fake_handles(monkeypatch, lambda _pgid, buffer, _size: _write(buffer, [4242, 88]), lambda *_a: 0)
    monkeypatch.setattr(memory, "_darwin_proc_listchildpids", lambda: None)

    # No descendant walk: the group is still the tree's best-known membership.
    assert memory.process_tree_pids(4242) == [4242, 88]


def _write(buffer: Any, pids: list[int]) -> int:
    for index, pid in enumerate(pids):
        buffer[index] = pid
    return len(pids)


def test_the_descendant_walk_descends_through_every_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One level would be enough for a shell tool that execs its command, and not
    # enough for one that forks first — which is the difference between a command
    # and a pipeline, not something the cap gets to depend on.
    _fake_children(monkeypatch, {1: [2], 2: [3], 3: [4]})

    assert sorted(memory.process_descendant_pids(1) or []) == [2, 3, 4]


def test_the_descendant_walk_never_revisits_a_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    # A pid recycled into the walk's own results would otherwise loop forever on
    # a supervisor tick that must finish inside the cap's sampling interval.
    _fake_children(monkeypatch, {1: [2], 2: [3], 3: [1, 2]})

    assert sorted(memory.process_descendant_pids(1) or []) == [2, 3]


def test_the_descendant_walk_stops_at_the_member_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory, "_MAX_TREE_MEMBERS", 3)
    _fake_children(monkeypatch, {1: [2, 3, 4, 5, 6]})

    found = memory.process_descendant_pids(1)

    assert found is not None and len(found) == 3


def test_the_child_walk_grows_past_a_saturated_first_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    def always_full(_pid: int, buffer: Any, size: int) -> int:
        capacity = size // ctypes.sizeof(ctypes.c_int32)
        attempts.append(capacity)
        for index in range(capacity):
            buffer[index] = index + 1
        return capacity

    monkeypatch.setattr(memory, "_darwin_proc_listchildpids", lambda: always_full)

    # A truncated child list would silently drop a whole subtree, which is worse
    # than reporting nothing: the cap would read a runaway tree as a small one.
    assert memory.process_child_pids(4242) is None
    assert attempts[0] == memory._PID_LIST_INITIAL_CAPACITY
    assert attempts[-1] <= memory._PID_LIST_MAX_CAPACITY


def test_a_failed_child_walk_is_none_rather_than_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory, "_darwin_proc_listchildpids", lambda: lambda *_args: -1)

    assert memory.process_child_pids(4242) is None
    assert memory.process_descendant_pids(4242) is None


def test_a_child_walk_that_raises_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object) -> int:
        raise OSError("libproc refused the call")

    monkeypatch.setattr(memory, "_darwin_proc_listchildpids", lambda: explode)

    assert memory.process_child_pids(4242) is None


def test_the_child_walk_rejects_non_positive_pids(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object) -> int:  # pragma: no cover - must not be reached
        raise AssertionError("libproc was called with a non-positive identifier")

    monkeypatch.setattr(memory, "_darwin_proc_listchildpids", lambda: refuse)

    # ``proc_listchildpids(0, …)`` answers for the kernel's own children, which
    # is never what a caller holding an instance pid meant to ask.
    assert memory.process_child_pids(0) is None
    assert memory.process_child_pids(-1) is None


def test_an_unsupported_platform_has_no_child_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory.sys, "platform", "linux")
    memory._darwin_proc_listchildpids.cache_clear()

    assert memory._darwin_proc_listchildpids() is None


def test_libproc_without_the_child_walk_symbol_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OlderLibproc:
        """A ``libproc`` carrying the group walk but not the child walk."""

        def __init__(self) -> None:
            self.proc_pidinfo = lambda *_args: 0
            self.proc_listpgrppids = lambda *_args: 0

    monkeypatch.setattr(memory.sys, "platform", "darwin")
    monkeypatch.setattr(memory.ctypes, "CDLL", lambda *_args, **_kwargs: OlderLibproc())
    memory._darwin_proc_listchildpids.cache_clear()

    assert memory._darwin_proc_listchildpids() is None


def test_tree_members_that_cannot_be_read_are_skipped_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readable = {4242: 4096, 77: 8192}

    def sample(pid: int, _flavor: int, _arg: int, buffer: Any, _size: int) -> int:
        resident = readable.get(pid)
        if resident is None:
            return 0
        struct.pack_into("=Q", buffer, memory._PTI_RESIDENT_SIZE_OFFSET, resident)
        return memory._PROC_TASKINFO_SIZE

    _fake_handles(monkeypatch, lambda _pgid, buffer, _size: _write(buffer, [4242, 99]), sample)
    _fake_children(monkeypatch, {4242: [77]})

    assert memory.process_tree_pids(4242) == [4242, 99, 77]
    assert memory.process_tree_memory_bytes(4242) == 4096 + 8192


def test_sampler_depends_on_nothing_inside_nanobot() -> None:
    # The bead requires this module not to reach into process_runtime.py's private
    # ctypes helpers, so a chokepoint the gateway imports on every start is left
    # untouched, and to add no dependency such as psutil.
    source = Path(memory.__file__).read_text(encoding="utf-8")
    imports = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "__future__" not in line
    ]

    assert imports == ["import ctypes", "import struct", "import sys"] + [
        "from functools import lru_cache",
        "from typing import Any",
    ]

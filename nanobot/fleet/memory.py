"""Resident memory sampling for a whole instance process tree.

The fleet supervisor caps each instance on the memory of its *entire* process
tree, not just the instance process, because a shell tool can fork a child that
allocates without bound. Every instance is started in its own process group, so
the group is the obvious enumeration of that tree and the one the kernel will
let us walk in a single call.

*The group alone is not the tree.* nanobot's own shell tool spawns every command
with ``start_new_session=True`` (``agent/tools/shell.py``) so that it can kill a
runaway command's whole tree by process group. That ``setsid`` moves the command
— and everything it goes on to start — out of the instance's process group
immediately, which is precisely where the allocation the cap exists to catch
happens. A group-only walk cannot see it. So the tree is defined here as the
union of two enumerations that fail in different ways:

* the process group, which catches a descendant that has been orphaned and
  reparented away but has not left the group; and
* the descendant closure of the group leader, which catches a child that left
  the group by starting its own session.

Neither is a superset of the other, and only their union matches what an
operator means by "everything that instance started". A process that does both —
leaves the group *and* is orphaned — is unreachable by either, and by then it is
no longer attributable to the instance by any means short of process accounting.

Summing per-process resident sizes double-counts pages shared between the
members of a tree (the Python runtime's own text and any copy-on-write pages
inherited across ``fork``). The sum is therefore an over-estimate and the cap
built on it is conservative. That is the metric the fleet spec asks for: the
total resident memory of the tree.

Only macOS is implemented, which is the platform the fleet's Seatbelt
confinement requires; every other platform samples as ``None``. ``psutil`` is
deliberately not used — it is not a dependency of nanobot and would duplicate
the ``ctypes`` work below.
"""

from __future__ import annotations

import ctypes
import struct
import sys
from functools import lru_cache
from typing import Any

# ``proc_pidinfo`` flavor for ``struct proc_taskinfo`` (``sys/proc_info.h``).
_PROC_PIDTASKINFO = 4
# ``sizeof(struct proc_taskinfo)``: six uint64 fields followed by twelve int32.
_PROC_TASKINFO_SIZE = 96
# ``pti_resident_size`` is the second uint64, right after ``pti_virtual_size``.
_PTI_RESIDENT_SIZE_OFFSET = 8

# ``proc_listpgrppids`` cannot report how many members a group has without
# writing them somewhere, so we start with a generous buffer and grow when the
# result saturates. The ceiling only exists so a hostile or pathological group
# cannot make us allocate without bound.
_PID_LIST_INITIAL_CAPACITY = 256
_PID_LIST_MAX_CAPACITY = 65536

# A ceiling on the descendant walk. One ``proc_listchildpids`` call per node is
# cheap, but the walk is driven by a table the processes under it are free to
# grow, and this runs on every supervisor tick. An instance tree that reaches
# this size has already lost whatever argument the cap was going to settle.
_MAX_TREE_MEMBERS = 4096


@lru_cache(maxsize=1)
def _darwin_libproc() -> tuple[Any, Any] | None:
    """Bind this module's own ``libproc`` handle, or ``None`` off macOS.

    ``nanobot/process_runtime.py`` binds ``proc_pidinfo`` as well, for process
    birth times. That binding is private to a module the gateway runtime imports
    on every start, so the fleet keeps its own handle here instead of reaching
    into it or widening its API.
    """
    if sys.platform != "darwin":
        return None
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_listpgrppids = libproc.proc_listpgrppids
    except (AttributeError, OSError):
        return None
    proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    proc_pidinfo.restype = ctypes.c_int
    proc_listpgrppids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    proc_listpgrppids.restype = ctypes.c_int
    return proc_pidinfo, proc_listpgrppids


@lru_cache(maxsize=1)
def _darwin_proc_listchildpids() -> Any | None:
    """Bind ``proc_listchildpids``, or ``None`` off macOS.

    Bound separately from :func:`_darwin_libproc` rather than added to its tuple
    so that the group walk and the descendant walk stay independently
    substitutable: a kernel that stopped exporting one must not take the other
    down with it, and the union in :func:`process_tree_pids` is worth more than
    either half.
    """
    if sys.platform != "darwin":
        return None
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_listchildpids = libproc.proc_listchildpids
    except (AttributeError, OSError):
        return None
    proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    proc_listchildpids.restype = ctypes.c_int
    return proc_listchildpids


def process_child_pids(pid: int) -> list[int] | None:
    """List the direct children of ``pid``, or ``None`` if unsupported.

    A process with no children is an empty list. So is a pid that no longer
    exists: the kernel reports "nothing has this parent" for both, and they are
    not worth distinguishing here — a dead process has no live children either
    way.
    """
    proc_listchildpids = _darwin_proc_listchildpids()
    if proc_listchildpids is None or pid <= 0:
        return None
    capacity = _PID_LIST_INITIAL_CAPACITY
    while capacity <= _PID_LIST_MAX_CAPACITY:
        buffer = (ctypes.c_int32 * capacity)()
        try:
            # Like ``proc_listpgrppids``: the number of pids written, saturating
            # at the buffer's capacity. A full buffer is indistinguishable from
            # an overflowing one, so grow and retry rather than lose a subtree.
            written = proc_listchildpids(pid, buffer, ctypes.sizeof(buffer))
        except (OSError, ValueError):
            return None
        if written < 0:
            return None
        if written >= capacity:
            capacity *= 2
            continue
        return [int(child) for child in buffer[:written] if child > 0]
    return None


def process_descendant_pids(pid: int) -> list[int] | None:
    """Walk ``pid``'s whole descendant closure, excluding ``pid`` itself.

    The half of the tree the process group cannot see: a child started with
    ``setsid`` keeps its parent, so descent by parentage finds what descent by
    group does not.

    Returns ``None`` only when the walk cannot be done at all — an unsupported
    platform, or a failure reading ``pid``'s own children. A failure deeper in
    the tree is skipped rather than fatal, on the same reasoning as a member that
    exits mid-sample: the alternative is discarding an otherwise usable reading
    of a tree that is always changing while it is read.

    Already-visited pids are never revisited, so a pid recycled into the walk's
    own results cannot make it loop.
    """
    roots = process_child_pids(pid)
    if roots is None:
        return None
    seen: set[int] = {pid}
    found: list[int] = []
    pending = list(roots)
    while pending and len(found) < _MAX_TREE_MEMBERS:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        found.append(current)
        children = process_child_pids(current)
        if children:
            pending.extend(children)
    return found


def process_group_pids(process_group: int) -> list[int] | None:
    """List the live members of ``process_group``, or ``None`` if unsupported.

    A group with no live members is an empty list, not ``None``: the group is
    knowably gone rather than unknowable.
    """
    handles = _darwin_libproc()
    if handles is None or process_group <= 0:
        return None
    _, proc_listpgrppids = handles
    capacity = _PID_LIST_INITIAL_CAPACITY
    while capacity <= _PID_LIST_MAX_CAPACITY:
        buffer = (ctypes.c_int32 * capacity)()
        try:
            # Returns the number of pids written, saturating at the capacity of
            # the buffer, or -1 on error. A saturated result is indistinguishable
            # from an exactly-full one, so both grow and retry rather than risk
            # silently under-reporting a tree's memory.
            written = proc_listpgrppids(process_group, buffer, ctypes.sizeof(buffer))
        except (OSError, ValueError):
            return None
        if written < 0:
            return None
        if written >= capacity:
            capacity *= 2
            continue
        return [int(pid) for pid in buffer[:written] if pid > 0]
    return None


def process_resident_bytes(pid: int) -> int | None:
    """Read one process's resident size in bytes, or ``None`` if unreadable.

    ``None`` covers an unsupported platform, a process that has already exited,
    and a process this user may not inspect.
    """
    handles = _darwin_libproc()
    if handles is None or pid <= 0:
        return None
    proc_pidinfo, _ = handles
    buffer = ctypes.create_string_buffer(_PROC_TASKINFO_SIZE)
    try:
        written = proc_pidinfo(pid, _PROC_PIDTASKINFO, 0, buffer, len(buffer))
    except (OSError, ValueError):
        return None
    # Anything short of the full struct means the layout this module hard-codes
    # is not the layout the running kernel has; fail closed instead of reading a
    # field out of a partially written buffer.
    if written != len(buffer):
        return None
    return int(struct.unpack_from("=Q", buffer, _PTI_RESIDENT_SIZE_OFFSET)[0])


def process_tree_pids(process_group: int) -> list[int] | None:
    """List every live process belonging to an instance's tree.

    The union of ``process_group``'s members and the descendant closure of its
    leader — whose pid *is* the group id, by definition of a process group. See
    this module's docstring for why neither enumeration alone is the tree.

    Returns ``None`` only when neither walk could be done; a tree with no live
    members is an empty list, so "this instance is using nothing" stays
    distinguishable from "this instance cannot be read".
    """
    members = process_group_pids(process_group)
    descendants = process_descendant_pids(process_group)
    if members is None and descendants is None:
        return None
    pids = dict.fromkeys(members or ())
    if descendants is not None:
        if members is None:
            # The group walk was the half that failed, so nothing has named the
            # leader yet. It is already a member whenever that walk succeeded,
            # and must not be invented when the group is knowably empty — that
            # is how an exited tree keeps sampling as zero rather than as its
            # own recycled pid.
            pids.setdefault(process_group)
        pids.update(dict.fromkeys(descendants))
    return list(pids)


def process_group_memory_bytes(process_group: int) -> int | None:
    """Sum the resident memory of every live process in ``process_group``.

    The group only. :func:`process_tree_memory_bytes` is what the fleet's cap is
    enforced on; this is the narrower reading, kept because "which of the two
    walks saw it" is exactly the question to ask when a tree's total surprises
    somebody.

    Returns bytes, or ``None`` when the group cannot be sampled at all — an
    unsupported platform, or a failed group walk. A group whose members have all
    exited samples as ``0``, so a supervisor never mistakes an already-dead
    instance for one over its cap.

    Members that exit between the walk and their own sample are skipped: a tree
    is sampled while it is changing, and the alternative is discarding an
    otherwise usable reading.
    """
    return _resident_total(process_group_pids(process_group))


def process_tree_memory_bytes(process_group: int) -> int | None:
    """Sum the resident memory of an instance's whole process tree.

    The fleet's metric, and the default sampler behind
    :class:`nanobot.fleet.cap.MemoryCap`. Covers the descendants that left the
    process group by starting their own session — which is every command the
    instance's shell tool runs, and so very nearly every way an instance can
    allocate without bound.

    Returns bytes, or ``None`` when the tree cannot be enumerated at all. A tree
    whose members have all exited samples as ``0``.
    """
    return _resident_total(process_tree_pids(process_group))


def _resident_total(pids: list[int] | None) -> int | None:
    """Sum ``pids``' resident sizes, or ``None`` for an enumeration that failed.

    A pid that exits between the walk and its own sample contributes nothing
    rather than voiding the reading: a tree is always sampled while it changes.
    """
    if pids is None:
        return None
    total = 0
    for pid in pids:
        resident = process_resident_bytes(pid)
        if resident is not None:
            total += resident
    return total

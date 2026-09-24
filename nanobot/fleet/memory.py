"""Resident memory sampling for a whole process group.

The fleet supervisor caps each instance on the memory of its *entire* process
tree, not just the instance process, because a shell tool can fork a child that
allocates without bound. Every instance is started in its own process group, so
the tree and the group are the same set of processes and the group is what the
kernel will let us enumerate cheaply.

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


def process_group_memory_bytes(process_group: int) -> int | None:
    """Sum the resident memory of every live process in ``process_group``.

    Returns bytes, or ``None`` when the group cannot be sampled at all — an
    unsupported platform, or a failed group walk. A group whose members have all
    exited samples as ``0``, so a supervisor never mistakes an already-dead
    instance for one over its cap.

    Members that exit between the walk and their own sample are skipped: a tree
    is sampled while it is changing, and the alternative is discarding an
    otherwise usable reading.
    """
    pids = process_group_pids(process_group)
    if pids is None:
        return None
    total = 0
    for pid in pids:
        resident = process_resident_bytes(pid)
        if resident is not None:
            total += resident
    return total

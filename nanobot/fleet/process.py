"""Process-tree inspection and termination for supervised instances."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass

PS_COMMAND = ("/bin/ps", "-axo", "pid=,ppid=,rss=")


@dataclass(frozen=True)
class ProcessSample:
    """One row of the process table: identity, parent and resident memory."""

    pid: int
    ppid: int
    rss_kb: int


def parse_ps_output(text: str) -> tuple[ProcessSample, ...]:
    """Parse ``pid ppid rss`` rows, skipping anything unparseable."""
    samples: list[ProcessSample] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            samples.append(ProcessSample(int(fields[0]), int(fields[1]), int(fields[2])))
        except ValueError:
            continue
    return tuple(samples)


def read_process_table() -> tuple[ProcessSample, ...]:
    """Snapshot every process on the host."""
    try:
        result = subprocess.run(
            list(PS_COMMAND), capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return parse_ps_output(result.stdout)


def _children_by_parent(samples: Iterable[ProcessSample]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for sample in samples:
        children.setdefault(sample.ppid, []).append(sample.pid)
    return children


def collect_tree_pids(root_pid: int, samples: Sequence[ProcessSample]) -> tuple[int, ...]:
    """Return *root_pid* and every descendant present in *samples*.

    ``seen`` also guards against a parent cycle, which a corrupt or racing
    ``ps`` snapshot can produce.
    """
    known = {sample.pid for sample in samples}
    if root_pid not in known:
        return ()
    children = _children_by_parent(samples)
    out: list[int] = []
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        stack.extend(children.get(pid, ()))
    return tuple(out)


def process_tree_rss_kb(root_pid: int, samples: Sequence[ProcessSample]) -> int:
    """Total resident memory of *root_pid* and every descendant, in KiB."""
    rss = {sample.pid: sample.rss_kb for sample in samples}
    return sum(rss.get(pid, 0) for pid in collect_tree_pids(root_pid, samples))


def process_is_alive(pid: int) -> bool:
    """Return whether *pid* still names a live process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_pid(pid: int, sig: int) -> None:
    with suppress(OSError):
        os.kill(pid, sig)


def _signal_group(pid: int, sig: int) -> None:
    with suppress(OSError):
        os.killpg(pid, sig)


def terminate_tree(
    root_pid: int,
    *,
    samples: Sequence[ProcessSample] = (),
    grace: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    is_alive: Callable[[int], bool] = process_is_alive,
    signal_pid: Callable[[int, int], None] = _signal_pid,
    signal_group: Callable[[int, int], None] = _signal_group,
) -> None:
    """Terminate *root_pid* and everything it started.

    Instances are launched in their own session, so signalling the process
    group reaches the shell tool's children too.  The pids captured from
    *samples* are signalled as well: a descendant that called ``setsid`` left
    the group, and once the root dies its ancestry is no longer discoverable.
    """
    pids = [pid for pid in collect_tree_pids(root_pid, samples) if pid != root_pid]
    signal_group(root_pid, signal.SIGTERM)
    signal_pid(root_pid, signal.SIGTERM)
    for pid in pids:
        signal_pid(pid, signal.SIGTERM)

    deadline = clock() + grace
    remaining = [root_pid, *pids]
    while clock() < deadline:
        remaining = [pid for pid in remaining if is_alive(pid)]
        if not remaining:
            return
        sleep(0.05)

    signal_group(root_pid, signal.SIGKILL)
    for pid in [root_pid, *pids]:
        if is_alive(pid):
            signal_pid(pid, signal.SIGKILL)

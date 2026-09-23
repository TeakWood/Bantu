"""The fleet supervisor: one OS process per instance, confined and capped.

The supervisor owns three jobs and nothing else.  It launches each instance
under its own Seatbelt profile with its own environment, it watches each
instance's process tree against that instance's memory cap, and it publishes
what it knows so `fleet status` and `fleet stop` can work from another shell.

It deliberately does not restart anything.  An instance that exits — because it
crashed, because it was killed, or because it outgrew its cap — stays exited,
and the others keep serving.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from nanobot.fleet.config import Fleet, FleetInstance
from nanobot.fleet.environment import instance_environment
from nanobot.fleet.process import (
    ProcessSample,
    process_is_alive,
    process_tree_rss_kb,
    read_process_table,
    terminate_tree,
)
from nanobot.fleet.sandbox import (
    confinement_available,
    instance_command,
    instance_profile,
    sandbox_command,
)
from nanobot.fleet.state import clear_state, fleet_run_dir, read_state, write_state

STATE_RUNNING = "running"
STATE_EXITED = "exited"

REASON_MEMORY = "memory"
REASON_SIGNAL = "signal"
REASON_EXIT = "exit"

# One second between samples leaves four to spare against the five-second
# deadline for killing an instance that crossed its cap.
POLL_INTERVAL = 1.0
KIB_PER_MIB = 1024


class LaunchedProcess(Protocol):
    """The part of :class:`subprocess.Popen` the supervisor depends on."""

    @property
    def pid(self) -> int: ...

    def poll(self) -> int | None: ...


Launcher = Callable[[FleetInstance, Sequence[str], Mapping[str, str], Path], LaunchedProcess]


@dataclass
class InstanceRecord:
    """The supervisor's view of one instance."""

    instance: FleetInstance
    pid: int | None = None
    # A record only becomes running once its process exists, so an instance
    # the supervisor never managed to launch is never reported as running.
    state: str = STATE_EXITED
    exit_reason: str | None = None
    process: LaunchedProcess | None = field(default=None, repr=False)

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.instance.name,
            "pid": self.pid,
            "state": self.state,
            "exit_reason": self.exit_reason,
            "workspace": str(self.instance.workspace),
            "config_dir": str(self.instance.config_dir),
            "memory_limit_mb": self.instance.memory_limit_mb,
        }


def spawn_instance(
    instance: FleetInstance,
    argv: Sequence[str],
    env: Mapping[str, str],
    log_path: Path,
) -> LaunchedProcess:
    """Start one instance detached into its own session.

    Its own session makes the instance a process-group leader, so signalling
    the group later reaches every process the instance started.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        return subprocess.Popen(
            list(argv),
            env=dict(env),
            cwd=str(instance.workspace),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )


class Supervisor:
    """Run every instance of a fleet, each in its own confined OS process."""

    def __init__(
        self,
        fleet: Fleet,
        *,
        environ: Mapping[str, str] | None = None,
        launcher: Launcher = spawn_instance,
        sampler: Callable[[], Sequence[ProcessSample]] = read_process_table,
        terminator: Callable[..., None] = terminate_tree,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval: float = POLL_INTERVAL,
        python_executable: str | None = None,
        confine: bool | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.fleet = fleet
        self._environ = dict(environ if environ is not None else os.environ)
        self._launcher = launcher
        self._sampler = sampler
        self._terminator = terminator
        self._sleep = sleep
        self._poll_interval = poll_interval
        self._python = python_executable or sys.executable
        self._confine = confinement_available() if confine is None else confine
        self._run_dir = run_dir or fleet_run_dir(fleet.path)
        self._records = {
            entry.name: InstanceRecord(instance=entry) for entry in fleet.instances
        }
        self._stopping = False

    # -- introspection ----------------------------------------------------

    @property
    def records(self) -> tuple[InstanceRecord, ...]:
        return tuple(self._records.values())

    def payload(self) -> dict[str, Any]:
        return {
            "fleet": str(self.fleet.path),
            "supervisor_pid": os.getpid(),
            "instances": [record.payload() for record in self._records.values()],
        }

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Launch every instance, or launch none of them."""
        self._run_dir.mkdir(parents=True, exist_ok=True)
        try:
            for record in self._records.values():
                self._launch(record)
        except Exception:
            # A fleet that cannot start whole must leave nothing behind.
            self.shutdown(reason=REASON_SIGNAL)
            raise
        self.publish()

    def _launch(self, record: InstanceRecord) -> None:
        instance = record.instance
        instance.workspace.mkdir(parents=True, exist_ok=True)
        argv = instance_command(instance, python_executable=self._python)
        if self._confine:
            profile = instance_profile(
                instance, others=self.fleet.instances, fleet_path=self.fleet.path
            )
            argv = sandbox_command(argv, profile=profile)
        process = self._launcher(
            instance,
            argv,
            instance_environment(instance, self._environ),
            self._run_dir / f"{instance.name}.log",
        )
        record.process = process
        record.pid = process.pid
        record.state = STATE_RUNNING
        record.exit_reason = None
        logger.info("fleet: started instance {} as pid {}", instance.name, process.pid)

    def run(self) -> None:
        """Start the fleet and supervise it until stopped or empty."""
        self.start()
        with self._signal_handlers():
            while not self._stopping and self._any_running():
                self._sleep(self._poll_interval)
                self.poll_once()
        if self._stopping:
            self.shutdown(reason=REASON_SIGNAL)

    def poll_once(self) -> None:
        """Reap exits, enforce memory caps, and publish the result."""
        self._reap()
        self._enforce_memory()
        self.publish()

    def shutdown(self, *, reason: str = REASON_SIGNAL) -> None:
        """Terminate every running instance's whole process tree."""
        samples = self._sampler()
        for record in self._records.values():
            if record.state != STATE_RUNNING or record.pid is None:
                continue
            self._terminator(record.pid, samples=samples)
            record.state = STATE_EXITED
            record.exit_reason = reason
        self.publish()

    # -- supervision ------------------------------------------------------

    def _any_running(self) -> bool:
        return any(record.state == STATE_RUNNING for record in self._records.values())

    def _reap(self) -> None:
        for record in self._records.values():
            if record.state != STATE_RUNNING or record.process is None:
                continue
            code = record.process.poll()
            if code is None:
                continue
            record.state = STATE_EXITED
            record.exit_reason = REASON_SIGNAL if code < 0 else REASON_EXIT
            logger.info(
                "fleet: instance {} exited ({}), not restarting",
                record.instance.name,
                record.exit_reason,
            )

    def _enforce_memory(self) -> None:
        running = [
            record
            for record in self._records.values()
            if record.state == STATE_RUNNING and record.pid is not None
        ]
        if not running:
            return
        samples = self._sampler()
        for record in running:
            assert record.pid is not None
            used_kb = process_tree_rss_kb(record.pid, samples)
            limit_kb = record.instance.memory_limit_mb * KIB_PER_MIB
            if used_kb <= limit_kb:
                continue
            logger.warning(
                "fleet: instance {} used {} MiB over its {} MiB cap; killing it",
                record.instance.name,
                used_kb // KIB_PER_MIB,
                record.instance.memory_limit_mb,
            )
            # Mark before killing so the reap that follows cannot relabel the
            # exit as a plain signal.
            record.state = STATE_EXITED
            record.exit_reason = REASON_MEMORY
            self._terminator(record.pid, samples=samples)

    def publish(self) -> None:
        write_state(self.fleet.path, self.payload())

    # -- signals ----------------------------------------------------------

    def request_stop(self) -> None:
        self._stopping = True

    @contextmanager
    def _signal_handlers(self) -> Iterator[None]:
        """Turn SIGTERM/SIGINT into an orderly shutdown of the whole fleet."""

        def handle(_signum: int, _frame: Any) -> None:
            self.request_stop()

        previous: dict[int, Any] = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            # signal.signal only works on the main thread of the main
            # interpreter; a supervisor driven from a test thread just runs
            # without handlers.
            with suppress(ValueError, OSError):
                previous[sig] = signal.signal(sig, handle)
        try:
            yield
        finally:
            for sig, handler in previous.items():
                with suppress(ValueError, OSError):
                    signal.signal(sig, handler)


def fleet_status(
    fleet_path: Path,
    *,
    is_alive: Callable[[int], bool] = process_is_alive,
) -> list[dict[str, Any]] | None:
    """Read the published status of the fleet at *fleet_path*.

    A supervisor that died without publishing would leave instances recorded as
    running, so each recorded pid is checked against the host before the entry
    is reported.  Returns ``None`` when no supervisor has published at all.
    """
    state = read_state(fleet_path)
    if state is None:
        return None
    entries: list[dict[str, Any]] = []
    for entry in state.get("instances", []):
        if not isinstance(entry, dict):
            continue
        record = dict(entry)  # pyright: ignore[reportUnknownArgumentType]
        pid = record.get("pid")
        if record.get("state") == STATE_RUNNING and (
            not isinstance(pid, int) or not is_alive(pid)
        ):
            record["state"] = STATE_EXITED
            record["exit_reason"] = record.get("exit_reason") or REASON_EXIT
        entries.append(record)
    return entries


@dataclass(frozen=True)
class StopResult:
    """Outcome of stopping a fleet."""

    stopped: bool
    message: str
    survivors: tuple[int, ...] = ()


def stop_fleet(
    fleet_path: Path,
    *,
    sampler: Callable[[], Sequence[ProcessSample]] = read_process_table,
    terminator: Callable[..., None] = terminate_tree,
    is_alive: Callable[[int], bool] = process_is_alive,
    signal_pid: Callable[[int, int], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    timeout: float = 10.0,
) -> StopResult:
    """Stop every instance of the fleet at *fleet_path*, then the supervisor.

    The process table is sampled *before* anything is signalled: once an
    instance dies its descendants are reparented and their ancestry is no
    longer discoverable, so the sweep needs the pids captured up front.
    """
    state = read_state(fleet_path)
    if state is None:
        return StopResult(False, f"no fleet supervisor state for {fleet_path}")

    send = signal_pid if signal_pid is not None else _default_signal
    samples = sampler()
    supervisor_pid = state.get("supervisor_pid")
    # A recycled pid in a stale state file must never turn `fleet stop` on the
    # process running it.
    if supervisor_pid == os.getpid():
        supervisor_pid = None
    if isinstance(supervisor_pid, int) and is_alive(supervisor_pid):
        send(supervisor_pid, signal.SIGTERM)
        deadline = clock() + timeout
        while clock() < deadline and is_alive(supervisor_pid):
            sleep(0.05)
        if is_alive(supervisor_pid):
            send(supervisor_pid, signal.SIGKILL)

    pids = [
        entry.get("pid")
        for entry in state.get("instances", [])
        if isinstance(entry, dict) and isinstance(entry.get("pid"), int)
    ]
    for pid in pids:
        assert isinstance(pid, int)
        if is_alive(pid):
            terminator(pid, samples=samples)

    survivors = tuple(pid for pid in pids if isinstance(pid, int) and is_alive(pid))
    clear_state(fleet_path)
    if survivors:
        return StopResult(False, f"instance processes still alive: {survivors}", survivors)
    return StopResult(True, "fleet stopped")


def _default_signal(pid: int, sig: int) -> None:
    with suppress(OSError):
        os.kill(pid, sig)

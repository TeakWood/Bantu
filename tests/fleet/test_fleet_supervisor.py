"""Supervisor lifecycle: launching, reaping, memory caps and stopping."""

from __future__ import annotations

import json
import signal
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from nanobot.fleet.config import Fleet, FleetInstance, load_fleet
from nanobot.fleet.process import ProcessSample
from nanobot.fleet.state import read_state, state_file, write_state
from nanobot.fleet.supervisor import (
    REASON_EXIT,
    REASON_MEMORY,
    REASON_SIGNAL,
    STATE_EXITED,
    STATE_RUNNING,
    Supervisor,
    fleet_status,
    stop_fleet,
)


class FakeProcess:
    """A launched instance the test can make exit on demand."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


class FakeLauncher:
    """Record every launch and hand back a controllable process."""

    def __init__(self, first_pid: int = 4100) -> None:
        self.calls: list[dict[str, Any]] = []
        self.processes: dict[str, FakeProcess] = {}
        self._next_pid = first_pid

    def __call__(
        self,
        instance: FleetInstance,
        argv: Sequence[str],
        env: Mapping[str, str],
        log_path: Path,
    ) -> FakeProcess:
        self.calls.append(
            {
                "name": instance.name,
                "argv": list(argv),
                "env": dict(env),
                "log_path": log_path,
            }
        )
        process = FakeProcess(self._next_pid)
        self._next_pid += 1
        self.processes[instance.name] = process
        return process


class RecordingTerminator:
    def __init__(self) -> None:
        self.killed: list[int] = []

    def __call__(self, pid: int, **_kwargs: Any) -> None:
        self.killed.append(pid)


@pytest.fixture
def fleet(
    make_entry: Callable[..., dict[str, Any]],
    write_fleet: Callable[..., Path],
) -> Fleet:
    return load_fleet(
        write_fleet(
            {
                "research": make_entry("research", memoryLimitMb=1, env=["SECRET_A"]),
                "trader": make_entry("trader", memoryLimitMb=1, env=["SECRET_B"]),
            }
        )
    )


def make_supervisor(
    fleet: Fleet,
    *,
    launcher: FakeLauncher | None = None,
    terminator: RecordingTerminator | None = None,
    samples: Sequence[ProcessSample] = (),
    environ: Mapping[str, str] | None = None,
    confine: bool = False,
) -> tuple[Supervisor, FakeLauncher, RecordingTerminator]:
    launcher = launcher or FakeLauncher()
    terminator = terminator or RecordingTerminator()
    supervisor = Supervisor(
        fleet,
        environ=environ or {"PATH": "/usr/bin", "SECRET_A": "a", "SECRET_B": "b", "SENTINEL": "s"},
        launcher=launcher,
        sampler=lambda: samples,
        terminator=terminator,
        sleep=lambda _seconds: None,
        confine=confine,
    )
    return supervisor, launcher, terminator


class TestStart:
    def test_each_instance_gets_its_own_process(self, fleet: Fleet) -> None:
        supervisor, launcher, _ = make_supervisor(fleet)

        supervisor.start()

        assert [call["name"] for call in launcher.calls] == ["research", "trader"]
        pids = [record.pid for record in supervisor.records]
        assert len(set(pids)) == 2
        assert all(record.state == STATE_RUNNING for record in supervisor.records)

    def test_publishes_status_readable_from_another_shell(self, fleet: Fleet) -> None:
        supervisor, _, _ = make_supervisor(fleet)

        supervisor.start()

        entries = fleet_status(fleet.path, is_alive=lambda _pid: True)
        assert entries is not None
        assert [entry["name"] for entry in entries] == ["research", "trader"]
        for entry in entries:
            assert entry["state"] == STATE_RUNNING
            assert entry["exit_reason"] is None
            assert entry["memory_limit_mb"] == 1
            assert set(entry) >= {
                "name",
                "pid",
                "state",
                "exit_reason",
                "workspace",
                "config_dir",
                "memory_limit_mb",
            }

    def test_published_pids_are_never_the_supervisors(self, fleet: Fleet) -> None:
        import os

        supervisor, _, _ = make_supervisor(fleet)

        supervisor.start()

        payload = json.loads(state_file(fleet.path).read_text(encoding="utf-8"))
        assert payload["supervisor_pid"] == os.getpid()
        assert all(entry["pid"] != os.getpid() for entry in payload["instances"])

    def test_each_instance_only_sees_its_own_credentials(self, fleet: Fleet) -> None:
        supervisor, launcher, _ = make_supervisor(fleet)

        supervisor.start()

        research = next(call for call in launcher.calls if call["name"] == "research")
        assert research["env"]["SECRET_A"] == "a"
        assert "SECRET_B" not in research["env"]
        assert "SENTINEL" not in research["env"]

    def test_confinement_wraps_the_command_in_sandbox_exec(self, fleet: Fleet) -> None:
        supervisor, launcher, _ = make_supervisor(fleet, confine=True)

        supervisor.start()

        research = next(call for call in launcher.calls if call["name"] == "research")
        assert research["argv"][0] == "/usr/bin/sandbox-exec"
        trader = fleet.instance("trader")
        assert trader is not None
        assert str(trader.workspace) in research["argv"][2]
        assert str(fleet.path) in research["argv"][2]

    def test_creates_each_workspace_before_launching(self, fleet: Fleet) -> None:
        supervisor, _, _ = make_supervisor(fleet)

        supervisor.start()

        assert all(entry.workspace.is_dir() for entry in fleet.instances)

    def test_a_failed_launch_leaves_no_instance_running(self, fleet: Fleet) -> None:
        launcher = FakeLauncher()
        terminator = RecordingTerminator()

        class FailingLauncher(FakeLauncher):
            def __call__(self, instance: FleetInstance, *args: Any) -> FakeProcess:
                if instance.name == "trader":
                    raise OSError("no such executable")
                return super().__call__(instance, *args)

        failing = FailingLauncher()
        supervisor, _, _ = make_supervisor(fleet, launcher=failing, terminator=terminator)

        with pytest.raises(OSError, match="no such executable"):
            supervisor.start()

        assert terminator.killed == [failing.processes["research"].pid]
        assert all(record.state == STATE_EXITED for record in supervisor.records)
        assert launcher.calls == []


class TestReaping:
    def test_an_exited_instance_is_reported_and_not_restarted(self, fleet: Fleet) -> None:
        supervisor, launcher, _ = make_supervisor(fleet)
        supervisor.start()
        launcher.processes["research"].returncode = -signal.SIGKILL

        supervisor.poll_once()
        supervisor.poll_once()

        entries = {record.instance.name: record for record in supervisor.records}
        assert entries["research"].state == STATE_EXITED
        assert entries["research"].exit_reason == REASON_SIGNAL
        assert entries["trader"].state == STATE_RUNNING
        # Two launches total: the dead instance was never started again.
        assert len(launcher.calls) == 2

    def test_a_clean_exit_is_reported_as_exit(self, fleet: Fleet) -> None:
        supervisor, launcher, _ = make_supervisor(fleet)
        supervisor.start()
        launcher.processes["research"].returncode = 0

        supervisor.poll_once()

        record = next(r for r in supervisor.records if r.instance.name == "research")
        assert record.exit_reason == REASON_EXIT

    def test_the_surviving_instance_keeps_its_pid(self, fleet: Fleet) -> None:
        supervisor, launcher, _ = make_supervisor(fleet)
        supervisor.start()
        trader_pid = launcher.processes["trader"].pid
        launcher.processes["research"].returncode = -9

        supervisor.poll_once()

        record = next(r for r in supervisor.records if r.instance.name == "trader")
        assert record.pid == trader_pid


class TestMemoryCap:
    def _samples(self, pid: int, rss_kb: int) -> tuple[ProcessSample, ...]:
        return (
            ProcessSample(pid=pid, ppid=1, rss_kb=rss_kb // 2),
            ProcessSample(pid=pid + 9000, ppid=pid, rss_kb=rss_kb - rss_kb // 2),
        )

    def test_a_tree_over_the_cap_is_killed_and_recorded_as_memory(self, fleet: Fleet) -> None:
        launcher = FakeLauncher()
        terminator = RecordingTerminator()
        supervisor = Supervisor(
            fleet,
            environ={},
            launcher=launcher,
            sampler=lambda: self._samples(launcher.processes["research"].pid, 4096),
            terminator=terminator,
            sleep=lambda _seconds: None,
            confine=False,
        )
        supervisor.start()

        supervisor.poll_once()

        entries = {record.instance.name: record for record in supervisor.records}
        assert entries["research"].state == STATE_EXITED
        assert entries["research"].exit_reason == REASON_MEMORY
        assert terminator.killed == [launcher.processes["research"].pid]
        assert entries["trader"].state == STATE_RUNNING

    def test_a_tree_within_the_cap_is_left_alone(self, fleet: Fleet) -> None:
        launcher = FakeLauncher()
        terminator = RecordingTerminator()
        supervisor = Supervisor(
            fleet,
            environ={},
            launcher=launcher,
            # 1 MiB cap, 512 KiB used.
            sampler=lambda: self._samples(launcher.processes["research"].pid, 512),
            terminator=terminator,
            sleep=lambda _seconds: None,
            confine=False,
        )
        supervisor.start()

        supervisor.poll_once()

        assert terminator.killed == []
        assert all(record.state == STATE_RUNNING for record in supervisor.records)

    def test_a_memory_kill_is_not_relabelled_by_the_next_reap(self, fleet: Fleet) -> None:
        launcher = FakeLauncher()
        supervisor = Supervisor(
            fleet,
            environ={},
            launcher=launcher,
            sampler=lambda: self._samples(launcher.processes["research"].pid, 4096),
            terminator=RecordingTerminator(),
            sleep=lambda _seconds: None,
            confine=False,
        )
        supervisor.start()
        supervisor.poll_once()
        launcher.processes["research"].returncode = -signal.SIGKILL

        supervisor.poll_once()

        record = next(r for r in supervisor.records if r.instance.name == "research")
        assert record.exit_reason == REASON_MEMORY


class TestRunAndShutdown:
    def test_run_returns_once_every_instance_has_exited(self, fleet: Fleet) -> None:
        launcher = FakeLauncher()
        supervisor = Supervisor(
            fleet,
            environ={},
            launcher=launcher,
            sampler=lambda: (),
            terminator=RecordingTerminator(),
            sleep=lambda _seconds: _exit_all(launcher),
            confine=False,
        )

        supervisor.run()

        assert all(record.state == STATE_EXITED for record in supervisor.records)

    def test_a_stop_request_terminates_every_running_tree(self, fleet: Fleet) -> None:
        launcher = FakeLauncher()
        terminator = RecordingTerminator()
        supervisor = Supervisor(
            fleet,
            environ={},
            launcher=launcher,
            sampler=lambda: (),
            terminator=terminator,
            sleep=lambda _seconds: supervisor.request_stop(),
            confine=False,
        )

        supervisor.run()

        assert sorted(terminator.killed) == sorted(
            process.pid for process in launcher.processes.values()
        )
        assert all(record.exit_reason == REASON_SIGNAL for record in supervisor.records)

    def test_shutdown_skips_instances_that_already_exited(self, fleet: Fleet) -> None:
        supervisor, launcher, terminator = make_supervisor(fleet)
        supervisor.start()
        launcher.processes["research"].returncode = 0
        supervisor.poll_once()

        supervisor.shutdown()

        assert terminator.killed == [launcher.processes["trader"].pid]


def _exit_all(launcher: FakeLauncher) -> None:
    for process in launcher.processes.values():
        process.returncode = 0


class TestFleetStatus:
    def test_returns_none_without_a_supervisor(self, tmp_path: Path) -> None:
        assert fleet_status(tmp_path / "fleet.json") is None

    def test_reports_a_vanished_process_as_exited(self, fleet: Fleet) -> None:
        supervisor, _, _ = make_supervisor(fleet)
        supervisor.start()

        entries = fleet_status(fleet.path, is_alive=lambda _pid: False)

        assert entries is not None
        assert all(entry["state"] == STATE_EXITED for entry in entries)
        assert all(entry["exit_reason"] == REASON_EXIT for entry in entries)

    def test_keeps_a_recorded_exit_reason(self, fleet: Fleet) -> None:
        supervisor, launcher, _ = make_supervisor(fleet)
        supervisor.start()
        launcher.processes["research"].returncode = -9
        supervisor.poll_once()

        entries = fleet_status(fleet.path, is_alive=lambda _pid: False)

        assert entries is not None
        research = next(entry for entry in entries if entry["name"] == "research")
        assert research["exit_reason"] == REASON_SIGNAL


class TestStopFleet:
    def test_reports_when_no_supervisor_ever_ran(self, tmp_path: Path) -> None:
        result = stop_fleet(tmp_path / "fleet.json")

        assert result.stopped is False
        assert result.survivors == ()

    def test_stops_the_supervisor_then_sweeps_the_instances(self, fleet: Fleet) -> None:
        supervisor, launcher, _ = make_supervisor(fleet)
        supervisor.start()
        instance_pids = [process.pid for process in launcher.processes.values()]
        supervisor_pid = _repoint_supervisor_pid(fleet)
        alive: set[int] = {supervisor_pid, *instance_pids}
        signalled: list[tuple[int, int]] = []
        terminator = RecordingTerminator()

        def send(pid: int, sig: int) -> None:
            signalled.append((pid, sig))
            alive.discard(pid)

        def kill_tree(pid: int, **_kwargs: Any) -> None:
            terminator(pid)
            alive.discard(pid)

        result = stop_fleet(
            fleet.path,
            sampler=lambda: (),
            terminator=kill_tree,
            is_alive=lambda pid: pid in alive,
            signal_pid=send,
            sleep=lambda _seconds: None,
            clock=lambda: 0.0,
        )

        assert result.stopped is True
        assert sorted(terminator.killed) == sorted(instance_pids)
        # The supervisor is asked to shut down first, before the sweep.
        assert signalled == [(supervisor_pid, signal.SIGTERM)]
        assert read_state(fleet.path) is None

    def test_a_wedged_supervisor_is_killed(self, fleet: Fleet) -> None:
        supervisor, _, _ = make_supervisor(fleet)
        supervisor.start()
        supervisor_pid = _repoint_supervisor_pid(fleet)
        signalled: list[tuple[int, int]] = []

        result = stop_fleet(
            fleet.path,
            sampler=lambda: (),
            terminator=lambda _pid, **_kwargs: None,
            is_alive=lambda pid: pid == supervisor_pid,
            signal_pid=lambda pid, sig: signalled.append((pid, sig)),
            sleep=lambda _seconds: None,
            clock=iter([0.0, 0.0, 100.0]).__next__,
        )

        assert signalled == [
            (supervisor_pid, signal.SIGTERM),
            (supervisor_pid, signal.SIGKILL),
        ]
        assert result.stopped is True

    def test_never_signals_the_process_calling_it(self, fleet: Fleet) -> None:
        # A recycled pid in a stale state file must not stop `fleet stop`
        # itself.
        import os

        supervisor, _, _ = make_supervisor(fleet)
        supervisor.start()
        signalled: list[tuple[int, int]] = []

        stop_fleet(
            fleet.path,
            sampler=lambda: (),
            terminator=lambda _pid, **_kwargs: None,
            is_alive=lambda pid: pid == os.getpid(),
            signal_pid=lambda pid, sig: signalled.append((pid, sig)),
            sleep=lambda _seconds: None,
            clock=lambda: 0.0,
        )

        assert signalled == []

    def test_reports_survivors_that_refuse_to_die(self, fleet: Fleet) -> None:
        supervisor, launcher, _ = make_supervisor(fleet)
        supervisor.start()
        stubborn = launcher.processes["research"].pid

        result = stop_fleet(
            fleet.path,
            sampler=lambda: (),
            terminator=lambda _pid, **_kwargs: None,
            is_alive=lambda pid: pid == stubborn,
            signal_pid=lambda _pid, _sig: None,
            sleep=lambda _seconds: None,
            clock=lambda: 0.0,
        )

        assert result.stopped is False
        assert result.survivors == (stubborn,)


def _repoint_supervisor_pid(fleet: Fleet, pid: int = 999_001) -> int:
    """Publish a supervisor pid other than the test runner's own.

    ``stop_fleet`` refuses to signal the process calling it, and here that
    process is the one that published the state.
    """
    state = read_state(fleet.path)
    assert state is not None
    state["supervisor_pid"] = pid
    write_state(fleet.path, state)
    return pid

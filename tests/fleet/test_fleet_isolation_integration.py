"""End-to-end checks against the real operating system.

These exercise the two pieces that only the OS can confirm: the Seatbelt
profile actually denies another instance's files, and terminating an instance
actually takes its descendants with it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nanobot.fleet.config import load_fleet
from nanobot.fleet.process import (
    process_is_alive,
    process_tree_rss_kb,
    read_process_table,
    terminate_tree,
)
from nanobot.fleet.sandbox import confinement_available, instance_profile, sandbox_command

requires_seatbelt = pytest.mark.skipif(
    not confinement_available(), reason="fleet confinement needs macOS sandbox-exec"
)


def _run_confined(script: str, *, profile: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        sandbox_command(["/bin/sh", "-c", script], profile=profile),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(cwd),
        timeout=60,
    )


@pytest.fixture
def two_instances(
    make_entry: Callable[..., dict[str, Any]],
    write_fleet: Callable[..., Path],
) -> Any:
    fleet = load_fleet(
        write_fleet(
            {
                "research": make_entry("research", env=["SECRET_A"]),
                "trader": make_entry("trader", env=["SECRET_B"]),
            }
        )
    )
    for instance in fleet.instances:
        instance.workspace.mkdir(parents=True, exist_ok=True)
        (instance.workspace / "secret.txt").write_text(f"{instance.name} data", encoding="utf-8")
    return fleet


@requires_seatbelt
class TestFilesystemConfinement:
    def test_an_instance_cannot_reach_another_instances_files(self, two_instances: Any) -> None:
        research = two_instances.instance("research")
        trader = two_instances.instance("trader")
        profile = instance_profile(
            research, others=two_instances.instances, fleet_path=two_instances.path
        )

        denied = {
            "read_workspace": f"cat {trader.workspace}/secret.txt",
            "read_config": f"cat {trader.config_path}",
            "list_workspace": f"ls {trader.workspace}",
            "write_workspace": f"touch {trader.workspace}/intruder",
            "read_fleet_file": f"cat {two_instances.path}",
        }
        outcomes = {
            label: _run_confined(script, profile=profile, cwd=research.workspace).returncode
            for label, script in denied.items()
        }

        assert all(code != 0 for code in outcomes.values()), outcomes
        assert not (trader.workspace / "intruder").exists()
        assert (trader.workspace / "secret.txt").read_text(encoding="utf-8") == "trader data"

    def test_the_same_shell_still_owns_its_own_workspace(self, two_instances: Any) -> None:
        research = two_instances.instance("research")
        profile = instance_profile(
            research, others=two_instances.instances, fleet_path=two_instances.path
        )

        result = _run_confined(
            f"cat {research.workspace}/secret.txt && echo written > {research.workspace}/own.txt",
            profile=profile,
            cwd=research.workspace,
        )

        assert result.returncode == 0, result.stderr
        assert "research data" in result.stdout
        assert (research.workspace / "own.txt").read_text(encoding="utf-8") == "written\n"

    def test_an_instance_owns_its_config_directory(self, two_instances: Any) -> None:
        research = two_instances.instance("research")
        profile = instance_profile(
            research, others=two_instances.instances, fleet_path=two_instances.path
        )

        result = _run_confined(
            f"cat {research.config_path} && touch {research.config_dir}/sessions.db",
            profile=profile,
            cwd=research.workspace,
        )

        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout.splitlines()[0])["agents"]["defaults"]["workspace"]

    def test_a_fleet_file_inside_the_config_directory_stays_denied(
        self,
        make_entry: Callable[..., dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        # The deny must outrank the instance's own config-directory allow.
        research_entry = make_entry("research")
        config_dir = Path(research_entry["config"]).parent
        fleet_path = config_dir / "fleet.json"
        fleet_path.write_text(
            json.dumps({"instances": {"research": research_entry}}), encoding="utf-8"
        )
        fleet = load_fleet(fleet_path)
        research = fleet.instance("research")
        assert research is not None
        research.workspace.mkdir(parents=True, exist_ok=True)
        profile = instance_profile(research, others=fleet.instances, fleet_path=fleet.path)

        result = _run_confined(
            f"cat {fleet.path}", profile=profile, cwd=research.workspace
        )

        assert result.returncode != 0
        assert "research" not in result.stdout


@requires_seatbelt
class TestSupervisedFleet:
    """Drive a real supervisor over real, confined OS processes."""

    def _script(self, instance: Any, peer: Any, fleet_path: Path) -> str:
        workspace = instance.workspace
        return "\n".join(
            [
                f"/usr/bin/env > {workspace}/env.txt",
                f"cat {peer.workspace}/secret.txt > {workspace}/stolen.txt 2>/dev/null",
                f"cat {fleet_path} > {workspace}/fleet.txt 2>/dev/null",
                f"touch {peer.workspace}/intruder 2>/dev/null",
                f"echo mine > {workspace}/own.txt",
                "sleep 60 &",
                "echo $! > " f"{workspace}/child.pid",
                "wait",
            ]
        )

    def _wait_for(self, path: Path, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return
            time.sleep(0.05)
        raise AssertionError(f"{path} never appeared")

    def test_confined_instances_run_kill_and_stop_independently(
        self, two_instances: Any
    ) -> None:
        from nanobot.fleet.supervisor import (
            REASON_SIGNAL,
            STATE_EXITED,
            STATE_RUNNING,
            Supervisor,
            fleet_status,
            spawn_instance,
            stop_fleet,
        )

        fleet = two_instances
        research = fleet.instance("research")
        trader = fleet.instance("trader")
        scripts = {
            "research": self._script(research, trader, fleet.path),
            "trader": self._script(trader, research, fleet.path),
        }

        def launcher(instance: Any, argv: Any, env: Any, log_path: Path) -> Any:
            # Stand in for `nanobot gateway`: same confinement, same
            # environment, same session, but a scripted shell instead.
            return spawn_instance(
                instance,
                [*argv[:3], "/bin/sh", "-c", scripts[instance.name]],
                env,
                log_path,
            )

        supervisor = Supervisor(
            fleet,
            environ={
                "PATH": "/usr/bin:/bin",
                "HOME": str(research.config_dir),
                "SECRET_A": "a-value",
                "SECRET_B": "b-value",
                "SENTINEL": "leak",
            },
            launcher=launcher,
            confine=True,
            sleep=lambda _seconds: None,
        )

        try:
            supervisor.start()
            self._wait_for(research.workspace / "own.txt")
            self._wait_for(trader.workspace / "own.txt")

            entries = {entry["name"]: entry for entry in (fleet_status(fleet.path) or [])}
            assert {entry["state"] for entry in entries.values()} == {STATE_RUNNING}
            pids = {entry["pid"] for entry in entries.values()}
            assert len(pids) == 2
            assert os.getpid() not in pids
            assert all(process_is_alive(pid) for pid in pids)

            # Criterion 4: A reached nothing of B's, and kept its own files.
            assert (research.workspace / "stolen.txt").read_text(encoding="utf-8") == ""
            assert (research.workspace / "fleet.txt").read_text(encoding="utf-8") == ""
            assert not (trader.workspace / "intruder").exists()
            assert (research.workspace / "own.txt").read_text(encoding="utf-8") == "mine\n"
            assert (trader.workspace / "secret.txt").read_text(encoding="utf-8") == "trader data"

            # Criterion 5: only its own secret reached the process.
            environment = (research.workspace / "env.txt").read_text(encoding="utf-8")
            assert "SECRET_A=a-value" in environment
            assert "SECRET_B" not in environment
            assert "SENTINEL" not in environment

            # Criterion 3: killing A leaves B running, and A is not restarted.
            research_pid = entries["research"]["pid"]
            trader_pid = entries["trader"]["pid"]
            os.kill(research_pid, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                supervisor.poll_once()
                after = {entry["name"]: entry for entry in (fleet_status(fleet.path) or [])}
                if after["research"]["state"] == STATE_EXITED:
                    break
                time.sleep(0.05)
            assert after["research"]["state"] == STATE_EXITED
            assert after["research"]["exit_reason"] == REASON_SIGNAL
            assert after["trader"]["state"] == STATE_RUNNING
            assert after["trader"]["pid"] == trader_pid

            # Criterion 8: stopping leaves nothing behind, shell children included.
            child_pid = int(
                (trader.workspace / "child.pid").read_text(encoding="utf-8").strip()
            )
            def alive(pid: int) -> bool:
                # This test process is the instances' parent, so a killed
                # instance stays visible to kill(pid, 0) until it is reaped.
                # poll() reaps it and reports the real lifecycle state. A real
                # supervisor is killed first, so its children are reparented
                # and reaped by the OS.
                for record in supervisor.records:
                    if record.process is not None:
                        record.process.poll()
                return process_is_alive(pid)

            assert stop_fleet(fleet.path, is_alive=alive).stopped is True
            assert not alive(trader_pid)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and process_is_alive(child_pid):
                time.sleep(0.05)
            assert not process_is_alive(child_pid)
        finally:
            supervisor.shutdown()


@requires_seatbelt
class TestRealMemoryCap:
    def test_an_instance_over_its_cap_is_killed_while_its_peer_serves(
        self,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        import sys

        from nanobot.fleet.supervisor import (
            REASON_MEMORY,
            STATE_EXITED,
            STATE_RUNNING,
            Supervisor,
            spawn_instance,
        )

        fleet = load_fleet(
            write_fleet(
                {
                    "research": make_entry("research", memoryLimitMb=64),
                    "trader": make_entry("trader", memoryLimitMb=64),
                }
            )
        )
        # research starts a child that allocates far past its 64 MiB cap;
        # trader just idles.
        hog = f"{sys.executable} -c 'x = bytearray(400 * 1024 * 1024); import time; time.sleep(60)'"
        scripts = {"research": f"{hog}\nwait", "trader": "sleep 60"}

        def launcher(instance: Any, argv: Any, env: Any, log_path: Path) -> Any:
            return spawn_instance(
                instance, [*argv[:3], "/bin/sh", "-c", scripts[instance.name]], env, log_path
            )

        supervisor = Supervisor(
            fleet,
            environ={"PATH": "/usr/bin:/bin", "HOME": str(fleet.instances[0].config_dir)},
            launcher=launcher,
            confine=True,
            sleep=lambda _seconds: None,
        )
        try:
            supervisor.start()
            records = {record.instance.name: record for record in supervisor.records}
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                supervisor.poll_once()
                if records["research"].state == STATE_EXITED:
                    break
                time.sleep(0.2)

            assert records["research"].state == STATE_EXITED
            assert records["research"].exit_reason == REASON_MEMORY
            assert records["trader"].state == STATE_RUNNING
            assert process_is_alive(records["trader"].pid or 0)
        finally:
            supervisor.shutdown()


class TestRealProcessTree:
    def test_terminating_an_instance_takes_its_children_with_it(self, tmp_path: Path) -> None:
        marker = tmp_path / "child.pid"
        process = subprocess.Popen(
            [
                "/bin/sh",
                "-c",
                f"sleep 30 & echo $! > {marker}; sleep 30",
            ],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.05)
        child_pid = int(marker.read_text(encoding="utf-8").strip())
        assert process_is_alive(child_pid)

        samples = read_process_table()
        assert process_tree_rss_kb(process.pid, samples) > 0
        terminate_tree(process.pid, samples=samples, grace=3.0)
        process.wait(timeout=10)

        assert not process_is_alive(process.pid)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and process_is_alive(child_pid):
            time.sleep(0.05)
        assert not process_is_alive(child_pid)

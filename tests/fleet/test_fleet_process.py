"""Process-tree measurement and termination."""

from __future__ import annotations

import signal

from nanobot.fleet.process import (
    ProcessSample,
    collect_tree_pids,
    parse_ps_output,
    process_is_alive,
    process_tree_rss_kb,
    read_process_table,
    terminate_tree,
)

# supervisor 100 -> instance 200 -> shell 300 -> hog 400, plus unrelated 500.
TREE = (
    ProcessSample(pid=100, ppid=1, rss_kb=1_000),
    ProcessSample(pid=200, ppid=100, rss_kb=10_000),
    ProcessSample(pid=300, ppid=200, rss_kb=2_000),
    ProcessSample(pid=400, ppid=300, rss_kb=500_000),
    ProcessSample(pid=500, ppid=1, rss_kb=999_999),
)


class TestParsing:
    def test_parses_pid_ppid_rss_rows(self) -> None:
        samples = parse_ps_output("  100     1   1024\n  200   100   2048\n")

        assert samples == (
            ProcessSample(100, 1, 1024),
            ProcessSample(200, 100, 2048),
        )

    def test_skips_short_and_unparseable_rows(self) -> None:
        samples = parse_ps_output("PID PPID RSS\n100 1\n\nbad row here\n200 100 5\n")

        assert samples == (ProcessSample(200, 100, 5),)

    def test_reads_the_real_process_table(self) -> None:
        samples = read_process_table()

        assert any(sample.pid > 0 for sample in samples)


class TestTreeMeasurement:
    def test_sums_the_whole_descendant_tree(self) -> None:
        assert process_tree_rss_kb(200, TREE) == 10_000 + 2_000 + 500_000

    def test_ignores_processes_outside_the_tree(self) -> None:
        assert process_tree_rss_kb(500, TREE) == 999_999

    def test_unknown_root_measures_nothing(self) -> None:
        assert process_tree_rss_kb(9999, TREE) == 0
        assert collect_tree_pids(9999, TREE) == ()

    def test_survives_a_parent_cycle(self) -> None:
        cyclic = (
            ProcessSample(pid=1, ppid=2, rss_kb=10),
            ProcessSample(pid=2, ppid=1, rss_kb=20),
        )

        assert sorted(collect_tree_pids(1, cyclic)) == [1, 2]
        assert process_tree_rss_kb(1, cyclic) == 30


class TestLiveness:
    def test_current_process_is_alive(self) -> None:
        import os

        assert process_is_alive(os.getpid()) is True

    def test_non_positive_pid_is_never_alive(self) -> None:
        assert process_is_alive(0) is False
        assert process_is_alive(-1) is False


class TestTermination:
    def test_signals_the_group_and_every_known_descendant(self) -> None:
        sent: list[tuple[str, int, int]] = []

        terminate_tree(
            200,
            samples=TREE,
            sleep=lambda _seconds: None,
            clock=lambda: 0.0,
            is_alive=lambda _pid: False,
            signal_pid=lambda pid, sig: sent.append(("pid", pid, sig)),
            signal_group=lambda pid, sig: sent.append(("group", pid, sig)),
        )

        assert ("group", 200, signal.SIGTERM) in sent
        assert {entry[1] for entry in sent if entry[0] == "pid"} == {200, 300, 400}
        assert all(entry[2] == signal.SIGTERM for entry in sent)

    def test_escalates_to_sigkill_when_the_tree_outlives_the_grace_period(self) -> None:
        sent: list[tuple[str, int, int]] = []
        clock = iter([0.0, 0.0, 10.0]).__next__

        terminate_tree(
            200,
            samples=TREE,
            sleep=lambda _seconds: None,
            clock=clock,
            is_alive=lambda _pid: True,
            signal_pid=lambda pid, sig: sent.append(("pid", pid, sig)),
            signal_group=lambda pid, sig: sent.append(("group", pid, sig)),
        )

        kills = [entry for entry in sent if entry[2] == signal.SIGKILL]
        assert ("group", 200, signal.SIGKILL) in kills
        assert {entry[1] for entry in kills if entry[0] == "pid"} == {200, 300, 400}

    def test_stops_early_once_nothing_is_left(self) -> None:
        sent: list[tuple[str, int, int]] = []

        terminate_tree(
            200,
            samples=TREE,
            sleep=lambda _seconds: None,
            clock=iter([0.0, 0.0]).__next__,
            is_alive=lambda _pid: False,
            signal_pid=lambda pid, sig: sent.append(("pid", pid, sig)),
            signal_group=lambda pid, sig: sent.append(("group", pid, sig)),
        )

        assert not [entry for entry in sent if entry[2] == signal.SIGKILL]

    def test_signals_an_unknown_root_on_its_own(self) -> None:
        sent: list[tuple[str, int, int]] = []

        terminate_tree(
            9999,
            samples=TREE,
            sleep=lambda _seconds: None,
            clock=lambda: 0.0,
            is_alive=lambda _pid: False,
            signal_pid=lambda pid, sig: sent.append(("pid", pid, sig)),
            signal_group=lambda pid, sig: sent.append(("group", pid, sig)),
        )

        assert ("pid", 9999, signal.SIGTERM) in sent

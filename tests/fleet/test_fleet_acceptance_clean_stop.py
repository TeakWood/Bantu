"""Acceptance criterion 8: a clean stop, checked from outside with ``ps``.

Asked the way the criterion words it. A real fleet is started by the real
``nanobot fleet start`` in its own process, one of its instances is driven
*through its own shell tool* to start a long-running child, and then the real
``nanobot fleet stop`` is invoked. The moment that command returns — with
nothing waited for in between, because the contract is that stop does not return
until the fleet is gone — the process table is read and must show nothing left:
no instance, nothing in any instance's process group, not the descendant, and
not the supervisor.

Four things about the shape are load-bearing rather than incidental.

*The supervisor is a real separate process.* ``tests/fleet/test_fleet_stop.py``
already drives :func:`~nanobot.fleet.stop.stop_fleet` against real processes, but
in-process and against scripts merely *shaped* like a fleet. Neither can make the
claim this criterion ends on. A supervisor that is the pytest session itself is
one :func:`~nanobot.fleet.stop._supervisor` refuses to name at all — it is in the
``forbidden`` set — so "none from the supervisor" would be vacuous. Here the
supervisor is ``python -m nanobot fleet start``, found by parentage exactly as an
operator's would be, and it is stopped through the CLI rather than through the
function the CLI calls.

*The descendant is started by the instance, not by the test.* The scripted model
turn calls ``exec`` with ``yield_time_ms``, which is the shell tool's own way of
leaving a command running past the turn that started it
(``agent/tools/exec_session.py``). That command is spawned with
``start_new_session``, so it is out of the instance's process group before it
does anything — which is the whole reason this criterion exists. A stop that
signalled process groups alone, or that terminated only its direct children the
way ``tests/webui/test_gateway_webui_smoke.py``'s ``_stop_gateway`` helper does,
leaves it running. Every link of the chain is asserted as a premise before the
stop, so a future change to how the shell tool spawns cannot quietly turn this
into a test of nothing.

*The chain is two deep, and the second link is the one that matters.* Mutation
testing found that a one-deep descendant proves very little here: on the way
down, a SIGTERM'd instance kills its own exec sessions by process group, so the
command dies whatever ``fleet stop`` does or does not do about trees. Versions
of this test with a single descendant passed with group-only signalling, with
the descendant closure never walked, and with remembered members forgotten
between rounds. So the command starts one further process with
``start_new_session`` of its own and *that* is what the criterion is asserted
about: it is outside the instance's group, outside the command's group, and
ignores SIGTERM — so it survives the instance's own cleanup and the stop's first
round, and is orphaned by its parent's death before the second. By then it is
reachable by no group and by no walk downwards from the instance, and only what
the stop remembered about it can still find it. All three of those mutations are
caught in this shape.

*A zombie is not a remaining process.* The supervisor is this test's own child,
so between its death and the moment pytest reaps it, ``ps`` still lists it in
state ``Z``. That is an artifact of the test being its parent — an operator's
shell reaps its own job — and it is not a process in any sense the criterion is
about: it holds no memory, no file handles, and cannot run. The sweep classifies
by the state column and says so, which is the same posture
:func:`~nanobot.process_runtime.process_is_running` takes; a test below pins that
classification against a real zombie rather than trusting it.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from stub_llm_server import StubCompletion, StubLLMServer, free_port

from nanobot.fleet.instance import SANDBOX_EXEC
from nanobot.fleet.memory import process_parent_pid
from nanobot.fleet.state import FleetStateError, InstanceRecord, read_fleet_state
from nanobot.process_runtime import process_is_running

#: The process table is read through the real tool, because the criterion says
#: ``ps`` and what an operator would see is the point.
PS = "/bin/ps"

confinement_available = pytest.mark.skipif(
    sys.platform != "darwin"
    or not Path(SANDBOX_EXEC).is_file()
    or not Path(PS).is_file(),
    reason="the criterion needs native Seatbelt and a POSIX ps",
)

has_ps = pytest.mark.skipif(
    os.name != "posix" or not Path(PS).is_file(),
    reason="the criterion is asserted through a POSIX ps",
)

CAP_MB = 512

STARTUP_TIMEOUT_SECONDS = 90.0
TURN_TIMEOUT_SECONDS = 120.0
COMMAND_TIMEOUT_SECONDS = 120.0
APPEAR_TIMEOUT_SECONDS = 30.0

#: How long the scripted ``exec`` call waits before handing the turn back while
#: leaving its command running. Short on purpose: the descendant announces itself
#: on stdout, so the poll returns as soon as that arrives rather than on a timer.
YIELD_TIME_MS = 500

#: Written by the descendant to stdout, which is what the shell tool hands back
#: to the model — so finding it in a captured request is proof the process was
#: started *by the instance's own tool* and not by this test.
LINGERING = "lingering-9c41ab"

#: What the stub answers for every turn that is not the scripted tool call.
FALLBACK_ANSWER = "the command is running"

# Run by alpha's shell tool as an exec session, so it outlives the turn that
# started it. It starts one further process of its own, exactly as a command an
# agent runs commonly does, and announces both before going quiet.
DESCENDANT = """
import json
import os
import subprocess
import sys
import time

marker, ready, script, announce = sys.argv[1:5]

# ``start_new_session`` puts this one outside *this* process's group as well as
# outside the instance's, which is what makes it survive everything the instance
# itself does on the way down.
child = subprocess.Popen([sys.executable, script, ready], start_new_session=True)

pending = marker + ".partial"
with open(pending, "w") as handle:
    json.dump(
        {"pid": os.getpid(), "pgid": os.getpgid(0), "child": child.pid}, handle
    )
    handle.flush()
    os.fsync(handle.fileno())
os.replace(pending, marker)

sys.stdout.write(announce + "\\n")
sys.stdout.flush()
time.sleep(600)
"""

# The grandchild, and the process this criterion is really about. It is in its
# own process group, so no group signal aimed at the instance or at its shell
# tool's command reaches it; and it ignores SIGTERM, so it survives the stop's
# first round and is orphaned by its parent's death before the second. By then
# it is reachable neither by any group nor by any walk downwards from the
# instance, and only what the stop remembered about it can still find it.
ORPHAN = """
import os
import pathlib
import signal
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(sys.argv[1]).write_text(str(os.getpgid(0)), encoding="utf-8")
time.sleep(600)
"""


# ----------------------------------------------------------------------------
# The process table
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Process:
    """One line of the process table: pid, process group, state and argv."""

    pid: int
    pgid: int
    stat: str
    args: str

    @property
    def zombie(self) -> bool:
        """Whether this is an exited process nobody has reaped yet.

        A zombie is not a process the criterion is about: it runs nothing and
        holds nothing. ``process_is_running`` draws the same line.
        """
        return self.stat.startswith("Z")


def parse_processes(text: str) -> tuple[Process, ...]:
    """Parse ``pid pgid stat args`` lines, dropping anything that is not one.

    A line whose first two fields are not numbers is dropped rather than guessed
    at — inventing a process in a sweep whose job is to find processes would be
    the more dangerous mistake, but so would inventing a *group*, which is why
    both numeric fields have to parse.
    """
    found: list[Process] = []
    for line in text.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) < 4:
            continue
        pid, pgid, stat, args = fields
        if not pid.isdigit() or not pgid.isdigit() or not args.strip():
            continue
        found.append(
            Process(pid=int(pid), pgid=int(pgid), stat=stat, args=args.strip())
        )
    return tuple(found)


def process_table() -> tuple[Process, ...]:
    """Every process on this host, with its group, its state and its full argv.

    ``-ww`` matters: without it macOS truncates the argv to the terminal width,
    and an instance's ``--config`` path is at the far end of a long command line.
    """
    result = subprocess.run(
        [PS, "-eww", "-o", "pid=,pgid=,stat=,args="],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=True,
    )
    return parse_processes(result.stdout)


def remaining(
    table: tuple[Process, ...],
    *,
    pids: set[int],
    groups: set[int],
) -> tuple[Process, ...]:
    """The live processes belonging to any of *pids* or to any of *groups*.

    Both halves are needed and neither implies the other: a pid answers "is this
    instance gone", a group answers "is anything this instance started gone",
    and the criterion asks both.
    """
    return tuple(
        process
        for process in table
        if not process.zombie
        and (process.pid in pids or process.pgid in groups)
    )


def describe(processes: tuple[Process, ...]) -> str:
    """Survivors as lines fit for a failure message."""
    return "\n".join(
        f"  pid={one.pid} pgid={one.pgid} stat={one.stat} {one.args}"
        for one in processes
    )


# ----------------------------------------------------------------------------
# The fleet
# ----------------------------------------------------------------------------


def write_instance(root: Path, name: str, stub: StubLLMServer) -> Path:
    """Lay out one instance's config the way nanobot lays one out by itself.

    ``<root>/<name>/config.json`` with ``<root>/<name>/workspace`` beneath it.
    The workspace is not created here: creating it belongs to ``prepare_fleet``,
    which the ``fleet start`` subprocess runs for itself.
    """
    config_dir = root / name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps(stub.instance_config(config_dir / "workspace", api_port=free_port())),
        encoding="utf-8",
    )
    return config_path


def write_fleet(root: Path, configs: dict[str, Path]) -> Path:
    """A fleet document naming *configs*, in the order given."""
    fleet_path = root / "fleet.json"
    fleet_path.write_text(
        json.dumps({
            "instances": {
                name: {
                    "config": str(path),
                    "mode": "serve",
                    "memoryLimitMb": CAP_MB,
                }
                for name, path in configs.items()
            }
        }),
        encoding="utf-8",
    )
    return fleet_path


def api_port_of(config_path: Path) -> int:
    """The port the instance will bind, read back from its own config."""
    return int(json.loads(config_path.read_text(encoding="utf-8"))["api"]["port"])


def log_tail(path: Path, limit: int = 4000) -> str:
    """A log's tail, which for a confined instance is the only diagnostic there is."""
    if not path.exists():
        return f"(no log at {path})"
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def child_environment(**overrides: str) -> dict[str, str]:
    """This process's environment, cleaned of everything a child must not inherit.

    ``NANOBOT_*`` is dropped because :class:`nanobot.config.Config` reads those,
    so a developer with one exported could make a child behave differently from a
    clean machine. ``COV_CORE_*`` and ``COVERAGE_*`` are dropped because
    ``pytest-cov`` asks every child process to start measuring — and a child
    started outside the repository finds no ``pyproject.toml``, so it measures
    with none of the configured ``omit`` rules and its data quietly changes the
    whole suite's coverage denominator. Nothing spawned here is part of the
    measurement.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NANOBOT_", "COV_CORE_", "COVERAGE_"))
    }
    return env | overrides


def wait_for_file(path: Path, what: str, *, log_path: Path) -> None:
    """Block until *path* exists, or fail with whatever log might explain it.

    There is no ``pytest-timeout`` in this repo, so every wait carries its own
    deadline — the pattern ``tests/webui/test_gateway_webui_smoke.py`` sets.
    """
    deadline = time.monotonic() + APPEAR_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    pytest.fail(f"{what} never appeared at {path}\n{log_tail(log_path)}")


def wait_for_health(port: int, name: str, log_path: Path) -> None:
    """Block until the instance answers on its own API, or fail with its log."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        with suppress(httpx.HTTPError, OSError):
            response = httpx.get(
                f"http://127.0.0.1:{port}/health", timeout=5.0, trust_env=False
            )
            if response.status_code == 200:
                return
        time.sleep(0.2)
    pytest.fail(f"instance {name} never became healthy\n{log_tail(log_path)}")


def wait_for_running_instances(
    state_path: Path,
    names: tuple[str, ...],
    *,
    log_path: Path,
) -> dict[str, InstanceRecord]:
    """Block until the supervisor's state file reports every instance running.

    Read through :func:`~nanobot.fleet.state.read_fleet_state`, which is the
    reader an operator's second shell uses — and the one ``fleet stop`` itself
    uses. The file is written atomically, so the only transient state to wait
    through is its absence.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        with suppress(FleetStateError):
            found = {record.name: record for record in read_fleet_state(state_path)}
            if set(found) == set(names) and all(
                record.state == "running" for record in found.values()
            ):
                return found
        time.sleep(0.05)
    pytest.fail(f"{state_path} never reported {names} running\n{log_tail(log_path)}")


def normalised(result: subprocess.CompletedProcess[str]) -> str:
    """A command's whole output with runs of whitespace collapsed.

    Refusals and confirmations are rendered by ``rich``, which wraps to the
    terminal width, so an unnormalised assertion on a message would be a test of
    the terminal.
    """
    return " ".join(f"{result.stdout}\n{result.stderr}".split())


def kill_tree(pid: int | None) -> None:
    """Best-effort SIGKILL, so a failing test leaves nothing sleeping behind."""
    if not pid or pid <= 0:
        return
    with suppress(OSError):
        os.kill(pid, signal.SIGKILL)


# ----------------------------------------------------------------------------
# The criterion
# ----------------------------------------------------------------------------


@confinement_available
def test_stop_leaves_no_instance_descendant_or_supervisor_running(
    tmp_path: Path,
    stub_llm_server: StubLLMServer,
) -> None:
    """Criterion 8, literally: through both commands, asserted through ``ps``."""
    pytest.importorskip("aiohttp")

    root = tmp_path.resolve()
    home = root / "home"
    home.mkdir()
    alpha_config = write_instance(root, "alpha", stub_llm_server)
    beta_config = write_instance(root, "beta", stub_llm_server)
    fleet_path = write_fleet(root, {"alpha": alpha_config, "beta": beta_config})
    state_path = root / "fleet.json.state.json"
    alpha_log = root / "alpha" / "logs" / "fleet.log"
    beta_log = root / "beta" / "logs" / "fleet.log"

    # Outside every instance's workspace, so neither instance's Seatbelt profile
    # has an opinion about them; the markers they write go into alpha's own
    # workspace, which is the one place alpha may write.
    descendant_script = root / "linger.py"
    descendant_script.write_text(DESCENDANT, encoding="utf-8")
    orphan_script = root / "orphan.py"
    orphan_script.write_text(ORPHAN, encoding="utf-8")
    marker = root / "alpha" / "workspace" / "descendant.json"
    orphan_marker = root / "alpha" / "workspace" / "orphan.txt"

    supervisor_out = root / "supervisor.out"
    #: Every pid this test learns about, whether or not it got as far as using
    #: it. A stop that failed to kill something must not leave this suite with a
    #: confined instance still holding a port.
    started: list[int] = []
    with supervisor_out.open("wb") as sink:
        supervisor = subprocess.Popen(
            [
                sys.executable, "-m", "nanobot",
                "fleet", "start", "--fleet", str(fleet_path),
            ],
            stdout=sink,
            stderr=subprocess.STDOUT,
            cwd=root,
            env=child_environment(HOME=str(home), COLUMNS="200"),
            # Its own session, so that the group swept below is the supervisor's
            # and not this test runner's. Without it ``os.getpgid(supervisor)``
            # is pytest's own group and the sweep would report the whole test
            # session as fleet survivors.
            start_new_session=True,
        )
    started.append(supervisor.pid)
    try:
        records = wait_for_running_instances(
            state_path, ("alpha", "beta"), log_path=supervisor_out
        )
        wait_for_health(api_port_of(alpha_config), "alpha", alpha_log)
        wait_for_health(api_port_of(beta_config), "beta", beta_log)

        alpha_pid, beta_pid = records["alpha"].pid, records["beta"].pid
        started.extend((alpha_pid, beta_pid))
        alpha_pgid, beta_pgid = os.getpgid(alpha_pid), os.getpgid(beta_pid)

        # One scripted tool call is the whole trigger: the model tells alpha to
        # run a command with ``yield_time_ms``, and alpha's own shell tool starts
        # a process that outlives the turn. Everything else the stub is asked
        # answers with the fallback, so nothing else can consume this entry.
        stub_llm_server.set_fallback(StubCompletion(content=FALLBACK_ANSWER))
        stub_llm_server.script_tool_call("exec", {
            "command": " ".join(shlex.quote(word) for word in (
                sys.executable,
                str(descendant_script),
                str(marker),
                str(orphan_marker),
                str(orphan_script),
                LINGERING,
            )),
            "yield_time_ms": YIELD_TIME_MS,
        })
        answer = httpx.post(
            f"http://127.0.0.1:{api_port_of(alpha_config)}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "start the command"}]},
            timeout=TURN_TIMEOUT_SECONDS,
            trust_env=False,
        )
        assert answer.status_code == 200, f"{answer.text}\n{log_tail(alpha_log)}"

        wait_for_file(marker, "alpha's long-running child", log_path=alpha_log)
        announced = json.loads(marker.read_text(encoding="utf-8"))
        descendant_pid, descendant_pgid = int(announced["pid"]), int(announced["pgid"])
        orphan_pid = int(announced["child"])
        started.extend((descendant_pid, orphan_pid))
        wait_for_file(orphan_marker, "the grandchild", log_path=alpha_log)
        orphan_pgid = int(orphan_marker.read_text(encoding="utf-8"))

        # The premises, checked rather than assumed. Each one is a way this test
        # could otherwise pass while testing nothing.
        #
        # It was started through the instance's shell tool: its stdout came back
        # to the model as a tool result.
        tool_output = "".join(
            json.dumps(request.body) for request in stub_llm_server.requests
        )
        assert LINGERING in tool_output, "the descendant's output never reached the model"
        # The chain is instance -> command -> grandchild, and every link is in a
        # process group of its own. That is the entire reason this criterion is
        # not satisfied by a signal aimed at the instance's group, and not
        # satisfied either by whatever the instance does to its own commands on
        # the way down: the grandchild is outside both.
        assert process_parent_pid(descendant_pid) == alpha_pid
        assert process_parent_pid(orphan_pid) == descendant_pid
        assert len({alpha_pgid, beta_pgid, descendant_pgid, orphan_pgid}) == 4
        assert len({alpha_pid, beta_pid, descendant_pid, orphan_pid, supervisor.pid}) == 5
        # Nothing being swept is this test runner's own group: an instance whose
        # recorded group were pytest's would turn the assertions below into a
        # report that the whole session had survived the stop.
        own_group = os.getpgid(0)
        groups = {
            alpha_pgid, beta_pgid, descendant_pgid, orphan_pgid, supervisor.pid
        }
        assert own_group not in groups, groups

        # Everything the stop has to account for is running right now, so
        # "nothing is left afterwards" cannot be true by default.
        pids = {alpha_pid, beta_pid, descendant_pid, orphan_pid, supervisor.pid}
        before = process_table()
        assert {one.pid for one in remaining(before, pids=pids, groups=set())} == pids

        stopped = subprocess.run(
            [
                sys.executable, "-m", "nanobot",
                "fleet", "stop", "--fleet", str(fleet_path),
            ],
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            cwd=root,
            env=child_environment(HOME=str(home), COLUMNS="200"),
        )
        # Read with nothing waited for in between: the contract is that the
        # command does not return until the fleet is gone, so any settling time
        # this test allowed itself would be testing a weaker claim.
        after = process_table()
        survivors = remaining(after, pids=pids, groups=groups)
    finally:
        for pid in started:
            kill_tree(pid)
        with suppress(subprocess.TimeoutExpired):
            supervisor.wait(timeout=STARTUP_TIMEOUT_SECONDS)

    output = normalised(stopped)
    assert stopped.returncode == 0, f"{output}\n{log_tail(supervisor_out)}"
    assert "Stopped alpha" in output, output
    assert "Stopped beta" in output, output
    assert f"Stopped the supervisor (pid {supervisor.pid})" in output, output

    # The criterion. One assertion covering the instances, their groups, the
    # descendant and its group, and the supervisor — named individually below so
    # a failure says which of them survived rather than only that one did.
    assert survivors == (), f"the fleet outlived its stop:\n{describe(survivors)}"
    live_pids = {one.pid for one in after if not one.zombie}
    live_groups = {one.pgid for one in after if not one.zombie}
    for name, pid, group in (
        ("alpha", alpha_pid, alpha_pgid),
        ("beta", beta_pid, beta_pgid),
        ("alpha's shell-tool command", descendant_pid, descendant_pgid),
        ("the grandchild it started", orphan_pid, orphan_pgid),
        ("the supervisor", supervisor.pid, supervisor.pid),
    ):
        assert pid not in live_pids, name
        assert group not in live_groups, name

    # The same claim through the helper the fleet itself uses, which reads each
    # pid directly rather than through a table: two independent readings of the
    # kernel have to agree before this criterion is called met.
    for pid in sorted(pids):
        assert not process_is_running(pid), pid

    # The state file the operator's next command would read agrees: nothing here
    # is running, and the stop is not merely invisible to ``ps``.
    assert [record.state for record in read_fleet_state(state_path)] == [
        "exited",
        "exited",
    ]


# ----------------------------------------------------------------------------
# The sweep's own sensitivity
# ----------------------------------------------------------------------------


@has_ps
def test_the_sweep_finds_a_process_by_its_own_process_group(tmp_path: Path) -> None:
    """A decoy in its own session is found by group as well as by pid.

    The half of the sweep the criterion leans on hardest: a descendant is found
    through its group, not through an argv this test could have recognised.
    """
    marker = tmp_path / "ready"
    decoy = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib, sys, time\n"
            "pathlib.Path(sys.argv[1]).write_text('ready')\n"
            "time.sleep(30)\n",
            str(marker),
        ],
        env=child_environment(),
        start_new_session=True,
    )
    try:
        wait_for_file(marker, "the decoy", log_path=marker)
        table = process_table()
        found = remaining(table, pids=set(), groups={decoy.pid})
    finally:
        kill_tree(decoy.pid)
        with suppress(subprocess.TimeoutExpired):
            decoy.wait(timeout=APPEAR_TIMEOUT_SECONDS)

    assert [one.pid for one in found] == [decoy.pid], found
    assert found[0].pgid == decoy.pid


@has_ps
def test_a_zombie_is_not_counted_as_a_remaining_process(tmp_path: Path) -> None:
    """An exited-but-unreaped process is listed by ``ps`` and is not a survivor.

    Pinned against a real zombie rather than a hand-written ``ps`` line, because
    this is the one exception the criterion's assertion makes and the whole
    question is whether the state column really says what it is assumed to say.
    The supervisor above becomes exactly this between its death and the moment
    pytest reaps it.
    """
    marker = tmp_path / "zombie.json"
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import json, pathlib, subprocess, sys, time\n"
            # Never waited on, so it stays a zombie for as long as this lives.
            "child = subprocess.Popen([sys.executable, '-c', ''])\n"
            "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid': child.pid}))\n"
            "time.sleep(30)\n",
            str(marker),
        ],
        env=child_environment(),
        start_new_session=True,
    )
    try:
        wait_for_file(marker, "the zombie's parent", log_path=marker)
        zombie_pid = int(json.loads(marker.read_text(encoding="utf-8"))["pid"])
        listed = None
        deadline = time.monotonic() + APPEAR_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            found = [one for one in process_table() if one.pid == zombie_pid]
            if found and found[0].zombie:
                listed = found[0]
                break
            time.sleep(0.05)
    finally:
        kill_tree(parent.pid)
        with suppress(subprocess.TimeoutExpired):
            parent.wait(timeout=APPEAR_TIMEOUT_SECONDS)

    assert listed is not None, "no zombie was ever observed, so this proves nothing"
    # ``ps`` lists it; the sweep does not count it; and the fleet's own liveness
    # helper agrees, which is the line this test exists to keep the two on.
    assert remaining((listed,), pids={zombie_pid}, groups={listed.pgid}) == ()
    assert not process_is_running(zombie_pid)


@has_ps
def test_this_host_really_reports_processes() -> None:
    """``ps`` answers here, so an empty sweep means empty and not unavailable."""
    table = process_table()

    assert len(table) > 1
    mine = [one for one in table if one.pid == os.getpid()]
    assert len(mine) == 1
    assert mine[0].pgid == os.getpgid(0)


# ----------------------------------------------------------------------------
# The sweep's machinery
# ----------------------------------------------------------------------------


def test_the_process_table_is_parsed_into_pid_group_state_and_arguments() -> None:
    parsed = parse_processes(
        "    1     1 Ss   /sbin/launchd\n"
        "  902   900 S+   python -m nanobot serve --config /tmp/c.json\n"
    )

    assert parsed == (
        Process(pid=1, pgid=1, stat="Ss", args="/sbin/launchd"),
        Process(
            pid=902,
            pgid=900,
            stat="S+",
            args="python -m nanobot serve --config /tmp/c.json",
        ),
    )


def test_a_line_that_is_not_a_process_is_dropped_rather_than_guessed_at() -> None:
    """A header, a blank line or a wrapped argv must not become a fake process.

    A line whose *group* does not parse is dropped too: a sweep that invented a
    group would answer "is anything this instance started still running" with a
    number it made up.
    """
    assert parse_processes(
        "  PID  PGID STAT ARGS\n\n   \n-1 -1 S x\n123\n7 seven S sleep\n"
    ) == ()


def test_a_process_is_a_survivor_by_its_pid_or_by_its_group() -> None:
    """Either half is enough, which is what makes the two together the criterion."""
    table = (
        Process(pid=10, pgid=10, stat="S", args="the instance"),
        Process(pid=20, pgid=20, stat="S", args="its descendant"),
        Process(pid=30, pgid=10, stat="S", args="something in the instance's group"),
        Process(pid=40, pgid=40, stat="S", args="unrelated"),
    )

    assert {one.pid for one in remaining(table, pids={20}, groups=set())} == {20}
    assert {one.pid for one in remaining(table, pids=set(), groups={10})} == {10, 30}
    assert {one.pid for one in remaining(table, pids={20}, groups={10})} == {10, 20, 30}


def test_an_unrelated_process_is_never_a_survivor() -> None:
    table = (Process(pid=40, pgid=40, stat="S", args="/usr/bin/vim notes.md"),)

    assert remaining(table, pids={10, 20}, groups={10, 20}) == ()


def test_a_zombie_matching_by_pid_or_by_group_is_still_not_a_survivor() -> None:
    """The exception is stated once and applies to both halves of the sweep."""
    table = (Process(pid=10, pgid=10, stat="Z+", args="(python)"),)

    assert remaining(table, pids={10}, groups=set()) == ()
    assert remaining(table, pids=set(), groups={10}) == ()


def test_a_live_process_is_not_mistaken_for_a_zombie() -> None:
    """``S`` and ``Z`` are not the only states, and only one of them is excluded."""
    for stat in ("S", "Ss", "S+", "R", "SN", "U", "I"):
        assert not Process(pid=10, pgid=10, stat=stat, args="x").zombie, stat
    for stat in ("Z", "Z+"):
        assert Process(pid=10, pgid=10, stat=stat, args="x").zombie, stat


def test_survivors_are_described_with_everything_needed_to_find_them() -> None:
    """A failure has to name the process well enough to go and look at it."""
    described = describe((Process(pid=10, pgid=9, stat="S", args="sleep 300"),))

    assert described == "  pid=10 pgid=9 stat=S sleep 300"

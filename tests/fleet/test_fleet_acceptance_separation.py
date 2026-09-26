"""Acceptance criteria 2 and 3: separate processes, and one death that is local.

Both criteria are asked the way they are worded, from outside a real fleet. A
real ``python -m nanobot fleet start`` holds the foreground of its own process;
everything asserted about it is read either through the real
``nanobot fleet status --json`` run from a *second* shell, through ``/bin/ps``, or
through an instance's own HTTP API. Nothing is read out of an in-process
supervisor object, because "these are separate OS processes" is precisely the
claim an in-process test cannot make.

**Criterion 2** — two instances, both reported ``running``, each a distinct live
process, and neither one the supervisor. Three things keep that from being
satisfied trivially.

*The supervisor's pid is a number genuinely in play.* Asserting "no instance pid
equals the supervisor's" proves nothing on its own — the supervisor's pid is
never recorded anywhere, so a status command could hardly publish it by accident.
What makes the absence meaningful is that both instances are first shown to be
the supervisor's own children, by parentage read from the kernel, and that the
supervisor's whole child set is exactly the two instances. The number is
therefore one step away from every record, and is then shown to appear in no
field of either object rather than merely not in the field named ``pid``.

*Separate means separate to the kernel, not to the reader.* Two records with
different pids could still describe one process observed twice. So each pid is
confirmed live two independent ways — through ``ps`` and through
:func:`~nanobot.process_runtime.process_is_running`, which reads each pid
directly — and the two are shown to lead *different process groups*, which is the
kernel's own account of their being distinct trees rather than a number this test
compared.

*Running means serving.* An instance can hold a pid without being usable, so both
are required to answer on their own API port before anything else is asserted.

**Criterion 3** — after ``kill -9`` of A, B still serves, B is the same process,
and A stays dead. The load-bearing decisions here:

*The kill is the criterion's own tool.* ``/bin/kill -9``, not
:func:`os.kill`, and aimed at the pid ``status --json`` published rather than at
one this test remembered from spawning — so what is killed is what an operator
reading the status output would kill.

*"B answers" is a round trip, not a status code.* The request goes to B's
``/v1/chat/completions``, and the assertion is both that B answered with the
model's words and that B's request reached the stub carrying this test's own
prompt. A process wedged by its peer's death could still complete a TCP
handshake; it could not drive a turn through the model and back.

*"B's pid is unchanged" is checked as identity, not as a number.* The whole record
is compared field by field against the one taken before the kill, and B's process
group is re-read from the kernel: a pid that had been recycled onto a different
process would be a different group.

*"Not restarted" is given time to be false.* A supervisor that replaced a dead
instance would not do it instantly, so A's death is followed by a settling window
of several of the supervisor's own sweeps before the claim is made. The evidence
is then threefold and none of it depends on an argv: A's record still names the
same dead pid, the supervisor's live child set is exactly ``{B}``, and nothing has
rebound A's API port. An argv sweep would be the natural fourth and is
deliberately absent — ``set_cli_process_identity`` renames a ``serve`` instance to
the bare word ``nanobot`` and *replaces* its argv on this platform, so the
``--config`` path that would identify a replacement is gone from the process
table before the sweep could find it.

There is no ``pytest-timeout`` in this repo, so every wait below carries its own
:func:`time.monotonic` deadline and fails with the instance log inlined — the
pattern ``tests/webui/test_gateway_webui_smoke.py`` sets.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from stub_llm_server import StubCompletion, StubLLMServer, free_port

from nanobot.fleet.instance import SANDBOX_EXEC
from nanobot.fleet.memory import process_parent_pid
from nanobot.fleet.state import (
    RECORD_FIELDS,
    FleetStateError,
    InstanceRecord,
    read_fleet_state,
)
from nanobot.process_runtime import process_is_running

#: Read through the real tool: the criteria name ``ps`` and ``kill``, and what an
#: operator would see and do is the point.
PS = "/bin/ps"
KILL = "/bin/kill"

confinement_available = pytest.mark.skipif(
    sys.platform != "darwin"
    or not Path(SANDBOX_EXEC).is_file()
    or not Path(PS).is_file()
    or not Path(KILL).is_file(),
    reason="the criteria need native Seatbelt, a POSIX ps and a POSIX kill",
)

has_ps = pytest.mark.skipif(
    os.name != "posix" or not Path(PS).is_file(),
    reason="the sweep is asserted through a POSIX ps",
)

#: Comfortably above the ~122 MB a serve instance's tree occupies, so nothing
#: here can be explained by the memory cap firing.
CAP_MB = 512

STARTUP_TIMEOUT_SECONDS = 90.0
TURN_TIMEOUT_SECONDS = 120.0
COMMAND_TIMEOUT_SECONDS = 120.0
DEATH_TIMEOUT_SECONDS = 30.0

#: Long enough for several of the supervisor's own sweeps (its poll interval is
#: half a second) to pass with A dead. A replacement would have to appear inside
#: this window for the "not restarted" claim below to be the one being tested.
RESTART_WINDOW_SECONDS = 3.0

#: How wide the second shell's terminal is when it asks for JSON. Narrow on
#: purpose: the published document must not depend on it.
STATUS_COLUMNS = "40"

#: What the stub answers every turn. No script queue is used at all here — the
#: queue is shared by the whole fleet, and these criteria need no tool call.
FALLBACK_ANSWER = "beta is still answering"


# ----------------------------------------------------------------------------
# The process table
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Process:
    """One line of the process table: pid, parent, group, state and argv."""

    pid: int
    ppid: int
    pgid: int
    stat: str
    args: str

    @property
    def zombie(self) -> bool:
        """Whether this is an exited process nobody has reaped yet.

        A zombie is not a process either criterion is about: it runs nothing and
        holds nothing. :func:`~nanobot.process_runtime.process_is_running` draws
        the same line, and a killed instance is exactly this for the moment
        between its death and the supervisor reaping it.
        """
        return self.stat.startswith("Z")


def parse_processes(text: str) -> tuple[Process, ...]:
    """Parse ``pid ppid pgid stat args`` lines, dropping anything that is not one.

    Every numeric field has to parse. A sweep whose job is to say "nothing new
    appeared" must not invent a *parent* any more than it invents a pid: a made-up
    ppid would answer the question about restarts with a number nobody read.
    """
    found: list[Process] = []
    for line in text.splitlines():
        fields = line.split(maxsplit=4)
        if len(fields) < 5:
            continue
        pid, ppid, pgid, stat, args = fields
        if not (pid.isdigit() and ppid.isdigit() and pgid.isdigit()):
            continue
        if not args.strip():
            continue
        found.append(
            Process(
                pid=int(pid),
                ppid=int(ppid),
                pgid=int(pgid),
                stat=stat,
                args=args.strip(),
            )
        )
    return tuple(found)


def process_table() -> tuple[Process, ...]:
    """Every process on this host, with its parent, group, state and full argv.

    ``-ww`` matters: without it macOS truncates the argv to the terminal width.
    """
    result = subprocess.run(
        [PS, "-eww", "-o", "pid=,ppid=,pgid=,stat=,args="],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=True,
    )
    return parse_processes(result.stdout)


def children_of(table: tuple[Process, ...], pid: int) -> set[int]:
    """The live processes the kernel says *pid* is the parent of.

    The one reading that answers "was a replacement started" without depending on
    an argv — which is unavailable here, because an instance renames itself and
    loses its ``--config`` argument on the way up.
    """
    return {one.pid for one in table if one.ppid == pid and not one.zombie}


def live_pids(table: tuple[Process, ...]) -> set[int]:
    """Every pid on the host that is not an unreaped corpse."""
    return {one.pid for one in table if not one.zombie}


def describe(processes: tuple[Process, ...]) -> str:
    """Processes as lines fit for a failure message."""
    return "\n".join(
        f"  pid={one.pid} ppid={one.ppid} pgid={one.pgid} stat={one.stat} {one.args}"
        for one in processes
    )


# ----------------------------------------------------------------------------
# The fleet on disk
# ----------------------------------------------------------------------------


def write_instance(root: Path, name: str, stub: StubLLMServer) -> Path:
    """Lay out one instance's config the way nanobot lays one out by itself.

    ``<root>/<name>/config.json`` with ``<root>/<name>/workspace`` beneath it.
    The workspace is deliberately not created: creating it belongs to
    ``prepare_fleet``, which the ``fleet start`` subprocess runs for itself.
    """
    config_dir = root / name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps(
            stub.instance_config(config_dir / "workspace", api_port=free_port())
        ),
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


# ----------------------------------------------------------------------------
# Waiting, always against a deadline
# ----------------------------------------------------------------------------


def health(port: int, *, timeout: float = 5.0) -> bool:
    """Whether something answers ``/health`` on *port* right now.

    Used in both directions — an instance that must be serving, and a port that
    must have nothing on it — so a probe stuck on one answer would be caught by
    the other.
    """
    with suppress(httpx.HTTPError, OSError):
        response = httpx.get(
            f"http://127.0.0.1:{port}/health", timeout=timeout, trust_env=False
        )
        return response.status_code == 200
    return False


def wait_for_health(port: int, name: str, log_path: Path) -> None:
    """Block until the instance answers on its own API, or fail with its log."""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if health(port):
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
    reader an operator's second shell uses. The file is written atomically, so
    the only transient state to wait through is its absence.
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


def wait_until_gone(pid: int, what: str, *, log_path: Path) -> None:
    """Block until *pid* is no longer a running process, or fail."""
    deadline = time.monotonic() + DEATH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not process_is_running(pid):
            return
        time.sleep(0.05)
    pytest.fail(f"{what} (pid {pid}) never died\n{log_tail(log_path)}")


def wait_for_exit_reason(
    state_path: Path,
    name: str,
    reason: str,
    *,
    log_path: Path,
) -> None:
    """Block until the supervisor has published *why* *name* stopped.

    Waited for separately from the death itself, and that separation is required
    rather than tidy: :func:`read_fleet_state` reconciles liveness as it reads, so
    an instance reports ``exited`` from the instant its process is gone — before
    the supervisor has reaped the child and recorded a reason. Asserting the
    reason off the back of the state change would race the supervisor's own write.
    """
    deadline = time.monotonic() + DEATH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        with suppress(FleetStateError):
            found = {record.name: record for record in read_fleet_state(state_path)}
            if found.get(name) is not None and found[name].exit_reason == reason:
                return
        time.sleep(0.05)
    pytest.fail(
        f"{name} was never recorded as having exited by {reason}\n{log_tail(log_path)}"
    )


# ----------------------------------------------------------------------------
# The contact points
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Fleet:
    """A real fleet running under a real supervisor in its own process."""

    root: Path
    path: Path
    state_path: Path
    home: Path
    shell: Path
    supervisor: subprocess.Popen[bytes]
    supervisor_log: Path
    configs: dict[str, Path]
    logs: dict[str, Path]
    #: Every instance pid learned at startup, so teardown can reach them even
    #: after a test has killed the supervisor or failed halfway through.
    pids: dict[str, int] = field(default_factory=dict)

    def port(self, name: str) -> int:
        return api_port_of(self.configs[name])


def status_json(fleet: Fleet) -> list[dict[str, object]]:
    """Run the real ``nanobot fleet status --json`` in a second shell.

    A separate process, a working directory outside the fleet, and a deliberately
    narrow terminal: a supervisor holds the foreground of the shell it was started
    in, so the whole point of fleet state being a file is that this command works
    from somewhere else entirely and depends on nothing about where it is run.
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "nanobot",
            "fleet", "status", "--fleet", str(fleet.path), "--json",
        ],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        cwd=fleet.shell,
        env=child_environment(HOME=str(fleet.home), COLUMNS=STATUS_COLUMNS),
    )
    assert result.returncode == 0, (
        f"fleet status refused\n{result.stdout}\n{result.stderr}"
    )
    reported = json.loads(result.stdout)
    assert isinstance(reported, list), reported
    return reported


def kill_nine(pid: int) -> None:
    """``kill -9`` through the tool the criterion names, not through :mod:`os`."""
    result = subprocess.run(
        [KILL, "-9", str(pid)],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, f"kill -9 {pid}: {result.stdout}{result.stderr}"


def stop_tree(pid: int | None) -> None:
    """Best-effort SIGKILL of *pid*'s process group, never of this test's own.

    The guard is not defensive padding. Every process this file starts is put in
    a session of its own, so ``os.getpgid(pid)`` is that process's group — but if
    a future change dropped ``start_new_session`` anywhere, the group would be
    pytest's and this cleanup would kill the test runner and everything above it.
    That mutation presents as a silent runner death with an empty log rather than
    as a failing assertion, which is why the check lives here rather than in a
    comment.
    """
    if not pid or pid <= 0:
        return
    own_group = os.getpgid(0)
    group: int | None = None
    with suppress(OSError):
        group = os.getpgid(pid)
    if group is not None and group != own_group:
        with suppress(OSError):
            os.killpg(group, signal.SIGKILL)
        return
    with suppress(OSError):
        os.kill(pid, signal.SIGKILL)


@contextmanager
def running_fleet(
    root: Path,
    stub: StubLLMServer,
    names: tuple[str, ...] = ("alpha", "beta"),
) -> Iterator[Fleet]:
    """Start a real two-instance fleet, healthy and serving, and tear it down.

    The supervisor is ``python -m nanobot fleet start`` in its own session. Its
    own session matters for two reasons: the teardown below addresses process
    groups, and a supervisor sharing pytest's group would make every group-shaped
    reading in these tests a reading of the test session.
    """
    home = root / "home"
    home.mkdir()
    # Somewhere with no relationship to the fleet, so the status command below is
    # run the way a second shell would run it.
    shell = root / "shell"
    shell.mkdir()

    configs = {name: write_instance(root, name, stub) for name in names}
    fleet_path = write_fleet(root, configs)
    supervisor_log = root / "supervisor.out"

    stub.set_fallback(StubCompletion(content=FALLBACK_ANSWER))

    with supervisor_log.open("wb") as sink:
        supervisor = subprocess.Popen(
            [
                sys.executable, "-m", "nanobot",
                "fleet", "start", "--fleet", str(fleet_path),
            ],
            stdout=sink,
            stderr=subprocess.STDOUT,
            cwd=root,
            env=child_environment(HOME=str(home), COLUMNS="200"),
            start_new_session=True,
        )
    fleet = Fleet(
        root=root,
        path=fleet_path,
        state_path=root / "fleet.json.state.json",
        home=home,
        shell=shell,
        supervisor=supervisor,
        supervisor_log=supervisor_log,
        configs=configs,
        logs={name: root / name / "logs" / "fleet.log" for name in names},
    )
    try:
        records = wait_for_running_instances(
            fleet.state_path, names, log_path=supervisor_log
        )
        fleet.pids.update({name: records[name].pid for name in names})
        for name in names:
            wait_for_health(fleet.port(name), name, fleet.logs[name])
        yield fleet
    finally:
        # Instances first: each is in its own session, so the supervisor's own
        # group signal would not reach them, and a confined instance left holding
        # a port would outlive this suite.
        for pid in fleet.pids.values():
            stop_tree(pid)
        stop_tree(supervisor.pid)
        with suppress(subprocess.TimeoutExpired):
            supervisor.wait(timeout=STARTUP_TIMEOUT_SECONDS)


# ----------------------------------------------------------------------------
# Criterion 2: separate processes
# ----------------------------------------------------------------------------


@confinement_available
def test_two_running_instances_are_distinct_live_processes_and_not_the_supervisor(
    tmp_path: Path,
    stub_llm_server: StubLLMServer,
) -> None:
    """Criterion 2, literally: through ``fleet status --json`` and through ``ps``."""
    pytest.importorskip("aiohttp")

    # Asserted inside the block, because every reading below is a reading of a
    # live kernel: on the way out the fixture kills the fleet, and a claim made
    # after that would be a claim about corpses.
    with running_fleet(tmp_path.resolve(), stub_llm_server) as fleet:
        reported = status_json(fleet)
        table = process_table()

        # The document's shape is the published contract, so it is checked before
        # anything is read out of it: exactly the seven fields, no more, in every
        # object, and one object per instance in fleet order.
        assert [entry["name"] for entry in reported] == ["alpha", "beta"]
        for entry in reported:
            assert sorted(entry) == sorted(RECORD_FIELDS), entry
            assert entry["state"] == "running", entry
            assert isinstance(entry["pid"], int), entry
            assert entry["exit_reason"] is None, entry
            assert entry["memory_limit_mb"] == CAP_MB, entry

        pids = {str(entry["name"]): int(entry["pid"]) for entry in reported}
        assert len(set(pids.values())) == 2, pids

        # Each is a live process, read two independent ways: out of the table,
        # and pid by pid through the helper the fleet itself uses.
        listed = {one.pid: one for one in table if not one.zombie}
        for name, pid in pids.items():
            assert pid in listed, f"{name} (pid {pid}) is not in the process table"
            assert process_is_running(pid), name
            assert health(fleet.port(name)), (
                f"{name} holds a pid but serves nothing\n{log_tail(fleet.logs[name])}"
            )

        # Distinct to the kernel and not merely to this test: two records with
        # different numbers could describe one process seen twice, but two
        # process groups cannot.
        assert len({listed[pid].pgid for pid in pids.values()}) == 2
        assert len({entry["workspace"] for entry in reported}) == 2
        assert len({entry["config_dir"] for entry in reported}) == 2
        assert len({fleet.port(name) for name in pids}) == 2

        # The supervisor's pid is one step from every record — both instances are
        # its children, and it has no others — which is what makes its absence
        # below a fact about this document rather than about an unreachable
        # number.
        supervisor_pid = fleet.supervisor.pid
        for name, pid in pids.items():
            assert process_parent_pid(pid) == supervisor_pid, name
        assert children_of(table, supervisor_pid) == set(pids.values())
        assert supervisor_pid in listed, "the supervisor died, so it watched nothing"

        # Absent from every field rather than only from ``pid``: a future edit
        # that published it anywhere at all is caught.
        for entry in reported:
            for key, value in entry.items():
                assert value != supervisor_pid, key
                assert str(value) != str(supervisor_pid), key


# ----------------------------------------------------------------------------
# Criterion 3: kill A, B still serves
# ----------------------------------------------------------------------------


@confinement_available
def test_killing_one_instance_leaves_the_other_serving_and_never_restarts_it(
    tmp_path: Path,
    stub_llm_server: StubLLMServer,
) -> None:
    """Criterion 3, literally: ``kill -9`` A, then ask B to answer a request."""
    pytest.importorskip("aiohttp")

    prompt = f"is beta still serving? {uuid.uuid4().hex}"

    with running_fleet(tmp_path.resolve(), stub_llm_server) as fleet:
        before = {str(entry["name"]): entry for entry in status_json(fleet)}
        assert set(before) == {"alpha", "beta"}
        assert [before[name]["state"] for name in ("alpha", "beta")] == [
            "running",
            "running",
        ]
        alpha_pid, beta_pid = int(before["alpha"]["pid"]), int(before["beta"]["pid"])
        beta_group = os.getpgid(beta_pid)
        alpha_port, beta_port = fleet.port("alpha"), fleet.port("beta")

        # Premises. Both are serving right now, so alpha's later silence and
        # beta's later answer are each changes rather than initial conditions.
        assert health(alpha_port), f"alpha was not serving\n{log_tail(fleet.logs['alpha'])}"
        assert health(beta_port), f"beta was not serving\n{log_tail(fleet.logs['beta'])}"

        # The kill, through the criterion's own tool, aimed at the pid the status
        # command published rather than at one this test remembered.
        kill_nine(alpha_pid)
        wait_until_gone(alpha_pid, "alpha", log_path=fleet.logs["alpha"])
        # Separately from the death: see ``wait_for_exit_reason``.
        wait_for_exit_reason(
            fleet.state_path, "alpha", "signal", log_path=fleet.supervisor_log
        )

        # B answers — and answers by driving a turn all the way through the model
        # and back, which a process merely holding a socket could not do.
        answer = httpx.post(
            f"http://127.0.0.1:{beta_port}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}]},
            timeout=TURN_TIMEOUT_SECONDS,
            trust_env=False,
        )
        assert answer.status_code == 200, (
            f"{answer.text}\n{log_tail(fleet.logs['beta'])}"
        )
        assert FALLBACK_ANSWER in answer.json()["choices"][0]["message"]["content"]
        asked = "".join(
            json.dumps(request.body) for request in stub_llm_server.requests
        )
        assert prompt in asked, "beta's turn never reached the model"

        after = {str(entry["name"]): entry for entry in status_json(fleet)}

        # B is the same process in every respect the fleet publishes, and in the
        # one it does not: a pid recycled onto another process would lead another
        # group.
        assert after["beta"] == before["beta"], (after["beta"], before["beta"])
        assert int(after["beta"]["pid"]) == beta_pid
        assert after["beta"]["state"] == "running"
        assert process_is_running(beta_pid)
        assert os.getpgid(beta_pid) == beta_group

        # A is reported exited, under the pid that was killed, with the
        # supervisor's own account of why.
        assert after["alpha"]["state"] == "exited"
        assert int(after["alpha"]["pid"]) == alpha_pid
        assert after["alpha"]["exit_reason"] == "signal"
        # Every other fact about A is untouched: only its liveness changed.
        for key in RECORD_FIELDS:
            if key in ("state", "exit_reason"):
                continue
            assert after["alpha"][key] == before["alpha"][key], key

        # Give a restart every chance to happen before denying that one did: the
        # supervisor sweeps several times inside this window.
        time.sleep(RESTART_WINDOW_SECONDS)
        settled = {str(entry["name"]): entry for entry in status_json(fleet)}
        table = process_table()

        # A stays dead. Three independent readings, none of which relies on an
        # argv: the record still names the same dead pid, the supervisor has
        # exactly one child left, and nothing has rebound A's port.
        assert settled["alpha"] == after["alpha"], (settled["alpha"], after["alpha"])
        assert settled["beta"] == before["beta"], (settled["beta"], before["beta"])
        assert not process_is_running(alpha_pid)
        assert alpha_pid not in live_pids(table)
        assert fleet.supervisor.poll() is None, (
            "the supervisor exited, so nothing was watching for a restart\n"
            f"{log_tail(fleet.supervisor_log)}"
        )
        assert children_of(table, fleet.supervisor.pid) == {beta_pid}, describe(
            tuple(one for one in table if one.ppid == fleet.supervisor.pid)
        )
        assert not health(alpha_port), "something is serving on alpha's port again"
        assert health(beta_port), (
            f"beta stopped serving\n{log_tail(fleet.logs['beta'])}"
        )


# ----------------------------------------------------------------------------
# The sweep's own sensitivity
# ----------------------------------------------------------------------------


@has_ps
def test_the_sweep_finds_a_live_child_of_a_given_process(tmp_path: Path) -> None:
    """``children_of`` really can see a child, so an empty answer means empty.

    The half criterion 3 leans on hardest. "The supervisor has no new children"
    is worth nothing from a reading that could not have found one.
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
    )
    try:
        deadline = time.monotonic() + DEATH_TIMEOUT_SECONDS
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.05)
        assert marker.exists(), "the decoy never started"
        table = process_table()
    finally:
        with suppress(OSError):
            decoy.kill()
        with suppress(subprocess.TimeoutExpired):
            decoy.wait(timeout=DEATH_TIMEOUT_SECONDS)

    assert decoy.pid in children_of(table, os.getpid())
    assert children_of(table, decoy.pid) == set()


@has_ps
def test_a_zombie_child_is_not_counted_as_a_live_child(tmp_path: Path) -> None:
    """An exited-but-unreaped child is listed by ``ps`` and is not a process.

    Pinned against a real zombie rather than a hand-written ``ps`` line, because
    a killed instance is exactly this between its death and the supervisor's
    reaping of it — so counting one would make "A is gone" fail intermittently.
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
        deadline = time.monotonic() + DEATH_TIMEOUT_SECONDS
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.05)
        assert marker.exists(), "the zombie's parent never started"
        zombie_pid = int(json.loads(marker.read_text(encoding="utf-8"))["pid"])
        observed = False
        table: tuple[Process, ...] = ()
        deadline = time.monotonic() + DEATH_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            table = process_table()
            found = [one for one in table if one.pid == zombie_pid]
            if found and found[0].zombie:
                observed = True
                break
            time.sleep(0.05)
    finally:
        stop_tree(parent.pid)
        with suppress(subprocess.TimeoutExpired):
            parent.wait(timeout=DEATH_TIMEOUT_SECONDS)

    assert observed, "no zombie was ever observed, so this proves nothing"
    assert zombie_pid not in children_of(table, parent.pid)
    assert zombie_pid not in live_pids(table)
    assert not process_is_running(zombie_pid)


@has_ps
def test_this_host_really_reports_processes_with_their_parents() -> None:
    """``ps`` answers here, so an empty sweep means empty and not unavailable."""
    table = process_table()

    assert len(table) > 1
    mine = [one for one in table if one.pid == os.getpid()]
    assert len(mine) == 1
    assert mine[0].ppid == os.getppid()
    assert mine[0].pgid == os.getpgid(0)


def test_the_health_probe_answers_false_for_a_port_nothing_is_listening_on() -> None:
    """Otherwise "nothing rebound A's port" would be true of every port."""
    assert not health(free_port(), timeout=1.0)


# ----------------------------------------------------------------------------
# The sweep's machinery
# ----------------------------------------------------------------------------


def test_the_process_table_is_parsed_into_pid_parent_group_state_and_arguments() -> None:
    parsed = parse_processes(
        "    1     0     1 Ss   /sbin/launchd\n"
        "  902   900   900 S+   python -m nanobot serve --config /tmp/c.json\n"
    )

    assert parsed == (
        Process(pid=1, ppid=0, pgid=1, stat="Ss", args="/sbin/launchd"),
        Process(
            pid=902,
            ppid=900,
            pgid=900,
            stat="S+",
            args="python -m nanobot serve --config /tmp/c.json",
        ),
    )


def test_a_line_that_is_not_a_process_is_dropped_rather_than_guessed_at() -> None:
    """A header, a blank line or a wrapped argv must not become a fake process.

    A line whose *parent* does not parse is dropped too: a sweep that invented a
    ppid would answer "was a replacement started" with a number it made up.
    """
    assert parse_processes(
        "  PID  PPID  PGID STAT ARGS\n"
        "\n"
        "   \n"
        "-1 -1 -1 S x\n"
        "123\n"
        "7 seven 7 S sleep\n"
        "7 7 seven S sleep\n"
        "7 7 7 S    \n"
    ) == ()


def test_children_are_found_by_parent_and_nothing_else_is() -> None:
    table = (
        Process(pid=10, ppid=1, pgid=10, stat="S", args="the supervisor"),
        Process(pid=20, ppid=10, pgid=20, stat="S", args="alpha"),
        Process(pid=30, ppid=10, pgid=30, stat="S", args="beta"),
        Process(pid=40, ppid=20, pgid=20, stat="S", args="something alpha started"),
        Process(pid=50, ppid=1, pgid=50, stat="S", args="unrelated"),
    )

    assert children_of(table, 10) == {20, 30}
    assert children_of(table, 20) == {40}
    assert children_of(table, 50) == set()


def test_a_zombie_is_neither_a_child_nor_a_live_pid() -> None:
    """The one exception is stated once and applies to both readings."""
    table = (
        Process(pid=20, ppid=10, pgid=20, stat="Z+", args="(python)"),
        Process(pid=30, ppid=10, pgid=30, stat="S", args="beta"),
    )

    assert children_of(table, 10) == {30}
    assert live_pids(table) == {30}


def test_a_live_process_is_not_mistaken_for_a_zombie() -> None:
    """``S`` and ``Z`` are not the only states, and only one of them is excluded."""
    for stat in ("S", "Ss", "S+", "R", "RN", "SN", "U", "I"):
        assert not Process(pid=10, ppid=1, pgid=10, stat=stat, args="x").zombie, stat
    for stat in ("Z", "Z+"):
        assert Process(pid=10, ppid=1, pgid=10, stat=stat, args="x").zombie, stat


def test_processes_are_described_with_everything_needed_to_find_them() -> None:
    """A failure has to name the process well enough to go and look at it."""
    described = describe(
        (Process(pid=10, ppid=9, pgid=8, stat="S", args="sleep 300"),)
    )

    assert described == "  pid=10 ppid=9 pgid=8 stat=S sleep 300"

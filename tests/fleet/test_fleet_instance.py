"""Tests for launching one fleet instance as a confined child.

Three layers, deliberately separated.

The argv tests are pure and run anywhere: the command is a data structure until
something spawns it, and the property the fleet depends on — ``sandbox-exec``
outermost, ``env -i`` next, the interpreter innermost — is visible in that data
structure. Getting the order wrong would produce a fleet that starts and serves
normally while confining nothing.

The spawn tests drive the real :func:`~nanobot.fleet.instance.launch_instance`
with an injected ``Popen`` so the flags it passes can be inspected without a real
process. They are macOS-gated only because the launcher refuses to run at all
without ``/usr/bin/sandbox-exec``, which is itself one of the behaviours tested.

The last test starts a real confined process through the real kernel policy. It
substitutes a shell script for the interpreter — the same reason
``ManagedProcessRuntime`` accepts ``python_executable`` — so the chain can be
exercised end to end without booting a whole agent behind it, and so the child can
be asked the questions only a live process can answer: which pid am I, is my tree
one process group, can I read a peer, and what does my environment contain.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, get_args

import pytest

from nanobot.fleet.config import FleetInstance, InstanceMode
from nanobot.fleet.instance import (
    ENV_EXEC,
    INSTANCE_LOG_NAME,
    LAUNCH_MODES,
    SANDBOX_EXEC,
    InstanceLaunchError,
    build_instance_command,
    ensure_instance_directories,
    instance_identity,
    instance_identity_match,
    instance_log_path,
    launch_instance,
)
from nanobot.fleet.profile import SeatbeltProfileError, build_instance_profile
from nanobot.fleet.validate import ResolvedInstance

confinement_available = pytest.mark.skipif(
    sys.platform != "darwin" or not Path(SANDBOX_EXEC).is_file(),
    reason="the launcher refuses to start an instance without native Seatbelt",
)

MINIMAL_PROFILE = "(version 1)\n(allow default)"
READY_TIMEOUT_SECONDS = 30.0


def make_instance(
    root: Path,
    name: str,
    *,
    mode: InstanceMode = "serve",
    env: list[str] | None = None,
) -> ResolvedInstance:
    """Resolve one instance laid out the way nanobot lays one out by itself.

    ``<root>/<name>/config.json`` with ``<root>/<name>/workspace`` beneath it. The
    workspace is deliberately *not* created: creating it is the launcher's job,
    and one of the things under test is that it does it.
    """
    config_dir = root / name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    return ResolvedInstance(
        name=name,
        entry=FleetInstance(
            config=str(config_path),
            mode=mode,
            memory_limit_mb=512,
            env=env or [],
        ),
        config_path=config_path,
        config_dir=config_dir,
        workspace=config_dir / "workspace",
        port=None,
        port_setting="api.port",
    )


class FakePopen:
    """A stand-in for ``subprocess.Popen`` that records how it was called."""

    def __init__(self, pid: int = 0, returncode: int | None = None) -> None:
        self.pid = pid or os.getpid()
        self.returncode = returncode
        self.command: list[str] = []
        self.kwargs: dict[str, Any] = {}
        self.calls = 0

    def __call__(self, command: list[str], **kwargs: Any) -> FakePopen:
        self.calls += 1
        self.command = command
        self.kwargs = kwargs
        return self

    def poll(self) -> int | None:
        return self.returncode


def build(instance: ResolvedInstance, **kwargs: Any) -> list[str]:
    """Build one instance's command over the trivial profile."""
    environment = kwargs.pop("environment", {"PATH": "/usr/bin", "HOME": "/home/a"})
    return build_instance_command(
        instance, profile=MINIMAL_PROFILE, environment=environment, **kwargs
    )


# --------------------------------------------------------------------------
# The shape of the child command
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["gateway", "serve"])
def test_sandbox_exec_comes_before_the_interpreter(tmp_path, mode) -> None:
    """The ordering the whole feature rests on.

    A Seatbelt profile applies to the process it is given and to everything that
    process starts. With ``sandbox-exec`` outermost the instance, its shell tool
    and every grandchild are inside one policy; with it anywhere else the
    interpreter would already be running unconfined by the time the profile was
    applied, and nothing about the running fleet would look different.
    """
    command = build(make_instance(tmp_path.resolve(), "a", mode=mode))

    interpreter = command.index(sys.executable)
    assert command[0] == SANDBOX_EXEC
    assert command.index(SANDBOX_EXEC) < command.index(ENV_EXEC) < interpreter
    assert command[interpreter:] == [
        sys.executable,
        "-m",
        "nanobot",
        mode,
        "--config",
        str(tmp_path.resolve() / "a" / "config.json"),
    ]


def test_the_profile_is_passed_inline_right_after_sandbox_exec(tmp_path) -> None:
    """``-p <profile>``, not a file: there is no profile on disk to read or race."""
    command = build(make_instance(tmp_path.resolve(), "a"))

    assert command[1:3] == ["-p", MINIMAL_PROFILE]


def test_the_environment_is_emptied_before_it_is_populated(tmp_path) -> None:
    """``env -i`` then assignments, in iteration order, then the interpreter.

    Order inside the assignment run is asserted because it is what makes the
    command reproducible: :mod:`nanobot.fleet.env` returns the base variables
    first and the instance's declared names after, and a supervisor that hashes or
    logs the command should get the same words for the same fleet every time.
    """
    command = build(
        make_instance(tmp_path.resolve(), "a"),
        environment={"PATH": "/usr/bin", "HOME": "/home/a", "OPENAI_API_KEY": "k"},
    )

    env_at = command.index(ENV_EXEC)
    assert command[env_at + 1] == "-i"
    assert command[env_at + 2 : env_at + 5] == [
        "PATH=/usr/bin",
        "HOME=/home/a",
        "OPENAI_API_KEY=k",
    ]
    assert command[env_at + 5] == sys.executable


def test_an_empty_environment_still_yields_a_runnable_command(tmp_path) -> None:
    """``env -i`` with nothing after it is the fail-closed case, not an error."""
    command = build(make_instance(tmp_path.resolve(), "a"), environment={})

    assert command[command.index(ENV_EXEC) : command.index(ENV_EXEC) + 3] == [
        ENV_EXEC,
        "-i",
        sys.executable,
    ]


def test_both_absolute_tools_are_named_rather_than_looked_up() -> None:
    """A ``PATH`` lookup would let the supervisor's environment pick the jailer."""
    assert Path(SANDBOX_EXEC).is_absolute()
    assert Path(ENV_EXEC).is_absolute()


def test_the_launch_modes_are_the_documents_own_modes() -> None:
    """Pinned so the launcher cannot accept a mode the fleet schema refuses."""
    assert LAUNCH_MODES == ("gateway", "serve")
    assert LAUNCH_MODES == get_args(InstanceMode)


def test_a_substituted_interpreter_replaces_only_the_innermost_link(tmp_path) -> None:
    command = build(make_instance(tmp_path.resolve(), "a"), python_executable="/opt/py")

    assert command[0] == SANDBOX_EXEC
    assert command[command.index("/opt/py") :][:3] == ["/opt/py", "-m", "nanobot"]


# --------------------------------------------------------------------------
# Refusals in building
# --------------------------------------------------------------------------


def test_an_unknown_mode_is_refused(tmp_path) -> None:
    """Re-checked where it is used, the way peer names are in the profile builder.

    ``FleetInstance`` already constrains ``mode``, but the value becomes an argv
    word selecting which server runs and which port is bound, so the launcher does
    not take it on trust from a model that could be constructed directly.
    """
    instance = make_instance(tmp_path.resolve(), "a")
    instance.entry.mode = "webui"  # type: ignore[assignment]

    with pytest.raises(InstanceLaunchError, match="is not a launch mode"):
        build(instance)


def test_an_environment_key_that_would_smuggle_a_variable_is_refused(tmp_path) -> None:
    """``NAME=value`` is one argv word, so a ``=`` in the key declares two."""
    with pytest.raises(InstanceLaunchError, match="not a valid environment variable"):
        build(
            make_instance(tmp_path.resolve(), "a"),
            environment={"PATH=/evil:/usr/bin\nHOME": "/tmp"},
        )


def test_an_unavailable_interpreter_is_refused(tmp_path, monkeypatch) -> None:
    """An empty ``sys.executable`` would otherwise make ``-m`` the command."""
    monkeypatch.setattr(sys, "executable", "")

    with pytest.raises(InstanceLaunchError, match="no Python interpreter"):
        build(make_instance(tmp_path.resolve(), "a"))


@pytest.mark.parametrize("tool", ["SANDBOX_EXEC", "ENV_EXEC"])
def test_a_missing_confinement_tool_refuses_the_launch(tmp_path, monkeypatch, tool) -> None:
    """Refuse rather than fall back: the fallback is an unconfined instance.

    Nothing is created and nothing is spawned on this path, so a host that cannot
    confine an instance is left exactly as it was.
    """
    monkeypatch.setattr(
        f"nanobot.fleet.instance.{tool}", str(tmp_path / "absent" / "tool")
    )
    instance = make_instance(tmp_path.resolve(), "a")
    popen = FakePopen()

    with pytest.raises(InstanceLaunchError, match="isolation cannot be applied"):
        launch_instance(instance, profile=MINIMAL_PROFILE, popen=popen)

    assert popen.calls == 0
    assert not instance.workspace.exists()
    assert not instance_log_path(instance).parent.exists()


# --------------------------------------------------------------------------
# Directories, and the order they have to exist in
# --------------------------------------------------------------------------


def test_the_log_lives_inside_the_instances_own_config_dir(tmp_path) -> None:
    """Which is what makes it unreadable by the rest of the fleet.

    Every peer's profile denies ``(subpath <peer config_dir>)``, so a log placed
    here is peer-denied for free. A shared supervisor-side log directory would not
    be: the profile builder denies the fleet file and the state file as literals,
    and a sibling directory beside them is not covered by either.
    """
    instance = make_instance(tmp_path.resolve(), "a")

    log_path = instance_log_path(instance)

    assert log_path.is_relative_to(instance.config_dir)
    assert log_path.name == INSTANCE_LOG_NAME


def test_preparing_directories_creates_the_workspace_and_log_directory(tmp_path) -> None:
    instance = make_instance(tmp_path.resolve(), "a")

    directories = ensure_instance_directories(instance)

    assert directories.workspace.is_dir()
    assert directories.logs_dir.is_dir()
    assert directories.log_path == instance_log_path(instance)
    assert stat.S_IMODE(directories.workspace.stat().st_mode) == 0o700


def test_preparing_directories_is_idempotent_and_keeps_existing_content(tmp_path) -> None:
    instance = make_instance(tmp_path.resolve(), "a")
    ensure_instance_directories(instance)
    (instance.workspace / "notes.md").write_text("kept", encoding="utf-8")

    ensure_instance_directories(instance)

    assert (instance.workspace / "notes.md").read_text(encoding="utf-8") == "kept"


def test_an_uncreatable_workspace_is_refused_naming_the_instance(tmp_path) -> None:
    root = tmp_path.resolve()
    instance = make_instance(root, "a")
    blocker = root / "a" / "workspace"
    blocker.write_text("not a directory", encoding="utf-8")

    with pytest.raises(InstanceLaunchError, match="instance a: unable to create"):
        ensure_instance_directories(instance)


def test_directories_must_be_prepared_before_a_profile_can_be_built(tmp_path) -> None:
    """The ordering constraint the two modules share, asserted across both.

    ``nanobot.fleet.profile`` refuses a path that does not exist, because Seatbelt
    would accept a rule naming one and then confine nothing. An instance's
    workspace does not exist before its first start, so a supervisor that built
    profiles first would be refused — and a supervisor that silently skipped the
    refusal would run a fleet that only looked isolated.
    """
    root = tmp_path.resolve()
    a, b = make_instance(root, "a"), make_instance(root, "b")
    fleet, state = root / "fleet.json", root / "state.json"
    fleet.write_text("{}", encoding="utf-8")
    state.write_text("{}", encoding="utf-8")

    with pytest.raises(SeatbeltProfileError, match="not an existing directory"):
        build_instance_profile(a, [b], fleet_path=fleet, state_path=state)

    for instance in (a, b):
        ensure_instance_directories(instance)
    profile = build_instance_profile(a, [b], fleet_path=fleet, state_path=state)

    assert str(b.workspace) in profile


# --------------------------------------------------------------------------
# Spawning
# --------------------------------------------------------------------------


@confinement_available
@pytest.mark.parametrize("mode", ["gateway", "serve"])
def test_the_child_is_started_in_its_own_session(tmp_path, mode) -> None:
    """``start_new_session=True`` is what makes the instance a group leader.

    Without it the child would join the supervisor's process group, and both
    tree-wide memory measurement and tree-wide termination would then be aimed at
    the supervisor itself.
    """
    instance = make_instance(tmp_path.resolve(), "a", mode=mode)
    popen = FakePopen()

    launched = launch_instance(instance, profile=MINIMAL_PROFILE, popen=popen)

    assert popen.kwargs["start_new_session"] is True
    assert popen.kwargs["close_fds"] is True
    assert launched.mode == mode
    assert launched.command[0] == SANDBOX_EXEC
    assert list(launched.command) == popen.command


@confinement_available
def test_the_child_gets_no_terminal_and_one_merged_output_stream(tmp_path) -> None:
    """stdin is closed off so an instance cannot compete for the operator's terminal."""
    instance = make_instance(tmp_path.resolve(), "a")
    popen = FakePopen()

    launched = launch_instance(instance, profile=MINIMAL_PROFILE, popen=popen)

    assert popen.kwargs["stdin"] is subprocess.DEVNULL
    assert popen.kwargs["stderr"] is subprocess.STDOUT
    handle = popen.kwargs["stdout"]
    assert Path(handle.name) == launched.log_path
    # The child holds its own duplicate; a supervisor copy left open would pin the
    # log file for as long as the fleet ran.
    assert handle.closed


@confinement_available
def test_launching_creates_the_log_and_appends_to_it(tmp_path) -> None:
    """Append, not truncate: the fleet never restarts, so a log is one run's record.

    Keeping earlier content means a new fleet over the same config dir does not
    erase the evidence of why the previous one stopped.
    """
    instance = make_instance(tmp_path.resolve(), "a")
    ensure_instance_directories(instance)
    instance_log_path(instance).write_text("previous run\n", encoding="utf-8")

    launched = launch_instance(instance, profile=MINIMAL_PROFILE, popen=FakePopen())

    assert launched.log_path.read_text(encoding="utf-8") == "previous run\n"


@confinement_available
def test_only_the_instances_own_declared_variables_are_forwarded(
    tmp_path, monkeypatch
) -> None:
    """The default environment comes from this entry's list, never a peer's."""
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("FLEET_MINE", "mine")
    monkeypatch.setenv("FLEET_THEIRS", "theirs")
    instance = make_instance(tmp_path.resolve(), "a", env=["FLEET_MINE"])
    popen = FakePopen()

    launched = launch_instance(instance, profile=MINIMAL_PROFILE, popen=popen)

    assert "FLEET_MINE=mine" in launched.command
    assert "FLEET_THEIRS=theirs" not in launched.command
    # The two wrapper processes are given the same environment, so the
    # supervisor's other variables do not survive even as far as ``env -i``.
    assert popen.kwargs["env"].get("FLEET_THEIRS") is None
    assert popen.kwargs["env"]["FLEET_MINE"] == "mine"


@confinement_available
def test_a_failed_spawn_is_reported_against_the_instance(tmp_path) -> None:
    instance = make_instance(tmp_path.resolve(), "a")

    def refuse(command: list[str], **kwargs: Any) -> Any:
        raise OSError(13, "Permission denied")

    with pytest.raises(InstanceLaunchError, match="instance a: unable to start"):
        launch_instance(instance, profile=MINIMAL_PROFILE, popen=refuse)


@confinement_available
def test_an_unopenable_log_refuses_the_launch(tmp_path) -> None:
    """Refused before spawning: an instance whose output goes nowhere is unobservable."""
    instance = make_instance(tmp_path.resolve(), "a")
    ensure_instance_directories(instance)
    instance_log_path(instance).mkdir()
    popen = FakePopen()

    with pytest.raises(InstanceLaunchError, match="instance a: unable to open"):
        launch_instance(instance, profile=MINIMAL_PROFILE, popen=popen)

    assert popen.calls == 0


@confinement_available
def test_a_group_lookup_that_fails_falls_back_to_the_leaders_pid(tmp_path) -> None:
    """A child that exits before the lookup still led the group named by its pid.

    ``start_new_session=True`` makes the two equal, so the fallback is the right
    answer rather than a guess — and it keeps a fast-failing instance from
    producing a group id the supervisor would signal blindly.
    """
    instance = make_instance(tmp_path.resolve(), "a")
    # Above every platform's PID ceiling, so the lookup cannot succeed by accident.
    absent = 2**31 - 1

    launched = launch_instance(
        instance, profile=MINIMAL_PROFILE, popen=FakePopen(pid=absent)
    )

    assert launched.pgid == absent


@confinement_available
def test_liveness_falls_back_to_the_pid_when_the_handle_cannot_answer(tmp_path) -> None:
    """A handle that cannot poll must not be read as "not running"."""

    class UnpollableHandle(FakePopen):
        def poll(self) -> int | None:
            raise OSError("no such process")

    instance = make_instance(tmp_path.resolve(), "a")

    launched = launch_instance(
        instance, profile=MINIMAL_PROFILE, popen=UnpollableHandle(pid=os.getpid())
    )

    assert launched.is_running() is True


@confinement_available
def test_a_recorded_identity_accompanies_the_pid(tmp_path) -> None:
    """A pid alone is not an identity; a recycled one would be reported as live."""
    instance = make_instance(tmp_path.resolve(), "a")

    launched = launch_instance(
        instance, profile=MINIMAL_PROFILE, popen=FakePopen(pid=os.getpid())
    )

    assert launched.pid == os.getpid()
    assert launched.identity == instance_identity(os.getpid())
    assert instance_identity_match(launched.identity["stable_identity"], os.getpid()) == "match"


@confinement_available
def test_liveness_prefers_the_owned_handle(tmp_path) -> None:
    """Which is the only way to tell a live instance from an unreaped zombie.

    On POSIX an exited child stays visible to ``kill(pid, 0)`` until its parent
    waits for it, so a pid probe alone would report a dead instance as running.
    """
    instance = make_instance(tmp_path.resolve(), "a")

    live = launch_instance(
        instance, profile=MINIMAL_PROFILE, popen=FakePopen(pid=os.getpid())
    )
    exited = launch_instance(
        instance,
        profile=MINIMAL_PROFILE,
        popen=FakePopen(pid=os.getpid(), returncode=1),
    )

    assert live.is_running() is True
    assert exited.is_running() is False


# --------------------------------------------------------------------------
# Identity, delegated rather than reimplemented
# --------------------------------------------------------------------------


def test_a_stale_identity_does_not_match_a_live_pid() -> None:
    """The reconciliation the supervisor's state file depends on."""
    assert instance_identity_match("darwin:1:1:1", os.getpid()) == "mismatch"


def test_an_unreadable_process_identity_is_unknown_rather_than_mismatched() -> None:
    """Three-valued on purpose: a failed read must not look like a recycled pid.

    Reported as a mismatch, a transient failure would mark a healthy instance
    exited; reported as a match, a recycled pid would be signalled as if it were
    still the instance.
    """
    recorded = instance_identity(os.getpid())["stable_identity"]

    assert instance_identity_match(recorded, 0) == "unknown"


# --------------------------------------------------------------------------
# The kernel, not just the flags
# --------------------------------------------------------------------------

# A stand-in for the instance interpreter. It answers the questions only a live
# confined process can: the pid it actually runs as, whether a peer's file is
# readable, and what its environment contains. Then it forks a child the way an
# instance's shell tool would and waits, so the tree can be inspected.
#
# Deliberately free of ``ps``: ``sandbox-exec`` refuses to exec a setgid binary
# even under ``(allow default)``, and ``/bin/ps`` is setgid ``kmem``. The process
# group is read from the supervisor side instead, which is where it matters.
INTERPRETER_STUB = """#!/bin/sh
echo "argv: $*"
echo "pid: $$"
if cat "__PEER_SENTINEL__" >/dev/null 2>&1; then
    echo "peer: readable"
else
    echo "peer: denied"
fi
echo "declared: ${FLEET_DECLARED:-absent}"
echo "undeclared: ${FLEET_UNDECLARED:-absent}"
sleep 300 &
echo "child: $!"
echo "ready"
wait
"""


def write_interpreter_stub(path: Path, peer_sentinel: Path) -> Path:
    """Write the stub with the peer path baked in — ``env -i`` would strip it."""
    path.write_text(
        INTERPRETER_STUB.replace("__PEER_SENTINEL__", str(peer_sentinel)),
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def wait_for_ready(log_path: Path) -> str:
    """Return the log once the child has reported in, or fail with what it said.

    There is no ``pytest-timeout`` in this repo, so the deadline is explicit and
    the log is inlined on failure — the child's own output is the only diagnostic
    a confined process leaves behind.
    """
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    text = ""
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        if "ready" in text:
            return text
        time.sleep(0.05)
    pytest.fail(f"instance never reported ready within {READY_TIMEOUT_SECONDS}s:\n{text}")


def stop_tree(launched: Any) -> None:
    """Kill the instance's process group, never the test runner's.

    The guard is not defensive padding: if ``start_new_session`` were ever dropped
    from the launcher, the child would join *this* process's group and an
    unguarded ``killpg`` would kill pytest itself — which is how that mutation
    first presented, as a silent runner death rather than a failed assertion.
    Tree termination is its own bead; the same guard belongs there.
    """
    if launched.pgid != os.getpgid(0):
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(launched.pgid, signal.SIGKILL)
    else:  # pragma: no cover - only reachable if the launcher stops isolating
        with suppress(ProcessLookupError, PermissionError):
            os.kill(launched.pid, signal.SIGKILL)


def field(text: str, name: str) -> str:
    """Return the value the stub reported for ``name``."""
    prefix = f"{name}: "
    line = next((one for one in text.splitlines() if one.startswith(prefix)), None)
    assert line is not None, f"{name} missing from instance log:\n{text}"
    return line[len(prefix) :]


@confinement_available
@pytest.mark.parametrize("mode", ["gateway", "serve"])
def test_a_launched_instance_is_confined_and_owns_its_process_group(
    tmp_path, monkeypatch, mode
) -> None:
    """One real process through the real policy, in both modes.

    Everything above proves the launcher *says* the right thing. This proves the
    OS agrees: the argv survives both wrappers intact, the pid the supervisor
    recorded is the pid the confined process runs as (so the chain is an exec
    chain, not three nested children), a forked descendant lands in the same
    process group, the peer's workspace is unreadable, and the environment holds
    only what this instance declared.
    """
    root = tmp_path.resolve()
    monkeypatch.setenv("FLEET_DECLARED", "for-a")
    monkeypatch.setenv("FLEET_UNDECLARED", "not-for-a")
    a = make_instance(root, "a", mode=mode, env=["FLEET_DECLARED"])
    b = make_instance(root, "b")
    fleet, state = root / "fleet.json", root / "state.json"
    fleet.write_text("{}", encoding="utf-8")
    state.write_text("{}", encoding="utf-8")
    for instance in (a, b):
        ensure_instance_directories(instance)
    sentinel = b.workspace / "sentinel"
    sentinel.write_text("peer-secret", encoding="utf-8")
    profile = build_instance_profile(a, [b], fleet_path=fleet, state_path=state)
    stub = write_interpreter_stub(root / "stub.sh", sentinel)

    launched = launch_instance(a, profile=profile, python_executable=str(stub))
    try:
        text = wait_for_ready(launched.log_path)
        child_pid = int(field(text, "child"))

        assert field(text, "argv") == (
            f"-m nanobot {mode} --config {a.config_path}"
        )
        # An exec chain, so the pid the supervisor recorded is the confined
        # process itself rather than a wrapper that has already gone away.
        assert int(field(text, "pid")) == launched.pid
        assert launched.pgid == launched.pid
        assert os.getpgid(child_pid) == launched.pgid
        assert field(text, "peer") == "denied"
        assert "peer-secret" not in text
        assert field(text, "declared") == "for-a"
        assert field(text, "undeclared") == "absent"
        assert launched.is_running() is True
    finally:
        # The group, not the leader: the forked descendant would otherwise outlive
        # the test. Stopping a tree properly is its own bead; this is cleanup.
        stop_tree(launched)
        launched.process.wait(timeout=READY_TIMEOUT_SECONDS)

    assert launched.is_running() is False

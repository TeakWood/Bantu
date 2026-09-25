"""Launch one fleet instance as a confined child of the supervisor.

The child command is a three-link chain, and the order of the links is the whole
point::

    /usr/bin/sandbox-exec -p <profile> \\
        /usr/bin/env -i <minimal environment> \\
        <python> -m nanobot <gateway|serve> --config <instance config>

``sandbox-exec`` comes first because a Seatbelt profile is applied to a process
and inherited by everything it goes on to start. Putting it outermost means the
instance, its shell tool, and every grandchild a prompt-injected agent manages to
spawn are all inside the same policy, and none of them can step outside it —
which is the difference between the fleet's isolation and the in-app guards it
replaces. ``env -i`` comes next so the *confined* process is the one that gets
the environment, and an instance never sees the supervisor's credentials, only
the variables its own fleet entry named (:mod:`nanobot.fleet.env`).

*One process group per instance.* The child is spawned with
``start_new_session=True``, so it becomes a session and process-group leader and
its group id equals its pid. Every descendant inherits that group unless it
deliberately leaves, which makes the group and the instance's process tree the
same set of processes. That equivalence is what
:mod:`nanobot.fleet.memory` measures and what tree termination will signal; a
child launched into the supervisor's own group would make both of those target
the supervisor.

*Composition over inheritance.* ``nanobot/process_runtime.py`` is reused, not
subclassed, and not modified. ``ManagedProcessRuntime`` is built for a different
shape of problem: one detached background process per service, its own state file
under ``<data_dir>/run/``, a lifecycle lock, restart-in-place. The fleet has many
foreground children, one supervisor-owned aggregate state file kept deliberately
outside every instance's reach, and never restarts anything. What it does want is
that module's PID-reuse-safe identity, because a pid alone is not an identity:
between the write of a state file and the read of it the pid may have been
recycled, and a supervisor that signalled on a recycled pid would kill an
unrelated process. So identity is *delegated* rather than reimplemented — see
:func:`instance_identity`.

*Logs are per instance, and they live inside the instance's own data directory.*
``<config_dir>/logs/fleet.log``, alongside the logs nanobot writes for itself.
That placement is a confinement decision, not a convention: every peer's profile
denies ``(subpath <peer config_dir>)``, so an instance's log is unreadable by the
rest of the fleet for free. A shared supervisor-side log directory would not be —
:mod:`nanobot.fleet.profile` denies the fleet file and the state file as
literals, so a sibling directory beside them would stay world-readable within the
fleet.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, get_args

from nanobot.fleet.config import ENV_NAME_PATTERN, InstanceMode
from nanobot.fleet.env import instance_environment
from nanobot.fleet.validate import ResolvedInstance
from nanobot.process_runtime import (
    ManagedProcessRuntime,
    ProcessRuntimePaths,
    ProcessStartOptions,
    process_identity_record,
    process_is_running,
)

#: The Seatbelt entry point. An absolute path rather than a ``PATH`` lookup: this
#: is the process that applies the confinement, so which binary runs it is not
#: something the supervisor's environment gets to influence.
SANDBOX_EXEC = "/usr/bin/sandbox-exec"

#: ``env -i`` is what empties the environment. Also absolute, for the same
#: reason — a substituted ``env`` would hand the instance whatever it liked.
ENV_EXEC = "/usr/bin/env"

#: The modes an instance may be started in, taken from the fleet document's own
#: type so the launcher cannot drift from what the schema accepts.
LAUNCH_MODES: tuple[str, ...] = get_args(InstanceMode)

#: Where an instance's combined stdout and stderr go, relative to its config dir.
INSTANCE_LOG_SUBDIR = "logs"
INSTANCE_LOG_NAME = "fleet.log"

_ENV_NAME_RE = re.compile(ENV_NAME_PATTERN)

# ``ManagedProcessRuntime`` requires paths for the three things the fleet does
# not use it for: its own per-service state file, its log tail, and its lifecycle
# lock. The identity methods called through it read only the host platform and
# never touch ``paths``. Pointing them at a path that cannot be created keeps
# that honest — a future edit reaching for a state-file method would fail rather
# than quietly write fleet state somewhere an instance can read it.
_IDENTITY_PROBE_ROOT = Path("/nonexistent/nanobot-fleet-identity-probe")


class InstanceLaunchError(RuntimeError):
    """An instance could not be started in a form that is actually confined.

    Always a refusal to launch. Starting an instance unconfined is worse than not
    starting it: the fleet would report a running instance and the operator would
    have no way to tell that its peers' workspaces were readable all along.
    """

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"instance {name}: {reason}")
        self.name = name
        self.reason = reason


@dataclass(frozen=True)
class InstanceDirectories:
    """The directories an instance needs on disk before it can be launched."""

    workspace: Path
    logs_dir: Path
    log_path: Path


@dataclass(frozen=True)
class LaunchedInstance:
    """A live instance process and the facts the supervisor records about it.

    ``pid`` and ``pgid`` are equal by construction (the child is a process-group
    leader) but both are kept: the pid identifies the instance process and the
    pgid identifies its whole tree, and code that means one should not have to
    rely on them coinciding.
    """

    name: str
    mode: str
    pid: int
    pgid: int
    command: tuple[str, ...]
    log_path: Path
    identity: Mapping[str, str | int | None]
    #: The owned ``Popen`` handle. Kept so the supervisor can reap the child: on
    #: POSIX an exited child stays visible to ``kill(pid, 0)`` until its parent
    #: waits for it, so polling this handle is the only way to distinguish a live
    #: instance from a zombie the supervisor has not collected yet.
    process: Any = field(repr=False, compare=False)

    def is_running(self) -> bool:
        """Whether this instance's process is still live.

        Prefers the owned handle, which both reaps and reports, and falls back to
        ``process_is_running`` for a handle that cannot answer.
        """
        poll = getattr(self.process, "poll", None)
        if callable(poll):
            try:
                return poll() is None
            except OSError:
                pass
        return process_is_running(self.pid)


def instance_log_path(instance: ResolvedInstance) -> Path:
    """Where this instance's combined stdout and stderr are written."""
    return instance.config_dir / INSTANCE_LOG_SUBDIR / INSTANCE_LOG_NAME


def ensure_instance_directories(instance: ResolvedInstance) -> InstanceDirectories:
    """Create the workspace and log directory this instance will need.

    Must run **before** :func:`nanobot.fleet.profile.build_instance_profile` is
    called for any instance in the fleet. That builder refuses a path that does
    not exist, precisely because Seatbelt would accept a rule naming one, match
    nothing, and confine nothing; and a workspace commonly does not exist until
    its instance first starts. ``nanobot.fleet.paths`` and
    ``nanobot.fleet.validate`` create nothing on purpose, so that a fleet which
    fails validation leaves no directories behind — which makes creating them the
    launcher's job, and makes the ordering load-bearing.

    Idempotent. Directories are created at mode ``0700``; Seatbelt is what
    actually enforces the fleet's isolation, but a permission bit that does not
    depend on it costs nothing. An existing directory's mode is left alone rather
    than tightened under an operator who chose it.

    Raises:
        InstanceLaunchError: a directory could not be created.
    """
    log_path = instance_log_path(instance)
    directories = InstanceDirectories(
        workspace=instance.workspace,
        logs_dir=log_path.parent,
        log_path=log_path,
    )
    for path in (directories.workspace, directories.logs_dir):
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            detail = exc.strerror or type(exc).__name__
            raise InstanceLaunchError(
                instance.name, f"unable to create {path}: {detail}"
            ) from exc
    return directories


def build_instance_command(
    instance: ResolvedInstance,
    *,
    profile: str,
    environment: Mapping[str, str],
    python_executable: str | None = None,
) -> list[str]:
    """Build the confined child command for one instance.

    Pure: nothing is created, spawned or stat'd, so the argv shape can be checked
    on any platform even though only macOS can run it.

    Args:
        instance: the validated instance to launch.
        profile: the SBPL profile from :mod:`nanobot.fleet.profile`, passed
            inline with ``-p`` rather than through a file so there is no profile
            on disk for an instance to read or race.
        environment: the complete environment the instance may see, normally
            from :func:`nanobot.fleet.env.instance_environment`. Rendered into
            ``NAME=value`` words after ``env -i``, in iteration order.
        python_executable: the interpreter to run ``-m nanobot`` with. Defaults
            to the supervisor's own ``sys.executable``, which is the interpreter
            that has nanobot installed. Overridable for the same reason
            ``ManagedProcessRuntime`` takes it: so a test can exercise the real
            ``sandbox-exec`` and ``env -i`` links of the chain without booting a
            whole agent behind them.

    Raises:
        InstanceLaunchError: the mode is not a launch mode, an environment key is
            not a variable name, or no interpreter is available.
    """
    if instance.mode not in LAUNCH_MODES:
        raise InstanceLaunchError(
            instance.name,
            f"{instance.mode!r} is not a launch mode (expected one of "
            f"{', '.join(LAUNCH_MODES)})",
        )
    interpreter = python_executable or sys.executable
    if not interpreter:
        raise InstanceLaunchError(
            instance.name,
            "no Python interpreter is available to run the instance "
            "(sys.executable is empty)",
        )
    return [
        SANDBOX_EXEC,
        "-p",
        profile,
        ENV_EXEC,
        "-i",
        *_env_words(instance.name, environment),
        interpreter,
        "-m",
        "nanobot",
        instance.mode,
        "--config",
        str(instance.config_path),
    ]


def launch_instance(
    instance: ResolvedInstance,
    *,
    profile: str,
    environment: Mapping[str, str] | None = None,
    python_executable: str | None = None,
    popen: Any = subprocess.Popen,
) -> LaunchedInstance:
    """Start one instance as a confined child in its own process group.

    Args:
        instance: the validated instance to launch.
        profile: its Seatbelt profile. Build it *after*
            :func:`ensure_instance_directories` has run for every instance in the
            fleet; see that function for why the order matters.
        environment: the environment to hand the instance. Defaults to
            :func:`nanobot.fleet.env.instance_environment` over this instance's
            own declared names — never a peer's.
        python_executable: see :func:`build_instance_command`.
        popen: the spawn primitive, injectable for tests.

    Raises:
        InstanceLaunchError: the confinement tools are missing, the command
            cannot be built, the log file cannot be opened, or the spawn fails.
            A missing ``sandbox-exec`` is a refusal rather than a fallback,
            because the fallback is an unconfined instance the fleet would go on
            to report as running.
    """
    _require_confinement_tools(instance.name)
    env = (
        instance_environment(instance.entry.env) if environment is None else environment
    )
    command = build_instance_command(
        instance,
        profile=profile,
        environment=env,
        python_executable=python_executable,
    )
    directories = ensure_instance_directories(instance)

    try:
        log_handle = directories.log_path.open("ab")
    except OSError as exc:
        detail = exc.strerror or type(exc).__name__
        raise InstanceLaunchError(
            instance.name, f"unable to open {directories.log_path}: {detail}"
        ) from exc
    try:
        process = popen(
            command,
            # The instance is not attached to the supervisor's terminal. Without
            # this a child that read stdin would compete with the operator for
            # it, and a gateway that prompted would block the whole fleet.
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            # The one flag that makes the instance a process-group leader, and so
            # makes its tree addressable as a group.
            start_new_session=True,
            # Only what the profile and ``env -i`` already imply: the child's own
            # environment is replaced by ``env -i`` regardless, so this keeps the
            # supervisor's variables out of even the two intermediate processes.
            env=dict(env),
            close_fds=True,
        )
    except (OSError, ValueError) as exc:
        detail = getattr(exc, "strerror", None) or str(exc) or type(exc).__name__
        raise InstanceLaunchError(
            instance.name, f"unable to start {SANDBOX_EXEC}: {detail}"
        ) from exc
    finally:
        # The child holds its own duplicate of this descriptor; keeping the
        # supervisor's copy open would pin the log file across the fleet's life.
        log_handle.close()

    pid = int(process.pid)
    return LaunchedInstance(
        name=instance.name,
        mode=instance.mode,
        pid=pid,
        pgid=_process_group(pid),
        command=tuple(command),
        log_path=directories.log_path,
        identity=instance_identity(pid),
        process=process,
    )


def instance_identity(pid: int) -> dict[str, str | int | None]:
    """Return a PID-reuse-safe identity record for ``pid``.

    The value the supervisor's state file stores next to a pid so that a later
    read can tell "this instance is still running" from "this pid has been handed
    to somebody else". Produced by ``process_runtime.process_identity_record``,
    and compared by :func:`instance_identity_match`; the two must stay the same
    implementation, which is the reason neither is reimplemented here. A second
    copy of the platform-specific identity format could drift from the comparison
    it is meant to feed, and the failure mode of that drift is the supervisor
    signalling a process it does not own.

    Inherited behaviour worth knowing when reconciling state: on a platform where
    the identity cannot be read the record's ``identity`` is ``None``, and a
    ``None`` record compares as ``"match"`` — ``process_runtime`` prefers trusting
    a live pid over killing a process it merely failed to identify.
    """
    return process_identity_record(_identity_probe().process_identity(pid))


def instance_identity_match(
    recorded: object,
    pid: int,
) -> Literal["match", "mismatch", "unknown"]:
    """Compare a recorded identity with the process currently holding ``pid``.

    Three-valued on purpose. ``"unknown"`` means the identity could not be read
    *now*, which is not the same as a mismatch: treating it as one would let a
    transient read failure turn a healthy instance into an exited one, and treating
    it as a match would report a recycled pid as running.
    """
    return _identity_probe().process_identity_match(recorded, pid)


def _require_confinement_tools(name: str) -> None:
    """Refuse to launch on a host that cannot apply the confinement.

    Read from module globals at call time so both paths are one monkeypatch away
    in a test, and checked before anything is created or spawned.
    """
    for what, tool in (("Seatbelt", SANDBOX_EXEC), ("environment", ENV_EXEC)):
        if not Path(tool).is_file():
            raise InstanceLaunchError(
                name,
                f"{tool} is missing, so {what} isolation cannot be applied; "
                f"the fleet refuses to start an unconfined instance",
            )


def _env_words(name: str, environment: Mapping[str, str]) -> list[str]:
    """Render ``environment`` as ``env -i`` assignment words.

    Names are re-checked here even though ``nanobot.fleet.config`` constrains
    every declared name and ``nanobot.fleet.env`` re-checks them again, because
    this function also accepts a caller-supplied mapping. A key containing ``=``
    would become an argv word declaring a second variable the fleet document
    never named, and a key containing whitespace or a newline is not a variable
    at all. The invariant is enforced where it is relied upon.
    """
    words: list[str] = []
    for key, value in environment.items():
        if not _ENV_NAME_RE.fullmatch(key):
            raise InstanceLaunchError(
                name,
                f"{key!r} is not a valid environment variable name (must match "
                f"{ENV_NAME_PATTERN}); it cannot be passed to {ENV_EXEC} -i",
            )
        words.append(f"{key}={value}")
    return words


def _process_group(pid: int) -> int:
    """Return ``pid``'s process group, falling back to the pid itself.

    ``start_new_session=True`` makes the child its own group leader, so its group
    id *is* its pid; the lookup exists only to observe that rather than assume it.
    A child that exits between the spawn and this call makes ``getpgid`` fail, and
    the pid is then still the right answer — it is the group the child led.
    """
    try:
        return os.getpgid(pid)
    except OSError:
        return pid


@lru_cache(maxsize=1)
def _identity_probe() -> ManagedProcessRuntime[ProcessStartOptions]:
    """A ``ManagedProcessRuntime`` used only as a pid-identity oracle.

    Composition, not inheritance: nothing here subclasses it, overrides it, or
    starts anything through it. Only its three platform-inspection methods are
    called, and none of them reads ``paths`` — see ``_IDENTITY_PROBE_ROOT``.
    """
    return ManagedProcessRuntime[ProcessStartOptions](
        paths=ProcessRuntimePaths(
            run_dir=_IDENTITY_PROBE_ROOT,
            logs_dir=_IDENTITY_PROBE_ROOT,
            state_path=_IDENTITY_PROBE_ROOT / "state.json",
            log_path=_IDENTITY_PROBE_ROOT / "log",
        )
    )

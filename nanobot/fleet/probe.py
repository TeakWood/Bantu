"""Prove a built profile actually binds, before the fleet reports it started.

Everything else in the fleet package can be checked by reading it. This cannot,
because the OS gives no feedback. Verified on the pinned host: ``sandbox-exec``
accepts a deny naming a non-canonical path (``/tmp/x`` where the real path is
``/private/tmp/x``), a deny naming a path that does not exist, and a deny with an
invalid relative subpath — and in all three cases it starts the process cleanly,
confines nothing, and exits 0. There is no error, no warning, and no difference
an operator could observe. ``nanobot.fleet.profile`` closes that gap by refusing
to *emit* such a rule, which is the right place for it; this module closes the
remaining gap, which is that "the builder was satisfied" is a claim about the
builder and not about the kernel. The only way to know a profile confines is to
run it and watch a read fail.

*How the two runs decide.* The probe reads one path that the profile must deny,
then — only if that read was refused — reads the same path again under a control
profile that denies nothing:

1. the read succeeds under the instance's profile → the deny did not bind, and no
   control run is needed, because a successful read settles it;
2. the read is refused under the instance's profile *and* permitted under the
   control → the profile is what refused it, which is the thing being proven;
3. the read is refused under both → the refusal is not evidence of anything. The
   path may have been deleted, the probe tool may be broken, the host may be
   refusing for its own reasons. Reported as *not* proven.

The control run comes second, rather than first, precisely because of case 3. A
control-first probe that saw the path vanish between its two runs would attribute
the second refusal to the profile and report a confinement that was never tested.
Running it second makes every ambiguity fail closed, and costs nothing in the
case that matters least — an unconfined instance is caught in one spawn.

*Why the outcome is a value and not an exception.* A missing ``sandbox-exec`` is
returned as a negative result, not raised. The caller decides policy: ``nanobot
fleet start`` refuses to start anything, while a diagnostic command may well want
to report every instance's probe rather than stop at the first host that cannot
run one. :class:`nanobot.fleet.instance.InstanceLaunchError` is the refusal, and
it belongs to the launcher.

*The probe reads metadata only.* ``stat`` on the denied path, never ``cat``: a
``file-read*`` deny covers metadata, so a denied path cannot even be stat'd,
which makes a metadata read a sufficient probe. It is also the one that cannot
leak — a probe that read file contents would pull a peer's data through the
supervisor's own pipes, and the supervisor is the one process in the fleet that
every profile allows to see everything. No shell is involved either; the probe
tool is exec'd directly, so no path needs quoting.

*Independent of the supervisor on purpose.* This module takes instances, profiles
and a state path rather than a :class:`~nanobot.fleet.supervisor.FleetPlan`, so
that ``nanobot.fleet.supervisor`` could one day call the probe without an import
cycle. The probe runs between ``prepare_fleet`` and ``start_fleet``; it must not
be something only the layer above both can reach.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from nanobot.fleet.instance import SANDBOX_EXEC as _LAUNCHER
from nanobot.fleet.validate import ResolvedInstance

#: The Seatbelt entry point, taken from :mod:`nanobot.fleet.instance` so the
#: profile is proven with the same binary that will apply it. Rebound as a module
#: global rather than used through that module, so the name is read here at call
#: time — a host missing the launcher is one of the outcomes under test.
SANDBOX_EXEC = _LAUNCHER

#: The read probe. Absolute, for the reason ``SANDBOX_EXEC`` is: a ``PATH`` lookup
#: would let the supervisor's environment choose what "a read" means.
PROBE_TOOL = "/usr/bin/stat"

#: Ask for the size and nothing else. The point is to perform a metadata read, not
#: to learn anything — the file's contents never enter the supervisor.
PROBE_ARGUMENTS: tuple[str, ...] = ("-f", "%z")

#: The control profile: valid SBPL that denies nothing. Distinguishes "this path
#: cannot be read" from "this profile refused this read", which is the whole
#: question — a path that cannot be read either way proves nothing about a deny.
CONTROL_PROFILE = "(version 1)\n(allow default)"

#: Generous: the probe is two ``stat`` calls, but it runs on a host that is also
#: starting a fleet, and a timeout is reported as an unproven profile, so erring
#: long costs a slow start while erring short costs a false refusal to start.
DEFAULT_PROBE_TIMEOUT_SECONDS = 30.0

#: Captured output is for a human reading a refusal, so it is bounded. A profile
#: string is megabytes-capable and an error message is not expected to be long.
MAX_DETAIL_CHARACTERS = 300

#: The spawn primitive, injectable so the decision logic can be tested on any
#: platform. Must accept ``subprocess.run``'s keyword arguments.
Runner = Callable[..., Any]

#: What one read attempt found. ``"broken"`` is separate from ``"refused"``
#: because a probe that could not run is not evidence that a rule bound.
_Outcome = Literal["read", "refused", "broken"]


@dataclass(frozen=True)
class ProbeResult:
    """Whether one instance's profile was observed to refuse one denied read.

    ``confined`` is only ever ``True`` when a read was actually refused under this
    profile *and* actually permitted without it. Every other outcome — the read
    succeeded, the probe could not run, the host has no ``sandbox-exec``, the path
    was unreadable either way — is ``False`` with a reason, because an unproven
    profile and a broken profile are the same thing to an operator deciding
    whether to start a fleet.
    """

    name: str
    path: Path
    confined: bool
    reason: str

    @property
    def summary(self) -> str:
        """One line naming the instance, for a caller reporting a refusal."""
        return f"instance {self.name}: {self.reason}"


def probe_confinement(
    name: str,
    profile: str,
    denied_path: str | Path,
    *,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    run: Runner = subprocess.run,
) -> ProbeResult:
    """Run ``profile`` and report whether it really refused a read of ``denied_path``.

    Args:
        name: the instance the profile belongs to, carried through to the result
            so a caller can name it without tracking the pairing itself.
        profile: the SBPL profile from :mod:`nanobot.fleet.profile`, passed inline
            with ``-p`` exactly as the launcher passes it. Proving a profile read
            from somewhere else would prove the wrong artefact.
        denied_path: a path this profile must deny. Normally a peer's workspace;
            see :func:`probe_target`.
        timeout: seconds allowed for each of the at most two read attempts.
        run: the spawn primitive, injectable for tests.

    Returns:
        A :class:`ProbeResult`. Never raises for a host or probe problem: a
        missing ``sandbox-exec``, a missing probe tool, a failed spawn and a
        timeout are all returned as unproven results, so the caller keeps the
        policy decision.
    """
    path = Path(denied_path)
    for what, tool in (
        ("the Seatbelt launcher", SANDBOX_EXEC),
        ("the read probe", PROBE_TOOL),
    ):
        if not Path(tool).is_file():
            return ProbeResult(
                name=name,
                path=path,
                confined=False,
                reason=(
                    f"{tool} is missing, so {what} cannot be run and this "
                    f"profile's confinement cannot be proven"
                ),
            )

    attempt = _read_probe(profile, path, timeout=timeout, run=run)
    if attempt.outcome == "read":
        return ProbeResult(
            name=name,
            path=path,
            confined=False,
            reason=(
                f"reading {path} succeeded under this profile, so nothing in it "
                f"denied the read ({attempt.detail})"
            ),
        )
    if attempt.outcome == "broken":
        return ProbeResult(
            name=name,
            path=path,
            confined=False,
            reason=(
                f"the read of {path} could not be carried out, so this profile's "
                f"confinement is untested ({attempt.detail})"
            ),
        )

    control = _read_probe(CONTROL_PROFILE, path, timeout=timeout, run=run)
    if control.outcome != "read":
        return ProbeResult(
            name=name,
            path=path,
            confined=False,
            reason=(
                f"reading {path} was refused under this profile, but it is also "
                f"refused under a profile that denies nothing, so the refusal is "
                f"not evidence that any rule bound ({control.detail})"
            ),
        )
    return ProbeResult(
        name=name,
        path=path,
        confined=True,
        reason=(
            f"reading {path} was refused under this profile ({attempt.detail}) and "
            f"permitted under a profile that denies nothing"
        ),
    )


def probe_target(
    instance: ResolvedInstance,
    peers: Iterable[ResolvedInstance],
    *,
    state_path: Path,
) -> Path:
    """The denied path ``instance`` is probed against.

    The first peer's workspace, because that is the isolation an operator cares
    about and the one emitted as a ``subpath`` rule. A fleet of one has no peer,
    and falls back to the supervisor's state file, which every profile denies as a
    ``literal`` — so a single-instance fleet is still probed against a real deny
    rather than skipped, and skipping is what would let a one-instance fleet grow
    a second instance with nothing ever having tested the mechanism.

    One path per instance and not all of them: every deny in a profile comes from
    the same builder in the same run, so the question the probe answers is whether
    *this profile* binds on *this host*, not whether one of its rules is
    individually malformed — that is what :mod:`nanobot.fleet.profile` refuses to
    emit, and what its tests cover rule by rule.
    """
    for peer in peers:
        return peer.workspace
    return state_path


def probe_fleet_confinement(
    instances: Sequence[ResolvedInstance],
    profiles: Mapping[str, str],
    *,
    state_path: Path,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    run: Runner = subprocess.run,
) -> tuple[ProbeResult, ...]:
    """Probe every instance's profile, in fleet order.

    Args:
        instances: the validated fleet, normally ``FleetPlan.instances``.
        profiles: instance name to SBPL profile, normally ``FleetPlan.profiles``.
        state_path: the supervisor's state file, normally ``FleetPlan.state_path``.
        timeout: forwarded to :func:`probe_confinement`.
        run: forwarded to :func:`probe_confinement`.

    Returns:
        One result per instance. An instance with no profile in ``profiles`` yields
        an unproven result without anything being run, rather than being omitted:
        a fleet start that iterated the results would otherwise see a clean sweep
        and launch an instance for which no confinement exists at all.
    """
    results: list[ProbeResult] = []
    for instance in instances:
        peers = [peer for peer in instances if peer.name != instance.name]
        path = probe_target(instance, peers, state_path=state_path)
        profile = profiles.get(instance.name)
        if profile is None:
            results.append(
                ProbeResult(
                    name=instance.name,
                    path=path,
                    confined=False,
                    reason=(
                        "no Seatbelt profile was built for this instance, so there "
                        "is nothing to prove and nothing confining it"
                    ),
                )
            )
            continue
        results.append(
            probe_confinement(
                instance.name, profile, path, timeout=timeout, run=run
            )
        )
    return tuple(results)


def unproven(results: Iterable[ProbeResult]) -> tuple[ProbeResult, ...]:
    """The results whose confinement was not proven, in the order given."""
    return tuple(result for result in results if not result.confined)


@dataclass(frozen=True)
class _Attempt:
    """One read of the probe path under one profile."""

    outcome: _Outcome
    detail: str


def _read_probe(
    profile: str,
    path: Path,
    *,
    timeout: float,
    run: Runner,
) -> _Attempt:
    """Read ``path``'s metadata under ``profile`` and classify what happened.

    Three outcomes rather than a boolean: a spawn that failed or timed out is
    ``"broken"``, never ``"refused"``. Collapsing the two would make a host that
    cannot run ``sandbox-exec`` at all look like a host whose policy bound
    perfectly — the exact inversion this module exists to prevent.
    """
    command = [SANDBOX_EXEC, "-p", profile, PROBE_TOOL, *PROBE_ARGUMENTS, str(path)]
    try:
        completed = run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _Attempt("broken", f"{PROBE_TOOL} did not finish within {timeout:g}s")
    except (OSError, ValueError) as exc:
        detail = getattr(exc, "strerror", None) or str(exc) or type(exc).__name__
        return _Attempt("broken", f"{SANDBOX_EXEC} could not be run: {detail}")

    code = completed.returncode
    if code == 0:
        return _Attempt("read", "exit 0")
    return _Attempt("refused", _detail(code, completed))


def _detail(code: object, completed: Any) -> str:
    """Render a failed attempt compactly enough to print in a refusal."""
    streams = (
        _text(getattr(completed, "stderr", None)),
        _text(getattr(completed, "stdout", None)),
    )
    message = next((stream for stream in streams if stream), "")
    if len(message) > MAX_DETAIL_CHARACTERS:
        message = f"{message[:MAX_DETAIL_CHARACTERS]}…"
    return f"exit {code}: {message}" if message else f"exit {code}"


def _text(value: object) -> str:
    """One line of captured output, or empty if there was none."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())

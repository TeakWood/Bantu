"""Tests for the confinement self-probe.

Two halves. The first drives the decision logic over a stub spawn primitive and
runs on any platform: every way a probe can be inconclusive has to be reported as
*not proven*, and getting that wrong in the lenient direction is the one bug this
module could have — a probe that called an unrunnable host confined would be
worse than no probe at all, because the fleet would then start on the strength of
it.

The second half runs the real ``sandbox-exec`` and proves both directions through
the kernel: a profile from :mod:`nanobot.fleet.profile` is detected as binding,
and a hand-written profile with the flaw that module refuses to emit — a deny
naming a non-canonical path, or one naming a path that is not the probe target —
is detected as *not* binding. Both of those profiles are accepted by
``sandbox-exec``, which starts the process and exits 0 either way, so the probe's
verdict is the only thing that tells them apart.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from nanobot.fleet import probe as probe_module
from nanobot.fleet.config import FleetInstance
from nanobot.fleet.probe import (
    CONTROL_PROFILE,
    MAX_DETAIL_CHARACTERS,
    PROBE_ARGUMENTS,
    PROBE_TOOL,
    SANDBOX_EXEC,
    ProbeResult,
    probe_confinement,
    probe_fleet_confinement,
    probe_target,
    unproven,
)
from nanobot.fleet.profile import build_fleet_profiles
from nanobot.fleet.validate import ResolvedInstance

darwin_only = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="requires native macOS Seatbelt",
)

PROFILE = '(version 1)\n(allow default)\n(deny file-read* (subpath "/nowhere"))'


def make_instance(root: Path, name: str) -> ResolvedInstance:
    """Resolve one instance laid out the way nanobot lays one out by itself.

    ``<root>/<name>/config.json`` with ``<root>/<name>/workspace`` beneath it,
    matching the fixture in ``test_fleet_profile.py`` so a profile built here is
    the same artefact that module's tests parse.
    """
    home = root / name
    workspace = home / "workspace"
    home.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = home / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    return ResolvedInstance(
        name=name,
        entry=FleetInstance(config=str(config_path), mode="serve", memory_limit_mb=512),
        config_path=config_path,
        config_dir=home,
        workspace=workspace,
        port=None,
        port_setting="api.port",
    )


def completed(
    returncode: int, *, stdout: Any = "", stderr: Any = ""
) -> subprocess.CompletedProcess[Any]:
    """A finished spawn with the given outcome."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class StubRun:
    """A stub spawn primitive that answers by the profile it was handed.

    Records every command so a test can assert *what* was run, which is where two
    of this module's guarantees live: the probe reads metadata through no shell,
    and the control profile is a separate run with a profile that denies nothing.
    """

    def __init__(self, answer: Callable[[str], Any]) -> None:
        self._answer = answer
        self.commands: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, command: Sequence[str], **kwargs: Any) -> Any:
        self.commands.append(list(command))
        self.kwargs.append(kwargs)
        answer = self._answer(command[2])
        if isinstance(answer, BaseException):
            raise answer
        return answer

    @property
    def profiles(self) -> list[str]:
        """The profile each run was given, in order."""
        return [command[2] for command in self.commands]


def answers(*, probe: Any, control: Any = None) -> Callable[[str], Any]:
    """Answer the control profile with ``control`` and anything else with ``probe``."""
    return lambda profile: control if profile == CONTROL_PROFILE else probe


@pytest.fixture
def denied(tmp_path: Path) -> Path:
    """An existing path standing in for a peer's workspace."""
    path = (tmp_path / "peer" / "workspace").resolve()
    path.mkdir(parents=True)
    return path


# --------------------------------------------------------------------------
# The decision: refused under the profile, permitted without it
# --------------------------------------------------------------------------


def test_a_refused_read_that_succeeds_unconfined_is_proof(denied: Path) -> None:
    """The positive result, and the only shape that produces one."""
    run = StubRun(
        answers(probe=completed(1, stderr="stat: Operation not permitted"), control=completed(0))
    )

    result = probe_confinement("alpha", PROFILE, denied, run=run)

    assert result.confined is True
    assert result.name == "alpha"
    assert result.path == denied
    assert "Operation not permitted" in result.reason
    # The control run is second, and it is a different profile: a probe that ran
    # the instance's profile twice would conclude the same thing for any host.
    assert run.profiles == [PROFILE, CONTROL_PROFILE]


def test_a_successful_read_is_reported_as_not_confined(denied: Path) -> None:
    """The negative result, settled in one spawn.

    A read that succeeded needs no control: nothing in the profile denied it, and
    that is true whatever an unconfined run would have done.
    """
    run = StubRun(answers(probe=completed(0, stdout="96")))

    result = probe_confinement("alpha", PROFILE, denied, run=run)

    assert result.confined is False
    assert "nothing in it denied the read" in result.reason
    assert run.profiles == [PROFILE]


def test_a_path_refused_even_without_a_profile_proves_nothing(denied: Path) -> None:
    """Refused both ways is not evidence, and must not read as confinement.

    The case the control run exists for: the path may be gone, the probe tool may
    be broken, the host may be refusing for reasons of its own. Reporting this as
    proven is how a fleet starts unconfined believing it was tested.
    """
    run = StubRun(
        answers(
            probe=completed(1, stderr="stat: No such file or directory"),
            control=completed(1, stderr="stat: No such file or directory"),
        )
    )

    result = probe_confinement("alpha", PROFILE, denied, run=run)

    assert result.confined is False
    assert "also refused under a profile that denies nothing" in result.reason
    assert run.profiles == [PROFILE, CONTROL_PROFILE]


@pytest.mark.parametrize(
    "failure",
    [
        OSError(2, "No such file or directory"),
        subprocess.TimeoutExpired(cmd=[SANDBOX_EXEC], timeout=30.0),
        ValueError("embedded null byte"),
    ],
    ids=["spawn-failed", "timed-out", "rejected-argv"],
)
def test_a_spawn_that_never_ran_is_not_mistaken_for_a_refusal(
    denied: Path, failure: BaseException
) -> None:
    """A probe that could not be carried out is untested, not confined.

    This is the inversion that matters most. ``sandbox-exec`` failing to run and
    ``sandbox-exec`` refusing the read both end with a non-zero outcome, and only
    one of them says anything about the policy.
    """
    run = StubRun(answers(probe=failure))

    result = probe_confinement("alpha", PROFILE, denied, run=run)

    assert result.confined is False
    assert "could not be carried out" in result.reason
    # No control run: there is nothing to compare an attempt that did not happen
    # against, and a control that then succeeded would look like proof.
    assert run.profiles == [PROFILE]


def test_a_timeout_reports_the_limit_it_exceeded(denied: Path) -> None:
    run = StubRun(
        answers(probe=subprocess.TimeoutExpired(cmd=[SANDBOX_EXEC], timeout=2.5))
    )

    result = probe_confinement("alpha", PROFILE, denied, timeout=2.5, run=run)

    assert result.confined is False
    assert "did not finish within 2.5s" in result.reason
    assert run.kwargs[0]["timeout"] == 2.5


# --------------------------------------------------------------------------
# A host that cannot run the probe
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["SANDBOX_EXEC", "PROBE_TOOL"])
def test_a_missing_tool_yields_a_result_rather_than_an_exception(
    denied: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """Missing confinement tools are the caller's policy, not this module's.

    ``nanobot fleet start`` refuses to start anything; a diagnostic command may
    want to report every instance. Raising here would take that choice away, and
    the result still has to be negative — an unprovable profile and a broken one
    are the same thing to an operator.
    """
    missing = str(denied / "definitely-not-installed")
    monkeypatch.setattr(probe_module, tool, missing)
    run = StubRun(answers(probe=completed(1)))

    result = probe_confinement("alpha", PROFILE, denied, run=run)

    assert result.confined is False
    assert missing in result.reason
    assert "cannot be proven" in result.reason
    # Checked before anything is spawned, so a host without Seatbelt does not
    # produce a refusal that came from somewhere else entirely.
    assert run.commands == []


# --------------------------------------------------------------------------
# What the probe actually runs
# --------------------------------------------------------------------------


def test_the_probe_reads_metadata_directly_with_no_shell(denied: Path) -> None:
    """The command shape is part of the contract, in three ways.

    It is a metadata read, so a peer's contents never reach the supervisor — the
    one process in the fleet that every profile allows to see everything. It is
    exec'd directly, so no path needs quoting and a path cannot become a second
    command. And the profile is passed inline with ``-p``, exactly as the launcher
    passes it, so the artefact proven is the artefact used.
    """
    run = StubRun(answers(probe=completed(1), control=completed(0)))

    probe_confinement("alpha", PROFILE, denied, run=run)

    assert run.commands[0] == [
        SANDBOX_EXEC,
        "-p",
        PROFILE,
        PROBE_TOOL,
        *PROBE_ARGUMENTS,
        str(denied),
    ]
    assert all("sh" not in Path(command[3]).name for command in run.commands)
    assert all(kwargs.get("shell") is not True for kwargs in run.kwargs)
    assert all(kwargs["capture_output"] is True for kwargs in run.kwargs)


def test_the_control_profile_denies_nothing() -> None:
    """Pinned as text: a control that denied anything could refuse the probe path.

    The control run's only job is to show the path is readable when no rule stands
    in the way. A stray deny in it would turn every proof into "refused both
    ways", which reads as unproven and would stop every fleet from starting.
    """
    assert "deny" not in CONTROL_PROFILE
    assert CONTROL_PROFILE.splitlines() == ["(version 1)", "(allow default)"]


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        ("", "on stderr", "on stderr"),
        ("on stdout", "", "on stdout"),
        (b"bytes stderr", b"", "bytes stderr"),
        ("", "", "exit 1"),
        (None, None, "exit 1"),
    ],
    ids=["stderr", "stdout-fallback", "bytes", "silent", "uncaptured"],
)
def test_captured_output_is_reported(
    denied: Path, stdout: Any, stderr: Any, expected: str
) -> None:
    """Whatever the probe said reaches the reason, including from a bytes stream."""
    run = StubRun(
        answers(probe=completed(1, stdout=stdout, stderr=stderr), control=completed(0))
    )

    result = probe_confinement("alpha", PROFILE, denied, run=run)

    assert result.confined is True
    assert expected in result.reason


def test_captured_output_is_bounded(denied: Path) -> None:
    """A refusal is printed, so its detail cannot be unbounded."""
    run = StubRun(
        answers(probe=completed(1, stderr="x" * 5000), control=completed(0))
    )

    result = probe_confinement("alpha", PROFILE, denied, run=run)

    assert "x" * MAX_DETAIL_CHARACTERS in result.reason
    assert "x" * (MAX_DETAIL_CHARACTERS + 1) not in result.reason
    assert "…" in result.reason


# --------------------------------------------------------------------------
# Choosing what to probe, and probing a whole fleet
# --------------------------------------------------------------------------


def test_the_probe_target_is_the_first_peers_workspace(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    a, b, c = (make_instance(root, name) for name in ("a", "b", "c"))

    assert probe_target(a, [b, c], state_path=root / "state.json") == b.workspace


def test_a_single_instance_fleet_is_probed_against_the_state_file(tmp_path: Path) -> None:
    """A fleet of one has no peer, and is still probed rather than skipped.

    Every profile denies the state file, so there is always a real deny to test.
    Skipping is what would let a one-instance fleet grow a second instance with
    the mechanism never once having been exercised on the host.
    """
    root = tmp_path.resolve()
    state = root / "state.json"

    assert probe_target(make_instance(root, "a"), [], state_path=state) == state


def test_every_instance_is_probed_in_fleet_order(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    instances = [make_instance(root, name) for name in ("a", "b", "c")]
    profiles = {instance.name: f"{PROFILE}\n; {instance.name}" for instance in instances}
    run = StubRun(answers(probe=completed(1), control=completed(0)))

    results = probe_fleet_confinement(
        instances, profiles, state_path=root / "state.json", run=run
    )

    assert [result.name for result in results] == ["a", "b", "c"]
    assert all(result.confined for result in results)
    # Each instance is probed against a peer's workspace, never its own.
    assert [result.path for result in results] == [
        instances[1].workspace,
        instances[0].workspace,
        instances[0].workspace,
    ]


def test_an_instance_with_no_profile_is_unproven_without_being_run(tmp_path: Path) -> None:
    """The gap a per-instance loop could otherwise skip straight past.

    An instance missing from ``profiles`` has no confinement at all. Omitting it
    from the results would give a caller a clean sweep to iterate over and an
    unconfined instance to start.
    """
    root = tmp_path.resolve()
    a, b = (make_instance(root, name) for name in ("a", "b"))
    run = StubRun(answers(probe=completed(1), control=completed(0)))

    results = probe_fleet_confinement(
        [a, b], {"a": PROFILE}, state_path=root / "state.json", run=run
    )

    assert [result.name for result in results] == ["a", "b"]
    assert results[0].confined is True
    assert results[1].confined is False
    assert "no Seatbelt profile was built" in results[1].reason
    # Only alpha's two runs happened; nothing was spawned for the instance that
    # has no profile to spawn under.
    assert run.profiles == [PROFILE, CONTROL_PROFILE]


def test_unproven_keeps_only_the_results_that_were_not_proven() -> None:
    proven = ProbeResult(name="a", path=Path("/x"), confined=True, reason="proven")
    broken = ProbeResult(name="b", path=Path("/y"), confined=False, reason="not proven")

    assert unproven([proven, broken, proven]) == (broken,)
    assert unproven([]) == ()


def test_a_result_summary_names_its_instance() -> None:
    result = ProbeResult(name="alpha", path=Path("/x"), confined=False, reason="because")

    assert result.summary == "instance alpha: because"


# --------------------------------------------------------------------------
# The kernel, not just the decision table
# --------------------------------------------------------------------------


@pytest.fixture
def host_tmp_root() -> Iterator[Path]:
    """A fleet root under ``/tmp``, yielded in its non-canonical spelling.

    ``/tmp`` is a symlink to ``/private/tmp`` on macOS, which is the discrepancy
    the profile builder refuses to emit and the one this file needs in order to
    build a flawed profile on purpose. ``tmp_path`` would not do: its own
    non-canonical spelling is not guaranteed to exist.
    """
    with tempfile.TemporaryDirectory(prefix="nanobot-fleet-probe-", dir="/tmp") as raw:
        yield Path(raw)


@pytest.fixture
def native_fleet(host_tmp_root: Path) -> tuple[Path, list[ResolvedInstance], dict[str, str]]:
    """Two real instances, real supervisor files, and their real profiles."""
    root = host_tmp_root.resolve()
    fleet_path = root / "fleet.json"
    state_path = root / "state.json"
    fleet_path.write_text("{}", encoding="utf-8")
    state_path.write_text("[]", encoding="utf-8")
    instances = [make_instance(root, name) for name in ("a", "b")]
    profiles = build_fleet_profiles(
        instances, fleet_path=fleet_path, state_path=state_path
    )
    return state_path, instances, profiles


@darwin_only
def test_a_generated_profile_is_proven_to_bind(
    native_fleet: tuple[Path, list[ResolvedInstance], dict[str, str]],
) -> None:
    """The positive direction, through the real kernel policy."""
    state_path, instances, profiles = native_fleet
    a, b = instances
    (b.workspace / "sentinel").write_text("peer-secret", encoding="utf-8")

    result = probe_confinement("a", profiles["a"], b.workspace)

    assert result.confined is True, result.reason
    assert result.path == b.workspace
    assert "peer-secret" not in result.reason


@darwin_only
@pytest.mark.parametrize("flaw", ["non-canonical", "wrong-path"])
def test_a_deny_that_matches_nothing_is_detected_as_not_binding(
    host_tmp_root: Path, native_fleet: tuple[Path, list[ResolvedInstance], dict[str, str]], flaw: str
) -> None:
    """The negative direction, and the reason this module exists.

    Both of these profiles are *accepted* by ``sandbox-exec``, which then starts
    the process, confines nothing and exits 0. ``nanobot.fleet.profile`` refuses to
    emit either, so they are hand-written here; the probe's verdict is the only
    thing that distinguishes them from the profile above.
    """
    _, instances, _ = native_fleet
    peer = instances[1].workspace
    assert host_tmp_root != peer.parents[1], "expected /tmp to be a symlink"
    target = {
        # /tmp/... where the real path is /private/tmp/...
        "non-canonical": host_tmp_root / peer.relative_to(peer.parents[1]),
        "wrong-path": peer.with_name("workspace-that-does-not-exist"),
    }[flaw]
    flawed = f'(version 1)\n(allow default)\n(deny file-read* (subpath "{target}"))'

    result = probe_confinement("a", flawed, peer)

    assert result.confined is False
    assert "nothing in it denied the read" in result.reason


@darwin_only
def test_an_unreadable_probe_path_proves_nothing_natively(
    native_fleet: tuple[Path, list[ResolvedInstance], dict[str, str]],
) -> None:
    """A path the supervisor cannot read either is reported as untested.

    The kernel-level form of the control run's purpose: a probe target that has
    gone away is refused under every profile, and a probe that took the first
    refusal as proof would report a confinement it never observed.
    """
    _, instances, profiles = native_fleet
    peer = instances[1].workspace
    shutil.rmtree(peer)

    result = probe_confinement("a", profiles["a"], peer)

    assert result.confined is False
    assert "also refused under a profile that denies nothing" in result.reason


@darwin_only
def test_an_instances_own_workspace_is_not_reported_as_confined(
    native_fleet: tuple[Path, list[ResolvedInstance], dict[str, str]],
) -> None:
    """The probe measures the path it was handed, not the profile in general.

    Also the allow-default posture in one assertion: an instance keeps its own
    workspace, which the shell sandbox's ``(deny default)`` profile would take
    away. A probe that always answered "confined" would pass every other native
    test in this file and fail this one.
    """
    _, instances, profiles = native_fleet
    a = instances[0]

    result = probe_confinement("a", profiles["a"], a.workspace)

    assert result.confined is False
    assert "nothing in it denied the read" in result.reason


@darwin_only
def test_a_whole_fleet_is_proven_through_the_real_launcher(
    native_fleet: tuple[Path, list[ResolvedInstance], dict[str, str]],
) -> None:
    state_path, instances, profiles = native_fleet

    results = probe_fleet_confinement(instances, profiles, state_path=state_path)

    assert unproven(results) == ()
    assert [result.path for result in results] == [
        instances[1].workspace,
        instances[0].workspace,
    ]


@darwin_only
def test_a_single_instance_fleets_state_file_deny_binds(
    native_fleet: tuple[Path, list[ResolvedInstance], dict[str, str]],
) -> None:
    """The fallback target is a real deny, not a formality.

    The state file is denied as a ``literal`` rather than a ``subpath``, so this is
    the only native test covering that rule — and it is the rule a fleet of one is
    probed against.
    """
    state_path, instances, profiles = native_fleet
    a = instances[0]

    results = probe_fleet_confinement([a], {a.name: profiles["a"]}, state_path=state_path)

    assert [result.path for result in results] == [state_path]
    assert results[0].confined is True, results[0].reason

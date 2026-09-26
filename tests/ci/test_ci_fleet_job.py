"""The CI workflow's carve-outs for tests that cannot share an xdist worker.

Fleet tests start supervisors, launch confined instances on real ports and send
signals to whole process groups. Under ``-n auto`` two workers can bind the same
port, or one worker's group signal can reach another's children — failures that
present as unrelated flakes far from the fleet suite. The workflow therefore
runs ``tests/fleet`` in its own serial macOS job, exactly as it already does for
the Windows process-tree tests and the Rich-output CLI tests.

That arrangement is only a comment in YAML, so these tests derive the claims
from the workflow itself: which invocations use xdist, which paths each one
carves out, and whether a carved-out path is actually run somewhere else. A
matrix entry added without the fleet exclusion fails here rather than silently
reintroducing the races.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
FLEET_TESTS = "tests/fleet"

# Carve-outs that apply to every platform, so every xdist run must exclude them.
# The Windows process tests are deliberately absent: they run inside the main
# xdist job on Linux and are only carved out of the Windows entry.
UNIVERSAL_CARVE_OUTS = (FLEET_TESTS, "tests/cli/test_commands.py")


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))


def _jobs() -> dict[str, Any]:
    return _workflow()["jobs"]


def _matrix_entries(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Every matrix combination of a job, or one empty entry if it has none."""
    include = job.get("strategy", {}).get("matrix", {}).get("include")
    return list(include) if include else [{}]


def _pytest_invocations(job: dict[str, Any]) -> list[str]:
    """Every ``python -m pytest`` command a job runs, matrix values expanded.

    A step's ``run`` is folded to a single line so the workflow's YAML line
    continuations do not change how the command tokenizes.
    """
    commands: list[str] = []
    for step in job.get("steps", []):
        run = step.get("run")
        if not isinstance(run, str) or "-m pytest" not in run:
            continue
        for entry in _matrix_entries(job):
            text = run
            for key, value in entry.items():
                text = text.replace("${{ matrix." + key + " }}", str(value))
            commands.append(" ".join(text.split()))
    return commands


def _all_invocations() -> list[tuple[str, str]]:
    return [(name, command) for name, job in _jobs().items() for command in _pytest_invocations(job)]


def _ignored_paths(command: str) -> set[str]:
    return {
        token.removeprefix("--ignore=") for token in shlex.split(command) if token.startswith("--ignore=")
    }


def _target_paths(command: str) -> set[str]:
    """Positional path arguments — what the invocation actually runs."""
    return {
        token
        for token in shlex.split(command)[1:]
        if not token.startswith("-") and (REPO_ROOT / token).exists()
    }


def _uses_xdist(command: str) -> bool:
    tokens = shlex.split(command)
    return "-n" in tokens or any(token.startswith("-n") and len(token) > 2 for token in tokens)


def _fleet_job() -> tuple[str, dict[str, Any]]:
    named = [
        (name, job)
        for name, job in _jobs().items()
        if any(FLEET_TESTS in _target_paths(command) for command in _pytest_invocations(job))
    ]
    assert len(named) == 1, f"expected exactly one job running {FLEET_TESTS}, found {[n for n, _ in named]}"
    return named[0]


def test_the_workflow_is_parseable_and_runs_pytest_somewhere() -> None:
    """Guard: every other test here is vacuous if nothing is discovered."""
    invocations = _all_invocations()
    assert invocations, "no pytest invocations found in the CI workflow"
    assert any(_uses_xdist(command) for _, command in invocations)


def test_the_fleet_suite_exists_and_is_not_empty() -> None:
    """The ignored path must name real tests, or the carve-out is a typo."""
    collected = sorted((REPO_ROOT / FLEET_TESTS).glob("test_*.py"))
    assert collected, f"{FLEET_TESTS} holds no test modules"


@pytest.mark.parametrize("path", UNIVERSAL_CARVE_OUTS)
def test_every_xdist_run_excludes_the_universal_carve_outs(path: str) -> None:
    for name, command in _all_invocations():
        if not _uses_xdist(command):
            continue
        assert path in _ignored_paths(command), f"job {name!r} runs {path} under xdist: {command}"


@pytest.mark.parametrize("path", UNIVERSAL_CARVE_OUTS)
def test_every_carved_out_path_is_still_run_serially(path: str) -> None:
    """Excluding a path from xdist must not drop it from CI altogether."""
    runners = [
        name
        for name, command in _all_invocations()
        if path in _target_paths(command) and not _uses_xdist(command)
    ]
    assert runners, f"{path} is excluded from every xdist run but never run serially"


def test_the_fleet_job_runs_on_macos() -> None:
    """Seatbelt confinement is macOS-only, which is why the job is separate."""
    _, job = _fleet_job()
    assert str(job["runs-on"]).startswith("macos")

    gated = [
        path
        for path in sorted((REPO_ROOT / FLEET_TESTS).glob("test_*.py"))
        if "darwin" in path.read_text(encoding="utf-8")
    ]
    assert gated, f"{FLEET_TESTS} has no darwin-specific tests; a macOS job would prove nothing"


def test_the_fleet_job_runs_pytest_outside_xdist() -> None:
    name, job = _fleet_job()
    for command in _pytest_invocations(job):
        assert not _uses_xdist(command), f"job {name!r} runs the fleet suite under xdist: {command}"


def test_the_fleet_job_is_bounded_like_the_existing_jobs() -> None:
    """There is no pytest-timeout here, so the job-level bound is the only one."""
    name, job = _fleet_job()
    timeouts = {
        other: other_job["timeout-minutes"]
        for other, other_job in _jobs().items()
        if "timeout-minutes" in other_job
    }
    assert name in timeouts, f"job {name!r} has no timeout-minutes"
    assert timeouts[name] <= max(timeouts.values())


def test_the_fleet_job_is_gated_like_the_other_python_jobs() -> None:
    """It must skip on a docs-only change exactly as the main test job does."""
    name, job = _fleet_job()
    main = _jobs()["test"]
    assert job["needs"] == main["needs"]
    assert job["if"] == main["if"], f"job {name!r} is gated differently from the main test job"


def test_every_ignored_path_in_the_workflow_exists() -> None:
    """A renamed or deleted carve-out silently stops excluding anything."""
    for name, command in _all_invocations():
        for path in _ignored_paths(command):
            assert (REPO_ROOT / path).exists(), f"job {name!r} ignores a missing path: {path}"

"""CONTRIBUTING.md's runner-cost policy, enforced against the real workflows.

``CONTRIBUTING.md`` asks contributors to keep CI inside GitHub Actions' free
tier: standard Linux and Windows runners only, no larger or self-hosted runners,
and macOS only where a stated reason justifies the ten-times billing multiplier.
That text is unenforceable on its own — a new ``runs-on: macos-latest``, a
``group:`` naming a self-hosted pool, or an ``ubuntu-latest-8-cores`` all merge
green, and the cost shows up on a bill rather than in a diff.

These tests read the allowlist and the exception list out of the policy section
itself, so the document is the single source of truth. A workflow that steps
outside them fails here; so does an exception that is documented but no longer
exists, or one whose stated justification it does not actually run. Widening the
policy therefore means editing the prose, which is exactly the discussion the
section asks for.
"""

from __future__ import annotations

import itertools
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
POLICY_HEADING = "## Modifying CI Workflows"
ALLOWLIST_MARKER = "standard GitHub-hosted runners"

_BACKTICKED = re.compile(r"`([^`]+)`")
_MATRIX_REF = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}")
_EXCEPTION_BULLET = re.compile(
    r"^- `(?P<job>[A-Za-z0-9_-]+)` in `(?P<workflow>\.github/workflows/[^`]+)` on `(?P<label>[^`]+)`"
)


# --- the policy document ---------------------------------------------------


def _policy_section() -> str:
    """The body of the ``Modifying CI Workflows`` section, heading excluded."""
    text = CONTRIBUTING.read_text(encoding="utf-8")
    start = text.index(POLICY_HEADING) + len(POLICY_HEADING)
    rest = text[start:]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def _allowed_labels() -> set[str]:
    section = _policy_section()
    lines = [line for line in section.splitlines() if ALLOWLIST_MARKER in line]
    assert lines, f"the policy section no longer names the {ALLOWLIST_MARKER!r} allowlist"
    return {label for line in lines for label in _BACKTICKED.findall(line) if "*" not in label}


def _forbidden_globs() -> set[str]:
    """Runner shapes the policy spells as globs, e.g. ``*-cores``."""
    return {token for token in _BACKTICKED.findall(_policy_section()) if "*" in token}


def _documented_exceptions() -> list[dict[str, Any]]:
    """Every approved exception bullet, with the paths it offers as justification."""
    exceptions: list[dict[str, Any]] = []
    for bullet in _bullets(_policy_section()):
        match = _EXCEPTION_BULLET.match(bullet)
        if match is None:
            continue
        workflow = match["workflow"]
        justification = {
            token
            for token in _BACKTICKED.findall(bullet)
            if token != workflow and (REPO_ROOT / token).exists()
        }
        exceptions.append(
            {
                "job": match["job"],
                "workflow": workflow,
                "label": match["label"],
                "justification": justification,
            }
        )
    return exceptions


def _bullets(section: str) -> list[str]:
    """Top-level list items, continuation lines folded back onto their bullet."""
    bullets: list[str] = []
    for line in section.splitlines():
        if line.startswith("- "):
            bullets.append(line)
        elif bullets and line.startswith("  ") and line.strip():
            bullets[-1] += " " + line.strip()
        else:
            bullets.append("")
    return [bullet for bullet in bullets if bullet]


# --- the workflows ---------------------------------------------------------


def _workflow_files() -> list[Path]:
    return sorted(path for path in WORKFLOW_DIR.iterdir() if path.suffix in {".yml", ".yaml"})


def _matrix_values(job: dict[str, Any], key: str) -> list[str]:
    matrix = job.get("strategy", {}).get("matrix", {})
    values = [str(value) for value in matrix.get(key, []) if not isinstance(value, dict)]
    values += [str(entry[key]) for entry in matrix.get("include") or [] if key in entry]
    return values


def _declared_labels(runs_on: Any) -> list[str]:
    """Labels named by a ``runs-on``, in any of its three accepted shapes."""
    if isinstance(runs_on, str):
        return [runs_on]
    if isinstance(runs_on, list):
        return [str(item) for item in runs_on]
    if isinstance(runs_on, dict):
        # A runner group is a self-hosted pool; its name is the label that matters.
        labels = runs_on.get("labels") or []
        group = [runs_on["group"]] if "group" in runs_on else []
        return [str(item) for item in (*group, *(labels if isinstance(labels, list) else [labels]))]
    return [str(runs_on)]


def _expand(label: str, job: dict[str, Any]) -> list[str]:
    """Every concrete label a ``matrix``-templated ``runs-on`` can resolve to."""
    keys = list(dict.fromkeys(_MATRIX_REF.findall(label)))
    if not keys:
        return [label]
    choices = [_matrix_values(job, key) for key in keys]
    if not all(choices):
        return [label]  # unresolvable; the vacuity guard below reports it
    expanded: list[str] = []
    for combination in itertools.product(*choices):
        text = label
        for key, value in zip(keys, combination, strict=True):
            text = re.sub(r"\$\{\{\s*matrix\." + re.escape(key) + r"\s*\}\}", value, text)
        expanded.append(text)
    return expanded


def _jobs() -> list[tuple[str, str, dict[str, Any]]]:
    """``(workflow path, job name, job)`` for every job in every workflow."""
    found: list[tuple[str, str, dict[str, Any]]] = []
    for path in _workflow_files():
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        relative = path.relative_to(REPO_ROOT).as_posix()
        for name, job in (workflow.get("jobs") or {}).items():
            found.append((relative, name, job))
    return found


def _runners() -> list[tuple[str, str, str]]:
    """``(workflow path, job name, concrete runner label)`` across all workflows."""
    return [
        (path, name, label)
        for path, name, job in _jobs()
        if "runs-on" in job
        for declared in _declared_labels(job["runs-on"])
        for label in _expand(declared, job)
    ]


def _step_text(job: dict[str, Any]) -> str:
    return "\n".join(
        str(value)
        for step in job.get("steps", [])
        for key, value in step.items()
        if isinstance(value, str) and key in {"run", "name", "working-directory"}
    )


def _job(workflow: str, name: str) -> dict[str, Any] | None:
    for path, other, job in _jobs():
        if path == workflow and other == name:
            return job
    return None


# --- guards ----------------------------------------------------------------


def test_the_workflow_directory_holds_jobs_with_runners() -> None:
    """Guard: every policy test below is vacuous if nothing is discovered."""
    assert _workflow_files(), "no workflow files found"
    runners = _runners()
    assert runners, "no runs-on found in any workflow"
    assert {path for path, _, _ in runners} == {
        path.relative_to(REPO_ROOT).as_posix() for path in _workflow_files()
    }, "a workflow file contributed no runner; its jobs were not parsed"


def test_the_policy_section_is_machine_readable() -> None:
    """Guard: a reworded section must not silently stop constraining anything."""
    assert _allowed_labels() == {"ubuntu-latest", "windows-latest"}
    assert _forbidden_globs(), "the policy no longer names any forbidden runner shape"
    assert _documented_exceptions(), "the policy lists no approved exception"


def test_every_runner_label_resolves_to_a_concrete_runner() -> None:
    """An unexpanded ``${{ … }}`` would pass every allowlist check vacuously."""
    for path, name, label in _runners():
        assert "${{" not in label, f"{path}: job {name!r} has an unresolved runs-on: {label}"
        assert label.strip() == label and label, f"{path}: job {name!r} has a blank runs-on"


# --- the policy ------------------------------------------------------------


def test_every_runner_is_allowlisted_or_a_documented_exception() -> None:
    allowed = _allowed_labels()
    exceptions = {(e["workflow"], e["job"], e["label"]) for e in _documented_exceptions()}
    for path, name, label in _runners():
        if label in allowed or (path, name, label) in exceptions:
            continue
        pytest.fail(
            f"{path}: job {name!r} runs on {label!r}, which is neither on "
            f"CONTRIBUTING.md's allowlist {sorted(allowed)} nor a documented exception. "
            "Add it to the 'Modifying CI Workflows' section or move the job."
        )


def test_no_job_uses_a_larger_or_self_hosted_runner() -> None:
    """Forbidden outright: no exception bullet can approve a paid runner shape."""
    globs = _forbidden_globs()
    for path, name, label in _runners():
        assert label != "self-hosted", f"{path}: job {name!r} uses a self-hosted runner"
        for glob in globs:
            assert not fnmatch(label, glob), f"{path}: job {name!r} runs on {label!r} ({glob})"


def test_every_documented_exception_still_exists() -> None:
    """A stale bullet would keep approving a runner nothing uses any more."""
    for exception in _documented_exceptions():
        workflow, name = exception["workflow"], exception["job"]
        assert (REPO_ROOT / workflow).exists(), f"documented exception names a missing {workflow}"
        job = _job(workflow, name)
        assert job is not None, f"CONTRIBUTING.md documents {name!r} in {workflow}, which has no such job"
        labels = [label for p, n, label in _runners() if (p, n) == (workflow, name)]
        assert labels == [exception["label"]], (
            f"CONTRIBUTING.md documents {name!r} on {exception['label']!r}, but it runs on {labels}"
        )


def test_every_documented_exception_runs_what_it_claims_to_need() -> None:
    """The stated justification must be a path the job actually exercises."""
    for exception in _documented_exceptions():
        justification = exception["justification"]
        assert justification, (
            f"the exception for {exception['job']!r} names no existing path as its reason; "
            "cite the tests that need this runner"
        )
        job = _job(exception["workflow"], exception["job"])
        assert job is not None
        steps = _step_text(job)
        for path in sorted(justification):
            assert path in steps, (
                f"{exception['job']!r} is approved for {path!r}, but no step mentions it"
            )


def test_no_undocumented_exception_is_needed() -> None:
    """Both directions: the exception list is exactly the off-allowlist jobs."""
    allowed = _allowed_labels()
    off_allowlist = {
        (path, name, label) for path, name, label in _runners() if label not in allowed
    }
    documented = {(e["workflow"], e["job"], e["label"]) for e in _documented_exceptions()}
    assert off_allowlist == documented, (
        "CONTRIBUTING.md's exception list and the workflows disagree; "
        f"undocumented: {sorted(off_allowlist - documented)}, stale: {sorted(documented - off_allowlist)}"
    )

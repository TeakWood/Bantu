"""Acceptance criterion 1: no fleet, no change.

Half of this criterion cannot be asserted by a test at all. "The existing suite
passes" is the *harness's* gate — a test that claimed it about its own suite
would be asserting its own result. So what is pinned here is the structural claim
the criterion rests on, which is checkable and which is the thing a future edit
could quietly break: the fleet is **additive**. It is a new package plus a new
command group, and nothing an instance-less nanobot does goes anywhere near it.

The claim is pinned four independent ways, because each one fails for a different
kind of edit:

*No core module mentions the fleet.* The epic names six files it does not modify
— the config schema, the agent loop and runner, the shell tool, the shell
sandbox, and ``process_runtime`` — and the cheapest complete check that none of
them grew a fleet branch is that the word does not occur in them. Coarse on
purpose: a conditional, an import, a comment or a flag all trip it.

*Importing those modules loads no fleet module.* The stronger form of the same
claim, and the one that survives a rename: what matters is not the spelling but
that the fleet package is unreachable from the agent's own import graph. The
probe runs in a subprocess (this pytest session has already imported nearly
everything) and ends by importing a fleet module itself, so a probe that could no
longer see one fails instead of reporting a clean sweep.

*The only edge into the fleet package is the CLI mount.* An AST sweep over every
module under ``nanobot/``: exactly one file may import ``nanobot.fleet``, exactly
one file may import that file, and in the other direction the fleet package may
not reach into ``nanobot.agent`` or the config schema. That is the whole
dependency contract the epic's design states, expressed so that a future import
has to break a test rather than a convention.

*Mounting the group and naming the role change nothing else.* ``fleet`` is one
more command beside the existing ones and one more entry in ``_ROLES``; every
other command still comes from its own module and every other role still produces
the process title it produced before.

One runtime check closes it: the whole CLI — fleet group mounted, fleet package
imported — is loaded in a subprocess with a throwaway ``HOME``, and neither that
directory nor the working directory gains a single entry. "No fleet, no change"
includes leaving no trace of a fleet on a machine that never asked for one.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import typer

from nanobot.cli.commands import app as root_app
from nanobot.cli.process_identity import _ROLES, set_cli_process_identity

#: The repository root, three levels up from ``tests/fleet/<this file>``.
REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "nanobot"
FLEET_ROOT = PACKAGE_ROOT / "fleet"

#: The files the epic promises the fleet does not modify. Read from disk rather
#: than imported: the claim is about their text, not their behaviour.
CORE_MODULES: tuple[str, ...] = (
    "config/schema.py",
    "agent/loop.py",
    "agent/runner.py",
    "agent/tools/shell.py",
    "agent/tools/sandbox.py",
    "process_runtime.py",
)

#: Imported by the probe below, in place of "the agent path". Every module the
#: epic promises is untouched, plus the two entry points that would drag the
#: whole agent tree in behind them.
AGENT_IMPORTS: tuple[str, ...] = (
    "nanobot.config.schema",
    "nanobot.agent.loop",
    "nanobot.agent.runner",
    "nanobot.agent.tools.shell",
    "nanobot.agent.tools.sandbox",
    "nanobot.process_runtime",
    "nanobot.nanobot",
)

#: The one fleet module the probe imports last, to prove it can see one at all.
CONTROL_IMPORT = "nanobot.fleet.supervisor"

#: The single file allowed to import the fleet package, and the single file
#: allowed to import *it*.
FLEET_CLI = "cli/fleet.py"
FLEET_CLI_MODULE = "nanobot.cli.fleet"
CLI_COMMANDS = "cli/commands.py"

#: Roles that named a process before the fleet existed. ``fleet`` is checked
#: separately: the point is that adding it moved none of these.
PRIOR_ROLES: tuple[str, ...] = ("agent", "gateway", "webui")

#: Commands the root app exposed before the fleet group was mounted.
PRIOR_COMMANDS: frozenset[str] = frozenset({
    "agent",
    "channels",
    "gateway",
    "onboard",
    "plugins",
    "provider",
    "serve",
    "sessions",
    "status",
    "trigger",
    "webui",
})

SUBPROCESS_TIMEOUT_SECONDS = 120.0


def python_modules(root: Path) -> tuple[Path, ...]:
    """Every Python module under *root*, sorted, so a sweep is reproducible."""
    return tuple(sorted(root.rglob("*.py")))


def imported_names(path: Path) -> set[str]:
    """Every module path *path* imports, absolute forms only.

    ``from a.b import c`` contributes both ``a.b`` and ``a.b.c`` so that a
    package can be spotted whether it is imported by name or reached through its
    parent. Relative imports contribute nothing: they cannot name another
    top-level package, which is what the sweeps below are about.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def imports_package(names: set[str], package: str) -> bool:
    """Whether *names* reaches ``package`` itself or anything inside it."""
    return any(name == package or name.startswith(f"{package}.") for name in names)


def run_python(
    code: str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run *code* under this interpreter, from outside the pytest session.

    Every claim about the import graph has to be made in a fresh process: by the
    time a test runs, the session has already imported the fleet package several
    times over, so ``sys.modules`` here proves nothing.
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        cwd=cwd,
        env=env or clean_environment(Path.home()),
    )


def clean_environment(home: Path) -> dict[str, str]:
    """This process's environment with a throwaway ``HOME`` and no inherited state.

    Three families are dropped. ``NANOBOT_*`` because :class:`nanobot.config.Config`
    is a ``BaseSettings`` that reads them, so a developer with one exported could
    otherwise make a subprocess behave differently from a clean machine.
    ``COV_CORE_*`` and ``COVERAGE_*`` because ``pytest-cov`` asks every child
    process to start measuring — and a child started outside the repository finds
    no ``pyproject.toml``, so it measures with none of the configured ``omit``
    rules and its data quietly changes the whole suite's coverage denominator.
    These children are probes of import behaviour, not part of the measurement.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NANOBOT_", "COV_CORE_", "COVERAGE_"))
    }
    env["HOME"] = str(home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def titles_for(args: list[str], monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The process titles :func:`set_cli_process_identity` would set for *args*."""
    titles: list[str] = []
    monkeypatch.setattr("nanobot.cli.process_identity.os.name", "posix")
    monkeypatch.setattr(
        "nanobot.cli.process_identity._set_process_title", titles.append
    )
    set_cli_process_identity(args)
    return titles


# ----------------------------------------------------------------------------
# The core modules the epic promises are untouched
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("relative", CORE_MODULES)
def test_a_core_module_carries_no_reference_to_the_fleet(relative: str) -> None:
    """None of the six files the epic names knows the fleet exists.

    Deliberately a substring search rather than an import check: a fleet-shaped
    edit to any of these would arrive as a conditional, a flag, a comment or an
    import, and all four spell the word.
    """
    path = PACKAGE_ROOT / relative

    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"{relative} is empty, so this check proves nothing"
    assert "fleet" not in text.lower(), f"{relative} refers to the fleet"


def test_the_agent_path_imports_no_fleet_module() -> None:
    """A nanobot with no fleet never loads a line of the fleet package.

    The probe's last two statements are its own positive control: it imports a
    fleet module and asserts it then appears. Without that, a probe whose earlier
    imports had silently stopped working would report an empty sweep and pass.
    """
    probe = "\n".join([
        "import json, sys",
        *(f"import {module}" for module in AGENT_IMPORTS),
        "loaded = lambda: sorted(m for m in sys.modules if m.startswith('nanobot.fleet'))",
        "before = loaded()",
        f"import {CONTROL_IMPORT}",
        "print(json.dumps({'before': before, 'after': loaded()}))",
    ])

    result = run_python(probe)

    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert seen["before"] == [], f"the agent path loaded {seen['before']}"
    assert CONTROL_IMPORT in seen["after"], "the probe cannot see a fleet module"


def test_loading_the_whole_cli_leaves_no_trace_on_the_filesystem(
    tmp_path: Path,
) -> None:
    """Importing the CLI — fleet group and all — creates nothing anywhere.

    The command group is mounted at import time, so this is the one place the
    fleet is guaranteed to be reached on a machine that never declared one. An
    import that created a directory, read a default config or laid down a state
    file would be a change to a nanobot with no fleet.
    """
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()

    result = run_python(
        f"import nanobot.cli.commands, {FLEET_CLI_MODULE}, {CONTROL_IMPORT}",
        cwd=work,
        env=clean_environment(home),
    )

    assert result.returncode == 0, result.stderr
    assert list(home.iterdir()) == [], "importing the CLI wrote into HOME"
    assert list(work.iterdir()) == [], "importing the CLI wrote into the cwd"


def test_the_cli_still_answers_without_a_fleet(tmp_path: Path) -> None:
    """The contact point an operator with no fleet actually uses still works.

    ``COLUMNS`` is pinned because the help is rendered by ``rich``, which wraps
    to the terminal and would otherwise split a command name across lines.
    """
    home = tmp_path / "home"
    home.mkdir()

    result = subprocess.run(
        [sys.executable, "-m", "nanobot", "--help"],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        cwd=tmp_path,
        env=clean_environment(home) | {"COLUMNS": "200"},
    )

    assert result.returncode == 0, result.stderr
    for name in ("serve", "agent", "gateway", "fleet"):
        assert name in result.stdout
    assert list(home.iterdir()) == [], "asking for help wrote into HOME"


# ----------------------------------------------------------------------------
# The dependency contract: one edge in, none out
# ----------------------------------------------------------------------------


def test_only_the_fleet_cli_module_imports_the_fleet_package() -> None:
    """Exactly one module under ``nanobot/`` reaches into ``nanobot.fleet``.

    The sweep covers the whole package, including the channel packages, so a
    future feature that grew a fleet dependency anywhere has to fail this rather
    than merely break a convention.
    """
    importers = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in python_modules(PACKAGE_ROOT)
        if not path.is_relative_to(FLEET_ROOT)
        and imports_package(imported_names(path), "nanobot.fleet")
    }

    assert importers == {FLEET_CLI}


def test_only_the_root_command_imports_the_fleet_cli_module() -> None:
    """The mount is the single edge, and it is where the epic said it would be."""
    importers = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in python_modules(PACKAGE_ROOT)
        if path != PACKAGE_ROOT / FLEET_CLI
        and imports_package(imported_names(path), FLEET_CLI_MODULE)
    }

    assert importers == {CLI_COMMANDS}


def test_the_fleet_package_does_not_reach_into_the_agent_or_the_schema() -> None:
    """The other direction: the fleet composes with nanobot, it does not extend it.

    ``nanobot.agent`` is excluded because the design forbids growing the core
    path, and ``nanobot.config.schema`` because the fleet is not allowed to add
    settings to it — the defaults it needs are duplicated and pinned by their own
    tests instead, so that importing a fleet module cannot drag the agent tool
    tree in behind it.
    """
    modules = python_modules(FLEET_ROOT)
    assert modules, "the fleet package has no modules, so this proves nothing"

    for path in modules:
        names = imported_names(path)
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        assert not imports_package(names, "nanobot.agent"), relative
        assert not imports_package(names, "nanobot.config.schema"), relative


def test_the_fleet_package_uses_only_public_process_runtime_helpers() -> None:
    """``process_runtime`` is composed with through its public surface only.

    The epic's own wording. A private helper would couple the fleet to an
    implementation detail of the module it promised not to modify — at which
    point "not modified" stops meaning "unaffected".
    """
    import nanobot.process_runtime as process_runtime

    borrowed: set[str] = set()
    for path in python_modules(FLEET_ROOT):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "nanobot.process_runtime"
            ):
                borrowed.update(alias.name for alias in node.names)

    assert borrowed, "no fleet module borrows from process_runtime at all"
    for name in sorted(borrowed):
        assert not name.startswith("_"), f"{name} is private to process_runtime"
        assert hasattr(process_runtime, name), name


# ----------------------------------------------------------------------------
# The two files the fleet does touch, touched additively
# ----------------------------------------------------------------------------


def test_the_fleet_group_is_mounted_beside_the_existing_commands() -> None:
    """Every command the root app had is still there, and still its own.

    The second half is the one worth pinning: a mount that shadowed an existing
    command would leave the name in place while quietly re-pointing it at the
    fleet module.
    """
    command = typer.main.get_command(root_app)
    commands = getattr(command, "commands", {})

    assert "fleet" in commands
    assert PRIOR_COMMANDS <= set(commands), PRIOR_COMMANDS - set(commands)
    for name, sub in commands.items():
        if name == "fleet":
            continue
        module = getattr(sub.callback, "__module__", "")
        assert not module.startswith(FLEET_CLI_MODULE), f"{name} came from the fleet"


def test_the_fleet_role_is_added_without_renaming_any_other_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_ROLES`` gained an entry; no existing role changed what it produces."""
    assert "fleet" in _ROLES
    assert set(PRIOR_ROLES) <= _ROLES

    assert titles_for(["fleet", "start"], monkeypatch) == ["nanobot-fleet"]
    for role in PRIOR_ROLES:
        assert titles_for([role], monkeypatch) == [f"nanobot-{role}"]


@pytest.mark.parametrize("args", [[], ["serve"], ["status"], ["--help"], ["fleeting"]])
def test_a_command_that_is_not_a_role_is_still_named_plain_nanobot(
    args: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The role lookup is exact, so ``fleet`` did not widen it.

    ``fleeting`` is the case that matters: a prefix or substring match would
    title an unrelated command after the supervisor.
    """
    assert titles_for(args, monkeypatch) == ["nanobot"]


# ----------------------------------------------------------------------------
# The sweeps' own machinery
# ----------------------------------------------------------------------------


def test_an_import_of_a_subpackage_is_attributed_to_the_package(
    tmp_path: Path,
) -> None:
    module = tmp_path / "m.py"
    module.write_text(
        "from nanobot.fleet.state import read_fleet_state\n", encoding="utf-8"
    )

    assert imports_package(imported_names(module), "nanobot.fleet")


def test_a_package_reached_through_its_parent_is_still_attributed_to_it(
    tmp_path: Path,
) -> None:
    """``from nanobot import fleet`` names the package without spelling its path."""
    module = tmp_path / "m.py"
    module.write_text("from nanobot import fleet\n", encoding="utf-8")

    assert imports_package(imported_names(module), "nanobot.fleet")


def test_a_plain_import_of_the_package_is_detected(tmp_path: Path) -> None:
    module = tmp_path / "m.py"
    module.write_text("import nanobot.fleet.supervisor as s\n", encoding="utf-8")

    assert imports_package(imported_names(module), "nanobot.fleet")


def test_a_similarly_named_package_is_not_mistaken_for_it(tmp_path: Path) -> None:
    """``nanobot.fleetwood`` is not the fleet, and neither is ``fleet`` alone."""
    module = tmp_path / "m.py"
    module.write_text(
        "import nanobot.fleetwood\nfrom fleet import thing\n", encoding="utf-8"
    )

    assert not imports_package(imported_names(module), "nanobot.fleet")


def test_a_relative_import_cannot_be_mistaken_for_another_package(
    tmp_path: Path,
) -> None:
    """``from .fleet import x`` inside some other package is not this one."""
    module = tmp_path / "m.py"
    module.write_text("from .fleet import x\nfrom ..fleet import y\n", encoding="utf-8")

    assert imported_names(module) == set()


def test_the_module_sweep_finds_every_file_in_the_fleet_package() -> None:
    """The sweeps are only as good as the file list they run over."""
    found = {path.name for path in python_modules(FLEET_ROOT)}

    assert {"__init__.py", "supervisor.py", "validate.py", "profile.py"} <= found

"""Acceptance criterion 7: an overlapping fleet is refused, and nothing is spawned.

Asked the way the criterion words it: the real ``nanobot fleet start`` command, in
its own process, against a real fleet document whose second instance has its
workspace inside the first's — then ``ps``, from outside, to see whether anything
was started.

Running the command as a subprocess rather than through ``CliRunner`` is the whole
point of this file existing beside ``tests/cli/test_fleet_commands.py``, which
already drives the same gate in-process. "No instance process exists afterwards"
is a claim about the operating system, and an in-process runner cannot make it:
the instances would be children of the test session itself, so nothing would
distinguish a fleet that refused from one that started two instances and then
happened to have them reaped before the assertions ran.

Two things make the ``ps`` sweep worth trusting.

*It looks for a spawned instance both ways it could appear.* An instance's argv is
``sandbox-exec … env -i … python -m nanobot serve --config <path>``, which carries
this fleet's own root directory — but only until the instance names itself, at
which point (verified on the pinned host) ``setproctitle`` replaces the argv
entirely and ``ps`` shows the bare word ``nanobot``. Either form is evidence, so
both are searched, and the comparison is against a snapshot taken before the
command ran so that a long-lived unrelated process cannot be mistaken for one of
ours. Matching the word ``nanobot`` anywhere in an argv is deliberately wider than
it needs to be: the two mistakes are not symmetric, since a false positive fails
loudly and gets investigated while a false negative passes this criterion while a
fleet is running.

*Its own sensitivity is tested.* Two tests spawn a decoy — one carrying this
fleet's root in its argv, one titled exactly as an instance would title itself —
and assert the sweep finds it. Without them, a sweep that had silently stopped
working (a renamed title, an unparsed ``ps`` line) would report "nothing was
spawned" for every fleet, including one that started everything.

The filesystem is checked alongside the process table, because the two fail
independently: an instance that was spawned and died before ``ps`` ran would still
have left the workspace and the log directory that ``prepare_fleet`` creates
before any process exists. Comparing the whole tree before and after means the
state file, the profiles and the log directory are all covered without naming any
of them.

This file needs no Seatbelt and no stub LLM: validation is the *first* gate, so
the refusal happens before anything platform-specific is reached. It is gated only
on having a POSIX ``ps`` to ask.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

#: The process table is read through the real tool, not through a library: the
#: criterion says ``ps``, and what an operator would see is the point.
PS = "/bin/ps"

has_ps = pytest.mark.skipif(
    os.name != "posix" or not Path(PS).is_file(),
    reason="the criterion is asserted through a POSIX ps",
)

#: Common to every name a nanobot process gives itself, which is what the sweep
#: can rely on. Worth being precise about, because the obvious guess is wrong:
#: ``serve`` is *not* in ``_ROLES``, so a ``serve`` instance titles itself the bare
#: word ``nanobot`` rather than ``nanobot-serve`` — verified on this host, where
#: the title replaces the argv and the ``--config`` path disappears with it. A
#: sweep looking for ``nanobot-serve`` would therefore find no instance at all.
NANOBOT_MARK = "nanobot"

COMMAND_TIMEOUT_SECONDS = 120.0
DECOY_TIMEOUT_SECONDS = 30.0
APPEAR_TIMEOUT_SECONDS = 20.0

#: Ports are explicit because every ``serve`` instance binds ``api.port`` and a
#: fleet whose instances collide on it is refused for a reason this file is not
#: about.
ALPHA_PORT = 8911
BETA_PORT = 8912


@dataclass(frozen=True)
class Process:
    """One line of the process table: a pid and the argv ``ps`` reported."""

    pid: int
    args: str


@dataclass(frozen=True)
class Attempt:
    """What one ``fleet start`` did, and what it left behind.

    ``output`` is stdout and stderr joined with runs of whitespace collapsed:
    refusals are rendered by ``rich``, which wraps to the terminal width, so a
    message can be split mid-sentence and an unnormalised assertion on it would
    be a test of the terminal.
    """

    returncode: int
    output: str
    appeared: tuple[Process, ...]
    changed: tuple[str, ...]


def parse_processes(text: str) -> tuple[Process, ...]:
    """Parse ``pid args`` lines, ignoring anything that is not one.

    A line whose first word is not a pid is dropped rather than guessed at — the
    alternative is inventing a process, which in a sweep whose job is to find
    processes would be the more dangerous mistake.
    """
    found: list[Process] = []
    for line in text.splitlines():
        pid, _, args = line.strip().partition(" ")
        if not pid.isdigit() or not args.strip():
            continue
        found.append(Process(pid=int(pid), args=args.strip()))
    return tuple(found)


def processes() -> tuple[Process, ...]:
    """Every process on this host, with its full argv.

    ``-ww`` matters: without it macOS truncates the argv to the terminal width,
    and an instance's ``--config`` path is at the far end of a long command line.
    """
    result = subprocess.run(
        [PS, "-eww", "-o", "pid=,args="],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=True,
    )
    return parse_processes(result.stdout)


def suspects(found: Iterable[Process], marker: str) -> tuple[Process, ...]:
    """The processes that could be part of the fleet under *marker*.

    Two independent shapes, because an instance passes through both: an argv
    still naming this fleet's own directory, and an argv already replaced by the
    title the instance gives itself. Only ever applied to processes that were not
    running before, which is what lets the second shape be as broad as it is.
    """
    return tuple(
        process
        for process in found
        if marker in process.args or NANOBOT_MARK in process.args
    )


def appeared_since(before: Iterable[Process], marker: str) -> tuple[Process, ...]:
    """Fleet-shaped processes that are running now and were not running before.

    Diffing by pid is what keeps the sweep honest on a developer's machine, where
    an editor or an agent may hold the word ``nanobot-serve`` in its own command
    line for the whole session.
    """
    known = {process.pid for process in before}
    return tuple(
        process for process in suspects(processes(), marker) if process.pid not in known
    )


def tree(root: Path) -> dict[str, str]:
    """Every path under *root*, and what kind of thing it is.

    Paths rather than contents: what a started instance leaves behind is new
    *entries* — a workspace, a ``logs`` directory, a profile, the supervisor's
    state file — so comparing the entry set covers all of them without this test
    having to know their names.
    """
    return {
        path.relative_to(root).as_posix(): "dir" if path.is_dir() else "file"
        for path in sorted(root.rglob("*"))
    }


def describe_changes(before: dict[str, str], after: dict[str, str]) -> tuple[str, ...]:
    """Every difference between two trees, as lines fit for a failure message."""
    changes: list[str] = []
    for relative in sorted(set(before) | set(after)):
        was, now = before.get(relative), after.get(relative)
        if was is None:
            changes.append(f"created: {relative}")
        elif now is None:
            changes.append(f"removed: {relative}")
        elif was != now:
            changes.append(f"changed: {relative}")
    return tuple(changes)


def write_instance(root: Path, name: str, *, port: int, workspace: Path) -> Path:
    """Lay down one instance config naming *workspace*, and return its path."""
    config_dir = root / name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps({
            "api": {"port": port},
            "agents": {"defaults": {"workspace": str(workspace)}},
        }),
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
                    "memoryLimitMb": 512,
                }
                for name, path in configs.items()
            }
        }),
        encoding="utf-8",
    )
    return fleet_path


def child_environment(**overrides: str) -> dict[str, str]:
    """This process's environment, cleaned of everything a child must not inherit.

    ``NANOBOT_*`` is dropped because :class:`nanobot.config.Config` reads those, so
    a developer with one exported could make a child behave differently from a
    clean machine. ``COV_CORE_*`` and ``COVERAGE_*`` are dropped because
    ``pytest-cov`` asks every child process to start measuring — and a child
    started outside the repository finds no ``pyproject.toml``, so it measures with
    none of the configured ``omit`` rules and its data quietly changes the whole
    suite's coverage denominator. Nothing spawned here is part of the measurement.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NANOBOT_", "COV_CORE_", "COVERAGE_"))
    }
    return env | overrides


def attempt_start(root: Path, fleet_path: Path) -> Attempt:
    """Run the real command against *fleet_path* and record what it did.

    ``HOME`` is a throwaway directory inside *root* and every ``NANOBOT_``
    variable is dropped, so nothing here can reach the developer's own nanobot and
    nothing the developer exported can change what the fleet resolves to. The
    ``COV_CORE_*``/``COVERAGE_*`` variables go too — see
    :func:`child_environment`.
    """
    home = root / "home"
    home.mkdir(exist_ok=True)
    env = child_environment(
        HOME=str(home), COLUMNS="200", PYTHONDONTWRITEBYTECODE="1"
    )

    before_tree = tree(root)
    before_processes = processes()
    result = subprocess.run(
        [sys.executable, "-m", "nanobot", "fleet", "start", "--fleet", str(fleet_path)],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        cwd=root,
        env=env,
    )
    appeared = appeared_since(before_processes, root.name)

    return Attempt(
        returncode=result.returncode,
        output=" ".join(f"{result.stdout}\n{result.stderr}".split()),
        appeared=appeared,
        changed=describe_changes(before_tree, tree(root)),
    )


def assert_refused_without_starting_anything(
    attempt: Attempt, *names: str
) -> None:
    """The criterion, in the three parts it names.

    Non-zero, both instances named, and nothing spawned — asserted against the
    process table and against the filesystem, which fail independently: an
    instance that started and died would be gone from ``ps`` but would have left
    the directories ``prepare_fleet`` creates before any process exists.
    """
    assert attempt.returncode != 0, attempt.output
    for name in names:
        assert f"instances.{name}" in attempt.output, attempt.output
    assert attempt.appeared == (), f"a process was spawned: {attempt.appeared}"
    assert attempt.changed == (), f"the fleet left something behind: {attempt.changed}"


def wait_for_suspect(marker: str, before: Iterable[Process]) -> tuple[Process, ...]:
    """Poll until the sweep sees a new fleet-shaped process, or give up.

    There is no ``pytest-timeout`` in this repo, so the wait carries its own
    deadline — the pattern ``tests/webui/test_gateway_webui_smoke.py`` sets.
    """
    known = tuple(before)
    deadline = time.monotonic() + APPEAR_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        found = appeared_since(known, marker)
        if found:
            return found
        time.sleep(0.1)
    return ()


def wait_for_file(path: Path) -> None:
    """Block until *path* exists, or fail saying it never did."""
    deadline = time.monotonic() + APPEAR_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    pytest.fail(f"{path} never appeared")


def stop(decoy: subprocess.Popen[bytes]) -> None:
    """Take a decoy down and wait for it, so no test leaks a process."""
    decoy.terminate()
    try:
        decoy.wait(timeout=DECOY_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        decoy.kill()
        decoy.wait(timeout=DECOY_TIMEOUT_SECONDS)


# ----------------------------------------------------------------------------
# The criterion
# ----------------------------------------------------------------------------


@has_ps
def test_a_nested_workspace_is_refused_and_no_instance_is_spawned(
    tmp_path: Path,
) -> None:
    """Criterion 7, literally: beta's workspace inside alpha's, through the CLI."""
    root = tmp_path.resolve()
    alpha_workspace = root / "alpha" / "workspace"
    alpha = write_instance(
        root, "alpha", port=ALPHA_PORT, workspace=alpha_workspace
    )
    beta = write_instance(
        root, "beta", port=BETA_PORT, workspace=alpha_workspace / "nested"
    )
    fleet_path = write_fleet(root, {"alpha": alpha, "beta": beta})

    attempt = attempt_start(root, fleet_path)

    assert_refused_without_starting_anything(attempt, "alpha", "beta")
    # Both names in one location, and the reason an operator can act on.
    assert "instances.alpha, instances.beta" in attempt.output, attempt.output
    assert "is inside" in attempt.output, attempt.output
    # Neither workspace was so much as created, which is the refusal restated:
    # ``prepare_fleet`` makes these, and it was never reached.
    assert not alpha_workspace.exists()
    assert not (alpha_workspace / "nested").exists()


@has_ps
@pytest.mark.parametrize(
    ("label", "beta_workspace_of"),
    [
        # Beta's workspace inside alpha's *config* directory rather than its
        # workspace: a different rule in the same check, and the shape that would
        # make alpha's own deny wall it off from its own files.
        ("inside the peer config dir", lambda root: root / "alpha" / "shared"),
        # The same workspace twice: sharing rather than nesting.
        ("shared with the peer", lambda root: root / "alpha" / "workspace"),
    ],
)
def test_another_overlapping_layout_is_refused_the_same_way(
    tmp_path: Path,
    label: str,
    beta_workspace_of: object,
) -> None:
    """Nesting is not a special case; every overlap starts nothing."""
    root = tmp_path.resolve()
    alpha = write_instance(
        root, "alpha", port=ALPHA_PORT, workspace=root / "alpha" / "workspace"
    )
    beta = write_instance(
        root,
        "beta",
        port=BETA_PORT,
        workspace=beta_workspace_of(root),  # type: ignore[operator]
    )
    fleet_path = write_fleet(root, {"alpha": alpha, "beta": beta})

    attempt = attempt_start(root, fleet_path)

    assert_refused_without_starting_anything(attempt, "alpha", "beta")


@has_ps
def test_a_peer_config_dir_inside_a_workspace_is_refused(tmp_path: Path) -> None:
    """The overlap in the other direction: alpha's workspace contains beta itself.

    Worth its own test because the containment is discovered from the *first*
    instance rather than the second, which is the branch a fixture that always
    nests the later declaration never reaches.
    """
    root = tmp_path.resolve()
    alpha = write_instance(root, "alpha", port=ALPHA_PORT, workspace=root)
    beta = write_instance(
        root, "beta", port=BETA_PORT, workspace=root / "beta" / "workspace"
    )
    fleet_path = write_fleet(root, {"alpha": alpha, "beta": beta})

    attempt = attempt_start(root, fleet_path)

    assert_refused_without_starting_anything(attempt, "alpha", "beta")


def test_the_same_fixture_without_the_overlap_is_accepted(tmp_path: Path) -> None:
    """The vacuity guard: nothing but the overlap makes these fleets unstartable.

    Without it, a fleet refused for an unrelated reason — a malformed document, a
    port clash, a workspace this fixture spelled wrongly — would satisfy every
    assertion above while proving nothing about overlap at all.

    Asked of the validator in this process rather than through the command on
    purpose: a separable fleet on a host with Seatbelt passes all three gates and
    then *runs in the foreground*, so driving the command here would start a real
    fleet to prove a fixture is well formed. Whether such a fleet comes up is
    criterion 4's business; all this needs to know is that these instances are
    separable and that the overlapping ones differ in exactly that.
    """
    from nanobot.fleet.validate import validate_fleet_file

    root = tmp_path.resolve()
    alpha = write_instance(
        root, "alpha", port=ALPHA_PORT, workspace=root / "alpha" / "workspace"
    )
    beta = write_instance(
        root, "beta", port=BETA_PORT, workspace=root / "beta" / "workspace"
    )
    fleet_path = write_fleet(root, {"alpha": alpha, "beta": beta})
    before = tree(root)

    instances = validate_fleet_file(fleet_path)

    assert [one.name for one in instances] == ["alpha", "beta"]
    # Validation is the gate that must not create anything, since it is the one
    # that refuses: the criterion's "starts no instance" begins here.
    assert describe_changes(before, tree(root)) == ()


# ----------------------------------------------------------------------------
# The sweep's own sensitivity
# ----------------------------------------------------------------------------


@has_ps
def test_the_sweep_sees_a_process_carrying_this_fleets_directory(
    tmp_path: Path,
) -> None:
    """A decoy whose argv names the fleet root is found.

    The first of the two shapes an instance takes: before it names itself, its
    argv still carries the ``--config`` path this fleet handed it. The decoy is an
    interpreter holding that path in its own argv rather than a shell command,
    because ``sh -c`` execs a lone ``sleep`` in place and the path would vanish
    with the shell.
    """
    root = tmp_path.resolve()
    before = processes()
    decoy = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import time; time.sleep({int(DECOY_TIMEOUT_SECONDS)})",
            str(root / "alpha" / "config.json"),
        ],
        env=child_environment(),
    )
    try:
        found = wait_for_suspect(root.name, before)
    finally:
        stop(decoy)

    assert any(process.pid == decoy.pid for process in found), found


@has_ps
def test_the_sweep_sees_a_process_that_named_itself_the_way_an_instance_does(
    tmp_path: Path,
) -> None:
    """A decoy that named itself through the production code is still found.

    The second shape, and the one that makes the first insufficient. The decoy
    calls :func:`set_cli_process_identity` with a ``serve`` instance's own
    arguments rather than hard-coding a title, so this is the name a real instance
    takes — and the assertions below record what that costs the sweep: the fleet's
    directory is no longer anywhere in the process table, so an argv search alone
    would report that nothing had been started.

    The decoy announces itself with a file written *after* it has renamed itself,
    because it is visible in ``ps`` under its original argv for as long as the
    interpreter takes to start — and a sweep that caught it in that window would
    be testing the argv half again rather than the title half.
    """
    root = tmp_path.resolve()
    config = root / "alpha" / "config.json"
    ready = root / "renamed"
    before = processes()
    decoy = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import pathlib, sys, time\n"
            "from nanobot.cli.process_identity import set_cli_process_identity\n"
            "set_cli_process_identity(['serve', '--config', sys.argv[1]])\n"
            "pathlib.Path(sys.argv[2]).write_text('renamed')\n"
            f"time.sleep({int(DECOY_TIMEOUT_SECONDS)})\n",
            str(config),
            str(ready),
        ],
        env=child_environment(),
    )
    try:
        wait_for_file(ready)
        found = appeared_since(before, root.name)
    finally:
        stop(decoy)

    assert any(process.pid == decoy.pid for process in found), found
    matched = next(process for process in found if process.pid == decoy.pid)
    assert NANOBOT_MARK in matched.args
    # The argv really was replaced: this is why the title half has to exist.
    assert root.name not in matched.args


# ----------------------------------------------------------------------------
# The sweep's machinery
# ----------------------------------------------------------------------------


def test_the_process_table_is_parsed_into_pids_and_arguments() -> None:
    parsed = parse_processes("  1 /sbin/launchd\n 902 python -m nanobot serve\n")

    assert parsed == (
        Process(pid=1, args="/sbin/launchd"),
        Process(pid=902, args="python -m nanobot serve"),
    )


def test_a_line_that_is_not_a_process_is_dropped_rather_than_guessed_at() -> None:
    """A header, a blank line or a wrapped argv must not become a fake process."""
    assert parse_processes("  PID ARGS\n\n   \n-1 broken\n123\n") == ()


@has_ps
def test_this_host_really_reports_processes() -> None:
    """``ps`` answers here, so an empty sweep means empty and not unavailable."""
    found = processes()

    assert len(found) > 1
    assert any(process.pid == os.getpid() for process in found)


def test_a_process_is_a_suspect_by_its_arguments_or_by_its_name() -> None:
    """Both shapes, plus the one an instance takes once it has renamed itself."""
    marker = "fleet-root-1d2e3f"
    table = (
        Process(pid=1, args="/sbin/launchd"),
        Process(pid=2, args=f"python -m nanobot serve --config /tmp/{marker}/c.json"),
        Process(pid=3, args="nanobot"),
        Process(pid=4, args="nanobot-fleet"),
        Process(pid=5, args=f"/bin/sh -c sleep 30 /tmp/{marker}"),
    )

    assert {process.pid for process in suspects(table, marker)} == {2, 3, 4, 5}


def test_a_process_with_no_connection_to_nanobot_is_not_a_suspect() -> None:
    table = (Process(pid=9, args="/usr/bin/vim notes.md"),)

    assert suspects(table, "fleet-root-1d2e3f") == ()


def test_the_sweep_errs_towards_suspicion_rather_than_silence() -> None:
    """A new process merely mentioning nanobot is treated as one of ours, on purpose.

    The rule is wider than it strictly needs to be, and this pins that as a
    decision rather than an accident: the sweep only ever sees processes that were
    not running before the command, so the cost of being wide is a loud failure
    somebody looks at, while the cost of being narrow is this criterion passing
    with a fleet still running.
    """
    table = (Process(pid=9, args="/usr/bin/vim notes-about-nanobot.md"),)

    assert len(suspects(table, "fleet-root-1d2e3f")) == 1


def test_a_tree_gains_and_loses_entries_visibly(tmp_path: Path) -> None:
    before = tree(tmp_path)
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "log").write_text("x", encoding="utf-8")

    assert describe_changes(before, tree(tmp_path)) == (
        "created: workspace",
        "created: workspace/log",
    )


def test_a_path_that_changes_kind_is_reported(tmp_path: Path) -> None:
    """A workspace appearing where a file was is still a change."""
    (tmp_path / "thing").write_text("x", encoding="utf-8")
    before = tree(tmp_path)
    (tmp_path / "thing").unlink()
    (tmp_path / "thing").mkdir()

    assert describe_changes(before, tree(tmp_path)) == ("changed: thing",)


def test_an_unchanged_tree_reports_nothing(tmp_path: Path) -> None:
    (tmp_path / "kept").mkdir()
    before = tree(tmp_path)

    assert describe_changes(before, tree(tmp_path)) == ()


def test_every_name_a_nanobot_process_can_take_is_one_the_sweep_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep's one assumption, checked against the production code that sets it.

    The titles are not enumerated here — they are asked for. Every role in
    ``_ROLES`` and the roleless default are run through
    :func:`set_cli_process_identity`, and each resulting title must be something
    :func:`suspects` would match. A future role, or a renamed title, fails this
    instead of quietly becoming a process the sweep cannot see.
    """
    from nanobot.cli.process_identity import _ROLES, set_cli_process_identity

    titles: list[str] = []
    monkeypatch.setattr("nanobot.cli.process_identity.os.name", "posix")
    monkeypatch.setattr(
        "nanobot.cli.process_identity._set_process_title", titles.append
    )
    for args in [[], ["serve"], *([role] for role in sorted(_ROLES))]:
        set_cli_process_identity(list(args))

    assert len(titles) == len(_ROLES) + 2
    for title in titles:
        assert suspects((Process(pid=1, args=title),), "unrelated-marker") != (), title

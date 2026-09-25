"""Acceptance criterion 4: filesystem isolation, against two real confined instances.

The central criterion of the fleet. Two real instances run under the real Seatbelt
policy with **their own guards off** — ``tools.restrictToWorkspace`` false and the
shell sandbox disabled (``tools.exec.sandbox`` empty), which is what
:meth:`StubLLMServer.instance_config` defaults to. Nothing inside the agent is
checking anything, so the only thing left that can explain a refusal is the
kernel.

The question is asked the way an attacker would ask it: through instance A's own
``exec`` tool, driven from outside by a scripted assistant turn, with every probe
run one level deeper again (``sandbox-exec`` → ``bash`` → ``sh`` → the tool, and
once → ``python``). Confinement was verified to inherit through exactly that chain
on the pinned host, which is what makes reaching the filesystem through the shell
tool's *descendants* a valid test of the boundary rather than a test of whatever
the first process happened to be allowed.

Three design decisions carry most of the weight.

*Every probe reports its own exit status and its own output, into A's workspace.*
The report is not returned through the model — it is raw bytes written to a file
only A and the supervisor can reach, so what is asserted on is what the command
actually produced. Capturing each probe's combined output into that report is the
point rather than a convenience: a denial is asserted three times over — as a
non-zero status, as the kernel's own ``EPERM`` wording (a mistyped tool path exits
non-zero too, and would otherwise satisfy the whole criterion), and as the absence
of the thing that would have been read.

*Each denied operation has its own marker, in its own place.* B's workspace holds
:data:`PEER_WORKSPACE_SECRET`, B's config file holds :data:`PEER_CONFIG_MARKER` as
its provider key, and the fleet file holds :data:`FLEET_FILE_MARKER` as a variable
name in B's entry. Nothing else on disk holds any of them. So "A could not read
the fleet file" is not inferred from an exit status alone — the string that only
the fleet file contains is absent from everything A produced.

*A positive control runs in the same turn, from the same tool.* An instance that
was simply broken — a mis-built argv, a dead interpreter, a shell that cannot run
at all — would fail all four denied operations for reasons that have nothing to do
with confinement. So the same shell tool, in the same command, writes a file into
A's own workspace and reads it back, and both must succeed with the value
intact.

The criterion's closing clause is checked as a before/after snapshot of B's
workspace and config directory, file by file, by digest. One exception is
deliberate and narrowed to it: files under B's own ``logs/`` directory are
compared as *append-only* — B is a live process whose stdout is block-buffered
(the fleet's minimal environment carries no ``PYTHONUNBUFFERED``), so a flush
landing between the two snapshots is B talking about itself, not A writing. New
paths there are still a failure, and the log is searched for A's intruder marker
like everything else.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from stub_llm_server import StubCompletion, StubLLMServer, free_port

from nanobot.fleet.instance import INSTANCE_LOG_SUBDIR, SANDBOX_EXEC
from nanobot.fleet.supervisor import FleetSupervisor, prepare_fleet, start_fleet
from nanobot.fleet.validate import validate_fleet_file

confinement_available = pytest.mark.skipif(
    not Path(SANDBOX_EXEC).is_file(),
    reason="a fleet cannot be started without native Seatbelt",
)

#: Planted in B's workspace. If A reads it, the value shows up in A's report.
PEER_WORKSPACE_SECRET = "beta-workspace-secret-6c31af"

#: B's provider key, and the only distinctive string in B's config file.
PEER_CONFIG_MARKER = "beta-config-marker-0b92d7"

#: Declared as a variable *name* in B's fleet entry, so it appears in the fleet
#: file and nowhere else. Never set in the supervisor's environment, so the
#: launcher omits it — a declaration is all this needs to be.
FLEET_FILE_MARKER = "FLEET_FILE_MARKER_47ad1e"

#: Written and read back by A inside its own workspace: the positive control.
OWN_VALUE = "alpha-own-workspace-value-5fe80c"

#: What A tries to leave behind in B's workspace.
INTRUDER = "alpha-was-here-8d40c2"

PEER_SECRET_NAME = "secret.txt"
TOUCH_NAME = "intruder-touch.txt"
REDIRECT_NAME = "intruder-redirect.txt"
REPORT_NAME = "probe-report.txt"
OWN_NAME = "own.txt"

#: Absolute paths throughout: the shell tool's minimal environment has no ``PATH``
#: on Unix (``agent/tools/shell.py``, ``_build_env``), so a bare command name
#: would be testing bash's compiled-in fallback rather than the fleet.
SH = "/bin/sh"
CAT = "/bin/cat"
LS = "/bin/ls"
TOUCH = "/usr/bin/touch"
TR = "/usr/bin/tr"

#: ``strerror`` for ``EPERM``, which is what Seatbelt returns for a denied path.
#: Every denied probe must say this, not merely exit non-zero: a mistyped tool
#: path, a dead interpreter or a syntax error all exit non-zero too, and each one
#: would let the whole criterion pass without the kernel refusing anything. This
#: file is gated to darwin, so the wording is the pinned host's.
REFUSAL = "Operation not permitted"

CAP_MB = 512
STARTUP_TIMEOUT_SECONDS = 90.0
TURN_TIMEOUT_SECONDS = 120.0

#: Answers every turn that is not the scripted tool call, including the one
#: carrying the tool's output back. The script queue is shared by the whole fleet.
FALLBACK_ANSWER = "reported"

DENIED = "denied"
ALLOWED = "allowed"


@dataclass(frozen=True)
class Probe:
    """One filesystem operation A attempts, and what the fleet owes it."""

    label: str
    shell: str
    expect: str


#: The probe set, in the order it runs. The first four are the criterion's own
#: denied operations; the next four are denied by the same two rules in the same
#: profile and cost nothing to ask. The last two are the positive control.
#:
#: Every probe runs through ``sh -c``, so each one is a grandchild of the shell
#: tool rather than the process the tool started — the inheritance the criterion
#: is really about.
PROBES: tuple[Probe, ...] = (
    Probe("read_peer_workspace_file", f'{CAT} "$PEER_WORKSPACE_FILE"', DENIED),
    Probe("read_peer_config_file", f'{CAT} "$PEER_CONFIG"', DENIED),
    Probe("create_file_in_peer_workspace", f'{TOUCH} "$PEER_TOUCH"', DENIED),
    Probe("read_fleet_file", f'{CAT} "$FLEET_FILE"', DENIED),
    Probe(
        "write_file_in_peer_workspace",
        'printf %s "$INTRUDER" > "$PEER_WRITE"',
        DENIED,
    ),
    Probe("list_peer_workspace", f'{LS} -a "$PEER_WORKSPACE"', DENIED),
    Probe(
        "list_peer_workspace_via_python",
        '"$PY" -c "import os, sys; print(os.listdir(sys.argv[1]))" "$PEER_WORKSPACE"',
        DENIED,
    ),
    Probe("read_supervisor_state_file", f'{CAT} "$STATE_FILE"', DENIED),
    Probe(
        "write_own_workspace_file",
        'printf %s "$OWN_VALUE" > "$OWN_FILE"',
        ALLOWED,
    ),
    Probe("read_own_workspace_file", f'{CAT} "$OWN_FILE"', ALLOWED),
)

DENIED_PROBES = tuple(probe.label for probe in PROBES if probe.expect == DENIED)
ALLOWED_PROBES = tuple(probe.label for probe in PROBES if probe.expect == ALLOWED)

#: The four operations the criterion names literally, so a future edit that
#: reshuffles :data:`PROBES` cannot quietly drop one of them.
CRITERION_PROBES = (
    "read_peer_workspace_file",
    "read_peer_config_file",
    "create_file_in_peer_workspace",
    "read_fleet_file",
)


@dataclass(frozen=True)
class Outcome:
    """One probe's exit status and its combined output, as A recorded them."""

    rc: int
    output: str


@dataclass(frozen=True)
class TreeSnapshot:
    """What a directory tree held, file by file.

    ``entries`` maps every relative path to a digest (files), ``"dir"``, or a
    symlink target, so a created, deleted or rewritten path all show up.
    ``appendable`` holds the raw bytes of the files exempted from byte-equality,
    which are compared as a prefix instead.
    """

    entries: dict[str, str]
    appendable: dict[str, bytes]


def snapshot_tree(root: Path, *, append_only: tuple[str, ...] = ()) -> TreeSnapshot:
    """Digest every path under *root*, reading the bytes of ``append_only`` files.

    ``append_only`` names relative directories whose files are still tracked as
    paths — a new one is a change — but whose contents are compared as a prefix
    by :func:`describe_changes` rather than by digest.
    """
    entries: dict[str, str] = {}
    appendable: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        exempt = any(
            relative == prefix or relative.startswith(f"{prefix}/")
            for prefix in append_only
        )
        if path.is_symlink():
            entries[relative] = f"link:{os.readlink(path)}"
        elif path.is_dir():
            entries[relative] = "dir"
        elif exempt:
            entries[relative] = "file:appendable"
            appendable[relative] = path.read_bytes()
        else:
            entries[relative] = f"file:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    return TreeSnapshot(entries=entries, appendable=appendable)


def describe_changes(before: TreeSnapshot, after: TreeSnapshot) -> list[str]:
    """Every difference between two snapshots, as lines fit for a failure message."""
    changes: list[str] = []
    for relative in sorted(set(before.entries) | set(after.entries)):
        was, now = before.entries.get(relative), after.entries.get(relative)
        if was is None:
            changes.append(f"created: {relative}")
        elif now is None:
            changes.append(f"removed: {relative}")
        elif was != now:
            changes.append(f"changed: {relative}")
    for relative, content in before.appendable.items():
        now_bytes = after.appendable.get(relative)
        if now_bytes is None or not now_bytes.startswith(content):
            changes.append(f"rewritten (not appended to): {relative}")
    return changes


def write_instance(
    root: Path,
    name: str,
    stub: StubLLMServer,
    *,
    api_key: str | None = None,
) -> tuple[Path, Path]:
    """Lay out one instance and return ``(config_path, workspace)``.

    The workspace is a *sibling* of the config directory rather than a child of
    it. Both layouts are legal, and nanobot's own default nests them — but nesting
    them here would mean the peer-workspace deny and the peer-config-directory
    deny cover the same paths, so a fleet that emitted only one of the two would
    still pass every probe below. Separating them makes the criterion's first two
    operations test two different rules.
    """
    config_dir = root / name
    workspace = root / f"{name}-workspace"
    config_dir.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    data = stub.instance_config(workspace, api_port=free_port())
    if api_key is not None:
        data["providers"]["custom"]["apiKey"] = api_key
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return config_path, workspace


def api_port_of(config_path: Path) -> int:
    """The port the instance will bind, read back from its own config."""
    return int(json.loads(config_path.read_text(encoding="utf-8"))["api"]["port"])


def log_tail(path: Path, limit: int = 4000) -> str:
    """An instance's own log, the only diagnostic a confined process leaves."""
    if not path.exists():
        return "(no log)"
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def wait_for_health(port: int, name: str, log_path: Path) -> None:
    """Block until the instance answers on its API, or fail with its log.

    There is no ``pytest-timeout`` in this repo, so every wait carries its own
    deadline — the pattern ``tests/webui/test_gateway_webui_smoke.py`` sets.
    """
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


def probe_script(variables: Mapping[str, str]) -> str:
    """Render the one shell command A's tool runs, covering every probe.

    Paths and values are passed as exported variables so that each probe's own
    text stays short enough to read, and so the report cannot be confused with the
    thing being probed. ``run`` records the label, the exit status, and the
    combined output flattened to a single line — a denial that still returned
    bytes would put those bytes in the report, where the marker assertions find
    them.
    """
    assignments = "\n".join(
        f"{name}={shlex.quote(value)}" for name, value in variables.items()
    )
    commands = "\n".join(
        f"run {probe.label} {shlex.quote(probe.shell)}" for probe in PROBES
    )
    return f"""set -u
{assignments}
export {" ".join(variables)}
: > "$REPORT"
run() {{
  label=$1
  out=$({SH} -c "$2" 2>&1)
  rc=$?
  printf '%s|%s|%s\\n' "$label" "$rc" \
"$(printf '%s' "$out" | {TR} '\\n\\t' '  ')" >> "$REPORT"
}}
{commands}
"""


def parse_report(text: str) -> dict[str, Outcome]:
    """Parse the report A wrote into ``label -> outcome``.

    Each probe contributes exactly one line, because ``run`` flattens newlines
    before writing; a line without the separator is not a probe result and is
    ignored rather than guessed at.
    """
    outcomes: dict[str, Outcome] = {}
    for line in text.splitlines():
        label, separator, rest = line.partition("|")
        if not separator:
            continue
        status, _, output = rest.partition("|")
        if not status.lstrip("-").isdigit():
            continue
        outcomes[label] = Outcome(rc=int(status), output=output)
    return outcomes


def stop_fleet(supervisor: FleetSupervisor, loop: threading.Thread) -> None:
    """Ask the supervisor to stop and wait for its loop to leave."""
    supervisor.request_stop()
    loop.join(timeout=STARTUP_TIMEOUT_SECONDS)


@confinement_available
def test_an_instance_cannot_reach_its_peers_files_but_can_reach_its_own(
    tmp_path: Path,
    stub_llm_server: StubLLMServer,
) -> None:
    pytest.importorskip("aiohttp")
    assert sys.executable, "the python probe needs the supervisor's own interpreter"

    root = tmp_path.resolve()
    alpha_config, alpha_workspace = write_instance(root, "alpha", stub_llm_server)
    beta_config, beta_workspace = write_instance(
        root, "beta", stub_llm_server, api_key=PEER_CONFIG_MARKER
    )
    beta_config_dir = beta_config.parent

    # The known file in B's workspace the criterion names.
    (beta_workspace / PEER_SECRET_NAME).write_text(
        PEER_WORKSPACE_SECRET, encoding="utf-8"
    )

    fleet_path = root / "fleet.json"
    fleet_path.write_text(
        json.dumps({
            "instances": {
                "alpha": {
                    "config": str(alpha_config),
                    "mode": "serve",
                    "memoryLimitMb": CAP_MB,
                },
                "beta": {
                    "config": str(beta_config),
                    "mode": "serve",
                    "memoryLimitMb": CAP_MB,
                    # A name, never a value: this is the fleet file's marker.
                    "env": [FLEET_FILE_MARKER],
                },
            }
        }),
        encoding="utf-8",
    )

    # The premise, checked rather than assumed: each marker is where the criterion
    # needs it and in exactly one place, and both instances really do run with
    # their own guards off.
    for config_path in (alpha_config, beta_config):
        guards = json.loads(config_path.read_text(encoding="utf-8"))["tools"]
        assert guards["restrictToWorkspace"] is False, config_path
        assert guards["exec"]["sandbox"] == "", config_path
        assert guards["exec"]["enable"] is True, config_path
    assert PEER_CONFIG_MARKER in beta_config.read_text(encoding="utf-8")
    assert PEER_CONFIG_MARKER not in alpha_config.read_text(encoding="utf-8")
    assert FLEET_FILE_MARKER in fleet_path.read_text(encoding="utf-8")

    instances = {one.name: one for one in validate_fleet_file(fleet_path)}
    alpha, beta = instances["alpha"], instances["beta"]
    assert (alpha.workspace, beta.workspace) == (alpha_workspace, beta_workspace)
    assert beta.config_dir == beta_config_dir

    plan = prepare_fleet(list(instances.values()), fleet_path=fleet_path)

    # The production spawn path: real argv, real Seatbelt wrapper, real ``env -i``.
    supervisor = start_fleet(plan)
    alpha_log = supervisor.launched("alpha").log_path
    beta_log = supervisor.launched("beta").log_path
    loop = threading.Thread(
        target=supervisor.run,
        kwargs={"handle_signals": False},
        name="fleet-supervisor",
        daemon=True,
    )
    loop.start()
    report_path = alpha_workspace / REPORT_NAME
    try:
        wait_for_health(api_port_of(alpha_config), "alpha", alpha_log)
        wait_for_health(api_port_of(beta_config), "beta", beta_log)

        # Bracket the probes, not the fleet's whole lifetime: B is fully up, so
        # anything that changes from here has to be explained.
        before_workspace = snapshot_tree(beta_workspace)
        before_config_dir = snapshot_tree(
            beta_config_dir, append_only=(INSTANCE_LOG_SUBDIR,)
        )

        stub_llm_server.set_fallback(StubCompletion(content=FALLBACK_ANSWER))
        stub_llm_server.script_tool_call("exec", {
            "command": probe_script({
                "REPORT": str(report_path),
                "PEER_WORKSPACE": str(beta_workspace),
                "PEER_WORKSPACE_FILE": str(beta_workspace / PEER_SECRET_NAME),
                "PEER_CONFIG": str(beta_config),
                "PEER_TOUCH": str(beta_workspace / TOUCH_NAME),
                "PEER_WRITE": str(beta_workspace / REDIRECT_NAME),
                "FLEET_FILE": str(fleet_path),
                "STATE_FILE": str(plan.state_path),
                "OWN_FILE": str(alpha_workspace / OWN_NAME),
                "PY": sys.executable,
                "OWN_VALUE": OWN_VALUE,
                "INTRUDER": INTRUDER,
            }),
        })
        response = httpx.post(
            f"http://127.0.0.1:{api_port_of(alpha_config)}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "run the probes"}]},
            timeout=TURN_TIMEOUT_SECONDS,
            trust_env=False,
        )
        assert response.status_code == 200, f"{response.text}\n{log_tail(alpha_log)}"
        if not report_path.exists():
            pytest.fail(
                f"alpha's shell tool never produced {report_path}\n"
                f"{log_tail(alpha_log)}"
            )
        report = report_path.read_text(encoding="utf-8", errors="replace")

        after_workspace = snapshot_tree(beta_workspace)
        after_config_dir = snapshot_tree(
            beta_config_dir, append_only=(INSTANCE_LOG_SUBDIR,)
        )
    finally:
        stop_fleet(supervisor, loop)

    assert not loop.is_alive()

    outcomes = parse_report(report)
    assert set(outcomes) == {probe.label for probe in PROBES}, report

    # The positive control, first: an instance that cannot do this is broken, and
    # every denial below would be worthless.
    assert outcomes["write_own_workspace_file"].rc == 0, report
    assert outcomes["read_own_workspace_file"].rc == 0, report
    assert outcomes["read_own_workspace_file"].output == OWN_VALUE, report
    assert (alpha_workspace / OWN_NAME).read_text(encoding="utf-8") == OWN_VALUE

    # The criterion. Asserted over the whole denied set, and separately over the
    # four operations named in the criterion so neither can be lost.
    for label in DENIED_PROBES:
        outcome = outcomes[label]
        assert outcome.rc != 0, f"{label} was not refused\n{report}"
        # The refusal has to be the kernel's, not a probe that could not run.
        assert REFUSAL in outcome.output, f"{label} failed for another reason\n{report}"
    for label in CRITERION_PROBES:
        assert label in DENIED_PROBES
        assert outcomes[label].rc != 0, f"{label} was not refused\n{report}"
        assert REFUSAL in outcomes[label].output, report

    # Statuses can be misread; contents cannot be un-read. Each marker lives in
    # exactly one of the places A was denied, so its absence from everything A
    # produced is the denial stated a second way.
    assert PEER_WORKSPACE_SECRET not in report
    assert PEER_CONFIG_MARKER not in report
    assert FLEET_FILE_MARKER not in report

    # Nothing A tried to leave behind exists.
    assert not (beta_workspace / TOUCH_NAME).exists()
    assert not (beta_workspace / REDIRECT_NAME).exists()

    # The closing clause, file by file. The vacuity guard comes first: an empty
    # tree compares equal to an empty tree.
    assert f"file:{hashlib.sha256(PEER_WORKSPACE_SECRET.encode()).hexdigest()}" == (
        before_workspace.entries[PEER_SECRET_NAME]
    )
    assert "config.json" in before_config_dir.entries
    assert describe_changes(before_workspace, after_workspace) == [], (
        f"beta's workspace changed\n{log_tail(beta_log)}"
    )
    assert describe_changes(before_config_dir, after_config_dir) == [], (
        f"beta's config directory changed\n{log_tail(beta_log)}"
    )

    # The one file allowed to differ is B's own log, so it is searched too.
    assert INTRUDER not in log_tail(beta_log, limit=1_000_000)


def test_the_probe_script_covers_every_declared_probe() -> None:
    """The command really contains one ``run`` per probe, with the right target.

    The script and the assertions are generated from the same tuple, so this is
    cheap; it is worth pinning because a quoting mistake in
    :func:`probe_script` would drop a probe silently and
    ``set(outcomes) == {...}`` in the acceptance test would then be the only thing
    standing between a missing denial and a green run.
    """
    script = probe_script({"REPORT": "/tmp/report", "OWN_FILE": "/tmp/own"})

    assert script.count("\nrun ") == len(PROBES)
    for probe in PROBES:
        assert f"run {probe.label} " in script
        assert probe.shell in script
    assert "REPORT=/tmp/report" in script
    assert "export REPORT OWN_FILE" in script


def test_the_probe_script_quotes_a_path_that_would_otherwise_split() -> None:
    """A path with a space survives into the script as one word."""
    script = probe_script({"REPORT": "/tmp/a b/report"})

    assert "REPORT='/tmp/a b/report'" in script


def test_the_report_parser_reads_a_status_and_a_flattened_output() -> None:
    parsed = parse_report(
        "read_peer|1|cat: /x: Operation not permitted\nread_own|0|value\n"
    )

    assert parsed == {
        "read_peer": Outcome(rc=1, output="cat: /x: Operation not permitted"),
        "read_own": Outcome(rc=0, output="value"),
    }


def test_the_report_parser_keeps_a_separator_inside_an_output() -> None:
    """Only the first two fields are structural; the rest is the command's output."""
    assert parse_report("probe|2|a|b") == {"probe": Outcome(rc=2, output="a|b")}


def test_the_report_parser_ignores_lines_that_are_not_probe_results() -> None:
    """Stray output must not be mistaken for a probe that passed.

    A line with no separator, or with something other than a status where the
    status belongs, is dropped — which makes the acceptance test's
    ``set(outcomes)`` check fail loudly rather than inventing an ``rc`` of 0 for a
    probe that never ran.
    """
    assert parse_report("noise\nbash: line 1: oops\nprobe|x|y\n") == {}


def test_a_created_file_is_reported_as_a_change(tmp_path: Path) -> None:
    before = snapshot_tree(tmp_path)
    (tmp_path / "new.txt").write_text("x", encoding="utf-8")

    assert describe_changes(before, snapshot_tree(tmp_path)) == ["created: new.txt"]


def test_a_rewritten_file_is_reported_as_a_change(tmp_path: Path) -> None:
    (tmp_path / "kept.txt").write_text("before", encoding="utf-8")
    before = snapshot_tree(tmp_path)
    (tmp_path / "kept.txt").write_text("after", encoding="utf-8")

    assert describe_changes(before, snapshot_tree(tmp_path)) == ["changed: kept.txt"]


def test_a_removed_file_is_reported_as_a_change(tmp_path: Path) -> None:
    (tmp_path / "doomed.txt").write_text("x", encoding="utf-8")
    before = snapshot_tree(tmp_path)
    (tmp_path / "doomed.txt").unlink()

    assert describe_changes(before, snapshot_tree(tmp_path)) == ["removed: doomed.txt"]


def test_a_nested_file_is_compared_by_its_relative_path(tmp_path: Path) -> None:
    (tmp_path / "deep").mkdir()
    before = snapshot_tree(tmp_path)
    (tmp_path / "deep" / "leaf.txt").write_text("x", encoding="utf-8")

    assert describe_changes(before, snapshot_tree(tmp_path)) == [
        "created: deep/leaf.txt"
    ]


def test_an_append_only_file_may_grow_but_not_be_rewritten(tmp_path: Path) -> None:
    """The narrow exemption B's live log needs, and nothing wider.

    Growth is what a block-buffered process flushing its own output looks like;
    a rewrite is not, and neither is a new path appearing beside it.
    """
    logs = tmp_path / INSTANCE_LOG_SUBDIR
    logs.mkdir()
    log = logs / "fleet.log"
    log.write_bytes(b"first line\n")
    before = snapshot_tree(tmp_path, append_only=(INSTANCE_LOG_SUBDIR,))

    log.write_bytes(b"first line\nsecond line\n")
    assert describe_changes(
        before, snapshot_tree(tmp_path, append_only=(INSTANCE_LOG_SUBDIR,))
    ) == []

    log.write_bytes(b"truncated\n")
    assert describe_changes(
        before, snapshot_tree(tmp_path, append_only=(INSTANCE_LOG_SUBDIR,))
    ) == ["rewritten (not appended to): logs/fleet.log"]


def test_an_append_only_exemption_does_not_cover_a_new_neighbour(
    tmp_path: Path,
) -> None:
    logs = tmp_path / INSTANCE_LOG_SUBDIR
    logs.mkdir()
    (logs / "fleet.log").write_bytes(b"x")
    before = snapshot_tree(tmp_path, append_only=(INSTANCE_LOG_SUBDIR,))

    (logs / "planted.txt").write_bytes(b"y")

    assert describe_changes(
        before, snapshot_tree(tmp_path, append_only=(INSTANCE_LOG_SUBDIR,))
    ) == ["created: logs/planted.txt"]


def test_an_append_only_prefix_does_not_match_a_sibling_by_name(
    tmp_path: Path,
) -> None:
    """``logs`` must not exempt ``logs-backup``, which only starts the same way."""
    (tmp_path / f"{INSTANCE_LOG_SUBDIR}-backup").mkdir()
    sibling = tmp_path / f"{INSTANCE_LOG_SUBDIR}-backup" / "file"
    sibling.write_bytes(b"before")
    before = snapshot_tree(tmp_path, append_only=(INSTANCE_LOG_SUBDIR,))
    sibling.write_bytes(b"before and after")

    assert describe_changes(
        before, snapshot_tree(tmp_path, append_only=(INSTANCE_LOG_SUBDIR,))
    ) == [f"changed: {INSTANCE_LOG_SUBDIR}-backup/file"]


def test_a_replaced_directory_is_reported_as_a_change(tmp_path: Path) -> None:
    """A path that stops being a directory is a change, not a match."""
    (tmp_path / "thing").mkdir()
    before = snapshot_tree(tmp_path)
    (tmp_path / "thing").rmdir()
    (tmp_path / "thing").write_text("now a file", encoding="utf-8")

    assert describe_changes(before, snapshot_tree(tmp_path)) == ["changed: thing"]


def test_a_symlink_is_compared_by_its_target(tmp_path: Path) -> None:
    """Retargeting a link changes what a path means without changing any file."""
    link = tmp_path / "link"
    link.symlink_to("/etc/hosts")
    before = snapshot_tree(tmp_path)
    link.unlink()
    link.symlink_to("/etc/passwd")

    assert describe_changes(before, snapshot_tree(tmp_path)) == ["changed: link"]


def test_every_marker_is_distinct_and_self_contained() -> None:
    """No marker assertion above can be satisfied by the wrong string.

    Each marker stands for one denied operation; if two were equal, or one
    contained another, a single leak would look like several and several like one.
    Cheap to pin, and it would not fail loudly on its own.
    """
    markers = (
        PEER_WORKSPACE_SECRET,
        PEER_CONFIG_MARKER,
        FLEET_FILE_MARKER,
        OWN_VALUE,
        INTRUDER,
    )

    assert len(set(markers)) == len(markers)
    for marker in markers:
        assert sum(marker in other for other in markers) == 1


def test_the_probe_set_declares_both_directions() -> None:
    """Denials and the positive control are both present, and labels are unique.

    A probe set that lost its ``ALLOWED`` half would still pass every denial
    assertion while no longer proving the instance works at all — which is the
    reading of this criterion the bead explicitly rules out.
    """
    labels = [probe.label for probe in PROBES]

    assert len(set(labels)) == len(labels)
    assert set(DENIED_PROBES) and set(ALLOWED_PROBES)
    assert set(DENIED_PROBES).isdisjoint(ALLOWED_PROBES)
    assert set(DENIED_PROBES) | set(ALLOWED_PROBES) == set(labels)
    assert set(CRITERION_PROBES) <= set(DENIED_PROBES)
    assert {probe.expect for probe in PROBES} == {DENIED, ALLOWED}

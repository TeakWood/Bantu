"""Tests for the supervisor's state file and its liveness reconciliation.

Four layers, and the split follows the module's own seams.

The *write* tests care about durability and about the file's mode, because the
mode is a security property here: the file lists every instance's workspace and
config directory, which is the same map the fleet file deny exists to withhold.

The *path* tests pin one fact that cannot be tested from inside either module
alone — that the path the supervisor writes is the path the profile builder
denies. A mismatch would be silent in the worst direction: ``sandbox-exec``
accepts a deny naming a path nothing writes, confines nothing, and reports no
error.

The *reconciliation* tests are the point of the bead. They run against real
processes wherever a real process can express the case: a live pid carrying a
forged identity is a genuine stand-in for a recycled pid, because what makes a
recycled pid dangerous is exactly that it is alive and is not the process that
was recorded. Only the "identity momentarily unreadable" branch is driven through
injected probes, since arranging a pid whose identity cannot be read right now is
not something a test can do honestly.

The *parse* tests are refusals. The only legitimate writer of this file is
``write_fleet_state``, so anything that does not round-trip means the file did
not come from the supervisor — and its contents drive process measurement and
termination.
"""

from __future__ import annotations

import builtins
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from nanobot.fleet.config import FleetInstance
from nanobot.fleet.instance import instance_identity
from nanobot.fleet.profile import SeatbeltProfileError, build_fleet_profiles
from nanobot.fleet.state import (
    EXIT_REASONS,
    IDENTITY_FIELDS,
    RECORD_FIELDS,
    STATE_FILE_MODE,
    FleetStateError,
    InstanceRecord,
    ensure_fleet_state_file,
    fleet_state_path,
    load_fleet_state,
    parse_fleet_state,
    read_fleet_state,
    reconcile_fleet_state,
    running_record,
    write_fleet_state,
)
from nanobot.fleet.validate import ResolvedInstance

STATE_PATH = Path("/srv/fleet.json.state.json")

#: An identity that can never belong to a live process on any supported
#: platform: process group 999999 with a birth time of 1970. Used as the stand-in
#: for "this pid has been handed to somebody else since it was recorded".
FORGED_IDENTITY = "darwin:999999:1:2"


def make_instance(root: Path, name: str, *, memory_limit_mb: int = 512) -> ResolvedInstance:
    """Resolve one instance laid out the way nanobot lays one out by itself.

    ``<root>/<name>/config.json`` with ``<root>/<name>/workspace`` beneath it.
    Both directories are created, because the profile builder refuses a path that
    does not exist.
    """
    config_dir = root / name
    workspace = config_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    return ResolvedInstance(
        name=name,
        entry=FleetInstance(
            config=str(config_path),
            mode="serve",
            memory_limit_mb=memory_limit_mb,
        ),
        config_path=config_path,
        config_dir=config_dir,
        workspace=workspace,
        port=None,
        port_setting="api.port",
    )


def record(**overrides: Any) -> InstanceRecord:
    """A plausible running record, overridable field by field."""
    fields: dict[str, Any] = {
        "name": "alpha",
        "pid": 4242,
        "state": "running",
        "exit_reason": None,
        "workspace": Path("/srv/alpha/workspace"),
        "config_dir": Path("/srv/alpha"),
        "memory_limit_mb": 512,
        "identity": {"identity": 4242, "stable_identity": "darwin:4242:100:200"},
    }
    fields.update(overrides)
    return InstanceRecord(**fields)


def payload(**overrides: Any) -> dict[str, Any]:
    """One well-formed state-file object, overridable key by key."""
    document = record().payload()
    document.update(overrides)
    return document


def refusal(document: object) -> FleetStateError:
    """Parse ``document`` as a state file and return the refusal it produces."""
    with pytest.raises(FleetStateError) as caught:
        parse_fleet_state(STATE_PATH, json.dumps(document))
    return caught.value


def dead_pid() -> int:
    """A pid whose process has exited *and* been reaped, so it is truly gone."""
    process = subprocess.Popen([sys.executable, "-c", ""])
    process.wait()
    return process.pid


def temporaries(directory: Path) -> list[Path]:
    """Every leftover temporary file in ``directory``."""
    return [path for path in directory.iterdir() if path.name.endswith(".tmp")]


# --- writing ------------------------------------------------------------------


def test_a_written_record_carries_exactly_the_seven_required_fields(tmp_path: Path) -> None:
    """The seven facts the status contract names, plus the identity that makes
    the pid trustworthy, and nothing else."""
    path = tmp_path / "fleet.json.state.json"
    write_fleet_state([record()], path=path)

    written = json.loads(path.read_text(encoding="utf-8"))

    assert len(written) == 1
    assert set(written[0]) == set(RECORD_FIELDS) | set(IDENTITY_FIELDS)
    assert written[0]["name"] == "alpha"
    assert written[0]["pid"] == 4242
    assert written[0]["state"] == "running"
    assert written[0]["exit_reason"] is None
    assert written[0]["workspace"] == "/srv/alpha/workspace"
    assert written[0]["config_dir"] == "/srv/alpha"
    assert written[0]["memory_limit_mb"] == 512


def test_a_written_state_file_reads_back_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    records = (
        record(),
        record(name="beta", pid=99, state="exited", exit_reason="memory", identity={}),
    )
    write_fleet_state(records, path=path)

    assert load_fleet_state(path) == records


def test_the_state_file_is_written_at_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_fleet_state([record()], path=path)

    assert stat.S_IMODE(path.stat().st_mode) == STATE_FILE_MODE


def test_a_relaxed_mode_on_an_existing_file_is_not_preserved(tmp_path: Path) -> None:
    """The divergence from ``helpers._write_text_atomic``, which copies the
    existing mode onto its replacement.

    That is right for a user-owned artefact and wrong here. If a write preserved
    whatever mode it found, a file that was once world-readable — by an operator's
    umask, an unpacked archive, a stray ``chmod`` — would stay world-readable for
    the life of the fleet, and every instance's directories would stay listed in
    it for anyone on the host to read.
    """
    path = tmp_path / "state.json"
    write_fleet_state([], path=path)
    path.chmod(0o644)

    write_fleet_state([record()], path=path)

    assert stat.S_IMODE(path.stat().st_mode) == STATE_FILE_MODE


def test_an_empty_fleet_writes_an_empty_array(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_fleet_state([], path=path)

    assert json.loads(path.read_text(encoding="utf-8")) == []
    assert load_fleet_state(path) == ()


def test_writing_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_fleet_state([record()], path=path)
    write_fleet_state([record(name="beta")], path=path)

    assert temporaries(tmp_path) == []
    assert [entry.name for entry in tmp_path.iterdir()] == ["state.json"]


def test_a_failed_write_leaves_the_previous_state_intact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader in another shell sees one whole array or the other, never a
    truncated one — which is why the file is replaced rather than edited.

    The failure is injected at the rename, the last step, because that is the
    only point at which a partially written state could become visible.
    """
    path = tmp_path / "state.json"
    write_fleet_state([record()], path=path)
    before = path.read_text(encoding="utf-8")

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(Path, "replace", refuse)

    with pytest.raises(FleetStateError) as caught:
        write_fleet_state([record(name="beta")], path=path)

    assert caught.value.kind == "io_error"
    assert path.read_text(encoding="utf-8") == before
    assert temporaries(tmp_path) == []


def test_writing_into_a_missing_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FleetStateError) as caught:
        write_fleet_state([record()], path=tmp_path / "absent" / "state.json")

    assert caught.value.kind == "io_error"


def test_writing_an_unserializable_record_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    unserializable = record(identity={"identity": object()})  # type: ignore[dict-item]

    with pytest.raises(FleetStateError) as caught:
        write_fleet_state([unserializable], path=path)

    assert caught.value.kind == "invalid_record"
    assert not path.exists()


# --- the state file's location ------------------------------------------------


def test_the_state_file_sits_beside_the_fleet_file(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet.json"

    path = fleet_state_path(fleet)

    assert path.parent == tmp_path.resolve()
    assert path.name == "fleet.json.state.json"


def test_two_fleet_files_in_one_directory_never_share_a_state_file(tmp_path: Path) -> None:
    """Derived from the fleet file's whole name, not its stem.

    ``fleet`` and ``fleet.json`` side by side would otherwise map to the same
    state file, and each supervisor would overwrite the other's pids — then
    measure, report and terminate against the other fleet's processes.
    """
    assert fleet_state_path(tmp_path / "fleet") != fleet_state_path(tmp_path / "fleet.json")


def test_the_state_path_is_canonical_through_a_symlinked_parent(tmp_path: Path) -> None:
    """The profile builder refuses a non-canonical path, because the kernel
    matches Seatbelt rules against the real path and a rule naming the link
    silently matches nothing."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    path = fleet_state_path(link / "fleet.json")

    assert path == real.resolve() / "fleet.json.state.json"
    assert path.resolve(strict=False) == path


def test_the_state_path_expands_a_home_relative_fleet_path() -> None:
    path = fleet_state_path("~/fleet.json")

    assert path == Path.home().resolve() / "fleet.json.state.json"


def test_the_state_path_is_the_one_denied_by_the_profile_builder(tmp_path: Path) -> None:
    """The acceptance criterion that spans two modules.

    Both sides call :func:`fleet_state_path`, so this is the test that would fail
    if either grew its own idea of where fleet state lives. It has to be checked
    from outside: a deny naming a path the supervisor never writes is accepted by
    ``sandbox-exec``, confines nothing, and produces no error anywhere.
    """
    fleet = tmp_path / "fleet.json"
    fleet.write_text("{}", encoding="utf-8")
    instances = [make_instance(tmp_path, "alpha"), make_instance(tmp_path, "beta")]
    state_path = ensure_fleet_state_file(fleet_state_path(fleet))

    profiles = build_fleet_profiles(
        instances,
        fleet_path=fleet.resolve(),
        state_path=state_path,
    )

    assert set(profiles) == {"alpha", "beta"}
    for profile in profiles.values():
        assert f'(literal "{state_path}")' in profile


def test_a_profile_cannot_be_built_before_the_state_file_exists(tmp_path: Path) -> None:
    """Why :func:`ensure_fleet_state_file` exists, and why it must run first.

    The builder refuses a deny for a path that does not exist rather than
    emitting one that would match nothing — so the first fleet to start would be
    unconfinable if nothing laid the file down beforehand.
    """
    fleet = tmp_path / "fleet.json"
    fleet.write_text("{}", encoding="utf-8")
    instances = [make_instance(tmp_path, "alpha"), make_instance(tmp_path, "beta")]

    with pytest.raises(SeatbeltProfileError) as caught:
        build_fleet_profiles(
            instances,
            fleet_path=fleet.resolve(),
            state_path=fleet_state_path(fleet),
        )

    assert "supervisor state file" in str(caught.value)


def test_ensuring_the_state_file_is_idempotent_and_never_truncates(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_fleet_state([record()], path=path)
    before = path.read_text(encoding="utf-8")

    assert ensure_fleet_state_file(path) == path
    assert path.read_text(encoding="utf-8") == before


def test_ensuring_a_missing_state_file_creates_an_empty_one_at_mode_0600(
    tmp_path: Path,
) -> None:
    path = ensure_fleet_state_file(tmp_path / "state.json")

    assert load_fleet_state(path) == ()
    assert stat.S_IMODE(path.stat().st_mode) == STATE_FILE_MODE


# --- records ------------------------------------------------------------------


def test_a_launch_record_carries_the_instance_the_supervisor_started(
    tmp_path: Path,
) -> None:
    instance = make_instance(tmp_path, "alpha", memory_limit_mb=1024)
    identity = instance_identity(os.getpid())

    launched = running_record(instance, pid=os.getpid(), identity=identity)

    assert launched.name == "alpha"
    assert launched.pid == os.getpid()
    assert launched.state == "running"
    assert launched.exit_reason is None
    assert launched.workspace == instance.workspace
    assert launched.config_dir == instance.config_dir
    assert launched.memory_limit_mb == 1024
    assert launched.identity == identity


def test_an_identity_carrying_an_unowned_key_is_refused() -> None:
    """The flat merge is what makes this dangerous rather than untidy: a ``pid``
    key inside the identity mapping would overwrite the record's real pid on its
    way to disk, and the supervisor would measure and signal that value."""
    with pytest.raises(ValueError, match="identity record may only carry"):
        record(identity={"identity": 1, "pid": 999})  # type: ignore[dict-item]


def test_marking_a_record_exited_does_not_overwrite_an_observed_reason() -> None:
    """``"memory"`` is a claim only the cap enforcement is in a position to make.
    A later reader that discovered the process was gone must not blur it."""
    capped = record(state="exited", exit_reason="memory")

    assert capped.exited("exit").exit_reason == "memory"
    assert record().exited("signal").exit_reason == "signal"
    assert record().exited().exit_reason is None


def test_the_stable_identity_is_the_value_compared() -> None:
    assert record().recorded_identity == "darwin:4242:100:200"
    assert record(identity={"identity": 4242}).recorded_identity == 4242
    assert record(identity={}).recorded_identity is None


# --- liveness reconciliation --------------------------------------------------


def test_a_recycled_pid_is_reported_as_exited(tmp_path: Path) -> None:
    """The acceptance criterion. The pid is alive — it is this very test process —
    but the identity recorded next to it belongs to a process that no longer
    exists, which is precisely the situation a pid handed out a second time
    produces. Reporting it as running would tell an operator a dead instance is
    serving, and would aim tree measurement and termination at a stranger."""
    path = tmp_path / "state.json"
    write_fleet_state(
        [record(pid=os.getpid(), identity={"stable_identity": FORGED_IDENTITY})],
        path=path,
    )

    (reconciled,) = read_fleet_state(path)

    assert reconciled.state == "exited"
    assert reconciled.exit_reason is None
    # Written state still says running: liveness is re-derived, not repaired.
    assert load_fleet_state(path)[0].state == "running"


def test_reconciliation_prefers_the_stable_identity_over_the_process_group(
    tmp_path: Path,
) -> None:
    """A record carries both, and only one of them is reuse-safe.

    ``process_identity_record`` stores the bare process group under ``identity``
    for pre-upgrade readers, and a recycled pid that happened to lead the same
    group matches it. So the reconciliation has to compare ``stable_identity``,
    and this test fails if that precedence is ever flipped: the group here is
    genuinely ours, and only the stable value is forged.
    """
    path = tmp_path / "state.json"
    write_fleet_state(
        [
            record(
                pid=os.getpid(),
                identity={
                    "identity": os.getpgid(os.getpid()),
                    "stable_identity": FORGED_IDENTITY,
                },
            )
        ],
        path=path,
    )

    (reconciled,) = read_fleet_state(path)

    assert reconciled.state == "exited"


def test_a_live_instance_with_its_own_identity_is_reported_as_running(
    tmp_path: Path,
) -> None:
    """The other half of the acceptance criterion: reconciliation must not be a
    blanket "assume everything is dead", or the fleet would never report a
    healthy instance."""
    path = tmp_path / "state.json"
    write_fleet_state(
        [record(pid=os.getpid(), identity=instance_identity(os.getpid()))],
        path=path,
    )

    (reconciled,) = read_fleet_state(path)

    assert reconciled.state == "running"
    assert reconciled.exit_reason is None


def test_a_dead_pid_is_reported_as_exited(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_fleet_state([record(pid=dead_pid(), identity={})], path=path)

    (reconciled,) = read_fleet_state(path)

    assert reconciled.state == "exited"


def test_a_record_without_an_identity_falls_back_to_trusting_a_live_pid(
    tmp_path: Path,
) -> None:
    """``process_runtime``'s own posture: an identity that could not be read is
    not evidence of death, and killing a process one merely failed to identify is
    the worse error."""
    path = tmp_path / "state.json"
    write_fleet_state([record(pid=os.getpid(), identity={})], path=path)

    (reconciled,) = read_fleet_state(path)

    assert reconciled.state == "running"


def test_an_unreadable_identity_leaves_a_running_instance_running() -> None:
    """The third value. ``"unknown"`` means the identity could not be read *now*;
    treating it as a mismatch would flip a healthy instance to exited on a
    transient failure."""
    (reconciled,) = reconcile_fleet_state(
        [record()],
        is_running=lambda _pid: True,
        identity_match=lambda _recorded, _pid: "unknown",
    )

    assert reconciled.state == "running"


def test_an_already_exited_record_is_never_reprobed() -> None:
    """The supervisor never restarts an instance, so a stopped instance stays
    stopped. Probing it again could only produce a worse answer — a pid recycled
    into a live process would come back as running."""

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an exited record must not be probed")

    stopped = record(state="exited", exit_reason="signal")

    (reconciled,) = reconcile_fleet_state(
        [stopped],
        is_running=forbidden,
        identity_match=forbidden,
    )

    assert reconciled == stopped


def test_reconciliation_keeps_every_field_but_the_liveness(tmp_path: Path) -> None:
    written = record(pid=dead_pid(), exit_reason=None)
    (reconciled,) = reconcile_fleet_state([written])

    assert reconciled.state == "exited"
    for name in ("name", "pid", "workspace", "config_dir", "memory_limit_mb", "identity"):
        assert getattr(reconciled, name) == getattr(written, name)


def test_reconciliation_reports_each_instance_independently() -> None:
    live, gone = reconcile_fleet_state(
        [
            record(name="alpha", pid=os.getpid(), identity={}),
            record(name="beta", pid=dead_pid(), identity={}),
        ]
    )

    assert (live.name, live.state) == ("alpha", "running")
    assert (gone.name, gone.state) == ("beta", "exited")


def test_reading_does_not_rewrite_the_state_file(tmp_path: Path) -> None:
    """Readers run in a second shell while the supervisor owns the file. A read
    that wrote back would race the writer that reconciliation exists to avoid
    trusting."""
    path = tmp_path / "state.json"
    write_fleet_state([record(pid=dead_pid(), identity={})], path=path)
    before = (path.read_text(encoding="utf-8"), path.stat().st_mtime_ns)

    read_fleet_state(path)

    assert (path.read_text(encoding="utf-8"), path.stat().st_mtime_ns) == before


# --- refusals -----------------------------------------------------------------


def test_a_missing_state_file_is_refused(tmp_path: Path) -> None:
    """Not an empty fleet. A reader that answered "nothing is running" for a file
    it could not read invites an operator to start a second fleet on the same
    ports and workspaces."""
    with pytest.raises(FleetStateError) as caught:
        load_fleet_state(tmp_path / "absent.json")

    assert caught.value.kind == "io_error"


def test_a_file_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_bytes(b"\xff\xfe[]")

    with pytest.raises(FleetStateError) as caught:
        load_fleet_state(path)

    assert caught.value.kind == "io_error"
    assert "UTF-8" in caught.value.summary


def test_malformed_json_is_reported_with_a_line_and_column() -> None:
    with pytest.raises(FleetStateError) as caught:
        parse_fleet_state(STATE_PATH, "[{]")

    assert caught.value.kind == "invalid_json"
    assert "line 1, column 3" in caught.value.summary


def test_a_root_that_is_not_an_array_is_refused() -> None:
    error = refusal({"alpha": payload()})

    assert error.kind == "invalid_root"
    assert "must be a JSON array" in error.summary


def test_an_element_that_is_not_an_object_is_refused() -> None:
    error = refusal(["alpha"])

    assert error.kind == "invalid_record"
    assert error.issues == ("[0]: expected an object, but found str.",)


@pytest.mark.parametrize("missing", RECORD_FIELDS)
def test_every_required_field_is_required(missing: str) -> None:
    document = payload()
    del document[missing]

    error = refusal([document])

    assert error.kind == "invalid_record"
    assert f"missing required field(s) {missing}" in error.issues[0]


def test_a_misspelled_field_is_refused_rather_than_ignored() -> None:
    """The typo that matters most: silently ignoring ``memory_limt_mb`` would
    report a cap the supervisor is not enforcing."""
    document = payload()
    document["memory_limt_mb"] = document.pop("memory_limit_mb")

    error = refusal([document])

    assert "missing required field(s) memory_limit_mb" in error.issues[0]
    document["memory_limit_mb"] = 512
    assert "unknown field(s) memory_limt_mb" in refusal([document]).issues[0]


def test_an_invalid_instance_name_is_refused() -> None:
    error = refusal([payload(name="Alpha/../beta")])

    assert "name must match" in error.issues[0]


@pytest.mark.parametrize("pid", [0, -1, "4242", True, 4242.0, None])
def test_a_pid_that_is_not_a_positive_integer_is_refused(pid: object) -> None:
    """``0`` is the case worth naming: ``os.kill(0, …)`` signals the caller's own
    process group, so a zero pid reaching a later terminate would take the
    supervisor and the whole fleet down with it. ``True`` is an ``int`` in Python
    but is not a JSON integer."""
    error = refusal([payload(pid=pid)])

    assert "pid must be a positive integer" in error.issues[0]


def test_an_unknown_state_is_refused() -> None:
    error = refusal([payload(state="starting")])

    assert "state must be one of running, exited" in error.issues[0]


def test_an_unknown_exit_reason_is_refused() -> None:
    error = refusal([payload(state="exited", exit_reason="oom")])

    assert f"one of {', '.join(EXIT_REASONS)}" in error.issues[0]


def test_a_running_record_carrying_an_exit_reason_is_refused() -> None:
    """A contradiction no writer here can produce, so a file containing one is
    not a file this module wrote."""
    error = refusal([payload(exit_reason="exit")])

    assert "running instance cannot carry an exit_reason" in error.issues[0]


@pytest.mark.parametrize("key", ["workspace", "config_dir"])
def test_a_relative_directory_is_refused(key: str) -> None:
    error = refusal([payload(**{key: "workspace"})])

    assert f"{key} must be an absolute path" in error.issues[0]


@pytest.mark.parametrize("key", ["workspace", "config_dir"])
def test_a_directory_that_is_not_a_string_is_refused(key: str) -> None:
    error = refusal([payload(**{key: ""})])

    assert f"{key} must be a non-empty string" in error.issues[0]


@pytest.mark.parametrize("limit", [0, -1, "512", True])
def test_a_memory_limit_that_is_not_a_positive_integer_is_refused(limit: object) -> None:
    error = refusal([payload(memory_limit_mb=limit)])

    assert "memory_limit_mb must be a positive integer" in error.issues[0]


def test_an_identity_of_the_wrong_type_is_refused() -> None:
    error = refusal([payload(stable_identity=["darwin", 1])])

    assert "stable_identity must be a string, an integer, or null" in error.issues[0]


def test_an_absent_identity_is_accepted() -> None:
    document = payload()
    for key in IDENTITY_FIELDS:
        document.pop(key, None)

    (parsed,) = parse_fleet_state(STATE_PATH, json.dumps([document]))

    assert parsed.identity == {}
    assert parsed.recorded_identity is None


def test_a_repeated_instance_name_is_refused() -> None:
    """Records are addressed by name downstream, so one entry would shadow the
    other and a live process tree would drop out of the fleet's view."""
    error = refusal([payload(), payload(pid=99)])

    assert "alpha appears more than once" in error.issues[0]


def test_one_bad_record_refuses_the_whole_file() -> None:
    """No partial answers. Dropping the unreadable record would leave the
    supervisor reporting the rest as healthy while having lost track of one
    running instance."""
    error = refusal([payload(), payload(name="beta", pid=0)])

    assert error.kind == "invalid_record"
    assert len(error.issues) == 1
    assert "beta" in error.issues[0]


def test_every_problem_in_a_record_is_reported_at_once() -> None:
    error = refusal([payload(pid=0, state="starting", memory_limit_mb=0)])

    assert len(error.issues) == 3


def test_rendering_stops_after_ten_issues() -> None:
    error = refusal([payload(name=f"i{index}", pid=0) for index in range(12)])

    text = str(error)
    assert len(error.issues) == 12
    assert text.count("pid must be a positive integer") == 10
    assert "… and 2 more issue(s)" in text


def test_a_refusal_never_echoes_the_value_it_rejected() -> None:
    """Same reason ``nanobot/config/errors.py`` passes ``include_input=False``.
    Nothing in a state file is meant to be a credential, but a corrupt or
    tampered file can contain anything, and an error is the one place its
    contents get copied to a terminal and a log."""
    error = refusal([payload(workspace="sk-live-secret", state="sk-live-secret")])

    rendered = str(error)
    assert "sk-live" not in rendered
    assert not any("sk-live" in issue for issue in error.issues)
    assert "sk-live" not in error.summary


# --- purity -------------------------------------------------------------------


def test_parsing_touches_no_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    """``parse_fleet_state`` takes a ``Path`` purely so its errors can name the
    file. This keeps that true: a future edit that stat'd a recorded workspace —
    or, worse, probed a recorded pid — would fail here rather than quietly turn
    parsing into an operation with side effects."""

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("parsing must not touch the filesystem")

    for module, name in (
        (builtins, "open"),
        (os, "open"),
        (os, "stat"),
        (os, "lstat"),
        (os, "listdir"),
        (os, "scandir"),
        (os, "mkdir"),
        (os, "makedirs"),
    ):
        monkeypatch.setattr(module, name, forbidden)

    (parsed,) = parse_fleet_state(STATE_PATH, json.dumps([payload()]))

    assert parsed == record()
    # And the refusal paths are as pure as the accepting one.
    assert refusal({}).kind == "invalid_root"
    assert refusal([payload(pid=0)]).kind == "invalid_record"

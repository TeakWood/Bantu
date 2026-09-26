"""The supervisor's own record of the fleet: written atomically, reconciled on read.

One file, one writer. The supervisor writes it on every transition; ``nanobot
fleet status`` and any other reader only ever reads it. That asymmetry is the
whole design, and the two halves of this module exist because of it.

*Why the supervisor owns the file, and why it lives where it does.* The obvious
alternative was to reuse each instance's existing ``<config_dir>/run/gateway*.json``
— the per-service state ``ManagedProcessRuntime`` already maintains. It was
rejected on a security ground, not a tidiness one: that file lives inside the
instance's own config directory, which the instance can write. An instance that
could write its own status could report a pid of its choosing, and the supervisor
would then measure, report and signal whatever process the *instance* named. So
fleet state is one aggregate file beside the fleet document, outside every
instance's reach, and :mod:`nanobot.fleet.profile` denies its path in *every*
instance profile — see :func:`fleet_state_path`.

The contents make that denial necessary rather than merely tidy. Each record
carries an instance's workspace and config directory, so the file as a whole is a
map of where every peer keeps its sessions — effectively a copy of what the fleet
document says. Leaving it readable would hand an instance the very index the
fleet file deny exists to withhold, by a second route.

*Why liveness is reconciled on read rather than trusted.* The last write is a
statement about the past. A reader in another shell cannot ask the supervisor
anything, and between the write and the read an instance may have exited and its
pid may have been handed to an unrelated process. A pid is therefore not an
identity: reporting a recycled pid as running would tell an operator a dead
instance is serving, and — worse, downstream — would aim a tree measurement or a
termination at somebody else's process. :func:`reconcile_fleet_state` re-derives
``state`` from ``process_is_running`` plus the PID-reuse-safe identity recorded
next to the pid, and it is a pure function of what it is given: reading never
writes the file back.

*Mode 0600 on every write, not the mode the file happens to have.*
``nanobot/utils/helpers.py`` ``_write_text_atomic`` copies an existing file's mode
onto its replacement, which is right for a user-owned artefact and wrong here: a
mode is a security property of this file, not an operator preference, and
preserving a relaxed one would quietly keep it relaxed for the life of the fleet.
The durability idiom is otherwise the one already used there and in
``process_runtime.py``: write a temporary file in the same directory, ``fsync``
it, ``os.replace`` it into place — which is atomic, so a reader in another shell
sees either the whole previous state or the whole new one, never a half-written
array — then ``fsync`` the directory.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, get_args

from nanobot.fleet.config import INSTANCE_NAME_PATTERN
from nanobot.fleet.instance import instance_identity_match
from nanobot.fleet.validate import ResolvedInstance
from nanobot.process_runtime import process_is_running

#: Appended to the fleet file's *whole* name, not its stem: two fleet documents
#: in one directory named ``fleet`` and ``fleet.json`` would otherwise share a
#: state file, and each supervisor would clobber the other's pids.
STATE_SUFFIX = ".state.json"

#: The only mode this file is ever written with. It lists every instance's
#: directories, so it is readable by its owner and nobody else.
STATE_FILE_MODE = 0o600

InstanceState = Literal["running", "exited"]
ExitReason = Literal["memory", "signal", "exit"]

#: Derived from the annotations above so the reader cannot accept a state or a
#: reason the writer has no way to produce.
INSTANCE_STATES: tuple[str, ...] = get_args(InstanceState)
EXIT_REASONS: tuple[str, ...] = get_args(ExitReason)

#: The seven facts the supervisor records about each instance.
RECORD_FIELDS: tuple[str, ...] = (
    "name",
    "pid",
    "state",
    "exit_reason",
    "workspace",
    "config_dir",
    "memory_limit_mb",
)

#: The keys ``process_runtime.process_identity_record`` may produce. Stored flat
#: beside the seven fields rather than nested, so the recorded shape *is* that
#: function's output and a reader can apply the same precedence
#: ``ManagedProcessRuntime`` applies to its own state files.
IDENTITY_FIELDS: tuple[str, ...] = ("identity", "stable_identity")

StateErrorKind = Literal[
    "io_error",
    "invalid_json",
    "invalid_root",
    "invalid_record",
]

#: The two probes :func:`reconcile_fleet_state` needs, named so they can be
#: substituted in a test without loosening the production signature.
IsRunning = Callable[[int], bool]
IdentityMatch = Callable[[object, int], Literal["match", "mismatch", "unknown"]]

_INSTANCE_NAME_RE = re.compile(INSTANCE_NAME_PATTERN)

_MAX_RENDERED_ISSUES = 10


class FleetStateError(ValueError):
    """The fleet state file cannot be read, written, or believed.

    Always a refusal, never a partial answer. A reader that returned "no
    instances" for a state file it could not parse would tell an operator the
    fleet is stopped, and the next thing that operator does is start a second
    fleet on the same ports and workspaces. Dropping a single unreadable record
    is worse still: the supervisor would lose track of one live process tree
    while reporting the rest as healthy.

    Messages name the offending key and the type found, never the value, for the
    reason ``nanobot/config/errors.py`` passes ``include_input=False``.
    """

    def __init__(
        self,
        path: Path,
        *,
        kind: StateErrorKind,
        summary: str,
        issues: Sequence[str] = (),
    ) -> None:
        self.path = path
        self.kind = kind
        self.summary = summary
        self.issues = tuple(issues)
        super().__init__(summary)

    def __str__(self) -> str:
        lines = [f"Invalid fleet state file: {self.path}", "", self.summary]
        for issue in self.issues[:_MAX_RENDERED_ISSUES]:
            lines.extend(("", f"  {issue}"))
        remaining = len(self.issues) - _MAX_RENDERED_ISSUES
        if remaining > 0:
            lines.extend(("", f"  … and {remaining} more issue(s)"))
        return "\n".join(lines)


@dataclass(frozen=True)
class InstanceRecord:
    """What the supervisor records about one instance.

    The seven fields the fleet's status contract names, plus the identity record
    that makes ``pid`` trustworthy. ``exit_reason`` is ``None`` while an instance
    runs and may stay ``None`` after it stops: a reader that discovers an
    instance is gone knows *that* it is gone and cannot honestly say why, and the
    three reasons are claims only the supervisor is in a position to make —
    ``"signal"`` and ``"exit"`` from reaping the child, ``"memory"`` from
    enforcing the cap.
    """

    name: str
    pid: int
    state: InstanceState
    exit_reason: ExitReason | None
    workspace: Path
    config_dir: Path
    memory_limit_mb: int
    #: ``process_identity_record`` output for ``pid``, as recorded at launch. An
    #: empty mapping means the platform could not read an identity, which
    #: reconciliation treats the way ``process_runtime`` does — see
    #: :attr:`recorded_identity`.
    identity: Mapping[str, str | int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse an identity mapping carrying a key the record does not own.

        Unreachable from ``process_identity_record``, which produces exactly
        :data:`IDENTITY_FIELDS`, and checked anyway because :meth:`payload`
        merges the mapping flat: a stray ``"pid"`` key would overwrite the
        record's real pid on its way to disk, and the supervisor would go on to
        measure and signal that value instead.
        """
        if unknown := sorted(set(self.identity) - set(IDENTITY_FIELDS)):
            raise ValueError(
                f"{self.name}: an identity record may only carry "
                f"{', '.join(IDENTITY_FIELDS)}, but found {', '.join(unknown)}"
            )

    @property
    def recorded_identity(self) -> object:
        """The identity value to compare against the process now holding ``pid``.

        ``stable_identity`` is preferred and ``identity`` is the fallback, which
        is exactly the precedence ``ManagedProcessRuntime`` applies to its own
        records — and it matters: on macOS the bare ``identity`` is only the
        process group, which a recycled pid leading the same group would match.
        ``None`` (including an absent identity) compares as a match, so a record
        from a platform that cannot read identities falls back to trusting a live
        pid rather than declaring a healthy instance dead.
        """
        recorded = self.identity.get("stable_identity")
        if recorded is None:
            recorded = self.identity.get("identity")
        return recorded

    def exited(self, reason: ExitReason | None = None) -> InstanceRecord:
        """This record as an exited one, keeping any reason already recorded.

        ``reason`` is only applied when nothing better is known. The supervisor
        observes why an instance stopped at the moment it stops; a later reader
        must not overwrite ``"memory"`` with a blander guess.
        """
        return replace(
            self,
            state="exited",
            exit_reason=self.exit_reason if self.exit_reason is not None else reason,
        )

    def payload(self) -> dict[str, object]:
        """This record as the JSON object written to the state file.

        Keys are ``snake_case``, deliberately unlike the fleet *document*'s
        camelCase: this file is not operator-authored, and its field names are
        the ones ``nanobot fleet status --json`` publishes.
        """
        return {
            "name": self.name,
            "pid": self.pid,
            "state": self.state,
            "exit_reason": self.exit_reason,
            "workspace": str(self.workspace),
            "config_dir": str(self.config_dir),
            "memory_limit_mb": self.memory_limit_mb,
            **self.identity,
        }


def status_payload(record: InstanceRecord) -> dict[str, object]:
    """One record as ``nanobot fleet status --json`` publishes it.

    Exactly :data:`RECORD_FIELDS`, projected out of :meth:`InstanceRecord.payload`
    rather than assembled a second time, so a published field cannot drift in
    name or in type from the one the supervisor writes. The command that renders
    this therefore makes no decision about what a fleet's status *is*; that
    belongs here, beside the writer.

    The identity record is deliberately not published, and its absence is the
    contract rather than an oversight. It is an internal token for telling one
    process from another that happens to hold the same pid: its shape is
    platform-specific — on macOS the bare ``identity`` is a process *group* —
    so publishing it would freeze a private format into an interface, and would
    invite a consumer to signal a group that only the supervisor and
    :mod:`nanobot.fleet.stop` are in a position to address.

    Note what is absent for a second reason: the supervisor's own pid. Nothing in
    this file records it — :func:`nanobot.fleet.stop._supervisor` has to recover
    it from parentage precisely because the state file does not know it — so an
    instance's ``pid`` cannot be the supervisor's simply because there is nowhere
    for that number to come from.
    """
    payload = record.payload()
    return {key: payload[key] for key in RECORD_FIELDS}


def fleet_state_path(fleet_path: str | Path) -> Path:
    """Where the supervisor keeps its state for the fleet declared at ``fleet_path``.

    The single definition of that location. :mod:`nanobot.fleet.profile` denies
    whatever it is told to deny, so if the supervisor and the profile builder
    derived the path separately they could disagree, and the disagreement would
    be silent in the direction that matters — a deny naming a path nothing writes
    confines nothing, and ``sandbox-exec`` reports no error. Both call this.

    The parent directory is canonicalised but the file name is not resolved,
    mirroring :func:`nanobot.fleet.paths.instance_paths`: the state file may not
    exist yet, and the profile builder requires a canonical path because the
    kernel matches Seatbelt rules against real paths.
    """
    path = Path(fleet_path).expanduser()
    return path.parent.resolve(strict=False) / f"{path.name}{STATE_SUFFIX}"


def running_record(
    instance: ResolvedInstance,
    *,
    pid: int,
    identity: Mapping[str, str | int | None] | None = None,
) -> InstanceRecord:
    """The record for an instance that has just been started.

    ``identity`` is normally :func:`nanobot.fleet.instance.instance_identity` for
    the same pid, taken at launch. It is a separate argument rather than read
    here because it must describe the process as it was *when it was spawned*:
    read later, it would describe whatever holds the pid by then, which is the
    very substitution it exists to detect.
    """
    return InstanceRecord(
        name=instance.name,
        pid=pid,
        state="running",
        exit_reason=None,
        workspace=instance.workspace,
        config_dir=instance.config_dir,
        memory_limit_mb=instance.entry.memory_limit_mb,
        identity=dict(identity or {}),
    )


def write_fleet_state(
    records: Iterable[InstanceRecord],
    *,
    path: Path,
) -> None:
    """Write the whole fleet's state to ``path`` atomically, at mode 0600.

    The file is replaced, never edited in place, so a reader in another shell
    sees one complete array or the other and never a truncated one. The
    temporary file is created in the destination directory (``os.replace`` is
    only atomic within a filesystem) at mode 0600, and the mode is set on the
    temporary file rather than afterwards, so the contents are never briefly
    visible to another user.

    Raises:
        FleetStateError: the records cannot be serialized (``invalid_record``),
            or the directory is missing or unwritable (``io_error``). The
            existing file is left untouched in both cases.
    """
    payload = [record.payload() for record in records]
    try:
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise FleetStateError(
            path,
            kind="invalid_record",
            summary=f"The fleet state could not be serialized: {_sentence(str(exc))}",
        ) from exc

    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
    except OSError as exc:
        raise FleetStateError(
            path,
            kind="io_error",
            summary=f"Unable to write beside {path.parent}: {_sentence(_detail(exc))}",
        ) from exc

    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            # ``mkstemp`` already creates at 0600; setting it explicitly on the
            # descriptor means the mode is this module's decision rather than a
            # detail of the primitive, and it is set before anything is written.
            os.fchmod(handle.fileno(), STATE_FILE_MODE)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except OSError as exc:
        raise FleetStateError(
            path,
            kind="io_error",
            summary=f"Unable to write the fleet state: {_sentence(_detail(exc))}",
        ) from exc
    finally:
        # Present only if the replace did not happen; the replace renames it away.
        temporary_path.unlink(missing_ok=True)

    # The rename is durable only once the directory entry is. Best effort: a
    # platform that cannot fsync a directory still has the file.
    with suppress(OSError, NotImplementedError):
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def ensure_fleet_state_file(path: Path) -> Path:
    """Create an empty state file at ``path`` if it does not exist yet.

    Called before :func:`nanobot.fleet.profile.build_fleet_profiles`, and the
    ordering is load-bearing rather than cosmetic: that builder refuses to emit a
    deny for a path that does not exist, because Seatbelt would accept such a
    rule, match nothing, confine nothing and report no error. The state file does
    not exist before a fleet's first start, so something has to lay it down
    first, and laying it down empty is the honest initial state — no instance has
    been started yet.

    Idempotent, and never truncates: an existing file is left exactly as it is,
    so calling this cannot lose the state of a running fleet.

    Raises:
        FleetStateError: the file could not be created (``io_error``).
    """
    if path.exists():
        return path
    write_fleet_state((), path=path)
    return path


def read_fleet_state(
    path: Path,
    *,
    is_running: IsRunning | None = None,
    identity_match: IdentityMatch | None = None,
) -> tuple[InstanceRecord, ...]:
    """Read the fleet's state and reconcile every record's liveness.

    The function a reader should use: what is on disk describes the last
    transition the supervisor observed, and :func:`reconcile_fleet_state` turns
    it into what is true now. The file is not rewritten — reading fleet state
    from a second shell must not race the supervisor's own writes.

    ``is_running`` and ``identity_match`` are injection points for tests; see
    :func:`reconcile_fleet_state`.

    Raises:
        FleetStateError: any failure of :func:`load_fleet_state`.
    """
    return reconcile_fleet_state(
        load_fleet_state(path),
        is_running=is_running,
        identity_match=identity_match,
    )


def load_fleet_state(path: Path) -> tuple[InstanceRecord, ...]:
    """Read the state file exactly as it was written, probing nothing.

    Separated from the reconciliation for the reason
    :mod:`nanobot.fleet.config` separates ``load_fleet_file`` from
    ``parse_fleet_file``: it lets the file's contents be examined as a record of
    what the supervisor said, which is what a test of the *writer* needs, and it
    keeps the liveness decision in one place.

    Raises:
        FleetStateError: the file is missing, unreadable, not valid UTF-8
            (``io_error``), or any failure :func:`parse_fleet_state` raises.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FleetStateError(
            path,
            kind="io_error",
            summary="The file is not valid UTF-8.",
        ) from exc
    except OSError as exc:
        raise FleetStateError(
            path,
            kind="io_error",
            summary=f"Unable to read the file: {_sentence(_detail(exc))}",
        ) from exc
    return parse_fleet_state(path, text)


def parse_fleet_state(path: Path, text: str) -> tuple[InstanceRecord, ...]:
    """Parse state file ``text`` into records, strictly.

    Pure: ``path`` is carried for error messages only and is never opened,
    resolved or stat'd, so the shape of a state document can be checked without
    touching a filesystem — a test pins that by making every filesystem
    primitive raise while this runs.

    Strict because the only legitimate writer of this file is
    :func:`write_fleet_state`. An unknown key, a missing one or a wrong type
    means the file did not come from the supervisor, and the values in it drive
    process measurement and termination. Two checks are worth naming: a pid must
    be a positive integer, because ``os.kill(0, …)`` signals the *caller's* whole
    process group, and a record cannot be ``running`` while carrying an exit
    reason, which is a contradiction no writer here can produce.

    Raises:
        FleetStateError: ``invalid_json`` (syntax, reported with line and
            column), ``invalid_root`` (the top level is not an array) or
            ``invalid_record`` (one issue per malformed record).
    """
    try:
        data: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FleetStateError(
            path,
            kind="invalid_json",
            summary=(
                f"JSON syntax error at line {exc.lineno}, column {exc.colno}: "
                f"{_sentence(exc.msg)}"
            ),
        ) from exc

    if not isinstance(data, list):
        raise FleetStateError(
            path,
            kind="invalid_root",
            summary=(
                f"The top level of a fleet state file must be a JSON array, but "
                f"found {type(data).__name__}."
            ),
        )

    records: list[InstanceRecord] = []
    issues: list[str] = []
    seen: set[str] = set()
    for index, element in enumerate(data):
        record, element_issues = _parse_record(index, element)
        if record is None:
            issues.extend(element_issues)
            continue
        if record.name in seen:
            # Records are addressed by name downstream, so a repeated name would
            # let one entry shadow the other and hide a live process tree.
            issues.append(f"[{index}]: {record.name} appears more than once.")
            continue
        seen.add(record.name)
        records.append(record)

    if issues:
        raise FleetStateError(
            path,
            kind="invalid_record",
            summary=f"Found {len(issues)} invalid record(s).",
            issues=issues,
        )
    return tuple(records)


def reconcile_fleet_state(
    records: Iterable[InstanceRecord],
    *,
    is_running: IsRunning | None = None,
    identity_match: IdentityMatch | None = None,
) -> tuple[InstanceRecord, ...]:
    """Re-derive every record's ``state`` from the processes that exist now.

    Three-valued on purpose, because a pid can be in three states and only one
    of them is "running":

    * the pid is gone, or is a zombie the supervisor has not reaped — exited;
    * the pid is alive but the process holding it is not the one that was
      recorded — exited, because the pid has been recycled. This is the case a
      pid-only check gets wrong, and getting it wrong means reporting a dead
      instance as serving and aiming later measurement or termination at an
      unrelated process;
    * the pid is alive and the identity cannot be read *right now* — left
      running. A transient read failure is not evidence of death, and treating it
      as such would flip a healthy instance to exited on a whim.

    A record already marked ``exited`` is returned untouched and its process is
    not probed at all. The supervisor never restarts an instance, so a stopped
    instance stays stopped and stays reported with the reason that was observed
    when it stopped; re-probing could only turn that into a worse answer if the
    pid were recycled into a live process.

    Pure with respect to the file: nothing is written, and the input records are
    frozen, so callers keep whatever they passed in.

    Args:
        records: the records as written, normally from :func:`load_fleet_state`.
        is_running: liveness probe, defaulting to
            ``process_runtime.process_is_running`` — which is also what rules out
            an unreaped zombie.
        identity_match: identity comparison, defaulting to
            :func:`nanobot.fleet.instance.instance_identity_match`. Both are
            injectable so the ``"unknown"`` branch can be exercised without
            arranging a pid whose identity is momentarily unreadable.
    """
    alive = process_is_running if is_running is None else is_running
    matches = instance_identity_match if identity_match is None else identity_match
    reconciled: list[InstanceRecord] = []
    for record in records:
        if record.state == "exited":
            reconciled.append(record)
            continue
        if not alive(record.pid):
            reconciled.append(record.exited())
            continue
        if matches(record.recorded_identity, record.pid) == "mismatch":
            reconciled.append(record.exited())
            continue
        reconciled.append(record)
    return tuple(reconciled)


def _parse_record(index: int, element: object) -> tuple[InstanceRecord | None, list[str]]:
    """Validate one element of the state array; ``(None, issues)`` if it is wrong."""
    where = f"[{index}]"
    if not isinstance(element, dict):
        return None, [f"{where}: expected an object, but found {type(element).__name__}."]

    keys = set(element)
    if missing := sorted(set(RECORD_FIELDS) - keys):
        return None, [f"{where}: missing required field(s) {', '.join(missing)}."]
    if unknown := sorted(keys - set(RECORD_FIELDS) - set(IDENTITY_FIELDS)):
        return None, [f"{where}: unknown field(s) {', '.join(unknown)}."]

    issues: list[str] = []
    name = element["name"]
    if not isinstance(name, str) or not _INSTANCE_NAME_RE.fullmatch(name):
        issues.append(f"{where}: name must match {INSTANCE_NAME_PATTERN}.")
    else:
        where = f"[{index}] {name}"

    pid = element["pid"]
    if not _is_int(pid) or pid <= 0:
        # A pid of 0 is not merely wrong: signalling it would hit the
        # supervisor's own process group, and negatives address groups too.
        issues.append(f"{where}: pid must be a positive integer.")

    state = element["state"]
    if state not in INSTANCE_STATES:
        issues.append(f"{where}: state must be one of {', '.join(INSTANCE_STATES)}.")

    reason = element["exit_reason"]
    if reason is not None and reason not in EXIT_REASONS:
        issues.append(
            f"{where}: exit_reason must be null or one of {', '.join(EXIT_REASONS)}."
        )
    elif reason is not None and state == "running":
        issues.append(f"{where}: a running instance cannot carry an exit_reason.")

    paths: dict[str, Path] = {}
    for key in ("workspace", "config_dir"):
        value = element[key]
        if not isinstance(value, str) or not value:
            issues.append(f"{where}: {key} must be a non-empty string.")
        elif not Path(value).is_absolute():
            issues.append(f"{where}: {key} must be an absolute path.")
        else:
            paths[key] = Path(value)

    limit = element["memory_limit_mb"]
    if not _is_int(limit) or limit <= 0:
        issues.append(f"{where}: memory_limit_mb must be a positive integer.")

    identity: dict[str, str | int | None] = {}
    for key in IDENTITY_FIELDS:
        if key not in element:
            continue
        value = element[key]
        if value is not None and not isinstance(value, str) and not _is_int(value):
            issues.append(f"{where}: {key} must be a string, an integer, or null.")
        else:
            identity[key] = value

    if issues:
        return None, issues
    return (
        InstanceRecord(
            name=name,
            pid=pid,
            state=state,
            exit_reason=reason,
            workspace=paths["workspace"],
            config_dir=paths["config_dir"],
            memory_limit_mb=limit,
            identity=identity,
        ),
        [],
    )


def _is_int(value: object) -> bool:
    """Whether ``value`` is a JSON integer. ``bool`` is not one, despite ``int``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _detail(exc: OSError) -> str:
    """The most useful description an ``OSError`` offers."""
    return exc.strerror or type(exc).__name__


def _sentence(message: str) -> str:
    """Trim and terminate a fragment so it reads as a sentence."""
    message = message.strip()
    if message and message[-1] not in ".!?":
        message += "."
    return message

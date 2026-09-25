"""The fleet document: what a fleet file may declare, and how it refuses.

A fleet file is a single JSON object naming the instances one supervisor runs.
Each entry says where that instance's nanobot config lives, which mode to start
it in, how much memory its whole process tree may use, and which environment
variable *names* it is allowed to inherit. Those four facts are the inputs to
every confinement decision the supervisor makes, so the document is parsed
strictly and refused loudly rather than defaulted into something plausible.

Three consequences shape the models below.

*Every confinement control is required.* ``config``, ``mode`` and
``memoryLimitMb`` have no defaults. There is no safe value to invent for a
memory cap or a launch mode, and an instance that silently ran uncapped because
a key was omitted would defeat the point of declaring a fleet at all. ``env`` is
the one optional field, and its default — the empty list — is the fail-closed
one: an instance inherits only the minimal base environment unless the document
names more.

*Unknown keys are refused.* ``Base`` does not forbid extras and the root config
schema explicitly allows them, but a fleet file cannot afford that: a misspelled
``memoryLimtMb`` under the usual "ignore" policy would be dropped on the floor
and the instance would start unconfined. ``extra="forbid"`` turns every typo
into a refusal.

*Environment entries are names, never values.* ``env`` holds variable names, and
each is constrained to the shape of an identifier. That is not cosmetic: it is
what catches an operator who writes ``"OPENAI_API_KEY=sk-…"`` and thereby commits
a live credential to a file that is meant to contain none.

Errors follow ``nanobot/config/errors.py`` rather than reimplementing it. That
module's ``ConfigIssue.location`` already redacts user-controlled key names, and
``validation_issues`` already passes ``include_input=False`` so a rejected value
never reaches the rendered message. Both are security-relevant; a second copy
here could drift and start leaking. Unlike ``nanobot.fleet.paths`` — which stays
off the config package because ``load_config`` has global side effects and the
``ensure_dir`` helpers create directories — nothing imported here touches the
filesystem or mutates global state.

Scope: this module knows nothing about paths. It never resolves, canonicalises,
compares or stats anything; ``config`` is validated as a non-empty string and
handed on. Parsing is a pure function of the document text, and a test pins that
by making every filesystem primitive raise while it runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, ValidationError

from nanobot.config.errors import ConfigIssue, validation_issues
from nanobot.config_base import Base

FleetErrorKind = Literal[
    "invalid_json",
    "invalid_root",
    "invalid_schema",
    "io_error",
]

#: Instance names are directory-safe, shell-safe and case-insensitively unique
#: by construction: they become process labels, state-file keys and Seatbelt
#: profile names, none of which want spaces, slashes or case collisions.
INSTANCE_NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"

#: The POSIX shape of an environment variable name. Anything else — most
#: importantly anything containing ``=`` — is a value that does not belong here.
ENV_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"

#: Pydantic's synthetic location part for a rejected mapping key. It is an
#: internal marker, not a config identifier, so it is rewritten rather than left
#: to render as ``<redacted>`` and imply something was hidden.
_KEY_LOCATION_MARKER = "[key]"

_MAX_RENDERED_ISSUES = 10

InstanceName = Annotated[str, StringConstraints(pattern=INSTANCE_NAME_PATTERN)]
EnvVarName = Annotated[str, StringConstraints(pattern=ENV_NAME_PATTERN)]
InstanceMode = Literal["gateway", "serve"]


class FleetInstance(Base):
    """One instance entry. The dict key that carries its name is not part of it."""

    model_config = ConfigDict(extra="forbid")

    config: str = Field(min_length=1)
    mode: InstanceMode
    memory_limit_mb: int = Field(gt=0)
    env: list[EnvVarName] = Field(default_factory=list)


class FleetFile(Base):
    """A whole fleet document: instance names mapped to their entries."""

    model_config = ConfigDict(extra="forbid")

    instances: dict[InstanceName, FleetInstance] = Field(min_length=1)


class FleetConfigError(ValueError):
    """A structured, user-safe fleet document failure.

    Mirrors ``ConfigLoadError``: the same ``kind`` discrimination, the same
    issue list, and the same rendering that stops after ten issues so a badly
    broken document produces a readable error instead of a wall of text.
    """

    def __init__(
        self,
        path: Path,
        *,
        kind: FleetErrorKind,
        summary: str,
        issues: tuple[ConfigIssue, ...] = (),
    ) -> None:
        self.path = path
        self.kind = kind
        self.summary = summary
        self.issues = issues
        super().__init__(summary)

    def __str__(self) -> str:
        lines = [f"Invalid fleet file: {self.path}", "", self.summary]
        for issue in self.issues[:_MAX_RENDERED_ISSUES]:
            lines.extend(("", f"  {issue.location}", f"    {issue.message}"))
        remaining = len(self.issues) - _MAX_RENDERED_ISSUES
        if remaining > 0:
            lines.extend(("", f"  … and {remaining} more issue(s)"))
        return "\n".join(lines)


def load_fleet_file(path: Path) -> FleetFile:
    """Read and parse a fleet file.

    The only function here that touches the filesystem, and it does nothing but
    read: callers validate fleet documents for instances they may go on to
    refuse, so nothing is created, moved or written.

    Raises:
        FleetConfigError: the file is unreadable or not valid UTF-8
            (``io_error``), or any failure ``parse_fleet_file`` raises.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FleetConfigError(
            path,
            kind="io_error",
            summary="The file is not valid UTF-8.",
        ) from exc
    except OSError as exc:
        detail = exc.strerror or type(exc).__name__
        raise FleetConfigError(
            path,
            kind="io_error",
            summary=f"Unable to read the file: {_sentence(detail)}",
        ) from exc
    return parse_fleet_file(path, text)


def parse_fleet_file(path: Path, text: str) -> FleetFile:
    """Parse fleet document ``text`` into typed models.

    Pure: ``path`` is carried for error messages only and is never opened,
    resolved or stat'd. Splitting the read out of the parse is what lets a test
    prove the document shape can be checked without any filesystem access.

    Raises:
        FleetConfigError: with ``kind`` ``invalid_json`` (syntax, reported with
            line and column), ``invalid_root`` (top level is not an object) or
            ``invalid_schema`` (one issue per rejected field, values withheld).
    """
    try:
        data: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FleetConfigError(
            path,
            kind="invalid_json",
            summary=(
                f"JSON syntax error at line {exc.lineno}, column {exc.colno}: "
                f"{_sentence(exc.msg)}"
            ),
        ) from exc

    if not isinstance(data, dict):
        raise FleetConfigError(
            path,
            kind="invalid_root",
            summary="The top level of a fleet file must be a JSON object.",
            issues=(
                ConfigIssue(
                    path=(),
                    message=f"Expected an object, but found {type(data).__name__}.",
                ),
            ),
        )

    try:
        return FleetFile.model_validate(data)
    except ValidationError as exc:
        issues = _fleet_issues(exc)
        raise FleetConfigError(
            path,
            kind="invalid_schema",
            summary=f"Found {len(issues)} invalid setting(s).",
            issues=issues,
        ) from exc


def _fleet_issues(error: ValidationError) -> tuple[ConfigIssue, ...]:
    """``validation_issues`` plus a readable message for a rejected instance name.

    Pydantic reports a bad mapping key at ``(…, <the key>, "[key]")`` with the
    generic "String should match pattern" text. Left alone that renders as a
    trailing ``<redacted>``, which reads as though a value were withheld when
    the marker is merely pydantic's internal notation. The location loses the
    marker and the message names what is actually wrong; the offending key stays
    in the location, where ``ConfigIssue`` redaction already governs it.
    """
    issues: list[ConfigIssue] = []
    for issue in validation_issues(error):
        if issue.path and issue.path[-1] == _KEY_LOCATION_MARKER:
            issues.append(
                ConfigIssue(
                    path=issue.path[:-1],
                    message=f"Instance name must match {INSTANCE_NAME_PATTERN}.",
                )
            )
        else:
            issues.append(issue)
    return tuple(issues)


def _sentence(message: str) -> str:
    """Trim and terminate a fragment so it reads as a sentence."""
    message = message.strip()
    if message and message[-1] not in ".!?":
        message += "."
    return message

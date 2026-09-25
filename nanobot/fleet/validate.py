"""Refuse a fleet whose instances would not actually be isolated.

``nanobot.fleet.config`` says whether a fleet *document* is well formed and
``nanobot.fleet.paths`` says which two directories one instance owns. Neither can
see the property that matters: that no instance's filesystem view overlaps any
other's. This module combines them and refuses the layouts that would produce a
fleet which looks confined and is not.

Why each rule is a refusal rather than a warning:

*A config path must be absolute after expansion, and must exist.* This is the
security rule, not an ergonomic one. Every deny rule in an instance's Seatbelt
profile is a peer's canonical directory, and — verified on the pinned host — a
profile naming a path that does not resolve starts the process cleanly, confines
nothing, and reports no error at all. There is no failure signal downstream of a
bad path, so the failure has to happen here. A relative path is refused for the
same reason one step earlier: it would silently resolve against whatever
directory the supervisor happened to be started from, which is not a property of
the fleet document and not something the operator declared.

*No two instances may share, contain, or be contained by each other's
directories.* Nesting is the interesting case. The profile builder denies
``(subpath <peer dir>)``, and a subpath rule covers everything beneath it — so if
one instance's workspace sits inside another's config dir, the deny that walls
the peer off also walls the instance off from its own files, and the allow that
lets it reach its own files punches a hole straight into the peer. Either way the
pair is not separable by the profile, so the fleet is refused before one is built.

*Within one instance, a workspace inside its own config dir is normal.* It is the
default layout: ``~/.nanobot-x/workspace`` under ``~/.nanobot-x/``. The check
above is therefore strictly about *distinct* instances.

*A config dir inside its own workspace is refused*, because nanobot itself
refuses it — ``JsonlSessionStore`` raises ``RuntimeError`` for exactly this shape
("session storage must be outside the agent workspace"). Left to run, the
instance would put its session store inside the directory its own agent can
freely write, so the agent could rewrite its own transcripts. Catching it during
validation turns a crash partway through starting a fleet into a refusal that
starts nothing.

*Duplicate ports are reported.* Two instances that bind the same TCP port cannot
both come up, and the loser fails after the supervisor has already started it.
Whether this should refuse or merely warn is the open question recorded in the
process-isolation ADR; it is implemented here as a reported error, which is the
fail-closed reading.

Two deliberate narrowings in the port rule, both to avoid refusing a fleet that
would in fact run. Only the port an instance's declared ``mode`` actually binds
is compared — ``gateway.port`` for ``gateway``, ``api.port`` for ``serve`` —
because a fleet of ``serve`` instances left on the default ``gateway.port`` binds
that port exactly zero times, and refusing it would make the commonest layout
unusable. And the comparison is by port *number*, not by setting name, so a
``serve`` instance moved onto the gateway's default port is still caught. The
gap this leaves: a ``gateway`` instance whose WebUI later starts the on-demand
API server binds ``api.port`` too, and that is not checked, because the
supervisor does not start it.

Every refusal names the instances involved, so the caller can exit non-zero with
a message an operator can act on without opening the fleet file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from nanobot.fleet.config import FleetFile, FleetInstance, load_fleet_file
from nanobot.fleet.paths import (
    FleetPathError,
    InstancePaths,
    declared_workspace,
    instance_paths,
    read_instance_config,
)

#: Mirrors ``GatewayConfig.port`` and ``ApiConfig.port`` in
#: nanobot/config/schema.py. Duplicated rather than imported for the same reason
#: ``nanobot.fleet.paths`` duplicates the workspace default — importing the
#: schema drags in the whole agent tool tree — and pinned by a test that asserts
#: each stays equal to the model's own default.
DEFAULT_GATEWAY_PORT = 18790
DEFAULT_API_PORT = 8900

#: The config setting each launch mode binds.
_MODE_PORT_SETTING: Mapping[str, tuple[str, str, int]] = {
    "gateway": ("gateway", "gateway.port", DEFAULT_GATEWAY_PORT),
    "serve": ("api", "api.port", DEFAULT_API_PORT),
}

_MAX_RENDERED_ISSUES = 10

#: How each of an instance's two directories is named in a refusal.
_WORKSPACE = "workspace"
_CONFIG_DIR = "config directory"


@dataclass(frozen=True)
class FleetValidationIssue:
    """One reason a fleet will not be started, and whom it is about.

    ``instances`` is never empty and is ordered as the fleet document orders
    them. The names need no redaction: ``nanobot.fleet.config`` has already
    constrained every one of them to ``^[a-z0-9][a-z0-9_-]*$``, so an instance
    name cannot carry a credential-shaped value by the time it reaches here.
    """

    instances: tuple[str, ...]
    message: str

    @property
    def location(self) -> str:
        """The offending instances, rendered the way a config location is."""
        return ", ".join(f"instances.{name}" for name in self.instances)


@dataclass(frozen=True)
class ResolvedInstance:
    """One validated instance: its entry plus everything derived from it.

    The handoff object for the rest of the fleet package — the profile builder
    needs the canonical directories, the launcher needs ``config_path`` and the
    entry's ``env`` and memory cap. Only produced for a fleet that validated, so
    every path on it is absolute, canonical, and known to exist (the config file)
    or known not to overlap a peer (the workspace).
    """

    name: str
    entry: FleetInstance
    config_path: Path
    config_dir: Path
    workspace: Path
    port: int | None
    port_setting: str

    @property
    def mode(self) -> str:
        """The launch mode declared for this instance."""
        return self.entry.mode


class FleetValidationError(ValueError):
    """A fleet that parsed but describes instances that are not separable.

    Rendering follows ``ConfigLoadError`` and ``FleetConfigError``: a header, a
    summary, then one location/message pair per issue, stopping after ten so a
    badly wrong fleet file produces something readable.
    """

    def __init__(self, path: Path, issues: Iterable[FleetValidationIssue]) -> None:
        self.path = path
        self.issues = tuple(issues)
        self.summary = f"Found {len(self.issues)} problem(s) that prevent this fleet from starting."
        super().__init__(self.summary)

    @property
    def instances(self) -> tuple[str, ...]:
        """Every instance named by any issue, deduplicated, in document order."""
        seen: dict[str, None] = {}
        for issue in self.issues:
            for name in issue.instances:
                seen.setdefault(name, None)
        return tuple(seen)

    def __str__(self) -> str:
        lines = [f"Invalid fleet: {self.path}", "", self.summary]
        for issue in self.issues[:_MAX_RENDERED_ISSUES]:
            lines.extend(("", f"  {issue.location}", f"    {issue.message}"))
        remaining = len(self.issues) - _MAX_RENDERED_ISSUES
        if remaining > 0:
            lines.extend(("", f"  … and {remaining} more issue(s)"))
        return "\n".join(lines)


def validate_fleet_file(path: Path) -> tuple[ResolvedInstance, ...]:
    """Load a fleet file and validate it, the whole gate in one call.

    Raises:
        FleetConfigError: the document itself is unreadable or malformed.
        FleetValidationError: the document is well formed but its instances are
            not separable.
    """
    return validate_fleet(path, load_fleet_file(path))


def validate_fleet(path: Path, fleet: FleetFile) -> tuple[ResolvedInstance, ...]:
    """Resolve every instance in ``fleet`` and refuse any layout that is unsafe.

    Creates nothing — a fleet that is about to be refused must leave no
    directories behind — and starts nothing. ``path`` is used only to name the
    fleet file in the error.

    Collects every problem before raising rather than stopping at the first, so
    an operator fixing a fleet file sees the whole list instead of discovering
    the next one on the next run.

    Returns:
        The validated instances, in document order.

    Raises:
        FleetValidationError: one or more rules were broken.
    """
    issues: list[FleetValidationIssue] = []
    resolved: list[ResolvedInstance] = []
    for name, entry in fleet.instances.items():
        instance, instance_issues = _resolve_instance(name, entry)
        issues.extend(instance_issues)
        if instance is not None:
            resolved.append(instance)

    issues.extend(_overlap_issues(resolved))
    issues.extend(_duplicate_port_issues(resolved))
    if issues:
        raise FleetValidationError(path, issues)
    return tuple(resolved)


def _resolve_instance(
    name: str,
    entry: FleetInstance,
) -> tuple[ResolvedInstance | None, list[FleetValidationIssue]]:
    """Validate one instance in isolation and resolve its directories.

    Returns ``(None, issues)`` when the instance could not be resolved at all,
    which excludes it from the cross-instance checks — there is nothing to
    compare. A self-inverted instance *is* returned: its paths are perfectly
    good, they are just arranged wrongly, so it still takes part in the overlap
    check and the operator sees every problem in one pass.
    """
    declared_config = Path(entry.config).expanduser()
    if not declared_config.is_absolute():
        return None, [
            FleetValidationIssue(
                (name,),
                f"config path must be absolute after expansion, but {entry.config!r} is "
                f"relative; a relative path would resolve against whatever directory the "
                f"supervisor was started from.",
            )
        ]

    try:
        config = read_instance_config(declared_config)
        workspace_value = declared_workspace(config)
    except FleetPathError as exc:
        return None, [FleetValidationIssue((name,), f"{exc.reason}: {exc.path}")]

    if not Path(workspace_value).expanduser().is_absolute():
        return None, [
            FleetValidationIssue(
                (name,),
                f"agents.defaults.workspace must be absolute after expansion, but "
                f"{workspace_value!r} in {config.path} is relative.",
            )
        ]

    paths = instance_paths(config)
    port, port_setting = _bound_port(entry, config.data)
    instance = ResolvedInstance(
        name=name,
        entry=entry,
        config_path=config.path,
        config_dir=paths.config_dir,
        workspace=paths.workspace,
        port=port,
        port_setting=port_setting,
    )
    return instance, _self_inversion_issues(instance, paths)


def _self_inversion_issues(
    instance: ResolvedInstance,
    paths: InstancePaths,
) -> list[FleetValidationIssue]:
    """Refuse a config dir at or inside its own workspace.

    The reason is nanobot's own, from ``JsonlSessionStore``: the session store
    lives under the config dir, so this layout puts it inside the directory the
    agent may freely write. The equality case counts as inside — the sessions
    directory would then be ``<workspace>/sessions`` — which is exactly the
    comparison ``JsonlSessionStore`` makes before it raises.
    """
    if paths.config_dir != paths.workspace and not paths.config_dir.is_relative_to(paths.workspace):
        return []
    return [
        FleetValidationIssue(
            (instance.name,),
            f"session storage must be outside the agent workspace, but the config "
            f"directory {paths.config_dir} is inside its own workspace "
            f"{paths.workspace}; move the config file outside the workspace, or choose "
            f"a workspace nested under the config directory.",
        )
    ]


def _overlap_issues(instances: list[ResolvedInstance]) -> list[FleetValidationIssue]:
    """Refuse any pair of distinct instances whose directories touch.

    At most one issue per pair, because a pair that overlaps usually overlaps in
    several ways at once (two instances sharing a config dir also share the
    workspace nested in it) and four restatements of one mistake are harder to
    act on than one.
    """
    issues: list[FleetValidationIssue] = []
    for index, first in enumerate(instances):
        for second in instances[index + 1 :]:
            message = _overlap_message(first, second)
            if message is not None:
                issues.append(FleetValidationIssue((first.name, second.name), message))
    return issues


def _overlap_message(first: ResolvedInstance, second: ResolvedInstance) -> str | None:
    """Describe the first way two instances' directories touch, if they do."""
    for first_kind, first_path in _directories(first):
        for second_kind, second_path in _directories(second):
            if first_path == second_path:
                return (
                    f"{first.name}'s {first_kind} and {second.name}'s {second_kind} are "
                    f"the same directory: {first_path}."
                )
            if first_path.is_relative_to(second_path):
                return (
                    f"{first.name}'s {first_kind} {first_path} is inside {second.name}'s "
                    f"{second_kind} {second_path}."
                )
            if second_path.is_relative_to(first_path):
                return (
                    f"{second.name}'s {second_kind} {second_path} is inside {first.name}'s "
                    f"{first_kind} {first_path}."
                )
    return None


def _directories(instance: ResolvedInstance) -> tuple[tuple[str, Path], ...]:
    """The two directories an instance owns, each with its human-readable name."""
    return ((_WORKSPACE, instance.workspace), (_CONFIG_DIR, instance.config_dir))


def _duplicate_port_issues(instances: list[ResolvedInstance]) -> list[FleetValidationIssue]:
    """Report each port more than one instance would try to bind."""
    by_port: dict[int, list[ResolvedInstance]] = {}
    for instance in instances:
        if instance.port is not None:
            by_port.setdefault(instance.port, []).append(instance)

    issues: list[FleetValidationIssue] = []
    for port, sharing in by_port.items():
        if len(sharing) < 2:
            continue
        who = ", ".join(f"{instance.name} ({instance.port_setting})" for instance in sharing)
        issues.append(
            FleetValidationIssue(
                tuple(instance.name for instance in sharing),
                f"{len(sharing)} instances are configured to bind port {port}: {who}.",
            )
        )
    return issues


def _bound_port(entry: FleetInstance, data: Mapping[str, object]) -> tuple[int | None, str]:
    """Return the port this instance's mode will bind, and the setting naming it.

    ``None`` means "not comparable": the config does say something about this
    port and what it says is not a plain integer, so nanobot will refuse the file
    when the instance starts. Validating an instance's own config is not this
    module's job, and substituting the default for an unreadable value would
    invent a conflict that does not exist.

    Only a genuinely *absent* declaration falls back to the schema default, the
    same distinction ``nanobot.fleet.paths`` draws with its ``_MISSING``
    sentinel — an explicit ``null`` is a declaration, not a silence.
    """
    section_name, setting, default = _MODE_PORT_SETTING[entry.mode]
    if section_name not in data:
        return default, setting
    section = data[section_name]
    if not isinstance(section, dict):
        return None, setting
    ported = cast(Mapping[str, object], section)
    if "port" not in ported:
        return default, setting
    value = ported["port"]
    # bool is an int subclass, and ``"port": true`` is a config error, not 1.
    if isinstance(value, bool) or not isinstance(value, int):
        return None, setting
    return value, setting

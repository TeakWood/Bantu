"""Build one instance's Seatbelt profile: allow by default, deny the peers.

This is the module that turns a validated fleet into OS-enforced isolation. For
instance X it emits an SBPL profile that denies reading and writing every *other*
instance's workspace and config directory, plus the fleet file and the
supervisor's own state file. Everything else stays permitted, because an instance
still has to be an ordinary nanobot: talk to providers, spawn its shell tool,
read the system libraries it links against.

*Why a new module rather than a change to* ``nanobot/agent/tools/sandbox.py``.
The posture here is the exact inverse. That file builds ``(deny default)`` and
then allows a small list, which is right for confining a single shell command,
and it is wrong here in three specific ways. It denies the workspace's *parent*
(``sandbox.py``), which in the fleet layout is the instance's own config
directory — the one place the instance must be able to read, since its config,
sessions, cron and logs all live there. It pins ``HOME`` and ``TMPDIR`` to the
workspace, which would cut the instance off from its own data directory. And it
returns a shell command line, not a profile. Reusing it would mean inverting its
default and deleting three of its rules, so the fleet gets its own builder and
``sandbox.py`` is not touched.

*Why every path is checked before it is emitted.* This is the chokepoint, and the
reason it is one is that Seatbelt gives no feedback. Verified on the pinned host:
a deny naming a non-canonical path (``/tmp/x`` where the real path is
``/private/tmp/x``), a deny naming a path that does not exist, and a deny with an
invalid relative subpath are all *accepted* by ``sandbox-exec``, which then starts
the process cleanly, confines nothing, and exits 0. There is no error, no
warning, and no difference an operator could observe. A builder that emitted such
a rule would produce a fleet that reports every instance as confined while every
instance can read every peer's sessions. So a path that cannot be denied
meaningfully raises :class:`SeatbeltProfileError` instead of being written into a
rule.

Two consequences for callers, both load-bearing:

*The directories must exist by the time a profile is built.* An instance's
workspace is commonly created on first start, and the supervisor state file does
not exist before the first fleet is started. Both must be laid down *before* this
runs, because "does not exist yet" and "silently unconfined" are the same thing
to the kernel. ``nanobot.fleet.paths`` and ``nanobot.fleet.validate`` deliberately
create nothing, so that a fleet which fails validation leaves no directories
behind; creating them is the launcher's job, and it has to happen first.

*Renaming an ancestor is a bypass, and it is closed here.* A ``subpath`` rule
matches by path, not by inode, so an instance that renames a directory *above* a
peer's workspace makes the deny stop matching and can then read the peer through
the new name. Confirmed on the pinned host. The final rule therefore denies
``file-write-unlink`` on every ancestor of every denied path, which fixes those
directory entries in place without denying writes to their other children — the
same defence, and the same reasoning, as the ancestor rule in ``sandbox.py``. No
ancestor of a peer's directory can be inside this instance's own workspace (that
would be an overlap, which ``nanobot.fleet.validate`` refuses and
:func:`build_instance_profile` re-checks), so this cannot wall an instance off
from its own files.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from nanobot.fleet.config import INSTANCE_NAME_PATTERN
from nanobot.fleet.validate import ResolvedInstance

#: The operations every fleet deny covers. ``file-read*`` includes metadata, so a
#: denied peer directory cannot even be stat'd, and ``file-write*`` includes
#: unlink, so its directory entry cannot be removed either.
DENIED_OPERATIONS = "file-read* file-write*"

#: Appended to every refusal about a path. The point of the message is that the
#: OS would *not* complain, so the builder has to.
_SILENT_FAILURE = (
    "sandbox-exec would accept a rule naming it, match nothing, and confine "
    "nothing, with no error"
)

_INSTANCE_NAME_RE = re.compile(INSTANCE_NAME_PATTERN)


class SeatbeltProfileError(ValueError):
    """A profile cannot be built in a form that would actually confine.

    Always a refusal to emit, never a repair: there is no such thing as a
    best-effort Seatbelt rule, because a rule that does not match is
    indistinguishable from no rule at all.
    """

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        super().__init__(message)
        self.path = path


def build_fleet_profiles(
    instances: Sequence[ResolvedInstance],
    *,
    fleet_path: Path,
    state_path: Path,
) -> dict[str, str]:
    """Build every instance's profile, keyed by instance name.

    The intended entry point, because it is the one that cannot get the peer set
    wrong: each instance's peers are exactly the rest of the fleet. Building
    profiles one at a time is supported (:func:`build_instance_profile`) but puts
    the "peers are everyone else" invariant on the caller.

    Raises:
        SeatbeltProfileError: any path in the fleet cannot be denied meaningfully,
            two instances are not separable, or a name is repeated. Nothing is
            returned partially — a fleet in which one instance cannot be confined
            is not startable.
    """
    names = [instance.name for instance in instances]
    if repeated := sorted({name for name in names if names.count(name) > 1}):
        # Unreachable from a parsed fleet document, whose instances are dict keys.
        # Checked anyway because the failure is silent and total: peers are
        # matched by name, so both copies of a repeated name would filter each
        # other out and neither would be denied the other's directories.
        raise SeatbeltProfileError(
            f"the fleet declares {', '.join(repeated)} more than once; each "
            f"instance's peers are selected by name, so a repeated name would "
            f"leave both copies unconfined from each other."
        )
    return {
        instance.name: build_instance_profile(
            instance,
            [peer for peer in instances if peer.name != instance.name],
            fleet_path=fleet_path,
            state_path=state_path,
        )
        for instance in instances
    }


def build_instance_profile(
    instance: ResolvedInstance,
    peers: Iterable[ResolvedInstance],
    *,
    fleet_path: Path,
    state_path: Path,
) -> str:
    """Return the SBPL profile confining ``instance`` away from ``peers``.

    Args:
        instance: the instance the profile is for. Its own directories are never
            denied; they are used only to re-check that no peer overlaps them.
        peers: every *other* instance in the fleet. Must not contain ``instance``
            itself — denying an instance its own workspace is refused rather than
            quietly dropped, because a caller that passed the whole fleet here
            would otherwise get a profile that confines the instance out of its
            own files.
        fleet_path: the fleet file, denied to every instance. It names every
            peer's config path, which is a map of what to go and read.
        state_path: the supervisor's state file, denied to every instance. It
            carries the pids the supervisor manages, so an instance that could
            write it could redirect the supervisor at processes of its choosing.

    Returns:
        A profile string suitable for ``sandbox-exec -p``. Deterministic: peers
        are emitted in the order given, so the same fleet always produces the
        same profile and :mod:`nanobot.fleet.probe` can test a stable artefact.

    Raises:
        SeatbeltProfileError: a path is relative, non-canonical, or absent; a peer
            overlaps ``instance``; ``peers`` contains ``instance``; or a peer name
            is not a legal instance name.
    """
    fleet_file = _denied_path("the fleet file", fleet_path, directory=False)
    state_file = _denied_path("the supervisor state file", state_path, directory=False)

    # The instance's own directories are never denied, but they are compared
    # against every peer's, and a non-canonical path compares wrong: /tmp/a/ws
    # would not look nested inside /private/tmp/a. Checking them makes
    # _check_separable sound, and costs nothing — for a fleet of two or more each
    # of these paths is validated anyway on the pass where it is somebody's peer.
    own = tuple(
        _denied_path(f"{instance.name}'s own {role}", path, directory=True)
        for role, path in (
            ("workspace", instance.workspace),
            ("config directory", instance.config_dir),
        )
    )

    rules = ["(version 1)", "(allow default)"]
    denied: list[Path] = [fleet_file, state_file]

    for peer in peers:
        name = _checked_name(peer.name)
        if name == instance.name:
            raise SeatbeltProfileError(
                f"{name} was passed as a peer of itself; peers must be every "
                f"other instance in the fleet, never this one."
            )
        _check_separable(instance.name, own, peer)
        targets = list(
            dict.fromkeys(
                (
                    _denied_path(f"peer {name}'s workspace", peer.workspace, directory=True),
                    _denied_path(f"peer {name}'s config directory", peer.config_dir, directory=True),
                )
            )
        )
        denied.extend(targets)
        rules.append(f"; peer {name}")
        rules.append(_deny_rule("subpath", targets))

    rules.append("; the fleet file names every peer's config; the state file drives the supervisor")
    rules.append(_deny_rule("literal", (fleet_file, state_file)))

    # A subpath rule matches by path, so pin the directory entries above every
    # denied path: renaming one would otherwise slide the peer out from under its
    # own deny. Only these entries are fixed; their other children stay writable.
    rules.append("; a renamed ancestor would move a peer out of its own deny")
    rules.append(f"(deny file-write-unlink {_literals(_ancestors(denied))})")

    return "\n".join(rules)


def _deny_rule(kind: str, paths: Iterable[Path]) -> str:
    """Render one deny covering ``paths`` as ``subpath`` or ``literal`` filters."""
    filters = " ".join(f"({kind} {sbpl_quote(str(path))})" for path in paths)
    return f"(deny {DENIED_OPERATIONS} {filters})"


def _literals(paths: Iterable[str]) -> str:
    """Render ``paths`` as a run of SBPL ``literal`` filters."""
    return " ".join(f"(literal {sbpl_quote(path)})" for path in paths)


def sbpl_quote(path: str) -> str:
    """Render ``path`` as an SBPL string literal.

    SBPL uses C-style string escaping, so an unescaped backslash or double quote
    in a path would terminate the literal early and change the meaning of every
    rule after it — in an allow-default profile, dropping a deny. Mirrors
    ``_sbpl_quote`` in ``nanobot/agent/tools/sandbox.py`` rather than importing a
    private name across packages; a test pins that the two agree, including on
    the quote-bearing workspace name the native Seatbelt tests deliberately use.
    """
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _denied_path(what: str, path: Path, *, directory: bool) -> Path:
    """Return ``path`` if it can be denied meaningfully, else raise.

    Absoluteness is checked before canonicality so a relative path is reported as
    what it is, rather than as a mismatch against wherever the supervisor was
    started. Existence is checked last because a non-canonical path usually
    *does* exist — ``/tmp/x`` resolves fine — and naming the real path is the more
    useful message.

    ``directory`` distinguishes the two filter kinds: a ``subpath`` rule is only
    meaningful over a directory, and a ``literal`` rule over a directory would
    cover the entry alone and leave everything inside it reachable.
    """
    if not path.is_absolute():
        raise SeatbeltProfileError(
            f"{what} {path} is not absolute; {_SILENT_FAILURE}.", path=path
        )
    resolved = path.resolve(strict=False)
    if resolved != path:
        raise SeatbeltProfileError(
            f"{what} {path} is not canonical (the real path is {resolved}); "
            f"{_SILENT_FAILURE}.",
            path=path,
        )
    if directory and not path.is_dir():
        raise SeatbeltProfileError(
            f"{what} {path} is not an existing directory; {_SILENT_FAILURE}.", path=path
        )
    if not directory and not path.is_file():
        raise SeatbeltProfileError(
            f"{what} {path} is not an existing file; {_SILENT_FAILURE}.", path=path
        )
    return path


def _checked_name(name: str) -> str:
    """Return ``name`` if it is a legal instance name, else raise.

    ``nanobot.fleet.config`` already constrains every name in a fleet document to
    this pattern. It is re-checked at the point it is relied upon, for the reason
    ``nanobot.fleet.env`` re-checks variable names: here a name is rendered into a
    profile comment, and a name carrying a newline could close the comment and
    append an ``(allow …)`` rule that re-opens a peer's directory, since the last
    matching rule wins.
    """
    if not _INSTANCE_NAME_RE.fullmatch(name):
        raise SeatbeltProfileError(
            f"{name!r} is not a valid instance name (must match "
            f"{INSTANCE_NAME_PATTERN}); it cannot be written into a profile."
        )
    return name


def _check_separable(
    name: str,
    own_directories: Iterable[Path],
    peer: ResolvedInstance,
) -> None:
    """Refuse a peer whose directories touch this instance's own.

    ``nanobot.fleet.validate`` already refuses such a fleet. Re-checked here
    because this is where the consequence lands: the deny that walls the peer off
    would also wall the instance out of its own config or workspace, and an
    instance that cannot read its own config file fails in a way that looks
    nothing like a fleet layout mistake.
    """
    for own in own_directories:
        for theirs in (peer.workspace, peer.config_dir):
            if own == theirs or own.is_relative_to(theirs) or theirs.is_relative_to(own):
                raise SeatbeltProfileError(
                    f"peer {peer.name}'s directory {theirs} overlaps {name}'s own "
                    f"directory {own}; denying it would confine {name} out of its own "
                    f"files.",
                    path=theirs,
                )


def _ancestors(paths: Iterable[Path]) -> list[str]:
    """Every ancestor directory of ``paths``, closest first, deduplicated."""
    out: dict[str, None] = {}
    for path in paths:
        for ancestor in path.parents:
            out.setdefault(str(ancestor), None)
    return list(out)

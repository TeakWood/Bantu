"""Tests for the per-instance Seatbelt profile builder.

These parse the generated profile string rather than executing it, so they run on
any platform — mirroring ``tests/tools/test_sandbox.py::TestSeatbeltBackend``,
which does the same for the shell sandbox. One darwin-gated test at the end drives
a generated profile through the real ``sandbox-exec`` in both directions.

Most of what is asserted here is a *refusal*, because every way of getting this
wrong is silent. Verified on the pinned host: ``sandbox-exec`` accepts a deny
naming a non-canonical path, a deny naming a path that does not exist, and a deny
with an invalid relative subpath, then starts the process unconfined and exits 0.
So "the builder raised" is the only observable difference between a profile that
confines and one that merely looks like it does.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nanobot.fleet.config import FleetInstance
from nanobot.fleet.profile import (
    DENIED_OPERATIONS,
    SeatbeltProfileError,
    build_fleet_profiles,
    build_instance_profile,
    sbpl_quote,
)
from nanobot.fleet.validate import ResolvedInstance


def make_instance(
    root: Path,
    name: str,
    *,
    workspace: Path | None = None,
    config_dir: Path | None = None,
) -> ResolvedInstance:
    """Resolve one instance laid out the way nanobot lays one out by itself.

    ``<root>/<name>/config.json`` with ``<root>/<name>/workspace`` beneath it, the
    default ``~/.nanobot-x/workspace`` under ``~/.nanobot-x/`` shape. The two
    directories can be overridden to build the layouts that must be refused.
    """
    home = root / name
    resolved_config_dir = home if config_dir is None else config_dir
    resolved_workspace = home / "workspace" if workspace is None else workspace
    resolved_config_dir.mkdir(parents=True, exist_ok=True)
    resolved_workspace.mkdir(parents=True, exist_ok=True)
    config_path = resolved_config_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    return ResolvedInstance(
        name=name,
        entry=FleetInstance(config=str(config_path), mode="serve", memory_limit_mb=512),
        config_path=config_path,
        config_dir=resolved_config_dir,
        workspace=resolved_workspace,
        port=None,
        port_setting="api.port",
    )


@pytest.fixture
def supervisor_files(tmp_path: Path) -> tuple[Path, Path]:
    """The fleet file and supervisor state file, both existing, both canonical."""
    root = tmp_path.resolve()
    fleet = root / "fleet.json"
    state = root / "state.json"
    fleet.write_text("{}", encoding="utf-8")
    state.write_text("{}", encoding="utf-8")
    return fleet, state


def build(
    instance: ResolvedInstance,
    peers: list[ResolvedInstance],
    supervisor_files: tuple[Path, Path],
) -> str:
    """Build one profile from the standard supervisor file pair."""
    fleet, state = supervisor_files
    return build_instance_profile(instance, peers, fleet_path=fleet, state_path=state)


def subpath_deny(*paths: Path) -> str:
    """The deny rule the builder emits for a peer's directories."""
    filters = " ".join(f"(subpath {sbpl_quote(str(path))})" for path in paths)
    return f"(deny {DENIED_OPERATIONS} {filters})"


def unlink_rule(profile: str) -> str:
    """The trailing rule that pins denied paths' ancestor directory entries."""
    return next(
        line for line in profile.splitlines() if line.startswith("(deny file-write-unlink ")
    )


# --------------------------------------------------------------------------
# Posture
# --------------------------------------------------------------------------


def test_posture_is_version_one_then_allow_default(tmp_path, supervisor_files) -> None:
    """The inverse of the shell sandbox, and the reason this is its own module."""
    root = tmp_path.resolve()
    profile = build(make_instance(root, "a"), [make_instance(root, "b")], supervisor_files)

    assert profile.splitlines()[:2] == ["(version 1)", "(allow default)"]


def test_no_allow_rule_ever_follows_a_deny(tmp_path, supervisor_files) -> None:
    """Last matching rule wins, so a later allow would silently re-open a peer.

    ``(allow default)`` is the only allow in the profile; everything after it
    narrows. A future rule added below a deny would break confinement without
    breaking any other assertion in this file, so it is pinned directly.
    """
    root = tmp_path.resolve()
    profile = build(make_instance(root, "a"), [make_instance(root, "b")], supervisor_files)

    allows = [line for line in profile.splitlines() if line.startswith("(allow")]
    assert allows == ["(allow default)"]


def test_every_peer_workspace_and_config_dir_is_denied(tmp_path, supervisor_files) -> None:
    root = tmp_path.resolve()
    a = make_instance(root, "a")
    peers = [make_instance(root, "b"), make_instance(root, "c")]

    profile = build(a, peers, supervisor_files)

    for peer in peers:
        assert subpath_deny(peer.workspace, peer.config_dir) in profile


def test_the_deny_covers_both_reading_and_writing(tmp_path, supervisor_files) -> None:
    root = tmp_path.resolve()
    b = make_instance(root, "b")

    profile = build(make_instance(root, "a"), [b], supervisor_files)

    assert DENIED_OPERATIONS == "file-read* file-write*"
    assert f"(deny file-read* file-write* (subpath {sbpl_quote(str(b.workspace))})" in profile


def test_the_instances_own_directories_are_never_denied(tmp_path, supervisor_files) -> None:
    """An instance must reach its own config dir; ``sandbox.py`` denies that parent."""
    root = tmp_path.resolve()
    a = make_instance(root, "a")

    profile = build(a, [make_instance(root, "b")], supervisor_files)

    for own in (a.workspace, a.config_dir):
        assert f"(subpath {sbpl_quote(str(own))})" not in profile
        assert f"(literal {sbpl_quote(str(own))})" not in profile


def test_the_fleet_file_and_state_file_are_denied_as_literals(tmp_path, supervisor_files) -> None:
    root = tmp_path.resolve()
    fleet, state = supervisor_files

    profile = build(make_instance(root, "a"), [make_instance(root, "b")], supervisor_files)

    assert (
        f"(deny {DENIED_OPERATIONS} (literal {sbpl_quote(str(fleet))}) "
        f"(literal {sbpl_quote(str(state))}))"
    ) in profile


def test_a_single_instance_fleet_still_denies_the_supervisor_files(
    tmp_path, supervisor_files
) -> None:
    """No peers is not the same as nothing to hide: the fleet file still is."""
    root = tmp_path.resolve()
    fleet, _ = supervisor_files

    profile = build(make_instance(root, "a"), [], supervisor_files)

    assert f"(literal {sbpl_quote(str(fleet))})" in profile


def test_the_profile_is_deterministic(tmp_path, supervisor_files) -> None:
    """The same fleet must always produce the same artefact, for F5 to probe."""
    root = tmp_path.resolve()
    a = make_instance(root, "a")
    peers = [make_instance(root, "b"), make_instance(root, "c")]

    assert build(a, peers, supervisor_files) == build(a, peers, supervisor_files)


# --------------------------------------------------------------------------
# The ancestor-rename bypass
# --------------------------------------------------------------------------


def test_ancestors_of_every_denied_path_are_unlink_denied(tmp_path, supervisor_files) -> None:
    """Renaming a directory above a peer slides it out from under its own deny.

    Confirmed on the pinned host: with the ancestor rule removed, ``mv`` of the
    peer's grandparent followed by a read of the peer's file succeeds.
    """
    root = tmp_path.resolve()
    b = make_instance(root, "b")
    fleet, state = supervisor_files

    rule = unlink_rule(build(make_instance(root, "a"), [b], supervisor_files))

    for denied in (b.workspace, b.config_dir, fleet, state):
        for ancestor in denied.parents:
            assert f"(literal {sbpl_quote(str(ancestor))})" in rule


def test_the_ancestor_rule_denies_only_unlinking(tmp_path, supervisor_files) -> None:
    """Ancestors stay readable and writable; only their entries are pinned.

    A broader deny here would cut the instance off from directories it shares
    with its peers — commonly the fleet root, and ultimately ``/``.
    """
    root = tmp_path.resolve()
    b = make_instance(root, "b")

    profile = build(make_instance(root, "a"), [b], supervisor_files)

    assert f"(deny {DENIED_OPERATIONS} (subpath {sbpl_quote(str(root))}))" not in profile
    assert unlink_rule(profile).startswith("(deny file-write-unlink ")


# --------------------------------------------------------------------------
# Refusals: a rule that cannot bind is never emitted
# --------------------------------------------------------------------------


def test_a_relative_peer_workspace_is_refused(tmp_path, supervisor_files) -> None:
    root = tmp_path.resolve()
    b = make_instance(root, "b")
    relative = replace(b, workspace=Path("relative/workspace"))

    with pytest.raises(SeatbeltProfileError, match="is not absolute"):
        build(make_instance(root, "a"), [relative], supervisor_files)


def test_a_non_canonical_peer_workspace_is_refused(tmp_path, supervisor_files) -> None:
    """The ``/tmp`` versus ``/private/tmp`` mistake, which confines nothing."""
    root = tmp_path.resolve()
    real = root / "real"
    real.mkdir()
    link = root / "link"
    link.symlink_to(real, target_is_directory=True)
    b = make_instance(root, "b")
    through_link = replace(b, workspace=link)

    with pytest.raises(SeatbeltProfileError, match="is not canonical") as caught:
        build(make_instance(root, "a"), [through_link], supervisor_files)

    assert str(real) in str(caught.value)


def test_a_path_containing_dotdot_is_refused(tmp_path, supervisor_files) -> None:
    """An invalid relative subpath is accepted by sandbox-exec and matches nothing."""
    root = tmp_path.resolve()
    b = make_instance(root, "b")
    dotted = replace(b, workspace=root / "b" / ".." / "b" / "workspace")

    with pytest.raises(SeatbeltProfileError, match="is not canonical"):
        build(make_instance(root, "a"), [dotted], supervisor_files)


def test_a_non_existent_peer_workspace_is_refused(tmp_path, supervisor_files) -> None:
    """The workspace is created on first start, so the launcher must create it first."""
    root = tmp_path.resolve()
    b = make_instance(root, "b")
    shutil.rmtree(b.workspace)

    with pytest.raises(SeatbeltProfileError, match="is not an existing directory"):
        build(make_instance(root, "a"), [b], supervisor_files)


def test_a_non_existent_peer_config_dir_is_refused(tmp_path, supervisor_files) -> None:
    root = tmp_path.resolve()
    b = make_instance(root, "b")
    absent = replace(b, config_dir=root / "gone")

    with pytest.raises(SeatbeltProfileError, match="is not an existing directory"):
        build(make_instance(root, "a"), [absent], supervisor_files)


@pytest.mark.parametrize("missing", ["fleet", "state"])
def test_a_non_existent_supervisor_file_is_refused(tmp_path, supervisor_files, missing) -> None:
    """The state file does not exist before the first start, and must by then."""
    root = tmp_path.resolve()
    fleet, state = supervisor_files
    (fleet if missing == "fleet" else state).unlink()

    with pytest.raises(SeatbeltProfileError, match="is not an existing file"):
        build(make_instance(root, "a"), [make_instance(root, "b")], supervisor_files)


def test_a_directory_passed_as_the_state_file_is_refused(tmp_path) -> None:
    """A ``literal`` over a directory covers the entry and leaves the contents open."""
    root = tmp_path.resolve()
    fleet = root / "fleet.json"
    fleet.write_text("{}", encoding="utf-8")
    state_dir = root / "state"
    state_dir.mkdir()

    with pytest.raises(SeatbeltProfileError, match="is not an existing file"):
        build_instance_profile(
            make_instance(root, "a"), [], fleet_path=fleet, state_path=state_dir
        )


def test_a_relative_fleet_path_is_refused(tmp_path, supervisor_files) -> None:
    root = tmp_path.resolve()
    _, state = supervisor_files

    with pytest.raises(SeatbeltProfileError, match="the fleet file.*is not absolute"):
        build_instance_profile(
            make_instance(root, "a"), [], fleet_path=Path("fleet.json"), state_path=state
        )


def test_an_overlapping_peer_is_refused(tmp_path, supervisor_files) -> None:
    """Denying it would confine the instance out of its own files.

    ``nanobot.fleet.validate`` already refuses this fleet; the check is repeated
    where the consequence lands.
    """
    root = tmp_path.resolve()
    a = make_instance(root, "a")
    nested = make_instance(root, "b", workspace=a.workspace / "nested")

    with pytest.raises(SeatbeltProfileError, match="overlaps a's own directory"):
        build(a, [nested], supervisor_files)


def test_a_peer_sharing_the_instances_config_dir_is_refused(tmp_path, supervisor_files) -> None:
    root = tmp_path.resolve()
    a = make_instance(root, "a")
    twin = make_instance(root, "b", config_dir=a.config_dir)

    with pytest.raises(SeatbeltProfileError, match="overlaps"):
        build(a, [twin], supervisor_files)


def test_a_non_canonical_workspace_on_the_instance_itself_is_refused(
    tmp_path, supervisor_files
) -> None:
    """Its own paths are not denied, but they are compared against every peer's.

    ``/tmp/a/ws`` would not look nested inside ``/private/tmp/a``, so an overlap
    that should be refused would be missed and the instance would be confined out
    of its own files instead.
    """
    root = tmp_path.resolve()
    real = root / "real"
    real.mkdir()
    link = root / "link"
    link.symlink_to(real, target_is_directory=True)
    a = replace(make_instance(root, "a"), workspace=link)

    with pytest.raises(SeatbeltProfileError, match="a's own workspace.*is not canonical"):
        build(a, [make_instance(root, "b")], supervisor_files)


def test_an_instance_may_not_be_its_own_peer(tmp_path, supervisor_files) -> None:
    """A caller that passed the whole fleet must be told, not quietly obeyed."""
    root = tmp_path.resolve()
    a = make_instance(root, "a")

    with pytest.raises(SeatbeltProfileError, match="was passed as a peer of itself"):
        build(a, [a], supervisor_files)


def test_a_peer_name_that_could_inject_a_rule_is_refused(tmp_path, supervisor_files) -> None:
    """A newline in a name would close the comment and append an allow.

    The last matching rule wins in an allow-default profile, so an injected
    ``(allow …)`` after the peer deny would re-open exactly what was denied.
    """
    root = tmp_path.resolve()
    b = make_instance(root, "b")
    injected = replace(b, name='b\n(allow file-read* (subpath "/"))')

    with pytest.raises(SeatbeltProfileError, match="is not a valid instance name"):
        build(make_instance(root, "a"), [injected], supervisor_files)


# --------------------------------------------------------------------------
# Quoting
# --------------------------------------------------------------------------


def test_quoting_escapes_backslash_and_double_quote() -> None:
    assert sbpl_quote(r'/a/pro"ject\back') == r'"/a/pro\"ject\\back"'


def test_quoting_matches_the_shell_sandbox() -> None:
    """Guard the one helper this module duplicates instead of importing."""
    from nanobot.agent.tools.sandbox import _sbpl_quote

    for path in ('/a/project with "quotes', r"/a/back\slash", "/plain", r'/both"\ '):
        assert sbpl_quote(path) == _sbpl_quote(path)


def test_a_quote_bearing_peer_workspace_is_escaped_in_the_profile(
    tmp_path, supervisor_files
) -> None:
    """The native Seatbelt tests deliberately use a workspace name with a quote.

    Unescaped, the literal would end early and every rule after it would be
    reinterpreted — in an allow-default profile, silently dropping a deny.
    """
    root = tmp_path.resolve()
    b = make_instance(root, "b", workspace=root / 'project with "quotes')

    profile = build(make_instance(root, "a"), [b], supervisor_files)

    assert f"(subpath {sbpl_quote(str(b.workspace))})" in profile
    # The escaped form is present and the raw one appears nowhere: an unescaped
    # quote would close the literal early and swallow the rules after it.
    assert 'with \\"quotes' in profile
    assert 'with "quotes' not in profile


# --------------------------------------------------------------------------
# The whole fleet
# --------------------------------------------------------------------------


def test_each_profile_denies_every_other_instance_and_no_more(tmp_path, supervisor_files) -> None:
    root = tmp_path.resolve()
    fleet, state = supervisor_files
    instances = [make_instance(root, name) for name in ("a", "b", "c")]

    profiles = build_fleet_profiles(instances, fleet_path=fleet, state_path=state)

    assert set(profiles) == {"a", "b", "c"}
    for subject in instances:
        profile = profiles[subject.name]
        for peer in instances:
            rule = subpath_deny(peer.workspace, peer.config_dir)
            assert (rule in profile) is (peer.name != subject.name)


def test_a_repeated_instance_name_refuses_the_fleet(tmp_path, supervisor_files) -> None:
    """Peers are selected by name, so a repeat would filter both copies out.

    Unreachable from a parsed fleet document, where names are dict keys, but the
    resulting hole is total and silent: neither copy would deny the other.
    """
    root = tmp_path.resolve()
    fleet, state = supervisor_files
    a = make_instance(root, "a")
    twin = replace(make_instance(root, "b"), name="a")

    with pytest.raises(SeatbeltProfileError, match="declares a more than once"):
        build_fleet_profiles([a, twin], fleet_path=fleet, state_path=state)


def test_one_unconfinable_instance_refuses_the_whole_fleet(tmp_path, supervisor_files) -> None:
    """A fleet in which one instance cannot be confined is not startable at all."""
    root = tmp_path.resolve()
    fleet, state = supervisor_files
    broken = make_instance(root, "b")
    shutil.rmtree(broken.workspace)

    with pytest.raises(SeatbeltProfileError):
        build_fleet_profiles(
            [make_instance(root, "a"), broken], fleet_path=fleet, state_path=state
        )


# --------------------------------------------------------------------------
# The kernel, not just the text
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="requires native macOS Seatbelt",
)
def test_a_generated_profile_binds_natively(tmp_path, supervisor_files) -> None:
    """Both directions through the real kernel policy.

    The portable tests above prove the profile *says* the right thing. This proves
    it *does* it: a peer's file is unreadable and unwritable under the generated
    profile, while the instance keeps full read and write access to its own
    workspace — which is the whole reason for the allow-default posture, and what
    the shell sandbox's profile would take away.
    """
    root = tmp_path.resolve()
    a = make_instance(root, "a")
    b = make_instance(root, "b")
    (b.workspace / "sentinel").write_text("peer-secret", encoding="utf-8")
    (a.workspace / "sentinel").write_text("own-data", encoding="utf-8")
    profile = build(a, [b], supervisor_files)

    def run(script: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", profile, "/bin/sh", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )

    peer_file = b.workspace / "sentinel"
    peer_read = run(f"cat '{peer_file}'")
    assert peer_read.returncode != 0, peer_read.stdout
    assert "peer-secret" not in peer_read.stdout

    assert run(f"printf tampered > '{peer_file}'").returncode != 0
    assert peer_file.read_text(encoding="utf-8") == "peer-secret"

    # The instance's own workspace must stay fully usable: the ancestor
    # file-write-unlink rule pins directory entries above the peer, and the fleet
    # root is one of them, so a too-broad rule would surface here.
    own = run(
        f"cat '{a.workspace / 'sentinel'}' "
        f"&& printf fresh > '{a.workspace / 'new'}' "
        f"&& rm '{a.workspace / 'new'}'"
    )
    assert own.returncode == 0, own.stderr
    assert own.stdout == "own-data"

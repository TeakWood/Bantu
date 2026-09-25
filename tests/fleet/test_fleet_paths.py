"""Tests for side-effect-free instance path resolution.

Three of these are the reason the module exists rather than a call to
``load_config`` plus ``get_workspace_path``: resolution must create nothing,
must canonicalise through symlinks, and must refuse a missing file instead of
silently handing back the shared default workspace.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nanobot.fleet.paths import (
    DEFAULT_WORKSPACE,
    FleetPathError,
    InstanceConfig,
    InstancePaths,
    declared_workspace,
    instance_paths,
    read_instance_config,
    resolve_instance_paths,
)


def write_config(directory: Path, data: object) -> Path:
    """Write one instance config file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def config_with_workspace(directory: Path, workspace: object) -> Path:
    """Write a config declaring ``agents.defaults.workspace``."""
    return write_config(directory, {"agents": {"defaults": {"workspace": workspace}}})


def tree_snapshot(root: Path) -> dict[str, object]:
    """Capture every path under ``root`` with its bytes and mtime.

    Fine-grained enough that creating a directory, touching a file, or rewriting
    one byte all show up as a difference.
    """
    snapshot: dict[str, object] = {}
    for item in sorted(root.rglob("*")):
        key = str(item.relative_to(root))
        if item.is_symlink():
            snapshot[key] = ("symlink", os.readlink(item))
        elif item.is_dir():
            snapshot[key] = ("dir", item.stat().st_mtime_ns)
        else:
            snapshot[key] = ("file", item.read_bytes(), item.stat().st_mtime_ns)
    return snapshot


def test_returns_canonical_absolute_config_dir_and_workspace(tmp_path: Path) -> None:
    instance = tmp_path / "instance-a"
    config = config_with_workspace(instance, str(instance / "workspace"))

    paths = resolve_instance_paths(config)

    assert paths == InstancePaths(
        config_dir=instance.resolve(),
        workspace=(instance / "workspace").resolve(),
    )
    assert paths.config_dir.is_absolute()
    assert paths.workspace.is_absolute()


def test_result_unpacks_as_a_config_dir_workspace_pair(tmp_path: Path) -> None:
    config = config_with_workspace(tmp_path / "instance-a", str(tmp_path / "ws"))

    config_dir, workspace = resolve_instance_paths(config)

    assert config_dir == (tmp_path / "instance-a").resolve()
    assert workspace == (tmp_path / "ws").resolve()


def test_absent_workspace_key_falls_back_to_the_schema_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config = write_config(tmp_path / "instance-a", {"agents": {"defaults": {}}})

    paths = resolve_instance_paths(config)

    assert paths.workspace == (home / ".nanobot" / "workspace").resolve()


def test_empty_config_object_falls_back_to_the_schema_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config = write_config(tmp_path / "instance-a", {})

    paths = resolve_instance_paths(config)

    assert paths.workspace == (home / ".nanobot" / "workspace").resolve()


def test_default_workspace_constant_matches_the_config_schema() -> None:
    """Guard the one value this module duplicates instead of importing."""
    from nanobot.config.schema import AgentDefaults

    assert DEFAULT_WORKSPACE == AgentDefaults().workspace


def test_tilde_in_workspace_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config = config_with_workspace(tmp_path / "instance-a", "~/ws-x")

    assert resolve_instance_paths(config).workspace == (home / "ws-x").resolve()


def test_tilde_in_config_path_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config_with_workspace(home / "instance-a", str(home / "instance-a" / "workspace"))

    paths = resolve_instance_paths("~/instance-a/config.json")

    assert paths.config_dir == (home / "instance-a").resolve()


def test_relative_config_path_resolves_against_the_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_with_workspace(tmp_path / "instance-a", str(tmp_path / "ws"))
    monkeypatch.chdir(tmp_path)

    paths = resolve_instance_paths("instance-a/config.json")

    assert paths.config_dir == (tmp_path / "instance-a").resolve()


def test_a_missing_config_file_raises_instead_of_yielding_defaults(tmp_path: Path) -> None:
    """``load_config`` returns ``Config()`` here; that would collide every typo.

    Every mistyped instance path would otherwise resolve to the same
    ``~/.nanobot/workspace``, so a fleet of confined instances would silently
    share one workspace — and share it with the unsupervised install.
    """
    missing = tmp_path / "instance-a" / "config.json"

    with pytest.raises(FleetPathError) as excinfo:
        resolve_instance_paths(missing)

    assert excinfo.value.path == missing
    assert "does not exist" in str(excinfo.value)
    assert str(missing) in str(excinfo.value)


def test_a_dangling_symlink_config_path_raises(tmp_path: Path) -> None:
    link = tmp_path / "config.json"
    link.symlink_to(tmp_path / "gone.json")

    with pytest.raises(FleetPathError, match="does not exist"):
        resolve_instance_paths(link)


def test_a_directory_in_place_of_the_config_file_raises(tmp_path: Path) -> None:
    directory = tmp_path / "config.json"
    directory.mkdir()

    with pytest.raises(FleetPathError, match="is not a file"):
        resolve_instance_paths(directory)


def test_a_config_path_traversing_a_symlink_resolves_to_its_real_path(tmp_path: Path) -> None:
    real = tmp_path / "real-instance"
    config_with_workspace(real, str(real / "workspace"))
    link = tmp_path / "linked-instance"
    link.symlink_to(real, target_is_directory=True)

    paths = resolve_instance_paths(link / "config.json")

    assert paths.config_dir == real.resolve()
    assert paths.workspace == (real / "workspace").resolve()
    assert "linked-instance" not in str(paths.config_dir)


def test_a_workspace_traversing_a_symlink_resolves_to_its_real_path(tmp_path: Path) -> None:
    real = tmp_path / "real-workspaces"
    real.mkdir()
    link = tmp_path / "linked-workspaces"
    link.symlink_to(real, target_is_directory=True)
    config = config_with_workspace(tmp_path / "instance-a", str(link / "ws"))

    paths = resolve_instance_paths(config)

    assert paths.workspace == (real / "ws").resolve()
    assert str(link) not in str(paths.workspace)


def test_a_symlinked_config_file_keeps_the_data_dir_where_nanobot_puts_it(
    tmp_path: Path,
) -> None:
    """The config *file* is not resolved: nanobot derives its data dir from the
    parent as written, so canonicalising through the file would relocate it."""
    shared = tmp_path / "shared"
    shared.mkdir()
    target = shared / "instance-a.json"
    target.write_text(json.dumps({"agents": {"defaults": {"workspace": "/tmp/ws-a"}}}), "utf-8")
    instance = tmp_path / "instance-a"
    instance.mkdir()
    (instance / "config.json").symlink_to(target)

    paths = resolve_instance_paths(instance / "config.json")

    assert paths.config_dir == instance.resolve()
    assert paths.config_dir != shared.resolve()


def test_resolution_creates_nothing(tmp_path: Path) -> None:
    instance = tmp_path / "instance-a"
    workspace = instance / "nested" / "workspace"
    config = config_with_workspace(instance, str(workspace))
    before = tree_snapshot(tmp_path)

    paths = resolve_instance_paths(config)

    assert tree_snapshot(tmp_path) == before
    assert paths.workspace == workspace.resolve()
    assert not workspace.exists()
    assert not workspace.parent.exists()


def test_resolution_creates_nothing_for_the_default_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default path runs through the same helpers, so cover it too."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    write_config(tmp_path / "instance-a", {})
    before = tree_snapshot(tmp_path)

    resolve_instance_paths(tmp_path / "instance-a" / "config.json")

    assert tree_snapshot(tmp_path) == before
    assert not (home / ".nanobot").exists()


_IMPORT_PROBE = """
import json, sys
from nanobot.fleet.paths import resolve_instance_paths
resolve_instance_paths(sys.argv[1])
print(json.dumps(sorted(
    name for name in sys.modules
    if name.startswith(("nanobot.agent", "nanobot.config"))
)))
"""


def test_resolution_does_not_import_the_agent_tool_or_config_modules(tmp_path: Path) -> None:
    """Checked in a subprocess: the test session has already imported everything.

    ``nanobot.config`` is in the probe alongside ``nanobot.agent`` because it is
    the stronger claim — it proves neither ``load_config`` nor the ``ensure_dir``
    path helpers are on this code path at all.
    """
    config = config_with_workspace(tmp_path / "instance-a", str(tmp_path / "ws"))
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, str(config)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(completed.stdout) == []


def test_invalid_json_raises(tmp_path: Path) -> None:
    instance = tmp_path / "instance-a"
    instance.mkdir()
    config = instance / "config.json"
    config.write_text("{not json", encoding="utf-8")

    with pytest.raises(FleetPathError, match="invalid JSON"):
        resolve_instance_paths(config)


def test_a_non_object_config_root_raises(tmp_path: Path) -> None:
    config = write_config(tmp_path / "instance-a", ["agents"])

    with pytest.raises(FleetPathError, match="top level must be a JSON object, found list"):
        resolve_instance_paths(config)


@pytest.mark.parametrize("value", [123, None, True, ["/ws"], {"path": "/ws"}, "", "   "])
def test_an_unusable_workspace_value_raises(tmp_path: Path, value: object) -> None:
    """A blank workspace would resolve to the working directory and confine nothing."""
    config = config_with_workspace(tmp_path / "instance-a", value)

    with pytest.raises(FleetPathError, match="workspace must be a non-empty string"):
        resolve_instance_paths(config)


@pytest.mark.parametrize(
    "data",
    [{"agents": 5}, {"agents": None}, {"agents": {"defaults": "yes"}}, {"agents": {"defaults": 0}}],
)
def test_a_mistyped_agents_section_raises(tmp_path: Path, data: object) -> None:
    config = write_config(tmp_path / "instance-a", data)

    with pytest.raises(FleetPathError, match="must be a JSON object"):
        resolve_instance_paths(config)


def test_an_unreadable_config_file_raises(tmp_path: Path) -> None:
    config = config_with_workspace(tmp_path / "instance-a", str(tmp_path / "ws"))
    config.chmod(0o000)
    try:
        if os.access(config, os.R_OK):  # running as root: the mode does not bite
            pytest.skip("cannot make a file unreadable for this user")
        with pytest.raises(FleetPathError, match="unable to read file"):
            resolve_instance_paths(config)
    finally:
        config.chmod(0o600)


def test_a_config_file_that_is_not_utf8_raises(tmp_path: Path) -> None:
    instance = tmp_path / "instance-a"
    instance.mkdir()
    config = instance / "config.json"
    config.write_bytes(b'{"agents": {"defaults": {"workspace": "\xff\xfe"}}}')

    with pytest.raises(FleetPathError, match="not valid UTF-8"):
        resolve_instance_paths(config)


# ---------------------------------------------------------------------------
# The read/derive split
#
# ``nanobot.fleet.validate`` needs more out of a config file than the two
# directories — the raw workspace value and the declared ports — and reading a
# confinement-critical file twice would let the two reads disagree.
# ---------------------------------------------------------------------------


def test_reading_and_deriving_compose_into_resolve(tmp_path: Path) -> None:
    config = config_with_workspace(tmp_path / "instance-a", str(tmp_path / "ws"))

    assert instance_paths(read_instance_config(config)) == resolve_instance_paths(config)


def test_read_instance_config_returns_the_document_and_the_unresolved_path(
    tmp_path: Path,
) -> None:
    """The path is expanded but not resolved: the config dir comes from the
    parent *as written*, so resolving the file here would relocate it."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "target.json").write_text(json.dumps({"x": 1}), encoding="utf-8")
    instance = tmp_path / "instance-a"
    instance.mkdir()
    (instance / "config.json").symlink_to(real / "target.json")

    config = read_instance_config(instance / "config.json")

    assert isinstance(config, InstanceConfig)
    assert config.path == instance / "config.json"
    assert config.data == {"x": 1}


def test_declared_workspace_returns_the_value_as_written(tmp_path: Path) -> None:
    """Raw, because "the operator declared a relative workspace" is only visible
    before expansion — and that is the refusal ``validate`` has to make."""
    config = read_instance_config(config_with_workspace(tmp_path / "instance-a", "./ws"))

    assert declared_workspace(config) == "./ws"


def test_declared_workspace_falls_back_to_the_schema_default(tmp_path: Path) -> None:
    config = read_instance_config(write_config(tmp_path / "instance-a", {}))

    assert declared_workspace(config) == DEFAULT_WORKSPACE


def test_declared_workspace_raises_on_an_unusable_value(tmp_path: Path) -> None:
    config = read_instance_config(config_with_workspace(tmp_path / "instance-a", ""))

    with pytest.raises(FleetPathError, match="workspace must be a non-empty string"):
        declared_workspace(config)


def test_deriving_paths_never_re_reads_the_config_file(tmp_path: Path) -> None:
    """The one read is the one that counts.

    Proved by deleting the file: ``instance_paths`` and ``declared_workspace``
    still work, so ``validate`` cannot get a workspace from one read of a
    confinement-critical file and a port from a different one. (Both still
    canonicalise, which stats the path — they create nothing, they just do not
    re-open the document.)
    """
    instance = tmp_path / "instance-a"
    config = read_instance_config(config_with_workspace(instance, str(tmp_path / "ws")))
    (instance / "config.json").unlink()

    assert declared_workspace(config) == str(tmp_path / "ws")
    assert instance_paths(config) == InstancePaths(
        config_dir=instance.resolve(),
        workspace=(tmp_path / "ws").resolve(),
    )

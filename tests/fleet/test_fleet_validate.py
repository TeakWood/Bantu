"""Tests for the fleet's refusal rules.

The rules exist because the failures they prevent are silent: a Seatbelt deny
naming a path that does not resolve confines nothing and reports nothing, and two
instances sharing a directory look perfectly healthy right up until one reads the
other's sessions. So most of what is asserted here is that a *plausible-looking*
fleet is refused, and that the refusal names whom to go and fix.

Multi-instance fixtures give every instance an explicit, distinct port unless the
test is about ports. Left on the defaults they would all bind 8900, and a port
issue would quietly inflate the issue counts the overlap tests assert on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nanobot.fleet.config import FleetFile
from nanobot.fleet.validate import (
    DEFAULT_API_PORT,
    DEFAULT_GATEWAY_PORT,
    FleetValidationError,
    ResolvedInstance,
    validate_fleet,
    validate_fleet_file,
)

FLEET_PATH = Path("/fleet/fleet.json")


def write_instance_config(
    directory: Path,
    *,
    workspace: object | None = None,
    api_port: object | None = None,
    raw: object | None = None,
) -> Path:
    """Write one instance's nanobot config and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    data: dict[str, object] = {}
    if raw is None:
        if workspace is not None:
            data["agents"] = {"defaults": {"workspace": workspace}}
        if api_port is not None:
            data["api"] = {"port": api_port}
    path.write_text(json.dumps(data if raw is None else raw), encoding="utf-8")
    return path


def default_layout(root: Path, name: str, *, api_port: object | None = None) -> Path:
    """Write an instance laid out the way nanobot lays one out by itself.

    ``<root>/<name>/config.json`` with ``<root>/<name>/workspace`` beneath it —
    the ``~/.nanobot-x/workspace`` under ``~/.nanobot-x/`` shape.
    """
    directory = root / name
    return write_instance_config(
        directory, workspace=str(directory / "workspace"), api_port=api_port
    )


def entry(
    config: Path | str,
    *,
    mode: str = "serve",
    memory_limit_mb: int = 512,
) -> dict[str, object]:
    """One fleet-document instance entry."""
    return {"config": str(config), "mode": mode, "memoryLimitMb": memory_limit_mb}


def fleet_of(**instances: dict[str, object]) -> FleetFile:
    """Build a parsed fleet document from instance entries."""
    return FleetFile.model_validate({"instances": instances})


def refusal(fleet: FleetFile) -> FleetValidationError:
    """Validate and return the error, failing the test if it was accepted."""
    with pytest.raises(FleetValidationError) as excinfo:
        validate_fleet(FLEET_PATH, fleet)
    return excinfo.value


def tree_snapshot(root: Path) -> dict[str, object]:
    """Capture every path under ``root`` with its bytes and mtime."""
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


# --------------------------------------------------------------------------
# The accepted layouts
# --------------------------------------------------------------------------


def test_the_default_layout_is_accepted(tmp_path: Path) -> None:
    """A workspace inside its *own* config dir is normal, not an overlap."""
    config = default_layout(tmp_path, "alpha")

    resolved = validate_fleet(FLEET_PATH, fleet_of(alpha=entry(config)))

    assert len(resolved) == 1
    instance = resolved[0]
    assert instance.name == "alpha"
    assert instance.config_dir == (tmp_path / "alpha").resolve()
    assert instance.workspace == (tmp_path / "alpha" / "workspace").resolve()
    assert instance.workspace.is_relative_to(instance.config_dir)


def test_a_two_instance_default_layout_is_accepted(tmp_path: Path) -> None:
    """The layout the feature is actually for: side-by-side ``~/.nanobot-x``."""
    resolved = validate_fleet(
        FLEET_PATH,
        fleet_of(
            alpha=entry(default_layout(tmp_path, "alpha", api_port=9001)),
            beta=entry(default_layout(tmp_path, "beta", api_port=9002)),
        ),
    )

    assert [instance.name for instance in resolved] == ["alpha", "beta"]
    assert resolved[0].workspace != resolved[1].workspace
    assert resolved[0].config_dir != resolved[1].config_dir


def test_instances_are_returned_in_document_order(tmp_path: Path) -> None:
    resolved = validate_fleet(
        FLEET_PATH,
        fleet_of(
            zulu=entry(default_layout(tmp_path, "zulu", api_port=9001)),
            alpha=entry(default_layout(tmp_path, "alpha", api_port=9002)),
            mike=entry(default_layout(tmp_path, "mike", api_port=9003)),
        ),
    )

    assert [instance.name for instance in resolved] == ["zulu", "alpha", "mike"]


def test_the_resolved_instance_carries_its_entry_for_downstream_beads(tmp_path: Path) -> None:
    config = default_layout(tmp_path, "alpha")

    instance = validate_fleet(FLEET_PATH, fleet_of(alpha=entry(config, memory_limit_mb=256)))[0]

    assert isinstance(instance, ResolvedInstance)
    assert instance.entry.memory_limit_mb == 256
    assert instance.mode == "serve"
    assert instance.config_path == config


def test_validation_creates_nothing(tmp_path: Path) -> None:
    """Workspaces do not exist yet, and a fleet may still be refused."""
    config = default_layout(tmp_path, "alpha")
    before = tree_snapshot(tmp_path)

    instance = validate_fleet(FLEET_PATH, fleet_of(alpha=entry(config)))[0]

    assert tree_snapshot(tmp_path) == before
    assert not instance.workspace.exists()


def test_a_refused_fleet_creates_nothing_either(tmp_path: Path) -> None:
    shared = str(tmp_path / "shared-ws")
    write_instance_config(tmp_path / "alpha", workspace=shared, api_port=9001)
    write_instance_config(tmp_path / "beta", workspace=shared, api_port=9002)
    before = tree_snapshot(tmp_path)

    refusal(
        fleet_of(
            alpha=entry(tmp_path / "alpha" / "config.json"),
            beta=entry(tmp_path / "beta" / "config.json"),
        )
    )

    assert tree_snapshot(tmp_path) == before


def test_a_missing_workspace_key_is_accepted_and_uses_the_schema_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config = write_instance_config(tmp_path / "alpha")

    instance = validate_fleet(FLEET_PATH, fleet_of(alpha=entry(config)))[0]

    assert instance.workspace == (home / ".nanobot" / "workspace").resolve()


# --------------------------------------------------------------------------
# Rule 1: absolute after expansion, and existing
# --------------------------------------------------------------------------


def test_a_relative_config_path_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It would resolve against the supervisor's cwd, which nobody declared."""
    default_layout(tmp_path, "alpha")
    monkeypatch.chdir(tmp_path)

    error = refusal(fleet_of(alpha=entry("alpha/config.json")))

    assert error.instances == ("alpha",)
    assert "must be absolute" in str(error)
    assert "instances.alpha" in str(error)


def test_a_tilde_config_path_is_absolute_after_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``~`` is not relative — the rule is about the expanded path."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    default_layout(home, "alpha")

    resolved = validate_fleet(FLEET_PATH, fleet_of(alpha=entry("~/alpha/config.json")))

    assert resolved[0].config_dir == (home / "alpha").resolve()


def test_a_missing_config_file_is_refused(tmp_path: Path) -> None:
    error = refusal(fleet_of(alpha=entry(tmp_path / "alpha" / "config.json")))

    assert error.instances == ("alpha",)
    assert "does not exist" in str(error)


def test_a_config_path_that_is_a_directory_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "alpha" / "config.json"
    directory.mkdir(parents=True)

    assert "is not a file" in str(refusal(fleet_of(alpha=entry(directory))))


def test_an_unparseable_config_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "alpha").mkdir()
    config = tmp_path / "alpha" / "config.json"
    config.write_text("{not json", encoding="utf-8")

    assert "invalid JSON" in str(refusal(fleet_of(alpha=entry(config))))


def test_a_relative_workspace_declaration_is_refused(tmp_path: Path) -> None:
    """Checked on the raw value: by the time it is resolved it looks absolute."""
    config = write_instance_config(tmp_path / "alpha", workspace="workspace")

    error = refusal(fleet_of(alpha=entry(config)))

    assert error.instances == ("alpha",)
    assert "agents.defaults.workspace must be absolute" in str(error)


def test_a_tilde_workspace_declaration_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    config = write_instance_config(tmp_path / "alpha", workspace="~/ws-alpha")

    instance = validate_fleet(FLEET_PATH, fleet_of(alpha=entry(config)))[0]

    assert instance.workspace == (home / "ws-alpha").resolve()


def test_a_blank_workspace_declaration_is_refused(tmp_path: Path) -> None:
    config = write_instance_config(tmp_path / "alpha", workspace="   ")

    assert "non-empty string" in str(refusal(fleet_of(alpha=entry(config))))


# --------------------------------------------------------------------------
# Rule 2: no overlap between distinct instances
# --------------------------------------------------------------------------


def test_two_instances_sharing_a_workspace_are_refused(tmp_path: Path) -> None:
    shared = str(tmp_path / "shared-ws")
    error = refusal(
        fleet_of(
            alpha=entry(write_instance_config(tmp_path / "alpha", workspace=shared, api_port=9001)),
            beta=entry(write_instance_config(tmp_path / "beta", workspace=shared, api_port=9002)),
        )
    )

    assert error.instances == ("alpha", "beta")
    assert "alpha's workspace and beta's workspace are the same directory" in str(error)
    assert "instances.alpha, instances.beta" in str(error)


def test_a_workspace_nested_in_a_peers_workspace_is_refused(tmp_path: Path) -> None:
    outer = tmp_path / "outer-ws"
    error = refusal(
        fleet_of(
            alpha=entry(
                write_instance_config(tmp_path / "alpha", workspace=str(outer), api_port=9001)
            ),
            beta=entry(
                write_instance_config(
                    tmp_path / "beta", workspace=str(outer / "inner"), api_port=9002
                )
            ),
        )
    )

    assert error.instances == ("alpha", "beta")
    assert "beta's workspace" in str(error)
    assert "is inside" in str(error)


def test_nesting_is_caught_in_either_document_order(tmp_path: Path) -> None:
    """The containing instance listed second must be caught just the same."""
    outer = tmp_path / "outer-ws"
    inner = entry(
        write_instance_config(tmp_path / "alpha", workspace=str(outer / "inner"), api_port=9001)
    )
    containing = entry(
        write_instance_config(tmp_path / "beta", workspace=str(outer), api_port=9002)
    )

    error = refusal(fleet_of(alpha=inner, beta=containing))

    assert error.instances == ("alpha", "beta")
    assert f"alpha's workspace {(outer / 'inner').resolve()} is inside beta's workspace" in str(
        error
    )


def test_a_workspace_nested_in_a_peers_config_dir_is_refused(tmp_path: Path) -> None:
    """The same nesting that is legal inside one instance is fatal across two."""
    alpha_dir = tmp_path / "alpha"
    error = refusal(
        fleet_of(
            alpha=entry(default_layout(tmp_path, "alpha", api_port=9001)),
            beta=entry(
                write_instance_config(
                    tmp_path / "beta", workspace=str(alpha_dir / "stolen"), api_port=9002
                )
            ),
        )
    )

    assert error.instances == ("alpha", "beta")
    assert "beta's workspace" in str(error)
    assert "alpha's config directory" in str(error)


def test_two_instances_sharing_a_config_dir_are_refused(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    for name, port in (("alpha", 9001), ("beta", 9002)):
        (shared / f"{name}.json").write_text(
            json.dumps(
                {
                    "agents": {"defaults": {"workspace": str(tmp_path / f"ws-{name}")}},
                    "api": {"port": port},
                }
            ),
            encoding="utf-8",
        )

    error = refusal(
        fleet_of(alpha=entry(shared / "alpha.json"), beta=entry(shared / "beta.json"))
    )

    assert len(error.issues) == 1
    assert "config directory" in str(error)
    assert "the same directory" in str(error)


def test_a_config_dir_nested_in_a_peers_config_dir_is_refused(tmp_path: Path) -> None:
    error = refusal(
        fleet_of(
            alpha=entry(
                write_instance_config(
                    tmp_path / "alpha", workspace=str(tmp_path / "wa"), api_port=9001
                )
            ),
            beta=entry(
                write_instance_config(
                    tmp_path / "alpha" / "beta", workspace=str(tmp_path / "wb"), api_port=9002
                )
            ),
        )
    )

    assert error.instances == ("alpha", "beta")
    assert "config directory" in str(error)


def test_two_instances_pointing_at_the_same_config_file_are_refused(tmp_path: Path) -> None:
    config = default_layout(tmp_path, "alpha")

    error = refusal(fleet_of(alpha=entry(config), beta=entry(config)))

    assert error.instances == ("alpha", "beta")


def test_overlap_through_a_symlink_is_caught(tmp_path: Path) -> None:
    """Textually distinct, identical on disk — the reason paths are canonical."""
    real = tmp_path / "real-ws"
    real.mkdir()
    link = tmp_path / "linked-ws"
    link.symlink_to(real, target_is_directory=True)

    error = refusal(
        fleet_of(
            alpha=entry(
                write_instance_config(tmp_path / "alpha", workspace=str(real), api_port=9001)
            ),
            beta=entry(
                write_instance_config(tmp_path / "beta", workspace=str(link), api_port=9002)
            ),
        )
    )

    assert error.instances == ("alpha", "beta")
    assert "the same directory" in str(error)


def test_one_issue_per_overlapping_pair(tmp_path: Path) -> None:
    """A pair usually overlaps several ways at once; say it once."""
    shared = tmp_path / "shared"
    error = refusal(
        fleet_of(
            alpha=entry(
                write_instance_config(
                    shared / "alpha", workspace=str(shared / "ws"), api_port=9001
                )
            ),
            beta=entry(
                write_instance_config(shared / "beta", workspace=str(shared / "ws"), api_port=9002)
            ),
        )
    )

    assert len(error.issues) == 1


def test_every_overlapping_pair_is_named(tmp_path: Path) -> None:
    shared = str(tmp_path / "shared-ws")
    error = refusal(
        fleet_of(
            **{
                name: entry(write_instance_config(tmp_path / name, workspace=shared, api_port=port))
                for name, port in (("alpha", 9001), ("beta", 9002), ("gamma", 9003))
            }
        )
    )

    assert {issue.instances for issue in error.issues} == {
        ("alpha", "beta"),
        ("alpha", "gamma"),
        ("beta", "gamma"),
    }


def test_an_unresolvable_instance_does_not_suppress_its_peers(tmp_path: Path) -> None:
    """A missing config file is reported alongside an unrelated overlap."""
    shared = str(tmp_path / "shared-ws")
    error = refusal(
        fleet_of(
            gone=entry(tmp_path / "gone" / "config.json"),
            alpha=entry(write_instance_config(tmp_path / "alpha", workspace=shared, api_port=9001)),
            beta=entry(write_instance_config(tmp_path / "beta", workspace=shared, api_port=9002)),
        )
    )

    assert error.instances == ("gone", "alpha", "beta")
    assert len(error.issues) == 2


# --------------------------------------------------------------------------
# Rule 3: a config dir inside its own workspace
# --------------------------------------------------------------------------


def test_a_config_dir_equal_to_its_own_workspace_is_refused(tmp_path: Path) -> None:
    """``JsonlSessionStore`` treats equality as inside, and so does this."""
    instance = tmp_path / "alpha"
    config = write_instance_config(instance, workspace=str(instance))

    error = refusal(fleet_of(alpha=entry(config)))

    assert error.instances == ("alpha",)
    assert "session storage must be outside the agent workspace" in str(error)


def test_a_config_dir_inside_its_own_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    config = write_instance_config(workspace / "conf", workspace=str(workspace))

    error = refusal(fleet_of(alpha=entry(config)))

    assert error.instances == ("alpha",)
    assert "session storage must be outside the agent workspace" in str(error)


def test_the_self_inversion_reason_is_nanobots_own(tmp_path: Path) -> None:
    """Pin the wording to the session store's, so the two do not drift apart."""
    from nanobot.session.manager import JsonlSessionStore

    source = Path(JsonlSessionStore.__init__.__code__.co_filename).read_text(encoding="utf-8")
    assert "session storage must be outside the agent workspace" in source

    config = write_instance_config(tmp_path / "ws", workspace=str(tmp_path / "ws"))
    assert "session storage must be outside the agent workspace" in str(
        refusal(fleet_of(alpha=entry(config)))
    )


def test_a_self_inverted_instance_still_takes_part_in_overlap_checks(tmp_path: Path) -> None:
    """Report everything in one pass rather than one problem per run."""
    workspace = tmp_path / "ws"
    error = refusal(
        fleet_of(
            alpha=entry(write_instance_config(workspace, workspace=str(workspace), api_port=9001)),
            beta=entry(
                write_instance_config(tmp_path / "beta", workspace=str(workspace), api_port=9002)
            ),
        )
    )

    assert len(error.issues) == 2
    assert error.issues[0].instances == ("alpha",)
    assert error.issues[1].instances == ("alpha", "beta")


# --------------------------------------------------------------------------
# Rule 4: duplicate ports
# --------------------------------------------------------------------------


def test_two_serve_instances_on_the_default_api_port_are_reported(tmp_path: Path) -> None:
    """Neither declares a port, and both would bind 8900."""
    error = refusal(
        fleet_of(
            alpha=entry(default_layout(tmp_path, "alpha"), mode="serve"),
            beta=entry(default_layout(tmp_path, "beta"), mode="serve"),
        )
    )

    assert error.instances == ("alpha", "beta")
    assert f"bind port {DEFAULT_API_PORT}" in str(error)
    assert "alpha (api.port)" in str(error)


def test_two_gateway_instances_on_the_default_gateway_port_are_reported(tmp_path: Path) -> None:
    error = refusal(
        fleet_of(
            alpha=entry(default_layout(tmp_path, "alpha"), mode="gateway"),
            beta=entry(default_layout(tmp_path, "beta"), mode="gateway"),
        )
    )

    assert f"bind port {DEFAULT_GATEWAY_PORT}" in str(error)
    assert "gateway.port" in str(error)


def test_distinct_declared_ports_are_accepted(tmp_path: Path) -> None:
    resolved = validate_fleet(
        FLEET_PATH,
        fleet_of(
            alpha=entry(default_layout(tmp_path, "alpha", api_port=9001)),
            beta=entry(default_layout(tmp_path, "beta", api_port=9002)),
        ),
    )

    assert [instance.port for instance in resolved] == [9001, 9002]
    assert {instance.port_setting for instance in resolved} == {"api.port"}


def test_only_the_port_the_mode_binds_is_compared(tmp_path: Path) -> None:
    """Two ``serve`` instances left on the default ``gateway.port`` bind it zero times."""
    for name, port in (("alpha", 9001), ("beta", 9002)):
        directory = tmp_path / name
        write_instance_config(
            directory,
            raw={
                "agents": {"defaults": {"workspace": str(directory / "workspace")}},
                "api": {"port": port},
                "gateway": {"port": DEFAULT_GATEWAY_PORT},
            },
        )

    resolved = validate_fleet(
        FLEET_PATH,
        fleet_of(
            alpha=entry(tmp_path / "alpha" / "config.json", mode="serve"),
            beta=entry(tmp_path / "beta" / "config.json", mode="serve"),
        ),
    )

    assert [instance.port for instance in resolved] == [9001, 9002]
    assert all(instance.port_setting == "api.port" for instance in resolved)


def test_a_serve_and_a_gateway_instance_on_their_defaults_do_not_clash(tmp_path: Path) -> None:
    resolved = validate_fleet(
        FLEET_PATH,
        fleet_of(
            alpha=entry(default_layout(tmp_path, "alpha"), mode="serve"),
            beta=entry(default_layout(tmp_path, "beta"), mode="gateway"),
        ),
    )

    assert [instance.port for instance in resolved] == [DEFAULT_API_PORT, DEFAULT_GATEWAY_PORT]


def test_a_clash_across_the_two_settings_is_reported(tmp_path: Path) -> None:
    """Compared by number, so a ``serve`` instance moved onto 18790 is caught."""
    alpha = default_layout(tmp_path, "alpha", api_port=DEFAULT_GATEWAY_PORT)
    beta = default_layout(tmp_path, "beta")

    error = refusal(fleet_of(alpha=entry(alpha, mode="serve"), beta=entry(beta, mode="gateway")))

    assert f"bind port {DEFAULT_GATEWAY_PORT}" in str(error)
    assert "alpha (api.port)" in str(error)
    assert "beta (gateway.port)" in str(error)


def test_three_instances_on_one_port_produce_one_issue_naming_all_three(tmp_path: Path) -> None:
    error = refusal(
        fleet_of(
            alpha=entry(default_layout(tmp_path, "alpha")),
            beta=entry(default_layout(tmp_path, "beta")),
            gamma=entry(default_layout(tmp_path, "gamma")),
        )
    )

    assert len(error.issues) == 1
    assert error.issues[0].instances == ("alpha", "beta", "gamma")
    assert "3 instances" in str(error)


@pytest.mark.parametrize("value", ["8900", True, None, [8900], {"n": 8900}])
def test_a_port_that_is_not_an_integer_is_not_compared(tmp_path: Path, value: object) -> None:
    """Validating an instance's own config is nanobot's job, not the fleet's, and
    substituting the default here would invent a clash that does not exist."""
    alpha = write_instance_config(
        tmp_path / "alpha",
        raw={
            "agents": {"defaults": {"workspace": str(tmp_path / "alpha" / "ws")}},
            "api": {"port": value},
        },
    )
    beta = default_layout(tmp_path, "beta", api_port=DEFAULT_API_PORT)

    resolved = validate_fleet(FLEET_PATH, fleet_of(alpha=entry(alpha), beta=entry(beta)))

    assert resolved[0].port is None
    assert resolved[1].port == DEFAULT_API_PORT


def test_a_mistyped_port_section_is_not_compared(tmp_path: Path) -> None:
    alpha = write_instance_config(
        tmp_path / "alpha",
        raw={
            "agents": {"defaults": {"workspace": str(tmp_path / "alpha" / "ws")}},
            "api": "nope",
        },
    )

    instance = validate_fleet(FLEET_PATH, fleet_of(alpha=entry(alpha)))[0]

    assert instance.port is None


def test_an_absent_port_section_falls_back_to_the_schema_default(tmp_path: Path) -> None:
    instance = validate_fleet(
        FLEET_PATH, fleet_of(alpha=entry(default_layout(tmp_path, "alpha")))
    )[0]

    assert instance.port == DEFAULT_API_PORT
    assert instance.port_setting == "api.port"


def test_a_port_section_without_a_port_key_falls_back_to_the_default(tmp_path: Path) -> None:
    alpha = write_instance_config(
        tmp_path / "alpha",
        raw={
            "agents": {"defaults": {"workspace": str(tmp_path / "alpha" / "ws")}},
            "api": {"host": "127.0.0.1"},
        },
    )

    assert validate_fleet(FLEET_PATH, fleet_of(alpha=entry(alpha)))[0].port == DEFAULT_API_PORT


def test_default_port_constants_match_the_config_schema() -> None:
    """Guard the two values this module duplicates instead of importing."""
    from nanobot.config.schema import ApiConfig, GatewayConfig

    assert DEFAULT_GATEWAY_PORT == GatewayConfig().port
    assert DEFAULT_API_PORT == ApiConfig().port


# --------------------------------------------------------------------------
# Error rendering and the loading entry point
# --------------------------------------------------------------------------


def test_the_error_names_the_fleet_file_and_every_instance(tmp_path: Path) -> None:
    shared = str(tmp_path / "shared-ws")
    error = refusal(
        fleet_of(
            alpha=entry(write_instance_config(tmp_path / "alpha", workspace=shared, api_port=9001)),
            beta=entry(write_instance_config(tmp_path / "beta", workspace=shared, api_port=9002)),
        )
    )

    rendered = str(error)
    assert rendered.startswith(f"Invalid fleet: {FLEET_PATH}")
    assert "Found 1 problem(s)" in rendered
    assert error.issues[0].location == "instances.alpha, instances.beta"


def test_rendering_stops_after_ten_issues(tmp_path: Path) -> None:
    shared = str(tmp_path / "shared-ws")
    error = refusal(
        fleet_of(
            **{
                f"a{index}": entry(
                    write_instance_config(
                        tmp_path / f"a{index}", workspace=shared, api_port=9000 + index
                    )
                )
                for index in range(1, 7)
            }
        )
    )

    assert len(error.issues) == 15
    rendered = str(error)
    assert rendered.count("are the same directory") == 10
    assert "… and 5 more issue(s)" in rendered


def test_validate_fleet_file_loads_and_validates(tmp_path: Path) -> None:
    config = default_layout(tmp_path, "alpha")
    fleet_file = tmp_path / "fleet.json"
    fleet_file.write_text(json.dumps({"instances": {"alpha": entry(config)}}), encoding="utf-8")

    resolved = validate_fleet_file(fleet_file)

    assert [instance.name for instance in resolved] == ["alpha"]


def test_validate_fleet_file_refuses_an_overlapping_document(tmp_path: Path) -> None:
    shared = str(tmp_path / "shared-ws")
    fleet_file = tmp_path / "fleet.json"
    fleet_file.write_text(
        json.dumps(
            {
                "instances": {
                    "alpha": entry(
                        write_instance_config(tmp_path / "alpha", workspace=shared, api_port=9001)
                    ),
                    "beta": entry(
                        write_instance_config(tmp_path / "beta", workspace=shared, api_port=9002)
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FleetValidationError) as excinfo:
        validate_fleet_file(fleet_file)

    assert excinfo.value.path == fleet_file

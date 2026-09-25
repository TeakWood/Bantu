"""Tests for the fleet document schema and its refusals.

Two properties get more attention than the rest because they are the ones a
regression would make dangerous rather than merely annoying: a rejected value
must never reach the rendered message (an ``env`` entry that turns out to be
``NAME=<secret>`` is the motivating case), and parsing must not touch the
filesystem, so a fleet can be validated without creating or reading anything on
behalf of an instance that is about to be refused.
"""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
from typing import Any

import pytest

from nanobot.config.errors import ConfigIssue
from nanobot.fleet.config import (
    ENV_NAME_PATTERN,
    INSTANCE_NAME_PATTERN,
    FleetConfigError,
    FleetFile,
    FleetInstance,
    load_fleet_file,
    parse_fleet_file,
)

FLEET_PATH = Path("/fleet/fleet.json")


def entry(**overrides: object) -> dict[str, object]:
    """A valid instance entry, with fields replaced or removed by the caller."""
    data: dict[str, object] = {
        "config": "/srv/alpha/config.json",
        "mode": "serve",
        "memoryLimitMb": 2048,
        "env": ["OPENAI_API_KEY"],
    }
    data.update(overrides)
    return {key: value for key, value in data.items() if value is not ...}


def document(name: str = "alpha", **overrides: object) -> str:
    """Serialise a one-instance fleet document."""
    return json.dumps({"instances": {name: entry(**overrides)}})


def refusal(text: str, path: Path = FLEET_PATH) -> FleetConfigError:
    """Parse ``text`` expecting a refusal, and return it."""
    with pytest.raises(FleetConfigError) as caught:
        parse_fleet_file(path, text)
    return caught.value


def rendered(error: FleetConfigError) -> str:
    """Everything a user would see: the rendered error plus every issue field."""
    return "\n".join(
        [str(error), error.summary, *(f"{i.location} {i.message}" for i in error.issues)]
    )


# --- the accepting case -------------------------------------------------------


def test_parses_a_valid_fleet_file_into_typed_models() -> None:
    fleet = parse_fleet_file(
        FLEET_PATH,
        json.dumps(
            {
                "instances": {
                    "alpha": entry(),
                    "beta-2": entry(mode="gateway", memoryLimitMb=512, env=[]),
                }
            }
        ),
    )

    assert isinstance(fleet, FleetFile)
    assert sorted(fleet.instances) == ["alpha", "beta-2"]
    alpha = fleet.instances["alpha"]
    assert isinstance(alpha, FleetInstance)
    assert alpha.config == "/srv/alpha/config.json"
    assert alpha.mode == "serve"
    assert alpha.memory_limit_mb == 2048
    assert alpha.env == ["OPENAI_API_KEY"]
    assert fleet.instances["beta-2"].mode == "gateway"


def test_snake_case_keys_are_accepted_alongside_camel_case() -> None:
    """``Base`` sets ``populate_by_name``; the fleet document inherits it."""
    fleet = parse_fleet_file(
        FLEET_PATH,
        json.dumps({"instances": {"alpha": entry(memoryLimitMb=..., memory_limit_mb=64)}}),
    )

    assert fleet.instances["alpha"].memory_limit_mb == 64


def test_env_defaults_to_empty_rather_than_inheriting_anything() -> None:
    """The one optional field, and its default is the fail-closed one."""
    fleet = parse_fleet_file(FLEET_PATH, document(env=...))

    assert fleet.instances["alpha"].env == []


@pytest.mark.parametrize("name", ["a", "0", "alpha", "beta-2", "a_b-c", "0abc9"])
def test_accepts_conforming_instance_names(name: str) -> None:
    fleet = parse_fleet_file(FLEET_PATH, document(name))

    assert list(fleet.instances) == [name]


# --- one distinct refusal per error kind --------------------------------------


def test_malformed_json_reports_line_and_column() -> None:
    error = refusal('{\n  "instances": {\n')

    assert error.kind == "invalid_json"
    assert "line 3, column 1" in error.summary
    assert str(error).startswith(f"Invalid fleet file: {FLEET_PATH}")


def test_a_non_object_root_is_refused() -> None:
    error = refusal(json.dumps(["alpha"]))

    assert error.kind == "invalid_root"
    assert error.issues == (
        ConfigIssue(path=(), message="Expected an object, but found list."),
    )
    assert error.issues[0].location == "<root>"


def test_an_invalid_instance_name_is_refused_by_name_rule() -> None:
    error = refusal(document("Bad Name"))

    assert error.kind == "invalid_schema"
    assert [i.message for i in error.issues] == [
        f"Instance name must match {INSTANCE_NAME_PATTERN}."
    ]
    # Pydantic's synthetic "[key]" marker is dropped rather than rendered as a
    # redaction, which would imply something was withheld.
    assert error.issues[0].location == "instances.<redacted>"


def test_an_unknown_mode_is_refused() -> None:
    error = refusal(document(mode="daemon"))

    assert error.kind == "invalid_schema"
    assert [i.location for i in error.issues] == ["instances.alpha.mode"]
    assert error.issues[0].message == "Must be 'gateway' or 'serve'."


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_memory_limit_is_refused(limit: int) -> None:
    error = refusal(document(memoryLimitMb=limit))

    assert error.kind == "invalid_schema"
    assert [i.location for i in error.issues] == ["instances.alpha.memoryLimitMb"]
    assert error.issues[0].message == "Must be greater than 0."


def test_a_non_list_env_is_refused() -> None:
    error = refusal(document(env="OPENAI_API_KEY"))

    assert error.kind == "invalid_schema"
    assert [i.location for i in error.issues] == ["instances.alpha.env"]
    assert error.issues[0].message == "Must be a valid list."


def test_an_unreadable_file_is_an_io_error(tmp_path: Path) -> None:
    error: FleetConfigError
    with pytest.raises(FleetConfigError) as caught:
        load_fleet_file(tmp_path / "absent" / "fleet.json")
    error = caught.value

    assert error.kind == "io_error"
    assert error.summary.startswith("Unable to read the file: ")
    assert error.summary.endswith(".")


def test_a_non_utf8_file_is_an_io_error(tmp_path: Path) -> None:
    path = tmp_path / "fleet.json"
    path.write_bytes(b'{"instances": {"alpha": "\xff\xfe"}}')

    with pytest.raises(FleetConfigError) as caught:
        load_fleet_file(path)

    assert caught.value.kind == "io_error"
    assert caught.value.summary == "The file is not valid UTF-8."


def test_load_reads_a_valid_file_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "fleet.json"
    path.write_text(document(), encoding="utf-8")

    fleet = load_fleet_file(path)

    assert fleet.instances["alpha"].memory_limit_mb == 2048


# --- fail-closed shape rules --------------------------------------------------


@pytest.mark.parametrize("field", ["config", "mode", "memoryLimitMb"])
def test_every_confinement_control_is_required(field: str) -> None:
    error = refusal(document(**{field: ...}))

    assert [i.location for i in error.issues] == [f"instances.alpha.{field}"]
    assert error.issues[0].message == "This setting is required."


def test_an_unknown_instance_key_is_refused_rather_than_ignored() -> None:
    """A typo must not silently leave the instance without that control."""
    error = refusal(document(memoryLimitMb=..., memoryLimtMb=2048))

    assert error.kind == "invalid_schema"
    assert sorted(i.location for i in error.issues) == [
        "instances.alpha.memoryLimitMb",
        "instances.alpha.memoryLimtMb",
    ]
    assert "Unknown setting." in [i.message for i in error.issues]


def test_an_unknown_top_level_key_is_refused() -> None:
    error = refusal(json.dumps({"instances": {"alpha": entry()}, "defaults": {}}))

    assert [i.location for i in error.issues] == ["defaults"]
    assert error.issues[0].message == "Unknown setting."


def test_an_empty_fleet_is_refused() -> None:
    error = refusal(json.dumps({"instances": {}}))

    assert error.kind == "invalid_schema"
    assert [i.location for i in error.issues] == ["instances"]


def test_an_empty_config_path_is_refused() -> None:
    error = refusal(document(config=""))

    assert [i.location for i in error.issues] == ["instances.alpha.config"]


def test_an_env_entry_carrying_a_value_is_refused() -> None:
    """The rule that keeps credentials out of the file: names only, never pairs."""
    error = refusal(document(env=["OPENAI_API_KEY=sk-live-secret"]))

    assert error.kind == "invalid_schema"
    assert [i.location for i in error.issues] == ["instances.alpha.env.0"]
    assert error.issues[0].message == f"String should match pattern '{ENV_NAME_PATTERN}'."


# --- redaction ----------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "sk-live-secret"},
        {"config": ["sk-live-secret"]},
        {"memoryLimitMb": "sk-live-secret"},
        {"env": "sk-live-secret"},
        {"env": ["OPENAI_API_KEY=sk-live-secret"]},
        {"memoryLimitMb": ..., "memoryLimtMb": "sk-live-secret"},
    ],
)
def test_a_rejected_value_is_never_echoed(overrides: dict[str, object]) -> None:
    error = refusal(document(**overrides))

    assert "sk-live-secret" not in rendered(error)


def test_a_rejected_instance_name_is_not_echoed() -> None:
    error = refusal(document("Bearer sk-live-secret"))

    assert "sk-live-secret" not in rendered(error)


def test_a_credential_shaped_key_is_redacted_in_the_location() -> None:
    """Free-form keys reach the location; ``ConfigIssue`` redaction governs them."""
    error = refusal(json.dumps({"instances": {"alpha": entry()}, "?sk-live-secret": 1}))

    assert [i.location for i in error.issues] == ["<redacted>"]
    assert "sk-live-secret" not in rendered(error)


def test_rendering_stops_after_ten_issues() -> None:
    names = [f"i{index}" for index in range(12)]
    error = refusal(json.dumps({"instances": {name: entry(mode="bad") for name in names}}))

    text = str(error)
    assert len(error.issues) == 12
    assert text.count("Must be 'gateway' or 'serve'.") == 10
    assert "… and 2 more issue(s)" in text


def test_rendering_omits_the_overflow_line_at_exactly_ten_issues() -> None:
    names = [f"i{index}" for index in range(10)]
    error = refusal(json.dumps({"instances": {name: entry(mode="bad") for name in names}}))

    assert "more issue(s)" not in str(error)


# --- purity -------------------------------------------------------------------


def test_parsing_touches_no_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every filesystem primitive raises for the duration of the parse.

    ``parse_fleet_file`` takes a ``Path`` purely so its errors can name the
    document, and this is what keeps that true: a future edit that reached for
    ``path.exists()`` or opened a referenced instance config would fail here
    rather than quietly reintroduce side effects into fleet validation.
    """

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

    fleet = parse_fleet_file(FLEET_PATH, document())

    assert fleet.instances["alpha"].config == "/srv/alpha/config.json"
    # And the refusal paths are just as pure as the accepting one.
    assert refusal("{").kind == "invalid_json"
    assert refusal("[]").kind == "invalid_root"
    assert refusal(document(mode="daemon")).kind == "invalid_schema"

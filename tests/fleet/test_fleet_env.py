"""Tests for the minimal per-instance environment.

The three cases the fleet's isolation claim rests on: the result is exactly the
base plus this instance's own names; a name another instance declared is absent;
and a declared-but-unset name is missing rather than empty. The last one is the
subtle one — an empty value would satisfy nanobot's ``${VAR}`` resolution and
turn a loud refusal to start into a blank API key discovered much later.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanobot.fleet.config import ENV_NAME_PATTERN, FleetInstance, parse_fleet_file
from nanobot.fleet.env import (
    MINIMAL_ENV_NAMES,
    InstanceEnvError,
    instance_environment,
)

#: A supervisor environment with every base variable set plus two credentials,
#: one per instance, so "leaked from a peer" is distinguishable from "forwarded".
SUPERVISOR_ENV: dict[str, str] = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/Users/supervisor",
    "LANG": "en_US.UTF-8",
    "TMPDIR": "/var/folders/xx/T/",
    "ALPHA_API_KEY": "alpha-secret",
    "BETA_API_KEY": "beta-secret",
    "SHELL": "/bin/zsh",
    "TERM": "xterm-256color",
    "AWS_SECRET_ACCESS_KEY": "supervisor-only",
}

BASE_ONLY: dict[str, str] = {name: SUPERVISOR_ENV[name] for name in MINIMAL_ENV_NAMES}


def test_produces_exactly_the_base_plus_the_instances_own_names() -> None:
    env = instance_environment(["ALPHA_API_KEY"], source=SUPERVISOR_ENV)

    assert env == {**BASE_ONLY, "ALPHA_API_KEY": "alpha-secret"}


def test_an_empty_env_list_yields_only_the_minimal_base() -> None:
    assert instance_environment(source=SUPERVISOR_ENV) == BASE_ONLY
    assert instance_environment([], source=SUPERVISOR_ENV) == BASE_ONLY


def test_base_is_exactly_path_home_lang_tmpdir() -> None:
    # Pinned rather than derived: TERM and SHELL are set in SUPERVISOR_ENV and
    # both are the kind of variable a "reasonable defaults" edit would add back.
    assert MINIMAL_ENV_NAMES == ("PATH", "HOME", "LANG", "TMPDIR")
    assert set(instance_environment(source=SUPERVISOR_ENV)) == set(MINIMAL_ENV_NAMES)


def test_a_variable_listed_for_a_different_instance_is_absent() -> None:
    alpha = instance_environment(["ALPHA_API_KEY"], source=SUPERVISOR_ENV)
    beta = instance_environment(["BETA_API_KEY"], source=SUPERVISOR_ENV)

    assert "BETA_API_KEY" not in alpha
    assert "ALPHA_API_KEY" not in beta
    # Nor does either see a credential no entry declared at all.
    assert "AWS_SECRET_ACCESS_KEY" not in alpha
    assert "AWS_SECRET_ACCESS_KEY" not in beta


def test_peer_isolation_holds_for_instances_from_one_fleet_document(
    tmp_path: Path,
) -> None:
    """The same property, driven through a parsed fleet file rather than lists."""
    document = """
    {
      "instances": {
        "alpha": {
          "config": "/fleet/alpha/config.json",
          "mode": "serve",
          "memoryLimitMb": 512,
          "env": ["ALPHA_API_KEY"]
        },
        "beta": {
          "config": "/fleet/beta/config.json",
          "mode": "gateway",
          "memoryLimitMb": 512,
          "env": ["BETA_API_KEY"]
        }
      }
    }
    """
    fleet = parse_fleet_file(tmp_path / "fleet.json", document)

    built = {
        name: instance_environment(instance.env, source=SUPERVISOR_ENV)
        for name, instance in fleet.instances.items()
    }

    assert built["alpha"] == {**BASE_ONLY, "ALPHA_API_KEY": "alpha-secret"}
    assert built["beta"] == {**BASE_ONLY, "BETA_API_KEY": "beta-secret"}


def test_a_listed_but_unset_name_is_omitted_rather_than_set_empty() -> None:
    env = instance_environment(
        ["ALPHA_API_KEY", "NEVER_EXPORTED"], source=SUPERVISOR_ENV
    )

    assert "NEVER_EXPORTED" not in env
    assert env == {**BASE_ONLY, "ALPHA_API_KEY": "alpha-secret"}
    # The distinction that matters downstream: nanobot's ``${VAR}`` resolution
    # raises on a name that is absent and silently succeeds on one set to "".
    assert "" not in env.values()


def test_an_unset_base_variable_is_omitted_rather_than_invented() -> None:
    """A value the supervisor does not have is not synthesised for the base either."""
    source = {name: SUPERVISOR_ENV[name] for name in MINIMAL_ENV_NAMES}
    del source["TMPDIR"]

    env = instance_environment(source=source)

    assert "TMPDIR" not in env
    assert env == {
        "PATH": SUPERVISOR_ENV["PATH"],
        "HOME": SUPERVISOR_ENV["HOME"],
        "LANG": SUPERVISOR_ENV["LANG"],
    }


def test_an_explicitly_empty_supervisor_value_is_forwarded_unchanged() -> None:
    """Omission is for *unset* names only; a real empty export is not rewritten."""
    env = instance_environment(
        ["ALPHA_API_KEY"], source={**SUPERVISOR_ENV, "ALPHA_API_KEY": ""}
    )

    assert env["ALPHA_API_KEY"] == ""


def test_an_entirely_empty_supervisor_environment_yields_nothing() -> None:
    assert instance_environment(["ALPHA_API_KEY"], source={}) == {}


def test_listing_a_base_variable_does_not_duplicate_or_shadow_it() -> None:
    env = instance_environment(["HOME", "ALPHA_API_KEY"], source=SUPERVISOR_ENV)

    assert env == {**BASE_ONLY, "ALPHA_API_KEY": "alpha-secret"}


def test_order_is_base_then_declaration_order() -> None:
    """Deterministic ordering keeps the launcher's ``env -i`` argv reproducible."""
    env = instance_environment(
        ["BETA_API_KEY", "ALPHA_API_KEY"], source=SUPERVISOR_ENV
    )

    assert list(env) == [*MINIMAL_ENV_NAMES, "BETA_API_KEY", "ALPHA_API_KEY"]


@pytest.mark.parametrize(
    "name",
    [
        "OPENAI_API_KEY=sk-live-secret",
        "1_LEADING_DIGIT",
        "has-a-dash",
        "has a space",
        "TRAILING_NEWLINE\n",
        "",
    ],
)
def test_a_name_that_is_not_a_variable_name_is_refused(name: str) -> None:
    """Re-checked here because the result becomes ``NAME=value`` argv words.

    ``nanobot.fleet.config`` rejects these too, but this module is what the
    launcher calls, so an ``=`` smuggling in a second variable has to be
    impossible on this side as well.
    """
    with pytest.raises(InstanceEnvError) as caught:
        instance_environment([name], source=SUPERVISOR_ENV)

    assert caught.value.name == name
    assert ENV_NAME_PATTERN in str(caught.value)


def test_a_rejected_name_is_refused_before_any_value_is_read() -> None:
    """A bad name last in the list still refuses without reading a credential."""
    reads: list[str] = []

    class RecordingEnv(dict[str, str]):
        def get(self, key: str, default: str | None = None) -> str | None:  # type: ignore[override]
            reads.append(key)
            return super().get(key, default)

    with pytest.raises(InstanceEnvError):
        instance_environment(
            ["ALPHA_API_KEY", "BAD=NAME"], source=RecordingEnv(SUPERVISOR_ENV)
        )

    assert reads == []


def test_the_valid_name_shape_matches_the_fleet_document_rule() -> None:
    """Both sides accept the same names, so neither is a surprise gate."""
    accepted = "A_VALID_NAME_9"
    FleetInstance(config="c", mode="serve", memoryLimitMb=1, env=[accepted])

    assert accepted in instance_environment(
        [accepted], source={**SUPERVISOR_ENV, accepted: "v"}
    )


def test_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLEET_ENV_PROBE", "from-os-environ")
    monkeypatch.setenv("HOME", "/Users/probe")
    monkeypatch.delenv("FLEET_ENV_ABSENT", raising=False)

    env = instance_environment(["FLEET_ENV_PROBE", "FLEET_ENV_ABSENT"])

    assert env["FLEET_ENV_PROBE"] == "from-os-environ"
    assert env["HOME"] == "/Users/probe"
    assert "FLEET_ENV_ABSENT" not in env


def test_the_result_is_a_fresh_mutable_dict_not_a_view_of_os_environ() -> None:
    env = instance_environment(["ALPHA_API_KEY"], source=SUPERVISOR_ENV)
    env["ALPHA_API_KEY"] = "mutated"

    assert SUPERVISOR_ENV["ALPHA_API_KEY"] == "alpha-secret"
    assert os.environ is not env

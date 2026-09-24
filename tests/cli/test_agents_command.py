"""`nanobot agents list`: the machine-readable view of the configured agents."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app

runner = CliRunner()


def write_config(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def base_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "agents": {
            "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
        },
    }
    payload.update(overrides)
    return payload


def list_agents(config_path: Path) -> list[dict[str, Any]]:
    """Run the command and return the parsed document, failing on any exit or noise."""
    result = runner.invoke(app, ["agents", "list", "--json", "--config", str(config_path)])
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


# --- shape ------------------------------------------------------------------


def test_a_config_with_no_named_agents_lists_only_default(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "config.json", base_payload())

    entries = list_agents(config_path)

    assert [entry["name"] for entry in entries] == ["default"]
    assert entries[0]["workspace"] == str(Path.home() / "default-workspace")


def test_two_named_agents_list_after_default(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.json",
        base_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"research": {}, "ops": {}},
            }
        ),
    )

    entries = list_agents(config_path)

    assert [entry["name"] for entry in entries] == ["default", "research", "ops"]


def test_every_entry_carries_the_documented_keys(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.json",
        base_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"research": {}},
            }
        ),
    )

    entries = list_agents(config_path)

    for entry in entries:
        assert {"name", "workspace", "model", "channels"} <= set(entry)


# --- workspace --------------------------------------------------------------


def test_a_named_agents_workspace_is_absolute_and_tilde_expanded(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.json",
        base_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"research": {}, "ops": {"workspace": "~/ops-workspace"}},
            }
        ),
    )

    by_name = {entry["name"]: entry for entry in list_agents(config_path)}

    for entry in by_name.values():
        assert Path(entry["workspace"]).is_absolute()
        assert "~" not in entry["workspace"]
    assert by_name["research"]["workspace"] == str(Path.home() / ".nanobot" / "agents" / "research")
    assert by_name["ops"]["workspace"] == str(Path.home() / "ops-workspace")


def test_a_relative_workspace_is_anchored_rather_than_reported_relative(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.json",
        base_payload(
            agents={"defaults": {"workspace": "relative-workspace", "model": "base/model"}}
        ),
    )

    entries = list_agents(config_path)

    assert Path(entries[0]["workspace"]).is_absolute()
    assert Path(entries[0]["workspace"]).name == "relative-workspace"


# --- model ------------------------------------------------------------------


def test_a_named_agent_reports_its_own_model_or_the_inherited_one(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.json",
        base_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"research": {"model": "research/model"}, "ops": {}},
            }
        ),
    )

    by_name = {entry["name"]: entry["model"] for entry in list_agents(config_path)}

    assert by_name == {
        "default": "base/model",
        "research": "research/model",
        "ops": "base/model",
    }


def test_model_is_the_resolved_preset_not_the_raw_entry(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.json",
        base_payload(
            agents={
                "defaults": {
                    "workspace": "~/default-workspace",
                    "model": "base/model",
                    "modelPreset": "fast",
                },
                "named": {"research": {}},
            },
            modelPresets={"fast": {"model": "preset/fast-model"}},
        ),
    )

    by_name = {entry["name"]: entry["model"] for entry in list_agents(config_path)}

    # Both the raw agents.defaults.model and the named entry's absence of one
    # would report "base/model"; the preset is what actually runs.
    assert by_name == {"default": "preset/fast-model", "research": "preset/fast-model"}


# --- channels ---------------------------------------------------------------


def test_channels_list_the_runtime_names_bound_to_each_agent(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.json",
        base_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"research": {}, "ops": {}},
            },
            channels={
                "telegram": {
                    "instances": [
                        {"id": "default", "token": "default-token"},
                        {"id": "research", "token": "research-token", "agent": "research"},
                        {"id": "ops", "token": "ops-token", "agent": "ops"},
                    ]
                }
            },
        ),
    )

    by_name = {entry["name"]: entry["channels"] for entry in list_agents(config_path)}

    assert by_name == {
        "default": ["telegram"],
        "research": ["telegram.research"],
        "ops": ["telegram.ops"],
    }


def test_an_install_without_telegram_reports_no_channels(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "config.json", base_payload())

    assert list_agents(config_path)[0]["channels"] == []


# --- --config ---------------------------------------------------------------


def test_config_selects_the_file_that_is_read(tmp_path: Path) -> None:
    selected = write_config(
        tmp_path / "selected.json",
        base_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"selected-agent": {}},
            }
        ),
    )
    write_config(
        tmp_path / "other.json",
        base_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"other-agent": {}},
            }
        ),
    )

    names = [entry["name"] for entry in list_agents(selected)]

    assert names == ["default", "selected-agent"]


# --- output hygiene ---------------------------------------------------------


def test_json_output_carries_no_rich_markup_or_config_notice(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "config.json", base_payload())

    result = runner.invoke(app, ["agents", "list", "--json", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Using config" not in result.stdout
    assert "\x1b[" not in result.stdout
    assert "[dim]" not in result.stdout
    assert result.stdout.lstrip().startswith("[")
    json.loads(result.stdout)


def test_the_table_view_still_names_every_agent(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.json",
        base_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"research": {}},
            }
        ),
    )

    result = runner.invoke(app, ["agents", "list", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "default" in result.stdout
    assert "research" in result.stdout
    assert "Using config" in result.stdout


# --- no runtime -------------------------------------------------------------


def test_listing_starts_no_gateway_and_connects_no_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.channels.manager import ChannelManager

    config_path = write_config(
        tmp_path / "config.json",
        base_payload(
            agents={
                "defaults": {"workspace": "~/default-workspace", "model": "base/model"},
                "named": {"research": {}},
            },
            channels={
                "telegram": {
                    "enabled": True,
                    "instances": [
                        {"id": "default", "token": "default-token"},
                        {"id": "research", "token": "research-token", "agent": "research"},
                    ],
                }
            },
        ),
    )

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("listing agents must not start a runtime")

    # Every gateway entry point runs an event loop and builds a ChannelManager;
    # neither may be reached by a command that resolves purely from config.
    monkeypatch.setattr(asyncio, "run", unexpected)
    monkeypatch.setattr(ChannelManager, "__init__", unexpected)
    monkeypatch.setattr("nanobot.cli.commands._run_gateway", unexpected)

    names = [entry["name"] for entry in list_agents(config_path)]

    assert names == ["default", "research"]


def test_listing_does_not_resolve_secret_env_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NANOBOT_TEST_AGENTS_LIST_MISSING", raising=False)
    config_path = write_config(
        tmp_path / "config.json",
        base_payload(
            providers={"openrouter": {"apiKey": "${NANOBOT_TEST_AGENTS_LIST_MISSING}"}},
        ),
    )

    entries = list_agents(config_path)

    assert [entry["name"] for entry in entries] == ["default"]

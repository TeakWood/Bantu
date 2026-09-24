"""`nanobot agents list` — the external view of the agent registry.

Covers the CLI contact point in acceptance criteria 1, 2 and 7.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from nanobot.agents import named_agent_workspace
from nanobot.cli.commands import app

from .conftest import write_config

runner = CliRunner()


def _run_json(config_path: Path) -> list[dict[str, object]]:
    result = runner.invoke(app, ["agents", "list", "--json", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_a_config_without_named_agents_lists_only_default(instance_dir: Path) -> None:
    config_path = write_config(
        instance_dir,
        {
            "agents": {
                "defaults": {
                    "model": "anthropic/claude-sonnet-5",
                    "workspace": str(instance_dir / "ws"),
                }
            }
        },
    )

    payload = _run_json(config_path)

    assert payload == [
        {
            "name": "default",
            "workspace": str((instance_dir / "ws").resolve()),
            "model": "anthropic/claude-sonnet-5",
            "channels": [],
        }
    ]


def test_two_named_agents_are_listed_after_default(instance_dir: Path) -> None:
    config_path = write_config(
        instance_dir,
        {
            "agents": {
                "defaults": {
                    "model": "anthropic/claude-sonnet-5",
                    "workspace": str(instance_dir / "ws"),
                },
                "named": {
                    "research": {
                        "model": "anthropic/claude-opus-5-5",
                        "workspace": str(instance_dir / "research-ws"),
                    },
                    "trader": {},
                },
            }
        },
    )

    payload = _run_json(config_path)

    assert [entry["name"] for entry in payload] == ["default", "research", "trader"]
    # A named agent that sets its own model reports it.
    assert payload[1]["model"] == "anthropic/claude-opus-5-5"
    assert payload[1]["workspace"] == str((instance_dir / "research-ws").resolve())
    # One that sets none reports agents.defaults.model...
    assert payload[2]["model"] == "anthropic/claude-sonnet-5"
    # ...and gets ~/.nanobot/agents/<name>.
    assert payload[2]["workspace"] == str(named_agent_workspace("trader").resolve())


def test_each_bot_is_shown_under_its_agent(instance_dir: Path) -> None:
    config_path = write_config(
        instance_dir,
        {
            "channels": {
                "telegram": {
                    "enabled": True,
                    "instances": [
                        {"id": "default", "token": "111:aaa"},
                        {"id": "research", "token": "222:bbb", "agent": "research"},
                        {"id": "trader", "token": "333:ccc", "agent": "trader"},
                    ],
                }
            },
            "agents": {"named": {"research": {}, "trader": {}}},
        },
    )

    channels = {entry["name"]: entry["channels"] for entry in _run_json(config_path)}

    assert channels == {
        "default": ["telegram"],
        "research": ["telegram.research"],
        "trader": ["telegram.trader"],
    }


def test_a_pre_existing_single_bot_config_shows_telegram_under_default(
    instance_dir: Path,
) -> None:
    config_path = write_config(
        instance_dir,
        {"channels": {"telegram": {"enabled": True, "token": "111:aaa"}}},
    )

    payload = _run_json(config_path)

    assert len(payload) == 1
    assert payload[0]["name"] == "default"
    assert payload[0]["channels"] == ["telegram"]


def test_the_table_view_renders_every_agent(instance_dir: Path) -> None:
    config_path = write_config(
        instance_dir,
        {"agents": {"named": {"research": {}, "health": {}}}},
    )

    result = runner.invoke(app, ["agents", "list", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    for name in ("default", "research", "health"):
        assert name in result.stdout

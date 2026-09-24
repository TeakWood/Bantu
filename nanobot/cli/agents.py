"""Typer commands for inspecting the agents an install is configured to run.

Everything here resolves from the config file alone, through
:mod:`nanobot.agents.registry`: no gateway is started, no channel is connected
and no secret env ref is resolved.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from nanobot.cli.runtime_config import _load_inspection_config

__all__ = ["agents_app"]

console = Console()
agents_app = typer.Typer(help="Inspect configured agents")


def _absolute(workspace: Path) -> Path:
    """Return *workspace* as an absolute path without resolving symlinks.

    The registry already expands ``~``; a workspace configured as a relative
    path is interpreted against the working directory everywhere else, so
    anchoring it there is what the agent would actually use.
    """
    return workspace if workspace.is_absolute() else Path.cwd() / workspace


@agents_app.command("list")
def agents_list(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print machine-readable JSON instead of a table",
    ),
) -> None:
    """List every configured agent, default first."""
    from nanobot.agents.registry import agent_registry

    # Inspection rather than runtime loading: listing agents must not resolve
    # secret env refs.  --json also suppresses its "Using config" notice, so
    # stdout carries the document alone.
    _, loaded = _load_inspection_config(config=config, quiet=json_output)
    entries = [
        {**entry.to_dict(), "workspace": str(_absolute(entry.workspace))}
        for entry in agent_registry(loaded)
    ]

    if json_output:
        # Plain stdout, not console.print: Rich would interpret markup and wrap
        # long workspace paths, neither of which survives json.loads.
        typer.echo(json.dumps(entries, indent=2))
        return

    table = Table(title="Agents")
    table.add_column("Agent", style="cyan")
    table.add_column("Model")
    table.add_column("Workspace")
    table.add_column("Channels")
    for entry in entries:
        table.add_row(
            str(entry["name"]),
            str(entry["model"]),
            str(entry["workspace"]),
            ", ".join(entry["channels"]) or "-",
        )
    console.print(table)

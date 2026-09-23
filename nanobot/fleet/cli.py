"""`nanobot fleet` — the supervisor's contact points."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from nanobot.fleet.config import FleetConfigError, expand_path, load_fleet
from nanobot.fleet.sandbox import confinement_available
from nanobot.fleet.supervisor import StopResult, Supervisor, fleet_status, stop_fleet

__all__ = ["StopResult", "create_fleet_app"]

_FLEET_OPTION = typer.Option(..., "--fleet", help="Path to the fleet file")


def create_fleet_app(*, console: Console | None = None) -> typer.Typer:
    """Build the `fleet` sub-app.

    The console is injected so tests can capture output without depending on
    the module-level console the rest of the CLI shares.
    """
    out = console or Console()
    fleet_app = typer.Typer(
        help="Run several nanobot instances, each confined to its own OS process."
    )

    @fleet_app.command("start")
    def fleet_start(  # pyright: ignore[reportUnusedFunction]
        fleet: str = _FLEET_OPTION,
    ) -> None:
        """Validate the fleet file, start every instance, and supervise them."""
        try:
            loaded = load_fleet(fleet)
        except FleetConfigError as exc:
            out.print(f"[red]Error: {exc}[/red]")
            raise typer.Exit(1) from exc
        if not confinement_available():
            out.print(
                "[red]Error: fleet isolation needs macOS sandbox-exec; "
                f"this host is {sys.platform}.[/red]"
            )
            raise typer.Exit(1)
        out.print(f"Starting fleet of {len(loaded.instances)} instance(s) from {loaded.path}")
        supervisor = Supervisor(loaded)
        try:
            supervisor.run()
        except OSError as exc:
            # A fleet that cannot start whole starts nothing; say why.
            out.print(f"[red]Error: an instance could not be started: {exc}[/red]")
            raise typer.Exit(1) from exc
        out.print("Fleet supervisor stopped.")

    @fleet_app.command("status")
    def fleet_status_command(  # pyright: ignore[reportUnusedFunction]
        fleet: str = _FLEET_OPTION,
        as_json: bool = typer.Option(False, "--json", help="Print one JSON object per instance"),
    ) -> None:
        """Report each instance's process, state and limits."""
        entries = fleet_status(expand_path(fleet))
        if entries is None:
            out.print(f"[red]Error: no fleet supervisor has run for {fleet}[/red]")
            raise typer.Exit(1)
        if as_json:
            # Printed without rich so the JSON is never wrapped or styled.
            print(json.dumps(entries, indent=2))
            return
        columns = ("name", "pid", "state", "exit_reason", "memory_limit_mb", "workspace")
        table = Table(title="Fleet")
        for column in columns:
            table.add_column(column)
        for entry in entries:
            table.add_row(*(str(entry.get(column, "")) for column in columns))
        out.print(table)

    @fleet_app.command("stop")
    def fleet_stop(  # pyright: ignore[reportUnusedFunction]
        fleet: str = _FLEET_OPTION,
    ) -> None:
        """Stop every instance's process tree, then the supervisor."""
        result = stop_fleet(expand_path(fleet))
        if not result.stopped:
            out.print(f"[yellow]{result.message}[/yellow]")
            if result.survivors:
                raise typer.Exit(1)
            return
        out.print("[green]Fleet stopped.[/green]")

    return fleet_app

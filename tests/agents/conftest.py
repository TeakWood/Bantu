"""Shared fixtures for the named-agents suite."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

MCP_SERVER_TEMPLATE = textwrap.dedent(
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("{server}")


    @mcp.tool()
    def {tool}(value: str) -> str:
        \"\"\"Echo a value back from the {server} server.\"\"\"
        return f"{server}:" + value


    if __name__ == "__main__":
        mcp.run()
    """
)


@pytest.fixture
def instance_dir(tmp_path: Path) -> Path:
    """Return a directory that holds config.json, outside any agent workspace."""
    path = tmp_path.parent / f"{tmp_path.name}-instance"
    path.mkdir(exist_ok=True)
    return path


def write_config(instance_dir: Path, data: dict[str, Any]) -> Path:
    """Write a config file and return its path."""
    merged: dict[str, Any] = {
        "providers": {"openrouter": {"apiKey": "sk-test-key"}},
    }
    merged.update(data)
    path = instance_dir / "config.json"
    path.write_text(json.dumps(merged), encoding="utf-8")
    return path


def write_mcp_server(directory: Path, server: str, tool: str) -> dict[str, Any]:
    """Write a minimal stdio MCP server and return its config entry."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / f"{server}_server.py"
    script.write_text(MCP_SERVER_TEMPLATE.format(server=server, tool=tool), encoding="utf-8")
    return {"command": sys.executable, "args": [str(script)], "toolTimeout": 30}

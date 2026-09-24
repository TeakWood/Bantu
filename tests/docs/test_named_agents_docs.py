"""The named-agents docs, checked against the implementation they describe.

Docs drift silently: a renamed field or a moved anchor breaks nothing until a
reader tries it.  These tests read the published Markdown and assert it against
the schema, the registry and the Typer app, so a config key the docs invent, an
example that no longer validates, or a cross-reference that no longer resolves
fails here rather than in someone's config file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import BaseModel

from nanobot.agents.registry import agent_registry, route
from nanobot.agents.resolution import named_agent_workspace, resolve_agent_config
from nanobot.cli.agents import agents_app
from nanobot.config.schema import (
    AGENT_NAME_PATTERN,
    RESERVED_AGENT_NAME,
    AgentDefaults,
    Config,
    NamedAgentConfig,
)

DOCS = Path(__file__).resolve().parents[2] / "docs"

CONFIGURATION = DOCS / "configuration.md"
ARCHITECTURE = DOCS / "architecture.md"
CHAT_APPS = DOCS / "chat-apps.md"
MULTIPLE_INSTANCES = DOCS / "multiple-instances.md"
CLI_REFERENCE = DOCS / "cli-reference.md"

# Every page this bead touched; each is link-checked as a whole.
TOUCHED_PAGES = (
    CONFIGURATION,
    ARCHITECTURE,
    CHAT_APPS,
    MULTIPLE_INSTANCES,
    CLI_REFERENCE,
    DOCS / "README.md",
)

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
_EXPLICIT_ANCHOR = re.compile(r'<a\s+id="([^"]+)"')
_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_MD_LINK = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:)([^)]+)\)")
_BACKTICKED = re.compile(r"`([^`]+)`")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _slug(heading: str) -> str:
    """Return the GitHub-style anchor for a Markdown heading."""
    text = re.sub(r"`|\*|<[^>]+>", "", heading).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text)


def _anchors(path: Path) -> set[str]:
    """Return every anchor a link can target in *path*."""
    text = _read(path)
    found = {_slug(title) for _level, title in _HEADING.findall(text)}
    found.update(_EXPLICIT_ANCHOR.findall(text))
    return found


def _section(path: Path, title: str) -> str:
    """Return the body of the heading called *title*, up to the next same-or-higher one."""
    text = _read(path)
    headings = [(m.start(), len(m.group(1)), m.group(2)) for m in _HEADING.finditer(text)]
    for index, (start, level, name) in enumerate(headings):
        if name != title:
            continue
        end = len(text)
        for later_start, later_level, _name in headings[index + 1 :]:
            if later_level <= level:
                end = later_start
                break
        return text[start:end]
    raise AssertionError(f"docs/{path.name} has no heading {title!r}")


def _json_blocks(markdown: str) -> list[dict[str, Any]]:
    return [
        json.loads(body)
        for language, body in _FENCE.findall(markdown)
        if language == "json"
    ]


NAMED_AGENTS = _section(CONFIGURATION, "Named Agents")


# --- the documented schema ---------------------------------------------------


def test_every_named_agent_field_in_the_docs_exists_in_the_schema() -> None:
    """The field table invents nothing: each name is a real serialization alias."""
    table = _section(CONFIGURATION, "What an Entry Accepts")
    documented = {
        name
        for row in table.splitlines()
        if row.startswith("|") and not row.startswith("|---")
        for name in _BACKTICKED.findall(row.split("|")[1])
    }
    aliases = {
        field.serialization_alias or name
        for name, field in NamedAgentConfig.model_fields.items()
    }
    assert documented - aliases == set()


def test_the_field_table_covers_every_field_an_entry_accepts() -> None:
    """Nothing an entry accepts is left undocumented."""
    table = _section(CONFIGURATION, "What an Entry Accepts")
    documented = {
        name
        for row in table.splitlines()
        if row.startswith("|") and not row.startswith("|---")
        for name in _BACKTICKED.findall(row.split("|")[1])
    }
    aliases = {
        field.serialization_alias or name
        for name, field in NamedAgentConfig.model_fields.items()
    }
    assert aliases - documented == set()


def _resolve_config_path(parts: list[str]) -> bool:
    """Whether a dotted `a.b.c` config path names real fields all the way down."""
    model: type[BaseModel] | None = Config
    if parts[0] == "dream":
        # Documented relative to an agent block rather than to the config root.
        model, parts = AgentDefaults, ["dream", *parts[1:]]
    for part in parts:
        if model is None:
            return False
        field = next(
            (
                candidate
                for name, candidate in model.model_fields.items()
                if part in {name, candidate.serialization_alias, candidate.alias}
            ),
            None,
        )
        if field is None:
            return False
        model = _model_of(field.annotation)
    return True


def _model_of(annotation: Any) -> type[BaseModel] | None:
    """Return the BaseModel an annotation ultimately describes, if any."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for argument in get_args(annotation):
        found = _model_of(argument)
        if found is not None:
            return found
    return None


def test_every_dotted_config_path_in_the_section_exists() -> None:
    """`dream.enabled`, `tools.mcpServers`, … name real fields, not near-misses."""
    roots = set(Config.model_fields) | {"dream"}
    paths = {
        token
        # Fences first: their own backticks would otherwise pair with the prose's
        # and swallow every inline token that follows a code block.
        for token in _BACKTICKED.findall(_FENCE.sub("", NAMED_AGENTS))
        if "." in token and token.split(".")[0] in roots
    }
    assert paths, "no dotted config path found to check"
    assert {path for path in paths if not _resolve_config_path(path.split("."))} == set()


def test_tools_is_documented_as_the_one_field_defaults_does_not_have() -> None:
    extra = set(NamedAgentConfig.model_fields) - set(AgentDefaults.model_fields)
    assert extra == {"tools"}
    assert "`tools` is the one field an entry accepts that `agents.defaults` does not" in (
        NAMED_AGENTS
    )


def test_the_documented_name_rule_is_the_implemented_pattern() -> None:
    assert f"`{AGENT_NAME_PATTERN}`" in NAMED_AGENTS
    assert f"`{RESERVED_AGENT_NAME}` is reserved" in NAMED_AGENTS


def test_the_documented_default_workspace_is_the_implemented_one() -> None:
    assert f"`{named_agent_workspace('<name>')}/`" in NAMED_AGENTS


# --- the worked examples -----------------------------------------------------


@pytest.mark.parametrize(
    "page,title",
    [
        (CONFIGURATION, "Named Agents"),
        (CHAT_APPS, None),
    ],
    ids=["configuration", "chat-apps"],
)
def test_documented_examples_validate_against_the_schema(page: Path, title: str | None) -> None:
    """Every JSON example that configures agents or Telegram instances loads."""
    markdown = _section(page, title) if title else _read(page)
    blocks = [
        block
        for block in _json_blocks(markdown)
        if "agents" in block or "channels" in block
    ]
    assert blocks, "no configuration example found to validate"
    for block in blocks:
        Config.model_validate(block)


def _worked_example() -> Config:
    """The `## Named Agents` worked example, loaded exactly as printed."""
    blocks = [block for block in _json_blocks(NAMED_AGENTS) if "agents" in block]
    return Config.model_validate(blocks[0])


def test_the_worked_example_declares_the_agents_it_describes() -> None:
    config = _worked_example()
    assert list(config.agents.named) == ["research", "ops"]


def test_the_worked_example_routes_the_way_the_prose_says() -> None:
    """"research talks to research, ops to ops, everything else to default"."""
    config = _worked_example()
    assert route(config, "telegram.research", None) == "research"
    assert route(config, "telegram.ops", None) == "ops"
    for channel in ("telegram", "feishu.product", "cli", "websocket"):
        assert route(config, channel, None) == RESERVED_AGENT_NAME


def test_the_worked_example_binds_the_channels_the_docs_list() -> None:
    config = _worked_example()
    bound = {entry.name: entry.channels for entry in agent_registry(config)}
    assert bound == {
        "default": ("telegram",),
        "research": ("telegram.research",),
        "ops": ("telegram.ops",),
    }


def test_the_worked_example_inherits_and_overrides_as_documented() -> None:
    """The overlay rules the section states, read off the resolved configs."""
    config = _worked_example()
    research = resolve_agent_config(config, "research")
    ops = resolve_agent_config(config, "ops")

    # Unstated fields keep the configured default, not the schema default.
    assert research.agent.max_tokens == config.agents.defaults.max_tokens
    assert research.agent.timezone == config.agents.defaults.timezone

    # Stated fields win.
    assert research.agent.model_preset == "deep"
    assert ops.agent.model == "gpt-4.1-mini"
    assert ops.agent.timezone == "UTC"

    # workspace is never inherited.
    assert research.workspace != config.workspace_path
    assert research.workspace.as_posix().endswith("/.nanobot/agents/research")
    assert ops.workspace.as_posix().endswith("/ops-workspace")

    # tools.mcpServers is never inherited, in either direction.
    assert set(research.mcp_servers) == {"notes"}
    assert ops.mcp_servers == {}
    assert resolve_agent_config(config, RESERVED_AGENT_NAME).mcp_servers == (
        config.tools.mcp_servers
    )

    # Every other tools field merges field by field.
    assert ops.tools.exec.enable is False
    assert ops.tools.web.enable is config.tools.web.enable


def test_the_telegram_example_keeps_the_default_instance_unnamespaced() -> None:
    """`telegram.<id>` for every instance but `default`, as documented."""
    section = _section(CHAT_APPS, "Chat Apps for Self-Hosted AI Agents")
    example = next(
        block
        for block in _json_blocks(section)
        if isinstance(block.get("channels", {}).get("telegram", {}), dict)
        and "instances" in block["channels"]["telegram"]
    )
    config = Config.model_validate(example)
    bound = {entry.name: entry.channels for entry in agent_registry(config)}
    assert bound[RESERVED_AGENT_NAME] == ("telegram",)


# --- the documented CLI ------------------------------------------------------


def test_the_documented_agents_list_flags_exist() -> None:
    command = next(c for c in agents_app.registered_commands if c.name == "list")
    params = command.callback.__annotations__ if command.callback else {}
    assert {"config", "json_output"} <= set(params)
    for flag in ("nanobot agents list", "--json", "--config <path>"):
        assert flag in _read(CLI_REFERENCE)


def test_the_documented_json_keys_are_the_ones_the_registry_emits() -> None:
    documented = _json_blocks(_section(CLI_REFERENCE, "Agents"))[0]
    emitted = agent_registry(_worked_example())[0].to_dict()
    assert set(documented[0]) == set(emitted)


# --- the boundary and the cross-references -----------------------------------


def test_the_unsupported_boundary_is_stated_where_it_can_be_linked() -> None:
    boundary = _section(CONFIGURATION, "Not Supported Yet")
    assert '<a id="named-agents-not-supported-yet">' in NAMED_AGENTS
    for claim in ("Cron", "Dream", "heartbeat", "Telegram is the only channel"):
        assert claim in boundary
    assert "talk to `default`" in boundary


@pytest.mark.parametrize("page", TOUCHED_PAGES, ids=lambda p: p.name)
def test_internal_links_resolve(page: Path) -> None:
    for target in _MD_LINK.findall(_read(page)):
        path_part, _, anchor = target.partition("#")
        destination = (page.parent / path_part).resolve() if path_part else page
        assert destination.exists(), f"docs/{page.name} -> {target}"
        if anchor:
            assert anchor in _anchors(destination), f"docs/{page.name} -> {target}"


@pytest.mark.parametrize("page", TOUCHED_PAGES, ids=lambda p: p.name)
def test_no_trailing_whitespace(page: Path) -> None:
    """What `git diff --check` would flag."""
    offenders = [
        number
        for number, line in enumerate(_read(page).splitlines(), start=1)
        if line != line.rstrip()
    ]
    assert offenders == []

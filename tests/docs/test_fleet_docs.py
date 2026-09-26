"""Pin the fleet documentation to the code it documents.

Prose drifts silently. Every assertion here derives its expectation from the
running code — the model's own field aliases, the Typer app's own subcommand
names, the state module's own published field tuple, the constants the policy
modules define — so a change to the fleet that the docs do not follow fails
here instead of misleading a reader.

The limits section gets the same treatment for the opposite reason. Nothing in
the code can tell you that ``docs/fleet.md`` warned about the home directory, so
those checks are the one place where the *absence* of a sentence is the defect:
a reader who over-trusts this boundary does so because a limit went unnamed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nanobot.cli.fleet import fleet_app
from nanobot.fleet.cap import BYTES_PER_MB, KILL_DEADLINE_SECONDS
from nanobot.fleet.config import (
    ENV_NAME_PATTERN,
    INSTANCE_NAME_PATTERN,
    FleetFile,
    FleetInstance,
    parse_fleet_file,
)
from nanobot.fleet.env import MINIMAL_ENV_NAMES
from nanobot.fleet.instance import INSTANCE_LOG_NAME, INSTANCE_LOG_SUBDIR
from nanobot.fleet.state import (
    RECORD_FIELDS,
    STATE_SUFFIX,
    fleet_state_path,
    write_fleet_state,
)
from nanobot.fleet.stop import DEFAULT_STOP_GRACE_SECONDS

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
FLEET_DOC = DOCS / "fleet.md"
CLI_REFERENCE = DOCS / "cli-reference.md"
MULTIPLE_INSTANCES = DOCS / "multiple-instances.md"
SECURITY = REPO_ROOT / "SECURITY.md"

#: A fenced ``json`` block, captured without its fence.
_JSON_BLOCK = re.compile(r"```json\n(.*?)```", re.DOTALL)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def section(text: str, heading: str) -> str:
    """The body of a ``## heading``, up to the next heading of the same level.

    Scoping assertions to a section is what stops a limit named only in a
    passing aside from satisfying a check meant for the limits section.
    """
    match = re.search(
        rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"no '## {heading}' section"
    return match.group(1)


def json_blocks(text: str) -> list[object]:
    return [json.loads(block) for block in _JSON_BLOCK.findall(text)]


# --------------------------------------------------------------------------
# docs/fleet.md: the fleet file
# --------------------------------------------------------------------------


def test_the_fleet_document_exists() -> None:
    assert FLEET_DOC.is_file()


@pytest.mark.parametrize("name", sorted(FleetInstance.model_fields))
def test_every_instance_field_is_documented_under_its_json_name(name: str) -> None:
    """A field added to the fleet document must be documented before it ships.

    Keyed on the alias rather than the Python name, because the alias is what an
    operator types: ``memoryLimitMb``, not ``memory_limit_mb``.
    """
    field = FleetInstance.model_fields[name]
    alias = field.alias or name
    assert f"`{alias}`" in read(FLEET_DOC), f"fleet.md never mentions `{alias}`"


@pytest.mark.parametrize("name", sorted(FleetFile.model_fields))
def test_every_root_field_is_documented(name: str) -> None:
    field = FleetFile.model_fields[name]
    assert f"`{field.alias or name}`" in read(FLEET_DOC)


def test_the_required_fields_are_documented_as_required() -> None:
    """The three confinement controls have no defaults, and the doc says so.

    Derived from the model so that giving one of them a default — which is the
    change this note exists to catch — makes the assertion stop holding.
    """
    required = {
        (FleetInstance.model_fields[name].alias or name)
        for name in FleetInstance.model_fields
        if FleetInstance.model_fields[name].is_required()
    }
    assert required == {"config", "mode", "memoryLimitMb"}
    no_default = re.search(
        r"There is \*\*no default\*\* for (.+?)\.", read(FLEET_DOC), re.DOTALL
    )
    assert no_default is not None
    for alias in required:
        assert f"`{alias}`" in no_default.group(1)


def test_every_documented_fleet_file_is_accepted_by_the_real_parser() -> None:
    """The examples are run through ``parse_fleet_file``, not merely eyeballed.

    A fleet file in the docs that the parser refuses is worse than no example:
    it is the first thing a reader copies. Blocks without an ``instances`` key
    are the status output and other payloads, and are not fleet documents.
    """
    documents = [
        block
        for block in json_blocks(read(FLEET_DOC))
        if isinstance(block, dict) and "instances" in block
    ]
    assert documents, "fleet.md shows no fleet file at all"
    for document in documents:
        parse_fleet_file(Path("docs/fleet.md"), json.dumps(document))


def test_every_documented_instance_entry_is_accepted_by_the_real_model() -> None:
    """The standalone entry examples are validated as ``FleetInstance``.

    Together with the test above and the status-array test below, this accounts
    for every JSON block on the page — which is what ``json_blocks`` parsing
    them all without raising already established. An example fragment that
    drifted out of the model's shape would fail here rather than be copied.
    """
    entries = [
        block
        for block in json_blocks(read(FLEET_DOC))
        if isinstance(block, dict) and "instances" not in block
    ]
    assert entries, "fleet.md shows no instance entry on its own"
    for entry in entries:
        FleetInstance.model_validate(entry)


def test_the_cli_reference_example_fleet_file_is_accepted_too() -> None:
    documents = [
        block
        for block in json_blocks(section(read(CLI_REFERENCE), "Fleet"))
        if isinstance(block, dict) and "instances" in block
    ]
    assert documents, "the CLI reference Fleet section shows no fleet file"
    for document in documents:
        parse_fleet_file(Path("docs/cli-reference.md"), json.dumps(document))


def test_the_documented_name_rules_are_the_patterns_the_parser_enforces() -> None:
    text = read(FLEET_DOC)
    assert f"`{INSTANCE_NAME_PATTERN}`" in text
    assert f"`{ENV_NAME_PATTERN}`" in text


def test_the_documented_base_environment_is_the_one_instances_receive() -> None:
    """Named exactly, and nothing else claimed: this is a credential boundary."""
    text = read(FLEET_DOC)
    for name in MINIMAL_ENV_NAMES:
        assert f"`{name}`" in text, f"fleet.md never names the base variable {name}"


# --------------------------------------------------------------------------
# docs/fleet.md: the commands
# --------------------------------------------------------------------------


def subcommand_names() -> list[str]:
    return sorted(command.name or "" for command in fleet_app.registered_commands)


def test_the_app_still_has_exactly_the_three_documented_subcommands() -> None:
    assert subcommand_names() == ["start", "status", "stop"]


@pytest.mark.parametrize("name", subcommand_names())
def test_every_subcommand_has_its_own_section_in_the_fleet_doc(name: str) -> None:
    assert f"### `nanobot fleet {name}`" in read(FLEET_DOC)


@pytest.mark.parametrize("name", subcommand_names())
def test_every_subcommand_appears_in_the_cli_reference_fleet_section(
    name: str,
) -> None:
    assert f"nanobot fleet {name}" in section(read(CLI_REFERENCE), "Fleet")


def test_the_documented_stop_grace_default_is_the_real_one() -> None:
    for path in (FLEET_DOC, CLI_REFERENCE):
        assert f"`{DEFAULT_STOP_GRACE_SECONDS}`" in read(path), path


def test_the_documented_kill_deadline_is_the_real_one() -> None:
    assert f"within {KILL_DEADLINE_SECONDS:.0f} seconds" in read(FLEET_DOC)


def test_a_megabyte_is_documented_as_binary() -> None:
    """The cap is read the way ``ulimit`` and Activity Monitor read one.

    Asserted against a hard literal as well as the constant, so that a decimal
    conversion in either the code or the prose fails rather than agreeing with
    itself.
    """
    assert BYTES_PER_MB == 1_048_576
    assert f"1 MB = {BYTES_PER_MB} bytes" in read(FLEET_DOC)


def test_the_documented_state_file_suffix_and_log_path_are_the_real_ones() -> None:
    text = read(FLEET_DOC)
    assert STATE_SUFFIX in text
    assert f"{INSTANCE_LOG_SUBDIR}/{INSTANCE_LOG_NAME}" in text


# --------------------------------------------------------------------------
# docs/fleet.md: the --json contract
# --------------------------------------------------------------------------


def test_the_documented_status_object_carries_exactly_the_published_fields() -> None:
    """The example is compared against ``RECORD_FIELDS``, key for key.

    Both directions matter: a field dropped from the docs leaves a consumer
    unaware of it, and a field invented by the docs is a promise the command
    does not keep.
    """
    arrays = [
        block
        for block in json_blocks(read(FLEET_DOC))
        if isinstance(block, list) and block and isinstance(block[0], dict)
    ]
    assert len(arrays) == 1, "expected exactly one --json example"
    assert sorted(arrays[0][0]) == sorted(RECORD_FIELDS)


@pytest.mark.parametrize("field", RECORD_FIELDS)
def test_every_published_status_field_has_a_row_of_its_own(field: str) -> None:
    assert f"| `{field}` |" in read(FLEET_DOC)


def test_a_missing_state_file_is_documented_as_a_refusal_not_an_empty_report(
    tmp_path: Path,
) -> None:
    """Driven against the real command, because the prose got this wrong first.

    ``status`` on a fleet that was never started exits non-zero; the *empty*
    report is a different case, reached only when the file exists and lists
    nothing. Documenting the friendlier one for both would send an operator
    looking for a crashed supervisor when they had simply never started it.
    """
    runner = CliRunner()
    fleet_file = tmp_path / "fleet.json"
    fleet_file.write_text("{}", encoding="utf-8")

    missing = runner.invoke(fleet_app, ["status", "--fleet", str(fleet_file)])
    assert missing.exit_code != 0
    assert "Invalid fleet state file" in missing.output

    write_fleet_state((), path=fleet_state_path(fleet_file.resolve()))
    empty = runner.invoke(fleet_app, ["status", "--fleet", str(fleet_file)])
    assert empty.exit_code == 0
    assert "No instance has been started for this fleet." in empty.output

    troubleshooting = section(read(FLEET_DOC), "Troubleshooting")
    assert "Invalid fleet state file" in troubleshooting
    assert "No instance has been started for this fleet." in troubleshooting


def test_the_identity_token_is_documented_as_deliberately_unpublished() -> None:
    """Its absence is the contract, so the doc has to say that it is."""
    assert "not published" in read(FLEET_DOC)


def test_the_instance_pid_is_documented_as_never_the_supervisors() -> None:
    assert "Never the supervisor's." in read(FLEET_DOC)


# --------------------------------------------------------------------------
# docs/fleet.md: the limits
# --------------------------------------------------------------------------

#: Every limit the bead requires be named, with a phrase that can only have been
#: written on purpose. Matched inside the limits section alone, so a mention
#: elsewhere in the page cannot stand in for a warning.
REQUIRED_LIMITS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("home directory", ("rest of your home directory is not denied", "~/.ssh")),
    ("network", ("No network isolation",)),
    ("restart", ("No automatic restart",)),
    ("other resource limits", ("No CPU, file-descriptor, disk, or process-count",)),
    ("named agents", ("No per-named-agent isolation", "subagent")),
    ("platform", ("macOS only",)),
)


def test_the_fleet_doc_has_a_limits_section() -> None:
    assert "## Limits" in read(FLEET_DOC)


@pytest.mark.parametrize(
    ("limit", "phrases"),
    REQUIRED_LIMITS,
    ids=[limit for limit, _ in REQUIRED_LIMITS],
)
def test_the_limits_section_names_every_required_limit(
    limit: str,
    phrases: tuple[str, ...],
) -> None:
    body = section(read(FLEET_DOC), "Limits")
    for phrase in phrases:
        assert phrase in body, f"the limits section does not name {limit}: {phrase!r}"


def test_the_limits_are_prominent_rather_than_buried() -> None:
    """Linked from the top of the page, and before the guarantees are described.

    A limits section a reader reaches only after deciding to trust the boundary
    is a section that arrives too late.
    """
    text = read(FLEET_DOC)
    intro = text[: text.index("## Quick Start")]
    assert "#limits" in intro


def test_the_limits_section_disclaims_the_sandbox_reading_outright() -> None:
    body = section(read(FLEET_DOC), "Limits")
    assert "not a sandbox" in body
    assert "not a VM" in body


# --------------------------------------------------------------------------
# The three pointers into the fleet doc
# --------------------------------------------------------------------------


def test_the_cli_reference_gains_a_fleet_section_after_gateway() -> None:
    text = read(CLI_REFERENCE)
    assert "\n## Fleet\n" in text
    assert text.index("\n## Gateway\n") < text.index("\n## Fleet\n")


def test_the_cli_reference_fleet_section_points_at_the_fleet_doc() -> None:
    assert "./fleet.md" in section(read(CLI_REFERENCE), "Fleet")


def test_multiple_instances_says_its_separation_is_convention_only() -> None:
    """The document whose implied promise motivated the feature.

    Its own separation is not enforced, and a reader who leaves that page
    believing otherwise is the reason the fleet exists.
    """
    text = read(MULTIPLE_INSTANCES)
    assert "convention only" in text
    assert "./fleet.md" in text


def test_the_convention_only_pointer_is_at_the_top_of_multiple_instances() -> None:
    text = read(MULTIPLE_INSTANCES)
    assert text.index("convention only") < text.index("## Quick Start")


def test_security_covers_the_fleet_boundary_beside_the_sandbox_one() -> None:
    """Placed with the existing filesystem-containment statement, not appended.

    The two boundaries are easy to confuse — one confines a shell command inside
    an instance, the other confines an instance against its peers — so the fleet
    text has to sit where a reader comparing them will find it.
    """
    text = read(SECURITY)
    containment = text.index(
        "This is filesystem containment, not a VM or separate user identity."
    )
    fleet = text.index("**Fleet confinement (macOS only):**")
    shell_section = text.index("### 3. Shell Command Execution")
    file_section = text.index("### 4. File System Access")
    assert containment < fleet < file_section
    assert shell_section < fleet


@pytest.mark.parametrize(
    ("limit", "phrases"),
    REQUIRED_LIMITS,
    ids=[limit for limit, _ in REQUIRED_LIMITS],
)
def test_security_repeats_every_limit_rather_than_only_linking_to_them(
    limit: str,
    phrases: tuple[str, ...],
) -> None:
    """A reader auditing SECURITY.md must not have to follow a link to learn these."""
    text = read(SECURITY)
    assert any(phrase in text for phrase in phrases), f"SECURITY.md omits {limit}"


def test_security_points_at_the_fleet_doc() -> None:
    assert "docs/fleet.md" in read(SECURITY)


# --------------------------------------------------------------------------
# Links
# --------------------------------------------------------------------------


def test_every_relative_link_in_the_fleet_doc_resolves() -> None:
    """Including the anchors it uses into its own limits section."""
    text = read(FLEET_DOC)
    for target in re.findall(r"\]\((\.[^)]+)\)", text):
        path, _, anchor = target.partition("#")
        resolved = (FLEET_DOC.parent / path).resolve()
        assert resolved.is_file(), f"broken link: {target}"
        if anchor and resolved == FLEET_DOC:
            assert f"## {anchor.replace('-', ' ').title()}" in text


def test_the_docs_index_lists_the_fleet_page() -> None:
    assert "./fleet.md" in read(DOCS / "README.md")

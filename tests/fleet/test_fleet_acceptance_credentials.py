"""Acceptance criterion 5: credential isolation, against two real confined instances.

The supervisor's environment is the union of every credential on the machine.
The fleet's promise is that an instance sees the four base variables plus the
variables *its own* entry names, and nothing else — not a peer's, and not one the
operator exported but declared for nobody. :mod:`nanobot.fleet.env` is unit-tested
against an injected mapping; this is the criterion written out literally instead,
with three real variables, two real instances under the real Seatbelt policy, and
the answer read out of an instance's *own shell tool* rather than out of a mapping
the test built.

The criterion is amended from the spec, and the amendment makes it stronger.
nanobot's shell tool does not hand a command the instance's environment: it builds
a minimal one of its own (``agent/tools/shell.py``, ``_build_env``) holding
``HOME``, ``LANG``, ``TERM`` and ``PYTHONUNBUFFERED`` plus whatever
``tools.exec.allowedEnvKeys`` lists. Run as literally specified, the criterion
would therefore pass on a fleet with no isolation at all: ``SECRET_B`` would be
absent from A's ``env`` output because the shell tool dropped it, not because the
supervisor withheld it. So **both** instance configs name all three variables in
``allowedEnvKeys``. The shell tool is then willing to forward every one of them,
and the only thing left that can explain what A sees is what the supervisor gave
A's process in the first place.

That still leaves the vacuity one level down — an allowlist that silently did
nothing would look identical from A's side alone. The companion assertion closes
it from B's: B's shell tool, with the *same* allowlist and the *same* four names,
does forward ``FLEET_SECRET_B``. A name that crosses in B cannot be a name the
shell tool strips in A, so A's blindness to it is the supervisor's partition and
nothing else. ``FLEET_SENTINEL`` — exported by the supervisor, declared by
neither instance, allowlisted by both — is the third leg: it is invisible to both,
which is the rule that a peer's list is not the only thing an instance is denied.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import threading
import time
from contextlib import suppress
from pathlib import Path

import httpx
import pytest
from stub_llm_server import StubCompletion, StubLLMServer, free_port

from nanobot.fleet.instance import SANDBOX_EXEC
from nanobot.fleet.supervisor import FleetSupervisor, prepare_fleet, start_fleet
from nanobot.fleet.validate import validate_fleet_file

confinement_available = pytest.mark.skipif(
    not Path(SANDBOX_EXEC).is_file(),
    reason="a fleet cannot be started without native Seatbelt",
)

#: The three variables the criterion is about. ``FLEET_SECRET_A`` is declared by
#: alpha only, ``FLEET_SECRET_B`` by beta only, and ``FLEET_SENTINEL`` by neither
#: — but all three are exported by the supervisor and allowlisted by both
#: instances' shell tools, so every absence below is a decision the fleet made.
SECRET_A = "FLEET_SECRET_A"
SECRET_B = "FLEET_SECRET_B"
SENTINEL = "FLEET_SENTINEL"
ALLOWED_ENV_KEYS = (SECRET_A, SECRET_B, SENTINEL)

#: Distinct, searchable values, so the test can assert on the *value* as well as
#: the name: a leak that renamed the variable would still carry the secret.
VALUES = {
    SECRET_A: "alpha-only-credential-4f1c9d",
    SECRET_B: "beta-only-credential-9a7302",
    SENTINEL: "declared-by-nobody-e20d5b",
}

#: An instance's own shell tool is the reporter, so ``env`` is run with an
#: absolute path: the tool's minimal environment has no ``PATH`` on Unix, and a
#: probe that depended on bash's fallback lookup would be testing bash.
ENV_TOOL = "/usr/bin/env"

CAP_MB = 512
STARTUP_TIMEOUT_SECONDS = 90.0
TURN_TIMEOUT_SECONDS = 120.0

#: What the stub answers for every turn that is not the scripted tool call —
#: including the turn that carries the tool's output back. The script queue is
#: shared by the whole fleet, so anything left unscripted must be answerable.
FALLBACK_ANSWER = "reported"

_ENV_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def write_instance(root: Path, name: str, stub: StubLLMServer) -> Path:
    """Lay out one instance's config at ``<root>/<name>/config.json``.

    ``allowedEnvKeys`` names all three variables for *both* instances: that is
    the amendment this test turns on, and making it identical on both sides is
    what leaves the supervisor as the only asymmetry between them.
    """
    config_dir = root / name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps(stub.instance_config(
            config_dir / "workspace",
            api_port=free_port(),
            allowed_env_keys=ALLOWED_ENV_KEYS,
        )),
        encoding="utf-8",
    )
    return config_path


def api_port_of(config_path: Path) -> int:
    """The port the instance will bind, read back from its own config."""
    return int(json.loads(config_path.read_text(encoding="utf-8"))["api"]["port"])


def allowed_env_keys_of(config_path: Path) -> list[str]:
    """The allowlist the instance will actually run with, read back from disk."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return list(data["tools"]["exec"]["allowedEnvKeys"])


def log_tail(path: Path, limit: int = 4000) -> str:
    """An instance's own log, which is the only diagnostic a confined process leaves."""
    if not path.exists():
        return "(no log)"
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def wait_for_health(port: int, name: str, log_path: Path) -> None:
    """Block until the instance answers on its API, or fail with its log.

    There is no ``pytest-timeout`` in this repo, so every wait carries its own
    deadline — the pattern ``tests/webui/test_gateway_webui_smoke.py`` sets.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        with suppress(httpx.HTTPError, OSError):
            response = httpx.get(
                f"http://127.0.0.1:{port}/health", timeout=5.0, trust_env=False
            )
            if response.status_code == 200:
                return
        time.sleep(0.2)
    pytest.fail(f"instance {name} never became healthy\n{log_tail(log_path)}")


def parse_env_dump(text: str) -> dict[str, str]:
    """Parse ``env`` output into a mapping.

    A continuation line — part of a multi-line value — has no ``NAME=`` prefix
    and is appended to the variable it belongs to rather than dropped, so a
    secret smuggled into the tail of another variable's value still shows up in
    this mapping's values. The raw text is asserted on separately as well.
    """
    env: dict[str, str] = {}
    current: str | None = None
    for line in text.splitlines():
        match = _ENV_LINE.match(line)
        if match is None:
            if current is not None:
                env[current] += f"\n{line}"
            continue
        current = match.group(1)
        env[current] = match.group(2)
    return env


def run_env_tool(
    stub: StubLLMServer,
    *,
    port: int,
    destination: Path,
    log_path: Path,
    name: str,
) -> tuple[dict[str, str], str]:
    """Make one instance's shell tool run ``env`` and return what it saw.

    Drives the instance through its own OpenAI-compatible API: one scripted
    assistant turn calling ``exec``, which the *instance* executes inside its own
    Seatbelt policy with its own environment. The output is written into the
    instance's workspace — a place only that instance and the supervisor can
    reach — rather than returned through the model, so what is asserted on is the
    raw bytes the command produced.

    Instances are driven one at a time because the stub's script queue is shared
    by the whole fleet; a peer turn arriving mid-flight would otherwise consume
    the scripted tool call meant for this one.
    """
    stub.script_tool_call("exec", {
        "command": f"{ENV_TOOL} > {shlex.quote(str(destination))}",
    })
    response = httpx.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "run env and save it"}]},
        timeout=TURN_TIMEOUT_SECONDS,
        trust_env=False,
    )
    assert response.status_code == 200, f"{name}: {response.text}\n{log_tail(log_path)}"
    if not destination.exists():
        pytest.fail(
            f"{name}'s shell tool never produced {destination}\n{log_tail(log_path)}"
        )
    text = destination.read_text(encoding="utf-8", errors="replace")
    return parse_env_dump(text), text


def stop_fleet(supervisor: FleetSupervisor, loop: threading.Thread) -> None:
    """Ask the supervisor to stop and wait for its loop to leave."""
    supervisor.request_stop()
    loop.join(timeout=STARTUP_TIMEOUT_SECONDS)


@confinement_available
def test_an_instance_sees_only_the_credentials_its_own_entry_declared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_llm_server: StubLLMServer,
) -> None:
    pytest.importorskip("aiohttp")

    # The supervisor holds all three. ``launch_instance`` reads ``os.environ`` at
    # spawn time, so exporting them here is exporting them into the fleet's
    # parent — exactly the situation the criterion describes.
    for name, value in VALUES.items():
        monkeypatch.setenv(name, value)

    root = tmp_path.resolve()
    alpha_config = write_instance(root, "alpha", stub_llm_server)
    beta_config = write_instance(root, "beta", stub_llm_server)
    fleet_path = root / "fleet.json"
    fleet_path.write_text(
        json.dumps({
            "instances": {
                "alpha": {
                    "config": str(alpha_config),
                    "mode": "serve",
                    "memoryLimitMb": CAP_MB,
                    "env": [SECRET_A],
                },
                "beta": {
                    "config": str(beta_config),
                    "mode": "serve",
                    "memoryLimitMb": CAP_MB,
                    "env": [SECRET_B],
                },
            }
        }),
        encoding="utf-8",
    )

    # The premise, checked rather than assumed: both instances allow all three
    # names through their shell tool, and only the fleet document differs.
    assert allowed_env_keys_of(alpha_config) == list(ALLOWED_ENV_KEYS)
    assert allowed_env_keys_of(beta_config) == list(ALLOWED_ENV_KEYS)
    assert all(os.environ.get(name) == value for name, value in VALUES.items())

    instances = {one.name: one for one in validate_fleet_file(fleet_path)}
    alpha, beta = instances["alpha"], instances["beta"]
    assert (list(alpha.entry.env), list(beta.entry.env)) == ([SECRET_A], [SECRET_B])

    plan = prepare_fleet(list(instances.values()), fleet_path=fleet_path)

    # The production spawn path: real argv, real Seatbelt wrapper, real ``env -i``.
    supervisor = start_fleet(plan)
    alpha_log = supervisor.launched("alpha").log_path
    beta_log = supervisor.launched("beta").log_path
    loop = threading.Thread(
        target=supervisor.run,
        kwargs={"handle_signals": False},
        name="fleet-supervisor",
        daemon=True,
    )
    loop.start()
    try:
        wait_for_health(api_port_of(alpha_config), "alpha", alpha_log)
        wait_for_health(api_port_of(beta_config), "beta", beta_log)

        stub_llm_server.set_fallback(StubCompletion(content=FALLBACK_ANSWER))
        alpha_env, alpha_text = run_env_tool(
            stub_llm_server,
            port=api_port_of(alpha_config),
            destination=alpha.workspace / "env.txt",
            log_path=alpha_log,
            name="alpha",
        )
        beta_env, beta_text = run_env_tool(
            stub_llm_server,
            port=api_port_of(beta_config),
            destination=beta.workspace / "env.txt",
            log_path=beta_log,
            name="beta",
        )
    finally:
        stop_fleet(supervisor, loop)

    assert not loop.is_alive()

    # The dump is a real environment, so an empty or truncated file cannot be
    # mistaken for a clean partition. ``_build_env`` sets these four
    # unconditionally on Unix.
    for shell_base in ("HOME", "LANG", "TERM", "PYTHONUNBUFFERED"):
        assert shell_base in alpha_env, alpha_text
        assert shell_base in beta_env, beta_text

    # The criterion itself.
    assert alpha_env.get(SECRET_A) == VALUES[SECRET_A], alpha_text
    assert SECRET_B not in alpha_env, alpha_text
    assert SENTINEL not in alpha_env, alpha_text

    # The companion. Same allowlist, same shell tool, same four names: a variable
    # the tool forwards for beta is not a variable the tool strips for alpha, so
    # alpha's blindness to it can only be the supervisor's partition.
    assert beta_env.get(SECRET_B) == VALUES[SECRET_B], beta_text
    assert SECRET_A not in beta_env, beta_text
    assert SENTINEL not in beta_env, beta_text

    # Names can be renamed; values cannot be un-leaked. Asserted against the raw
    # bytes so a secret hidden in some other variable's value is caught too.
    assert VALUES[SECRET_B] not in alpha_text
    assert VALUES[SENTINEL] not in alpha_text
    assert VALUES[SECRET_A] not in beta_text
    assert VALUES[SENTINEL] not in beta_text

    # Stated once as a set, so a fourth criterion variable added later cannot
    # slip across unasserted: of the three allowlisted names, exactly the one
    # this instance declared is present.
    assert {name for name in ALLOWED_ENV_KEYS if name in alpha_env} == {SECRET_A}
    assert {name for name in ALLOWED_ENV_KEYS if name in beta_env} == {SECRET_B}


def test_the_env_dump_parser_keeps_a_multi_line_value_with_its_variable() -> None:
    """A continuation line belongs to the variable above it, not to nothing.

    Pinned separately because the whole point of parsing this way is that a
    secret appended to another variable's value still lands in the mapping the
    absence assertions run against; a parser that dropped unprefixed lines would
    make those assertions weaker without failing anything.
    """
    parsed = parse_env_dump("FIRST=one\nWRAPPED=head\ntail\nLAST=three\n")

    assert parsed == {"FIRST": "one", "WRAPPED": "head\ntail", "LAST": "three"}


def test_the_env_dump_parser_ignores_a_leading_continuation() -> None:
    """Output that starts mid-value has nothing to attach to and is dropped."""
    assert parse_env_dump("orphan\nNAME=value\n") == {"NAME": "value"}


def test_the_three_criterion_variables_are_distinct() -> None:
    """Names and values are all distinct, so no assertion above is trivially true.

    If two of the three shared a value, a "this secret did not leak" assertion
    could be satisfied by the wrong variable being present; if two shared a name,
    the partition under test would not exist. Both are cheap to pin and neither
    would fail loudly on its own.
    """
    assert len({SECRET_A, SECRET_B, SENTINEL}) == 3
    assert len(set(VALUES.values())) == 3
    assert set(VALUES) == set(ALLOWED_ENV_KEYS) == {SECRET_A, SECRET_B, SENTINEL}

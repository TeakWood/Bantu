"""Acceptance criterion 6: the memory cap, against two real confined instances.

Everything the fleet's cap is made of has been tested in isolation — the sampler
against known struct offsets, the threshold against injected numbers, the loop
against stub children on a fake clock. This is the criterion written out
literally instead: two nanobot instances are started under the real Seatbelt
policy by the real supervisor, one of them is driven *through its own shell tool*
to start a process that allocates past its declared cap, and the fleet must do
three things about it — kill that instance's whole tree inside the deadline,
report it as having died of memory, and leave its peer serving.

Two things about the shape are load-bearing rather than incidental.

*The allocation is in a descendant, and the descendant is not in the instance's
process group.* nanobot's shell tool starts every command with
``start_new_session`` so that it can kill a runaway command by group
(``agent/tools/shell.py``), which takes that command out of the instance's group
before it allocates its first byte. So this test is the one that says whether the
fleet measures and kills the *tree* or merely the group; a group-only fleet
passes every unit test written for it and reads a runaway instance as idle.

*The deadline is measured from before the crossing, not after it.* The allocator
records a timestamp and its pid before it allocates anything, so the cap is
crossed at some instant strictly after that mark. An elapsed time measured from
the mark is therefore at least the true time from the crossing, and an assertion
that it is under five seconds is a conservative statement of the criterion rather
than a flattering one. The alternative — timing from when the test *noticed* the
allocation — would measure from after the crossing and could pass a fleet that
missed the deadline.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import httpx
import pytest
from stub_llm_server import StubCompletion, StubLLMServer, free_port

from nanobot.fleet.cap import BYTES_PER_MB, KILL_DEADLINE_SECONDS
from nanobot.fleet.instance import SANDBOX_EXEC
from nanobot.fleet.memory import process_tree_memory_bytes
from nanobot.fleet.state import InstanceRecord, read_fleet_state
from nanobot.fleet.supervisor import prepare_fleet, start_fleet
from nanobot.fleet.validate import validate_fleet_file
from nanobot.process_runtime import process_is_running

confinement_available = pytest.mark.skipif(
    sys.platform != "darwin" or not Path(SANDBOX_EXEC).is_file(),
    reason="a fleet cannot be started without native Seatbelt",
)

#: The caps declared for the two instances, and what the descendant allocates.
#: A ``serve`` instance settles at roughly 120 MB resident, so the cap sits well
#: clear of an instance doing nothing and the allocation clears the cap several
#: times over. Both margins are asserted before the allocation is triggered, so a
#: baseline that drifts fails saying so rather than by a confusing timeout.
CAP_MB = 384
PEER_CAP_MB = 512
ALLOCATE_MB = 512

STARTUP_TIMEOUT_SECONDS = 90.0
#: How long the test is willing to wait past the deadline before giving up. Well
#: over ``KILL_DEADLINE_SECONDS`` on purpose: a fleet that kills the tree in eight
#: seconds has broken the criterion and should say so with a number, not stall.
OBSERVATION_TIMEOUT_SECONDS = 45.0

PEER_ANSWER = "beta is still serving"

# Run by the instance's shell tool, as a child of the instance. Writes its mark
# atomically and *before* it allocates anything: the marker is what the deadline
# is measured from, and a mark written afterwards would measure from after the
# cap was already crossed. ``bytearray`` is zero-filled by CPython, but the pages
# are touched explicitly so the resident size does not depend on that staying
# true.
ALLOCATOR = """
import json
import os
import sys
import time

marker, megabytes = sys.argv[1], int(sys.argv[2])
pending = marker + ".partial"
with open(pending, "w") as handle:
    json.dump({"pid": os.getpid(), "started": time.time()}, handle)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(pending, marker)

block = bytearray(megabytes * 1024 * 1024)
for offset in range(0, len(block), 4096):
    block[offset] = 1
sys.stdout.write("allocated\\n")
sys.stdout.flush()
time.sleep(600)
"""


def write_instance(root: Path, name: str, stub: StubLLMServer) -> Path:
    """Lay out one instance's config the way nanobot lays one out by itself.

    ``<root>/<name>/config.json`` with ``<root>/<name>/workspace`` beneath it.
    The workspace is not created here: creating it belongs to ``prepare_fleet``.
    """
    config_dir = root / name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps(stub.instance_config(config_dir / "workspace", api_port=free_port())),
        encoding="utf-8",
    )
    return config_path


def api_port_of(config_path: Path) -> int:
    """The port the instance will bind, read back from its own config."""
    return int(json.loads(config_path.read_text(encoding="utf-8"))["api"]["port"])


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


def observe(predicate: Callable[[], bool], what: str, *, log_path: Path) -> float:
    """Poll until ``predicate`` holds and return the wall-clock time it did.

    Wall clock rather than monotonic because the other end of the measurement is
    a timestamp taken inside another process; both read the same machine clock.
    """
    deadline = time.monotonic() + OBSERVATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return time.time()
        time.sleep(0.02)
    pytest.fail(f"{what} did not happen within {OBSERVATION_TIMEOUT_SECONDS}s\n{log_tail(log_path)}")


def status_of(state_path: Path) -> dict[str, InstanceRecord]:
    """What ``nanobot fleet status`` would report right now, by instance name.

    Read through :func:`read_fleet_state`, which is the reader an operator's
    second shell uses: it reconciles each record's liveness and never writes.
    """
    return {record.name: record for record in read_fleet_state(state_path)}


@confinement_available
def test_a_descendants_allocation_kills_its_instance_and_spares_its_peer(
    tmp_path: Path,
    stub_llm_server: StubLLMServer,
) -> None:
    pytest.importorskip("aiohttp")

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
                },
                "beta": {
                    "config": str(beta_config),
                    "mode": "serve",
                    "memoryLimitMb": PEER_CAP_MB,
                },
            }
        }),
        encoding="utf-8",
    )

    instances = {one.name: one for one in validate_fleet_file(fleet_path)}
    alpha, beta = instances["alpha"], instances["beta"]
    plan = prepare_fleet(list(instances.values()), fleet_path=fleet_path)

    allocator = alpha.workspace / "allocate.py"
    allocator.write_text(ALLOCATOR, encoding="utf-8")
    marker = alpha.workspace / "allocating.json"

    # The production spawn path: real argv, real Seatbelt wrapper, real session.
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
    allocator_pid = 0
    try:
        wait_for_health(api_port_of(alpha_config), "alpha", alpha_log)
        wait_for_health(api_port_of(beta_config), "beta", beta_log)

        # Both instances are nowhere near their caps before anything allocates,
        # so a kill below can only be about the allocation.
        at_rest = process_tree_memory_bytes(supervisor.launched("alpha").pgid)
        assert at_rest is not None and at_rest < CAP_MB * BYTES_PER_MB, at_rest
        assert supervisor.breaches == ()
        assert status_of(plan.state_path)["alpha"].state == "running"

        # One scripted tool call is the whole trigger: the model tells alpha to
        # run a command, and alpha's own shell tool starts the process that
        # outgrows the cap. Everything else the stub is asked answers with the
        # fallback, so beta's turn cannot race alpha's for a queue entry.
        stub_llm_server.set_fallback(StubCompletion(content=PEER_ANSWER))
        stub_llm_server.script_tool_call("exec", {
            "command": " ".join(shlex.quote(word) for word in (
                sys.executable, str(allocator), str(marker), str(ALLOCATE_MB)
            )),
        })

        def drive_alpha() -> None:
            # Never returns normally: alpha is killed mid-tool-call.
            with suppress(httpx.HTTPError, OSError):
                httpx.post(
                    f"http://127.0.0.1:{api_port_of(alpha_config)}/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "run the command"}]},
                    timeout=OBSERVATION_TIMEOUT_SECONDS,
                    trust_env=False,
                )

        threading.Thread(target=drive_alpha, name="drive-alpha", daemon=True).start()

        observe(marker.exists, "alpha's shell tool started the allocation", log_path=alpha_log)
        announced = json.loads(marker.read_text(encoding="utf-8"))
        allocator_pid, started = int(announced["pid"]), float(announced["started"])

        # The premise of the criterion, checked rather than assumed: the process
        # that allocates is a descendant of alpha and is not in alpha's group.
        assert allocator_pid != supervisor.launched("alpha").pid
        assert os.getpgid(allocator_pid) != supervisor.launched("alpha").pgid

        # The deadline is about the processes, so it is measured on the
        # processes. Both halves of the tree have to be gone: killing only the
        # instance would leave the allocation running, reparented and
        # attributable to nobody, which is the failure a group-only kill has.
        alpha_dead_at = observe(
            lambda: not process_is_running(supervisor.launched("alpha").pid),
            "alpha was killed",
            log_path=alpha_log,
        )
        tree_dead_at = observe(
            lambda: not process_is_running(allocator_pid),
            "alpha's descendant was killed",
            log_path=alpha_log,
        )

        # Measured from before the crossing, so these over-state the true elapsed
        # time rather than flattering it.
        assert alpha_dead_at - started <= KILL_DEADLINE_SECONDS, alpha_dead_at - started
        assert tree_dead_at - started <= KILL_DEADLINE_SECONDS, tree_dead_at - started

        # Reported separately, and waited for separately. A reader reconciles
        # liveness for itself, so alpha reads as exited from the instant it dies;
        # what has to arrive is the supervisor's own account of *why*, which only
        # the component that sent the signal can give.
        observe(
            lambda: status_of(plan.state_path)["alpha"].exit_reason is not None,
            "the supervisor published a reason for alpha's exit",
            log_path=alpha_log,
        )
        status = status_of(plan.state_path)
        assert (status["alpha"].state, status["alpha"].exit_reason) == ("exited", "memory")
        assert status["beta"].state == "running"

        # The reading that ended alpha, kept so an operator can tell whether the
        # cap or the workload was wrong.
        assert [one.name for one in supervisor.breaches] == ["alpha"]
        breach = supervisor.breaches[0]
        assert breach.resident_bytes > breach.limit_bytes == CAP_MB * BYTES_PER_MB

        # Serving, not merely holding a pid: beta answers a real request through
        # its own API, with its own turn against the stub.
        answer = httpx.post(
            f"http://127.0.0.1:{api_port_of(beta_config)}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "are you still there"}]},
            timeout=OBSERVATION_TIMEOUT_SECONDS,
            trust_env=False,
        )
        assert answer.status_code == 200, answer.text
        assert PEER_ANSWER in answer.json()["choices"][0]["message"]["content"]
        assert beta.workspace.exists()
    finally:
        supervisor.request_stop()
        loop.join(timeout=STARTUP_TIMEOUT_SECONDS)
        if allocator_pid:
            with suppress(ProcessLookupError, PermissionError):
                os.kill(allocator_pid, signal.SIGKILL)

    assert not loop.is_alive()
    assert [one.state for one in supervisor.records] == ["exited", "exited"]

---
title: Process isolation
status: accepted
date: 2026-09-23
superseded-by:
---

# Process isolation

Epic: `Bantu-beads-m7w` (23 children, `Bantu-beads-m7w.1` … `.23`).

## Context

`docs/multiple-instances.md` already tells operators how to run several nanobot
instances side by side, each with its own config file and workspace. Its own
table — config, workspace, sessions, cron, media, all derived from `--config` —
describes a naming convention. Nothing enforces it.

Every instance runs as the same OS user, can read and write every other
instance's workspace, config file and session store, inherits the whole
environment it was started from including secrets, and can exhaust the machine's
memory and take the others down with it.

nanobot's own guards do not close this. `tools.restrictToWorkspace`
(`nanobot/config/schema.py:401`) and the shell tool's sandbox are checks inside
the agent's own code — inside the blast radius of the thing that goes wrong. The
repo says so itself, at `docs/configuration.md:2123`: *"but this is not an OS
sandbox"*.

Three instances make the asymmetry concrete. A **research** instance reads
arbitrary web pages all day, which makes it the instance most exposed to prompt
injection. A **trading** instance holds brokerage credentials. A **health**
instance holds medical records. The exposed instance is not the valuable
instance, and today a compromise of the first reaches the second.

The requirement is that a compromised or runaway research instance can damage
nothing but itself, and that this holds **even when the instance's own guards are
switched off** (`restrictToWorkspace: false`, shell sandbox disabled). An
isolation property that evaporates when a config flag is flipped is not
isolation.

This decision adds a supervisor that runs a **fleet** — a set of named instances
declared in one JSON fleet file — with each instance in its own OS process,
confined by the operating system.

## Constraints that shaped the design

Four repo facts did more to determine the shape than the feature description did.

### 1. The existing Seatbelt profile has the opposite posture

nanobot already uses Seatbelt. `_seatbelt` (`nanobot/agent/tools/sandbox.py:184`)
builds a `(version 1)` + `(deny default)` allowlist for individual shell
commands. Three properties make it unusable here:

- It **denies the workspace's parent directory** (`sandbox.py:243-245`). In the
  fleet layout the workspace's parent *is the instance's own config directory* —
  where its config file and session store live. A gateway confined by this
  profile could not read its own config.
- It pins `HOME` and `TMPDIR` to the workspace and drops the rest of the
  environment (`sandbox.py:292-306`). An instance needs its provider credentials.
- It returns a shell string ending in `sh -c "cd … || exit\n<command>"`, shaped
  for a short command list rather than a service entrypoint.

So this feature builds a **new** profile module and leaves `sandbox.py`
untouched. That file also carries the native shell-sandbox test suite
(`tests/tools/test_seatbelt_native.py`); destabilising it to serve a different
feature would risk an unrelated security boundary.

### 2. Seatbelt fails open, silently — this is the central design constraint

Measured on the pinned macOS host during design:

| Profile content | Outcome |
| --- | --- |
| `(deny …(subpath "/private/tmp/…/b"))` — canonical | read, write and `listdir` all refused |
| `(deny …(subpath "/tmp/…/b"))` — via symlink to the same directory | **file readable, exit 0, no warning** |
| deny naming a path that does not exist | process starts, no warning |
| deny with an invalid *relative* `subpath` | process starts, no warning |

`sandbox-exec` never reports a bad rule. A profile that confines nothing is
indistinguishable, from the outside, from one that works — the instance starts,
the CLI reports success, and every acceptance check that only looks at process
state still passes.

This matters because `Config.workspace_path`
(`nanobot/config/schema.py:490-493`) calls only `.expanduser()`, never
`.resolve()`. On macOS `/tmp` is a symlink to `/private/tmp`. A fleet file with
one unresolved symlink anywhere in a path therefore produces total, silent loss
of isolation.

Two consequences run through the whole decomposition:

- The profile builder **raises** on any non-absolute or non-canonical input
  rather than emitting it (`Bantu-beads-m7w.4`).
- `fleet start` **proves** the confinement with a live probe before reporting an
  instance started, and exits non-zero if the deny does not bind
  (`Bantu-beads-m7w.5`, `.12`).

The opposite good news was also measured: confinement **is** inherited across
`exec` and process spawns. Through `sandbox-exec → sh → sh → python → cat`, reads,
writes and directory listings of a denied peer path were all refused. That is
what makes it valid to test the boundary by driving an instance's shell tool,
whose work happens in descendants.

### 3. `.agent/design.md` is normative

> Core stays small; extend at the edges. […] The files `agent/loop.py` and
> `agent/runner.py` form the critical core path.

and

> Prefer duplication over premature abstraction.

The first rules out threading fleet concepts through the agent loop; nothing
here needs to. The second decides the memory sampler: `nanobot/process_runtime.py`
already loads `libproc.dylib` via `ctypes` (`:678-720`) to read process *birth*
times, but that helper is private and `process_runtime.py` is a chokepoint
imported by the gateway runtime. Duplicating a small `ctypes` handle locally is
the sanctioned choice over widening a shared file.

### 4. The shell tool already strips the environment

`ExecTool._build_env` (`nanobot/agent/tools/shell.py:793-805`) forwards, on Unix,
only `HOME`, `LANG`, `TERM` and `PYTHONUNBUFFERED`, plus whatever
`tools.exec.allowedEnvKeys` names. This changes what the credential-isolation
criterion can prove — see *Amendment* below.

## Decision

A new `nanobot/fleet/` package plus a `nanobot/cli/fleet.py` command group.
Nothing in the existing runtime is modified beyond mounting the command group.

### Instance launch

```
/usr/bin/sandbox-exec -p <per-instance profile>
  /usr/bin/env -i <minimal environment>
    <python> -m nanobot <gateway|serve> --config <instance config>
```

spawned with `start_new_session=True`, so the instance and every descendant share
one process-group id. That single pgid is what makes both tree *measurement* and
tree *termination* possible.

### Filesystem confinement

Posture is `(version 1)` + `(allow default)` + targeted denies — deliberately the
inverse of `_seatbelt`. For instance *X*, for every peer *Y ≠ X*, deny
`file-read*` and `file-write*` on `(subpath <Y workspace>)` and
`(subpath <Y config_dir>)`, plus `(literal <fleet file>)` and
`(literal <supervisor state file>)`.

This is weaker than a deny-default allowlist: the rest of the home directory
stays readable. The feature description puts that explicitly out of scope, and
the trade is deliberate — a deny-default profile must enumerate the Python
installation, the virtualenv, `site-packages` and the dyld caches, and a missing
allow surfaces as a mystery crash rather than a clear error. Given that the
failure mode of this whole mechanism is *silence*, the profile that is easy to
verify beats the profile that is theoretically stronger.

### Credential isolation

Each instance's environment is exactly `PATH`, `HOME`, `LANG`, `TMPDIR` plus the
variables named in its own `env` list. A name listed but unset is omitted, never
forwarded as an empty string.

This composes with an existing behaviour rather than fighting it: nanobot's
loader resolves `${VAR}` references at load time and **raises**
`ConfigLoadError(kind="missing_env")` when one is unset
(`nanobot/config/loader.py:196-215`). An instance whose config references a
credential absent from its own `env` list fails loudly at startup, which is the
desired outcome.

### Memory cap

`nanobot/fleet/memory.py` sums resident memory across an instance's process group
via its own `ctypes` handle on `libproc.dylib`, calling `proc_pidinfo` with
`PROC_PIDTASKINFO`. The supervisor samples at most every second; a tree over its
cap has its whole process group killed and is recorded with
`exit_reason: memory`. The one-second bound is what makes the five-second kill
deadline achievable.

Summed tree RSS double-counts shared pages, so the cap is conservative. Accepted:
the feature defines the metric as the total resident memory of the whole tree.

### Status without IPC

The supervisor writes one atomic, `0600`, supervisor-owned JSON file beside the
fleet file, carrying `name`, `pid`, `state`, `exit_reason`, `workspace`,
`config_dir` and `memory_limit_mb` per instance. `fleet status` reads it and
reconciles liveness on read using the public `process_is_running`
(`process_runtime.py:518`) and `process_identity_record` (`:617`), so a recycled
PID is never reported as running.

That file is denied in **every** instance profile. It lists every instance's
workspace and config directory, making it effectively a copy of the fleet file's
contents; leaving it readable would defeat the requirement that an instance
cannot read the fleet file.

## Alternatives considered

**Extend `_seatbelt` rather than add a profile module.** Rejected: opposite
posture, it denies the config directory the instance must read, and it carries
the native shell-sandbox suite.

**A deny-default profile.** Rejected as out of stated scope, and because its
failure mode is a hard-to-diagnose startup crash in a mechanism whose whole
problem is silent failure.

**`psutil` for memory.** Rejected: a new runtime dependency that duplicates
`ctypes` work already in the repo. This feature adds no dependency.

**Reuse each instance's existing `<config_dir>/run/gateway*.json` for status.**
Rejected: those files live inside the instance's own config directory and are
writable by it, so a compromised instance could forge its own state and `status`
would report attacker-controlled data. It also covers only `gateway` mode.

**Supervisor IPC (unix socket or HTTP) for status.** Rejected: a new protocol and
auth surface, and it stops answering the moment the supervisor dies — exactly
when the operator most wants to know what happened.

**Subclassing `ManagedProcessRuntime`.** Rejected in favour of composition. That
class is built for background single-process start/stop with its own per-instance
state file under `<data_dir>/run/`, while the fleet keeps one aggregate
supervisor-owned file and never restarts. Its genuinely valuable parts — the
PID-reuse-safe identity helpers — are public module-level functions and are used
directly, leaving `process_runtime.py` unmodified.

## Amendment to acceptance criterion 5

As written, criterion 5 would pass for the wrong reason. Because
`ExecTool._build_env` (`shell.py:793-805`) forwards only a fixed minimal set plus
`tools.exec.allowedEnvKeys`, running `env` through instance A's shell tool would
show no `SECRET_B` and no `SENTINEL` **even with no isolation at all** — the shell
tool strips them regardless.

Both instance configs therefore set
`tools.exec.allowedEnvKeys: ["SECRET_A", "SECRET_B", "SENTINEL"]`, so the shell
tool is willing to forward all three and the only possible explanation for what A
sees is the supervisor's environment partitioning. The amended criterion is
strictly stronger than the original.

## Decomposition

Twenty-three children. The graph has four roots that can start immediately —
`.1` fleet file schema, `.2` path resolution, `.7` memory sampling and `.15` the
stub LLM server — and funnels through a narrow validation-and-profile spine
(`.1`/`.2` → `.3` → `.4` → `.5`), because a wrong path in the profile fails
silently and must therefore be established once, early, in one reviewable place.

`.4` (profile builder) and `.15` (stub LLM server) are the two chokepoints and
are the only P0 children. `.15` is the largest piece of new scaffolding in the
feature: no stub LLM server exists in the repo today — every current fake is
in-process (`httpx.MockTransport` or an `LLMProvider` subclass) and none can
serve a gateway running in a separate OS process. Five of the six behavioural
acceptance criteria depend on it.

Acceptance is one child per criterion (`.16`–`.21`) so that a failing criterion
rejects a single bead rather than the whole feature.

Sizing follows difficulty rather than directory. Measurement (`.7`, hand-laid
`ctypes` struct offsets) is split from policy (`.11`, a threshold and a deadline)
because a reviewer checking struct offsets and one checking a kill deadline are
looking for different things.

### Files deliberately not touched

- `nanobot/agent/tools/sandbox.py`, `shell.py`,
  `nanobot/security/workspace_access.py` — the guarantee must hold with these
  guards *off*; hardening them would be the wrong axis.
- `nanobot/process_runtime.py` — a chokepoint imported by the gateway runtime;
  composed with via its public helpers only.
- `nanobot/config/schema.py` — the fleet file is a separate document, which makes
  criterion 1 ("no fleet, no change") true by construction rather than by test.
- `nanobot/agent/loop.py`, `runner.py` — `.agent/design.md` forbids growing the
  core path.
- `pyproject.toml` — no new dependency.

## Risks

- **`ctypes` struct offsets** for `proc_taskinfo` are hand-laid and
  macOS-version-sensitive. `.7` states this in its own description so a reviewer
  knows what to check.
- **The stub LLM server** is new surface that six criteria rest on. If it is
  wrong, acceptance tests can pass or fail for reasons unrelated to isolation.
  `.15` therefore carries a self-test that drives a real instance subprocess
  through it.
- **Conservative memory accounting** from shared-page double-counting may kill an
  instance slightly early.
- **CI**: fleet tests bind ports, spawn process trees and send group signals. They
  run in a dedicated serial macOS job (`.22`), following the existing
  `windows_process` carve-out (`.github/workflows/ci.yml:232`) and its stated
  rationale. There is no `pytest-timeout` in this repo, so every wait carries an
  explicit `time.monotonic()` deadline, as the existing gateway smoke test does.

## Open questions

- **Duplicate ports across instances.** The feature description is silent on
  whether two instances configured on the same `gateway.port` or `api.port` should
  make `fleet start` refuse or merely warn. `.3` implements it as a reported
  error; the acceptance tests assign distinct ports explicitly either way.
  Revisit if operators find the refusal too strict.
- **A future Linux arm.** Whether it reuses this profile builder behind a backend
  dispatch, or gets a `bwrap` sibling, as `wrap_command`'s `_BACKENDS` table
  (`sandbox.py:309`) does for the shell tool. Out of scope here; the pinned host
  is macOS.

## Out of scope

Per-named-agent isolation (the `named-agents` feature isolates agents within one
process; this isolates whole instances), automatic restart, limits other than
memory, network isolation, operating systems other than the pinned macOS host,
protection against the supervisor's own user, denying the rest of the home
directory, instances outliving an unexpected supervisor crash, hot reload of the
fleet file, and any web UI for the fleet.

# Fleet

A **fleet** runs several nanobot instances under one supervisor, each in its own OS process, each confined by macOS Seatbelt so that it cannot read or write any other instance's files.

[Multiple Instances](./multiple-instances.md) describes how to run separate instances today. That separation is **convention only**: every instance runs as the same OS user, so each one can read and write every other one's workspace, config file, and session store, and each inherits the whole environment including every API key you exported. The in-app guards (`tools.restrictToWorkspace`, the exec sandbox) are checks *inside* the agent being contained, so a bug, a misconfiguration, or a prompt-injected agent gets past them.

A fleet moves that boundary into the kernel. Read [Limits](#limits) before you rely on it — the boundary is real, but it is narrower than "each instance is in its own box".

> **macOS only.** Confinement is enforced by `sandbox-exec(1)` (Seatbelt). On Linux and Windows `nanobot fleet start` refuses to start anything. See [Limits](#limits).

## Quick Start

Set up two ordinary nanobot instances first, exactly as [Multiple Instances](./multiple-instances.md) describes:

```bash
nanobot onboard --config ~/fleet/telegram/config.json --workspace ~/fleet/telegram/workspace
nanobot onboard --config ~/fleet/discord/config.json  --workspace ~/fleet/discord/workspace
```

Declare them in one fleet file, `~/fleet/fleet.json`:

```json
{
  "instances": {
    "telegram": {
      "config": "~/fleet/telegram/config.json",
      "mode": "gateway",
      "memoryLimitMb": 1024,
      "env": ["ANTHROPIC_API_KEY"]
    },
    "discord": {
      "config": "~/fleet/discord/config.json",
      "mode": "gateway",
      "memoryLimitMb": 1024,
      "env": ["OPENAI_API_KEY"]
    }
  }
}
```

Each instance's config must set a different `gateway.port`, because two instances cannot bind the same port. Start the fleet:

```bash
nanobot fleet start --fleet ~/fleet/fleet.json
```

The supervisor holds that terminal. From a second shell:

```bash
nanobot fleet status --fleet ~/fleet/fleet.json
nanobot fleet status --fleet ~/fleet/fleet.json --json
nanobot fleet stop   --fleet ~/fleet/fleet.json
```

## The Fleet File

One JSON object with a single required key, `instances`, mapping instance names to entries. Unknown keys are refused rather than ignored — a misspelled `memoryLimtMb` under an "ignore unknown keys" policy would leave that instance running uncapped, so every typo is an error.

| Field | Required | Type | Meaning |
|---|---|---|---|
| `config` | yes | string | Path to that instance's `config.json`. Must be absolute after `~` expansion, and must already exist. |
| `mode` | yes | `"gateway"` or `"serve"` | Which nanobot command the instance runs: `nanobot gateway` or `nanobot serve`. |
| `memoryLimitMb` | yes | integer > 0 | Resident memory cap for the instance's **whole process tree**, in binary MB (1 MB = 1048576 bytes). |
| `env` | no | list of strings | Environment variable **names** this instance may inherit. Defaults to `[]`. |

Instance names must match `^[a-z0-9][a-z0-9_-]*$`. They become process labels, state-file keys, and Seatbelt profile names, so spaces, slashes, and uppercase are refused.

There is **no default** for `config`, `mode`, or `memoryLimitMb`. There is no safe value to invent for a memory cap or a launch mode, and an instance that silently ran uncapped because a key was omitted would defeat the point of declaring a fleet.

### `env` holds names, never values

```json
{
  "config": "~/fleet/telegram/config.json",
  "mode": "gateway",
  "memoryLimitMb": 1024,
  "env": ["ANTHROPIC_API_KEY", "BRAVE_API_KEY"]
}
```

Each entry must match `^[A-Za-z_][A-Za-z0-9_]*$`. Writing `"ANTHROPIC_API_KEY=sk-..."` is refused — a fleet file is meant to contain no credentials, and the name rule is what catches an operator about to commit one.

A name that is listed but **unset** in the supervisor's environment is left unset in the instance, not forwarded as an empty string. That is deliberate: nanobot's config loader raises a `missing_env` error for an unresolvable `${VAR}` reference, so an instance whose config reaches for a key its entry never listed fails loudly at launch instead of coming up and talking to a provider with a blank key.

Beyond `env`, an instance receives only `PATH`, `HOME`, `LANG`, and `TMPDIR`. Nothing else from the supervisor's environment crosses — in particular, nothing another instance's entry names.

### Layout rules

A fleet is refused, before anything starts, if:

- any instance's `config` path is relative after expansion, or does not exist;
- any instance's `agents.defaults.workspace` is relative after expansion;
- two distinct instances' workspaces or config directories **overlap** — share a path, or one contains the other. A `subpath` deny covers everything beneath it, so a nested pair cannot be separated by a profile at all: the rule that walls the peer off also walls the instance off from its own files;
- an instance's config directory is at or inside its own workspace (nanobot's own session store refuses this shape);
- two instances would bind the same TCP port. Only the port the declared `mode` actually binds is compared — `gateway.port` for `gateway`, `api.port` for `serve` — but the comparison is by port *number*, so a `serve` instance moved onto the gateway's default port is still caught.

A workspace nested inside its **own** config directory is normal — that is the default `~/.nanobot-x/workspace` layout. The overlap rule is strictly about *distinct* instances.

## Commands

### `nanobot fleet start`

```bash
nanobot fleet start --fleet <path>
```

Runs four gates in order, then holds the foreground:

1. **Validate** the fleet document and every instance's paths.
2. **Prepare** the directories, the supervisor state file, and each instance's Seatbelt profile.
3. **Prove** each profile actually binds on this host.
4. **Start** every instance and watch it.

Steps 1–3 refuse by exiting non-zero and naming the offending instance, and none of them starts an instance process or leaves a directory behind. Step 4 is all-or-nothing: if any instance fails to launch, the ones already started are stopped and recorded before the command exits non-zero.

The proof step is a gate and not a warning because Seatbelt gives no feedback. A deny naming a path that does not exist, or a non-canonical spelling of one (`/tmp/x` where the real path is `/private/tmp/x`), is *accepted* by `sandbox-exec`, which then starts the process cleanly, confines nothing, and exits 0. So before any instance exists, the supervisor runs each profile for real and watches a read of a denied path fail. A profile that cannot be observed to refuse is reported as unproven and the fleet does not start.

Each instance is launched as:

```text
/usr/bin/sandbox-exec -p <profile> \
    /usr/bin/env -i <minimal environment> \
    <python> -m nanobot <gateway|serve> --config <instance config>
```

`sandbox-exec` is outermost because a Seatbelt profile is inherited by everything the process goes on to start, so the instance, its shell tool, and every grandchild are inside the same policy. Each instance is spawned into its own process group, which is what makes its whole tree measurable and terminable as a unit.

Per-instance output goes to `<config_dir>/logs/fleet.log`, inside the instance's own data directory — where every peer's profile already denies it.

### `nanobot fleet status`

```bash
nanobot fleet status --fleet <path>
nanobot fleet status --fleet <path> --json
```

Reads the supervisor's state file and writes nothing. Both facts matter: the supervisor holds the foreground of the terminal it was started in, so status has to be answerable from a *second* shell, and because the supervisor is the file's only writer a reader that repaired what it found would race the write it was reading.

Liveness is re-derived on the way past rather than trusted from the last write. A pid is not an identity — between the write and the read an instance may have exited and its pid may have been handed to an unrelated process — so each record's pid is checked against a PID-reuse-safe identity recorded beside it.

`--json` prints an array with exactly seven fields per instance:

```json
[
  {
    "name": "telegram",
    "pid": 78185,
    "state": "running",
    "exit_reason": null,
    "workspace": "/Users/you/fleet/telegram/workspace",
    "config_dir": "/Users/you/fleet/telegram",
    "memory_limit_mb": 1024
  }
]
```

| Field | Type | Meaning |
|---|---|---|
| `name` | string | The instance name from the fleet file. |
| `pid` | integer | The **instance's** own process. Never the supervisor's. |
| `state` | `"running"` or `"exited"` | Reconciled at read time. |
| `exit_reason` | `"memory"`, `"signal"`, `"exit"`, or `null` | Why it ended; `null` while running. |
| `workspace` | string | Canonical workspace path. |
| `config_dir` | string | Canonical config directory. |
| `memory_limit_mb` | integer | The cap from the fleet file. |

The PID-reuse identity is deliberately not published: it is platform-specific (on macOS the bare value is a process *group*), so exposing it would freeze a private format into an interface and invite a consumer to signal a group only the supervisor is positioned to address.

Status exits zero whatever the instances are doing — an exited instance is a report, not an error. Only a state file that cannot be read or believed is a refusal, and a **missing** file is one of those cases: `status` on a fleet that was never started exits non-zero rather than printing an empty array. Answering "no instances" over a file it could not read would tell you the fleet is stopped, and the next thing you would do is start a second fleet on the same ports and workspaces.

### `nanobot fleet stop`

```bash
nanobot fleet stop --fleet <path>
nanobot fleet stop --fleet <path> --grace 10
```

Terminates every instance's whole process tree, then the supervisor, escalating `SIGTERM` then `SIGKILL`. `--grace` (default `5.0` seconds) is how long everything is given to leave after `SIGTERM`.

It targets the **tree**, not the pid. An instance's shell tool starts each command in a new session, so a long-running child is outside the instance's process group from the moment it starts; a group signal alone would leave it running. Every process ever seen in a tree is remembered and re-signalled, which is what keeps a descendant that survived `SIGTERM` and was then orphaned by its parent's death reachable for the `SIGKILL`. Every remembered pid is re-checked against its identity before being signalled, so a recycled pid is dropped rather than killed.

The command returns only once nothing of the fleet is left, and exits non-zero naming whatever would not die. A stop that returned optimistically would hand you a prompt back while confined processes were still writing to their workspaces.

`stop` never reads the fleet document — it needs only the state file, whose location it derives from the fleet path. That is deliberate: a fleet that is already running must be stoppable even if its declaration has since been edited, moved, or made invalid. A `stop` that refused on a validation error would leave confined processes running with the only command that can reach them refusing to run.

## What the Fleet Guarantees

### Filesystem separation

Each instance runs under a Seatbelt profile that **denies reading and writing**:

- every *other* instance's workspace;
- every *other* instance's config directory (its sessions, media, cron, and logs);
- the fleet file itself;
- the supervisor's state file, `<fleet file name>.state.json`.

`file-read*` covers metadata, so a denied directory cannot even be `stat`'d. `file-write*` covers unlink, and a final rule additionally denies `file-write-unlink` on every *ancestor* of every denied path — because a `subpath` rule matches by path and not by inode, so renaming a directory above a peer's workspace would otherwise make the deny stop matching.

The state file is denied in every profile because it maps every instance's workspace and config directory; leaving it readable would hand an instance the index the fleet-file deny exists to withhold, by a second route. It is written `0600` on every write.

Everything **not** on that list stays permitted. See [Limits](#limits).

This holds with `tools.restrictToWorkspace` set to `false` and the exec sandbox disabled, because it is enforced by the kernel on the instance process and everything it starts, not by a check inside the agent.

### Credential separation

An instance sees `PATH`, `HOME`, `LANG`, `TMPDIR`, and the variables its own entry names. It does not see the supervisor's other credentials, and it does not see variables another instance's entry names.

### Memory cap

`memoryLimitMb` is enforced against the resident memory of the instance's **whole process tree**, not just its main process. A tree over its cap is `SIGKILL`ed within 5 seconds and recorded with `exit_reason: "memory"`. The supervisor samples at least once per second, which is what makes that deadline meaningful.

A tree that cannot be sampled is left alone rather than killed — a tree whose members have all exited samples as zero, which keeps "using nothing" distinguishable from "cannot tell".

### Proof before start

No instance is started until its profile has been observed to refuse a read on this host. See [`nanobot fleet start`](#nanobot-fleet-start).

## Limits

**Read this section before relying on the boundary.** A fleet gives you *filesystem separation between named instances on macOS*, plus a memory cap and per-instance credentials. It is not a sandbox, not a container, and not a VM. Specifically:

- **The rest of your home directory is not denied.** The profile is allow-by-default: it denies each instance's *peers*, the fleet file, and the state file, and permits everything else. An instance can still read and write `~/.ssh`, `~/Documents`, `~/.aws`, your browser profiles, and every other file your user account can reach. If you need that closed, use the exec sandbox (`tools.exec.sandbox: "seatbelt"`), a dedicated OS user, or a VM — the fleet does not replace any of them.
- **No network isolation.** Nothing in a fleet profile restricts network access. An instance can reach the internet, your LAN, and `127.0.0.1` — including another instance's gateway or API port. Port *clashes* are validated; port *access* is not restricted.
- **No automatic restart.** An instance that exits stays exited and keeps being reported as exited. This is a choice, not an omission: each instance is confined by a profile built from a snapshot of the fleet document and holds credentials derived from the supervisor's environment at start, so a restart would re-spawn it from state that may no longer be true — and a crash loop a supervisor hides is a crash loop nobody fixes. One instance's death does not stop the others.
- **No CPU, file-descriptor, disk, or process-count limits.** Memory is the only resource that is capped. An instance can spin a core, fill a disk, or fork until the machine's process table is exhausted. Memory is a sampled resident-set measurement, so a tree that allocates and frees faster than the sampling interval can briefly exceed its cap without being caught.
- **No per-named-agent isolation.** The unit of confinement is the *instance process*. Every session, subagent, skill, MCP server, and shell command inside one instance shares that instance's boundary and its credentials. A fleet separates instance from instance; it does nothing to separate agents, sessions, or tools from each other within one instance.
- **Same OS user, same identity.** Instances are ordinary processes owned by you. This is filesystem containment enforced by Seatbelt — the same posture the exec sandbox section of [SECURITY.md](../SECURITY.md) describes — not a separate user identity, and not a VM. Anything reachable through your user's credentials, keychain access, or running agents is still reachable.
- **The supervisor is not confined.** It runs unsandboxed with your full environment; that is how it can build profiles, read the fleet file, and sample and signal every tree. Trust it exactly as much as you trust `nanobot` itself.
- **macOS only.** Confinement is `sandbox-exec(1)`. On Linux and Windows there is no Seatbelt, the confinement proof cannot pass, and `nanobot fleet start` refuses to start anything rather than running a fleet that is unconfined. There is no bubblewrap fleet backend; on Linux, use containers or separate OS users.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Invalid fleet file: ...` with `invalid setting(s)` | A missing or misspelled key, a bad instance name, or an `env` entry that is not a bare variable name. |
| `instances.<a>, instances.<b>` naming two paths | Two instances' workspaces or config directories overlap, or they would bind the same port. Give each instance a directory tree of its own. |
| `config path must be absolute after expansion` | Use an absolute path (or one starting with `~`); a relative path would resolve against whatever directory the supervisor was started from. |
| `Confinement could not be proven, so no instance was started.` | Usually a host with no `/usr/bin/sandbox-exec` — i.e. not macOS. Every unproven instance is named, so one run tells you about all of them. |
| `session storage must be outside the agent workspace` | That instance's config file sits inside its own workspace. Move the config file out, or nest the workspace under the config directory. |
| `Invalid fleet state file: ... No such file or directory.` from `status` or `stop` | There is no state file beside the fleet document, so there is nothing to report on or stop. Either the fleet was never started, or the `--fleet` path points somewhere else. This is a refusal (exit 1), not an empty report — see below. |
| `status` says *No instance has been started for this fleet.* | The state file exists but lists no instances. Exits zero. |
| `This fleet was not fully stopped.` | Something would not die within the grace period even after `SIGKILL`. The surviving instances are named. |

## See Also

- [Multiple Instances](./multiple-instances.md) — running separate instances without a fleet, and what that does *not* isolate
- [CLI Reference: Fleet](./cli-reference.md#fleet) — the command table
- [SECURITY.md](../SECURITY.md) — the fleet boundary alongside the exec sandbox boundary
- [Configuration](./configuration.md) — the per-instance `config.json` a fleet entry points at

# Fleets: Process Isolation Between Instances

[Multiple instances](multiple-instances.md) keeps instances apart by convention:
each has its own config file and workspace, but they all run as the same OS
user, so any one of them can read every other one's files and inherits every
secret in the environment it was started from.

A **fleet** replaces that convention with confinement the operating system
enforces. `nanobot fleet start` runs each instance as a separate OS process
that:

- can read and write only its own workspace and config directory, and cannot
  reach another instance's workspace or config directory, or the fleet file
- receives only a minimal base environment plus the variables its own fleet
  entry names
- is killed if its process tree grows past its own memory cap

The confinement applies to the instance process and every process it starts —
the shell tool's commands included — and holds even when the instance's own
guards are off (`restrictToWorkspace: false`, shell sandbox disabled).

> Fleets require macOS, where confinement uses Seatbelt (`sandbox-exec`). On
> other platforms `fleet start` exits with an error.

## The fleet file

```json
{
  "instances": {
    "research": {
      "config": "~/.nanobot-research/config.json",
      "mode": "gateway",
      "memoryLimitMb": 1024,
      "env": ["OPENROUTER_API_KEY"]
    },
    "trader": {
      "config": "~/.nanobot-trader/config.json",
      "mode": "gateway",
      "memoryLimitMb": 512,
      "env": ["OPENROUTER_API_KEY", "BROKER_TOKEN"]
    }
  }
}
```

| Field | Meaning |
| --- | --- |
| instance name | Must match `[a-z0-9][a-z0-9_-]*` |
| `config` | Path to that instance's ordinary nanobot config file. Its parent is the instance's **config directory**; its `agents.defaults.workspace` is the instance's **workspace**. |
| `mode` | `gateway` (default) runs `nanobot gateway`; `serve` runs the OpenAI-compatible API. |
| `memoryLimitMb` | Required. Resident memory cap for the instance's whole process tree, in MiB. |
| `env` | Names of environment variables copied from the supervisor's environment. Values never appear in the fleet file. |

Two instances whose workspaces or config directories are the same, or nested
one inside the other, are a configuration error: `fleet start` names both
instances and starts nothing.

## Commands

```bash
# Validate the fleet file, start every instance, and supervise in the foreground
OPENROUTER_API_KEY=... BROKER_TOKEN=... nanobot fleet start --fleet ~/fleet.json

# From another shell
nanobot fleet status --fleet ~/fleet.json --json
nanobot fleet stop   --fleet ~/fleet.json
```

`fleet status --json` prints one object per instance:

```json
[
  {
    "name": "research",
    "pid": 41207,
    "state": "running",
    "exit_reason": null,
    "workspace": "/Users/you/.nanobot-research/workspace",
    "config_dir": "/Users/you/.nanobot-research",
    "memory_limit_mb": 1024
  }
]
```

`pid` is always the instance's own process, never the supervisor's. `state` is
`running` or `exited`; `exit_reason` is `memory`, `signal`, `exit` or `null`.

## Lifecycle

Instances live and die independently. When one exits — it crashed, it was
killed, or it outgrew its cap — the others keep running and serving, and the
supervisor does **not** restart it; it keeps reporting it as exited with the
reason. `nanobot fleet stop` terminates every instance's whole process tree,
then the supervisor, and returns once no instance process remains.

The supervisor publishes its state under `~/.nanobot/fleet/` (override with
`NANOBOT_FLEET_STATE_DIR`), which is how `status` and `stop` work from a
different shell.

## What a fleet does not do

- It does not isolate individual named agents; the unit of isolation is a whole
  instance.
- It does not restart exited instances.
- It caps memory only — not CPU, open files, disk or process count.
- It does not restrict network access.
- It does not protect instance files from other processes running as the same
  user outside the fleet.
- It does not deny the rest of your home directory, only other instances' paths
  and the fleet file.

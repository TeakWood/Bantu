---
title: Named agents
status: accepted
date: 2026-09-23
superseded-by:
---

# Named agents

Epic: `Bantu-beads-4bq` (15 children, `Bantu-beads-4bq.1` … `.15`).

## Context

A nanobot install runs exactly one agent. Every conversation, on every channel,
lands in the same memory, the same session store and the same tool set.
Background subagents exist, but they are ephemeral helpers of that one agent,
not independent agents.

That is the wrong shape for a personal assistant that does more than one kind of
job — a research agent whose memory should be only its research notebook, a
trading agent that alone holds the brokerage connection, a health agent whose
compartment must stay sealed. The requirement is stronger than convention:

> No other agent may even *see* the brokerage tools. Asking another agent, in
> its prompt, not to use them does not count as keeping it from them.

So isolation has to be structural. Anything achieved by instructing a model is
out of bounds by construction.

Running a separate process per agent works today (`docs/multiple-instances.md`)
at the cost of one config, one provider setup and one gateway per agent. This
decision makes several agents first-class within one install, one config file
and one gateway.

The existing single agent becomes the agent named `default`, configured exactly
as today by `agents.defaults` and the top-level `tools` block, answering every
chat no named agent claims.

## Constraints that shaped the design

Two repo facts did more to determine the shape than anything in the feature
description.

**1. `.agent/design.md` is normative and protects the core path.**

> Core stays small; extend at the edges. […] The files `agent/loop.py` and
> `agent/runner.py` form the critical core path; changes there should be minimal
> and justified.

Agent selection therefore cannot live in `AgentLoop`, even though `AgentLoop` is
where every inbound message already funnels.

**2. One `MessageBus` cannot serve N agents.** `MessageBus.inbound` is a single
`asyncio.Queue` and `consume_inbound()` *pops* it
(`nanobot/bus/queue.py:32,41-43`). N `AgentLoop`s sharing one bus would steal
each other's messages — not a filtering problem, a queue-semantics problem.

## Decision

### Per-agent bus, demuxed at the composition root

Each agent gets its own `MessageBus`. A new `MultiAgentRuntime` owns the
channel-facing bus and pumps between it and the per-agent buses:

```
ChannelManager ─┬─ channel bus ─→ route(config, channel, chat_id) ─→ agent bus ─→ AgentLoop
                └────────────── ← outbound pump ← agent bus ← ────────────────────┘
```

Everything new lives in a new `nanobot/agents/` package plus the gateway
composition root. **`agent/loop.py`, `agent/runner.py`, `bus/queue.py`,
`session/manager.py` and `channels/manager.py` are not modified at all.**

### Three isolation properties are inherited, not built

This is the part worth remembering, because it is why the change is smaller than
the feature sounds.

**Subagent results stay with their parent for free.** Subagent results are
delivered by a bus round-trip, not a callback: `SubagentManager._announce_result`
republishes `InboundMessage(channel="system", …, session_key_override=<parent's
key>)` onto `self.bus` (`nanobot/agent/subagent.py:554-563`). Because `self.bus`
is the spawning agent's own bus, a result cannot reach another agent. No
`channel="system"` routing rule is needed and `subagent.py` needs no change.
Had we chosen a shared bus, this would have required inventing an agent
discriminator for system messages — the per-agent bus removes the problem
instead of solving it.

**Sessions follow the workspace.** `JsonlSessionStore` claims a *workspace-id*
subdirectory under `<config-dir>/sessions/` and explicitly refuses to live
inside the workspace (`nanobot/session/manager.py:455-459,475`). Distinct
workspace ⇒ distinct session namespace, on the filesystem.

**Memory follows the workspace.** `MemoryStore` derives every path from the
workspace root (`nanobot/agent/memory.py:72-90`) and there is exactly one
construction site (`nanobot/agent/context.py:98`).

Giving each agent its own `SessionManager` also closes two latent leaks that
would otherwise have become real the moment a second agent existed:
`_pick_heartbeat_target_from_sessions` (`nanobot/cli/gateway_runtime.py:200-227`)
returns the first routable chat from *all* sessions, and
`AutoCompact.check_expired` iterates `list_sessions()` globally. Both become
scoped by construction.

### Configuration and resolution

Named agents are declared under `agents.named`, keyed by name, matching
`[a-z0-9][a-z0-9_-]*`, with `default` reserved. An entry accepts every field
`agents.defaults` accepts, plus a `tools` block of the same shape as the
top-level one.

An agent's effective settings are `agents.defaults` with its own entries laid
over, with two deliberate exceptions:

- **Workspace is never inherited.** A named agent with none gets
  `~/.nanobot/agents/<name>`, never the default agent's workspace. This is what
  makes the memory and session isolation above structural rather than
  conventional.
- **`tools.mcpServers` is never inherited.** An agent has exactly the servers in
  its own block; top-level servers belong to `default` alone. This is the direct
  expression of the trading-agent requirement.

The rest of `tools` merges over the top-level block.

This required a small type decision with disproportionate reach: an entry must
distinguish *unset* from *explicitly set to the default value*, or a field
merely carrying `AgentDefaults`' default would win over a configured default.
We use pydantic's `model_fields_set` rather than redeclaring ~25 fields as
`Optional`.

Note the nearest existing analogue deliberately behaves differently:
`Config.resolve_preset` (`nanobot/config/schema.py:472-488`) *replaces* rather
than merges, because `ModelPresetConfig` carries its own hard defaults. It is
not reused.

### Telegram binding

Telegram gains the multi-instance convention Feishu already has — an `instances`
list, each entry with an `id`, runtime channel name `telegram.<id>`, with the
unsuffixed `telegram` still meaning the default bot. An `agent` field on a bot
binds it to an agent; unbound bots and all other channels go to `default`.

`ChannelManager` needs no change: `_build_channel` already assigns
`channel.name = runtime_name` (`nanobot/channels/manager.py:216-218`), and
`BaseChannel._handle_message` stamps `channel=self.name`
(`nanobot/channels/base.py:311`), so replies route back through
`ChannelManager.channels[msg.channel]` to the exact bot the message arrived on.

The template is `nanobot/channels/feishu/instances.py` — in particular
`runtime_channel_name` (`:26-28`), the legacy flat-section path (`:92-106`) that
keeps existing single-bot configs working, and `FEISHU_MANAGEMENT` (`:249-255`).

### Scheduled work: excluded by construction

Cron, Dream and heartbeat keep running for `default` exactly as today. Named
agents receive `cron_service=None`, so the `cron` tool is never constructed for
them — `CronTool.enabled` gates purely on `ctx.cron_service is not None`
(`nanobot/agent/tools/cron.py:65`). This needs no new mechanism:
`Nanobot.from_config` already builds cron-less loops today
(`nanobot/nanobot.py:135-141`).

Sharing the one `CronService` was considered and rejected as a leak, not merely
as scope creep. `on_cron_job` (`nanobot/cli/gateway_runtime.py:556-698`)
dispatches by job *name* against one captured `agent`, the store lives at
`config.workspace_path` (i.e. `agents.defaults.workspace`), and neither
`CronJob` nor `CronPayload` carries an agent identifier — bound jobs carry
*session* delivery context. A named agent's bound job would therefore execute
against the default agent, replaying that agent's session content into
`default`. That is precisely the compartment breach this feature exists to
prevent.

### MCP failure semantics: isolate

A named agent whose MCP server fails to launch comes up with its built-in tools;
other agents are unaffected. This is the existing behaviour, not new code:
`MCPProvider.connect()` catches `BaseException`, marks those servers `"failed"`,
logs a warning and retries on the next readiness check
(`nanobot/agent/tools/mcp.py:1469-1474`), and `connect_mcp_servers` returns only
the subset that came up. Failing the whole gateway would have been the expensive
option.

### Workspace bootstrap

Each agent's workspace is created and seeded with `sync_workspace_templates`
(`nanobot/utils/helpers.py:897`) — the same treatment `default`'s workspace
gets — so every named agent has its own `SOUL.md`, `USER.md` and `memory/` from
turn one rather than a subtly blanker persona.

## Alternatives considered

**One bus, N loops, filter on consume.** Rejected: `consume_inbound()` pops, so
filtering means each loop drains messages meant for others. It would require
changing `MessageBus`, which the design rules protect.

**Route inside `AgentLoop._effective_session_key`**
(`nanobot/agent/loop.py:911-915`). Genuinely tempting — it is the single funnel
every message already passes through, after runtime-control and priority-command
filtering, and it is where the only existing routing policy (unified vs
per-channel sessions) already lives. Rejected: it couples routing to the thing
being routed to, and violates the core-stays-small rule.

**Agent-prefixed session keys (`agent:channel:chat_id`) over one shared store.**
Rejected: isolation would be a naming convention inside a shared directory. The
health-agent requirement demands a filesystem boundary, and a convention is
exactly the kind of guarantee that erodes.

**Per-agent `CronService`.** Rejected as out of scope; see above for why the
shared alternative is worse than merely broad.

## Consequences

Enabling. Several assistants in one install with enforced compartments; one
config, one provider setup, one gateway. `default` behaves exactly as today
when `agents.named` is absent.

Costs and limits.

- MCP startup cost and process footprint scale with agent count.
- Named agents run no cron, Dream or heartbeat until a follow-on feature.
- Channels other than Telegram serve `default` only.
- The CLI (`nanobot agent`) and WebUI still talk to `default`, so named agents'
  sessions are invisible in the UI.
- `nanobot agents list --json` introduces the first machine-readable output mode
  in the CLI; there is no existing house style to follow.

### A pre-existing bug this surfaced

Checking acceptance criterion 6 (`subagent_tool_names()` must be a subset of
`tool_names()`) exposed a bug that affects `default` today, independently of
this feature. `SubagentManager._subagent_tools_config`
(`nanobot/agent/subagent.py:204-211`) forwards only `exec`, `web`, `file` and
`restrict_to_workspace`, so `cli_apps` reverts to its own default of
`enable=True` (`nanobot/agent/tools/cli_apps.py:30`). With
`tools.cliApps.enable = false` the parent loses `run_cli_app` while its
subagents keep it. Filed as `Bantu-beads-4bq.5`.

### A latent cross-agent leak this design must fix

`nanobot/channels/telegram/runtime.py:1664-1670` builds
`f"telegram:{chat_id}:topic:{thread_id}"` as a **literal** rather than from
`self.name` — the one place Telegram diverges from the multi-instance
convention every other consumer follows (compare
`nanobot/channels/feishu/runtime.py:2795-2824`). Left unfixed it silently merges
group-topic sessions from all bots into one namespace: a cross-agent session
leak that no existing test would catch, because it only manifests once a second
bot exists. The fix is a provable no-op for the default instance, where
`self.name == "telegram"`.

## Open questions

Recorded rather than blocking.

- A named agent may set `dream:` in its entry, since an entry accepts every
  `agents.defaults` field. With scheduled work out of scope that field is
  **accepted but inert**. Rejecting it would be config validation, which is
  explicitly out of scope; revisit with the scheduled-work feature.
- Unknown keys under `agents.named.<name>` are silently ignored (`Base`
  inherits `extra="ignore"`). Consistent with the rest of the schema, but a
  typo'd field will not error.
- Detecting two agents that share a workspace, and a bot bound to an undeclared
  agent, are both out of scope. The first would defeat the isolation guarantee
  if an operator configured it; worth revisiting.

## Out of scope

Each is a candidate for a later, separate feature: scheduled work for named
agents; channels other than Telegram; per-chat binding; config validation beyond
names; a CLI to talk to a named agent; separate processes, resource limits and
per-agent credential environments (the `process-isolation` feature);
agent-to-agent messaging and delegation; per-agent web UI, per-agent
`nanobot serve`, and hot reload of the agent list; and the example agents' own
behaviour.

## Plan

15 children, dependency-ordered. The graph is a diamond: config (`.1`, `.2`) and
channels (`.3`, `.4`) are independent roots converging on the registry (`.6`);
the runtime spine (`.7` → `.9` → `.10`/`.11`) carries the risk; two acceptance
beads (`.13`, `.14`) close the contract at the two contact points the feature
description names; docs (`.15`) sink everything.

| Bead | Title |
|---|---|
| `.1` | `agents.named` config schema and name validation |
| `.2` | Effective agent config resolution |
| `.3` | Telegram multi-instance scaffolding |
| `.4` | Telegram runtime instance awareness and session-key namespacing |
| `.5` | Forward `cliApps` to the subagent tools config (pre-existing bug) |
| `.6` | Agent registry and `route()` |
| `.7` | Per-agent `AgentRuntime` bundle |
| `.8` | `Nanobot.from_config(agent=...)` facade |
| `.9` | `MultiAgentRuntime` inbound/outbound demux |
| `.10` | Rewire `nanobot gateway` onto `MultiAgentRuntime` |
| `.11` | `open_gateway()` in-process harness |
| `.12` | `nanobot agents list --json` |
| `.13` | Acceptance: criteria 1–6 through the facade and CLI |
| `.14` | Acceptance: criteria 7–9 through `open_gateway` |
| `.15` | Documentation |

Edges: `.2←.1`; `.4←.3`; `.6←.2,.3`; `.7←.2`; `.8←.7,.5`; `.9←.6,.7`;
`.10←.9`; `.11←.9`; `.12←.6`; `.13←.8,.12`; `.14←.11,.4,.12`;
`.15←.10,.11,.12`.

`.9` and `.10` are the two P0 beads. `.10` stands alone because of what the file
is: `_run_gateway` (`nanobot/cli/gateway_runtime.py:340-1034`) is a 695-line
function with roughly fifteen nested closures, nearly all capturing one `agent`
variable. `.9` lands first with its own tests so that rewiring the gateway onto
it is a substitution rather than a design.

`open_gateway` does not exist today — zero hits repo-wide, and there is no
fake-channel + bus + loop harness to extract. The idiomatic precedents to follow
are `nanobot/channels/websocket/tests/ws_test_client.py` (the only reusable
importable harness in the tree), the inject/observe pattern in
`tests/channels/test_base_channel.py`, and the bus-level pattern in
`tests/agent/test_loop_runner_integration.py`. Note that basedpyright runs
strict over `nanobot/` but excludes `**/tests`, so a harness shipped as library
code under `nanobot/agents/` is strictly type-checked.

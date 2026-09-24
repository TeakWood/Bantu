# Named Agents

Run several first-class agents inside one install, one config file and one
gateway. Each has its own workspace, memory, sessions, model settings, tool set
and Telegram bot.

A named agent is **not** a subagent. It is a full peer of the agent nanobot has
always run — which is now the agent named `default`, configured exactly as
before by `agents.defaults` and the top-level `tools` block.

If your config has no `agents.named`, nothing changes.

## When to use this

Use named agents when one assistant does more than one kind of job and those
jobs must not share a memory:

- a **research** agent whose memory is a research notebook and nothing else
- a **trading** agent that alone holds the brokerage tools — no other agent, and
  no subagent of another agent, can even see them
- a **health** agent whose symptoms, medications and test results stay sealed
  away from the general-purpose agent

For fully separate processes, ports and credential environments, see
[Multiple Instances](./multiple-instances.md) instead.

## Declaring agents

```jsonc
{
  "agents": {
    "defaults": { "model": "anthropic/claude-sonnet-5" },
    "named": {
      "research": {
        "model": "anthropic/claude-opus-5-5",
        "workspace": "~/.nanobot/agents/research"
      },
      "trader": {
        "tools": {
          "mcpServers": { "broker": { "command": "broker-mcp" } }
        }
      }
    }
  }
}
```

A name must match `[a-z0-9][a-z0-9_-]*`. `default` is reserved.

Each entry accepts every field `agents.defaults` accepts, plus a `tools` block
with the same shape as the top-level one.

## How settings resolve

An agent's effective settings are `agents.defaults` with its own entries laid
over it. Two things are **never** inherited:

| Setting | Rule |
|---|---|
| `workspace` | A named agent that sets none gets `~/.nanobot/agents/<name>`. It never gets the default agent's workspace. |
| `tools.mcpServers` | A named agent has exactly the servers in its own `tools.mcpServers`. Servers in the top-level `tools.mcpServers` belong to `default` only. |

The rest of a named agent's `tools` block merges over the top-level `tools`
block.

Check what resolved:

```bash
nanobot agents list
nanobot agents list --json --config ~/.nanobot/config.json
```

## Isolation

- **Memory.** Each agent reads and writes only its own workspace's identity
  files (`SOUL.md`, `USER.md`) and memory (`memory/MEMORY.md`,
  `memory/history.jsonl`).
- **Sessions.** Each agent's sessions are stored apart from every other agent's.
  A conversation with one agent never appears in another's session listing or
  history, even on the same channel and chat id.
- **Tools.** A tool one agent has is not a tool another agent has.
- **Subagents.** A subagent's tool set is always a subset of its parent's, and
  its result is delivered back only to the agent and session that spawned it.

## Binding a Telegram bot

Each agent talks to the user through its own Telegram bot, so every agent has
its own one-to-one chat. The `telegram` channel config takes an `instances`
list, following the same convention as Feishu:

```jsonc
{
  "channels": {
    "telegram": {
      "enabled": true,
      "instances": [
        { "id": "default",  "token": "111:aaa" },
        { "id": "research", "token": "222:bbb", "agent": "research" },
        { "id": "trader",   "token": "333:ccc", "agent": "trader" }
      ]
    }
  }
}
```

- A bot's runtime channel name is `telegram.<id>` — for example
  `telegram.research`. The unsuffixed `telegram` keeps meaning the default bot,
  so existing single-bot configs are unchanged.
- The `agent` field binds that bot to an agent. Every chat arriving through it
  goes to that agent.
- A bot with no `agent` field, and every other channel, goes to `default`.
- A reply always leaves through the same bot and chat the message arrived on.

## From Python

```python
from nanobot import Nanobot

bot = Nanobot.from_config("~/.nanobot/config.json", agent="research")
print(bot.agent_name, bot.workspace)
print(bot.tool_names())           # including MCP tools, after connect_mcp()
print(bot.subagent_tool_names())  # what its background subagents get
```

Route an inbound message without building anything:

```python
from nanobot.agents import route

route(config, "telegram.research", chat_id="42")  # -> "research"
```

Drive the whole multi-agent runtime in-process, with no external channel
connected:

```python
from nanobot.agents import open_gateway

async with open_gateway("~/.nanobot/config.json") as gateway:
    await gateway.inject("telegram.research", "42", "user-1", "what moved today?")
    out = await gateway.next_outbound(timeout=30)
    print(out.channel, out.chat_id, out.content)
```

`nanobot gateway` routes live Telegram traffic through the same per-agent
runtime. It starts every declared bot, keeps the `default` agent's own runtime —
cron, heartbeat, Dream and the WebUI — and runs each named agent beside it on a
private queue. A message on a bound bot's channel is handed to that agent, and
its reply goes back out through the same bot.

A bot bound to an agent that is not declared under `agents.named` is served by
nobody: the gateway warns at startup and drops its messages rather than letting
them accumulate in the `default` agent's memory and sessions.

## Not covered yet

Cron jobs, heartbeat and Dream consolidation keep running for `default` exactly
as today. Channels other than Telegram serve `default` only. Routing individual
chats within one bot to different agents, a CLI that talks to a named agent, and
agent-to-agent delegation are all separate features.

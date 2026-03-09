# Running Bantu Services

Bantu can be deployed in two modes:

- **Embedded (single-process) mode** — one `nanobot gateway` process runs
  everything.  This is the default and requires no extra configuration.  Use it
  for local development or simple self-hosted setups.
- **Distributed mode** — three independent processes (agent, admin, gateway)
  that communicate over HTTP.  Use it when you want to scale or containerise
  each service separately.

---

## Embedded mode (backward-compatible, default)

Run a single command:

```bash
nanobot gateway
```

This starts the full stack in one process on port **18790**:

| Component | What it does |
|-----------|-------------|
| ChannelManager | Connects to Telegram, Discord, Slack, etc. |
| MessageBus | In-process queue that routes messages between channels and the agent |
| AgentLoop | Processes inbound messages with the configured LLM |
| CronService | Executes scheduled tasks |
| HeartbeatService | Periodic background tasks |
| AdminServer | Config web UI at `http://127.0.0.1:18791` (when `gateway.admin.enabled = true`) |

### Configuration

Config is read from `~/.bantu/config.json`.  The minimal configuration to start
the agent with an LLM provider:

```json
{
  "providers": {
    "anthropic": { "apiKey": "sk-ant-..." }
  },
  "agents": {
    "defaults": { "model": "anthropic/<model-id>" }
  }
}
```

Replace `anthropic/<model-id>` with a valid model identifier for your
provider (e.g. `anthropic/claude-opus-4-5`).  See the
[LiteLLM providers docs](https://docs.litellm.ai/docs/providers) for the full
list of supported model strings.

The gateway starts even if no API key is configured — it prints a warning and
waits for you to add one via the admin UI.

### Port options

```bash
nanobot gateway --port 8080       # change gateway port (default 18790)
nanobot gateway --verbose         # enable debug logging
```

---

## Distributed mode (three independent services)

When `gateway.services.agent_url` is set, the gateway switches to distributed
mode: it forwards channel messages to the remote agent service and proxies
admin-API requests to the remote admin service.

### Service responsibilities

| Service | Command | Default port | What it runs |
|---------|---------|--------------|-------------|
| Agent | `nanobot serve-agent` | 18792 | MessageBus + AgentLoop + CronService + HeartbeatService |
| Admin | `nanobot serve-admin` | 18791 | Admin REST API + web UI |
| Gateway | `nanobot gateway` | 18790 | ChannelManager + HTTP proxy to agent and admin |

### REST API exposed by each service

**Agent service** (`nanobot serve-agent`):

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/inbound` | Accept an inbound message from the gateway |
| `GET` | `/api/outbound` | Long-poll for outbound messages |
| `GET` | `/api/health` | Liveness probe |

**Admin service** (`nanobot serve-admin`):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/config` | Full (masked) configuration |
| `GET` | `/api/providers` | List LLM providers |
| `PUT` | `/api/providers/{name}` | Update a provider's API key |
| `GET` | `/api/channels` | List channels |
| `PUT` | `/api/channels/{name}` | Update a channel's config |
| `GET` | `/api/mcp` | List MCP servers |
| `POST/PUT/DELETE` | `/api/mcp/{name}` | Create / update / delete an MCP server |
| `GET` | `/api/agent` | Agent defaults |
| `PUT` | `/api/agent` | Update agent defaults |
| `GET` | `/api/health` | Liveness probe |

**Gateway service** (distributed mode, `nanobot gateway`):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `*` | `/api/admin/{path}` | Transparent proxy to the admin service |

### Telling the gateway where the other services are

Set the service URLs via environment variables or config:

**Environment variables** (recommended for Docker / CI):

```bash
export NANOBOT_GATEWAY__SERVICES__AGENT_URL=http://localhost:18792
export NANOBOT_GATEWAY__SERVICES__ADMIN_URL=http://localhost:18791
nanobot gateway
```

**config.json**:

```json
{
  "gateway": {
    "services": {
      "agentUrl": "http://localhost:18792",
      "adminUrl": "http://localhost:18791"
    }
  }
}
```

When `agent_url` is non-empty the gateway automatically enters distributed
mode.  When it is empty (the default) the gateway runs the embedded stack.

---

## Starting all three services locally (non-Docker)

Use the bundled `start.sh` script:

```bash
./start.sh
```

This starts all three services with their default ports and streams
colour-coded logs to the terminal.  Press **Ctrl+C** once to stop everything
cleanly.

### Custom ports

```bash
./start.sh --agent-port 19792 --admin-port 19791 --gateway-port 19790
```

### What start.sh does

1. Starts the agent service (`nanobot serve-agent --port 18792`)
2. Starts the admin service (`nanobot serve-admin --host 0.0.0.0 --port 18791`)
3. Waits up to 30 s for both health endpoints to respond
4. Starts the gateway, injecting `NANOBOT_GATEWAY__SERVICES__AGENT_URL` and
   `NANOBOT_GATEWAY__SERVICES__ADMIN_URL` so it enters distributed mode

---

## Running with Docker Compose

```bash
# Start all three services
docker compose up

# Start only one service
docker compose up nanobot-agent

# Run a CLI command against a running stack
docker compose run --rm nanobot-cli status
```

The `docker-compose.yml` at the repo root defines three services:

| Compose service | Image built from | Port |
|----------------|-----------------|------|
| `nanobot-agent` | `services/agent/Dockerfile` | 18792 |
| `nanobot-admin` | `services/admin/Dockerfile` | 18791 |
| `nanobot-gateway` | `services/gateway/Dockerfile` | 18790 |

All three services mount `~/.bantu` so they share the same config file.

The gateway container has these environment variables pre-configured to use
Docker's internal DNS:

```yaml
NANOBOT_GATEWAY__SERVICES__AGENT_URL: http://nanobot-agent:18792
NANOBOT_GATEWAY__SERVICES__ADMIN_URL: http://nanobot-admin:18791
```

### Building individual images

Each `Dockerfile` is self-contained.  Build and run any service independently:

```bash
# Agent
docker build -f services/agent/Dockerfile -t bantu-agent .
docker run -v ~/.bantu:/root/.bantu -p 18792:18792 bantu-agent

# Admin
docker build -f services/admin/Dockerfile -t bantu-admin .
docker run -v ~/.bantu:/root/.bantu -p 18791:18791 bantu-admin

# Gateway (point it at the other two)
docker build -f services/gateway/Dockerfile -t bantu-gateway .
docker run -v ~/.bantu:/root/.bantu -p 18790:18790 \
  -e NANOBOT_GATEWAY__SERVICES__AGENT_URL=http://host.docker.internal:18792 \
  -e NANOBOT_GATEWAY__SERVICES__ADMIN_URL=http://host.docker.internal:18791 \
  bantu-gateway
```

---

## Health checks

Every service exposes a health endpoint you can poll:

| Service | URL | Expected response |
|---------|-----|-------------------|
| Agent | `http://localhost:18792/api/health` | `{"status": "ok", "service": "agent"}` |
| Admin | `http://localhost:18791/api/health` | `{"status": "ok", "service": "admin"}` |
| Gateway (distributed) | `http://localhost:18790/health` | `{"status": "ok", "service": "gateway"}` |

```bash
curl http://localhost:18792/api/health
curl http://localhost:18791/api/health
curl http://localhost:18790/health
```

---

## Choosing between modes

| Situation | Recommended mode |
|-----------|-----------------|
| Local development or simple personal use | Embedded (`nanobot gateway`) |
| You want to update the agent without restarting the gateway | Distributed |
| Containerised deployment with separate resource limits per service | Distributed + Docker Compose |
| Backward compatibility with an existing single-process setup | Embedded (no config change needed) |

The embedded mode is the default: no environment variables need to be set, and
existing `config.json` files continue to work without modification.

---

## Switching between modes

**Embedded → Distributed**: set `gateway.services.agent_url` (via env var or
config) and start the agent and admin services separately.

**Distributed → Embedded**: unset `NANOBOT_GATEWAY__SERVICES__AGENT_URL` (or
remove `gateway.services.agentUrl` from config) and restart with
`nanobot gateway`.  The gateway falls back to the embedded stack automatically.

#!/usr/bin/env bash
# start.sh — Start all three Bantu services locally (non-Docker).
#
# Usage:
#   ./start.sh                  # start all services with defaults
#   ./start.sh --agent-port 18792 --admin-port 18791 --gateway-port 18790
#
# Each service writes its logs to the terminal with a coloured prefix.
# Press Ctrl+C once to stop all services cleanly.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
AGENT_PORT=18792
ADMIN_PORT=18791
GATEWAY_PORT=18790

# ── Parse optional CLI flags ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent-port)   AGENT_PORT="$2";   shift 2 ;;
    --admin-port)   ADMIN_PORT="$2";   shift 2 ;;
    --gateway-port) GATEWAY_PORT="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── Logging helpers ───────────────────────────────────────────────────────────
log() { printf '\033[1;37m[start.sh]\033[0m %s\n' "$*"; }
prefix_lines() {
  local tag="$1"
  while IFS= read -r line; do
    printf '%s %s\n' "$tag" "$line"
  done
}

# ── Cleanup on exit ───────────────────────────────────────────────────────────
PIDS=()
cleanup() {
  log "Stopping all Bantu services..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${PIDS[@]}" 2>/dev/null || true
  log "All services stopped."
}
trap cleanup INT TERM EXIT

# ── Start Agent service ───────────────────────────────────────────────────────
log "Starting agent service on port ${AGENT_PORT}..."
nanobot serve-agent --port "${AGENT_PORT}" 2>&1 \
  | prefix_lines $'\033[1;34m[agent]\033[0m' &
PIDS+=($!)

# ── Start Admin service ───────────────────────────────────────────────────────
log "Starting admin service on port ${ADMIN_PORT}..."
nanobot serve-admin --host 0.0.0.0 --port "${ADMIN_PORT}" 2>&1 \
  | prefix_lines $'\033[1;35m[admin]\033[0m' &
PIDS+=($!)

# ── Wait briefly for the other services to be ready ──────────────────────────
log "Waiting for agent and admin services to initialise..."
READY_TIMEOUT=30
ELAPSED=0
for svc_port in "${AGENT_PORT}" "${ADMIN_PORT}"; do
  while ! curl -sf "http://localhost:${svc_port}/api/health" >/dev/null 2>&1 && \
        [[ $ELAPSED -lt $READY_TIMEOUT ]]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
  done
  if [[ $ELAPSED -ge $READY_TIMEOUT ]]; then
    log "WARNING: service on port ${svc_port} did not become ready in ${READY_TIMEOUT}s; continuing anyway."
    ELAPSED=0
  fi
done

# ── Start Gateway service ─────────────────────────────────────────────────────
log "Starting gateway service on port ${GATEWAY_PORT}..."
NANOBOT_GATEWAY__SERVICES__AGENT_URL="http://localhost:${AGENT_PORT}" \
NANOBOT_GATEWAY__SERVICES__ADMIN_URL="http://localhost:${ADMIN_PORT}" \
nanobot gateway --port "${GATEWAY_PORT}" 2>&1 \
  | prefix_lines $'\033[1;32m[gateway]\033[0m' &
PIDS+=($!)

log "All services started.  Press Ctrl+C to stop."
log "  Agent:   http://localhost:${AGENT_PORT}/api/health"
log "  Admin:   http://localhost:${ADMIN_PORT}"
log "  Gateway: http://localhost:${GATEWAY_PORT}/health"

# ── Wait until a child exits (error) or we receive a signal ──────────────────
wait -n "${PIDS[@]}" 2>/dev/null || true

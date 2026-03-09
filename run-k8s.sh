#!/usr/bin/env bash
# run-k8s.sh — Deploy all Bantu services to a local Kubernetes cluster and
#              stream their logs with colour-coded prefixes.
#
# Supported local cluster tools:
#   minikube  (default)
#   kind
#
# Usage:
#   ./run-k8s.sh                    # build images, deploy, stream logs
#   ./run-k8s.sh --tool kind        # use kind instead of minikube
#   ./run-k8s.sh --no-build         # skip image build (use existing images)
#   ./run-k8s.sh --logs-only        # skip deploy, only stream logs
#   ./run-k8s.sh --bantu-home /custom/path   # override config dir (default: ~/.bantu)
#
# Press Ctrl+C once to stop log streaming.  The Kubernetes workloads keep
# running; use `kubectl delete namespace bantu` to tear everything down.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TOOL="minikube"
BANTU_HOME="${HOME}/.bantu"
NO_BUILD=false
LOGS_ONLY=false
NAMESPACE="bantu"
K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/k8s" && pwd)"

# ── Parse CLI flags ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool)        TOOL="$2";        shift 2 ;;
    --bantu-home)  BANTU_HOME="$2";  shift 2 ;;
    --no-build)    NO_BUILD=true;    shift   ;;
    --logs-only)   LOGS_ONLY=true;   shift   ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
  esac
done

# ── Logging helpers ────────────────────────────────────────────────────────────
BOLD='\033[1;37m'
RED='\033[1;31m'
RESET='\033[0m'
COLOR_AGENT='\033[1;34m'    # blue
COLOR_ADMIN='\033[1;35m'    # magenta
COLOR_GATEWAY='\033[1;32m'  # green

log()  { printf "${BOLD}[run-k8s.sh]${RESET} %s\n" "$*"; }
err()  { printf "${RED}[run-k8s.sh] ERROR:${RESET} %s\n" "$*" >&2; }

prefix_lines() {
  local tag="$1"
  while IFS= read -r line; do
    printf '%b %s\n' "$tag" "$line"
  done
}

# ── Prerequisite checks ────────────────────────────────────────────────────────
require_cmd() {
  if ! command -v "$1" &>/dev/null; then
    err "'$1' is required but not found in PATH."
    exit 1
  fi
}

require_cmd kubectl
require_cmd envsubst

case "$TOOL" in
  minikube) require_cmd minikube ;;
  kind)     require_cmd kind; require_cmd docker ;;
  *)
    err "Unknown --tool '$TOOL'. Supported values: minikube, kind."
    exit 1
    ;;
esac

# ── Ensure Bantu config directory exists ──────────────────────────────────────
mkdir -p "${BANTU_HOME}"
log "Using Bantu config directory: ${BANTU_HOME}"

# ── Helper: apply a manifest file after substituting ${BANTU_HOME} ────────────
apply_manifest() {
  local file="$1"
  BANTU_HOME="${BANTU_HOME}" envsubst '${BANTU_HOME}' < "${file}" | kubectl apply -f -
}

# ── Build Docker images ────────────────────────────────────────────────────────
build_images() {
  log "Building Docker images..."

  if [[ "$TOOL" == "minikube" ]]; then
    log "Pointing Docker CLI at minikube's daemon..."
    # shellcheck disable=SC2046
    eval $(minikube docker-env)
  fi

  log "Building bantu-agent:local ..."
  docker build \
    -f services/agent/Dockerfile \
    -t bantu-agent:local \
    . 2>&1 | prefix_lines "${COLOR_AGENT}[build:agent]${RESET}"

  log "Building bantu-admin:local ..."
  docker build \
    -f services/admin/Dockerfile \
    -t bantu-admin:local \
    . 2>&1 | prefix_lines "${COLOR_ADMIN}[build:admin]${RESET}"

  log "Building bantu-gateway:local ..."
  docker build \
    -f services/gateway/Dockerfile \
    -t bantu-gateway:local \
    . 2>&1 | prefix_lines "${COLOR_GATEWAY}[build:gateway]${RESET}"

  if [[ "$TOOL" == "kind" ]]; then
    log "Loading images into kind cluster..."
    kind load docker-image bantu-agent:local
    kind load docker-image bantu-admin:local
    kind load docker-image bantu-gateway:local
  fi
}

# ── Deploy manifests ───────────────────────────────────────────────────────────
deploy() {
  log "Applying Kubernetes manifests from ${K8S_DIR}/ ..."

  apply_manifest "${K8S_DIR}/namespace.yaml"

  # Agent
  apply_manifest "${K8S_DIR}/agent-deployment.yaml"
  apply_manifest "${K8S_DIR}/agent-service.yaml"

  # Admin
  apply_manifest "${K8S_DIR}/admin-deployment.yaml"
  apply_manifest "${K8S_DIR}/admin-service.yaml"

  # Gateway (depends on agent + admin being reachable)
  apply_manifest "${K8S_DIR}/gateway-deployment.yaml"
  apply_manifest "${K8S_DIR}/gateway-service.yaml"

  log "All manifests applied."
}

# ── Wait for deployments to become ready ──────────────────────────────────────
wait_ready() {
  log "Waiting for deployments to become ready (timeout: 120s)..."
  for deployment in nanobot-agent nanobot-admin nanobot-gateway; do
    log "  Waiting for ${deployment}..."
    kubectl rollout status deployment/"${deployment}" \
      --namespace="${NAMESPACE}" \
      --timeout=120s
  done
  log "All deployments are ready."
}

# ── Stream logs from all pods ──────────────────────────────────────────────────
stream_logs() {
  log "Streaming logs from all Bantu pods (Ctrl+C to stop)..."
  log "  Agent:   http://localhost:18792/api/health"
  if [[ "$TOOL" == "minikube" ]]; then
    local gw_url
    gw_url=$(minikube service nanobot-gateway --namespace="${NAMESPACE}" --url 2>/dev/null || true)
    if [[ -n "$gw_url" ]]; then
      log "  Gateway: ${gw_url}/health"
    else
      log "  Gateway: run 'minikube service nanobot-gateway -n bantu --url' for the URL"
    fi
  else
    log "  Gateway: http://localhost:30790/health  (NodePort)"
  fi
  log ""

  PIDS=()

  cleanup() {
    for pid in "${PIDS[@]:-}"; do
      kill "$pid" 2>/dev/null || true
    done
    wait "${PIDS[@]:-}" 2>/dev/null || true
    log "Log streaming stopped."
  }
  trap cleanup INT TERM EXIT

  kubectl logs \
    --namespace="${NAMESPACE}" \
    --selector=app=nanobot-agent \
    --follow \
    --prefix \
    --all-containers \
    2>&1 | prefix_lines "${COLOR_AGENT}[agent]${RESET}" &
  PIDS+=($!)

  kubectl logs \
    --namespace="${NAMESPACE}" \
    --selector=app=nanobot-admin \
    --follow \
    --prefix \
    --all-containers \
    2>&1 | prefix_lines "${COLOR_ADMIN}[admin]${RESET}" &
  PIDS+=($!)

  kubectl logs \
    --namespace="${NAMESPACE}" \
    --selector=app=nanobot-gateway \
    --follow \
    --prefix \
    --all-containers \
    2>&1 | prefix_lines "${COLOR_GATEWAY}[gateway]${RESET}" &
  PIDS+=($!)

  # Wait until a log streamer exits (pod crash) or we get a signal.
  wait -n "${PIDS[@]}" 2>/dev/null || true
}

# ── Main ───────────────────────────────────────────────────────────────────────
if [[ "$LOGS_ONLY" == false ]]; then
  if [[ "$NO_BUILD" == false ]]; then
    build_images
  fi
  deploy
  wait_ready
fi

stream_logs

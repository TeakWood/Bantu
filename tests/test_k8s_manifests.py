"""Tests for the Kubernetes workload manifests and the run-k8s.sh runner script.

Covers:
- All seven YAML manifests exist and are valid YAML
- Namespace manifest correctness
- Deployment manifests: required fields, ports, probes, resources, volumes
- Service manifests: type, port numbers, selectors
- Gateway deployment has the two distributed-mode environment variables set
- run-k8s.sh is present, executable, and contains the critical behavioural sections
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest

try:
    import yaml  # PyYAML is pulled in transitively by many packages
    HAS_YAML = True
except ImportError:  # pragma: no cover
    HAS_YAML = False

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
K8S_DIR = REPO_ROOT / "k8s"
RUNNER = REPO_ROOT / "run-k8s.sh"

MANIFEST_FILES = [
    "namespace.yaml",
    "agent-deployment.yaml",
    "agent-service.yaml",
    "admin-deployment.yaml",
    "admin-service.yaml",
    "gateway-deployment.yaml",
    "gateway-service.yaml",
]


def _load_yaml(name: str) -> Any:
    """Load a YAML file from the k8s/ directory."""
    path = K8S_DIR / name
    with path.open() as fh:
        # Use safe_load; manifests contain ${BANTU_HOME} placeholders which
        # are plain strings — no special YAML handling needed.
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Existence and parse-ability
# ─────────────────────────────────────────────────────────────────────────────


def test_k8s_directory_exists():
    assert K8S_DIR.is_dir(), "k8s/ directory is missing"


@pytest.mark.parametrize("filename", MANIFEST_FILES)
def test_manifest_file_exists(filename: str):
    assert (K8S_DIR / filename).is_file(), f"k8s/{filename} is missing"


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
@pytest.mark.parametrize("filename", MANIFEST_FILES)
def test_manifest_is_valid_yaml(filename: str):
    doc = _load_yaml(filename)
    assert doc is not None, f"{filename} is empty or not valid YAML"


# ─────────────────────────────────────────────────────────────────────────────
# Namespace manifest
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
def test_namespace_kind_and_name():
    doc = _load_yaml("namespace.yaml")
    assert doc["kind"] == "Namespace"
    assert doc["metadata"]["name"] == "bantu"
    assert doc["apiVersion"] == "v1"


# ─────────────────────────────────────────────────────────────────────────────
# Deployment manifests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
@pytest.mark.parametrize(
    "filename, app_name, port",
    [
        ("agent-deployment.yaml", "nanobot-agent", 18792),
        ("admin-deployment.yaml", "nanobot-admin", 18791),
        ("gateway-deployment.yaml", "nanobot-gateway", 18790),
    ],
)
def test_deployment_basic_fields(filename: str, app_name: str, port: int):
    doc = _load_yaml(filename)
    assert doc["kind"] == "Deployment"
    assert doc["apiVersion"] == "apps/v1"
    assert doc["metadata"]["name"] == app_name
    assert doc["metadata"]["namespace"] == "bantu"

    spec = doc["spec"]
    assert spec["replicas"] == 1

    # selector must match template labels
    selector_label = spec["selector"]["matchLabels"]["app"]
    template_label = spec["template"]["metadata"]["labels"]["app"]
    assert selector_label == app_name
    assert template_label == app_name

    # container port
    containers = spec["template"]["spec"]["containers"]
    assert len(containers) == 1
    container_ports = [p["containerPort"] for p in containers[0]["ports"]]
    assert port in container_ports


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
@pytest.mark.parametrize(
    "filename, probe_path, port",
    [
        ("agent-deployment.yaml", "/api/health", 18792),
        ("admin-deployment.yaml", "/api/health", 18791),
        ("gateway-deployment.yaml", "/health", 18790),
    ],
)
def test_deployment_has_liveness_probe(filename: str, probe_path: str, port: int):
    doc = _load_yaml(filename)
    container = doc["spec"]["template"]["spec"]["containers"][0]
    probe = container["livenessProbe"]
    assert probe["httpGet"]["path"] == probe_path
    assert probe["httpGet"]["port"] == port


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
@pytest.mark.parametrize(
    "filename",
    ["agent-deployment.yaml", "admin-deployment.yaml", "gateway-deployment.yaml"],
)
def test_deployment_has_readiness_probe(filename: str):
    doc = _load_yaml(filename)
    container = doc["spec"]["template"]["spec"]["containers"][0]
    assert "readinessProbe" in container


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
@pytest.mark.parametrize(
    "filename",
    ["agent-deployment.yaml", "admin-deployment.yaml", "gateway-deployment.yaml"],
)
def test_deployment_has_resource_limits(filename: str):
    doc = _load_yaml(filename)
    container = doc["spec"]["template"]["spec"]["containers"][0]
    resources = container["resources"]
    assert "limits" in resources
    assert "requests" in resources


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
@pytest.mark.parametrize(
    "filename",
    ["agent-deployment.yaml", "admin-deployment.yaml", "gateway-deployment.yaml"],
)
def test_deployment_mounts_bantu_config_volume(filename: str):
    doc = _load_yaml(filename)
    pod_spec = doc["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    volume_names = [v["name"] for v in pod_spec["volumes"]]
    assert "bantu-config" in volume_names, "bantu-config volume not declared"

    mount_paths = [m["mountPath"] for m in container["volumeMounts"]]
    assert "/root/.bantu" in mount_paths, "/root/.bantu not mounted"


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
@pytest.mark.parametrize(
    "filename",
    ["agent-deployment.yaml", "admin-deployment.yaml", "gateway-deployment.yaml"],
)
def test_deployment_uses_hostpath_with_bantu_home_placeholder(filename: str):
    doc = _load_yaml(filename)
    pod_spec = doc["spec"]["template"]["spec"]
    volumes = {v["name"]: v for v in pod_spec["volumes"]}
    config_vol = volumes["bantu-config"]
    assert "hostPath" in config_vol
    # The placeholder must appear in the raw file text.
    raw = (K8S_DIR / filename).read_text()
    assert "${BANTU_HOME}" in raw, "BANTU_HOME placeholder missing from manifest"


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
@pytest.mark.parametrize(
    "filename",
    ["agent-deployment.yaml", "admin-deployment.yaml", "gateway-deployment.yaml"],
)
def test_deployment_image_pull_policy_never(filename: str):
    """All local deployments must use imagePullPolicy: Never."""
    doc = _load_yaml(filename)
    container = doc["spec"]["template"]["spec"]["containers"][0]
    assert container["imagePullPolicy"] == "Never"


# ─────────────────────────────────────────────────────────────────────────────
# Service manifests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
@pytest.mark.parametrize(
    "filename, app_name, port, svc_type",
    [
        ("agent-service.yaml", "nanobot-agent", 18792, "ClusterIP"),
        ("admin-service.yaml", "nanobot-admin", 18791, "ClusterIP"),
        ("gateway-service.yaml", "nanobot-gateway", 18790, "NodePort"),
    ],
)
def test_service_basic_fields(filename: str, app_name: str, port: int, svc_type: str):
    doc = _load_yaml(filename)
    assert doc["kind"] == "Service"
    assert doc["apiVersion"] == "v1"
    assert doc["metadata"]["name"] == app_name
    assert doc["metadata"]["namespace"] == "bantu"

    spec = doc["spec"]
    assert spec["type"] == svc_type
    assert spec["selector"]["app"] == app_name

    ports = spec["ports"]
    assert len(ports) >= 1
    assert ports[0]["port"] == port
    assert ports[0]["targetPort"] == port


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
def test_gateway_service_has_nodeport():
    doc = _load_yaml("gateway-service.yaml")
    ports = doc["spec"]["ports"]
    node_ports = [p["nodePort"] for p in ports if "nodePort" in p]
    assert node_ports, "gateway-service.yaml must define a nodePort"
    assert 30000 <= node_ports[0] <= 32767, "nodePort must be in 30000-32767 range"


# ─────────────────────────────────────────────────────────────────────────────
# Gateway distributed-mode environment variables
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
def test_gateway_deployment_has_agent_url_env():
    doc = _load_yaml("gateway-deployment.yaml")
    container = doc["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container.get("env", [])}
    assert "NANOBOT_GATEWAY__SERVICES__AGENT_URL" in env
    assert env["NANOBOT_GATEWAY__SERVICES__AGENT_URL"] == "http://nanobot-agent:18792"


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
def test_gateway_deployment_has_admin_url_env():
    doc = _load_yaml("gateway-deployment.yaml")
    container = doc["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container.get("env", [])}
    assert "NANOBOT_GATEWAY__SERVICES__ADMIN_URL" in env
    assert env["NANOBOT_GATEWAY__SERVICES__ADMIN_URL"] == "http://nanobot-admin:18791"


# ─────────────────────────────────────────────────────────────────────────────
# run-k8s.sh runner script
# ─────────────────────────────────────────────────────────────────────────────


def test_runner_script_exists():
    assert RUNNER.is_file(), "run-k8s.sh is missing from the repository root"


def test_runner_script_is_executable():
    mode = RUNNER.stat().st_mode
    assert mode & stat.S_IXUSR, "run-k8s.sh is not executable (missing user execute bit)"


def test_runner_script_has_shebang():
    first_line = RUNNER.read_text().splitlines()[0]
    assert first_line.startswith("#!/"), "run-k8s.sh must start with a shebang line"


@pytest.mark.parametrize(
    "expected",
    [
        # CLI flags
        "--tool",
        "--no-build",
        "--logs-only",
        "--bantu-home",
        # Core operations
        "build_images",
        "deploy",
        "wait_ready",
        "stream_logs",
        # Image names must match Dockerfile tags
        "bantu-agent:local",
        "bantu-admin:local",
        "bantu-gateway:local",
        # minikube support
        "minikube",
        # kind support
        "kind",
        # envsubst for BANTU_HOME substitution
        "envsubst",
        # Namespace
        "bantu",
        # Log colour prefixes
        "[agent]",
        "[admin]",
        "[gateway]",
        # kubectl log streaming
        "kubectl logs",
    ],
)
def test_runner_script_contains_section(expected: str):
    content = RUNNER.read_text()
    assert expected in content, f"run-k8s.sh is missing expected content: {expected!r}"


def test_runner_script_streams_all_three_services():
    """All three selector labels must appear in the kubectl logs commands."""
    content = RUNNER.read_text()
    for label in ("app=nanobot-agent", "app=nanobot-admin", "app=nanobot-gateway"):
        assert label in content, f"run-k8s.sh missing log selector for {label}"


def test_runner_script_builds_from_correct_dockerfiles():
    content = RUNNER.read_text()
    assert "services/agent/Dockerfile" in content
    assert "services/admin/Dockerfile" in content
    assert "services/gateway/Dockerfile" in content


# ─────────────────────────────────────────────────────────────────────────────
# Cross-consistency: service names match across deployment env vars and services
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
def test_gateway_env_agent_url_matches_agent_service_name_and_port():
    """The agent URL in the gateway deployment must match the agent service."""
    gw = _load_yaml("gateway-deployment.yaml")
    svc = _load_yaml("agent-service.yaml")

    container = gw["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container.get("env", [])}
    agent_url = env.get("NANOBOT_GATEWAY__SERVICES__AGENT_URL", "")

    svc_name = svc["metadata"]["name"]
    svc_port = svc["spec"]["ports"][0]["port"]

    assert svc_name in agent_url
    assert str(svc_port) in agent_url


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
def test_gateway_env_admin_url_matches_admin_service_name_and_port():
    """The admin URL in the gateway deployment must match the admin service."""
    gw = _load_yaml("gateway-deployment.yaml")
    svc = _load_yaml("admin-service.yaml")

    container = gw["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container.get("env", [])}
    admin_url = env.get("NANOBOT_GATEWAY__SERVICES__ADMIN_URL", "")

    svc_name = svc["metadata"]["name"]
    svc_port = svc["spec"]["ports"][0]["port"]

    assert svc_name in admin_url
    assert str(svc_port) in admin_url

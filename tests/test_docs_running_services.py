"""Tests verifying the running-services documentation.

These tests check two things:
1. The documentation file exists and contains required sections.
2. The runtime behaviour described in the docs matches the actual code
   (default ports, env-var configuration, embedded vs. distributed mode).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import inspect

from nanobot.config.schema import Config, GatewayConfig, ServicesConfig
from nanobot.services.agent_server import AgentRestServer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent.parent / "docs"
RUNNING_SERVICES_DOC = DOCS_DIR / "running-services.md"


# ---------------------------------------------------------------------------
# Documentation file structure
# ---------------------------------------------------------------------------


class TestDocumentationFile:
    """The running-services.md file must exist and cover all required topics."""

    def test_file_exists(self) -> None:
        assert RUNNING_SERVICES_DOC.exists(), (
            f"Expected documentation file at {RUNNING_SERVICES_DOC}"
        )

    def test_file_is_not_empty(self) -> None:
        content = RUNNING_SERVICES_DOC.read_text()
        assert len(content.strip()) > 0

    @pytest.mark.parametrize(
        "section",
        [
            # Mode headings
            "Embedded",
            "Distributed",
            # Commands that must be documented
            "nanobot gateway",
            "serve-agent",
            "serve-admin",
            # Deployment options
            "Docker Compose",
            "start.sh",
            # Configuration method
            "NANOBOT_GATEWAY__SERVICES__AGENT_URL",
            "NANOBOT_GATEWAY__SERVICES__ADMIN_URL",
            # Health check paths
            "/api/health",
            "/health",
        ],
    )
    def test_required_section_present(self, section: str) -> None:
        content = RUNNING_SERVICES_DOC.read_text()
        assert section in content, (
            f"Documentation must mention '{section}' but it was not found."
        )

    def test_default_ports_mentioned(self) -> None:
        content = RUNNING_SERVICES_DOC.read_text()
        for port in ("18790", "18791", "18792"):
            assert port in content, (
                f"Default port {port} must be documented."
            )

    def test_embedded_mode_described(self) -> None:
        content = RUNNING_SERVICES_DOC.read_text()
        assert "single" in content.lower() or "embedded" in content.lower(), (
            "Documentation must describe the single-process (embedded) mode."
        )

    def test_backward_compatibility_mentioned(self) -> None:
        content = RUNNING_SERVICES_DOC.read_text()
        lower = content.lower()
        assert "backward" in lower or "back" in lower or "compatible" in lower, (
            "Documentation must mention backward compatibility."
        )


# ---------------------------------------------------------------------------
# Embedded mode — default behaviour matches the docs
# ---------------------------------------------------------------------------


class TestEmbeddedModeDefaults:
    """ServicesConfig defaults must match what the docs say about embedded mode."""

    def test_services_config_defaults_to_embedded_mode(self) -> None:
        """Empty agent_url means embedded mode — no extra config required."""
        cfg = Config()
        assert cfg.gateway.services.agent_url == "", (
            "Default agent_url must be empty so the gateway starts in embedded mode."
        )
        assert cfg.gateway.services.admin_url == "", (
            "Default admin_url must be empty so the gateway starts in embedded mode."
        )

    def test_gateway_default_port_is_18790(self) -> None:
        cfg = Config()
        assert cfg.gateway.port == 18790

    def test_agent_service_default_port_is_18792(self) -> None:
        """AgentRestServer binds to port 18792 by default, as documented."""
        sig = inspect.signature(AgentRestServer.__init__)
        default_port = sig.parameters["port"].default
        assert default_port == 18792, (
            f"AgentRestServer default port must be 18792 (got {default_port})."
        )

    def test_admin_config_default_port_is_18791(self) -> None:
        cfg = Config()
        assert cfg.gateway.admin.port == 18791

    def test_admin_config_default_host_is_localhost(self) -> None:
        """Docs say admin binds to 127.0.0.1 by default for security."""
        cfg = Config()
        assert cfg.gateway.admin.host == "127.0.0.1"


# ---------------------------------------------------------------------------
# Distributed mode — env-var configuration matches the docs
# ---------------------------------------------------------------------------


class TestDistributedModeConfiguration:
    """Env-var configuration described in the docs works at runtime."""

    def test_agent_url_set_via_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NANOBOT_GATEWAY__SERVICES__AGENT_URL activates distributed mode."""
        monkeypatch.setenv(
            "NANOBOT_GATEWAY__SERVICES__AGENT_URL", "http://agent-host:18792"
        )
        cfg = Config()
        assert cfg.gateway.services.agent_url == "http://agent-host:18792"

    def test_admin_url_set_via_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NANOBOT_GATEWAY__SERVICES__ADMIN_URL configures the admin proxy."""
        monkeypatch.setenv(
            "NANOBOT_GATEWAY__SERVICES__ADMIN_URL", "http://admin-host:18791"
        )
        cfg = Config()
        assert cfg.gateway.services.admin_url == "http://admin-host:18791"

    def test_both_urls_set_together(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting both URLs simultaneously works as documented."""
        monkeypatch.setenv(
            "NANOBOT_GATEWAY__SERVICES__AGENT_URL", "http://localhost:18792"
        )
        monkeypatch.setenv(
            "NANOBOT_GATEWAY__SERVICES__ADMIN_URL", "http://localhost:18791"
        )
        cfg = Config()
        assert cfg.gateway.services.agent_url == "http://localhost:18792"
        assert cfg.gateway.services.admin_url == "http://localhost:18791"

    def test_services_config_json_field_name(self) -> None:
        """The config.json fields are agentUrl / adminUrl (camelCase)."""
        raw = {
            "agentUrl": "http://remote:18792",
            "adminUrl": "http://remote:18791",
        }
        svc = ServicesConfig.model_validate(raw)
        assert svc.agent_url == "http://remote:18792"
        assert svc.admin_url == "http://remote:18791"

    def test_gateway_config_round_trips_services(self) -> None:
        """GatewayConfig correctly nests ServicesConfig under 'services'."""
        raw = {
            "services": {
                "agentUrl": "http://agent:18792",
                "adminUrl": "http://admin:18791",
            }
        }
        gw = GatewayConfig.model_validate(raw)
        assert gw.services.agent_url == "http://agent:18792"
        assert gw.services.admin_url == "http://admin:18791"

    def test_switching_back_to_embedded_by_clearing_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Removing agent_url reverts to embedded mode (empty string)."""
        # First set, then clear — simulates the switch described in the docs.
        monkeypatch.setenv(
            "NANOBOT_GATEWAY__SERVICES__AGENT_URL", "http://localhost:18792"
        )
        cfg_distributed = Config()
        assert cfg_distributed.gateway.services.agent_url != ""

        monkeypatch.delenv("NANOBOT_GATEWAY__SERVICES__AGENT_URL")
        cfg_embedded = Config()
        assert cfg_embedded.gateway.services.agent_url == "", (
            "After removing the env var the gateway must fall back to embedded mode."
        )

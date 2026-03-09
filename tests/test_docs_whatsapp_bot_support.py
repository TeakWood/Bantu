"""Tests verifying the WhatsApp bot-support documentation.

These tests check two things:
1. The documentation file exists and contains every required section / term.
2. The runtime behaviour described in the docs (config schema, field names,
   default values) matches the actual code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.config.schema import ChannelsConfig, WhatsAppConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent.parent / "docs"
WHATSAPP_DOC = DOCS_DIR / "whatsapp-bot-support.md"


# ---------------------------------------------------------------------------
# Documentation file structure
# ---------------------------------------------------------------------------


class TestDocumentationFile:
    """The whatsapp-bot-support.md file must exist and cover all required topics."""

    def test_file_exists(self) -> None:
        assert WHATSAPP_DOC.exists(), (
            f"Expected documentation file at {WHATSAPP_DOC}"
        )

    def test_file_is_not_empty(self) -> None:
        content = WHATSAPP_DOC.read_text()
        assert len(content.strip()) > 0

    @pytest.mark.parametrize(
        "term",
        [
            # Evaluation outcome
            "WhatsApp",
            "bot",
            # Two distinct pathways
            "Cloud API",
            "Baileys",
            # Setup steps
            "QR",
            "Linked Device",
            # Config fields documented
            "allowFrom",
            "bridgeUrl",
            "bridgeToken",
            # Required content areas
            "Limitation",
            "Comparison",
            "Recommendation",
            # Official route concepts
            "webhook",
            "template",
            "Business",
        ],
    )
    def test_required_term_present(self, term: str) -> None:
        content = WHATSAPP_DOC.read_text()
        assert term in content, (
            f"Documentation must mention '{term}' but it was not found."
        )

    def test_official_api_section_present(self) -> None:
        content = WHATSAPP_DOC.read_text()
        lower = content.lower()
        assert "cloud api" in lower or "business platform" in lower, (
            "Documentation must describe the official WhatsApp Cloud API / "
            "Business Platform."
        )

    def test_unofficial_baileys_section_present(self) -> None:
        content = WHATSAPP_DOC.read_text()
        assert "Baileys" in content, (
            "Documentation must describe the Baileys-based approach used by Bantu."
        )

    def test_registration_steps_documented(self) -> None:
        content = WHATSAPP_DOC.read_text()
        lower = content.lower()
        assert "step" in lower or "register" in lower, (
            "Documentation must include registration / setup steps."
        )

    def test_both_pathways_compared(self) -> None:
        content = WHATSAPP_DOC.read_text()
        lower = content.lower()
        # Comparison section must contain both labels
        assert "official" in lower and "baileys" in lower, (
            "Documentation must compare the official Cloud API and Baileys approaches."
        )

    def test_recommendation_present(self) -> None:
        content = WHATSAPP_DOC.read_text()
        lower = content.lower()
        assert "recommendation" in lower or "recommend" in lower, (
            "Documentation must include a recommendation on which approach to use."
        )

    def test_terms_of_service_risk_mentioned(self) -> None:
        """Docs must warn about the ToS risk of the unofficial approach."""
        content = WHATSAPP_DOC.read_text()
        lower = content.lower()
        assert "terms of service" in lower or "tos" in lower, (
            "Documentation must mention the Terms of Service risk of using Baileys."
        )

    def test_default_bridge_url_documented(self) -> None:
        """The documented default bridge URL must match WhatsAppConfig.bridge_url."""
        content = WHATSAPP_DOC.read_text()
        expected = WhatsAppConfig().bridge_url  # ws://localhost:3001
        assert expected in content, (
            f"Documentation must state the default bridgeUrl as '{expected}'."
        )


# ---------------------------------------------------------------------------
# WhatsAppConfig schema — defaults must match what the docs describe
# ---------------------------------------------------------------------------


class TestWhatsAppConfigDefaults:
    """WhatsAppConfig defaults must be consistent with the documented values."""

    def test_whatsapp_disabled_by_default(self) -> None:
        cfg = WhatsAppConfig()
        assert cfg.enabled is False, "WhatsApp must be disabled by default."

    def test_default_bridge_url(self) -> None:
        cfg = WhatsAppConfig()
        assert cfg.bridge_url == "ws://localhost:3001", (
            "Default bridge URL must be ws://localhost:3001 as documented."
        )

    def test_default_bridge_token_is_empty(self) -> None:
        cfg = WhatsAppConfig()
        assert cfg.bridge_token == "", "Default bridge token must be empty string."

    def test_default_allow_from_is_empty_list(self) -> None:
        cfg = WhatsAppConfig()
        assert cfg.allow_from == [], (
            "Default allow_from must be an empty list (anyone can message)."
        )

    def test_allow_from_accepts_phone_numbers(self) -> None:
        cfg = WhatsAppConfig(allow_from=["+1234567890", "+0987654321"])
        assert len(cfg.allow_from) == 2
        assert "+1234567890" in cfg.allow_from
        assert "+0987654321" in cfg.allow_from

    def test_camel_case_field_names_accepted(self) -> None:
        """config.json uses camelCase; the schema must accept it."""
        data = {
            "enabled": True,
            "bridgeUrl": "ws://remote-host:3001",
            "bridgeToken": "shared-secret",
            "allowFrom": ["+15550001111"],
        }
        cfg = WhatsAppConfig.model_validate(data)
        assert cfg.enabled is True
        assert cfg.bridge_url == "ws://remote-host:3001"
        assert cfg.bridge_token == "shared-secret"
        assert cfg.allow_from == ["+15550001111"]

    def test_snake_case_field_names_accepted(self) -> None:
        """snake_case names must also be accepted (programmatic construction)."""
        data = {
            "enabled": False,
            "bridge_url": "ws://127.0.0.1:3001",
            "bridge_token": "",
            "allow_from": [],
        }
        cfg = WhatsAppConfig.model_validate(data)
        assert cfg.bridge_url == "ws://127.0.0.1:3001"

    def test_whatsapp_nested_in_channels_config(self) -> None:
        """WhatsAppConfig must be accessible via ChannelsConfig.whatsapp."""
        channels = ChannelsConfig()
        assert hasattr(channels, "whatsapp"), (
            "ChannelsConfig must expose a 'whatsapp' attribute."
        )
        assert isinstance(channels.whatsapp, WhatsAppConfig)

    def test_enable_whatsapp_via_channels_config(self) -> None:
        """Enabling WhatsApp through nested ChannelsConfig must work."""
        channels = ChannelsConfig.model_validate(
            {"whatsapp": {"enabled": True, "allowFrom": ["+1"]}}
        )
        assert channels.whatsapp.enabled is True
        assert channels.whatsapp.allow_from == ["+1"]

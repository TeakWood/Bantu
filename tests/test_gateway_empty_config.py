"""Tests for gateway command behaviour with empty / unconfigured config.

Acceptance criterion (Bantu-msf):
  `uv run gateway` should not exit with error if config.json is not configured.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from nanobot.config.loader import load_config
from nanobot.config.schema import Config
from nanobot.providers.null_provider import _NOT_CONFIGURED_MESSAGE, NullProvider

# ---------------------------------------------------------------------------
# NullProvider unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_null_provider_chat_returns_not_configured_message():
    provider = NullProvider()
    response = await provider.chat(messages=[{"role": "user", "content": "hello"}])
    assert response.content == _NOT_CONFIGURED_MESSAGE
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_null_provider_chat_ignores_tools_and_model():
    """NullProvider must not crash when tools or model are provided."""
    provider = NullProvider()
    response = await provider.chat(
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"name": "some_tool"}],
        model="gpt-4",
        max_tokens=100,
        temperature=0.5,
    )
    assert response.content == _NOT_CONFIGURED_MESSAGE


def test_null_provider_get_default_model():
    provider = NullProvider()
    assert provider.get_default_model() == "not-configured"


# ---------------------------------------------------------------------------
# _make_provider behaviour when config has no API key
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# load_config auto-create and no-overwrite tests
# ---------------------------------------------------------------------------


def test_load_config_creates_file_when_missing(tmp_path: Path):
    """load_config writes a default config file when the file does not exist."""
    config_path = tmp_path / "config.json"
    assert not config_path.exists()

    config = load_config(config_path)

    assert config_path.exists(), "load_config must create the config file when absent"
    with open(config_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert isinstance(saved, dict)
    assert isinstance(config, Config)


def test_load_config_does_not_overwrite_corrupt_file(tmp_path: Path):
    """load_config returns an in-memory default but must NOT overwrite a corrupt file."""
    config_path = tmp_path / "config.json"
    corrupt_content = "{not valid json"
    config_path.write_text(corrupt_content, encoding="utf-8")

    config = load_config(config_path)

    assert config_path.read_text(encoding="utf-8") == corrupt_content, (
        "load_config must not overwrite a corrupt config file"
    )
    assert isinstance(config, Config)


def test_make_provider_returns_none_when_raise_on_missing_false():
    """With raise_on_missing=False and no key, _make_provider returns None."""
    from nanobot.cli.commands import _make_provider

    config = Config()  # default config: no providers set up

    result = _make_provider(config, raise_on_missing=False)

    # None signals "no provider configured"; gateway wraps it in NullProvider
    assert result is None


def test_make_provider_raises_on_missing_key_by_default():
    """Default behaviour (raise_on_missing=True) still exits on missing key."""
    from nanobot.cli.commands import _make_provider

    config = Config()

    with pytest.raises(typer.Exit):
        _make_provider(config, raise_on_missing=True)


# ---------------------------------------------------------------------------
# Gateway command integration: does NOT exit when config is empty.
#
# The gateway function uses local imports, so we patch at the source modules.
# ---------------------------------------------------------------------------

_GATEWAY_PATCHES = [
    ("nanobot.config.loader.load_config", {}),
    ("nanobot.cli.commands.sync_workspace_templates", {}),
    ("nanobot.bus.queue.MessageBus", {}),
    ("nanobot.session.manager.SessionManager", {}),
    ("nanobot.cron.service.CronService", {}),
    ("nanobot.agent.loop.AgentLoop", {}),
    ("nanobot.channels.manager.ChannelManager", {}),
    ("nanobot.heartbeat.service.HeartbeatService", {}),
    ("nanobot.config.loader.get_data_dir", {}),
    ("asyncio.run", {}),
]


def _make_gateway_mocks(captured_provider: list | None = None):
    """Return a context manager that patches all gateway dependencies."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        empty_config = Config()

        def _agent_factory(*args, **kwargs):
            if captured_provider is not None:
                captured_provider.append(kwargs.get("provider"))
            m = MagicMock()
            m.model = "not-configured"
            return m

        mock_cron = MagicMock()
        mock_cron.status.return_value = {"jobs": 0}

        mock_channels = MagicMock()
        mock_channels.enabled_channels = []

        with (
            patch("nanobot.config.loader.load_config", return_value=empty_config),
            patch("nanobot.cli.commands.sync_workspace_templates"),
            patch("nanobot.bus.queue.MessageBus"),
            patch("nanobot.session.manager.SessionManager"),
            patch("nanobot.cron.service.CronService", return_value=mock_cron),
            patch("nanobot.agent.loop.AgentLoop", side_effect=_agent_factory),
            patch("nanobot.channels.manager.ChannelManager", return_value=mock_channels),
            patch("nanobot.heartbeat.service.HeartbeatService", return_value=MagicMock()),
            patch("nanobot.config.loader.get_data_dir", return_value=MagicMock()),
            patch("asyncio.run", side_effect=lambda coro: coro.close()),
        ):
            yield

    return _ctx()


def test_gateway_does_not_exit_on_empty_config():
    """The gateway command must not exit with a non-zero code when config.json
    has no API key configured (Bantu-msf acceptance criterion)."""
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    runner = CliRunner()

    with _make_gateway_mocks():
        result = runner.invoke(app, ["gateway"])

    assert result.exit_code == 0, (
        f"gateway exited with code {result.exit_code}.\nOutput:\n{result.output}"
    )


def test_gateway_prints_warning_on_empty_config():
    """When no API key is configured the gateway prints a helpful warning."""
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    runner = CliRunner()

    with _make_gateway_mocks():
        result = runner.invoke(app, ["gateway"])

    assert "No API key configured" in result.output
    assert "nanobot onboard" in result.output


def test_gateway_uses_null_provider_when_config_empty():
    """The AgentLoop must be constructed with a NullProvider when no key is set."""
    from typer.testing import CliRunner

    from nanobot.cli.commands import app

    runner = CliRunner()
    captured: list = []

    with _make_gateway_mocks(captured_provider=captured):
        runner.invoke(app, ["gateway"])

    assert len(captured) == 1
    assert isinstance(captured[0], NullProvider)

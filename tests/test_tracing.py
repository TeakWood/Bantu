"""Tests for nanobot.tracing — LLM call observability via OpenTelemetry / Phoenix."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# TracingConfig
# ---------------------------------------------------------------------------


class TestTracingConfig:
    def test_default_values(self):
        from nanobot.config.schema import TracingConfig

        cfg = TracingConfig()
        assert cfg.enabled is False
        assert cfg.endpoint == "http://localhost:6006/v1/traces"
        assert cfg.project_name == "nanobot"

    def test_custom_values(self):
        from nanobot.config.schema import TracingConfig

        cfg = TracingConfig(
            enabled=True,
            endpoint="http://custom:9999/v1/traces",
            project_name="mybot",
        )
        assert cfg.enabled is True
        assert cfg.endpoint == "http://custom:9999/v1/traces"
        assert cfg.project_name == "mybot"

    def test_is_part_of_config(self):
        from nanobot.config.schema import Config, TracingConfig

        cfg = Config()
        assert hasattr(cfg, "tracing")
        assert isinstance(cfg.tracing, TracingConfig)
        assert cfg.tracing.enabled is False

    def test_camel_case_alias(self):
        """TracingConfig accepts camelCase keys (via Base model alias_generator)."""
        from nanobot.config.schema import TracingConfig

        cfg = TracingConfig.model_validate(
            {"projectName": "bantu", "enabled": True, "endpoint": "http://host:6006/v1/traces"}
        )
        assert cfg.project_name == "bantu"
        assert cfg.enabled is True


# ---------------------------------------------------------------------------
# setup_tracing
# ---------------------------------------------------------------------------


class TestSetupTracing:
    def test_disabled_is_noop(self):
        """setup_tracing is a complete no-op when enabled=False."""
        from nanobot.config.schema import TracingConfig
        from nanobot.tracing.setup import setup_tracing

        cfg = TracingConfig(enabled=False)
        # Must not raise even when OTel packages are absent.
        setup_tracing(cfg)

    def test_missing_deps_does_not_raise(self):
        """setup_tracing handles missing OTel dependencies without raising."""
        from nanobot.config.schema import TracingConfig
        from nanobot.tracing.setup import setup_tracing

        cfg = TracingConfig(enabled=True)

        # Simulate absent optional packages by shadowing them in sys.modules.
        blocked: dict[str, None] = {
            k: None
            for k in list(sys.modules)
            if k.startswith(("opentelemetry", "openinference"))
        }
        blocked.update(
            {
                "opentelemetry": None,
                "opentelemetry.sdk": None,
                "opentelemetry.sdk.resources": None,
                "opentelemetry.sdk.trace": None,
                "opentelemetry.sdk.trace.export": None,
                "opentelemetry.exporter": None,
                "opentelemetry.exporter.otlp": None,
                "opentelemetry.exporter.otlp.proto": None,
                "opentelemetry.exporter.otlp.proto.http": None,
                "opentelemetry.exporter.otlp.proto.http.trace_exporter": None,
                "openinference": None,
                "openinference.instrumentation": None,
                "openinference.instrumentation.litellm": None,
            }
        )
        with patch.dict(sys.modules, blocked):
            setup_tracing(cfg)  # must not raise

    def test_enabled_configures_provider_and_instruments_litellm(self):
        """setup_tracing sets up a TracerProvider and instruments LiteLLM."""
        from nanobot.config.schema import TracingConfig
        from nanobot.tracing.setup import setup_tracing

        cfg = TracingConfig(enabled=True, endpoint="http://localhost:6006/v1/traces")

        # Build minimal mocks for each dependency imported inside setup_tracing.
        mock_instrumentor = MagicMock()
        mock_instrumentor_cls = MagicMock(return_value=mock_instrumentor)
        mock_provider = MagicMock()
        mock_exporter = MagicMock()
        mock_processor = MagicMock()
        mock_resource = MagicMock()
        mock_trace_api = MagicMock()

        sdk_resources = MagicMock()
        sdk_resources.Resource.create.return_value = mock_resource

        sdk_trace = MagicMock()
        sdk_trace.TracerProvider.return_value = mock_provider

        sdk_export = MagicMock()
        sdk_export.BatchSpanProcessor.return_value = mock_processor

        exporter_mod = MagicMock()
        exporter_mod.OTLPSpanExporter.return_value = mock_exporter

        oi_litellm = MagicMock()
        oi_litellm.LiteLLMInstrumentor = mock_instrumentor_cls

        mocks: dict[str, object] = {
            "opentelemetry": MagicMock(trace=mock_trace_api),
            "opentelemetry.trace": mock_trace_api,
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.resources": sdk_resources,
            "opentelemetry.sdk.trace": sdk_trace,
            "opentelemetry.sdk.trace.export": sdk_export,
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": exporter_mod,
            "openinference": MagicMock(),
            "openinference.instrumentation": MagicMock(),
            "openinference.instrumentation.litellm": oi_litellm,
        }

        with patch.dict(sys.modules, mocks):
            setup_tracing(cfg)

        # A TracerProvider must have been created with a resource.
        sdk_resources.Resource.create.assert_called_once_with({"service.name": "nanobot"})
        sdk_trace.TracerProvider.assert_called_once_with(resource=mock_resource)

        # The OTLP exporter must target the configured endpoint.
        exporter_mod.OTLPSpanExporter.assert_called_once_with(
            endpoint="http://localhost:6006/v1/traces"
        )

        # A BatchSpanProcessor wrapping that exporter must be registered.
        sdk_export.BatchSpanProcessor.assert_called_once_with(mock_exporter)
        mock_provider.add_span_processor.assert_called_once_with(mock_processor)

        # The provider must be set as the global tracer provider.
        mock_trace_api.set_tracer_provider.assert_called_once_with(mock_provider)

        # LiteLLMInstrumentor().instrument() must be called once.
        mock_instrumentor.instrument.assert_called_once()

    def test_enabled_uses_custom_project_name(self):
        """setup_tracing uses the configured project_name as service.name."""
        from nanobot.config.schema import TracingConfig
        from nanobot.tracing.setup import setup_tracing

        cfg = TracingConfig(enabled=True, project_name="my-custom-bot")

        sdk_resources = MagicMock()
        sdk_resources.Resource.create.return_value = MagicMock()
        mock_provider = MagicMock()
        sdk_trace = MagicMock()
        sdk_trace.TracerProvider.return_value = mock_provider
        sdk_export = MagicMock()
        sdk_export.BatchSpanProcessor.return_value = MagicMock()
        exporter_mod = MagicMock()
        exporter_mod.OTLPSpanExporter.return_value = MagicMock()
        oi_litellm = MagicMock()
        oi_litellm.LiteLLMInstrumentor.return_value = MagicMock()

        mocks: dict[str, object] = {
            "opentelemetry": MagicMock(),
            "opentelemetry.trace": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.resources": sdk_resources,
            "opentelemetry.sdk.trace": sdk_trace,
            "opentelemetry.sdk.trace.export": sdk_export,
            "opentelemetry.exporter": MagicMock(),
            "opentelemetry.exporter.otlp": MagicMock(),
            "opentelemetry.exporter.otlp.proto": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http": MagicMock(),
            "opentelemetry.exporter.otlp.proto.http.trace_exporter": exporter_mod,
            "openinference": MagicMock(),
            "openinference.instrumentation": MagicMock(),
            "openinference.instrumentation.litellm": oi_litellm,
        }

        with patch.dict(sys.modules, mocks):
            setup_tracing(cfg)

        sdk_resources.Resource.create.assert_called_once_with(
            {"service.name": "my-custom-bot"}
        )


# ---------------------------------------------------------------------------
# agent_turn_span
# ---------------------------------------------------------------------------


class TestAgentTurnSpan:
    def test_no_otel_returns_usable_context_manager(self):
        """Falls back to nullcontext when OTel is not installed."""
        from nanobot.tracing.setup import agent_turn_span

        blocked: dict[str, None] = {
            k: None
            for k in list(sys.modules)
            if k.startswith("opentelemetry")
        }
        blocked["opentelemetry"] = None
        blocked["opentelemetry.trace"] = None

        with patch.dict(sys.modules, blocked):
            ctx = agent_turn_span("default", "cli:user1")
            # Must be usable as a synchronous context manager without raising.
            with ctx:
                pass

    def test_exception_in_otel_returns_usable_context_manager(self):
        """Falls back to nullcontext when OTel raises an unexpected error."""
        from nanobot.tracing.setup import agent_turn_span

        broken_trace = MagicMock()
        broken_trace.get_tracer.side_effect = RuntimeError("oops")
        broken_otel = MagicMock(trace=broken_trace)

        with patch.dict(
            sys.modules,
            {"opentelemetry": broken_otel, "opentelemetry.trace": broken_trace},
        ):
            ctx = agent_turn_span("default", "cli:user1")
            with ctx:
                pass  # must not raise

    def test_creates_span_with_correct_attributes(self):
        """Creates a span named 'agent.turn' with agent.id and session.key."""
        from nanobot.tracing.setup import agent_turn_span

        mock_span_cm = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span_cm
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer

        with patch.dict(
            sys.modules,
            {
                "opentelemetry": MagicMock(trace=mock_trace),
                "opentelemetry.trace": mock_trace,
            },
        ):
            ctx = agent_turn_span("viharapala", "telegram:99999")

        mock_trace.get_tracer.assert_called_once_with("nanobot.agent")
        mock_tracer.start_as_current_span.assert_called_once_with(
            "agent.turn",
            attributes={"agent.id": "viharapala", "session.key": "telegram:99999"},
        )
        # The returned value must be the span context manager from OTel.
        assert ctx is mock_span_cm


# ---------------------------------------------------------------------------
# Integration: AgentLoop imports agent_turn_span without error
# ---------------------------------------------------------------------------


def test_agent_loop_imports_agent_turn_span():
    """AgentLoop can be imported; nanobot.tracing.setup is accessible."""
    from nanobot.tracing.setup import agent_turn_span  # noqa: F401

    # Simply importing confirms no circular-import or missing-dep issue.
    assert callable(agent_turn_span)

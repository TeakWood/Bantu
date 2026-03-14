"""OpenTelemetry tracing setup for nanobot LLM calls.

Tracing is optional — all functionality degrades gracefully to no-ops when
the ``tracing`` optional dependencies are not installed.

Install with::

    pip install 'nanobot-ai[tracing]'

Enable in config::

    {
      "tracing": {
        "enabled": true,
        "endpoint": "http://localhost:6006/v1/traces",
        "project_name": "nanobot"
      }
    }
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import TracingConfig


def setup_tracing(config: TracingConfig) -> None:
    """Initialise OpenTelemetry tracing with LiteLLM auto-instrumentation.

    When *config.enabled* is ``False``, or when the tracing optional
    dependencies are not installed, this function is a complete no-op.

    On success it:

    1. Creates an OTLP exporter pointed at *config.endpoint*.
    2. Registers a :class:`~opentelemetry.sdk.trace.TracerProvider` with a
       ``BatchSpanProcessor`` so spans are flushed asynchronously.
    3. Calls :class:`~openinference.instrumentation.litellm.LiteLLMInstrumentor`
       ``.instrument()`` so every ``litellm.acompletion`` / ``litellm.completion``
       call automatically emits child spans carrying model, token counts, etc.
    """
    if not config.enabled:
        return

    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "Tracing dependencies are not installed — tracing disabled. "
            "Enable with: pip install 'nanobot-ai[tracing]'"
        )
        return

    resource = Resource.create({"service.name": config.project_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=config.endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    LiteLLMInstrumentor().instrument()
    logger.info("Tracing enabled → {}", config.endpoint)


def agent_turn_span(agent_id: str, session_key: str) -> Any:
    """Return a context manager that traces one agent turn.

    Creates an OpenTelemetry span named ``agent.turn`` with the attributes:

    * ``agent.id``  — the owning agent identifier (e.g. ``"default"``)
    * ``session.key`` — the session routing key (e.g. ``"telegram:12345"``)

    LiteLLM calls made *inside* this context manager automatically become
    child spans of the agent-turn span (via OpenInference instrumentation and
    OpenTelemetry context propagation).

    Falls back silently to :func:`contextlib.nullcontext` when:

    * OpenTelemetry is not installed, or
    * No tracer provider has been configured (tracing is disabled).
    """
    try:
        from opentelemetry import trace

        return trace.get_tracer("nanobot.agent").start_as_current_span(
            "agent.turn",
            attributes={"agent.id": agent_id, "session.key": session_key},
        )
    except Exception:
        return nullcontext()

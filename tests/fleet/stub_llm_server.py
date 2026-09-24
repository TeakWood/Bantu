"""A local, scriptable OpenAI-compatible stub LLM server for fleet tests.

Fleet acceptance tests run nanobot instances in **separate OS processes**, so the
in-process fakes used elsewhere in this suite (``httpx.MockTransport``,
:class:`~nanobot.providers.base.LLMProvider` subclasses) cannot reach them. This
module serves the same contract over a real socket instead: a
:class:`http.server.ThreadingHTTPServer` bound to ``127.0.0.1:0`` speaking
``POST /v1/chat/completions`` (non-streaming JSON and SSE) plus ``GET /v1/models``.

Responses come from an ordered script queue, so a test can make an instance run a
chosen shell or file tool call and then observe the effect from outside. Every
request is captured for later assertions.

Wire an instance to it with ``providers.custom.apiBase`` — see
:meth:`StubLLMServer.instance_config`. ``conftest.py`` exposes the whole thing as
the ``stub_llm_server`` fixture.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

__all__ = (
    "DEFAULT_STUB_MODEL",
    "CapturedRequest",
    "StubCompletion",
    "StubLLMServer",
    "StubToolCall",
    "free_port",
)

DEFAULT_STUB_MODEL = "custom/stub-model"
_MAX_BODY_BYTES = 16 * 1024 * 1024


def free_port() -> int:
    """Return a currently-unused loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# Script entries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StubToolCall:
    """One ``tool_calls`` entry the stub should emit."""

    name: str
    arguments: Mapping[str, Any] | str = field(default_factory=dict)
    call_id: str | None = None

    def arguments_json(self) -> str:
        if isinstance(self.arguments, str):
            return self.arguments
        return json.dumps(dict(self.arguments), ensure_ascii=False)

    def wire_id(self) -> str:
        return self.call_id or f"call_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class StubCompletion:
    """One scripted assistant turn: visible text, tool calls, or both."""

    content: str | None = None
    tool_calls: tuple[StubToolCall, ...] = ()
    finish_reason: str | None = None

    @property
    def effective_finish_reason(self) -> str:
        if self.finish_reason:
            return self.finish_reason
        return "tool_calls" if self.tool_calls else "stop"


@dataclass(frozen=True)
class CapturedRequest:
    """One request the stub received, recorded verbatim."""

    method: str
    path: str
    headers: Mapping[str, str]
    body: Mapping[str, Any]

    @property
    def stream(self) -> bool:
        return self.body.get("stream") is True

    @property
    def model(self) -> str:
        model = self.body.get("model")
        return model if isinstance(model, str) else ""

    @property
    def messages(self) -> list[dict[str, Any]]:
        raw = self.body.get("messages")
        if not isinstance(raw, list):
            return []
        return [cast(dict[str, Any], m) for m in cast(list[object], raw) if isinstance(m, dict)]

    @property
    def tool_names(self) -> list[str]:
        raw = self.body.get("tools")
        if not isinstance(raw, list):
            return []
        names: list[str] = []
        for entry in cast(list[object], raw):
            if not isinstance(entry, dict):
                continue
            function = cast(dict[str, Any], entry).get("function")
            if isinstance(function, dict):
                name = cast(dict[str, Any], function).get("name")
                if isinstance(name, str):
                    names.append(name)
        return names


# ---------------------------------------------------------------------------
# Wire formatting
# ---------------------------------------------------------------------------


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _estimated_prompt_tokens(request: CapturedRequest) -> int:
    total = sum(len(json.dumps(message, ensure_ascii=False)) for message in request.messages)
    return max(1, total // 4)


def _split_for_stream(text: str, parts: int = 3) -> list[str]:
    """Split *text* into up to *parts* non-empty pieces for SSE deltas."""
    if not text:
        return []
    size = max(1, -(-len(text) // parts))
    return [text[i:i + size] for i in range(0, len(text), size)]


def _completion_payload(
    completion: StubCompletion,
    model: str,
    completion_id: str,
    prompt_tokens: int,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": completion.content}
    if completion.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.wire_id(),
                "type": "function",
                "index": index,
                "function": {"name": call.name, "arguments": call.arguments_json()},
            }
            for index, call in enumerate(completion.tool_calls)
        ]
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": completion.effective_finish_reason,
            }
        ],
        "usage": _usage(prompt_tokens, max(1, len(completion.content or "") // 4)),
    }


def _stream_chunks(
    completion: StubCompletion,
    model: str,
    completion_id: str,
    prompt_tokens: int,
) -> list[dict[str, Any]]:
    """Build the ordered SSE chunk payloads for one scripted turn."""
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    chunks: list[dict[str, Any]] = [chunk({"role": "assistant"})]
    for piece in _split_for_stream(completion.content or ""):
        chunks.append(chunk({"content": piece}))

    for index, call in enumerate(completion.tool_calls):
        call_id = call.wire_id()
        chunks.append(chunk({
            "tool_calls": [{
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": ""},
            }],
        }))
        # Arguments arrive fragmented on real providers; mirror that so
        # consumers exercise their delta-accumulation path.
        for piece in _split_for_stream(call.arguments_json(), parts=2):
            chunks.append(chunk({
                "tool_calls": [{"index": index, "function": {"arguments": piece}}],
            }))

    chunks.append(chunk({}, finish_reason=completion.effective_finish_reason))
    # Trailing usage-only chunk, as sent for stream_options.include_usage.
    chunks.append({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": _usage(prompt_tokens, max(1, len(completion.content or "") // 4)),
    })
    return chunks


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "nanobot-stub-llm/1"

    @property
    def stub(self) -> StubLLMServer:
        return cast(Any, self.server).stub

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr access log."""

    # -- routing --

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler contract
        route = self.path.split("?", 1)[0].rstrip("/")
        if route.endswith("/models"):
            self._send_json(200, self.stub.models_payload())
            return
        if route.endswith("/health"):
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": {"message": f"unknown route {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler contract
        route = self.path.split("?", 1)[0].rstrip("/")
        if not route.endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": f"unknown route {self.path}"}})
            return

        body = self._read_json_body()
        if body is None:
            self._send_json(400, {"error": {"message": "invalid JSON body"}})
            return

        request = CapturedRequest(
            method="POST",
            path=self.path,
            headers={key.lower(): value for key, value in self.headers.items()},
            body=body,
        )
        completion = self.stub.record(request)
        model = request.model or self.stub.model
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        prompt_tokens = _estimated_prompt_tokens(request)

        if request.stream:
            self._send_stream(
                _stream_chunks(completion, model, completion_id, prompt_tokens)
            )
        else:
            self._send_json(
                200,
                _completion_payload(completion, model, completion_id, prompt_tokens),
            )

    # -- transport --

    def _read_json_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length < 0 or length > _MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length) if length else b""
        try:
            parsed: object = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return None
        return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_stream(self, chunks: Sequence[Mapping[str, Any]]) -> None:
        # Length is unknown up front; close the connection to delimit the body
        # rather than hand-rolling chunked transfer encoding.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(
                b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n\n"
            )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class StubLLMServer:
    """A scriptable OpenAI-compatible endpoint on ``127.0.0.1``.

    Scripted completions are consumed in FIFO order, one per chat request. When
    the queue runs dry the configured fallback is returned instead, so an
    unexpected extra model call never hangs an instance under test.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_STUB_MODEL,
        fallback: StubCompletion | None = None,
    ) -> None:
        self.model = model
        self._fallback = fallback or StubCompletion(content="stub fallback response")
        self._lock = threading.Lock()
        self._new_request = threading.Condition(self._lock)
        self._script: deque[StubCompletion] = deque()
        self._requests: list[CapturedRequest] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle --

    def start(self) -> StubLLMServer:
        if self._httpd is not None:
            return self
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        httpd.daemon_threads = True
        cast(Any, httpd).stub = self
        thread = threading.Thread(
            target=httpd.serve_forever,
            name="stub-llm-server",
            daemon=True,
        )
        thread.start()
        self._httpd = httpd
        self._thread = thread
        return self

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        self._httpd, self._thread = None, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> StubLLMServer:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- addressing --

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("stub LLM server is not running")
        return int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        """The ``providers.custom.apiBase`` value for instances under test."""
        return f"http://127.0.0.1:{self.port}/v1"

    def models_payload(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nanobot-stub",
                }
            ],
        }

    # -- scripting --

    def script(self, *completions: StubCompletion) -> StubLLMServer:
        with self._lock:
            self._script.extend(completions)
        return self

    def script_text(self, content: str) -> StubLLMServer:
        return self.script(StubCompletion(content=content))

    def script_tool_call(
        self,
        name: str,
        arguments: Mapping[str, Any] | str | None = None,
        *,
        content: str | None = None,
        call_id: str | None = None,
    ) -> StubLLMServer:
        """Script one assistant turn that calls *name* with *arguments*."""
        return self.script(StubCompletion(
            content=content,
            tool_calls=(
                StubToolCall(name=name, arguments=arguments or {}, call_id=call_id),
            ),
        ))

    def set_fallback(self, completion: StubCompletion) -> None:
        with self._lock:
            self._fallback = completion

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._script)

    # -- capture --

    @property
    def requests(self) -> tuple[CapturedRequest, ...]:
        with self._lock:
            return tuple(self._requests)

    def clear(self) -> None:
        with self._lock:
            self._script.clear()
            self._requests.clear()

    def record(self, request: CapturedRequest) -> StubCompletion:
        """Capture *request* and pop the completion that answers it."""
        with self._new_request:
            self._requests.append(request)
            completion = self._script.popleft() if self._script else self._fallback
            self._new_request.notify_all()
        return completion

    def wait_for_requests(self, count: int, timeout: float = 30.0) -> tuple[CapturedRequest, ...]:
        """Block until at least *count* requests have arrived, then return them."""
        deadline = time.monotonic() + timeout
        with self._new_request:
            while len(self._requests) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"stub LLM server saw {len(self._requests)} requests, expected {count}"
                    )
                self._new_request.wait(remaining)
            return tuple(self._requests)

    def instance_config(
        self,
        workspace: Path,
        *,
        api_port: int,
        allowed_env_keys: Sequence[str] = (),
        max_tool_iterations: int = 6,
        restrict_to_workspace: bool = False,
        sandbox: str = "",
    ) -> dict[str, Any]:
        """Build an instance config wired to this stub via ``providers.custom``.

        Defaults match what the fleet acceptance criteria require of an instance
        under test: its own guards off (``restrictToWorkspace`` false, shell
        sandbox disabled) so that only OS-level confinement can explain a denial.
        """
        return {
            "agents": {
                "defaults": {
                    "workspace": str(workspace),
                    "provider": "custom",
                    "model": self.model,
                    "maxToolIterations": max_tool_iterations,
                    "dream": {"enabled": False},
                    "idleCompactAfterMinutes": 0,
                }
            },
            "providers": {
                "custom": {
                    "apiKey": "stub-key",
                    "apiBase": self.base_url,
                }
            },
            "tools": {
                "restrictToWorkspace": restrict_to_workspace,
                "exec": {
                    "enable": True,
                    "sandbox": sandbox,
                    "allowedEnvKeys": list(allowed_env_keys),
                },
            },
            "api": {"host": "127.0.0.1", "port": api_port, "timeout": 60.0},
        }

"""Self-tests for the stub LLM server fixture.

The wire-shape tests exercise the stub directly. The final test is the one that
matters for the fleet work: it runs a **real nanobot instance in a separate OS
process**, points it at the stub via ``providers.custom.apiBase``, and asserts a
scripted shell tool call actually executed on disk.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from stub_llm_server import StubCompletion, StubLLMServer, StubToolCall, free_port

from nanobot.providers.openai_compat_provider import OpenAICompatProvider

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STARTUP_TIMEOUT_S = 90.0
_TURN_TIMEOUT_S = 90.0


# ---------------------------------------------------------------------------
# Wire shape
# ---------------------------------------------------------------------------


def test_models_endpoint_lists_the_configured_model(stub_llm_server: StubLLMServer) -> None:
    payload = httpx.get(
        f"{stub_llm_server.base_url}/models", timeout=5.0, trust_env=False
    ).json()

    assert payload["object"] == "list"
    assert [model["id"] for model in payload["data"]] == [stub_llm_server.model]


def test_non_streaming_completion_serves_the_scripted_queue_in_order(
    stub_llm_server: StubLLMServer,
) -> None:
    stub_llm_server.script_text("first").script_tool_call("exec", {"command": "true"})

    def ask(text: str) -> dict[str, Any]:
        return httpx.post(
            f"{stub_llm_server.base_url}/chat/completions",
            json={"model": "custom/stub-model", "messages": [{"role": "user", "content": text}]},
            timeout=5.0,
            trust_env=False,
        ).json()

    first, second = ask("one"), ask("two")

    assert first["object"] == "chat.completion"
    assert first["choices"][0]["message"]["content"] == "first"
    assert first["choices"][0]["finish_reason"] == "stop"
    assert first["usage"]["total_tokens"] > 0

    call = second["choices"][0]["message"]["tool_calls"][0]
    assert second["choices"][0]["finish_reason"] == "tool_calls"
    assert call["function"]["name"] == "exec"
    assert json.loads(call["function"]["arguments"]) == {"command": "true"}

    # Queue drained — further requests fall back rather than hanging.
    assert stub_llm_server.pending == 0
    assert ask("three")["choices"][0]["message"]["content"] == "stub fallback response"

    captured = stub_llm_server.requests
    assert len(captured) == 3
    assert [request.messages[0]["content"] for request in captured] == ["one", "two", "three"]
    assert all(not request.stream for request in captured)


def test_streaming_completion_emits_sse_deltas_and_done(
    stub_llm_server: StubLLMServer,
) -> None:
    stub_llm_server.script(StubCompletion(
        content="hello there",
        tool_calls=(StubToolCall(name="exec", arguments={"command": "echo hi"}, call_id="c1"),),
    ))

    with httpx.stream(
        "POST",
        f"{stub_llm_server.base_url}/chat/completions",
        json={
            "model": "custom/stub-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        timeout=10.0,
        trust_env=False,
    ) as response:
        assert response.headers["content-type"] == "text/event-stream"
        payloads = [line[len("data: "):] for line in response.iter_lines() if line.startswith("data: ")]

    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(raw) for raw in payloads[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)

    content = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks
        if chunk["choices"]
    )
    assert content == "hello there"

    arguments = "".join(
        tool_call.get("function", {}).get("arguments", "")
        for chunk in chunks
        if chunk["choices"]
        for tool_call in chunk["choices"][0]["delta"].get("tool_calls", [])
    )
    assert json.loads(arguments) == {"command": "echo hi"}
    assert [
        tool_call["id"]
        for chunk in chunks
        if chunk["choices"]
        for tool_call in chunk["choices"][0]["delta"].get("tool_calls", [])
        if "id" in tool_call
    ] == ["c1"]

    finish_reasons = [
        chunk["choices"][0]["finish_reason"] for chunk in chunks if chunk["choices"]
    ]
    assert finish_reasons[-1] == "tool_calls"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] > 0

    assert stub_llm_server.requests[0].stream is True


def test_unknown_routes_are_rejected(stub_llm_server: StubLLMServer) -> None:
    base = f"http://127.0.0.1:{stub_llm_server.port}"
    assert httpx.get(f"{base}/v1/nope", timeout=5.0, trust_env=False).status_code == 404
    assert httpx.post(f"{base}/v1/nope", json={}, timeout=5.0, trust_env=False).status_code == 404
    assert httpx.post(
        f"{stub_llm_server.base_url}/chat/completions",
        content=b"not json",
        headers={"Content-Type": "application/json"},
        timeout=5.0,
        trust_env=False,
    ).status_code == 400


def test_wait_for_requests_times_out_when_nothing_arrives(
    stub_llm_server: StubLLMServer,
) -> None:
    with pytest.raises(AssertionError, match="expected 1"):
        stub_llm_server.wait_for_requests(1, timeout=0.2)


# ---------------------------------------------------------------------------
# The provider that instances actually use
# ---------------------------------------------------------------------------

_EXEC_TOOL_SCHEMA: list[dict[str, Any]] = [{
    "type": "function",
    "function": {
        "name": "exec",
        "description": "Execute a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def _provider(stub_llm_server: StubLLMServer) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="stub-key",
        api_base=stub_llm_server.base_url,
        default_model="stub-model",
    )


async def test_provider_parses_a_scripted_tool_call(stub_llm_server: StubLLMServer) -> None:
    stub_llm_server.script_tool_call("exec", {"command": "echo hi"}, call_id="call_abc")

    response = await _provider(stub_llm_server).chat(
        messages=[{"role": "user", "content": "go"}],
        tools=_EXEC_TOOL_SCHEMA,
    )

    assert response.finish_reason == "tool_calls", response.content
    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [
        ("call_abc", "exec", {"command": "echo hi"})
    ]
    assert response.usage is not None and response.usage.input_tokens > 0


async def test_provider_parses_a_scripted_streaming_turn(stub_llm_server: StubLLMServer) -> None:
    stub_llm_server.script(StubCompletion(
        content="hello there",
        tool_calls=(StubToolCall(name="exec", arguments={"command": "echo hi"}, call_id="c1"),),
    ))
    deltas: list[str] = []

    async def collect(text: str) -> None:
        deltas.append(text)

    response = await _provider(stub_llm_server).chat_stream(
        messages=[{"role": "user", "content": "go"}],
        tools=_EXEC_TOOL_SCHEMA,
        on_content_delta=collect,
    )

    assert response.finish_reason == "tool_calls", response.content
    assert "".join(deltas) == "hello there" and len(deltas) > 1
    assert response.content == "hello there"
    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [
        ("c1", "exec", {"command": "echo hi"})
    ]
    assert response.usage is not None and response.usage.input_tokens > 0
    assert stub_llm_server.requests[0].stream is True


# ---------------------------------------------------------------------------
# Real instance subprocess
# ---------------------------------------------------------------------------


def _start_instance(config_path: Path, log_path: Path) -> subprocess.Popen[bytes]:
    with log_path.open("wb") as log_file:
        return subprocess.Popen(
            [sys.executable, "-m", "nanobot", "serve", "--config", str(config_path)],
            cwd=_REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def _wait_for_health(
    base_url: str,
    process: subprocess.Popen[bytes],
    log_path: Path,
) -> None:
    deadline = time.monotonic() + _STARTUP_TIMEOUT_S
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            response = httpx.get(f"{base_url}/health", timeout=5.0, trust_env=False)
            if response.status_code == 200:
                return
            last_error = RuntimeError(f"health returned {response.status_code}")
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
        time.sleep(0.2)
    raise AssertionError(
        f"instance did not become healthy; last_error={last_error!r}\n"
        f"{log_path.read_text(encoding='utf-8', errors='replace')}"
    )


def _stop_instance(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def test_stub_drives_a_real_instance_subprocess_to_run_a_shell_tool_call(
    tmp_path: Path,
    stub_llm_server: StubLLMServer,
) -> None:
    """The acceptance case: a scripted tool call must actually execute on disk."""
    pytest.importorskip("aiohttp")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sentinel = workspace / "tool-ran.txt"
    api_port = free_port()
    config_path = tmp_path / "config.json"
    log_path = tmp_path / "instance.log"
    config_path.write_text(
        json.dumps(stub_llm_server.instance_config(workspace, api_port=api_port)),
        encoding="utf-8",
    )

    stub_llm_server.script_tool_call(
        "exec",
        {"command": f"printf tool-ran > {sentinel}"},
    ).script_text("the shell tool ran")

    base_url = f"http://127.0.0.1:{api_port}"
    process = _start_instance(config_path, log_path)
    try:
        _wait_for_health(base_url, process, log_path)

        answer = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "write the sentinel"}]},
            timeout=_TURN_TIMEOUT_S,
            trust_env=False,
        )
        assert answer.status_code == 200, answer.text
        assert "the shell tool ran" in answer.json()["choices"][0]["message"]["content"]
    finally:
        _stop_instance(process)

    # The tool call the stub scripted really ran inside the instance process.
    assert sentinel.read_text(encoding="utf-8") == "tool-ran"

    requests = stub_llm_server.requests
    assert len(requests) >= 2, log_path.read_text(encoding="utf-8", errors="replace")
    assert "exec" in requests[0].tool_names
    # The instance replayed the stub's tool call and its result back to the stub.
    replayed = [
        message
        for request in requests
        for message in request.messages
        if message.get("role") == "tool"
    ]
    assert replayed, json.dumps([dict(r.body) for r in requests])[:4000]

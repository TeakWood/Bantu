"""A provider whose replies are scripted, for driving real agent turns in tests."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

Script = Callable[[list[dict[str, Any]]], LLMResponse]


class ScriptedProvider(LLMProvider):
    """Answer every chat call from a caller-supplied script.

    The script receives the request messages and returns the response, so a test
    can branch on what the agent has done so far (for example: ask for a tool on
    the first call, then reply once its result is in context).
    """

    def __init__(self, script: Script, *, name: str = "scripted") -> None:
        super().__init__(api_key="test-key", provider_name=name)
        self.script = script
        self.calls: list[list[dict[str, Any]]] = []

    def get_default_model(self) -> str:
        return "scripted/model"

    def estimate_prompt_tokens(self, *_args: Any, **_kwargs: Any) -> tuple[int, str]:
        return 1000, "test"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        return self.script(messages)


def reply(content: str) -> LLMResponse:
    """A plain final answer."""
    return LLMResponse(content=content, finish_reason="stop")


_call_ids = itertools.count(1)


def tool_call(name: str, arguments: dict[str, Any], *, call_id: str | None = None) -> LLMResponse:
    """A single tool call, with a fresh id so replays stay valid."""
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id=call_id or f"call-{next(_call_ids)}",
                name=name,
                arguments=arguments,
            )
        ],
        finish_reason="tool_calls",
    )


def transcript(messages: list[dict[str, Any]]) -> str:
    """Flatten a request's messages so a script can branch on what it contains."""
    return "\n".join(str(message.get("content", "")) for message in messages)

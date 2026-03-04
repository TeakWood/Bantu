"""Null provider — returned when no LLM API key is configured."""

from typing import Any

from nanobot.providers.base import LLMProvider, LLMResponse

_NOT_CONFIGURED_MESSAGE = (
    "⚠️ No API key is configured. "
    "Run 'nanobot onboard' to set up your configuration, "
    "or add an API key to ~/.bantu/config.json under the providers section."
)


class NullProvider(LLMProvider):
    """Provider used when no API key has been configured.

    Instead of crashing, it responds to every chat request with a clear
    'not configured' message so the gateway can start and inform users.
    """

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=_NOT_CONFIGURED_MESSAGE,
            finish_reason="stop",
        )

    def get_default_model(self) -> str:
        return "not-configured"

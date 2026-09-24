"""Fixtures for fleet tests, centred on the local stub LLM server.

The server itself lives in :mod:`stub_llm_server` so test modules can import its
types directly; this file only wires it into pytest.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from stub_llm_server import StubLLMServer


@pytest.fixture
def stub_llm_server() -> Iterator[StubLLMServer]:
    """A running OpenAI-compatible stub on 127.0.0.1, with an empty script."""
    with StubLLMServer() as server:
        yield server

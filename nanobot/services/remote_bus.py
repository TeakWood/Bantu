"""HTTP-backed message bus connecting the gateway to a remote agent service.

:class:`RemoteMessageBus` has the same interface as
:class:`~nanobot.bus.queue.MessageBus` (``publish_inbound``,
``consume_outbound``, ``inbound_size``, ``outbound_size``) so the channel
manager and channel implementations can use it as a drop-in replacement.

Architecture
------------
- ``publish_inbound`` serialises the message and POSTs it to
  ``<agent_url>/api/inbound``.
- A long-poll background task continuously GETs ``<agent_url>/api/outbound``
  and feeds the returned messages into a local ``asyncio.Queue``.
- ``consume_outbound`` reads from that local queue.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from nanobot.bus.events import InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

_DEFAULT_CONNECT_RETRY_S = 2.0


def _inbound_to_dict(msg: InboundMessage) -> dict[str, Any]:
    return {
        "channel": msg.channel,
        "sender_id": msg.sender_id,
        "chat_id": msg.chat_id,
        "content": msg.content,
        "media": msg.media,
        "metadata": msg.metadata,
        "session_key_override": msg.session_key_override,
    }


def _dict_to_outbound(data: dict[str, Any]) -> OutboundMessage:
    return OutboundMessage(
        channel=data["channel"],
        chat_id=data["chat_id"],
        content=data["content"],
        reply_to=data.get("reply_to"),
        media=data.get("media", []),
        metadata=data.get("metadata", {}),
    )


class RemoteMessageBus:
    """Drop-in replacement for :class:`~nanobot.bus.queue.MessageBus` that
    communicates with a remote agent service via HTTP.

    Parameters
    ----------
    agent_url:
        Base URL of the agent service, e.g. ``"http://localhost:18792"``.
    poll_timeout_s:
        Long-poll timeout sent to the agent's ``/api/outbound`` endpoint.
    """

    def __init__(self, agent_url: str, poll_timeout_s: float = 5.0) -> None:
        self._agent_url = agent_url.rstrip("/")
        self._poll_timeout_s = poll_timeout_s
        self._outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self._poll_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public bus interface
    # ------------------------------------------------------------------

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Forward an inbound message to the agent service via HTTP POST."""
        url = f"{self._agent_url}/api/inbound"
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(url, json=_inbound_to_dict(msg), timeout=10.0)
                r.raise_for_status()
        except Exception as exc:
            logger.error("RemoteMessageBus: failed to POST inbound to %s: %s", url, exc)
            raise

    async def consume_outbound(self) -> OutboundMessage:
        """Return the next outbound message from the local queue (blocks)."""
        return await self._outbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Enqueue a message directly (used in tests / embedded fallback)."""
        await self._outbound.put(msg)

    @property
    def inbound_size(self) -> int:
        return 0

    @property
    def outbound_size(self) -> int:
        return self._outbound.qsize()

    # ------------------------------------------------------------------
    # Background poll loop
    # ------------------------------------------------------------------

    def start_polling(self) -> None:
        """Launch the background outbound-poll task."""
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())

    def stop_polling(self) -> None:
        """Cancel the background poll task."""
        if self._poll_task is not None:
            self._poll_task.cancel()
            self._poll_task = None

    async def _poll_loop(self) -> None:
        """Continuously long-poll the agent service for outbound messages."""
        url = f"{self._agent_url}/api/outbound"
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    r = await client.get(
                        url,
                        params={"timeout": str(self._poll_timeout_s)},
                        timeout=self._poll_timeout_s + 5.0,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        for msg_data in data.get("messages", []):
                            await self._outbound.put(_dict_to_outbound(msg_data))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "RemoteMessageBus: poll error (%s), retrying in %.1fs",
                        exc,
                        _DEFAULT_CONNECT_RETRY_S,
                    )
                    await asyncio.sleep(_DEFAULT_CONNECT_RETRY_S)

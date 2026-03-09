"""REST API server that wraps the MessageBus + AgentLoop for the agent service.

Endpoints
---------
GET  /api/health
    Liveness probe.  Returns ``{"status": "ok"}``.

POST /api/inbound
    Accept an :class:`~nanobot.bus.events.InboundMessage` from the gateway and
    enqueue it on the bus.

    Body (JSON)::

        {
            "channel": "telegram",
            "sender_id": "123",
            "chat_id": "456",
            "content": "Hello",
            "media": [],
            "metadata": {},
            "session_key_override": null
        }

GET  /api/outbound?timeout=<seconds>
    Long-poll for outbound messages.  Waits up to *timeout* seconds (default 5)
    for at least one message; then drains the rest of the queue and returns all
    available messages in a single response.

    Response::

        {"messages": [{"channel": ..., "chat_id": ..., "content": ..., ...}]}

    Returns ``{"messages": []}`` on timeout.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aiohttp import web

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus

# aiohttp Application key for the shared MessageBus instance.
_APP_KEY_BUS: web.AppKey[MessageBus] = web.AppKey("bus", MessageBus)


def _outbound_to_dict(msg: OutboundMessage) -> dict[str, Any]:
    return {
        "channel": msg.channel,
        "chat_id": msg.chat_id,
        "content": msg.content,
        "reply_to": msg.reply_to,
        "media": msg.media,
        "metadata": msg.metadata,
    }


async def _handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "agent"})


async def _handle_inbound(request: web.Request) -> web.Response:
    """Accept a message from the gateway and push it onto the bus."""
    try:
        data: dict[str, Any] = await request.json()
    except Exception:
        raise web.HTTPBadRequest(reason="Invalid JSON body")

    required = ("channel", "sender_id", "chat_id", "content")
    missing = [f for f in required if f not in data]
    if missing:
        raise web.HTTPBadRequest(reason=f"Missing fields: {', '.join(missing)}")

    msg = InboundMessage(
        channel=data["channel"],
        sender_id=data["sender_id"],
        chat_id=data["chat_id"],
        content=data["content"],
        media=data.get("media", []),
        metadata=data.get("metadata", {}),
        session_key_override=data.get("session_key_override"),
    )
    bus: MessageBus = request.app[_APP_KEY_BUS]
    await bus.publish_inbound(msg)
    return web.json_response({"status": "accepted"})


async def _handle_outbound(request: web.Request) -> web.Response:
    """Long-poll for outbound messages, returning all currently available."""
    timeout = float(request.rel_url.query.get("timeout", "5"))
    bus: MessageBus = request.app[_APP_KEY_BUS]

    try:
        first = await asyncio.wait_for(bus.consume_outbound(), timeout=timeout)
    except asyncio.TimeoutError:
        return web.json_response({"messages": []})

    messages = [_outbound_to_dict(first)]

    # Drain any additional queued messages without blocking.
    while not bus.outbound.empty():
        try:
            extra = bus.outbound.get_nowait()
            messages.append(_outbound_to_dict(extra))
        except asyncio.QueueEmpty:
            break

    return web.json_response({"messages": messages})


class AgentRestServer:
    """Lightweight aiohttp server exposing the bus over HTTP.

    Parameters
    ----------
    bus:
        The shared :class:`~nanobot.bus.queue.MessageBus` instance.
    host:
        Bind address (default ``"0.0.0.0"``).
    port:
        TCP port (default ``18792``).
    """

    def __init__(self, bus: MessageBus, host: str = "0.0.0.0", port: int = 18792) -> None:
        self._bus = bus
        self._host = host
        self._port = port
        self._runner: web.AppRunner | None = None

    # ------------------------------------------------------------------
    # Application factory
    # ------------------------------------------------------------------

    def _build_app(self) -> web.Application:
        app = web.Application()
        app[_APP_KEY_BUS] = self._bus
        app.router.add_get("/api/health", _handle_health)
        app.router.add_post("/api/inbound", _handle_inbound)
        app.router.add_get("/api/outbound", _handle_outbound)
        return app

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the REST server."""
        app = self._build_app()
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

    async def stop(self) -> None:
        """Stop the REST server and release resources."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

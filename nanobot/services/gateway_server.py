"""Gateway HTTP server — health endpoint and admin-API reverse proxy.

When the gateway runs in distributed mode it starts this lightweight aiohttp
server on the gateway port (default 18790) to serve:

- ``GET /health``
    Liveness probe; returns ``{"status": "ok", "service": "gateway"}``.

- ``/api/admin/{path}``   (only when ``admin_url`` is non-empty)
    Transparent reverse proxy to the admin service.  All HTTP methods,
    headers (minus ``Host``/``Content-Length``), and the request body are
    forwarded verbatim; the response is returned as-is.

This server does NOT handle channel traffic — channels connect directly to
their respective external services (Telegram, Discord, etc.).
"""

from __future__ import annotations

import logging

import httpx
from aiohttp import web

logger = logging.getLogger(__name__)

# Hop-by-hop headers that must not be forwarded.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

_APP_KEY_ADMIN_URL: web.AppKey[str] = web.AppKey("admin_url", str)


async def _handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "gateway"})


async def _handle_admin_proxy(request: web.Request) -> web.StreamResponse:
    """Reverse-proxy the request to the admin service."""
    admin_url: str = request.app[_APP_KEY_ADMIN_URL]
    path_info: str = request.match_info.get("path_info", "")

    target = f"{admin_url}/api/{path_info}"
    if request.query_string:
        target = f"{target}?{request.query_string}"

    forward_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    body = await request.read()

    try:
        async with httpx.AsyncClient() as client:
            r = await client.request(
                method=request.method,
                url=target,
                headers=forward_headers,
                content=body,
                timeout=30.0,
            )
    except httpx.RequestError as exc:
        logger.error("Gateway admin proxy error: %s", exc)
        return web.Response(status=502, text=f"Bad Gateway: {exc}")

    # Strip hop-by-hop from the upstream response too.
    response_headers = {
        k: v for k, v in r.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    return web.Response(
        status=r.status_code,
        body=r.content,
        headers=response_headers,
    )


class GatewayHttpServer:
    """HTTP server for the gateway service.

    Parameters
    ----------
    host:
        Bind address (default ``"0.0.0.0"``).
    port:
        TCP port (default ``18790``).
    admin_url:
        Base URL of the admin service, e.g. ``"http://localhost:18791"``.
        Pass an empty string to disable the admin proxy route.
    """

    def __init__(
        self, host: str = "0.0.0.0", port: int = 18790, admin_url: str = ""
    ) -> None:
        self._host = host
        self._port = port
        self._admin_url = admin_url.rstrip("/")
        self._runner: web.AppRunner | None = None

    def _build_app(self) -> web.Application:
        app = web.Application()
        app[_APP_KEY_ADMIN_URL] = self._admin_url

        app.router.add_get("/health", _handle_health)

        if self._admin_url:
            app.router.add_route("*", "/api/admin/{path_info:.*}", _handle_admin_proxy)

        return app

    async def start(self) -> None:
        """Start the HTTP server."""
        app = self._build_app()
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

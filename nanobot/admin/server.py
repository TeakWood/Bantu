"""Admin HTTP server backed by aiohttp with bearer-token auth and strict CORS policy."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp import web

from nanobot.admin import routes as _routes
from nanobot.config.schema import Config

# Static assets bundled with the package
_STATIC_DIR = Path(__file__).parent / "static"

# CORS headers sent on every matched-origin response
_CORS_METHODS = "GET, PUT, POST, DELETE, OPTIONS"
_CORS_HEADERS = "Authorization, Content-Type"
_CORS_CREDENTIALS = "false"


class AdminServer:
    """Lightweight HTTP server for the Bantu admin console.

    Features:
    - Optional Bearer token authentication (enforced only when
      ``config.gateway.admin.token`` is non-empty).
    - Strict CORS policy: only requests originating from
      ``http://127.0.0.1:<port>`` or ``http://localhost:<port>`` receive CORS
      response headers.  Wildcard origins are intentionally forbidden to
      prevent cross-site config exfiltration.
    - Serves a static admin UI from ``nanobot/admin/static/index.html`` at
      ``GET /``.
    """

    def __init__(self, config: Config, config_path: Path | None = None) -> None:
        self._config = config
        self._config_path = config_path
        self._runner: web.AppRunner | None = None

    # ------------------------------------------------------------------
    # Application factory
    # ------------------------------------------------------------------

    def _build_app(self) -> web.Application:
        """Build and return the configured aiohttp Application."""
        cfg = self._config.gateway.admin
        token: str = cfg.token
        allowed_origins: frozenset[str] = frozenset(
            {
                f"http://127.0.0.1:{cfg.port}",
                f"http://localhost:{cfg.port}",
            }
        )

        # ------------------------------------------------------------------
        # CORS middleware — must be outermost so OPTIONS bypasses auth
        # ------------------------------------------------------------------

        @web.middleware
        async def cors_middleware(
            request: web.Request,
            handler: web.RequestHandler,
        ) -> web.StreamResponse:
            origin = request.headers.get("Origin", "")
            origin_allowed = origin in allowed_origins

            # Respond to preflight immediately — do NOT invoke auth middleware
            if request.method == "OPTIONS":
                resp: web.StreamResponse = web.Response(status=200)
                if origin_allowed:
                    _add_cors_headers(resp, origin)
                return resp

            resp = await handler(request)
            if origin_allowed:
                _add_cors_headers(resp, origin)
            return resp

        # ------------------------------------------------------------------
        # Auth middleware — applied after CORS (only for non-OPTIONS requests)
        # ------------------------------------------------------------------

        @web.middleware
        async def auth_middleware(
            request: web.Request,
            handler: web.RequestHandler,
        ) -> web.StreamResponse:
            if token:
                auth_header = request.headers.get("Authorization", "")
                expected = f"Bearer {token}"
                if auth_header != expected:
                    raise web.HTTPUnauthorized(
                        reason="Invalid or missing Bearer token",
                        headers={"WWW-Authenticate": 'Bearer realm="nanobot-admin"'},
                    )
            return await handler(request)

        app = web.Application(middlewares=[cors_middleware, auth_middleware])

        # Per-app lock and config path for route handlers (typed keys avoid warnings).
        app[_routes.APP_KEY_CONFIG_LOCK] = asyncio.Lock()
        app[_routes.APP_KEY_CONFIG_PATH] = self._config_path

        # Static UI
        app.router.add_get("/", _handle_root)

        # --- Config ---
        app.router.add_get("/api/config", _routes.handle_get_config)

        # --- Providers ---
        app.router.add_get("/api/providers", _routes.handle_get_providers)
        app.router.add_put("/api/providers/{name}", _routes.handle_put_provider)

        # --- Channels ---
        app.router.add_get("/api/channels", _routes.handle_get_channels)
        app.router.add_get("/api/channels/{name}", _routes.handle_get_channel)
        app.router.add_put("/api/channels/{name}", _routes.handle_put_channel)

        # --- MCP servers ---
        app.router.add_get("/api/mcp", _routes.handle_get_mcp)
        app.router.add_post("/api/mcp/{name}", _routes.handle_post_mcp)
        app.router.add_put("/api/mcp/{name}", _routes.handle_put_mcp)
        app.router.add_delete("/api/mcp/{name}", _routes.handle_delete_mcp)

        # --- Agent defaults ---
        app.router.add_get("/api/agent", _routes.handle_get_agent)
        app.router.add_put("/api/agent", _routes.handle_put_agent)

        return app

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the admin HTTP server and begin accepting connections."""
        cfg = self._config.gateway.admin
        app = self._build_app()
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, cfg.host, cfg.port)
        await site.start()

    async def stop(self) -> None:
        """Stop the admin HTTP server and release all resources."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None


# ---------------------------------------------------------------------------
# Helpers (module-level so they can be referenced inside the factory closures)
# ---------------------------------------------------------------------------


def _add_cors_headers(response: web.StreamResponse, origin: str) -> None:
    """Attach CORS response headers to *response* for the given *origin*."""
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = _CORS_METHODS
    response.headers["Access-Control-Allow-Headers"] = _CORS_HEADERS
    response.headers["Access-Control-Allow-Credentials"] = _CORS_CREDENTIALS


async def _handle_root(request: web.Request) -> web.StreamResponse:
    """Serve the static admin UI (``static/index.html``)."""
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return web.FileResponse(index)
    # Fallback: plain-text placeholder
    return web.Response(
        text=(
            "<!DOCTYPE html><html><head><title>Bantu Admin</title></head>"
            "<body><h1>Bantu Admin</h1><p>UI coming soon.</p></body></html>"
        ),
        content_type="text/html",
    )

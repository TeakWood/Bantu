# ── Admin UI Service ───────────────────────────────────────────────────────────
# Serves the Bantu admin REST API and static UI.  The gateway proxies
# /api/admin/* requests to this service.
#
# Default port: 18791
# ──────────────────────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p nanobot bridge && touch nanobot/__init__.py && \
    uv pip install --system --no-cache . && \
    rm -rf nanobot bridge

# Copy full source and install
COPY nanobot/ nanobot/
COPY bridge/ bridge/
RUN uv pip install --system --no-cache .

RUN mkdir -p /root/.bantu

# Admin UI port
EXPOSE 18791

ENTRYPOINT ["nanobot"]
CMD ["serve-admin", "--host", "0.0.0.0"]

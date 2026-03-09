# ── Agent Service ──────────────────────────────────────────────────────────────
# Runs the MessageBus + AgentLoop + CronService + HeartbeatService and exposes
# them via a lightweight REST API so the gateway can forward messages.
#
# Default port: 18792
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

# Agent REST API port
EXPOSE 18792

ENTRYPOINT ["nanobot"]
CMD ["serve-agent"]

"""Service layer for distributed (monorepo) deployment of Bantu.

Three independent services share the same ``nanobot`` package:

- **gateway** – accepts channel connections and routes messages.
- **agent**   – runs the MessageBus + AgentLoop; exposes a REST API.
- **admin**   – serves the Admin UI; exposes the existing admin REST API.
"""

"""Named agents: several first-class agents within one nanobot install."""

from nanobot.agents.harness import GatewayHarness, open_gateway
from nanobot.agents.multi import MultiAgentRuntime
from nanobot.agents.registry import (
    AgentRegistryEntry,
    agent_names,
    agent_registry,
    route,
)
from nanobot.agents.resolution import (
    NAMED_AGENT_WORKSPACE_ROOT,
    ResolvedAgentConfig,
    named_agent_entry,
    named_agent_workspace,
    resolve_agent_config,
)
from nanobot.agents.runtime import (
    AgentRuntime,
    bootstrap_agent_workspace,
    build_agent_runtime,
)

__all__ = [
    "NAMED_AGENT_WORKSPACE_ROOT",
    "AgentRegistryEntry",
    "AgentRuntime",
    "GatewayHarness",
    "MultiAgentRuntime",
    "ResolvedAgentConfig",
    "agent_names",
    "agent_registry",
    "bootstrap_agent_workspace",
    "build_agent_runtime",
    "named_agent_entry",
    "named_agent_workspace",
    "open_gateway",
    "resolve_agent_config",
    "route",
]

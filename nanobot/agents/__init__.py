"""Named agents: several first-class agents within one nanobot install."""

from nanobot.agents.resolution import (
    NAMED_AGENT_WORKSPACE_ROOT,
    ResolvedAgentConfig,
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
    "AgentRuntime",
    "ResolvedAgentConfig",
    "bootstrap_agent_workspace",
    "build_agent_runtime",
    "named_agent_workspace",
    "resolve_agent_config",
]

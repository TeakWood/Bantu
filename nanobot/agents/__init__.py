"""Named agents: several first-class agents within one nanobot install."""

from nanobot.agents.resolution import (
    NAMED_AGENT_WORKSPACE_ROOT,
    ResolvedAgentConfig,
    named_agent_workspace,
    resolve_agent_config,
)

__all__ = [
    "NAMED_AGENT_WORKSPACE_ROOT",
    "ResolvedAgentConfig",
    "named_agent_workspace",
    "resolve_agent_config",
]

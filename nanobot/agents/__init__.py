"""Named agents: several first-class agents within one install and one config.

The agent nanobot has always run is the agent named ``default``. Agents declared
under ``agents.named`` are its full peers — each with its own workspace, memory,
sessions, model settings, tool set and Telegram bot — not subagents of it.
"""

from nanobot.agents.gateway import (
    AgentRuntime,
    InboundRouter,
    MultiAgentGateway,
    NamedAgentFleet,
    OutboundRecord,
    open_gateway,
)
from nanobot.agents.registry import (
    DEFAULT_AGENT_NAME,
    AgentSpec,
    agent_config,
    agent_names,
    agent_spec,
    named_agent_workspace,
    resolve_agent_specs,
)
from nanobot.agents.routing import (
    agent_channels,
    channel_agent_bindings,
    declared_agent_bindings,
    route,
)

__all__ = [
    "DEFAULT_AGENT_NAME",
    "AgentRuntime",
    "AgentSpec",
    "InboundRouter",
    "MultiAgentGateway",
    "NamedAgentFleet",
    "OutboundRecord",
    "agent_channels",
    "agent_config",
    "agent_names",
    "agent_spec",
    "channel_agent_bindings",
    "declared_agent_bindings",
    "named_agent_workspace",
    "open_gateway",
    "resolve_agent_specs",
    "route",
]

import json
import logging

from ..agent.types import Chat, Message, Reply, Role


logger = logging.getLogger(__name__)


ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Why this agent is the best fit for the query.",
        },
        "agent": {
            "type": "string",
            "description": "The name of the agent to handle the query.",
        },
        "needs_user_context": {
            "type": "boolean",
            "description": "Whether the query requires the user's location or timezone to answer (e.g. local weather, local time).",
        },
    },
    "required": ["reasoning", "agent", "needs_user_context"],
}


def _build_router_prompt(agents_cfg: dict) -> str:
    lines = ["Select the best agent to handle the user's query.", ""]
    lines.append("Available agents:")
    for name, cfg in agents_cfg.items():
        role = cfg.get("role", "general")
        lines.append(f"  - {name}: {role}")
    lines.append("")
    lines.append("Also decide if the query requires the user's location or timezone.")
    lines.append("Set needs_user_context to true only when the answer depends on where the user is")
    lines.append('(e.g. "what\'s the weather?", "what time is it?"), not for general questions.')
    lines.append("")
    lines.append('Respond with JSON: {"reasoning": "...", "agent": "...", "needs_user_context": true/false}')
    return "\n".join(lines)


async def route(
    model: Chat, query: str, agents_cfg: dict,
) -> tuple[str, str, bool, Reply]:
    """Route a query to the best agent.

    Returns (agent_name, reasoning, needs_user_context, reply).
    """
    system = _build_router_prompt(agents_cfg)
    messages = [
        Message(role=Role.SYSTEM, content=system),
        Message(role=Role.USER, content=query),
    ]

    reply = await model.query(messages, ROUTER_SCHEMA)

    agent_names = list(agents_cfg.keys())
    default_agent = agent_names[0]

    try:
        data = json.loads(reply.text)
        agent_name = data.get("agent", default_agent)
        reasoning = data.get("reasoning", "")
        needs_user_context = data.get("needs_user_context", False)
        if agent_name not in agents_cfg:
            logger.warning("Router returned unknown agent %r, falling back to %r", agent_name, default_agent)
            agent_name = default_agent
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Router parse error, falling back to %r", default_agent)
        agent_name = default_agent
        reasoning = "fallback"
        needs_user_context = False

    logger.info("Router: %r -> %s (context=%s) (%s)", query, agent_name, needs_user_context, reasoning)
    return agent_name, reasoning, needs_user_context, reply

import json
import logging

from ..agent.types import Chat, Message, Reply, Role


logger = logging.getLogger(__name__)


EVALUATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Analysis of whether the response adequately answers the query.",
        },
        "sufficient": {
            "type": "boolean",
            "description": "True if the response is good enough to return to the user.",
        },
        "next_agent": {
            "type": "string",
            "description": "If not sufficient, the name of the agent that should try next.",
        },
    },
    "required": ["reasoning", "sufficient"],
}


def _build_evaluator_prompt(agents_cfg: dict, current_agent: str) -> str:
    other_agents = {
        name: cfg for name, cfg in agents_cfg.items() if name != current_agent
    }
    lines = [
        "Evaluate whether the response adequately answers the user's query.",
        "",
        "IMPORTANT: Default to sufficient=true. Only mark as insufficient if:",
        "  - The response is factually wrong or completely off-topic",
        "  - The query clearly needed a DIFFERENT specialist (e.g. asked about stocks but got a weather agent)",
        "  - The agent failed to use a tool it clearly needed",
        "",
        "Mark as sufficient even if the response is:",
        "  - A simple greeting or acknowledgment (for greeting queries)",
        "  - Not perfect but reasonably addresses the query",
        "  - Short but correct",
        "",
    ]
    if other_agents:
        lines.append("Other available agents (for forwarding if insufficient):")
        for name, cfg in other_agents.items():
            role = cfg.get("role", "general")
            lines.append(f"  - {name}: {role}")
        lines.append("")
    lines.append('Respond with JSON: {"reasoning": "...", "sufficient": true/false, "next_agent": "..."}')
    return "\n".join(lines)


async def evaluate(
    model: Chat,
    query: str,
    response: str,
    agents_cfg: dict,
    current_agent: str,
) -> tuple[bool, str, str, Reply]:
    """Evaluate whether a response is sufficient.

    Returns (sufficient, reasoning, next_agent, reply).
    """
    system = _build_evaluator_prompt(agents_cfg, current_agent)
    user_content = f"Query: {query}\n\nResponse from '{current_agent}':\n{response}"
    messages = [
        Message(role=Role.SYSTEM, content=system),
        Message(role=Role.USER, content=user_content),
    ]

    reply = await model.query(messages, EVALUATOR_SCHEMA)

    other_agents = [name for name in agents_cfg if name != current_agent]
    default_next = other_agents[0] if other_agents else current_agent

    try:
        data = json.loads(reply.text)
        sufficient = data.get("sufficient", True)
        reasoning = data.get("reasoning", "")
        next_agent = data.get("next_agent", default_next)
        if next_agent not in agents_cfg or next_agent == current_agent:
            next_agent = default_next
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Evaluator parse error, defaulting to sufficient")
        sufficient = True
        reasoning = "parse error fallback"
        next_agent = default_next

    logger.info(
        "Evaluator: sufficient=%s, next=%s (%s)", sufficient, next_agent, reasoning,
    )
    return sufficient, reasoning, next_agent, reply

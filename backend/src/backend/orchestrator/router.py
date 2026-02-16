import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ..agent.types import Chat, Message, Reply, Role


logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    agent: str
    reasoning: str
    needs_user_context: bool
    clarification: str
    reply: Reply

    @property
    def needs_clarification(self) -> bool:
        return bool(self.clarification)


class _RouterSchema(BaseModel):
    reasoning         : str  = Field(description="Why this agent is the best fit for the query.")
    agent             : str  = Field(description="The name of the agent to handle the query.")
    needs_user_context: bool = Field(description="Whether the query requires the user's location or timezone to answer (e.g. local weather, local time).")
    clarification     : str  = Field(description="If the query is too vague or ambiguous to route confidently, a short question to ask the user. Empty string if the query is clear enough.")

_ROUTER_SCHEMA = _RouterSchema.model_json_schema()


_ROUTER_PROMPT = """\
Select the best agent to handle the user's query.

Available agents:
{agents}

Also decide if the query requires the user's location or timezone.
Set needs_user_context to true only when the answer depends on where the user is
(e.g. "what's the weather?", "what time is it?"), not for general questions.

If the query is too vague or ambiguous to answer well, set clarification to a short
question asking the user to be more specific. Only do this when the query is genuinely
unclear — most queries should get an empty clarification string.

Respond with JSON: {{"reasoning": "...", "agent": "...", "needs_user_context": true/false, "clarification": ""}}"""


def _build_router_prompt(agents_cfg: dict) -> str:
    agents = "\n".join(f"  - {name}: {cfg.get('role', 'general')}" for name, cfg in agents_cfg.items())
    return _ROUTER_PROMPT.format(agents=agents)


async def route(model: Chat, query: str, agents_cfg: dict) -> RouteResult:
    """Route a query to the best agent."""
    system = _build_router_prompt(agents_cfg)
    messages = [
        Message(role=Role.SYSTEM, content=system),
        Message(role=Role.USER, content=query),
    ]

    reply = await model.query(messages, _ROUTER_SCHEMA)

    agent_names = list(agents_cfg.keys())
    default_agent = agent_names[0]

    try:
        data = json.loads(reply.text)
        agent_name = data.get("agent", default_agent)
        reasoning = data.get("reasoning", "")
        needs_user_context = data.get("needs_user_context", False)
        clarification = data.get("clarification", "")
        if agent_name not in agents_cfg:
            logger.warning("Router returned unknown agent %r, falling back to %r", agent_name, default_agent)
            agent_name = default_agent
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Router parse error, falling back to %r", default_agent)
        agent_name = default_agent
        reasoning = "fallback"
        needs_user_context = False
        clarification = ""

    logger.info("Router: %r -> %s (context=%s, clarify=%r) (%s)", query, agent_name, needs_user_context, clarification, reasoning)
    return RouteResult(
        agent=agent_name,
        reasoning=reasoning,
        needs_user_context=needs_user_context,
        clarification=clarification,
        reply=reply,
    )

import logging
import operator
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ..llm import Message, Role, TokenUsage
from .moderator import Moderation, Moderator
from .reflector import Action, Reflection, Reflector
from .router import AgentRequest, Intent, Router


logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    intent: str = Intent.NONE  # Intent member or custom string for MCP tools
    model: str = ""
    answer: str = ""
    success: bool = True
    tools_used: list[str] = field(default_factory=list)


class State(TypedDict, total=False):
    query: str
    history: list[Message]
    model: Any  # Chat instance
    use_reflection: bool
    moderation: Moderation | None
    agent_requests: list[AgentRequest]
    agent_responses: list[AgentResponse]
    reflection: Reflection
    token_log: Annotated[list[TokenUsage], operator.add]


def build_state_graph(*, moderator: Moderator, router: Router, agents: dict, reflector: Reflector):
    builder = StateGraph(State)
    builder.add_node("moderation", partial(_moderation_node, moderator=moderator), destinations=("finalize", "router"))
    builder.add_node("router", partial(_router_node, router=router), destinations=("agent",))
    builder.add_node("agent", partial(_agent_node, agents=agents), destinations=("reflector", "check_next"))
    builder.add_node("reflector", partial(_reflector_node, reflector=reflector), destinations=("check_next",))
    builder.add_node("check_next", _check_next_node, destinations=("agent", "finalize"))
    builder.add_node("finalize", _finalize_node)
    builder.set_entry_point("moderation")

    return builder.compile()


async def _moderation_node(state: State, moderator: Moderator) -> Command:
    moderation = moderator.moderate(state["query"])
    goto = "finalize" if moderation.is_blocked else "router"
    return Command(
        update={"moderation": moderation},
        goto=goto,
    )


async def _router_node(state: State, router: Router) -> Command:
    intents, tokens = router.route(
        query=state["query"],
        history=state["history"],
    )

    return Command(
        update={
            "agent_requests": intents,
            "token_log": [tokens],
        },
        goto="agent",
    )


async def _agent_node(state: State, agents: dict) -> Command:
    refl = state.get("reflection", Reflection())
    is_retry = refl.action == Action.RETRY and refl.retry_query is not None

    agent_responses: list[AgentResponse] = list(state.get("agent_responses", []))
    idx = len(agent_responses) - 1 if is_retry else len(agent_responses)

    intent_data: AgentRequest = state["agent_requests"][idx]

    intent = intent_data.intent
    agent = agents.get(intent, agents[Intent.CHAT])

    params = intent_data.params

    query = params.get("query") or state["query"]
    history = state["history"]

    if is_retry:
        query = refl.retry_query
        history = refl.retry_history or history

    # TalkAgent uses the user's model; RAG/MCP have their own (model kwarg ignored via **_).
    result = await agent.execute(
        query=query,
        model=state["model"],
        history=history,
        params=params,
    )

    entry = AgentResponse(
        intent=intent,
        model=result.model,
        answer=result.response,
        success=getattr(result, "success", True),
        tools_used=getattr(result, "tools_used", []),
    )

    if is_retry and agent_responses:
        agent_responses[-1] = entry
    else:
        agent_responses.append(entry)

    tokens = TokenUsage(
        model=result.model,
        input_tokens=getattr(result, "input_tokens", 0),
        output_tokens=getattr(result, "output_tokens", 0),
    )  # Agent results aren't Response — manual construction

    updates: dict = {
        "agent_responses": agent_responses,
        "token_log": [tokens],
    }
    if not is_retry:
        updates["reflection"] = Reflection()

    used_tools = getattr(result, "tools_used", [])
    skip_reflection = intent_data.intent == Intent.SMALLTALK and not used_tools
    goto = "reflector" if state.get("use_reflection", False) and not skip_reflection else "check_next"
    return Command(update=updates, goto=goto)


async def _reflector_node(state: State, reflector: Reflector) -> Command:
    agent_responses = state.get("agent_responses", [])
    idx = len(agent_responses) - 1
    latest = agent_responses[idx] if agent_responses else None

    intents: list[AgentRequest] = state.get("agent_requests", [])
    intent_data = intents[idx] if idx < len(intents) else AgentRequest()

    agent_response = latest.answer if latest else ""
    agent_success = latest.success if latest else True

    # For multi-intent queries, scope evaluation to the current intent
    params = intent_data.params
    if len(intents) > 1:
        params_desc = ", ".join(f"{k}={v}" for k, v in params.items())
        intent_query = f"{intent_data.intent}: {params_desc}" if params_desc else state["query"]
    else:
        intent_query = params.get("query") or state["query"]

    info, tokens = reflector.reflect(
        query=intent_query,
        response=agent_response,
        intent=intent_data.intent,
        success=agent_success,
    )

    # For multi-intent queries, convert retry to accept to avoid looping
    # on a single intent's details.
    if len(intents) > 1 and info["action"] == Action.RETRY:
        logger.info("Multi-intent query: converting retry to accept")
        info["action"] = Action.ACCEPT

    prev_refl = state.get("reflection", Reflection())
    reflection_count = prev_refl.count + 1

    retry_query = None
    retry_history = None

    if info["action"] == Action.RETRY and reflection_count < reflector.max_reflections:
        retry_query = f"Previous reply had an error: {info['feedback']}"
        retry_history = list(state["history"]) + [
            Message(role=Role.USER, content=state["query"]),
            Message(role=Role.ASSISTANT, content=agent_response),
        ]

    updates: dict = {
        "token_log": [tokens],
        "reflection": Reflection(
            count=reflection_count,
            retry_query=retry_query,
            retry_history=retry_history,
            **info,
        ),
    }

    return Command(update=updates, goto="check_next")


async def _check_next_node(state: State) -> Command:
    refl = state.get("reflection", Reflection())
    if refl.action == Action.RETRY and refl.retry_query:
        return Command(update={}, goto="agent")

    requests = state.get("agent_requests", [])
    goto = "agent" if len(state.get("agent_responses", [])) < len(requests) else "finalize"
    return Command(update={}, goto=goto)


async def _finalize_node(state: State) -> Command:
    moderation = state.get("moderation")
    if moderation and moderation.is_blocked:
        agent_responses = [
            AgentResponse(
                intent=Intent.BLOCKED,
                answer="I'm sorry, but I can't assist with that request.",
                success=False,
            ),
        ]
    else:
        agent_responses = state.get("agent_responses", [])

    return Command(update={
        "agent_responses": agent_responses,
        "moderation": moderation,
    }, goto=END)

import logging
from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.types import Command

from .moderator import Moderator
from .reflector import Reflector
from .router import Router
from .state import (
    Action,
    AgentRequest,
    AgentResult,
    Intent,
    Reflection,
    State,
    TokenUsage,
)


logger = logging.getLogger(__name__)


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
            "routing": intents,
            "token_log": [tokens],
        },
        goto="agent",
    )


async def _agent_node(state: State, agents: dict) -> Command:
    refl = state.get("reflection", Reflection())
    is_retry = refl.action == Action.RETRY and refl.retry_query is not None

    agent_results: list[AgentResult] = list(state.get("agent_results", []))
    idx = len(agent_results) - 1 if is_retry else len(agent_results)

    intent_data: AgentRequest = state["routing"][idx]

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

    entry = AgentResult(
        intent=intent,
        model=result.model,
        answer=result.response,
        success=getattr(result, "success", True),
        tools_used=getattr(result, "tools_used", []),
    )

    if is_retry and agent_results:
        agent_results[-1] = entry
    else:
        agent_results.append(entry)

    tokens = TokenUsage(
        model=result.model,
        input_tokens=getattr(result, "input_tokens", 0),
        output_tokens=getattr(result, "output_tokens", 0),
    )

    updates: dict = {
        "agent_results": agent_results,
        "token_log": [tokens],
    }
    if not is_retry:
        updates["reflection"] = Reflection()

    used_tools = getattr(result, "tools_used", [])
    skip_reflection = intent_data.intent == Intent.SMALLTALK and not used_tools
    goto = "reflector" if state.get("use_reflection", False) and not skip_reflection else "check_next"
    return Command(update=updates, goto=goto)


async def _reflector_node(state: State, reflector: Reflector) -> Command:
    agent_results = state.get("agent_results", [])
    idx = len(agent_results) - 1
    latest = agent_results[idx] if agent_results else None

    intents: list[AgentRequest] = state.get("routing", [])
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
            {"role": "user", "content": state["query"]},
            {"role": "assistant", "content": agent_response},
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

    routing = state.get("routing", [])
    goto = "agent" if len(state.get("agent_results", [])) < len(routing) else "finalize"
    return Command(update={}, goto=goto)


async def _finalize_node(state: State) -> Command:
    moderation = state.get("moderation")
    if moderation and moderation.is_blocked:
        agent_results = [
            AgentResult(
                intent=Intent.BLOCKED,
                answer="I'm sorry, but I can't assist with that request.",
                success=False,
            ),
        ]
    else:
        agent_results = state.get("agent_results", [])

    return Command(update={
        "agent_results": agent_results,
        "moderation": moderation,
    }, goto=END)

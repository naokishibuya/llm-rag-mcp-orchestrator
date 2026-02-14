import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ..core import Message, Reply, Role, Tokens, UserContext
from .moderator import Moderation, Moderator
from .reflector import Action, Reflection, Reflector
from .router import Intent, Route, Router


logger = logging.getLogger(__name__)


class State(TypedDict, total=False):
    query: str
    history: list[Message]
    model: Any  # Chat instance
    use_reflection: bool
    context: UserContext
    moderation: Moderation | None
    routes: list[Route]
    cursor: int
    reflection: Reflection
    token_log: Annotated[list[tuple[str, Tokens]], operator.add]


class Nodes:
    def __init__(self, state: State):
        self._state = state

    @property
    def query(self) -> str:
        return self._state["query"]

    @property
    def history(self) -> list[Message]:
        return self._state["history"]

    @property
    def cursor(self) -> int:
        return self._state.get("cursor", 0)

    @property
    def route(self) -> Route:
        return self._state["routes"][self.cursor]

    @property
    def reflection(self) -> Reflection:
        return self._state.get("reflection", Reflection())

    @property
    def is_multi(self) -> bool:
        return len(self._state["routes"]) > 1

    def replace_route(self, **kwargs) -> list[Route]:
        routes = list(self._state["routes"])
        routes[self.cursor] = Route(**kwargs)
        return routes

    def token_entry(self, reply: Reply) -> list[tuple[str, Tokens]]:
        return [(reply.model, reply.tokens)]

    async def moderation(self, moderator: Moderator) -> Command:
        result = moderator.moderate(self.query)
        goto = "finalize" if result.is_blocked else "router"
        return Command(update={"moderation": result}, goto=goto)

    async def routing(self, router: Router) -> Command:
        routes, reply = await router.route(query=self.query, history=self.history, context=self._state["context"])
        return Command(
            update={"routes": routes, "token_log": self.token_entry(reply)},
            goto="agent",
        )

    async def agent(self, agents: dict) -> Command:
        refl = self.reflection
        route = self.route

        query = refl.retry_query if refl.is_retry else (route.params.get("query") or self.query)
        history = (refl.retry_history or self.history) if refl.is_retry else self.history

        # TalkAgent uses the user's model; RAG/MCP have their own (model kwarg ignored via **_).
        agent = agents.get(route.intent, agents[Intent.CHAT])
        reply = await agent.act(
            query=query, model=self._state["model"], history=history,
            context=self._state["context"], params=route.params,
        )

        routes = self.replace_route(intent=route.intent, params=route.params, reply=reply)

        updates: dict = {"routes": routes, "cursor": self.cursor, "token_log": self.token_entry(reply)}
        if not refl.is_retry:
            updates["reflection"] = Reflection()

        skip = route.intent == Intent.SMALLTALK and not reply.tools_used
        should_reflect = self._state.get("use_reflection", False) and not skip
        return Command(update=updates, goto="reflector" if should_reflect else "check_next")

    async def reflect(self, reflector: Reflector) -> Command:
        route = self.route

        # For multi-intent queries, scope evaluation to the current intent
        params = route.params
        if self.is_multi:
            params_desc = ", ".join(f"{k}={v}" for k, v in params.items())
            intent_query = f"{route.intent}: {params_desc}" if params_desc else self.query
        else:
            intent_query = params.get("query") or self.query

        info, reply = await reflector.reflect(
            query=intent_query,
            response=route.reply.text,
            intent=route.intent,
            context=self._state["context"],
            success=route.reply.success,
        )

        # For multi-intent queries, convert retry to accept to avoid looping
        # on a single intent's details.
        if self.is_multi and info["action"] == Action.RETRY:
            logger.info("Multi-intent query: converting retry to accept")
            info["action"] = Action.ACCEPT

        reflection_count = self.reflection.count + 1

        retry_query = None
        retry_history = None

        if info["action"] == Action.RETRY and reflection_count < reflector.max_reflections:
            retry_query = f"Previous reply had an error: {info['feedback']}"
            retry_history = list(self.history) + [
                Message(role=Role.USER, content=self.query),
                Message(role=Role.ASSISTANT, content=route.reply.text),
            ]

        return Command(
            update={
                "token_log": self.token_entry(reply),
                "reflection": Reflection(
                    count=reflection_count,
                    retry_query=retry_query,
                    retry_history=retry_history,
                    **info,
                ),
            },
            goto="check_next",
        )

    async def check_next(self) -> Command:
        if self.reflection.is_retry:
            return Command(update={}, goto="agent")

        cursor = self.cursor + 1
        goto = "agent" if cursor < len(self._state["routes"]) else "finalize"
        return Command(update={"cursor": cursor}, goto=goto)

    async def finalize(self) -> Command:
        moderation = self._state.get("moderation")
        if moderation and moderation.is_blocked:
            routes = [Route(intent=Intent.BLOCKED, reply=Reply(text="I'm sorry, but I can't assist with that request.", success=False))]
        else:
            routes = self._state.get("routes", [])

        return Command(update={"routes": routes, "moderation": moderation}, goto=END)


def build_state_graph(*, moderator: Moderator, router: Router, agents: dict, reflector: Reflector):
    async def moderation_node(state):
        return await Nodes(state).moderation(moderator)

    async def router_node(state):
        return await Nodes(state).routing(router)

    async def agent_node(state):
        return await Nodes(state).agent(agents)

    async def reflector_node(state):
        return await Nodes(state).reflect(reflector)

    async def check_next_node(state):
        return await Nodes(state).check_next()

    async def finalize_node(state):
        return await Nodes(state).finalize()

    builder = StateGraph(State)
    builder.add_node("moderation", moderation_node, destinations=("finalize", "router"))
    builder.add_node("router", router_node, destinations=("agent",))
    builder.add_node("agent", agent_node, destinations=("reflector", "check_next"))
    builder.add_node("reflector", reflector_node, destinations=("check_next",))
    builder.add_node("check_next", check_next_node, destinations=("agent", "finalize"))
    builder.add_node("finalize", finalize_node)
    builder.set_entry_point("moderation")

    return builder.compile()

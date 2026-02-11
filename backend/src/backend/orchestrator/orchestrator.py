import logging
from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ..config import Config
from ..llm import Chat, Registry
from ..mcp import MCPAgent, MCPClient, MCPHandler
from ..rag import RAGAgent
from ..talk import TalkAgent
from .moderator import Moderator
from .reflector import Reflector
from .router import Router, RoutingMeta
from .state import AgentState

logger = logging.getLogger(__name__)


def _extract_params(input_schema: dict) -> dict[str, str]:
    props = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    return {name: meta.get("description", name) for name, meta in props.items() if name in required}


async def _agent_node(state: dict, agents: dict, agent_models: dict) -> Command:
    idx = state["current_intent_index"]
    intent_data = state["intents"][idx]

    agent_name = intent_data.get("agent", "TalkAgent")
    agent = agents.get(agent_name, agents["TalkAgent"])

    params = intent_data.get("params", {})

    query = params.get("query") or state["query"]
    history = state["history"]

    feedback = state.get("reflection_feedback")
    if feedback and feedback.get("query"):
        query = feedback["query"]
    if feedback and feedback.get("history"):
        history = feedback["history"]

    # Pick model: TalkAgent uses user's choice, others use their pipeline model.
    model_key = agent_models.get(agent_name, "mcp_model")
    model = state[model_key]

    result = await agent.execute(
        query=query,
        model=model,
        history=history,
        params=params,
        embedder=state["embedder"],
    )
    is_reflection = feedback and feedback.get("action") in ("retry", "reroute")

    # On retry/reroute, replace the last response; otherwise append.
    responses = list(state.get("intent_responses", []))
    if is_reflection and responses:
        responses[-1] = result.response
    else:
        responses.append(result.response)

    updates = {
        "agent_response": result.response,
        "agent_success": getattr(result, "success", True),
        "intent_responses": responses,
        "current_intent_index": idx + 1,
        "reflection_feedback": None,
        "input_tokens": getattr(result, "input_tokens", 0),
        "output_tokens": getattr(result, "output_tokens", 0),
    }
    if not is_reflection:
        updates["reflection_count"] = 0

    goto = "reflector" if state.get("use_reflection", False) else "check_next"
    return Command(update=updates, goto=goto)


async def _check_next_node(state: dict) -> Command:
    idx = state.get("current_intent_index", 0)
    goto = "agent" if idx < len(state.get("intents", [])) else "finalize"
    return Command(goto=goto)


async def _finalize_node(state: dict) -> Command:
    responses = state.get("intent_responses", [])
    if len(responses) == 0:
        final = state.get("agent_response", "I couldn't process your request.")
    elif len(responses) == 1:
        final = responses[0]
    else:
        final = "\n\n".join(responses)
    return Command(update={"final_answer": final}, goto=END)


async def _blocked_node(state: dict) -> Command:
    return Command(
        update={"final_answer": "I'm sorry, but I can't assist with that request."},
        goto=END,
    )


class Orchestrator:
    def __init__(self, config: Config):
        self._config = config
        self._registry = Registry(config)
        self._mcp_client: MCPClient | None = None
        self._graph = None
        self._agents: dict = {}
        self._router: Router | None = None
        self._discovered_servers: set[str] = set()

        # Resolve pipeline models once from config.
        self._orchestrator_model: Chat = self._registry.resolve_model("orchestrator")
        self._rag_model: Chat = self._registry.resolve_model("rag")
        self._mcp_model: Chat = self._registry.resolve_model("mcp")

    async def initialize(self):
        mcp_client = MCPClient(self._config.mcp_services)
        await mcp_client.connect()
        discovered = await mcp_client.discover()
        self._mcp_client = mcp_client

        # Build agents: hardcoded non-MCP agents + dynamic MCP handlers
        self._agents = {
            "TalkAgent": TalkAgent(),
            "RAGAgent": RAGAgent(),
        }

        # Map agent name → state key for model selection
        agent_models = {
            "TalkAgent": "model",
            "RAGAgent": "rag_model",
        }

        routing: dict[str, RoutingMeta] = {
            "TalkAgent": RoutingMeta("chat", "General conversation, greetings", {}),
            "RAGAgent": RoutingMeta("rag", "Knowledge base / documentation questions", {"query": "question"}),
        }

        for service in discovered:
            agent_name = f"{service.server}_{service.name}"
            handler = MCPHandler(mcp_client, service)
            self._agents[agent_name] = MCPAgent(handler, service)
            agent_models[agent_name] = "mcp_model"
            routing[agent_name] = RoutingMeta(
                service.name,
                service.description,
                _extract_params(service.input_schema),
            )
            self._discovered_servers.add(service.server)
            logger.info(f"Registered MCP agent: {agent_name}")

        self._agent_models = agent_models
        self._graph = self._build_graph(self._agents, agent_models, routing)
        logger.info("Orchestrator initialized with %d agents", len(self._agents))

    def _build_graph(self, agents: dict, agent_models: dict, routing: dict[str, RoutingMeta]):
        moderator = Moderator()
        router = Router(routing)
        reflector = Reflector(available_agents=router.agent_descriptions)
        self._router = router

        builder = StateGraph(AgentState)
        builder.add_node("safety", moderator, destinations=("blocked", "router"))
        builder.add_node("router", router, destinations=("agent",))
        builder.add_node("agent", partial(_agent_node, agents=agents, agent_models=agent_models), destinations=("reflector", "check_next"))
        builder.add_node("reflector", reflector, destinations=("agent", "router", "check_next"))
        builder.add_node("check_next", _check_next_node, destinations=("agent", "finalize"))
        builder.add_node("finalize", _finalize_node)
        builder.add_node("blocked", _blocked_node)
        builder.set_entry_point("safety")

        return builder.compile()

    async def shutdown(self):
        if self._mcp_client:
            await self._mcp_client.close()
            self._mcp_client = None
            logger.info("Orchestrator shut down")

    async def _refresh_services(self):
        """Try to discover tools from MCP servers that weren't available at startup."""
        if not self._mcp_client or not self._router:
            return
        pending = self._mcp_client.servers - self._discovered_servers
        if not pending:
            return

        newly_connected: set[str] = set()
        for server in pending:
            if await self._mcp_client.reconnect(server):
                newly_connected.add(server)
        if not newly_connected:
            return

        new_tools = await self._mcp_client.discover(servers=newly_connected)
        new_routing: dict[str, RoutingMeta] = {}
        for service in new_tools:
            agent_name = f"{service.server}_{service.name}"
            handler = MCPHandler(self._mcp_client, service)
            self._agents[agent_name] = MCPAgent(handler, service)
            self._agent_models[agent_name] = "mcp_model"
            new_routing[agent_name] = RoutingMeta(
                service.name,
                service.description,
                _extract_params(service.input_schema),
            )
            logger.info(f"Late-discovered MCP agent: {agent_name}")
        self._discovered_servers.update(newly_connected)

        if new_routing:
            self._router.add_routes(new_routing)

    async def invoke(
        self,
        query: str,
        history: list,
        model_name: str,
        embedding_model: str,
        use_reflection: bool = False,
    ) -> dict:
        if not self._graph:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        await self._refresh_services()

        model = self._registry.get_talk_model(model_name)
        embedder = self._registry.get_embeddings(embedding_model)

        initial_state = {
            # Input
            "query": query,
            "history": history,
            "model": model,
            "orchestrator_model": self._orchestrator_model,
            "rag_model": self._rag_model,
            "mcp_model": self._mcp_model,
            "embedder": embedder,
            "use_reflection": use_reflection,
            "max_reflections": 2,
            # Multi-intent
            "intents": [],
            "current_intent_index": 0,
            # Execution
            "agent_response": "",
            "agent_success": True,
            "intent_responses": [],
            "tool_results": {},
            # Reflection
            "reflection_count": 0,
            "reflection_feedback": None,
            # Tokens (initialized for add reducer)
            "input_tokens": 0,
            "output_tokens": 0,
            # Safety
            "is_blocked": False,
            "moderation_reason": None,
            # Output
            "final_answer": "",
        }

        return await self._graph.ainvoke(initial_state)

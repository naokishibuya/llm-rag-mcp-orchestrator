import logging

from ..config import Config
from ..llm import Registry
from ..rag import RAGAgent, RAGClient
from ..talk import TalkAgent
from .moderator import Moderator
from ..core import UserContext
from .nodes import State, build_state_graph
from .reflector import Reflection, Reflector
from .router import Intent
from .router import Router, RoutingInfo
from .services import ServiceRegistry


logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: Config):
        self._config = config
        self._registry = Registry(config)
        self._services: ServiceRegistry | None = None
        self._router: Router | None = None
        self._graph = None

    async def startup(self):
        orchestrator_model = self._registry.resolve_model("orchestrator")

        self._services = ServiceRegistry(
            self._config.mcp_services,
            self._registry.resolve_model("mcp"),
        )
        await self._services.startup()

        agents = {
            Intent.CHAT: TalkAgent(),
            Intent.SMALLTALK: TalkAgent(),
            Intent.RAG: RAGAgent(
                self._registry.resolve_model("rag"),
                RAGClient(self._registry.resolve_embeddings()),
                top_k=self._config.rag_top_k),
            **self._services.agents,
        }

        routing = {
            Intent.CHAT: RoutingInfo("Complex general knowledge questions, explanations, or creative writing", {}),
            Intent.SMALLTALK: RoutingInfo("Simple greetings, social pleasantries, or very brief small talk", {}),
            Intent.RAG: RoutingInfo("Knowledge base / documentation questions", {"query": "question"}),
            **self._services.routing,
        }
        self._router = Router(routing, model=orchestrator_model)

        reflector = Reflector(
            model=orchestrator_model,
            max_reflections=self._config.max_reflections,
        )

        moderator = Moderator()

        self._graph = build_state_graph(
            moderator=moderator,
            router=self._router,
            agents=agents,
            reflector=reflector,
        )
        print(self._graph.get_graph().draw_mermaid())
        logger.info("Orchestrator started with %d agents", len(agents))

    async def shutdown(self):
        if self._services:
            await self._services.close()
            self._services = None
            logger.info("Orchestrator shut down")

    async def stream(
        self,
        query: str,
        history: list,
        model_name: str,
        context: UserContext,
        use_reflection: bool = False,
    ):
        """Async generator yielding (node_name, updates) for each graph step."""
        if not self._graph:
            raise RuntimeError("Orchestrator not started. Call startup() first.")

        if self._services and self._router:
            await self._services.refresh(self._router)

        initial_state = State(
            query=query,
            history=history,
            model=self._registry.get_talk_model(model_name),
            use_reflection=use_reflection,
            context=context,
            moderation=None,
            routes=[],
            cursor=0,
            reflection=Reflection(),
            token_log=[],
        )
        async for event in self._graph.astream(initial_state, stream_mode="updates"):
            for node_name, updates in event.items():
                yield node_name, updates or {}


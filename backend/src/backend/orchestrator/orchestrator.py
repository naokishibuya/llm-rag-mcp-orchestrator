import logging

from ..agent import Agent, Reply, UserContext
from ..agent.llm import Registry
from ..agent.mcp import MCPClient, MCPHandler
from ..agent.rag import RAGClient
from ..agent.tools import build_tools
from ..agent.types import Message
from ..config import Config
from .moderator import Moderator


logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: Config):
        self._config = config
        self._registry = Registry(config.llm, config.embedding)
        self._mcp_client: MCPClient | None = None
        self._mcp_handlers: dict[str, tuple[MCPHandler, object]] = {}
        self._rag_client: RAGClient | None = None
        self._agent: Agent | None = None
        self._moderator = Moderator()

    async def startup(self):
        # Build agent from config
        agent_cfg = self._config.agents.get("chat", {})
        self._agent = Agent(
            "chat",
            agent_cfg.get("system_prompt", "You are a helpful assistant."),
        )

        # Setup RAG
        self._rag_client = RAGClient(
            self._registry.resolve_embeddings(),
            topics=self._config.rag_topics,
        )

        # Setup MCP
        mcp_services = self._config.mcp_services
        if mcp_services:
            self._mcp_client = MCPClient(mcp_services)
            await self._mcp_client.connect()
            for service in await self._mcp_client.discover():
                handler = MCPHandler(self._mcp_client, service)
                self._mcp_handlers[service.name] = (handler, service)

        logger.info(
            "Orchestrator started: agent=%s, mcp_tools=%d",
            self._agent.name, len(self._mcp_handlers),
        )

    async def shutdown(self):
        if self._mcp_client:
            await self._mcp_client.close()
            self._mcp_client = None
            logger.info("Orchestrator shut down")

    async def stream(
        self,
        query: str,
        history: list[Message],
        model_name: str,
        context: UserContext,
        **_,
    ):
        """Async generator yielding (event_name, data) tuples."""
        if not self._agent:
            raise RuntimeError("Orchestrator not started. Call startup() first.")

        # Refresh MCP connections for servers that weren't available at startup
        await self._refresh_mcp()

        # Moderation
        moderation = self._moderator.moderate(query)
        yield "moderation", {"moderation": moderation}

        if moderation.is_blocked:
            reply = Reply(text="I'm sorry, but I can't assist with that request.", success=False)
            yield "agent", {"reply": reply}
            yield "done", {"moderation": moderation}
            return

        # Build tools for this request
        tools = build_tools(
            context=context,
            rag_client=self._rag_client,
            rag_top_k=self._config.rag_top_k,
            mcp_handlers=self._mcp_handlers,
        )

        # Run agent
        model = self._registry.get_talk_model(model_name)
        reply = await self._agent.act(
            model=model, query=query, history=history, context=context, tools=tools,
        )
        yield "agent", {"reply": reply}
        yield "done", {"moderation": moderation}

    async def _refresh_mcp(self):
        """Try to connect MCP servers that failed at startup."""
        if not self._mcp_client:
            return
        pending = self._mcp_client.servers - {
            s.server for _, s in self._mcp_handlers.values()
        }
        if not pending:
            return
        newly_connected: set[str] = set()
        for server in pending:
            if await self._mcp_client.reconnect(server):
                newly_connected.add(server)
        if not newly_connected:
            return
        for service in await self._mcp_client.discover(servers=newly_connected):
            handler = MCPHandler(self._mcp_client, service)
            self._mcp_handlers[service.name] = (handler, service)
            logger.info("Registered MCP tool: %s", service.name)

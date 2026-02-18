import logging

from ..agent import Agent, Reply, UserContext
from ..agent.llm import Registry
from ..agent.mcp import MCPClient, MCPHandler
from ..agent.rag import RAGClient
from ..agent.tools import build_tools, filter_tools
from ..agent.types import Message
from ..config import Config
from .evaluator import evaluate
from .moderator import Moderator
from .router import route


logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: Config):
        self._config = config
        self._registry = Registry(config.llm, config.embedding)
        self._mcp_client: MCPClient | None = None
        self._mcp_handlers: dict[str, tuple[MCPHandler, object]] = {}
        self._rag_client: RAGClient | None = None
        self._agents: dict[str, Agent] = {}
        self._moderator = Moderator()

    async def startup(self):
        # Build agents from config
        for name, cfg in self._config.agents.items():
            self._agents[name] = Agent(
                name,
                cfg.get("system_prompt", "You are a helpful assistant."),
            )

        # Setup RAG
        self._rag_client = RAGClient(
            self._registry.resolve_embeddings(),
            **self._config.rag_params,
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
            "Orchestrator started: agents=%s, mcp_tools=%d",
            list(self._agents.keys()), len(self._mcp_handlers),
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
        if not self._agents:
            raise RuntimeError("Orchestrator not started. Call startup() first.")

        # Refresh MCP connections for servers that weren't available at startup
        await self._refresh_mcp()

        # Moderation
        moderation = self._moderator.moderate(query)
        yield "moderation", {"moderation": moderation}

        if moderation.is_blocked:
            reply = Reply(text="I'm sorry, but I can't assist with that request.", success=False)
            yield "agent", {"reply": reply, "agent_name": "moderation"}
            yield "done", {"moderation": moderation}
            return

        # Build all tools once
        all_tools = build_tools(
            context=context,
            rag_client=self._rag_client,
            mcp_handlers=self._mcp_handlers)

        model = self._registry.get_talk_model(model_name)
        agents_cfg = self._config.agents
        max_forwards = self._config.workflow.get("max_forwards", 2)

        # Route
        result = await route(model, query, agents_cfg)
        yield "thinking", {
            "step": f"Router \u2192 {result.agent}",
            "detail": result.reasoning,
            "tokens": result.reply.tokens,
            "model": result.reply.model,
        }

        # Clarification — return early if query is too vague
        if result.needs_clarification:
            reply = Reply(text=result.clarification, model=result.reply.model, tokens=result.reply.tokens)
            yield "agent", {"reply": reply, "agent_name": result.agent}
            yield "done", {"moderation": moderation}
            return

        # Agent loop
        agent_name = result.agent
        agent_context = context if result.needs_user_context else None
        reply = None
        agents_used: list[str] = []
        for attempt in range(max_forwards + 1):
            # Run agent with filtered tools
            tool_names = agents_cfg[agent_name].get("tools", "all")
            agent_tools = filter_tools(all_tools, tool_names)
            agents_used.append(agent_name)
            reply = await self._agents[agent_name].act(
                model=model, query=query, history=history,
                tools=agent_tools, context=agent_context,
            )

            # Yield thinking for each tool used
            for tool_name in reply.tools_used:
                yield "thinking", {"step": f"Tool[{agent_name}]: {tool_name}"}

            # Yield agent reply as thinking (visible in UI before evaluation)
            yield "thinking", {
                "step": f"Agent[{agent_name}]",
                "detail": reply.text,
                "tokens": reply.tokens,
                "model": reply.model,
            }

            # Last attempt — skip evaluation
            if attempt >= max_forwards:
                break

            # Evaluate
            sufficient, eval_reasoning, next_agent, eval_reply = await evaluate(
                model, query, reply.text, agents_cfg, agent_name,
            )
            status = "\u2713 sufficient" if sufficient else f"\u2717 forwarding \u2192 {next_agent}"
            yield "thinking", {
                "step": f"Evaluator: {status}",
                "detail": eval_reasoning,
                "tokens": eval_reply.tokens,
                "model": eval_reply.model,
            }

            if sufficient:
                break
            agent_name = next_agent

        # Final answer — collect unique disclaimers from all agents in the chain
        disclaimers = dict.fromkeys(
            agents_cfg[a]["disclaimer"] for a in agents_used if agents_cfg[a].get("disclaimer")
        )
        disclaimer = "\n".join(disclaimers) if disclaimers else None
        yield "agent", {"reply": reply, "agent_name": agent_name, "disclaimer": disclaimer}
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

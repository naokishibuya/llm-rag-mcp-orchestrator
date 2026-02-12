import logging

from ..llm import Chat
from ..mcp import MCPAgent, MCPClient, MCPHandler
from .router import Router, RoutingInfo


logger = logging.getLogger(__name__)


class ServiceRegistry:
    def __init__(self, mcp_config: dict, mcp_model: Chat):
        self._mcp_model = mcp_model
        self._client = MCPClient(mcp_config)
        self._discovered: set[str] = set()
        self.agents: dict = {}
        self.routing: dict[str, RoutingInfo] = {}

    async def startup(self):
        await self._client.connect()
        for service in await self._client.discover():
            self._register(service)

    def _register(self, service):
        handler = MCPHandler(self._client, service)
        self.agents[service.name] = MCPAgent(handler, service, self._mcp_model)
        self.routing[service.name] = RoutingInfo(
            service.description, _extract_params(service.input_schema),
        )
        self._discovered.add(service.server)
        logger.info("Registered MCP intent: %s", service.name)

    async def refresh(self, router: Router):
        pending = self._client.servers - self._discovered
        if not pending:
            return
        newly_connected: set[str] = set()
        for server in pending:
            if await self._client.reconnect(server):
                newly_connected.add(server)
        if not newly_connected:
            return
        new_routing: dict[str, RoutingInfo] = {}
        for service in await self._client.discover(servers=newly_connected):
            self._register(service)
            new_routing[service.name] = self.routing[service.name]
        self._discovered.update(newly_connected)
        if new_routing:
            router.add_routes(new_routing)

    async def close(self):
        await self._client.close()


def _extract_params(input_schema: dict) -> dict[str, str]:
    props = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    return {name: meta.get("description", name) for name, meta in props.items() if name in required}

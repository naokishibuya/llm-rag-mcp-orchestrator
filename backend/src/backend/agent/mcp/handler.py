import logging

from .client import MCPClient, MCPService


logger = logging.getLogger(__name__)


class MCPHandler:
    def __init__(self, client: MCPClient, service: MCPService):
        self.client = client
        self.service = service

    async def handle(self, **params) -> dict:
        try:
            result = await self.client.call(
                self.service.server, self.service.name, params,
            )
        except ConnectionError as e:
            logger.error(f"MCP server unavailable for {self.service.server}/{self.service.name}: {e}")
            return {"error": str(e), "unavailable": True}
        except Exception as e:
            logger.error(f"MCP call failed for {self.service.server}/{self.service.name}: {e}")
            return {"error": f"The {self.service.name} tool encountered an error: {e}"}

        data = result.structured_content
        if isinstance(data, dict) and "error" in data:
            logger.error(f"Tool {self.service.name} returned error: {data['error']}")
        return data

import asyncio
import logging
import os
import re
from dataclasses import dataclass

from fastmcp.client.client import CallToolResult, Client
from fastmcp.client.transports import StreamableHttpTransport


logger = logging.getLogger(__name__)


@dataclass
class MCPService:
    server: str
    name: str
    description: str
    input_schema: dict
    format_hint: str = ""


DEFAULT_TIMEOUT = 15  # seconds


class MCPClient:
    def __init__(self, servers: dict):
        if not servers:
            raise ValueError("No MCP servers configured")

        self._configs: dict[str, dict] = {}
        for name, config in servers.items():
            url = config.get("url")
            if not url:
                raise ValueError(f"MCP server '{name}' is missing a URL")

            # Resolve ${ENV_VAR} placeholders in URL
            url = re.sub(
                r"\$\{(\w+)\}",
                lambda m: os.environ.get(m.group(1), ""),
                url,
            )

            self._configs[name] = {
                "url": url,
                "format": config.get("format", ""),
                "timeout": config.get("timeout", DEFAULT_TIMEOUT),
            }

        self._clients: dict[str, Client] = {}

    async def connect(self):
        await asyncio.gather(*(self._connect_one(name) for name in self._configs))

    async def _connect_one(self, name: str):
        cfg = self._configs[name]
        try:
            transport = StreamableHttpTransport(url=cfg["url"])
            client = Client(transport)
            await client.__aenter__()
            self._clients[name] = client
            logger.info(f"Connected to MCP server: {name}")
        except Exception as e:
            logger.warning(f"Failed to connect to MCP server '{name}': {e}")

    async def reconnect(self, server: str) -> bool:
        if server in self._clients:
            return True
        logger.info(f"Attempting to reconnect to MCP server: {server}")
        await self._connect_one(server)
        return server in self._clients

    @property
    def servers(self) -> set[str]:
        return set(self._configs.keys())

    async def discover(self, servers: set[str] | None = None) -> list[MCPService]:
        tools: list[MCPService] = []
        for name, client in self._clients.items():
            if servers is not None and name not in servers:
                continue
            try:
                server_tools = await client.list_tools()
                for tool in server_tools:
                    discovered = MCPService(
                        server=name,
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema or {},
                        format_hint=self._configs[name].get("format", ""),
                    )
                    tools.append(discovered)
                    logger.info(f"Discovered tool: {name}/{tool.name}")
            except Exception as e:
                logger.warning(f"Failed to discover tools from '{name}': {e}")
        return tools

    async def call(self, server: str, tool: str, args: dict) -> CallToolResult:
        timeout = self._configs[server]["timeout"]
        client = self._clients.get(server)
        if not client:
            if not await self.reconnect(server):
                raise ConnectionError(f"MCP server '{server}' is not available")
            client = self._clients[server]
        try:
            result = await asyncio.wait_for(client.call_tool(tool, args), timeout=timeout)
        except Exception:
            # Connection may have dropped — evict stale client and retry once
            self._clients.pop(server, None)
            if not await self.reconnect(server):
                raise ConnectionError(f"MCP server '{server}' is not available")
            result = await asyncio.wait_for(
                self._clients[server].call_tool(tool, args), timeout=timeout,
            )
        logger.info(f"MCP call {server}/{tool} completed")
        return result

    async def close(self):
        for name, client in self._clients.items():
            try:
                await client.__aexit__(None, None, None)
                logger.info(f"Disconnected from MCP server: {name}")
            except Exception as e:
                logger.warning(f"Error closing MCP server '{name}': {e}")
        self._clients.clear()

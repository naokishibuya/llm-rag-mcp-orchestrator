"""Quick script to inspect raw MCP responses from finance, weather, and tavily."""

import asyncio
from pprint import pprint

from fastmcp.client.client import Client
from fastmcp.client.transports import StreamableHttpTransport


async def inspect(name: str, url: str, tool: str, args: dict):
    print(f"{'=' * 60}")
    print(f"  {name}: {tool}({args})")
    print(f"{'=' * 60}")

    transport = StreamableHttpTransport(url=url)
    client = Client(transport)

    async with client:
        result = await client.call_tool(tool, args)

        print("\n--- structured_content ---")
        print(f"type: {type(result.structured_content)}")
        pprint(result.structured_content)

        print("\n--- content blocks ---")
        print(f"count: {len(result.content)}")
        for i, block in enumerate(result.content):
            print(f"\nblock[{i}] type: {type(block).__name__}")
            if hasattr(block, "text"):
                print(f"text: {block.text[:300]}")

        print("\n--- data ---")
        print(f"type: {type(getattr(result, 'data', None))}")
        pprint(getattr(result, "data", None))
        print()


async def main():
    await inspect(
        "Finance", "http://127.0.0.1:8030/mcp",
        "get_stock_price", {"symbol": "AAPL2"},
    )
    await inspect(
        "Weather", "http://127.0.0.1:8031/mcp",
        "get_weather", {"city": "Tokyo"},
    )


if __name__ == "__main__":
    asyncio.run(main())

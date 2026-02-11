"""Quick script to inspect Tavily MCP response and test formatting."""

import asyncio
import os
from pprint import pprint

from dotenv import load_dotenv
from fastmcp.client.client import Client
from fastmcp.client.transports import StreamableHttpTransport

load_dotenv()


async def main():
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        print("TAVILY_API_KEY not set")
        return

    url = f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
    transport = StreamableHttpTransport(url=url)
    client = Client(transport)

    async with client:
        result = await client.call_tool("tavily_search", {"query": "What are the latest news on NVIDIA?"})

        print("=== structured_content ===")
        print(f"type: {type(result.structured_content)}")
        pprint(result.structured_content)
        print()

        print("=== content blocks ===")
        for item in result.content:
            pprint(item)
            print()

        # # 1. Raw structured_content
        # print("=== structured_content ===")
        # print(f"type: {type(result.structured_content)}")
        # if result.structured_content:
        #     print(f"keys: {list(result.structured_content.keys())}")
        #     print(f"num results: {len(result.structured_content.get('results', []))}")
        #     print()
        #     print("First result:")
        #     pprint.pprint(result.structured_content["results"][0], width=100)
        # print()

        # # 2. Raw content blocks
        # print("=== content blocks ===")
        # print(f"num blocks: {len(result.content)}")
        # for i, block in enumerate(result.content):
        #     print(f"block[{i}] type: {type(block).__name__}, has text: {hasattr(block, 'text')}")
        # print()

        # # 3. How our handler would format it
        # data = result.structured_content or {}
        # print("=== formatted output ===")
        # for item in data.get("results", []):
        #     title = item.get("title", "")
        #     url = item.get("url", "")
        #     content = item.get("content", "")
        #     print(f"- [{title}]({url}): {content}")


if __name__ == "__main__":
    asyncio.run(main())

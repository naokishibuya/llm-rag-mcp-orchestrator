import inspect
import json
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .mcp.handler import MCPHandler, MCPService
from .rag.client import RAGClient
from .types import UserContext


def _tool(fn):
    """Decorator that attaches a `.tool_schema` dict derived from the function signature."""
    sig = inspect.signature(fn)
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        prop: dict = {"type": "string"}
        if param.annotation is float:
            prop["type"] = "number"
        elif param.annotation is int:
            prop["type"] = "integer"
        elif param.annotation is bool:
            prop["type"] = "boolean"
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    fn.tool_schema = {
        "name": fn.__name__,
        "description": inspect.getdoc(fn) or "",
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }
    return fn


@_tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression.

    Supports standard math functions (sqrt, sin, cos, log, etc.),
    abs, round, min, and max.

    Args:
        expression: A mathematical expression string, e.g. "sqrt(144)" or "0.15 * 230".
    """
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max})
    return float(eval(expression, {"__builtins__": {}}, allowed))


@_tool
def get_current_time(tz: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        tz: IANA timezone name, e.g. "Asia/Tokyo", "America/New_York". Defaults to "UTC".
    """
    try:
        zone = ZoneInfo(tz)
    except KeyError:
        zone = timezone.utc
    now = datetime.now(zone)
    return now.strftime(f"%Y-%m-%d %H:%M:%S %Z ({tz})")


def _make_context_tool(context: UserContext):
    @_tool
    def get_user_context() -> str:
        """Get user's information about:
         - city
         - timezone
         - local_time
        Use this to personalize responses or determine the user's timezone.
        """
        return str(context)

    return get_user_context


def _make_rag_tool(rag_client: RAGClient):
    topic_names = rag_client.topic_names()
    topic_descs = rag_client.topic_descriptions()
    topic_list = "\n".join(f"  - {t}" for t in topic_descs) if topic_descs else "  (general)"

    @_tool
    def search_knowledge_base(query: str, topic: str = "") -> str:
        """Search the knowledge base."""
        results = rag_client.search(query, topic=topic)
        if not results or results[0].score < 0.3:
            return "No relevant documents found in the knowledge base."
        chunks = []
        for r in results:
            if r.score >= 0.3:
                source = f" (Source: {r.document.source})" if r.document.source else ""
                chunks.append(f"{r.document.content[:500]}{source}")
        return "\n\n---\n\n".join(chunks)

    # Override auto-generated schema with topic-aware description and enum
    search_knowledge_base.tool_schema["description"] = (
        f"Search the knowledge base for information relevant to a query.\n\n"
        f"Available topics:\n{topic_list}\n\n"
        f"Use this tool when the user asks about any of these topics."
    )
    if topic_names:
        search_knowledge_base.tool_schema["input_schema"]["properties"]["topic"] = {
            "type": "string",
            "description": f"Topic to search within. One of: {', '.join(topic_names)}",
        }
    return search_knowledge_base


def _make_mcp_tool(handler: MCPHandler, service: MCPService):
    format_hint = service.format_hint

    async def mcp_tool(**params) -> str:
        data = await handler.handle(**params)
        if isinstance(data, dict) and data.get("unavailable"):
            return f"Service '{service.name}' is currently unavailable."
        if isinstance(data, dict) and "error" in data:
            return f"Error: {data['error']}"
        raw = json.dumps(data, indent=2, ensure_ascii=False) if not isinstance(data, str) else data
        if format_hint:
            return f"{raw}\n\nFormat: {format_hint}"
        return raw

    mcp_tool.__name__ = service.name
    mcp_tool.__doc__ = service.description
    mcp_tool.tool_schema = {
        "name": service.name,
        "description": service.description,
        "input_schema": service.input_schema,
    }
    return mcp_tool


def build_tools(*, context: UserContext, rag_client: RAGClient, mcp_handlers: dict[str, MCPHandler]=None):
    """Build the complete tools dict for a single request."""
    tools = {
        "calculate": calculate,
        "get_current_time": get_current_time,
        "get_user_context": _make_context_tool(context),
        "search_knowledge_base": _make_rag_tool(rag_client),
    }
    if mcp_handlers:
        for handler, service in mcp_handlers.values():
            tools[service.name] = _make_mcp_tool(handler, service)
    return tools


def filter_tools(all_tools: dict, tool_names: list[str] | str) -> dict:
    """Filter tools dict to only include the named tools."""
    if tool_names == "all" or (isinstance(tool_names, list) and "all" in tool_names):
        return all_tools
    return {name: fn for name, fn in all_tools.items() if name in tool_names}

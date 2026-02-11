from typing import Annotated, Any, TypedDict


def merge_dict(existing: dict, new: dict | None) -> dict:
    """Reducer that merges new keys into existing dict."""
    if new is None:
        return existing
    return {**existing, **new}


class Message(TypedDict):
    """Chat message."""

    role: str
    content: str


class AgentState(TypedDict, total=False):
    """State for the multi-agent orchestration graph.

    Uses Annotated reducers for automatic state merging:
    - tool_results: Merged via `merge_dict`
    """

    # === Input (set once at invocation) ===
    query: str
    history: list[Message]
    model: Any  # Chat instance (user's choice, for TalkAgent)
    orchestrator_model: Any  # Chat instance (for router + reflector)
    rag_model: Any  # Chat instance (for RAG agent)
    mcp_model: Any  # Chat instance (for MCP agent formatting)
    use_reflection: bool
    max_reflections: int

    # === Multi-intent routing ===
    intents: list[dict]  # [{"intent": "rag", "agent": "RAGAgent", "params": {...}}, ...]
    current_intent_index: int

    # === Execution ===
    agent_response: str  # Current agent's response
    agent_success: bool  # Whether the agent reported success
    intent_results: list[dict]  # Per-intent results with answer, tokens, reflection, etc.
    tool_results: Annotated[dict[str, dict], merge_dict]  # {tool_name: {tool, args, result/error}}

    # === Reflection ===
    reflection_count: int
    reflection_feedback: dict | None  # {"action": "accept|retry|reroute", "feedback": "..."}

    # === Token tracking (router overhead) ===
    router_input_tokens: int
    router_output_tokens: int

    # === Safety ===
    is_blocked: bool
    moderation_reason: str | None

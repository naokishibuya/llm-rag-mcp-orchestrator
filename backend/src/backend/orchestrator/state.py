from operator import add
from typing import Annotated, Any, TypedDict


def append_list(existing: list, new: list | None) -> list:
    """Reducer that appends new items to existing list."""
    if new is None:
        return existing
    return existing + new


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
    - input_tokens/output_tokens: Accumulated via `add`
    - intent_responses: Appended via `append_list`
    - tool_results: Merged via `merge_dict`
    """

    # === Input (set once at invocation) ===
    query: str
    history: list[Message]
    model: Any  # Chat instance
    embedder: Any  # Embeddings instance
    use_reflection: bool
    max_reflections: int

    # === Multi-intent routing ===
    intents: list[dict]  # [{"intent": "rag", "agent": "RAGAgent", "params": {...}}, ...]
    current_intent_index: int

    # === Execution ===
    agent_response: str  # Current agent's response
    agent_success: bool  # Whether the agent reported success
    intent_responses: Annotated[list[str], append_list]  # All responses (accumulated)
    tool_results: Annotated[dict[str, dict], merge_dict]  # {tool_name: {tool, args, result/error}}

    # === Reflection ===
    reflection_count: int
    reflection_feedback: dict | None  # {"action": "accept|retry|reroute", "feedback": "..."}

    # === Token tracking (accumulated) ===
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]

    # === Safety ===
    is_blocked: bool
    moderation_reason: str | None

    # === Output ===
    final_answer: str

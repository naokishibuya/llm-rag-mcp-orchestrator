from typing import Any, TypedDict


class Message(TypedDict):
    """Chat message."""

    role: str
    content: str


class AgentState(TypedDict, total=False):
    """State for the multi-agent orchestration graph."""

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
    intents: list[dict]  # [{"intent": "rag", "params": {...}}, ...]
    current_intent_index: int

    # === Execution ===
    agent_response: str  # Current agent's response
    agent_success: bool  # Whether the agent reported success
    intent_results: list[dict]  # Per-intent results with answer, tokens, reflection, etc.

    # === Reflection ===
    reflection_count: int
    reflection_feedback: dict | None  # {"action": "accept|retry|reroute", "feedback": "..."}

    # === Token tracking ===
    router_input_tokens: int
    router_output_tokens: int
    step_input_tokens: int   # Per-step tokens (overwritten each node, for streaming display)
    step_output_tokens: int

    # === Safety ===
    is_blocked: bool
    moderation_reason: str | None

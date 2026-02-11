"""API routes - LangGraph-based multi-agent orchestration."""

from fastapi import APIRouter
from pathlib import Path
from pydantic import BaseModel

from .config import Config
from .orchestrator import Orchestrator
from .pricer import Pricer


CONFIG_PATH = Path(__file__).parent.parent.parent / "config/config.yaml"

config = Config(CONFIG_PATH)
orchestrator = Orchestrator(config)
pricer = Pricer(config.pricing)

router = APIRouter()


# === Request/Response Models ===

class MessageModel(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[MessageModel]
    model: str | None = None
    embedding_model: str | None = None
    use_reflection: bool = True


# === Endpoints ===

@router.get("/models")
async def get_models():
    return {"models": config.list_chat_models()}


@router.get("/embeddings")
async def get_embeddings():
    return {"embeddings": config.list_embedding_models()}


@router.post("/chat")
async def chat(request: ChatRequest):
    """Multi-turn chat endpoint using LangGraph orchestration."""
    if not request.messages:
        return {"answer": "No messages provided", "error": True}

    # Get models
    models = config.list_chat_models()
    embeddings = config.list_embedding_models()

    model_name = request.model or (models[0] if models else "qwen2.5:7b")
    embedding_model = request.embedding_model or (embeddings[0] if embeddings else "nomic-embed-text")

    # Extract query and history
    query = request.messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    # Invoke orchestrator
    final_state = await orchestrator.invoke(
        query=query,
        history=history,
        model_name=model_name,
        embedding_model=embedding_model,
        use_reflection=request.use_reflection,
    )

    # Calculate cost
    input_tokens = final_state.get("input_tokens", 0)
    output_tokens = final_state.get("output_tokens", 0)
    cost = pricer.calc_cost(model_name, input_tokens, output_tokens)

    # Build metrics
    intents_processed = final_state.get("intents", [])
    metrics = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": round(cost, 6),
        "model": model_name,
        "tools_used": list(final_state.get("tool_results", {}).keys()),
        "intents_processed": [i["intent"] for i in intents_processed],
        "agents_used": [i["agent"] for i in intents_processed],
    }

    # Add reflection info if used
    if request.use_reflection and final_state.get("reflection_feedback"):
        feedback = final_state["reflection_feedback"]
        metrics["reflection"] = {
            "count": final_state.get("reflection_count", 0),
            "action": feedback.get("action", ""),
            "score": feedback.get("score", 0),
            "feedback": feedback.get("feedback", ""),
        }

    return {
        "answer": final_state.get("final_answer", ""),
        "intent": intents_processed[0]["intent"] if intents_processed else "chat",
        "moderation": {
            "verdict": final_state.get("moderation_verdict", "allow"),
            "reason": final_state.get("moderation_reason"),
        },
        "model": model_name,
        "embedding_model": embedding_model,
        "metrics": metrics,
    }

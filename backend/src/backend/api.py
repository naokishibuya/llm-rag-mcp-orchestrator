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
    use_reflection: bool = True


# === Endpoints ===

@router.get("/models")
async def get_models():
    return {"models": config.list_talk_models()}


@router.post("/chat")
async def chat(request: ChatRequest):
    """Multi-turn chat endpoint using LangGraph orchestration."""
    if not request.messages:
        return {"answer": "No messages provided", "error": True}

    # Get models
    models = config.list_talk_models()

    model_name = request.model or (models[0] if models else "qwen2.5:7b")

    # Extract query and history
    query = request.messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    # Invoke orchestrator
    final_state = await orchestrator.invoke(
        query=query,
        history=history,
        model_name=model_name,
        use_reflection=request.use_reflection,
    )

    # Build per-intent results
    intent_results = final_state.get("intent_results", [])
    results = []
    for ir in intent_results:
        ir_model = ir.get("model") or model_name
        ir_in = ir.get("input_tokens", 0)
        ir_out = ir.get("output_tokens", 0)
        cost = pricer.calc_cost(ir_model, ir_in, ir_out)

        reflection = ir.get("reflection")
        if reflection:
            ref_model = final_state.get("orchestrator_model")
            ref_model_name = ref_model.model if ref_model else model_name
            cost += pricer.calc_cost(
                ref_model_name,
                reflection.get("input_tokens", 0),
                reflection.get("output_tokens", 0),
            )

        results.append({
            "answer": ir.get("answer", ""),
            "intent": ir.get("intent", "chat"),
            "agent": ir.get("agent", ""),
            "model": ir_model,
            "metrics": {
                "input_tokens": ir_in,
                "output_tokens": ir_out,
                "cost": round(cost, 6),
                "tools_used": ir.get("tools_used", []),
                "reflection": reflection,
            },
        })

    # Router cost
    router_in = final_state.get("router_input_tokens", 0)
    router_out = final_state.get("router_output_tokens", 0)
    orch_model = final_state.get("orchestrator_model")
    router_model_name = orch_model.model if orch_model else model_name
    router_cost = pricer.calc_cost(router_model_name, router_in, router_out)

    is_blocked = final_state.get("is_blocked", False)
    verdict = "block" if is_blocked else "allow"
    if not is_blocked and final_state.get("moderation_reason"):
        verdict = "warn"

    return {
        "results": results,
        "moderation": {
            "verdict": verdict,
            "reason": final_state.get("moderation_reason"),
        },
        "router": {
            "input_tokens": router_in,
            "output_tokens": router_out,
            "cost": round(router_cost, 6),
        },
    }

"""API routes - LangGraph-based multi-agent orchestration."""

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pathlib import Path
from pydantic import BaseModel

from .config import Config
from .orchestrator import Orchestrator
from .pricer import Pricer

logger = logging.getLogger(__name__)


CONFIG_PATH = Path(__file__).parent.parent.parent / "config/config.yaml"

config = Config(CONFIG_PATH)
orchestrator = Orchestrator(config)
pricer = Pricer(config.pricing)

router = APIRouter()


# === Request/Response Models ===

class MessageModel(BaseModel):
    role: str
    content: str


class UserContext(BaseModel):
    city: str | None = None
    timezone: str | None = None
    local_time: str | None = None


class ChatRequest(BaseModel):
    messages: list[MessageModel]
    model: str | None = None
    use_reflection: bool = True
    user_context: UserContext | None = None


def _build_context_message(uctx: UserContext | None) -> dict | None:
    """Build a system message from user context, or None if empty."""
    if not uctx:
        return None
    parts = []
    if uctx.city:
        parts.append(uctx.city)
    if uctx.local_time:
        parts.append(uctx.local_time)
    if uctx.timezone:
        parts.append(f"({uctx.timezone})")
    if not parts:
        return None
    return {"role": "system", "content": "User context: " + " ".join(parts)}


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

    ctx_msg = _build_context_message(request.user_context)
    if ctx_msg:
        history = [ctx_msg] + history

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


def _sse(data: dict | str) -> str:
    """Format a single SSE event."""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _build_result(ir: dict, model_name: str, orch_model=None) -> dict:
    """Build a single intent result dict (shared by /chat and /chat/stream)."""
    ir_model = ir.get("model") or model_name
    ir_in = ir.get("input_tokens", 0)
    ir_out = ir.get("output_tokens", 0)
    cost = pricer.calc_cost(ir_model, ir_in, ir_out)

    reflection = ir.get("reflection")
    if reflection:
        ref_model_name = orch_model.model if orch_model else model_name
        cost += pricer.calc_cost(
            ref_model_name,
            reflection.get("input_tokens", 0),
            reflection.get("output_tokens", 0),
        )

    return {
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
    }


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming endpoint — emits thinking steps as they happen."""
    if not request.messages:
        async def empty():
            yield _sse({"type": "error", "message": "No messages provided"})
            yield _sse("[DONE]")
        return StreamingResponse(empty(), media_type="text/event-stream")

    models = config.list_talk_models()
    model_name = request.model or (models[0] if models else "qwen2.5:7b")

    query = request.messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    ctx_msg = _build_context_message(request.user_context)
    if ctx_msg:
        history = [ctx_msg] + history

    async def event_generator():
        # Accumulated state for cost calculation at the end
        acc = {
            "intent_results": [],
            "router_input_tokens": 0,
            "router_output_tokens": 0,
            "orchestrator_model": None,
            "is_blocked": False,
            "moderation_reason": None,
        }

        try:
            async for node_name, updates in orchestrator.stream(
                query=query,
                history=history,
                model_name=model_name,
                use_reflection=request.use_reflection,
            ):
                # Accumulate state
                if "intent_results" in updates:
                    acc["intent_results"] = updates["intent_results"]
                if "router_input_tokens" in updates:
                    acc["router_input_tokens"] += updates["router_input_tokens"]
                if "router_output_tokens" in updates:
                    acc["router_output_tokens"] += updates["router_output_tokens"]
                if "orchestrator_model" in updates:
                    acc["orchestrator_model"] = updates["orchestrator_model"]
                if "is_blocked" in updates:
                    acc["is_blocked"] = updates["is_blocked"]
                if "moderation_reason" in updates:
                    acc["moderation_reason"] = updates["moderation_reason"]

                # Map node → SSE event
                if node_name == "safety":
                    is_blocked = updates.get("is_blocked", False)
                    verdict = "block" if is_blocked else "allow"
                    if not is_blocked and updates.get("moderation_reason"):
                        verdict = "warn"
                    yield _sse({"type": "thinking", "step": f"Safety check: {verdict}"})

                elif node_name == "router":
                    intents = updates.get("intents", [])
                    parts = [i["intent"] for i in intents]
                    router_in = updates.get("router_input_tokens", 0)
                    router_out = updates.get("router_output_tokens", 0)
                    yield _sse({
                        "type": "thinking",
                        "step": f"Routing: {', '.join(parts)}",
                        "tokens": {"in": router_in, "out": router_out},
                    })

                elif node_name == "agent":
                    intent_results = updates.get("intent_results", acc["intent_results"])
                    if intent_results:
                        latest = intent_results[-1]
                        intent = latest.get("intent", "?")
                        for tool_name in latest.get("tools_used", []):
                            yield _sse({"type": "thinking", "step": f"Tool: {tool_name}"})
                        step_in = updates.get("step_input_tokens", 0)
                        step_out = updates.get("step_output_tokens", 0)
                        yield _sse({
                            "type": "thinking",
                            "step": f"Agent: {intent}",
                            "detail": latest.get("answer", ""),
                            "tokens": {"in": step_in, "out": step_out},
                        })

                elif node_name == "reflector":
                    # Read reflection data from intent_results (always present,
                    # even when max_reflections clears reflection_feedback).
                    ir_list = updates.get("intent_results", acc["intent_results"])
                    ref = ir_list[-1].get("reflection", {}) if ir_list else {}
                    if ref:
                        action = ref.get("action", "?")
                        score = ref.get("score")
                        score_str = f", score: {score:.2f}" if score is not None else ""
                        detail = ref.get("feedback", "")
                        step_in = updates.get("step_input_tokens", 0)
                        step_out = updates.get("step_output_tokens", 0)
                        yield _sse({
                            "type": "thinking",
                            "step": f"Reflection: {action}{score_str}",
                            "detail": detail,
                            "tokens": {"in": step_in, "out": step_out},
                        })

                elif node_name in ("finalize", "blocked"):
                    # Emit all results once the graph is done.
                    # We intentionally defer to here (not check_next)
                    # because reroute replaces intent_results entries
                    # in-place — emitting earlier would show stale answers.
                    final_results = acc["intent_results"]
                    if node_name == "blocked":
                        final_results = updates.get("intent_results", final_results)

                    # When multiple intents produced results, drop low-quality
                    # ones (reflection score < 0.7) if a good result exists.
                    if len(final_results) > 1:
                        good = [
                            r for r in final_results
                            if (r.get("reflection") or {}).get("score", 1.0) >= 0.7
                        ]
                        if good:
                            dropped = len(final_results) - len(good)
                            if dropped:
                                logger.info(f"SSE finalize: dropped {dropped} low-score results")
                            final_results = good

                    logger.info(
                        f"SSE finalize: {len(final_results)} results, "
                        f"intents={[r.get('intent') for r in final_results]}, "
                        f"agents={[r.get('agent') for r in final_results]}"
                    )
                    # Emit answer events for filtered results
                    for ir in final_results:
                        result = _build_result(ir, model_name, acc["orchestrator_model"])
                        yield _sse({"type": "answer", "result": result})

                    # Compute totals from ALL results (including dropped ones)
                    # since those tokens were actually consumed.
                    all_built = [
                        _build_result(ir, model_name, acc["orchestrator_model"])
                        for ir in acc["intent_results"]
                    ]

                    is_blocked = acc["is_blocked"]
                    verdict = "block" if is_blocked else "allow"
                    if not is_blocked and acc["moderation_reason"]:
                        verdict = "warn"

                    router_in = acc["router_input_tokens"]
                    router_out = acc["router_output_tokens"]
                    orch_model = acc["orchestrator_model"]
                    router_model_name = orch_model.model if orch_model else model_name
                    router_cost = pricer.calc_cost(router_model_name, router_in, router_out)

                    total_in = router_in
                    total_out = router_out
                    total_cost = router_cost
                    for r in all_built:
                        total_in += r["metrics"]["input_tokens"]
                        total_out += r["metrics"]["output_tokens"]
                        total_cost += r["metrics"]["cost"]
                        ref = r["metrics"].get("reflection")
                        if ref:
                            total_in += ref.get("input_tokens", 0)
                            total_out += ref.get("output_tokens", 0)

                    yield _sse({
                        "type": "done",
                        "moderation": {
                            "verdict": verdict,
                            "reason": acc["moderation_reason"],
                        },
                        "router": {
                            "input_tokens": router_in,
                            "output_tokens": router_out,
                            "cost": round(router_cost, 6),
                        },
                        "total": {
                            "input_tokens": total_in,
                            "output_tokens": total_out,
                            "cost": round(total_cost, 6),
                        },
                    })

        except Exception as e:
            logger.exception("Error during streaming")
            yield _sse({"type": "error", "message": str(e)})

        yield _sse("[DONE]")

    return StreamingResponse(event_generator(), media_type="text/event-stream")

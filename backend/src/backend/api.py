"""API routes - LangGraph-based multi-agent orchestration."""

import json
import logging
from dataclasses import asdict

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

router = APIRouter()


# === Request/Response Models ===

class MessageModel(BaseModel):
    role: str
    content: str


class UserContext(BaseModel):
    city: str | None = None
    timezone: str | None = None
    local_time: str | None = None

    @property
    def summary(self) -> str | None:
        parts = []
        if self.city:
            parts.append(self.city)
        if self.local_time:
            parts.append(self.local_time)
        if self.timezone:
            parts.append(f"({self.timezone})")
        return " ".join(parts) if parts else None


class ChatRequest(BaseModel):
    messages: list[MessageModel]
    model: str | None = None
    use_reflection: bool = True
    user_context: UserContext | None = None


# === Endpoints ===

@router.get("/models")
async def get_models():
    return {"models": config.list_talk_models()}


@router.post("/chat")
async def chat(request: ChatRequest):
    """SSE streaming endpoint — emits thinking steps as they happen."""
    if not request.messages:
        async def empty():
            yield _event("error", message="No messages provided")
        return StreamingResponse(_sse_wrap(empty()), media_type="text/event-stream")

    model_name, query, history = _parse_request(request)

    async def event_generator():
        pricer = Pricer(config.pricing)

        try:
            async for node_name, updates in orchestrator.stream(
                query=query,
                history=history,
                model_name=model_name,
                use_reflection=request.use_reflection,
            ):
                tokens = pricer.add(updates.get("token_log", []))

                if node_name == "moderation":
                    yield _event("thinking", step=f"Moderation: {updates['moderation'].verdict}")

                elif node_name == "router":
                    intents = updates["routing"]
                    yield _event("thinking", step=f"Routing: {', '.join(r.intent for r in intents)}", tokens=tokens)

                elif node_name == "agent":
                    latest = updates["agent_results"][-1]
                    for tool_name in latest.tools_used:
                        yield _event("thinking", step=f"Tool: {tool_name}")
                    yield _event("thinking", step=f"Agent: {latest.intent}", detail=latest.answer, tokens=tokens)

                elif node_name == "reflector":
                    ref = updates["reflection"]
                    score_str = f", score: {ref.score:.2f}" if ref.score is not None else ""
                    yield _event("thinking", step=f"Reflection: {ref.action}{score_str}", detail=ref.feedback, tokens=tokens)

                elif node_name == "finalize":
                    for ir in updates["agent_results"]:
                        yield _event("answer", result=asdict(ir))
                    yield _event("done", moderation=asdict(updates["moderation"]), **pricer.summary())

        except Exception as e:
            logger.exception("Error during streaming")
            yield _event("error", message=str(e))

    return StreamingResponse(_sse_wrap(event_generator()), media_type="text/event-stream")


def _parse_request(request: ChatRequest) -> tuple[str, str, list[dict]]:
    model_name = request.model or config.default_talk_model()
    query = request.messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]
    ctx = request.user_context
    if ctx and ctx.summary:
        history = [{"role": "system", "content": f"User context: {ctx.summary}"}] + history
    return model_name, query, history


def _event(type: str, **kwargs) -> dict:
    return {"type": type, **kwargs}


async def _sse_wrap(events):
    """Wrap an async generator of dicts into SSE (Server-Sent Events) wire format."""
    async for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

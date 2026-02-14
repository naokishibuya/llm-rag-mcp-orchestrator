import json
import logging
from dataclasses import asdict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pathlib import Path
from pydantic import BaseModel

from .config import Config
from .core import Message, UserContext
from .orchestrator import Orchestrator
from .llm import Pricer


logger = logging.getLogger(__name__)


CONFIG_PATH = Path(__file__).parent.parent.parent / "config/config.yaml"

config = Config(CONFIG_PATH)
orchestrator = Orchestrator(config)

router = APIRouter()


# === Request/Response Models ===

class MessageModel(BaseModel):
    role: str
    content: str


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

    model_name, query, history, context = _parse_request(request)

    async def event_generator():
        pricer = Pricer(config.pricing)

        try:
            async for node_name, updates in orchestrator.stream(
                query=query,
                history=history,
                model_name=model_name,
                use_reflection=request.use_reflection,
                context=context,
            ):
                token_log = updates.get("token_log", [])
                tokens = pricer.add(*token_log[0]) if token_log else None

                if node_name == "moderation":
                    yield _event("thinking", step=f"Moderation: {updates['moderation'].verdict}")

                elif node_name == "router":
                    routes = updates["routes"]
                    yield _event("thinking", step=f"Routing: {', '.join(r.intent for r in routes)}", tokens=tokens)

                elif node_name == "agent":
                    route = updates["routes"][updates.get("cursor", 0)]
                    for tool_name in route.reply.tools_used:
                        yield _event("thinking", step=f"Tool: {tool_name}")
                    yield _event("thinking", step=f"Agent: {route.intent}", detail=route.reply.text, tokens=tokens)

                elif node_name == "reflector":
                    ref = updates["reflection"]
                    score_str = f", score: {ref.score:.2f}" if ref.score is not None else ""
                    yield _event("thinking", step=f"Reflection: {ref.action}{score_str}", detail=ref.feedback, tokens=tokens)

                elif node_name == "finalize":
                    for route in updates["routes"]:
                        yield _event("answer", result={"intent": route.intent, **asdict(route.reply)})
                    yield _event("done", moderation=asdict(updates["moderation"]), **pricer.summary())

        except Exception as e:
            logger.exception("Error during streaming")
            yield _event("error", message=str(e))

    return StreamingResponse(_sse_wrap(event_generator()), media_type="text/event-stream")


def _parse_request(request: ChatRequest) -> tuple[str, str, list[Message], UserContext]:
    model_name = request.model or config.default_talk_model()
    query = request.messages[-1].content
    history = [Message(role=m.role, content=m.content) for m in request.messages[:-1]]
    return model_name, query, history, request.user_context or UserContext()


def _event(type: str, **kwargs) -> dict:
    return {"type": type, **kwargs}


async def _sse_wrap(events):
    """Wrap an async generator of dicts into SSE (Server-Sent Events) wire format."""
    async for event in events:
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

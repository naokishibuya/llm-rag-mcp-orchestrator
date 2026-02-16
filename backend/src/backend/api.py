import json
import logging
from dataclasses import asdict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pathlib import Path
from pydantic import BaseModel

from .agent import Pricer, UserContext
from .agent.types import Message
from .config import Config
from .orchestrator import Orchestrator


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
    user_context: UserContext | None = None


# === Endpoints ===

@router.get("/models")
async def get_models():
    return {"models": config.list_models()}


@router.post("/chat")
async def chat(request: ChatRequest):
    """SSE streaming endpoint."""
    if not request.messages:
        async def empty():
            yield _event("error", message="No messages provided")
        return StreamingResponse(_sse_wrap(empty()), media_type="text/event-stream")

    model_name, query, history, context = _parse_request(request)

    async def event_generator():
        pricer = Pricer(config.pricing)

        try:
            async for event_name, data in orchestrator.stream(
                query=query,
                history=history,
                model_name=model_name,
                context=context,
            ):
                if event_name == "moderation":
                    yield _event("thinking", step=f"Moderation: {data['moderation'].verdict}")

                elif event_name == "agent":
                    reply = data["reply"]
                    for tool_name in reply.tools_used:
                        yield _event("thinking", step=f"Tool: {tool_name}")
                    tokens = pricer.add(reply.model, reply.tokens)
                    yield _event("answer", result={"intent": "chat", **asdict(reply)})

                elif event_name == "done":
                    yield _event("done", moderation=asdict(data["moderation"]), **pricer.summary())

        except Exception as e:
            logger.exception("Error during streaming")
            yield _event("error", message=str(e))

    return StreamingResponse(_sse_wrap(event_generator()), media_type="text/event-stream")


def _parse_request(request: ChatRequest) -> tuple[str, str, list[Message], UserContext]:
    model_name = request.model or config.default_model()
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

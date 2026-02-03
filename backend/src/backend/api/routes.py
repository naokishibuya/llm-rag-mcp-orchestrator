from fastapi import APIRouter
from .models import AskRequest, ChatRequest
from ..agent import AskAgent, ChatAgent, Message
from ..metrics import get_tracker


router = APIRouter()

_ask_agent: AskAgent | None = None
_chat_agent: ChatAgent | None = None


def get_ask_agent() -> AskAgent:
    global _ask_agent
    if _ask_agent is None:
        _ask_agent = AskAgent()
    return _ask_agent


def get_chat_agent() -> ChatAgent:
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = ChatAgent()
    return _chat_agent


@router.post("/ask")
async def ask(request: AskRequest):
    agent = get_ask_agent()
    response = await agent.run(request.question)
    return response.to_dict()


@router.post("/chat")
async def chat(request: ChatRequest):
    agent = get_chat_agent()
    messages = [Message(role=m.role, content=m.content) for m in request.messages]
    response = await agent.run(messages)
    return response.to_dict()


@router.get("/metrics")
async def metrics():
    tracker = get_tracker()
    return tracker.get_summary()


@router.get("/metrics/requests")
async def metrics_requests(limit: int = 100):
    tracker = get_tracker()
    requests = tracker.get_requests(limit)
    return [
        {
            "request_id": r.request_id,
            "timestamp": r.timestamp.isoformat(),
            "model": r.model,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost": r.cost,
            "latency_ms": r.latency_ms,
            "operation": r.operation,
        }
        for r in requests
    ]

from fastapi import APIRouter
from .models import AskRequest, ChatRequest
from ..agent import AskAgent, ChatAgent, Message
from ..metrics import get_tracker
from ..llm import list_models, list_embeddings


router = APIRouter()


@router.get("/models")
async def get_models():
    return {"models": list_models()}


@router.get("/embeddings")
async def get_embeddings():
    return {"embeddings": list_embeddings()}


@router.post("/ask")
async def ask(request: AskRequest):
    agent = AskAgent(model=request.model, embedding_model=request.embedding_model)
    response = await agent.run(request.question)
    return response.to_dict()


@router.post("/chat")
async def chat(request: ChatRequest):
    agent = ChatAgent(model=request.model, embedding_model=request.embedding_model)
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

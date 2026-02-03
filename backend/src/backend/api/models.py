from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    model: str | None = None
    embedding_model: str | None = None


class MessageModel(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[MessageModel]
    model: str | None = None
    embedding_model: str | None = None

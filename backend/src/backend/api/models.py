from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class MessageModel(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[MessageModel]

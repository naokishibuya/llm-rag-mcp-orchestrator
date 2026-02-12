import operator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, TypedDict


class Message(TypedDict):
    role: str
    content: str


class Verdict(StrEnum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class Moderation:
    verdict: Verdict
    reason: str | None = None

    @property
    def is_blocked(self) -> bool:
        return self.verdict == Verdict.BLOCK


class Intent(StrEnum):
    NONE = ""
    CHAT = "chat"
    SMALLTALK = "smalltalk"
    RAG = "rag"
    BLOCKED = "blocked"


@dataclass
class AgentRequest:
    intent: str = Intent.CHAT  # Intent member or custom string for MCP tools
    params: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    intent: str = Intent.NONE  # Intent member or custom string for MCP tools
    model: str = ""
    answer: str = ""
    success: bool = True
    tools_used: list[str] = field(default_factory=list)


class Action(StrEnum):
    NONE = ""
    ACCEPT = "accept"
    RETRY = "retry"


@dataclass
class Reflection:
    count: int = 0
    action: str = Action.NONE
    score: float | None = None
    feedback: str = ""
    retry_query: str | None = None
    retry_history: list[dict] | None = None


@dataclass
class TokenUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class State(TypedDict, total=False):
    query: str
    history: list[Message]
    model: Any  # Chat instance
    use_reflection: bool
    moderation: Moderation | None
    routing: list[AgentRequest]
    agent_results: list[AgentResult]
    reflection: Reflection
    token_log: Annotated[list[TokenUsage], operator.add]

import logging
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ..llm import Chat, Message, Role, TokenUsage


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


logger = logging.getLogger(__name__)


@dataclass
class RoutingInfo:
    description: str
    params_schema: dict[str, str]


SYSTEM_PROMPT = """You classify user intent and route to specialized agents.

Available agents:
{descriptions}

Guidelines:
1. Be EXTREMELY STRICT. Only route to an agent if the user's query explicitly and clearly requires it.
2. 'smalltalk' and 'chat' are ALWAYS MUTUALLY EXCLUSIVE. A single question or statement MUST map to exactly one of them, NEVER both.
3. 'smalltalk' is ONLY for greetings and social pleasantries with NO informational content (e.g., "Hi", "Hello", "How are you?"). Any question that asks for real information — even if brief or conversational (e.g., "where am I?", "what time is it?") — is NOT smalltalk.
4. For ANY questions requiring information, calculation, general knowledge, explanations, reasoning, or creative writing, use the 'chat' intent. This includes brief math expressions like "14*37/53".
5. Do NOT add any extra intents, tools, or parameters that are not directly requested.
6. Only return multiple intents if the user has provided MULTIPLE distinct and explicit questions or commands (e.g., "Hi, tell me about Mars").
7. Each intent needs: "intent" (agent type) and "params" (extracted parameters).

Parameters by intent:
{params_lines}
"""


class Router:
    def __init__(self, routing: dict[str, RoutingInfo], *, model: Chat):
        self._model = model
        self._intent_names: list[str] = []
        self._descriptions: list[str] = []
        self._params_lines: list[str] = []
        self._update(routing)

    def add_routes(self, routing: dict[str, RoutingInfo]):
        self._update(routing)

    def _update(self, routing: dict[str, RoutingInfo]):
        for intent_name, meta in routing.items():
            self._intent_names.append(intent_name)
            self._descriptions.append(f"- {intent_name}: {meta.description}")
            self._params_lines.append(f"- {intent_name}: {meta.params_schema}")

        intent_names = list(dict.fromkeys(self._intent_names))
        IntentEnum = Enum("IntentEnum", {n: n for n in intent_names}, type=str)

        class Intent(BaseModel):
            intent: IntentEnum  # type: ignore[valid-type]
            params: dict[str, Any] = Field(default_factory=dict)

        class RoutingResult(BaseModel):
            intents: list[Intent]

        self._RoutingResult = RoutingResult
        self._system_prompt = SYSTEM_PROMPT.format(
            descriptions="\n".join(self._descriptions),
            params_lines="\n".join(self._params_lines),
        )
        logger.info("Router prompt:\n%s", self._system_prompt)

    def route(
        self, query: str, history: list[dict]
    ) -> tuple[list[AgentRequest], TokenUsage]:
        messages = [Message(role=Role.SYSTEM, content=self._system_prompt)]

        recent_history = history[-3:] if len(history) > 3 else history
        messages.extend(recent_history)
        messages.append(Message(role=Role.USER, content=query))

        response = self._model.chat(messages, self._RoutingResult)

        try:
            result = self._RoutingResult.model_validate_json(response.text)
            intents = [
                AgentRequest(
                    intent=item.intent.value,
                    params=item.params,
                )
                for item in result.intents
            ]
        except Exception:
            logger.warning(f"Failed to parse routing result, falling back to chat: {response.text}")
            intents = [AgentRequest()]

        logger.info(f"Routed query: {query!r} -> {intents}")

        return intents, response.tokens

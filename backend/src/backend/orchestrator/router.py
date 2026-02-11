import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langgraph.types import Command
from pydantic import BaseModel, Field

from ..llm import Chat

logger = logging.getLogger(__name__)


@dataclass
class RoutingMeta:
    intent: str
    description: str
    params_schema: dict[str, str]


@dataclass
class RouterResult:
    intents: list[dict]
    input_tokens: int = 0
    output_tokens: int = 0


SYSTEM_PROMPT = """You classify user intent and route to specialized agents.

Available agents:
{descriptions}

For queries with MULTIPLE questions, return multiple intents in order.
Each intent needs: "intent" (agent type) and "params" (extracted parameters).

Parameters by intent:
{params_lines}
"""


class Router:
    def __init__(self, routing: dict[str, RoutingMeta]):
        self._intent_to_agent: dict[str, str] = {}
        self._descriptions: list[str] = []
        self._params_lines: list[str] = []
        self._update(routing)

    @property
    def agent_descriptions(self) -> str:
        """Available agents formatted for the reflector prompt."""
        return "\n".join(self._descriptions)

    def add_routes(self, routing: dict[str, RoutingMeta]):
        self._update(routing)

    def _update(self, routing: dict[str, RoutingMeta]):
        for agent_name, meta in routing.items():
            self._intent_to_agent[meta.intent] = agent_name
            self._descriptions.append(f"- {meta.intent}: {meta.description}")
            self._params_lines.append(f"- {meta.intent}: {meta.params_schema}")

        intent_names = list(self._intent_to_agent.keys())
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

    async def __call__(self, state: dict) -> Command:
        query = state["query"]
        exclude_agent = None

        feedback = state.get("reflection_feedback")
        if feedback and feedback.get("action") == "reroute":
            query = feedback["query"]
            exclude_agent = feedback.get("exclude_agent")

        result = await self.execute(
            model=state["model"],
            query=query,
            history=state["history"],
        )

        # If the router picked the same agent that just failed, fall back.
        if exclude_agent:
            for intent in result.intents:
                if intent["agent"] == exclude_agent:
                    fallback = feedback.get("suggested_agent") or "TalkerAgent"
                    logger.info(f"Excluding failed agent {exclude_agent}, falling back to {fallback}")
                    intent["agent"] = fallback

        return Command(
            update={
                "intents": result.intents,
                "current_intent_index": 0,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
            goto="agent",
        )

    async def execute(
        self, model: Chat, query: str, history: list[dict]
    ) -> RouterResult:
        messages = [{"role": "system", "content": self._system_prompt}]

        recent_history = history[-3:] if len(history) > 3 else history
        messages.extend(recent_history)
        messages.append({"role": "user", "content": query})

        response = model.chat(messages, self._RoutingResult)

        try:
            result = self._RoutingResult.model_validate_json(response.text)
            intents = [
                {
                    "intent": item.intent.value,
                    "agent": self._intent_to_agent[item.intent.value],
                    "params": item.params,
                }
                for item in result.intents
            ]
        except Exception:
            logger.warning(f"Failed to parse routing result, falling back to chat: {response.text}")
            intents = [{"intent": "chat", "agent": self._intent_to_agent.get("chat", "TalkerAgent"), "params": {}}]

        logger.info(f"Routed query: {query!r} -> {intents}")

        return RouterResult(
            intents=intents,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

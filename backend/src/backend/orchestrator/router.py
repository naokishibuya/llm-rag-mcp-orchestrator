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
    def __init__(self, routing: dict[str, RoutingMeta]):
        self._intent_names: list[str] = []
        self._descriptions: list[str] = []
        self._params_lines: list[str] = []
        self._update(routing)

    @property
    def agent_descriptions(self) -> str:
        """Available intents formatted for the reflector prompt."""
        return "\n".join(self._descriptions)

    def add_routes(self, routing: dict[str, RoutingMeta]):
        self._update(routing)

    def _update(self, routing: dict[str, RoutingMeta]):
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

    async def __call__(self, state: dict) -> Command:
        query = state["query"]
        exclude_intent = None
        suggested_intent = None

        feedback = state.get("reflection_feedback")
        if feedback and feedback.get("action") == "reroute":
            query = feedback["query"]
            exclude_intent = feedback.get("exclude_intent")
            suggested_intent = feedback.get("suggested_intent")

        result = await self.execute(
            model=state["orchestrator_model"],
            query=query,
            history=state["history"],
        )

        # Handle reroute:
        # If we were rerouting a specific failed intent in a multi-intent sequence,
        # we want to replace JUST that intent in the original sequence, not the whole list.
        current_intents = list(state.get("intents", []))
        if exclude_intent and current_intents:
            idx = state.get("current_intent_index", 0) - 1
            if 0 <= idx < len(current_intents) and current_intents[idx]["intent"] == exclude_intent:
                # Use the first intent from the new routing as the replacement
                new_intent = result.intents[0] if result.intents else None
                if not new_intent or new_intent["intent"] == exclude_intent:
                    fallback = suggested_intent or "chat"
                    new_intent = {
                        "intent": fallback,
                        "params": {"query": query}
                    }

                logger.info(f"Rerouting intent {idx}: {exclude_intent} -> {new_intent['intent']}")
                current_intents[idx] = new_intent
                return Command(
                    update={
                        "intents": current_intents,
                        "current_intent_index": idx, # stay at this index to execute replacement
                        "router_input_tokens": result.input_tokens,
                        "router_output_tokens": result.output_tokens,
                    },
                    goto="agent",
                )

        return Command(
            update={
                "intents": result.intents,
                "current_intent_index": 0,
                "router_input_tokens": result.input_tokens,
                "router_output_tokens": result.output_tokens,
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
                    "params": item.params,
                }
                for item in result.intents
            ]
        except Exception:
            logger.warning(f"Failed to parse routing result, falling back to chat: {response.text}")
            intents = [{"intent": "chat", "params": {}}]

        logger.info(f"Routed query: {query!r} -> {intents}")

        return RouterResult(
            intents=intents,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

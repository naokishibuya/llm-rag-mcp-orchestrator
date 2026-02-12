import json
import logging
import re

from ..llm import Chat
from .state import Action, TokenUsage


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a quality evaluator for an AI assistant's responses.

Analyze the response and determine if it adequately answers the user's query.

Evaluate based on:
1. Relevance: Does the response address the query?
2. Completeness: Is the answer thorough enough?
3. Accuracy: Is the information likely correct?
4. Clarity: Is the response clear and well-structured?

Critical checks:
- If Agent Success is false, the current intent could not handle this query. Recommend "retry" with specific feedback.
- If the response suggests code or manual steps instead of providing actual data, it likely failed. Recommend "retry".
- If the response confidently states facts that weren't retrieved from a tool or knowledge base, it may be hallucinating.
- MATH FORMATTING: If the response uses \(, \), \[, or \] for LaTeX, it is INVALID. Recommend "retry" with feedback: "Use $...$ and $$...$$ for math. Delimiters \( \) and \[ \] are not supported."

Respond with JSON:
{{
    "action": "accept|retry",
    "score": 0.0-1.0,
    "feedback": "Specific, actionable feedback"
}}

Actions:
- "accept": Response is good enough. Use for scores >= 0.7
- "retry": Response needs improvement. Provide specific feedback.

Be concise but specific in feedback. Focus on actionable improvements."""

EVALUATION_PROMPT = """Evaluate this response:

User Query: {query}
Intent: {intent}
Agent Success: {success}

Response:
{response}

Provide your evaluation as JSON."""

VALID_ACTIONS = {Action.ACCEPT, Action.RETRY}


class Reflector:
    def __init__(self, *, model: Chat, max_reflections: int):
        self._model = model
        self._max_reflections = max_reflections

    def reflect(
        self,
        query: str,
        response: str,
        intent: str,
        success: bool = True,
    ) -> tuple[dict, TokenUsage]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EVALUATION_PROMPT.format(
                query=query,
                intent=intent,
                response=response,
                success=success,
            )},
        ]

        llm_response = self._model.chat(messages)
        logger.debug(f"Reflector raw response: {llm_response.text}")
        info = self._parse_reflection(llm_response.text, success)

        logger.info(
            f"Reflection on {intent}: "
            f"action={info['action']}, score={info['score']}, "
            f"feedback={info['feedback']!r}"
        )

        tokens = TokenUsage(
            model=self._model.model,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
        )
        return info, tokens

    @property
    def max_reflections(self) -> int:
        return self._max_reflections

    def _parse_reflection(self, text: str, agent_success: bool = True) -> dict:
        try:
            text = text.strip()
            if "```" in text:
                match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
                if match:
                    text = match.group(1).strip()

            data = json.loads(text)
            action = data.get("action", "").lower()

            if action not in VALID_ACTIONS:
                logger.warning(f"Invalid reflection action '{action}', defaulting to accept")
                action = Action.ACCEPT

            return {
                "action": action,
                "score": float(data["score"]) if "score" in data else None,
                "feedback": data.get("feedback", ""),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse reflection: {e}")
            fallback = Action.RETRY if not agent_success else Action.ACCEPT
            return {
                "action": fallback,
                "score": None,
                "feedback": "Reflection parse failed",
            }

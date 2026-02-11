import json
import logging
import re
from dataclasses import dataclass

from langgraph.types import Command

from ..llm import Chat


logger = logging.getLogger(__name__)


@dataclass
class ReflectionResult:
    action: str
    score: float | None = None
    feedback: str = ""
    suggested_agent: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


SYSTEM_PROMPT = """You are a quality evaluator for an AI assistant's responses. \
Analyze the response and determine if it adequately answers the user's query.

Evaluate based on:
1. Relevance: Does the response address the query?
2. Completeness: Is the answer thorough enough?
3. Accuracy: Is the information likely correct?
4. Clarity: Is the response clear and well-structured?

Critical checks:
- If Agent Success is false, the agent could not handle this query. Recommend "reroute" to a more suitable agent.
- If the response suggests code or manual steps instead of providing actual data, it likely failed. Recommend "reroute".
- If the response confidently states facts that weren't retrieved from a tool or knowledge base, it may be hallucinating.

Respond with JSON:
{{
    "action": "accept|retry|reroute",
    "score": 0.0-1.0,
    "feedback": "Specific, actionable feedback",
    "suggested_agent": "agent_name (only if action=reroute)"
}}

Actions:
- "accept": Response is good enough. Use for scores >= 0.7
- "retry": Response needs improvement from same agent. Provide specific feedback.
- "reroute": Wrong agent was used. Suggest correct agent and explain why.

Available agents for rerouting:
{available_agents}

Be concise but specific in feedback. Focus on actionable improvements."""

EVALUATION_PROMPT = """Evaluate this response:

User Query: {query}
Agent Used: {agent} (intent: {intent})
Agent Success: {success}

Response:
{response}

Provide your evaluation as JSON."""

VALID_ACTIONS = {"accept", "retry", "reroute"}


class Reflector:
    def __init__(self, available_agents: str = ""):
        self._available_agents = available_agents

    async def __call__(self, state: dict) -> Command:
        idx = state["current_intent_index"] - 1
        intent_data = state["intents"][idx] if idx < len(state["intents"]) else {}

        # For multi-intent queries, scope evaluation to the current intent
        # so the stock price agent isn't penalized for not returning weather.
        params = intent_data.get("params", {})
        if len(state.get("intents", [])) > 1:
            params_desc = ", ".join(f"{k}={v}" for k, v in params.items())
            intent_query = f"{intent_data.get('intent', 'chat')}: {params_desc}" if params_desc else state["query"]
        else:
            intent_query = params.get("query") or state["query"]

        result = await self.execute(
            model=state["model"],
            query=intent_query,
            agent_response=state["agent_response"],
            delegated_agent=intent_data.get("agent", "TalkerAgent"),
            intent=intent_data.get("intent", "chat"),
            agent_success=state.get("agent_success", True),
        )

        reflection_feedback = {
            "action": result.action,
            "score": result.score,
            "feedback": result.feedback,
            "suggested_agent": result.suggested_agent,
        }

        updates: dict = {
            "reflection_count": state.get("reflection_count", 0) + 1,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

        if result.action == "retry":
            reflection_feedback["query"] = f"Previous reply had an error: {result.feedback}"
            reflection_feedback["history"] = list(state["history"]) + [
                {"role": "user", "content": state["query"]},
                {"role": "assistant", "content": state["agent_response"]},
            ]
            updates["current_intent_index"] = idx
        elif result.action == "reroute":
            reflection_feedback["query"] = state["query"]
            reflection_feedback["exclude_agent"] = intent_data.get("agent")

        updates["reflection_feedback"] = reflection_feedback

        # Absorb edge routing logic
        reflection_count = state.get("reflection_count", 0) + 1
        if reflection_count >= state.get("max_reflections", 2):
            goto = "check_next"
        elif result.action == "retry":
            goto = "agent"
        elif result.action == "reroute":
            goto = "router"
        else:
            goto = "check_next"

        return Command(update=updates, goto=goto)

    async def execute(
        self,
        model: Chat,
        query: str,
        agent_response: str,
        delegated_agent: str,
        intent: str,
        agent_success: bool = True,
    ) -> ReflectionResult:
        system_prompt = SYSTEM_PROMPT.format(available_agents=self._available_agents)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": EVALUATION_PROMPT.format(
                query=query,
                intent=intent,
                agent=delegated_agent,
                response=agent_response,
                success=agent_success,
            )},
        ]

        response = model.chat(messages)
        logger.debug(f"Reflector raw response: {response.text}")
        reflection = self._parse_reflection(response.text, agent_success)

        logger.info(
            f"Reflection on {delegated_agent}/{intent}: "
            f"action={reflection['action']}, score={reflection['score']}, "
            f"feedback={reflection['feedback']!r}"
        )

        return ReflectionResult(
            action=reflection["action"],
            score=reflection["score"],
            feedback=reflection["feedback"],
            suggested_agent=reflection.get("suggested_agent"),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

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
                action = "accept"

            return {
                "action": action,
                "score": float(data["score"]) if "score" in data else None,
                "feedback": data.get("feedback", ""),
                "suggested_agent": data.get("suggested_agent"),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse reflection: {e}")
            # If the agent already reported failure, reroute instead of accepting.
            fallback = "reroute" if not agent_success else "accept"
            return {
                "action": fallback,
                "score": None,
                "feedback": "Reflection parse failed",
                "suggested_agent": None,
            }

import logging

from .types import Chat, Message, Reply, Role, UserContext


logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    async def act(
        self,
        *,
        model: Chat,
        query: str,
        history: list[Message],
        context: UserContext | None = None,
        tools: dict[str, callable] | None = None,
    ) -> Reply:
        system = self.system_prompt
        if context:
            system += f"\n\n{context}"

        messages = [Message(role=Role.SYSTEM, content=system)]
        messages.extend(history)
        messages.append(Message(role=Role.USER, content=query))

        logger.info("Agent[%s] query=%r", self.name, query)
        reply = await model.ask(messages, tools=tools)
        logger.info("Agent[%s] %s", self.name, reply)
        return reply

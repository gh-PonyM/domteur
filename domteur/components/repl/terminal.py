import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from domteur.components.base import MQTTClient
from domteur.components.llm_processor.constants import TOPIC_COMPONENT_LLM_PROC_ANSWER
from domteur.components.llm_processor.contracts import (
    Conversation,
    HistoryEntry,
    LLMResponse,
)
from domteur.components.repl.constants import TOPIC_TERMINAL_LLM_REQUEST
from domteur.config import Settings


class LLMTerminalChat(MQTTClient):
    def __init__(self, client, settings: Settings, name: str | None = None):
        super().__init__(client, name)
        self.conversation_history: list[HistoryEntry] = []
        self.settings = settings

    async def handle_message(self, msg):
        if msg.topic.matches(TOPIC_COMPONENT_LLM_PROC_ANSWER):
            answer = LLMResponse.model_validate_json(msg.payload.decode("utf-8"))
            self.conversation_history.extend(
                [
                    HistoryEntry(role="user", content=answer.original_message),
                    HistoryEntry(role="assistant", content=answer.content),
                ]
            )
            with patch_stdout():
                print(f"\n🤖 Assistant: {answer.content}")

    async def ask_questions(self):
        session = PromptSession()
        while True:
            with patch_stdout():
                user = await session.prompt_async("💬 You: ")
            await self.publish(
                TOPIC_TERMINAL_LLM_REQUEST.replace("+", self.name),
                Conversation(
                    content=user, conversation_history=self.conversation_history
                ),
            )

    async def listen(self):
        async for msg in self.client.messages:
            await self.handle_message(msg)

    async def start(self):
        await self.subscribe(TOPIC_COMPONENT_LLM_PROC_ANSWER)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.ask_questions())
            tg.create_task(self.listen())

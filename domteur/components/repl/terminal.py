import asyncio

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from domteur.components.base import MQTTClient, on_publish, on_receive
from domteur.components.llm_processor.constants import TOPIC_COMPONENT_LLM_PROC_ANSWER
from domteur.components.llm_processor.contracts import (
    Conversation,
    HistoryEntry,
    LLMResponse,
)
from domteur.components.repl.constants import COMPONENT_NAME, TOPIC_TERMINAL_LLM_REQUEST
from domteur.config import Settings


class LLMTerminalChat(MQTTClient):
    component_name = COMPONENT_NAME

    def __init__(self, client, settings: Settings, name: str | None = None):
        super().__init__(client, name)
        self.conversation_history: list[HistoryEntry] = []
        self.session_id = None
        self.settings = settings

    @on_receive(TOPIC_COMPONENT_LLM_PROC_ANSWER, LLMResponse)
    async def handle_llm_message(self, msg, response: LLMResponse):
        self.conversation_history.extend(
            [
                HistoryEntry(role="user", content=response.original_message),
                HistoryEntry(role="assistant", content=response.content),
            ]
        )
        with patch_stdout():
            print(f"\n🤖 Assistant: {response.content}\n")

    async def ask_questions(self):
        session = PromptSession()
        while True:
            with patch_stdout():
                query = await session.prompt_async("💬 You: ")
            if query:
                await self.send_to_llm(None, query)

    @on_publish(TOPIC_TERMINAL_LLM_REQUEST, Conversation)
    async def send_to_llm(self, msg, query: str):
        logger.info(f"Conversation lengths: {len(self.conversation_history)}")
        c = Conversation(content=query, conversation_history=self.conversation_history)
        if not self.session_id:
            self.session_id = c.session_id
        else:
            c.session_id = self.session_id
        return c

    async def start(self):
        await self.initialize_subscriptions()
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.ask_questions())
            tg.create_task(self.listen())

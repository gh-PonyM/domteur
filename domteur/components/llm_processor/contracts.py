from typing import Literal

from pydantic import BaseModel

from domteur.components.base import MessagePayload


class HistoryEntry(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Conversation(MessagePayload):
    content: str
    conversation_history: list[HistoryEntry]


class LLMResponse(MessagePayload):
    content: str
    model: str
    original_message: str


class LLMStreamResponse(MessagePayload):
    content: str

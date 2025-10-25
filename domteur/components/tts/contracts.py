from typing import Literal

from domteur.components.base import MessagePayload


class TTSControl(MessagePayload):
    action: Literal["STOP", "MUTE", "UNMUTE", "CLEAR_QUEUE"]


class TTSStreamChunk(MessagePayload):
    content: str
    priority: int = 2  # default NORMAL
    message_type: Literal["stream_chunk", "complete"] = "stream_chunk"

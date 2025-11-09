from typing import Literal

from domteur.components.base import MessagePayload


class STTextSegment(MessagePayload):
    """The response for a stt segment"""

    lang: str
    lang_probability: float
    content: str
    segment_start: float
    segment_end: float


class AudioFileTranscribeRequest(MessagePayload):
    """Request to SST to start sending text segments for the transcription of the file"""

    file_path: str


class STTControl(MessagePayload):
    """Control message for STT operations."""

    action: Literal["START", "STOP", "PAUSE", "RESUME"]


class LiveTranscriptionRequest(MessagePayload):
    """Request to start live transcription."""

    device_index: int | None = None
    sample_rate: int = 16000
    channels: int = 2
    output_format: Literal["txt", "json"] = "txt"
    output_file: str | None = None

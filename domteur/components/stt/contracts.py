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

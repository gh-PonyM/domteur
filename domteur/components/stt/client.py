from pathlib import Path
from typing import TYPE_CHECKING

from faster_whisper import BatchedInferencePipeline, WhisperModel
from loguru import logger

from domteur.components.base import MessagePayload, MQTTClient, on_publish, on_receive
from domteur.components.stt.contracts import (
    AudioFileTranscribeRequest,
    STTextSegment,
)

if TYPE_CHECKING:
    from domteur.config import Settings


class WhisperSTT(MQTTClient):
    def __init__(
        self,
        client,
        settings: "Settings",
        name: str | None = None,
    ):
        super().__init__(client, settings, name)

    async def pre_start(self):
        cfg = self.settings.sst
        # Downloads the model, do on pre-start
        self._model = WhisperModel(
            cfg.model_size, device=cfg.device, compute_type=cfg.compute_type
        )

    @on_receive("+", "audio_transcribe", AudioFileTranscribeRequest, event="request")
    async def transcribe_file(self, msg, event: AudioFileTranscribeRequest):
        logger.info(
            f"Audio transcription request {event.file_path}, {event.session_id=}"
        )
        if not Path(event.file_path).is_file():
            logger.warning("File did not exist")
            await self._send_error_response(
                event.session_id,
                f"file_path {event.file_path} not found",
                source_topic=msg.topic,
            )
            return

        batched_model = BatchedInferencePipeline(model=self._model)
        segments, info = batched_model.transcribe(
            event.file_path, batch_size=16, beam_size=5
        )
        for segment in segments:
            await self.publish_segment(msg, segment, info, event)

    @on_publish("audio_transcribe", STTextSegment, event="batch_output")
    async def publish_segment(
        self, msg, segment, info, original_msg: MessagePayload
    ) -> STTextSegment:
        seg = STTextSegment(
            content=segment.text,
            session_id=original_msg.session_id,
            lang=info.language,
            lang_probability=info.language_probability,
            segment_start=segment.start,
            segment_end=segment.end,
        )
        return seg

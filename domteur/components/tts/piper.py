"""Text-to-speech component using Piper TTS with direct audio output."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import piper
import sounddevice as sd
from loguru import logger
from piper import PiperVoice
from pydantic import BaseModel, Field

from domteur.components.base import MQTTClient, on_receive
from domteur.components.llm_processor.constants import TOPIC_COMPONENT_LLM_PROC_ANSWER
from domteur.components.llm_processor.contracts import LLMResponse
from domteur.components.tts.constants import TOPIC_PIPER_TTS_CONTROL
from domteur.components.tts.contracts import TTSControl

if TYPE_CHECKING:
    from domteur.config import Settings


class PiperTTSConfig(BaseModel):
    """Configuration for text-to-speech engine."""

    type: Literal["piper"] = "piper"
    # TODO: Make a list of the available voices
    voice_model_name: str = "de_DE-kerstin-low"
    volume: float = 1.0
    speed: float = 1.0
    use_cuda: bool = False
    auto_download_voice: bool = True
    sample_rate: int = 22050
    chunk_size: int = 1024
    model_storage_path: Path = Field(default_factory=lambda: Path("/tmp"))

    @property
    def voice_model_path(self) -> Path:
        return self.model_storage_path / f"{self.voice_model_name}.onnx"

    @property
    def voice_config_path(self) -> Path:
        return self.model_storage_path / f"{self.voice_model_name}.json"


def download_voice(voice_name: str, dl_dir: Path) -> None:
    """Download the voice model if it doesn't exist."""
    logger.info(f"Downloading voice model: {voice_name}")
    import subprocess
    import sys

    # Use piper's download utility
    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper.download_voices",
            "--download-dir",
            str(dl_dir),
            voice_name,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    logger.info("Voice model downloaded successfully")


def load_voice(
    voice_name: str,
    model_path: Path,
    auto_download_voice: bool = True,
    use_cuda: bool = False,
) -> PiperVoice:
    """Load the Piper TTS voice model."""
    # Auto-download voice if configured and not exists
    if auto_download_voice and not model_path.exists():
        # TODO: Download path not specified
        download_voice(voice_name, model_path.parent)

    if not model_path.exists():
        raise FileNotFoundError(f"Voice model not found: {model_path}")
    return PiperVoice.load(model_path=model_path, use_cuda=use_cuda)


def stream_play(voice: piper.PiperVoice, text: str):
    """Streaming piper voice output directly to a sounddevice"""
    stream = sd.OutputStream(
        samplerate=voice.config.sample_rate, channels=1, dtype="int16"
    )
    stream.start()
    try:
        for chunk in voice.synthesize(text):
            audio_array = chunk.audio_int16_array
            stream.write(audio_array)
    except Exception as err:
        logger.error(f"Exception during audio streaming: {err}")
    finally:
        stream.stop()
        stream.close()


class AudioPlayer:
    def __init__(self):
        self._task: asyncio.Task | None = None
        self._stream: sd.OutputStream | None = None

    async def play(self, voice: piper.PiperVoice, text: str) -> None:
        if self._task and not self._task.done():
            await self.cancel()

        self._task = asyncio.create_task(self._play_async(voice, text))
        try:
            await self._task
        except asyncio.CancelledError:
            logger.info("Audio playback cancelled")
            raise

    async def _play_async(self, voice: piper.PiperVoice, text: str) -> None:
        loop = asyncio.get_event_loop()
        self._stream = sd.OutputStream(
            samplerate=voice.config.sample_rate, channels=1, dtype="int16"
        )
        self._stream.start()

        try:
            for chunk in voice.synthesize(text):
                if self._task and self._task.cancelled():
                    break
                audio_array = chunk.audio_int16_array
                await loop.run_in_executor(None, self._stream.write, audio_array)
        except asyncio.CancelledError:
            logger.info("Audio synthesis cancelled")
            raise
        except Exception as err:
            logger.error(f"Exception during audio streaming: {err}")
        finally:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    async def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class PiperTTS(MQTTClient):
    """Text-to-speech component that converts text to speech using Piper TTS.
    This is meant to run in a separate thread if async is used"""

    def __init__(self, client, settings: "Settings", name: str | None = None):
        super().__init__(client, name)
        self.config = settings.tts
        self.voice: PiperVoice | None = None

    @staticmethod
    def _prepare_text(text):
        return text

    @on_receive(TOPIC_PIPER_TTS_CONTROL, TTSControl)
    async def handle_control_event(self, msg, event: TTSControl):
        logger.info(f"voice control event received: {event.action}")

    @on_receive(TOPIC_COMPONENT_LLM_PROC_ANSWER, LLMResponse)
    async def handle_tts_request(self, event: LLMResponse) -> None:
        """Handle LLM responses."""
        text = self._prepare_text(event.content)
        logger.info(f"Processing TTS request: {text[:50]}...")

        try:
            self._synthesize_and_play(text)
            logger.info("TTS synthesis and playback completed")
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            # Send a mqtt message in the background by raising exception
            raise

    def _synthesize_and_play(self, text: str) -> None:
        """Synthesize text to speech and play it directly."""
        if not self.voice:
            # maybe try except here?
            self.voice = load_voice(
                voice_name=self.config.voice_model_name,
                model_path=self.config.voice_model_path,
                auto_download_voice=self.config.auto_download_voice,
                use_cuda=self.config.use_cuda,
            )

        # TODO

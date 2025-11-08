"""Text-to-speech component using Piper TTS with direct audio output."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import piper
import sounddevice as sd
from loguru import logger
from piper import PiperVoice
from pydantic import BaseModel, Field

from domteur.components.base import MQTTClient, on_receive
from domteur.components.llm_processor.contracts import LLMResponse
from domteur.components.tts.contracts import TTSControl, TTSStreamChunk

if TYPE_CHECKING:
    from domteur.config import Settings


class PiperTTSConfig(BaseModel):
    """Configuration for text-to-speech engine."""

    type: Literal["piper"] = "piper"
    # TODO: Make a list of the available voices
    voice_model_name: str = "en_US-norman-medium"
    volume: float = 1.0
    speed: float = 1.0
    use_cuda: bool = False
    auto_download_voice: bool = True
    sample_rate: int = 22050
    chunk_size: int = 1024
    model_storage_path: Path = Field(default_factory=lambda: Path("/tmp/piper_tts"))

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


# This is working!
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


class AudioState(str, Enum):
    IDLE = "IDLE"
    PLAYING = "PLAYING"
    MUTED = "MUTED"
    PAUSED = "PAUSED"


# Audio chunks are typically numpy int16 arrays from Piper; accept any buffer-like
# TODO: fix type hint and allow only one
AudioChunk = Any

# Priority levels (lower is higher priority)
PRIORITY_CRITICAL = 0
PRIORITY_HIGH = 1
PRIORITY_NORMAL = 2
PRIORITY_LOW = 3


@contextmanager
def audio_stream(voice):
    stream = sd.OutputStream(
        samplerate=voice.config.sample_rate, channels=1, dtype="int16"
    )
    stream.start()
    try:
        yield stream
    except Exception as err:
        logger.error(f"Exception during audio streaming: {err}")
    finally:
        stream.stop()
        stream.close()


@dataclass
class AudioPlaybackManager:
    voice: piper.PiperVoice | None = None
    muted: bool = False
    running: bool = False

    def __post_init__(self):
        self._task = None
        self._queue = asyncio.Queue()

    async def clear_queue(self):
        """Remove all remaining messages from the queue."""
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        logger.info(f"Cleared {cleared} queued messages.")

    async def set_muted(self, on: bool):
        logger.info(f"Audioplayer: muted {on}")
        self.muted = on

    def _synthesize_and_stream_to_sd(self, text):
        with audio_stream(self.voice) as stream:
            for chunk in self.voice.synthesize(text):
                if not self.running:
                    break
                stream.write(chunk.audio_int16_array)

    async def start(self):
        """Start the player (spawns the play loop)."""
        if not self.running:
            self.running = True
            print("Audio player started.")
            self._task = asyncio.create_task(self._play_loop())

    async def pause(self):
        """Stop the player gracefully."""
        logger.info("Audio player pause (leave the queue)...")
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def stop(self):
        """Stop the player gracefully."""
        logger.info("Audio player stop...")
        self.running = False
        if self._task:
            self._task.cancel()
            await self._task
        await self.clear_queue()

    async def enqueue(self, text: str, start_playing: bool = True):
        await self._queue.put(text)
        if start_playing and not self.running:
            await self.start()

    async def _play_loop(self):
        """Main loop for the player (prints while playing)."""
        try:
            while self.running:
                text = await self._queue.get()
                logger.info("-----------------Playing audio...")
                await asyncio.to_thread(self._synthesize_and_stream_to_sd, text)
                self._queue.task_done()
                logger.info("------------------Finished playing")

        except asyncio.CancelledError:
            logger.info("CancelledError captured in _play_loop")

    async def close(self):
        """Clean up player task."""
        await self.stop()
        logger.info("Audio player closed.")


class PiperTTS(MQTTClient):
    """Text-to-speech component that converts text to speech using Piper TTS.
    This is meant to run in a separate thread if async is used"""

    def __init__(
        self,
        client,
        settings: Settings,
        name: str | None = None,
    ):
        super().__init__(client, settings, name)
        self.voice: PiperVoice | None = None
        self._audio = AudioPlaybackManager()

    @staticmethod
    def _prepare_text(text):
        return text

    @property
    def config(self):
        return self.settings.tts

    @on_receive("LLMTerminalChat", "tts_control", TTSControl)
    async def handle_control_event(self, msg, event: TTSControl):
        logger.info(f"voice control event received: {event.action}")
        match event.action:
            case "STOP" | "CLEAR_QUEUE":
                await self._audio.stop()
            case "MUTE":
                await self._audio.set_muted(True)
            case "UNMUTE":
                await self._audio.set_muted(False)
            case "PAUSE":
                await self._audio.pause()

    @on_receive("LLMProcessor", "output", LLMResponse, "complete")
    async def play_llm_output(self, msg, event: LLMResponse):
        logger.info(
            "Playing llm output text on event type 'complete' meaning unchunked"
        )
        chunk = TTSStreamChunk(content=event.content, message_type="complete")
        await self.handle_streaming_chunk(chunk)

    @on_receive("LLMTerminalChat", "output", TTSStreamChunk, "play")
    async def speak_terminal_user_query(self, msg, event: TTSStreamChunk):
        logger.info("Received play request for terminal text")
        await self.handle_streaming_chunk(event)

    async def handle_streaming_chunk(self, event: TTSStreamChunk) -> None:
        """Handle streaming text chunks from LLM."""
        logger.debug(
            f"Received stream chunk: {event.content[:30]}... (priority={event.priority})"
        )
        if event.message_type != "complete":
            logger.warning("Stream responses are not yet supported")
            return
        # TODO: do this not on every message but once only
        await self._ensure_voice()
        self._audio.voice = self.voice
        await self._audio.enqueue(event.content, start_playing=True)

    async def _ensure_voice(self) -> None:
        if not self.voice:
            self.voice = load_voice(
                voice_name=self.config.voice_model_name,
                model_path=self.config.voice_model_path,
                auto_download_voice=self.config.auto_download_voice,
                use_cuda=self.config.use_cuda,
            )

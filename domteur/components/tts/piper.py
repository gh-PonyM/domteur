"""Text-to-speech component using Piper TTS with direct audio output."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
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
from domteur.components.tts.audio import StreamingTextQueue
from domteur.components.tts.contracts import TTSControl, TTSStreamChunk

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
    STREAMING = "STREAMING"
    PLAYING = "PLAYING"  # Kept for compatibility; maps to active stream writes
    MUTED = "MUTED"


# Audio chunks are typically numpy int16 arrays from Piper; accept any buffer-like
# TODO: fix type hint and allow only one
AudioChunk = Any

# Priority levels (lower is higher priority)
PRIORITY_CRITICAL = 0
PRIORITY_HIGH = 1
PRIORITY_NORMAL = 2
PRIORITY_LOW = 3


class AudioPlaybackManager:
    """Priority-aware audio playback with mute and immediate stop.

    Notes
    - Lower integer means higher priority (0 is CRITICAL).
    - Only one active playback is handled at a time.
    - Same/low priority requests can be rejected here; upstream queue
      should typically ensure FIFO for same priority.
    """

    def __init__(self) -> None:
        self._state: AudioState = AudioState.IDLE
        self._muted: bool = False
        self._current_priority: int | None = None
        self._task: asyncio.Task | None = None
        self._stream: sd.OutputStream | None = None
        self._sample_rate: int | None = None
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> AudioState:
        return self._state

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def current_priority(self) -> int | None:
        return self._current_priority

    async def set_muted(self, muted: bool) -> None:
        async with self._lock:
            if self._muted == muted:
                return
            self._muted = muted
            if muted:
                self._state = AudioState.MUTED
                # Keep draining any active stream without audio output
                await self._close_stream()
            else:
                # When unmuting, the playback loop will lazily re-open the stream
                # on the next chunk write.
                if self._state == AudioState.MUTED:
                    # If we are not actively streaming data, return to IDLE
                    if not self._task or self._task.done():
                        self._state = AudioState.IDLE

    async def stop(self) -> None:
        """Immediately stop any active playback and reset state."""
        async with self._lock:
            self._stop_event.set()
            if self._task and not self._task.done():
                self._task.cancel()
            await self._close_stream()
            self._task = None
            self._current_priority = None
            self._sample_rate = None
            self._state = AudioState.MUTED if self._muted else AudioState.IDLE
            # Clear stop flag so future sessions can start
            self._stop_event = asyncio.Event()

    async def request_play_stream(
        self,
        sample_rate: int,
        chunks: AsyncIterator[AudioChunk],
        priority: int,
    ) -> bool:
        """Request to start streaming playback.

        Returns True if accepted. Rejects if lower priority than current.
        Higher priority interrupts current playback.
        """
        async with self._lock:
            if self._current_priority is not None and priority > self._current_priority:
                # Lower priority than current; reject
                return False

            # Interrupt current playback if any
            if self._task and not self._task.done():
                logger.debug(
                    f"Interrupting playback: new priority {priority} replaces {self._current_priority}"
                )
                self._stop_event.set()
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                finally:
                    await self._close_stream()

            # Prepare new session
            self._sample_rate = sample_rate
            self._current_priority = priority
            self._stop_event = asyncio.Event()
            self._state = AudioState.MUTED if self._muted else AudioState.STREAMING

            # Launch playback loop
            self._task = asyncio.create_task(self._playback_loop(chunks))
            return True

    async def _ensure_stream(self) -> None:
        if self._muted:
            return
        if self._stream is None:
            if self._sample_rate is None:
                raise RuntimeError("Sample rate not set before opening stream")
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate, channels=1, dtype="int16"
            )
            self._stream.start()
            self._state = AudioState.PLAYING

    async def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Error stopping audio stream: {e}")
            try:
                self._stream.close()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Error closing audio stream: {e}")
            finally:
                self._stream = None

    async def _playback_loop(self, chunks: AsyncIterator[AudioChunk]) -> None:
        loop = asyncio.get_running_loop()
        try:
            async for chunk in chunks:
                if self._stop_event.is_set():
                    break

                # Lazily open stream when needed and not muted
                if not self._muted:
                    await self._ensure_stream()

                try:
                    if self._muted:
                        # Drain without audio output
                        continue
                    stream = self._stream
                    if stream is None:
                        # If we became muted after ensure, just skip
                        continue
                    # Use executor to avoid blocking the event loop on writes
                    await loop.run_in_executor(None, stream.write, chunk)
                except asyncio.CancelledError:
                    raise
                except Exception as err:
                    logger.error(f"Exception during audio write: {err}")
                    break
        except asyncio.CancelledError:
            logger.info("Audio playback loop cancelled")
            raise
        except Exception as err:
            logger.error(f"Playback loop error: {err}")
        finally:
            async with self._lock:
                await self._close_stream()
                # Transition back to state depending on mute
                self._state = AudioState.MUTED if self._muted else AudioState.IDLE
                self._task = None
                self._current_priority = None
                self._sample_rate = None


class PiperTTS(MQTTClient):
    """Text-to-speech component that converts text to speech using Piper TTS.
    This is meant to run in a separate thread if async is used"""

    def __init__(
        self,
        client,
        settings: Settings,
        name: str | None = None,
        shutdown_event: asyncio.Event | None = None,
    ):
        super().__init__(client, name, shutdown_event=shutdown_event)
        self.config = settings.tts
        self.voice: PiperVoice | None = None
        self._audio = AudioPlaybackManager()
        self._shutdown_watcher: asyncio.Task | None = None
        self._streaming_queue = StreamingTextQueue()
        self._stream_processor_task: asyncio.Task | None = None

    @staticmethod
    def _prepare_text(text):
        return text

    @on_receive("LLMTerminalChat", "tts_control", TTSControl)
    async def handle_control_event(self, msg, event: TTSControl):
        logger.info(f"voice control event received: {event.action}")
        match event.action:
            case "STOP" | "CLEAR_QUEUE":
                await self._audio.stop()
                await self._streaming_queue.clear_all()
                if (
                    self._stream_processor_task
                    and not self._stream_processor_task.done()
                ):
                    self._stream_processor_task.cancel()
            case "MUTE":
                await self._audio.set_muted(True)
            case "UNMUTE":
                await self._audio.set_muted(False)

    @on_receive("LLMProcessor", "output", LLMResponse, "complete")
    async def play_llm_output(self, msg, event: LLMResponse):
        chunk = TTSStreamChunk(content=event.content, message_type="complete")
        await self.handle_streaming_chunk(chunk)

    @on_receive("LLMTerminalChat", "output", TTSStreamChunk, "play")
    async def speak_terminal_user_query(self, msg, event: TTSStreamChunk):
        await self.handle_streaming_chunk(event)

    async def handle_streaming_chunk(self, event: TTSStreamChunk) -> None:
        """Handle streaming text chunks from LLM."""
        logger.debug(
            f"Received stream chunk: {event.content[:30]}... (priority={event.priority})"
        )

        # Check for priority interrupt
        await self._streaming_queue.clear_lower_priority(event.priority)

        # Add token to buffer
        await self._streaming_queue.add_token(event.content, event.priority)

        # Start stream processor if not running
        if self._stream_processor_task is None or self._stream_processor_task.done():
            self._stream_processor_task = asyncio.create_task(
                self._process_streaming_queue()
            )

        # Mark complete if this is the final chunk
        if event.message_type == "complete":
            await self._streaming_queue.mark_complete()

    # @on_receive(TOPIC_COMPONENT_LLM_PROC_ANSWER, LLMResponse)
    # async def handle_tts_request(self, event: LLMResponse) -> None:
    #     """Handle LLM responses."""
    #     text = self._prepare_text(event.content)
    #     logger.info(f"Processing TTS request: {text[:50]}...")
    #
    #     try:
    #         await self._synthesize_and_play(text)
    #         logger.info("TTS synthesis and playback completed")
    #     except Exception as e:
    #         logger.error(f"TTS synthesis failed: {e}")
    #         # Send a mqtt message in the background by raising exception
    #         raise

    async def _process_streaming_queue(self) -> None:
        """Process sentences from streaming queue and synthesize them."""
        try:
            while True:
                sentence_data = await self._streaming_queue.get_next_sentence()
                if sentence_data is None:
                    await asyncio.sleep(0.1)
                    continue

                priority, sentence = sentence_data
                logger.info(
                    f"Processing streamed sentence: {sentence[:50]}... (priority={priority})"
                )

                try:
                    await self._ensure_voice()
                    cancel_evt = threading.Event()
                    chunks = self._chunks_from_text(sentence, cancel_evt)
                    accepted = await self._audio.request_play_stream(
                        self.voice.config.sample_rate if self.voice else 22050,
                        chunks,
                        priority=priority,
                    )
                    if not accepted:
                        logger.info(
                            f"Streaming playback rejected (priority={priority})"
                        )
                except Exception as e:
                    logger.error(f"Error processing streamed sentence: {e}")

        except asyncio.CancelledError:
            logger.info("Stream processor cancelled")
            raise

    def _iter_chunks_sync(self, text: str, cancel_evt: threading.Event):
        assert self.voice is not None
        for chunk in self.voice.synthesize(text):
            if cancel_evt.is_set():
                break
            yield chunk.audio_int16_array

    async def _chunks_from_text(
        self, text: str, cancel_evt: threading.Event
    ) -> AsyncIterator[Any]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=16)

        def worker():
            try:
                for a in self._iter_chunks_sync(text, cancel_evt):
                    if cancel_evt.is_set():
                        break
                    # Block if queue is full until consumer catches up
                    asyncio.run_coroutine_threadsafe(queue.put(a), loop).result()
            except Exception as e:  # pragma: no cover - defensive
                logger.error(f"Synthesis worker error: {e}")
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            cancel_evt.set()
            thread.join(timeout=1.0)

    async def _ensure_voice(self) -> None:
        if not self.voice:
            self.voice = load_voice(
                voice_name=self.config.voice_model_name,
                model_path=self.config.voice_model_path,
                auto_download_voice=self.config.auto_download_voice,
                use_cuda=self.config.use_cuda,
            )

    async def _watch_shutdown(self) -> None:
        if self.shutdown_event is None:
            return
        await self.shutdown_event.wait()
        await self._audio.stop()
        await self._streaming_queue.stop()
        if self._stream_processor_task and not self._stream_processor_task.done():
            self._stream_processor_task.cancel()

    async def start(self):
        await self.initialize_subscriptions()
        if self._shutdown_watcher is None:
            self._shutdown_watcher = asyncio.create_task(self._watch_shutdown())
        await self.listen()

    async def _synthesize_and_play(self, text: str) -> None:
        await self._ensure_voice()
        cancel_evt = threading.Event()
        chunks = self._chunks_from_text(text, cancel_evt)
        accepted = await self._audio.request_play_stream(
            self.voice.config.sample_rate if self.voice else 22050,
            chunks,
            priority=PRIORITY_NORMAL,
        )
        if not accepted:
            logger.info("Playback request rejected due to lower priority")

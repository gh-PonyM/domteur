import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from loguru import logger

from domteur.components.stt.configs import AudioStreamConfig, WhisperSTTConfig


def volume_bar(indata: np.ndarray):
    # Calculate volume
    rms = np.sqrt(np.mean(indata**2))
    db = 20 * np.log10(rms + 1e-10)

    # Visual volume meter
    bar_length = int(max(0, min(50, (db + 60) / 60 * 50)))
    bar = "█" * bar_length + "░" * (50 - bar_length)
    print(f"\rVolume: {bar} {db:.1f} dB", end="", flush=True)


def _audio_callback(indata: np.ndarray, frames: int, time_info, status, model) -> None:
    """Process audio chunks from the stream."""
    if status:
        logger.warning(f"Audio stream status: {status}")
    # TODO
    # Detect silence, put chunks into a buffer, if silence occurs, put the buffer into the async queue
    pass


# TODO: cache this function call depending in the cfg input
def load_model(cfg: WhisperSTTConfig):
    """Also downloads the model"""
    return WhisperModel(
        cfg.model_size,
        device=cfg.device,
        compute_type=cfg.compute_type,
        download_root=str(cfg.model_path.resolve()),
    )


# @contextmanager
# def streaming(cfg: WhisperSTTConfig, callback: Callable):
#     with sd.InputStream(
#         samplerate=cfg.sample_rate,
#         channels=cfg.channels,
#         blocksize=cfg.block_size,
#         callback=callback,
#     ):
#         sd.sleep(int(1e10))
#         yield


@dataclass
class StreamingTranscriber:
    """Async handler for live audio stream transcription with silence detection.

    Handles incoming audio stream from a live source hardware audio device,
    using silence detection and buffering audio chunks to hand over to speech to text transcription using
    faster_whisper.
    """

    model: WhisperModel
    cfg: AudioStreamConfig
    whisper_config: WhisperSTTConfig = field(default_factory=WhisperSTTConfig)

    _buffer: np.ndarray | None = None
    # Async queue for processed audio chunks
    _queue: asyncio.Queue[np.ndarray | None] = field(default_factory=asyncio.Queue)
    _silence_duration: float = field(default=0.0)
    _is_running: bool = field(default=False)

    def emtpy_buffer(self):
        return np.empty((0, self.cfg.channels), dtype=np.float32)

    def __post_init__(self):
        """Initialize buffer with correct shape based on config channels."""
        self._buffer = self.emtpy_buffer()

    async def _detect_silence(self, indata: np.ndarray) -> bool:
        """Detect if audio chunk is below silence threshold."""
        rms = np.sqrt(np.mean(indata**2))
        db = 20 * np.log10(rms + 1e-10)
        return db < self.cfg.limiter_threshold

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Synchronous callback from sounddevice - queues work for async processing."""
        if status:
            logger.warning(f"Audio stream status: {status}")

        # Put audio chunk into queue for async processing
        # volume_bar(indata)
        try:
            self._queue.put_nowait(indata.copy())
        except asyncio.QueueFull:
            logger.warning("Audio queue full, dropping frame")

    async def _process_audio_stream(self) -> AsyncIterator[np.ndarray]:
        """Process audio chunks with silence detection and buffering.

        Yields complete audio buffers when accumulated silence duration exceeds the configured threshold.
        """
        last_sec_log: int = 0

        def log_silence():
            nonlocal last_sec_log
            # Log silence every 2 seconds at info level
            if self._silence_duration == 0.0:
                logger.info("Silence detected for the first time")
            else:
                dur = int(self._silence_duration)
                if dur % 2 == 0 and dur != last_sec_log:
                    logger.info(
                        f"Silence detected: {self._silence_duration:.2f}s accumulated"
                    )
                    last_sec_log = dur

        while self._is_running:
            try:
                # Get audio chunk from queue with timeout
                indata = await asyncio.wait_for(self._queue.get(), timeout=1.0)

                if indata is None:  # Sentinel value to stop
                    break

                # Check for silence
                is_silent = await self._detect_silence(indata)

                if not is_silent:
                    # Reset silence counter and add to buffer
                    self._silence_duration = 0.0
                    self._buffer = np.concatenate([self._buffer, indata])
                    continue

                log_silence()
                # Accumulate silence duration
                chunk_duration = len(indata) / self.cfg.sample_rate
                self._silence_duration += chunk_duration

                # If silence exceeds threshold and buffer has audio, yield buffer
                if (
                    self._buffer.size > 0
                    and self._silence_duration >= self.cfg.split_after_silence_secs
                ):
                    logger.info(
                        f"Yielding audio chunk: {len(self._buffer)} samples ({len(self._buffer) / self.cfg.sample_rate:.2f}s)"
                    )
                    yield self._buffer.copy()
                    self._buffer = self.emtpy_buffer()
                    self._silence_duration = 0.0

            except TimeoutError:
                # Timeout waiting for audio - check if we should flush buffer
                if (
                    self._buffer.size > 0
                    and self._silence_duration >= self.cfg.split_after_silence_secs
                ):
                    logger.info(
                        f"Yielding audio chunk (timeout): {len(self._buffer)} samples ({len(self._buffer) / self.cfg.sample_rate:.2f}s)"
                    )
                    yield self._buffer.copy()
                    self._buffer = self.emtpy_buffer()
                    self._silence_duration = 0.0

    async def transcribe_stream(self) -> AsyncIterator[str]:
        """Main async generator for transcribing live audio stream.

        Yields transcribed text segments as they become available.
        """
        self._is_running = True
        try:
            async for audio_chunk in self._process_audio_stream():
                # Run transcription in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: self.model.transcribe(
                        audio_chunk,
                        beam_size=self.whisper_config.beam_size,
                        word_timestamps=False,
                    ),
                )

                # Yield transcribed text
                for segment in segments:
                    yield segment.text

        finally:
            self._is_running = False

    @asynccontextmanager
    async def stream(self):
        """Async context manager for audio streaming.

        Manages the lifecycle of the audio input stream.
        """

        # Run sounddevice stream in background
        def run_stream():
            with sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                blocksize=self.cfg.block_size,
                callback=self._audio_callback,
            ):
                while self._is_running:
                    sd.sleep(100)

        # Start stream in thread
        stream_thread = asyncio.to_thread(run_stream)
        task = asyncio.create_task(stream_thread)

        try:
            yield
        finally:
            self._is_running = False
            await task
            self._queue.put_nowait(None)  # Sentinel to stop processing

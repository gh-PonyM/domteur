import asyncio
import functools
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from loguru import logger

from domteur.components.stt.configs import AudioStreamConfig, WhisperSTTConfig


def volume_bar(db, silence_threshold: float = -40.0):
    """Display volume bar with colored bars based on silence threshold.

    Args:
        silence_threshold: dB threshold for silence detection (default: -40.0)
    """
    # Calculate volume

    # ANSI color codes
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    # Visual volume meter
    bar_length = int(max(0, min(50, (db + 60) / 60 * 50)))

    # Calculate threshold position in the bar (0-50 range)
    threshold_position = int(max(0, min(50, (silence_threshold + 60) / 60 * 50)))

    # Color bars: red below threshold, green at/above threshold
    red_bars = min(bar_length, threshold_position)
    green_bars = max(0, bar_length - threshold_position)
    empty_bars = 50 - bar_length

    bar = f"{RED}{'█' * red_bars}{GREEN}{'█' * green_bars}{RESET}{'░' * empty_bars}"
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
    whisper_config: WhisperSTTConfig = field(default_factory=WhisperSTTConfig)  # type: ignore
    dry_run: bool = False
    show_audio_bar: bool = False

    # Use deque for efficient buffering instead of np.concatenate
    _buffer_chunks: deque[np.ndarray] = field(default_factory=deque)
    # Async queue for processed audio chunks
    _queue: asyncio.Queue[np.ndarray | None] = field(default_factory=asyncio.Queue)
    _silence_duration: float = field(default=0.0)
    _is_running: bool = field(default=False)
    # Maximum buffer duration in seconds (safety limit)
    _max_buffer_duration: float = field(default=90.0)

    def _get_buffer_duration(self) -> float:
        """Calculate current buffer duration in seconds."""
        if not self._buffer_chunks:
            return 0.0
        return sum(len(chunk) for chunk in self._buffer_chunks) / self.cfg.sample_rate

    def _get_buffer_size_mb(self) -> float:
        """Calculate current buffer size in MB."""
        if not self._buffer_chunks:
            return 0.0
        total_samples = sum(len(chunk) for chunk in self._buffer_chunks)
        # Each sample is float32 (4 bytes), mono audio (1 channel)
        return (total_samples * 4) / (1024 * 1024)

    def _flush_buffer(self) -> np.ndarray | None:
        """Concatenate all buffered chunks and return as single array."""
        if not self._buffer_chunks:
            return None
        audio_buffer = np.concatenate(list(self._buffer_chunks))
        self._buffer_chunks.clear()
        return audio_buffer

    def _clear_buffer(self) -> None:
        """Clear all buffered chunks."""
        self._buffer_chunks.clear()

    async def _detect_silence(self, indata: np.ndarray) -> bool:
        """Detect if audio chunk is below silence threshold."""
        rms = np.sqrt(np.mean(indata**2))
        db = 20 * np.log10(rms + 1e-10)
        return db < self.cfg.limiter_threshold

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Synchronous callback from sounddevice - queues work for async processing.

        Args:
            indata: Audio data with shape (frames, channels) - e.g., (1024, 2) for stereo
        """
        if status:
            logger.warning(f"Audio stream status: {status}")

        # Convert stereo to mono by averaging channels
        # indata shape: (frames, channels) -> mono shape: (frames,)
        # faster-whisper expects 1D mono audio array
        if indata.shape[1] > 1:
            mono_audio = np.mean(indata, axis=1)
        else:
            mono_audio = indata.squeeze()

        if self.show_audio_bar:
            rms = np.sqrt(np.mean(indata ** 2))
            db = 20 * np.log10(rms + 1e-10)
            volume_bar(db, self.cfg.limiter_threshold)

        try:
            self._queue.put_nowait(mono_audio.copy())
        except asyncio.QueueFull:
            logger.warning("Audio queue full, dropping frame")

    async def _process_audio_stream(self) -> AsyncIterator[np.ndarray]:
        """Process audio chunks with silence detection and buffering.

        Yields complete audio buffers when accumulated silence duration exceeds the configured threshold
        or when buffer reaches maximum duration limit.
        """

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
                    self._buffer_chunks.append(indata)

                    # Check if buffer exceeds maximum duration
                    buffer_duration = self._get_buffer_duration()
                    if buffer_duration >= self._max_buffer_duration:
                        buffer_size_mb = self._get_buffer_size_mb()
                        logger.warning(
                            f"Buffer limit reached ({buffer_duration:.2f}s, {buffer_size_mb:.2f} MB), "
                            f"forcing yield to prevent memory overflow"
                        )
                        audio_buffer = self._flush_buffer()
                        if audio_buffer is not None:
                            yield audio_buffer
                        self._silence_duration = 0.0
                    continue

                # Accumulate silence duration
                chunk_duration = len(indata) / self.cfg.sample_rate
                self._silence_duration += chunk_duration

                # If silence exceeds threshold and buffer has audio, yield buffer
                if (
                    self._buffer_chunks
                    and self._silence_duration >= self.cfg.split_after_silence_secs
                ):
                    buffer_duration = self._get_buffer_duration()
                    buffer_size_mb = self._get_buffer_size_mb()
                    logger.info(
                        f"Yielding audio chunk: {buffer_duration:.2f}s ({buffer_size_mb:.2f} MB), silence:{self._silence_duration:1f}"
                    )
                    audio_buffer = self._flush_buffer()
                    if audio_buffer is not None:
                        yield audio_buffer
                    self._silence_duration = 0.0

            except TimeoutError:
                # Timeout waiting for audio - check if we should flush buffer
                if (
                    self._buffer_chunks
                    and self._silence_duration >= self.cfg.split_after_silence_secs
                ):
                    buffer_duration = self._get_buffer_duration()
                    buffer_size_mb = self._get_buffer_size_mb()
                    logger.info(
                        f"Yielding audio chunk (timeout): {buffer_duration:.2f}s ({buffer_size_mb:.2f} MB)"
                    )
                    audio_buffer = self._flush_buffer()
                    if audio_buffer is not None:
                        yield audio_buffer
                    self._silence_duration = 0.0

    async def transcribe_stream(self) -> AsyncIterator[str]:
        """Main async generator for transcribing live audio stream.

        Yields transcribed text segments as they become available.
        """
        self._is_running = True
        try:
            async for audio_chunk in self._process_audio_stream():
                # Run transcription in thread pool to avoid blocking
                if self.dry_run:
                    yield "DRY-RUN: yield audio chunk"
                    continue

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

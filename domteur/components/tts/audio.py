"""Streaming audio buffer utilities for TTS."""

from __future__ import annotations

import asyncio
import re
import time

from loguru import logger


class SentenceBuffer:
    """Accumulates streaming text tokens and detects sentence boundaries.

    Supports multi-language sentence detection (en, de, fr) with fallback
    mechanisms for timeout and buffer size limits.
    """

    def __init__(
        self,
        max_buffer_size: int = 500,
        sentence_timeout: float = 2.0,
        min_sentence_length: int = 10,
    ):
        self._buffer: list[str] = []
        self._max_buffer_size = max_buffer_size
        self._sentence_timeout = sentence_timeout
        self._min_sentence_length = min_sentence_length
        self._last_token_time: float = 0.0
        self._lock = asyncio.Lock()

        # Sentence boundary patterns (en, de, fr)
        self._sentence_end_pattern = re.compile(
            r"[.!?;]\s+|[.!?;]$|…\s+|…$|\.\.\.\s+|\.\.\.$"
        )

        # Abbreviations to avoid false positives
        self._abbreviations = re.compile(
            r"\b(mr|mrs|ms|dr|prof|sr|jr|vs|etc|e\.g|i\.e|ca|bzw|usw|z\.b|fr|m|mme)\.$",
            re.IGNORECASE,
        )

    async def add_token(self, token: str) -> str | None:
        """Add token to buffer. Returns complete sentence if boundary detected."""
        async with self._lock:
            self._buffer.append(token)
            self._last_token_time = time.monotonic()

            current_text = "".join(self._buffer)

            # Check for sentence boundary
            if self._is_sentence_complete(current_text):
                sentence = current_text.strip()
                self._buffer.clear()
                return sentence

            # Fallback: buffer size limit
            if len(current_text) >= self._max_buffer_size:
                logger.debug(
                    f"Buffer size limit reached ({self._max_buffer_size}), flushing"
                )
                sentence = current_text.strip()
                self._buffer.clear()
                return sentence

            return None

    async def flush(self) -> str | None:
        """Force flush remaining buffer content."""
        async with self._lock:
            if not self._buffer:
                return None
            sentence = "".join(self._buffer).strip()
            self._buffer.clear()
            return sentence

    async def check_timeout(self) -> str | None:
        """Check if timeout has elapsed since last token. Returns sentence if timeout."""
        async with self._lock:
            if not self._buffer:
                return None

            elapsed = time.monotonic() - self._last_token_time
            if elapsed >= self._sentence_timeout:
                current_text = "".join(self._buffer).strip()
                if len(current_text) >= self._min_sentence_length:
                    logger.debug(
                        f"Sentence timeout ({self._sentence_timeout}s), flushing"
                    )
                    self._buffer.clear()
                    return current_text

            return None

    def _is_sentence_complete(self, text: str) -> bool:
        """Detect if text contains a complete sentence boundary."""
        if len(text) < self._min_sentence_length:
            return False

        # Check for sentence-ending punctuation
        match = self._sentence_end_pattern.search(text)
        if not match:
            return False

        # Check for abbreviations before the punctuation
        before_punct = text[: match.start() + 1]
        if self._abbreviations.search(before_punct):
            return False

        return True

    async def clear(self) -> None:
        """Clear the buffer."""
        async with self._lock:
            self._buffer.clear()


class StreamingTextQueue:
    """Three-tier queue for streaming TTS: tokens → sentences → audio chunks.

    Handles priority-aware processing with interrupt capability.
    """

    def __init__(
        self,
        max_sentence_queue: int = 10,
        max_audio_queue: int = 5,
        sentence_timeout: float = 2.0,
    ):
        self._sentence_buffer = SentenceBuffer(sentence_timeout=sentence_timeout)
        self._sentence_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue(
            maxsize=max_sentence_queue
        )
        self._audio_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(
            maxsize=max_audio_queue
        )
        self._current_priority: int | None = None
        self._stop_event = asyncio.Event()
        self._timeout_task: asyncio.Task | None = None

    async def add_token(self, token: str, priority: int) -> None:
        """Add streaming token with priority."""
        if self._stop_event.is_set():
            return

        # Check priority interruption
        if self._current_priority is not None and priority < self._current_priority:
            await self.clear_all()

        self._current_priority = priority

        # Accumulate into sentence
        sentence = await self._sentence_buffer.add_token(token)
        if sentence:
            await self._sentence_queue.put((priority, sentence))

        # Start timeout checker if not running
        if self._timeout_task is None or self._timeout_task.done():
            self._timeout_task = asyncio.create_task(self._timeout_checker())

    async def mark_complete(self) -> None:
        """Mark streaming complete and flush remaining buffer."""
        sentence = await self._sentence_buffer.flush()
        if sentence and self._current_priority is not None:
            await self._sentence_queue.put((self._current_priority, sentence))

    async def get_next_sentence(self) -> tuple[int, str] | None:
        """Get next sentence from queue (non-blocking)."""
        try:
            return self._sentence_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def clear_all(self) -> None:
        """Clear all queues (tokens, sentences, audio)."""
        await self._sentence_buffer.clear()

        # Drain sentence queue
        while not self._sentence_queue.empty():
            try:
                self._sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Drain audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._current_priority = None
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

    async def clear_lower_priority(self, priority: int) -> None:
        """Clear queues if current priority is lower than given priority."""
        if self._current_priority is not None and priority < self._current_priority:
            await self.clear_all()

    async def _timeout_checker(self) -> None:
        """Periodically check for sentence timeout."""
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.5)  # Check every 500ms
                sentence = await self._sentence_buffer.check_timeout()
                if sentence and self._current_priority is not None:
                    await self._sentence_queue.put((self._current_priority, sentence))
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop timeout checker."""
        self._stop_event.set()
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()

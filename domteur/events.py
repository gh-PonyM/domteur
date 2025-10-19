"""Event system for domteur with topic-based architecture for future pub/sub compatibility."""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event types that flow through the system."""

    USER_INPUT = "user_input"
    LLM_RESPONSE = "llm_response"
    TTS_REQUEST = "tts_request"
    PERSIST_MESSAGE = "persist_message"
    SYSTEM_SHUTDOWN = "system_shutdown"


class Event(BaseModel):
    """Event message structure for inter-component communication."""

    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    payload: dict[str, Any] = Field(default_factory=dict)
    topic: str = Field(default="default")

    def to_json(self) -> str:
        """Serialize event to JSON for pub/sub compatibility."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "Event":
        """Deserialize event from JSON for pub/sub compatibility."""
        return cls.model_validate_json(json_str)


class EventBusInterface(ABC):
    """Abstract interface for event bus implementations."""

    @abstractmethod
    async def publish(self, topic: str, event: Event) -> None:
        """Publish an event to a topic."""
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        """Subscribe to a topic with a handler."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the event bus."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the event bus gracefully."""
        pass


class InMemoryEventBus(EventBusInterface):
    """In-memory event bus with topic support for future pub/sub migration."""

    def __init__(self):
        self._subscribers: dict[str, list[Callable[[Event], None]]] = {}
        self._queue: asyncio.Queue[tuple[str, Event]] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task | None = None

    async def publish(self, topic: str, event: Event) -> None:
        """Publish an event to a topic."""
        if not self._running:
            logger.warning("EventBus not running, event will be queued")

        # Set topic on event for consistency
        event.topic = topic
        await self._queue.put((topic, event))
        logger.debug(f"Published event {event.event_type} to topic {topic}")

    async def subscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        """Subscribe to a topic with a handler."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []

        self._subscribers[topic].append(handler)
        logger.debug(f"Subscribed handler to topic {topic}")

    async def start(self) -> None:
        """Start the event bus processing loop."""
        if self._running:
            logger.warning("EventBus already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("EventBus started")

    async def stop(self) -> None:
        """Stop the event bus gracefully."""
        if not self._running:
            return

        self._running = False

        # Send shutdown signal
        await self.publish("system", Event(event_type=EventType.SYSTEM_SHUTDOWN))

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("EventBus shutdown timeout, cancelling task")
                self._task.cancel()

        logger.info("EventBus stopped")

    async def _process_events(self) -> None:
        """Process events from the queue and dispatch to subscribers."""
        while self._running:
            try:
                # Use timeout to allow periodic checking of _running flag
                topic, event = await asyncio.wait_for(self._queue.get(), timeout=1.0)

                # Handle system shutdown
                if event.event_type == EventType.SYSTEM_SHUTDOWN:
                    logger.info("Received shutdown event")
                    break

                # Dispatch to subscribers
                handlers = self._subscribers.get(topic, [])
                if handlers:
                    # Run handlers concurrently but catch individual failures
                    await self._dispatch_to_handlers(handlers, event)
                else:
                    logger.debug(f"No subscribers for topic {topic}")

            except asyncio.TimeoutError:
                # Timeout is expected, continue loop
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    async def _dispatch_to_handlers(
        self, handlers: list[Callable], event: Event
    ) -> None:
        """Dispatch event to all handlers, catching individual failures."""
        tasks = []

        for handler in handlers:
            try:
                # Handle both sync and async handlers
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                else:
                    # Run sync handler in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    tasks.append(loop.run_in_executor(None, handler, event))
            except Exception as e:
                logger.error(f"Error creating task for handler {handler}: {e}")

        # Wait for all handlers to complete, catch failures
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Handler {handlers[i]} failed: {result}")


# Convenience functions for topic naming - makes migration to pub/sub easier
class Topics:
    """Standard topic names for the application."""

    USER_INPUT = "input"
    LLM_PROCESSING = "llm"
    TTS_OUTPUT = "tts"
    PERSISTENCE = "persistence"
    SYSTEM = "system"


# Global event bus instance (can be replaced with Redis/other implementation)
_event_bus: EventBusInterface | None = None


def get_event_bus() -> EventBusInterface:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = InMemoryEventBus()
    return _event_bus


def set_event_bus(event_bus: EventBusInterface) -> None:
    """Set a custom event bus implementation (for testing or Redis migration)."""
    global _event_bus
    _event_bus = event_bus

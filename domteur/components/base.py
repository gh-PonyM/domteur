"""Base component class for event-driven architecture."""

import asyncio
import logging
from abc import ABC, abstractmethod

from domteur.events import Event, EventBusInterface, get_event_bus

logger = logging.getLogger(__name__)


class Component(ABC):
    """Base class for all components in the event-driven system."""

    def __init__(self, name: str, event_bus: EventBusInterface | None = None):
        self.name = name
        self.event_bus = event_bus or get_event_bus()
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._subscriptions: dict[str, list[str]] = {}  # topic -> handler_names

        logger.info(f"Component {self.name} initialized")

    async def start(self) -> None:
        """Start the component and register event handlers."""
        if self._running:
            logger.warning(f"Component {self.name} already running")
            return

        self._running = True
        await self._register_handlers()
        await self._start_background_tasks()

        logger.info(f"Component {self.name} started")

    async def stop(self) -> None:
        """Stop the component gracefully."""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        await self._cleanup()
        logger.info(f"Component {self.name} stopped")

    async def publish(self, topic: str, event: Event) -> None:
        """Publish an event to a topic."""
        await self.event_bus.publish(topic, event)
        logger.debug(f"Component {self.name} published {event.event_type} to {topic}")

    async def subscribe(self, topic: str, handler_name: str) -> None:
        """Subscribe to a topic with a named handler method."""
        if not hasattr(self, handler_name):
            raise AttributeError(f"Handler method {handler_name} not found")

        handler = getattr(self, handler_name)
        await self.event_bus.subscribe(topic, handler)

        # Track subscription for debugging
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
        self._subscriptions[topic].append(handler_name)

        logger.debug(f"Component {self.name} subscribed to {topic} with {handler_name}")

    def add_background_task(self, coro) -> None:
        """Add a background coroutine to run while component is active."""
        if self._running:
            task = asyncio.create_task(coro)
            self._tasks.append(task)
        else:
            # Store for later start
            if not hasattr(self, "_pending_tasks"):
                self._pending_tasks = []
            self._pending_tasks.append(coro)

    @abstractmethod
    async def _register_handlers(self) -> None:
        """Register event handlers - must be implemented by subclasses."""
        pass

    async def _start_background_tasks(self) -> None:
        """Start any pending background tasks."""
        if hasattr(self, "_pending_tasks"):
            for coro in self._pending_tasks:
                task = asyncio.create_task(coro)
                self._tasks.append(task)
            delattr(self, "_pending_tasks")

    @abstractmethod
    async def _cleanup(self) -> None:
        """Override this method for component-specific cleanup."""
        pass

    def is_running(self) -> bool:
        """Check if component is currently running."""
        return self._running

    def get_subscriptions(self) -> dict[str, list[str]]:
        """Get current topic subscriptions for debugging."""
        return self._subscriptions.copy()

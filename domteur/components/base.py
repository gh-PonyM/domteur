"""Base component class for event-driven architecture."""

import asyncio
import inspect
import json
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Literal

import aiomqtt
import pydantic
from aiomqtt import Client
from loguru import logger
from pydantic import BaseModel, Field


@dataclass
class ContractMap:
    topic: str
    contract: type[BaseModel]
    component: str
    method_name: str
    type: Literal["pub", "sub"]


@dataclass
class ContractRegistry:
    items: list[ContractMap]


# Private instance of ContractRegistry for managing decorators
__contract_registry = ContractRegistry(items=[])


def on_receive(topic: str, payload_contract: type["MessagePayload"]):
    """Decorator for registering message receive handlers with automatic validation."""
    global __contract_registry

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, msg) -> None:
            src_topic = str(msg.topic) if msg else None
            # Decode JSON payload
            try:
                data = json.loads(msg.payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as err:
                await self._send_error_response(
                    session_id="unknown",
                    error=f"Invalid JSON payload: {err}",
                    source_topic=src_topic,
                )
                return

            # Validate against contract
            try:
                event = payload_contract.model_validate(data)
            except pydantic.ValidationError as err:
                await self._send_error_response(
                    session_id=data.get("session_id", "unknown"),
                    error=err,
                    source_topic=src_topic,
                )
                return

            # Call original function with full msg and validated event
            await func(self, msg, event)

        # Store handler info directly on the function for instance discovery
        wrapper._mqtt_handler_topic = topic

        # Register the contract in the global registry
        __contract_registry.items.append(
            ContractMap(
                topic=topic,
                contract=payload_contract,
                component=func.__qualname__.split(".")[0]
                if "." in func.__qualname__
                else "unknown",
                method_name=func.__name__,
                type="sub",
            )
        )

        return wrapper

    return decorator


def on_publish(topic: str, payload_contract: type["MessagePayload"]):
    """Decorator for registering message publish handlers."""
    global __contract_registry

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, msg: aiomqtt.Message | None, *args, **kwargs):
            try:
                response = await func(self, msg, *args, **kwargs)
            except Exception as err:
                await self._send_error_response(
                    session_id="unknown",
                    error=err,
                    source_topic=str(msg.topic) if msg else None,
                )
                return
            # TODO: proper identification if what the component_id of the instance is
            await self.publish(topic.replace("+", self.name), response)

        # Register the contract in the global registry
        __contract_registry.items.append(
            ContractMap(
                topic=topic,
                contract=payload_contract,
                component=func.__qualname__.split(".")[0]
                if "." in func.__qualname__
                else "unknown",
                method_name=func.__name__,
                type="pub",
            )
        )

        return wrapper

    return decorator


class MessagePayload(BaseModel):
    """Base type for payloads that all the payload contracts must adhere to"""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> str:
        """Serialize event to JSON for pub/sub compatibility."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "MessagePayload":
        """Deserialize event from JSON for pub/sub compatibility."""
        return cls.model_validate_json(json_str)


class Error(MessagePayload):
    content: str


class MQTTClient:
    """Base class for all mqtt client in the event-driven system."""

    component_name: str = "template"

    def __init__(
        self,
        client: Client,
        name: str | None = None,
        shutdown_event: asyncio.Event | None = None,
    ):
        self.name = name if name else self.default_name
        self.client = client
        self.shutdown_event = shutdown_event
        logger.info(f"Component {self.name} initialized")

    async def publish(self, topic: str, payload: MessagePayload | Error) -> None:
        """Publish an event to a topic."""
        await self.client.publish(topic, payload.model_dump_json())

    async def subscribe(self, topic: str) -> None:
        """Subscribe to a topic with a named handler method."""
        await self.client.subscribe(topic)

    @property
    def default_name(self):
        return f"{self.__class__.__name__.lower()}{random.randint(1000, 9999)}"

    @classmethod
    def _component_topic(cls):
        assert cls.component_name != "template", (
            "Please change the classes component_name attribute"
        )
        return f"component/{cls.component_name}"

    @property
    def instance_topic(self):
        return f"{self._component_topic()}/{self.name}"

    @property
    def error_topic(self):
        """The default topic for each component to send a standardized error message"""
        return f"{self.instance_topic}/error"

    @staticmethod
    def _format_error(err: Any) -> str:
        return str(err)

    async def _send_error_response(
        self, session_id: str, error: Any, source_topic: str | None = None
    ) -> None:
        """Send error response event with optional source topic information."""
        error_content = self._format_error(error)
        if source_topic:
            error_content = f"[from {source_topic}] {error_content}"

        await self.publish(
            self.error_topic,
            Error(
                session_id=session_id,
                content=error_content,
            ),
        )

    def _get_decorated_handlers(self) -> dict[str, Callable]:
        """Discover all methods decorated with @on_receive in this instance."""
        handlers = {}

        # Inspect all methods in this instance
        for method_name in dir(self):
            method = getattr(self, method_name)
            if inspect.ismethod(method) and hasattr(method, "_mqtt_handler_topic"):
                topic = method._mqtt_handler_topic
                handlers[topic] = method

        return handlers

    async def initialize_subscriptions(self):
        """Auto-discover and subscribe to all registered topics for this component."""
        handlers = self._get_decorated_handlers()

        for topic in handlers.keys():
            logger.info(f"Auto-subscribing {self.name} to topic: {topic}")
            await self.subscribe(topic)

    async def handle_message(self, msg):
        """Auto-route incoming messages to decorated handlers based on topic matching."""
        handlers = self._get_decorated_handlers()

        # Find matching handler by topic
        matched = 0
        for topic, method in handlers.items():
            if msg.topic.matches(topic):
                handler = method
                matched += 1
                await handler(msg)  # Pass full msg object to decorated handler
        if not matched:
            logger.critical(
                f"No handler found for topic: {msg.topic} in component {self.name}"
            )

    async def listen(self):
        async for msg in self.client.messages:
            await self.handle_message(msg)

    async def start(self):
        """Starts a 'serve forever' function"""
        await self.initialize_subscriptions()
        # waiting for messages is running forever: https://aiomqtt.bo3hm.com/subscribing-to-a-topic.html#listening-without-blocking
        await self.listen()


async def start_cli_client(
    client: aiomqtt.Client,
    mqtt_client: type[MQTTClient],
    shutdown_event: asyncio.Event,
    reconnect_interval: int = 5,
    **client_kwargs,
):
    """Start the aiomqtt client with reconnection"""
    while not shutdown_event.is_set():
        try:
            async with client:
                instance = mqtt_client(
                    client, shutdown_event=shutdown_event, **client_kwargs
                )

                # Create tasks for client and shutdown monitoring
                client_task = asyncio.create_task(instance.start())
                shutdown_task = asyncio.create_task(shutdown_event.wait())

                try:
                    # Wait for either client completion or shutdown
                    done, pending = await asyncio.wait(
                        [client_task, shutdown_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    # If shutdown was triggered, break out of retry loop
                    if shutdown_event.is_set():
                        break

                except asyncio.CancelledError:
                    # Handle cancellation gracefully
                    logger.info("Client task cancelled during shutdown")
                    for task in [client_task, shutdown_task]:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                    break

        except aiomqtt.MqttError as e:
            if shutdown_event.is_set():
                logger.info("MQTT error during shutdown, exiting")
                break
            logger.warning(
                f"Connection lost ({e}), reconnecting in {reconnect_interval} seconds"
            )
            try:
                await asyncio.wait_for(asyncio.sleep(reconnect_interval), timeout=1.0)
            except asyncio.TimeoutError:
                if shutdown_event.is_set():
                    break


def get_registry_items():
    return __contract_registry.items

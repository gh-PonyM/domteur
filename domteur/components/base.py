"""Base component class for event-driven architecture."""

import asyncio
import logging
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import aiomqtt
from aiomqtt import Client
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class ContractMap:
    topic: str
    contract: BaseModel
    component: str
    type: Literal["pub", "sub"]


@dataclass
class ContractRegistry:
    items: list[ContractMap]


class MessagePayload(BaseModel):
    """Base type for payloads that all the payload contracts must adhere to"""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> str:
        """Serialize event to JSON for pub/sub compatibility."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "MessagePayload":
        """Deserialize event from JSON for pub/sub compatibility."""
        return cls.model_validate_json(json_str)


class Error(BaseModel):
    session_id: str
    content: str


class MQTTClient(ABC):
    """Base class for all mqtt client in the event-driven system."""

    component_name: str = "template"

    def __init__(self, client: Client, name: str | None = None):
        self.name = name if name else self.default_name
        self.client = client
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

    async def _send_error_response(self, session_id: str, error: Any) -> None:
        """Send error response event."""
        await self.publish(
            self.error_topic,
            Error(
                session_id=session_id,
                content=self._format_error(error),
            ),
        )

    @abstractmethod
    async def initialize_subscriptions(self):
        """Initializes registered message contract topics"""
        ...

    @abstractmethod
    async def handle_message(self, msg):
        """Handles all incoming messages:"""
        ...

    async def start(self):
        """Starts a 'serve forever' function"""
        await self.initialize_subscriptions()
        # waiting for messages is running forever: https://aiomqtt.bo3hm.com/subscribing-to-a-topic.html#listening-without-blocking
        async for msg in self.client.messages:
            await self.handle_message(msg)


async def start_cli_client(
    client: aiomqtt.Client,
    mqtt_client: type[MQTTClient],
    reconnect_interval: int = 5,
    **client_kwargs,
):
    """Start the aiomqtt client with reconnection"""
    while True:
        try:
            async with client:
                instance = mqtt_client(client, **client_kwargs)
                await instance.start()
        except aiomqtt.MqttError:
            logger.info(
                f"Connection lost, reconnecting in {reconnect_interval} seconds"
            )
            await asyncio.sleep(reconnect_interval)

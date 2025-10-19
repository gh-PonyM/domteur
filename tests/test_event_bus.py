"""Tests for the event bus."""

import asyncio

import pytest

from domteur.events import Event, EventType, InMemoryEventBus, Topics


@pytest.mark.asyncio
async def test_publish_and_subscribe(event_bus):
    """Test basic publish/subscribe functionality."""
    received_events = []

    async def handler(event: Event):
        received_events.append(event)

    # Subscribe to topic
    await event_bus.subscribe(Topics.USER_INPUT, handler)

    # Publish event
    test_event = Event(
        event_type=EventType.USER_INPUT, payload={"content": "test message"}
    )
    await event_bus.publish(Topics.USER_INPUT, test_event)

    # Give event loop time to process
    await asyncio.sleep(0.1)

    # Verify event was received
    assert len(received_events) == 1
    assert received_events[0].event_type == EventType.USER_INPUT
    assert received_events[0].payload["content"] == "test message"


@pytest.mark.asyncio
async def test_multiple_subscribers(event_bus):
    """Test multiple subscribers to same topic."""
    handler1_events = []
    handler2_events = []

    async def handler1(event: Event):
        handler1_events.append(event)

    async def handler2(event: Event):
        handler2_events.append(event)

    # Subscribe both handlers
    await event_bus.subscribe(Topics.LLM_PROCESSING, handler1)
    await event_bus.subscribe(Topics.LLM_PROCESSING, handler2)

    # Publish event
    test_event = Event(event_type=EventType.LLM_RESPONSE)
    await event_bus.publish(Topics.LLM_PROCESSING, test_event)

    # Give event loop time to process
    await asyncio.sleep(0.1)

    # Both handlers should receive the event
    assert len(handler1_events) == 1
    assert len(handler2_events) == 1


@pytest.mark.asyncio
async def test_topic_isolation(event_bus):
    """Test that events are only delivered to correct topic subscribers."""
    topic1_events = []
    topic2_events = []

    async def handler1(event: Event):
        topic1_events.append(event)

    async def handler2(event: Event):
        topic2_events.append(event)

    # Subscribe to different topics
    await event_bus.subscribe("topic1", handler1)
    await event_bus.subscribe("topic2", handler2)

    # Publish to topic1 only
    test_event = Event(event_type=EventType.USER_INPUT)
    await event_bus.publish("topic1", test_event)

    # Give event loop time to process
    await asyncio.sleep(0.1)

    # Only handler1 should receive the event
    assert len(topic1_events) == 1
    assert len(topic2_events) == 0


@pytest.mark.asyncio
async def test_event_bus_start_stop():
    """Test event bus lifecycle."""
    bus = InMemoryEventBus()

    assert not bus._running

    await bus.start()
    assert bus._running

    await bus.stop()
    assert not bus._running

"""Tests for event models."""

from domteur.events import Event, EventType


def test_event_creation():
    """Test creating an event."""
    event = Event(event_type=EventType.USER_INPUT, payload={"content": "Hello world"})

    assert event.event_type == EventType.USER_INPUT
    assert event.payload["content"] == "Hello world"
    assert event.topic == "default"


def test_event_json_serialization():
    """Test event JSON serialization and deserialization."""
    event = Event(
        event_type=EventType.LLM_RESPONSE,
        payload={"response": "Hi there!"},
        topic="test",
    )

    json_str = event.to_json()
    restored_event = Event.from_json(json_str)

    assert restored_event.event_type == event.event_type
    assert restored_event.payload == event.payload
    assert restored_event.topic == event.topic


def test_event_default_values():
    """Test event default values are set correctly."""
    event = Event(event_type=EventType.SYSTEM_SHUTDOWN)

    assert event.event_type == EventType.SYSTEM_SHUTDOWN
    assert event.payload == {}
    assert event.topic == "default"
    assert event.session_id is not None
    assert event.timestamp is not None

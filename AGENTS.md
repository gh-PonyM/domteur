# Agent Guidelines for domteur

## POC Goals & Requirements
Event-driven AI assistant with:
- **LLM Integration**: Local (Ollama) + cloud (OpenRouter) providers
- **Interactive Chat**: CLI REPL for real-time conversation
- **Text-to-Speech**: Audio output for responses
- **Event Architecture**: Fully decoupled components with JSON messaging
- **Async Operations**: All components and functions use async/await
- **Local Persistence**: SQLite for chat history and state

## Component Names
- `repl`: CLI REPL for user input
- `llm_processor`: Handles LLM requests/responses (uses LangChain/LangGraph)
- `text_to_speech_engine`: TTS message sink
- `event_dispatcher`: Central message routing
- `persistence_manager`: SQLite database operations
- `config_manager`: Settings and provider configuration

## Event Architecture
```
┌─────────────────┐    JSON Messages    ┌──────────────────┐
│ repl            │ ──────────────────> │ mqtt broker      │
│                 │                     │                  │
└─────────────────┘                     └──────────────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   │                             │                             │
                   v                             v                             v
        ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
        │ llm_processor   │           │ text_to_speech_ │           │ persistence_    │
        │ (LangChain)     │           │ engine          │           │ manager         │
        └─────────────────┘           └─────────────────┘           └─────────────────┘
```

## Message Protocol (JSON)

The underlying library aiomqtt encodes the payload in binary or it can be None. Message contracts are used by topic.

## Configuration Structure

The example structure can be found in `config.example.yml`.

## Build/Test/Lint Commands
- Run tests: `uv run pytest` or `pytest tests/`
- Run single test: `uv run pytest tests/test_cli.py::test_help`
- Lint/format: `./lint.sh` or `ruff format && ruff check --fix`
- Type check: `uv run pyrefly check`

## Code Style
- Use ruff for linting (line length: 88 chars)
- Import style: `from typing import`, run lint command after each change
- Type hints: Use modern syntax (`str | None` not `Optional[str]`)
- Naming: snake_case for functions/variables, PascalCase for classes
- Error handling: Use raise typer.Exit() for CLI errors, ValidationError for pydantic
- Config: Use pydantic BaseSettings, YAML format preferred
- CLI: Use typer with pretty_exceptions_show_locals=False
- Secrets: Use pydantic SecretStr, exclude from serialization
- Private attrs: Use PrivateAttr() for non-serialized fields
- **Async**: All functions must be async, use async-compatible libraries

## Project Structure
- Main code in `domteur/` package
- Tests in `tests/` directory
- Config handling via pydantic-settings with YAML files
- CLI entry point: `domteur.main:cli`
- Event system spawns all components

## MQTT Message iterator

The aiomqtt message iterator per client is implemented as follows:

```python
class MessagesIterator:
    """Dynamic view of the client's message queue."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def __aiter__(self) -> AsyncIterator[Message]:
        return self

    async def __anext__(self) -> Message:
        # Wait until we either (1) receive a message or (2) disconnect
        task = self._client._loop.create_task(self._client._queue.get())  # noqa: SLF001
        try:
            done, _ = await asyncio.wait(
                (task, self._client._disconnected),  # noqa: SLF001
                return_when=asyncio.FIRST_COMPLETED,
            )
        # If the asyncio.wait is cancelled, we must also cancel the queue task
        except asyncio.CancelledError:
            task.cancel()
            raise
        # When we receive a message, return it
        if task in done:
            return task.result()
        # If we disconnect from the broker, stop the generator with an exception
        task.cancel()
        msg = "Disconnected during message iteration"
        raise MqttError(msg)
```
The `async for msg in client.messages` loops over the `MessagesIterator`

## Enhanced Piper TTS Component - Priority-Based Streaming System

### Core Requirements

**Priority-Based Message Processing:**
- Priority levels: CRITICAL(0), HIGH(1), NORMAL(2), LOW(3)
- Priority encoded in message payload, not separate topics
- Higher priority messages interrupt and clear lower priority processing

**Custom MQTT Queue Implementation:**
- Multi-priority queue system replacing standard aiomqtt message iteration
- Dynamic message reordering based on priority
- Queue inspection and management capabilities
- Thread-safe operations for concurrent access

**Streaming Text Processing:**
- Handle continuous LLM token streams with priority awareness
- Smart sentence boundary detection with fallback mechanisms
- Multi-level internal queues (tokens → sentences → audio)
- Priority-based queue clearing on interruption

**State Management:**
- States: IDLE, STREAMING, SYNTHESIZING, PLAYING, MUTED
- Muted state processes all messages but suppresses audio output
- No pause/resume - only stop/clear/play based on priority

### Message Priority System

**Priority Levels and Behavior:**
```
CRITICAL (0): System alerts, errors - immediate interruption
HIGH (1): User commands, urgent responses - interrupt current playback
NORMAL (2): Standard LLM responses - queue normally
LOW (3): Background info, notifications - queue after higher priorities
```

**Message Payload Structure:**
```json
{
  "content": "text content",
  "priority": 1,
  "session_id": "uuid",
  "timestamp": 1234567890.123,
  "message_type": "stream_chunk|complete|control"
}
```

**Priority Interrupt Logic:**
- **Higher Priority Arrives**: Stop current playback + clear all queues + play new message
- **Same Priority**: Queue in FIFO order
- **Lower Priority**: Queue behind existing messages
- **Muted State**: Process all priorities but suppress audio output

### Architecture Components

**1. PriorityMQTTQueue**: Custom message queue with priority ordering
- Multi-priority queues with separate queues for each priority level
- Dynamic message reordering and priority-based insertion
- Operations: enqueue, dequeue, peek_next, clear_lower_priority, clear_all

**2. StreamingTextBuffer**: Token accumulation with priority awareness
- Handle streaming text tokens from LLM with sentence boundary detection
- Priority-aware processing with interrupt capability
- Fallback mechanisms: timeout trigger, buffer size limits, manual boundaries

**3. TTSMessageQueue**: Three-tier processing pipeline
- Level 1: Raw token buffer (fast accumulation)
- Level 2: Sentence queue (synthesis ready)
- Level 3: Audio queue (playback ready)

**4. AudioPlaybackManager**: Thread-safe audio with interrupt capability
- Hybrid async/sync architecture with ThreadPoolExecutor
- Immediate stop capability using threading primitives
- Priority-based playback control

**5. PriorityInterruptHandler**: Manages stop/clear/play logic
- State transitions on priority interruption
- Queue clearing strategy across all processing levels
- Session state reset and recovery

### Control Signals

**Core Signals:**
- `STOP`: Immediate halt and queue clearing
- `MUTE`/`UNMUTE`: Toggle audio output while maintaining processing
- `CLEAR_QUEUE`: Manual queue clearing
- Priority-based automatic interruption

**State Transitions on Interrupt:**
```
PLAYING + HIGH_PRIORITY → STOP + CLEAR + PLAY_NEW
STREAMING + HIGH_PRIORITY → STOP + CLEAR + STREAM_NEW  
SYNTHESIZING + HIGH_PRIORITY → STOP + CLEAR + SYNTHESIZE_NEW
MUTED + ANY_PRIORITY → PROCESS_SILENTLY
```

### Streaming Text Challenges

Supported Languages:

- en
- de
- fr

**Token Fragmentation Issues:**
- Tokens may split mid-word or mid-sentence
- Punctuation might arrive separately from words
- Incomplete sentences create poor synthesis quality
- Buffer management for partial content

**Sentence Boundary Detection:**
- Multi-strategy approach: pattern-based, context-aware, timeout-based
- Handle abbreviations, numbers, URLs across different languages
- Fallback mechanisms for edge cases and incomplete streams

**Queue Management Complexity:**
- Memory management for large streaming queues
- Priority handling with concurrent streams
- Graceful degradation under load
- Latency balance between responsiveness and quality

### Implementation Priority

1. **Custom MQTT priority queue system**
2. **Priority-based interrupt handling**
3. **Streaming token processing with priority awareness**
4. **Multi-level queue management**
5. **State management and muted mode**
6. **Robust error handling and recovery**

For point 1, evaluate if a custom priority queue should be used according to the docs, here is an example:
```python
import asyncio
import aiomqtt


class CustomPriorityQueue(asyncio.PriorityQueue):
    def _put(self, item):
        priority = 2
        if item.topic.matches("humidity/#"):  # Assign priority
            priority = 1
        super()._put((priority, item))

    def _get(self):
        return super()._get()[1]


async def main():
    async with aiomqtt.Client(
        "test.mosquitto.org", queue_type=CustomPriorityQueue
    ) as client:
        await client.subscribe("temperature/#")
        await client.subscribe("humidity/#")
        async for message in client.messages:
            print(message.payload)


asyncio.run(main())
```

### Key Behaviors

- Long text playback interrupted by higher priority messages
- Complete queue clearing on priority interruption
- Immediate processing of high-priority messages
- Graceful handling of streaming session interruptions
- Muted state maintains processing without audio output

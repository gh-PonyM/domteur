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
- `tts`: TTS message sink
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
        │ llm_processor   │           │ tts             │           │ persistence_    │
        │ (LangChain)     │           │                 │           │ manager         │
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

This is already implemented.

## 🧭 Topic Naming Conventions

### Recommended Topic Structure

```
app/<app_name>/<component>/<instance>/<domain>/<event>
```

### Example Topics

| Component | Direction | Topic Example | Description |
|------------|------------|----------------|--------------|
| `repl` | pub | `app/assistant/repl/001/output/user_message` | Publishes user input |
| `llm_processor` | sub | `app/assistant/repl/+/output/user_message` | Consumes user input |
| `llm_processor` | pub | `app/assistant/llm_processor/001/output/response` | Publishes LLM output |
| `tts` | sub | `app/assistant/llm_processor/+/output/response` | Converts text to speech |
| `persistence_manager` | sub | `app/assistant/+/+/output/#` | Logs all output events |
| `config_manager` | pub/sub | `app/assistant/config/+/control/#` | Configuration updates |

### Naming Principles
- Use **lowercase** and **snake_case** only.
- Maintain **consistent hierarchy**: `namespace → component → domain → event`.
- Avoid embedding dynamic or sensitive data in topic names.

## 💬 Message Structure

All messages share a **common JSON envelope** for consistency and traceability.

### Envelope Schema

```json
{
  "event_type": "llm_response",
  "timestamp": "2025-10-25T12:00:00Z",
  "source": "llm_processor",
  "correlation_id": "c8b12f47-3f45-4e7a-9b16-93cfd51c73b0",
  "payload": {
    "text": "Sure! Here's an example...",
    "tokens_used": 342
  }
}
```

### Envelope Fields

| Field            | Description                                   |
| ---------------- | --------------------------------------------- |
| `event_type`     | Logical name of the event                     |
| `timestamp`      | ISO 8601 timestamp for event emission         |
| `source`         | Component name that produced the event        |
| `correlation_id` | Unique ID for tracing request–response chains |
| `payload`        | Schema-specific message data                  |

## 📦 Contract Definitions

Define message contracts as **Pydantic** or **dataclass** models for validation and typing.

### Example Models

```python
class Envelope(BaseModel):
    event_type: str
    timestamp: datetime
    source: str
    correlation_id: str
    payload: BaseModel

class Conversation(BaseModel):
    user_id: str
    message: str
    context: Optional[dict]

class LLMResponse(BaseModel):
    text: str
    tokens_used: int
```

## ⚙️ Event Routing Design

### Option A: Central Dispatcher

* Acts as an internal message bus or monitor.
* Pros: Simplified debugging and tracing.
* Cons: Creates coupling if it must know all topics.

### Option B: Direct Pub/Sub (Recommended)

* Each component subscribes directly to relevant topics.
* Fully decoupled and scalable.
* Dispatcher remains **optional** (for logging or dynamic routing).

## 📈 Observability and Control Topics

| Purpose             | Topic Example                                     | Payload Example                         |
| ------------------- | ------------------------------------------------- | --------------------------------------- |
| **Heartbeat**       | `app/assistant/<component>/<id>/status/heartbeat` | `{ "alive": true, "timestamp": "..." }` |
| **Error Reporting** | `app/assistant/<component>/<id>/status/error`     | `{ "message": "..." }`                  |
| **Config Reload**   | `app/assistant/config/<component>/control/reload` | `{ "action": "reload" }`                |

## 🧠 Example Full Flow

1. **User Input**

   ```
   Topic: app/assistant/repl/001/output/user_message
   Payload: { "event_type": "user_message", "payload": { "message": "What's the weather?" } }
   ```

2. **LLM Processing**

   ```
   Topic: app/assistant/llm_processor/001/output/response
   Payload: { "event_type": "llm_response", "payload": { "text": "It’s 20°C and sunny." } }
   ```

3. **Text-to-Speech**

   ```
   Topic: app/assistant/tts/001/output/audio_chunk
   Payload: { "event_type": "tts_stream_chunk", "payload": { "base64_audio": "..." } }
   ```

4. **Persistence Logging**

   ```
   Subscription: app/assistant/+/+/output/#
   ```

## ✅ Design Principles Summary

| Principle                        | Description                                         |
| -------------------------------- | --------------------------------------------------- |
| **Structured topic hierarchy**   | Namespace → Component → Domain → Event              |
| **Common envelope format**       | Metadata + typed payload                            |
| **Request–response correlation** | Use consistent `correlation_id`                     |
| **Component autonomy**           | Each service subscribes only to needed topics       |
| **Built-in observability**       | Heartbeat and error topics                          |
| **Versioning support**           | Prefix topic root with version (e.g., `v1/`)        |
| **Scalability first**            | Stateless, async components communicating over MQTT |

---

## 🧩 Next Steps

1. **Refactor topic names** across all components to follow the new pattern.
2. **Implement envelope schema** and correlation IDs in all message contracts.
3. **Adopt Pydantic-based validation** for all inbound/outbound messages.
4. **Add observability topics** (heartbeat/error/config).
5. **Document topic contracts** per component in `/docs/contracts.md`.
6. **(Optional)** Add dynamic routing or filtering in `event_dispatcher` for orchestration.

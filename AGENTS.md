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
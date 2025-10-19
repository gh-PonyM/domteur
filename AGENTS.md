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
│ repl            │ ──────────────────> │ event_dispatcher │
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
```json
{
    "event_type": "user_input | llm_response | tts_request | persist_message",
    "timestamp": "2025-10-19T...",
    "session_id": "uuid",
    "payload": {
        "content": "message content",
        "metadata": {"user": "...", "model": "..."}
    }
}
```

## Configuration Structure
```yaml
llm_providers:
  - type: "ollama"
    base_url: "http://localhost:11434"
    model: "llama2"
  - type: "openrouter"
    api_key: "sk-..."
    model: "anthropic/claude-3-haiku"

database:
  type: "sqlite"
  path: "./domteur.db"

tts:
  engine: "system"
```

## Build/Test/Lint Commands
- Run tests: `uv run pytest` or `pytest tests/`
- Run single test: `uv run pytest tests/test_cli.py::test_help`
- Lint/format: `./lint.sh` or `ruff format && ruff check --fix`
- Build: `uv build` (uses hatchling)

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
- JSON message protocol for inter-component communication
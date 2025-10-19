# Agent Guidelines for domteur

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

## Project Structure
- Main code in `domteur/` package
- Tests in `tests/` directory
- Config handling via pydantic-settings with YAML files
- CLI entry point: `domteur.main:cli`
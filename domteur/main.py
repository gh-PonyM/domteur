import sys
from os import environ
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from pydantic import ValidationError
from rich import print

from domteur import __version__
from domteur.cli.chat import chat_cli
from domteur.cli.llm import llm_cli
from domteur.config import APP_CFG, CONFIG_FN, ENV_CONFIG_KEY, Settings

cli = typer.Typer(pretty_exceptions_show_locals=False)

# Register chat commands
cli.add_typer(chat_cli, name="chat")
cli.add_typer(llm_cli, name="llm")


log_levels = {0: "ERROR", 1: "WARNING", 2: "INFO", 3: "DEBUG"}


def configure_logging(verbose: int):
    """Configure loguru logging based on verbosity level."""
    # Remove default handler
    logger.remove()

    logger.add(
        sys.stderr,
        level=log_levels[verbose],
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    logger.info(f"Logging configured at level: {log_levels[verbose]}")


def version_info(value: bool):
    if not value:
        return
    typer.echo(__version__)
    raise typer.Exit()


def error(msg: str, exit_code: int = 1):
    print(f"[bold red]ERROR: {msg}")
    raise typer.Exit(exit_code)


def ensure_config(ctx, option, cfg_file):
    """Ensure the config path and directory and inject the config into the context"""
    path = Path(cfg_file) if not isinstance(cfg_file, Path) else cfg_file
    env_overwrite = environ.get(ENV_CONFIG_KEY)
    path = Path(env_overwrite, CONFIG_FN) if env_overwrite else path

    # TODO: if path is default path and not exists, raise an error

    if not path.parent.is_dir():
        if not cfg_file == APP_CFG:
            error(f"{path.parent} does not exist")
        print(f"Creating app config folder {path.parent}")
        path.parent.mkdir()
    if not path.is_file():
        print(f"Creating empty config {path}")
        cfg = Settings()
        cfg.prompt_initial_config()
        path.write_text(cfg.dump())
    try:
        cfg = Settings.from_file(path)
    except ValidationError:
        raise

    # Write the new version if cli was updated
    if cfg.version != __version__:
        cfg.save()

    # Use this option to avoid having a global cfg object
    ctx.meta["cfg"] = cfg
    return path


@cli.callback(invoke_without_command=False)
def main(
    ctx: typer.Context,
    version: bool | None = typer.Option(None, "--version", callback=version_info),
    cfg_file: Path = typer.Option(APP_CFG, callback=ensure_config),
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, max=3, min=0)
    ] = 0,
):
    """domteur command line interface made with typer"""
    configure_logging(verbose)
    if ctx.invoked_subcommand is None:
        pass


@cli.command()
def open_config(ctx: typer.Context):
    """Open the place where the config is stored"""
    cfg = ctx.meta["cfg"]
    print(f"Opening config {cfg.file_path}")
    typer.launch(str(cfg.file_path.resolve()), locate=True)


@cli.command()
def show_config(ctx: typer.Context):
    """Open the place where the config is stored"""
    cfg = ctx.meta["cfg"]
    from rich import print

    print(cfg.dump())


@cli.command()
def list_components():
    """List the registry of components"""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    from domteur.components.base import get_registry_items

    items = sorted(get_registry_items(), key=lambda x: x.topic)

    table = Table(title="Component Registry")
    table.add_column("Component", style="cyan")
    table.add_column("Topic")
    table.add_column("Method", style="white")
    table.add_column("Contract", style="blue")

    for item in items:
        parts = item.topic.split("/")
        colorized_topic = Text()
        type_text = Text(
            item.type, style="bright_green" if item.type == "pub" else "bright_magenta"
        )
        if len(parts) >= 5:
            colorized_topic.append(type_text)
            colorized_topic.append(" | ", style="dim white")
            colorized_topic.append(parts[0], style="dim white")
            colorized_topic.append("/", style="dim white")
            colorized_topic.append(parts[1], style="dim white")
            colorized_topic.append("/", style="dim white")
            colorized_topic.append(parts[2], style="red")
            colorized_topic.append("/", style="dim white")
            colorized_topic.append(parts[3], style="bright_yellow")
            colorized_topic.append("/", style="dim white")
            colorized_topic.append(parts[4], style="yellow")
            if len(parts) > 5:
                colorized_topic.append("/", style="dim white")
                colorized_topic.append("/".join(parts[5:]), style="bright_green")
        else:
            colorized_topic.append(item.topic, style="white")

        table.add_row(
            item.component,
            colorized_topic,
            item.method_name,
            item.contract.__name__,
        )

    console = Console()
    console.print(table)

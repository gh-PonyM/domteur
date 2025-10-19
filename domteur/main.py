import logging
from os import environ
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich import print

from domteur import __version__
from domteur.config import APP_CFG, CONFIG_FN, ENV_CONFIG_KEY, Settings

cli = typer.Typer(pretty_exceptions_show_locals=False)


log_levels = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}


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
    logging.basicConfig(level=log_levels[verbose])
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


@cli.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def extra(ctx: typer.Context):
    for extra_arg in ctx.args:
        print(f"Got extra arg: {extra_arg}")

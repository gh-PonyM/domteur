"""Chat commands for domteur CLI."""

import typer
from loguru import logger

from domteur.cli.base import sync_run_client
from domteur.components.repl.terminal import LLMTerminalChat
from domteur.components.tts.piper import PiperTTS
from domteur.config import Settings


import piper
import sounddevice as sd

# Create standalone chat CLI
chat_cli = typer.Typer(name="chat", help="Interactive chat with LLM")


@chat_cli.command()
def repl(ctx: typer.Context, broker_host: str | None = None):
    """Output given text as speech"""
    cfg = ctx.meta["cfg"]
    # from prompt_toolkit import PromptSession
    # from prompt_toolkit.patch_stdout import patch_stdout
    typer.secho(f"Starting with {cfg.broker_host}:{cfg.broker_port}")
    if broker_host:
        cfg.broker_host = broker_host
    sync_run_client(cfg, LLMTerminalChat)



@chat_cli.command()
def piper(ctx: typer.Context):

    cfg: Settings = ctx.meta["cfg"]
    sync_run_client(cfg, PiperTTS)



@chat_cli.command()
def models(ctx: typer.Context):
    """List available LLM models."""
    cfg = ctx.meta["cfg"]
    if hasattr(cfg, "llm_providers") and cfg.llm_providers:
        print("📋 Available LLM models:")
        for i, provider in enumerate(cfg.llm_providers, 1):
            print(f"  {i}. {provider.type}: {provider.model}")
    else:
        print("❌ No LLM providers configured")

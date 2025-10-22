"""Chat commands for domteur CLI."""

import typer

from domteur.cli.base import sync_run_client
from domteur.components.llm_processor.client import LLMProcessor
from domteur.config import Settings

# Create standalone chat CLI
llm_cli = typer.Typer(name="chat", help="LLM Service Client")


@llm_cli.command()
def start(ctx: typer.Context):
    """Output given text as speech"""

    cfg: Settings = ctx.meta["cfg"]
    typer.secho(f"Starting with {cfg.broker_host}:{cfg.broker_port}")
    sync_run_client(cfg, LLMProcessor)


@llm_cli.command()
def models(ctx: typer.Context):
    """List available LLM models."""
    cfg = ctx.meta["cfg"]
    if hasattr(cfg, "llm_providers") and cfg.llm_providers:
        print("📋 Available LLM models:")
        for i, provider in enumerate(cfg.llm_providers, 1):
            print(f"  {i}. {provider.type}: {provider.model}")
    else:
        print("❌ No LLM providers configured")

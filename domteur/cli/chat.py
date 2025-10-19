"""Chat commands for domteur CLI."""

import asyncio

import typer

# Create standalone chat CLI
chat_cli = typer.Typer(name="chat", help="Interactive chat with LLM")


def error(msg: str, exit_code: int = 1):
    """Error handling function."""
    from rich import print

    print(f"[bold red]ERROR: {msg}")
    raise typer.Exit(exit_code)


@chat_cli.command()
def start(ctx: typer.Context):
    """Start interactive chat with LLM using event-driven architecture."""
    try:
        asyncio.run(_run_chat(ctx))
    except KeyboardInterrupt:
        print("\n👋 Chat interrupted by user")
    except Exception as e:
        error(f"Failed to start chat: {e}")


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


async def _run_chat(ctx: typer.Context):
    """Run the chat system with all components."""
    from domteur.components.llm_processor import create_llm_processor
    from domteur.components.repl import create_repl_component
    from domteur.events import get_event_bus

    cfg = ctx.meta["cfg"]
    event_bus = get_event_bus()

    try:
        # Start event bus
        await event_bus.start()
        print("🌟 Starting domteur chat system...")

        # Create and start components
        llm_processor = await create_llm_processor(cfg)
        repl_component = await create_repl_component()

        print(
            f"✅ LLM processor ready with models: {llm_processor.get_available_models()}"
        )

        # Start the interactive REPL
        await repl_component.start_repl()

    except Exception as e:
        print(f"❌ Error running chat: {e}")
        raise
    finally:
        # Cleanup components
        try:
            if "llm_processor" in locals():
                await llm_processor.stop()
            if "repl_component" in locals():
                await repl_component.stop()
            await event_bus.stop()
        except Exception as e:
            print(f"Error during cleanup: {e}")

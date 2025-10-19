"""Chat commands for domteur CLI."""

import asyncio

import typer

# Create standalone chat CLI
chat_cli = typer.Typer(name="chat", help="Interactive chat with LLM")


@chat_cli.command()
def start(ctx: typer.Context):
    """Start interactive chat with LLM using event-driven architecture."""
    import signal

    STOP = asyncio.Event()

    def ask_exit(*args):
        STOP.set()

    async def main():
        from domteur.components.llm_processor import create_llm_processor
        from domteur.events import Event, EventType, Topics, get_event_bus

        cfg = ctx.meta["cfg"]
        event_bus = get_event_bus()

        # Response handler to display LLM responses
        def handle_response(event):
            if event.event_type == EventType.LLM_RESPONSE:
                response = event.payload.get("content", "")
                print(f"🤖 Assistant: {response}")

        try:
            # Start event bus
            await event_bus.start()
            print("🌟 Starting domteur chat system...")

            # Subscribe to LLM responses
            await event_bus.subscribe(Topics.LLM_PROCESSING, handle_response)

            # Create and start LLM processor
            llm_processor = await create_llm_processor(cfg, event_bus=event_bus)
            print(
                f"✅ LLM processor ready with models: {llm_processor.get_available_models()}"
            )

            # Send a test message
            test_event = Event(
                event_type=EventType.USER_INPUT,
                session_id="test-session",
                payload={
                    "content": "Hello! Can you tell me a short joke?",
                    "user": "human",
                },
            )

            print("🧪 Sending test message: 'Hello! Can you tell me a short joke?'")
            await event_bus.publish(Topics.USER_INPUT, test_event)

            # Wait for stop signal (or timeout after 30 seconds for testing)
            try:
                await asyncio.wait_for(STOP.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                print("\n⏰ Test timeout reached")

        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            print("\n🧹 Cleaning up...")
            try:
                if "llm_processor" in locals():
                    await llm_processor.stop()
                await event_bus.stop()
            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, ask_exit)
    loop.add_signal_handler(signal.SIGTERM, ask_exit)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n👋 Chat interrupted")
    finally:
        print("💫 Goodbye!")


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

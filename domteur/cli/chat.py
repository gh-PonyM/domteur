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

    # Create STOP event inside async context
    STOP = asyncio.Event()

    def ask_exit(*args):
        STOP.set()

    # Set up signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, ask_exit)
    loop.add_signal_handler(signal.SIGTERM, ask_exit)

    async def main():
        import uuid

        from domteur.components.llm_processor import create_llm_processor
        from domteur.events import Event, EventType, Topics, get_event_bus

        cfg = ctx.meta["cfg"]
        event_bus = get_event_bus()
        session_id = str(uuid.uuid4())
        conversation_history = []

        # Response handler to display LLM responses and maintain history
        def handle_response(event):
            if event.event_type == EventType.LLM_RESPONSE:
                response = event.payload.get("content", "")
                print(f"🤖 Assistant: {response}")

                # Add response to conversation history
                conversation_history.append({"role": "assistant", "content": response})

        async def get_user_input():
            """Get user input asynchronously."""
            return typer.prompt("💬 You: ")

        # Start event bus
        await event_bus.start()
        print("🌟 Starting domteur chat system...")
        print("Type 'quit', 'exit', or press Ctrl+C to stop.\n")

        # Subscribe to LLM responses
        await event_bus.subscribe(Topics.LLM_PROCESSING, handle_response)

        # Create and start LLM processor
        llm_processor = await create_llm_processor(cfg, event_bus=event_bus)
        print(
            f"✅ LLM processor ready with models: {llm_processor.get_available_models()}"
        )

        # Continuous conversation loop
        try:
            while not STOP.is_set():
                try:
                    # Get user input
                    user_input = await get_user_input()

                    # Check for quit commands
                    if user_input.lower().strip() in ["quit", "exit", "q", "bye"]:
                        print("👋 Goodbye!")
                        break

                    # Skip empty inputs
                    if not user_input.strip():
                        continue

                    # Add user message to history
                    conversation_history.append(
                        {"role": "user", "content": user_input.strip()}
                    )

                    # Create event with conversation history
                    chat_event = Event(
                        event_type=EventType.USER_INPUT,
                        session_id=session_id,
                        payload={
                            "content": user_input.strip(),
                            "user": "human",
                            "conversation_history": conversation_history.copy(),
                            "message_count": len(conversation_history),
                        },
                    )

                    await event_bus.publish(Topics.USER_INPUT, chat_event)

                    # Small delay to allow response processing
                    await asyncio.sleep(0.1)

                except asyncio.TimeoutError:
                    # Timeout is expected, continue loop
                    continue
        except KeyboardInterrupt:
            print("\n👋 Chat interrupted by user")
            raise
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

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # loop = asyncio.get_event_loop()
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

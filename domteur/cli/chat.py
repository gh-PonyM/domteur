"""Chat commands for domteur CLI."""

import typer

from domteur.cli.base import sync_run_client
from domteur.components.repl.terminal import LLMTerminalChat

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

    # session = PromptSession()
    #
    # async def main():
    #     async with aiomqtt.Client(cfg.broker_url) as client:
    #         while True:
    #             with patch_stdout():
    #                 user_input = await session.prompt_async("💬 You: ")
    #                 print(user_input)

    # asyncio.run(main())


# @chat_cli.command()
# def start(ctx: typer.Context):
#     """Start interactive chat with LLM using event-driven architecture."""
#
#     async def main():
#         from domteur.components.llm_processor import create_llm_processor
#         from domteur.components.text_to_speech_engine import TextToSpeechEngine, PiperTTSConfig
#         from domteur.events import Event, EventType, Topics, get_event_bus
#
#         # Create stop event and signal handler
#         STOP = asyncio.Event()
#
#         def ask_exit(*args):
#             print("\n🛑 Shutting down...")
#             STOP.set()
#
#         # Set up signal handlers
#         loop = asyncio.get_running_loop()
#         for sig in (signal.SIGTERM, signal.SIGINT):
#             loop.add_signal_handler(sig, ask_exit)
#
#         cfg = ctx.meta["cfg"]
#         event_bus = get_event_bus()
#         session_id = str(uuid.uuid4())
#         conversation_history = []
#
#         # Create prompt session
#         session = PromptSession()
#
#         # Response handler to display LLM responses and maintain history
#         async def handle_response(event):
#             if event.event_type == EventType.LLM_RESPONSE:
#                 response = event.payload.get("content", "")
#
#                 # Use patch_stdout to ensure response is visible during input
#                 with patch_stdout():
#                     print(f"\n🤖 Assistant: {response}\n")
#
#                 # Add response to conversation history
#                 conversation_history.append({"role": "assistant", "content": response})
#
#                 # Trigger TTS for the response
#                 # tts_event = Event(
#                 #     event_type=EventType.TTS_REQUEST,
#                 #     session_id=session_id,
#                 #     payload={"text": response}
#                 # )
#                 # asyncio.create_task(event_bus.publish(Topics.TTS_OUTPUT, tts_event))
#
#         try:
#             # Start event bus
#             await event_bus.start()
#             print("🌟 Starting domteur chat system...")
#             print("Type 'quit', 'exit', or press Ctrl+C to stop.\n")
#
#             # Subscribe to LLM responses
#             await event_bus.subscribe(Topics.LLM_PROCESSING, handle_response)
#
#             # Create and start LLM processor
#             llm_processor = await create_llm_processor(cfg, event_bus=event_bus)
#             print(
#                 f"✅ LLM processor ready with models: {llm_processor.get_available_models()}\n"
#             )
#
#             # Create and start TTS engine
#             tts_config = PiperTTSConfig(**cfg.tts.model_dump())
#             tts_engine = TextToSpeechEngine(tts_config, event_bus=event_bus)
#             await tts_engine.start()
#             print("🔊 TTS engine ready\n")
#
#             # Continuous conversation loop
#             while not STOP.is_set():
#                 try:
#                     # Get user input with prompt_toolkit
#                     with patch_stdout():
#                         user_input = await session.prompt_async("💬 You: ")
#
#                     # Check for quit commands
#                     if user_input.lower().strip() in ["quit", "exit", "q", "bye"]:
#                         print("👋 Goodbye!")
#                         break
#
#                     # Skip empty inputs
#                     if not user_input.strip():
#                         continue
#
#                     # Add user message to history
#                     conversation_history.append(
#                         {"role": "user", "content": user_input.strip()}
#                     )
#
#                     # Create event with conversation history
#                     chat_event = Event(
#                         event_type=EventType.USER_INPUT,
#                         session_id=session_id,
#                         payload={
#                             "content": user_input.strip(),
#                             "user": "human",
#                             "conversation_history": conversation_history.copy(),
#                             "message_count": len(conversation_history),
#                         },
#                     )
#
#                     await event_bus.publish(Topics.USER_INPUT, chat_event)
#
#                     # Wait a bit for response to arrive
#                     await asyncio.sleep(0.5)
#
#                 except KeyboardInterrupt:
#                     print("\n👋 Chat interrupted by user")
#                     break
#                 except (EOFError, asyncio.CancelledError):
#                     print("\n👋 Chat session ended")
#                     break
#                 except Exception as e:
#                     print(f"\n❌ Error during chat: {e}")
#                     continue
#
#         except Exception as e:
#             print(f"❌ Fatal error: {e}")
#         finally:
#             print("\n🧹 Cleaning up...")
#             try:
#                 if "tts_engine" in locals():
#                     await tts_engine.stop()
#                 if "llm_processor" in locals():
#                     await llm_processor.stop()
#                 await event_bus.stop()
#                 print("✅ Cleanup completed")
#             except Exception as e:
#                 print(f"⚠️ Error during cleanup: {e}")
#
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\n👋 Chat interrupted")
#     except Exception as e:
#         print(f"❌ Error: {e}")
#     finally:
#         pass


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

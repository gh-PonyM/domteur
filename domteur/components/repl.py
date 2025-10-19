"""Interactive REPL component for chat input."""

import asyncio
import logging
import uuid

from domteur.components.base import Component
from domteur.events import Event, EventType, Topics

logger = logging.getLogger(__name__)


class InteractiveLLMInput(Component):
    """CLI REPL component for interactive chat with LLM."""

    def __init__(self, session_id: str | None = None, **kwargs):
        super().__init__("interactive_llm_input", **kwargs)
        self.session_id = session_id or str(uuid.uuid4())
        self._running_repl = False

    async def _register_handlers(self) -> None:
        """Register event handlers for this component."""
        # Subscribe to LLM responses to display them
        await self.subscribe(Topics.LLM_PROCESSING, "handle_llm_response")

    async def handle_llm_response(self, event: Event) -> None:
        """Handle LLM response events and display them to user."""
        if event.event_type == EventType.LLM_RESPONSE:
            response_content = event.payload.get("content", "")
            print(f"\n🤖 Assistant: {response_content}")

            # Show prompt again if REPL is still running
            if self._running_repl:
                print("\n💬 You: ", end="", flush=True)

    async def start_repl(self) -> None:
        """Start the interactive REPL loop."""
        if self._running_repl:
            logger.warning("REPL already running")
            return

        self._running_repl = True
        print("🚀 Domteur Chat Started! Type 'quit', 'exit', or Ctrl+C to stop.\n")

        try:
            while self._running_repl and self.is_running():
                try:
                    # Get user input
                    user_input = await self._get_user_input()

                    if user_input is None:  # EOF or quit command
                        break

                    # Create and publish user input event
                    event = Event(
                        event_type=EventType.USER_INPUT,
                        session_id=self.session_id,
                        payload={"content": user_input, "user": "human"},
                    )

                    await self.publish(Topics.USER_INPUT, event)

                except KeyboardInterrupt:
                    print("\n\n👋 Chat interrupted by user")
                    break
                except EOFError:
                    print("\n\n👋 Chat ended")
                    break
                except Exception as e:
                    logger.error(f"Error in REPL: {e}")
                    print(f"\n❌ Error: {e}")

        finally:
            self._running_repl = False
            print("💫 Goodbye!")

    async def stop_repl(self) -> None:
        """Stop the REPL loop."""
        self._running_repl = False

    async def _get_user_input(self) -> str | None:
        """Get user input asynchronously."""
        # Run input() in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        try:
            user_input = await loop.run_in_executor(None, lambda: input("💬 You: "))

            # Check for quit commands
            if user_input.lower().strip() in ["quit", "exit", "q", "bye"]:
                return None

            # Skip empty inputs
            if not user_input.strip():
                return await self._get_user_input()

            return user_input.strip()

        except (EOFError, KeyboardInterrupt):
            return None

    async def _cleanup(self) -> None:
        """Cleanup when component is stopped."""
        await self.stop_repl()


async def create_repl_component(
    session_id: str | None = None,
) -> InteractiveLLMInput:
    """Factory function to create and start a REPL component."""
    repl = InteractiveLLMInput(session_id=session_id)
    await repl.start()
    return repl

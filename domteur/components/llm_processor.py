"""LLM processor component using LangChain for Ollama integration."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM

from domteur.components.base import Component
from domteur.config import OllamaProvider, Settings
from domteur.events import Event, EventBusInterface, EventType, Topics

logger = logging.getLogger(__name__)


class LLMProcessor(Component):
    """LLM processor component that handles AI conversations using LangChain."""

    def __init__(self, settings: Settings, **kwargs):
        super().__init__("llm_processor", **kwargs)
        self.settings = settings
        self.llm_providers = {}
        self.current_provider = None
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize LLM providers from settings."""
        for provider_config in self.settings.llm_providers:
            try:
                if isinstance(provider_config, OllamaProvider):
                    # Initialize Ollama provider
                    llm = OllamaLLM(
                        model=provider_config.model, base_url=provider_config.base_url
                    )
                    self.llm_providers[provider_config.model] = llm

                    # Set first provider as current
                    if self.current_provider is None:
                        self.current_provider = provider_config.model
                        logger.info(
                            f"Set default LLM provider: {provider_config.model}"
                        )

                # TODO: Add OpenRouter support later
                # elif isinstance(provider_config, OpenRouterProvider):
                #     # Initialize OpenRouter provider
                #     pass

            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_config}: {e}")

    async def _register_handlers(self) -> None:
        """Register event handlers for this component."""
        # Subscribe to user input events
        await self.subscribe(Topics.USER_INPUT, "handle_user_input")

    async def handle_user_input(self, event: Event) -> None:
        """Handle user input events and generate LLM responses."""
        if event.event_type != EventType.USER_INPUT:
            return

        user_message = event.payload.get("content", "")
        if not user_message:
            logger.warning("Received empty user message")
            return

        try:
            # Get current LLM provider
            if (
                not self.current_provider
                or self.current_provider not in self.llm_providers
            ):
                error_msg = "No LLM provider available"
                logger.error(error_msg)
                await self._send_error_response(event.session_id, error_msg)
                return

            llm = self.llm_providers[self.current_provider]

            # Create chat prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="You are a helpful AI assistant. Provide concise and helpful responses."
                    ),
                    HumanMessage(content=user_message),
                ]
            )

            logger.info(
                f"Processing message with {self.current_provider}: {user_message[:50]}..."
            )

            # Generate response using LangChain
            formatted_prompt = prompt.format_messages(human_input=user_message)
            response = await self._generate_response(llm, formatted_prompt[-1].content)

            # Create and publish LLM response event
            response_event = Event(
                event_type=EventType.LLM_RESPONSE,
                session_id=event.session_id,
                payload={
                    "content": response,
                    "model": self.current_provider,
                    "original_message": user_message,
                },
            )

            await self.publish(Topics.LLM_PROCESSING, response_event)
            logger.info("LLM response generated and published")

        except Exception as e:
            logger.error(f"Error processing LLM request: {e}")
            await self._send_error_response(
                event.session_id, f"LLM processing error: {str(e)}"
            )

    async def _generate_response(self, llm, message: str) -> str:
        """Generate response using the LLM."""
        try:
            # Use LangChain's async interface if available
            if hasattr(llm, "ainvoke"):
                response = await llm.ainvoke(message)
            else:
                # Fallback to sync invoke in executor
                import asyncio

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, llm.invoke, message)

            return (
                response.strip() if isinstance(response, str) else str(response).strip()
            )

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise

    async def _send_error_response(self, session_id: str, error_message: str) -> None:
        """Send error response event."""
        error_event = Event(
            event_type=EventType.LLM_RESPONSE,
            session_id=session_id,
            payload={"content": f"❌ Error: {error_message}", "error": True},
        )
        await self.publish(Topics.LLM_PROCESSING, error_event)

    def get_available_models(self) -> list[str]:
        """Get list of available LLM models."""
        return list(self.llm_providers.keys())

    def switch_provider(self, model_name: str) -> bool:
        """Switch to a different LLM provider/model."""
        if model_name in self.llm_providers:
            self.current_provider = model_name
            logger.info(f"Switched to LLM provider: {model_name}")
            return True
        else:
            logger.warning(f"Model {model_name} not available")
            return False


async def create_llm_processor(
    settings: Settings, event_bus: EventBusInterface | None = None
) -> LLMProcessor:
    """Factory function to create and start an LLM processor component."""
    processor = LLMProcessor(settings, event_bus=event_bus)
    await processor.start()
    return processor

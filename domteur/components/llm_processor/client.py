"""LLM processor component using LangChain for Ollama integration."""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM

from domteur.components.base import MQTTClient, on_receive
from domteur.components.llm_processor.constants import (
    COMPONENT_NAME,
    TOPIC_COMPONENT_LLM_PROC_ANSWER,
)
from domteur.components.llm_processor.contracts import (
    Conversation,
    LLMResponse,
)
from domteur.components.repl.constants import TOPIC_TERMINAL_LLM_REQUEST
from domteur.config import OllamaProvider, Settings

logger = logging.getLogger(__name__)


MessagesT = list[AIMessage | HumanMessage | SystemMessage]
UnsetProvider = "unset"


class LLMProcessor(MQTTClient):
    """LLM processor component that handles AI conversations using LangChain."""

    component_name = COMPONENT_NAME

    def __init__(self, client, settings: Settings, name: str | None = None):
        super().__init__(client, name)
        self.settings = settings
        self.llm_providers = {}
        self.current_provider = UnsetProvider
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize LLM providers from settings."""
        for provider_config in self.settings.llm_providers:
            if isinstance(provider_config, OllamaProvider):
                # Initialize Ollama provider
                llm = OllamaLLM(
                    model=provider_config.model, base_url=provider_config.base_url
                )
                self.llm_providers[provider_config.model] = llm

                # Set first provider as current
                if self.current_provider == UnsetProvider:
                    self.current_provider = provider_config.model
                    logger.info(f"Set default LLM provider: {provider_config.model}")

                # TODO: Add OpenRouter support later
                # elif isinstance(provider_config, OpenRouterProvider):
                #     # Initialize OpenRouter provider

    # Auto-discovery methods will handle subscriptions and message routing

    @on_receive(TOPIC_TERMINAL_LLM_REQUEST, Conversation)
    async def handle_user_input(self, msg, event: Conversation) -> None:
        """Handle user input events and generate LLM responses."""
        user_message = event.content

        # Get current LLM provider
        if (
            self.current_provider == UnsetProvider
            or self.current_provider not in self.llm_providers
        ):
            error_msg = "No LLM provider available"
            logger.error(error_msg)
            await self._send_error_response(event.session_id, error_msg, str(msg.topic))
            return

        llm = self.llm_providers[self.current_provider]

        # Build conversation prompt with history
        messages: MessagesT = [
            SystemMessage(
                content="You are a helpful AI assistant. Provide concise and helpful responses."
            )
        ]
        conversation_history = event.conversation_history
        # Add conversation history
        mapping = {"user": HumanMessage, "assistant": AIMessage}
        for hist_msg in conversation_history[
            :-1
        ]:  # Exclude the current message as it's already the last one
            messages.append(mapping[hist_msg.role](content=hist_msg.content))

        messages.append(HumanMessage(user_message))

        try:
            logger.info(
                f"Processing message with {self.current_provider} (history: {len(conversation_history)} messages): {user_message[:50]}..."
            )

            # Generate response using conversation context
            response = await self._generate_response(llm, messages)

            # Create and publish LLM response event
            response_event = LLMResponse(
                session_id=event.session_id,
                content=response,
                model=self.current_provider,
                original_message=user_message,
            )
            await self.publish(
                TOPIC_COMPONENT_LLM_PROC_ANSWER.replace("+", self.name), response_event
            )
            logger.info("LLM response generated and published")

        except Exception as e:
            logger.error(f"Error processing LLM request: {e}")
            await self._send_error_response(
                event.session_id, f"LLM processing error: {str(e)}", str(msg.topic)
            )

    async def _generate_response(self, llm, messages: MessagesT) -> str:
        """Generate response using the LLM."""
        try:
            # Use LangChain's async interface that handles missing async methods
            response = await llm.ainvoke(messages)
            return (
                response.strip() if isinstance(response, str) else str(response).strip()
            )

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise

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

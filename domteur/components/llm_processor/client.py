"""LLM processor component using LangChain for Ollama integration."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM
from loguru import logger

from domteur.components.base import MQTTClient, on_publish, on_receive
from domteur.components.llm_processor.constants import (
    COMPONENT_NAME,
    TOPIC_COMPONENT_LLM_PROC_ANSWER,
)
from domteur.components.llm_processor.contracts import (
    Conversation,
    LLMResponse,
)
from domteur.components.repl.constants import TOPIC_TERMINAL_LLM_REQUEST
from domteur.config import BaseLLMProvider, OllamaProvider, Settings

MessagesT = list[AIMessage | HumanMessage | SystemMessage]


class LLMProcessor(MQTTClient):
    """LLM processor component that handles AI conversations using LangChain."""

    component_name = COMPONENT_NAME

    def __init__(self, client, settings: Settings, name: str | None = None):
        super().__init__(client, name)
        self.settings = settings
        self.llm_providers = {}
        self.current_provider: BaseLLMProvider | None = None
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize LLM providers from settings."""
        for provider_config in self.settings.llm_providers:
            if isinstance(provider_config, OllamaProvider):
                # Initialize Ollama provider
                llm = OllamaLLM(
                    model=provider_config.model, base_url=provider_config.base_url
                )
                self.llm_providers[provider_config.model_id] = llm

                # Set first provider as current
                if not self.current_provider:
                    self.current_provider = provider_config
                    logger.info(f"Set default LLM provider: {provider_config.model_id}")

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
            self.current_provider is None
            or self.current_provider.model_id not in self.llm_providers
        ):
            error_msg = "No LLM provider available"
            logger.error(error_msg)
            await self._send_error_response(event.session_id, error_msg, str(msg.topic))
            return

        # Build conversation prompt with history
        messages: MessagesT = [
            SystemMessage(content=self.current_provider.system_prompt)
        ]
        conversation_history = event.conversation_history
        # Add conversation history
        mapping = {"user": HumanMessage, "assistant": AIMessage}
        for hist_msg in conversation_history[
            :-1
        ]:  # Exclude the current message as it's already the last one
            messages.append(mapping[hist_msg.role](content=hist_msg.content))

        messages.append(HumanMessage(user_message))
        response_event = LLMResponse(
            session_id=event.session_id,
            content="No provider",
            model=self.current_provider.model_id,
            original_message=user_message,
        )
        await self.send_answer(msg, messages, response_event)

    @on_publish(TOPIC_COMPONENT_LLM_PROC_ANSWER, LLMResponse)
    async def send_answer(self, msg, messages: MessagesT, pre_response: LLMResponse):
        if not self.current_provider:
            return pre_response
        llm = self.llm_providers[self.current_provider.model_id]
        # Generate response using conversation context
        pre_response.content = await self._generate_response(llm, messages)
        # Create and publish LLM response event
        return pre_response

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

    def switch_provider(self, model_id: str) -> bool:
        """Switch to a different LLM provider/model."""
        if model_id in self.llm_providers:
            self.current_provider = self.llm_providers[model_id]
            logger.info(f"Switched to LLM provider: {model_id}")
            return True
        else:
            logger.warning(f"Model {model_id} not available")
            return False

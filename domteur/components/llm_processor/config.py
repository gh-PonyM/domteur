from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM
from loguru import logger
from pydantic import BaseModel, Discriminator, Field, SecretStr, Tag

MessagesT = list[AIMessage | HumanMessage | SystemMessage]


class BaseLLMProvider(BaseModel):
    name: str = Field(
        "default", description="A identifying name for this configuration"
    )
    model: str
    system_prompt: str = (
        "You are a helpful AI assistant. Provide concise and helpful responses."
    )

    @property
    def model_id(self):
        return f"{self.type}:{self.name}"

    async def ainvoke(self, messages: MessagesT):
        raise NotImplementedError


class OpenRouterProvider(BaseModel):
    """OpenRouter LLM provider configuration."""

    type: Literal["openrouter"]
    api_key: SecretStr
    model: str = "anthropic/claude-3-haiku"
    base_url: str = "https://openrouter.ai/api/v1"


def get_provider_type(v):
    """Discriminator function for LLM provider types."""
    if isinstance(v, dict):
        return v.get("type")
    return getattr(v, "type", None)


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider configuration."""

    type: Literal["ollama"]
    base_url: str = "http://localhost:11434"
    model: str = "llama2"
    model_download_on_startup: bool = False
    model_download_if_not_present: bool = True

    def model_post_init(self, context):
        self._instance = OllamaLLM(model=self.model, base_url=self.base_url)

    def pull_model(self):
        if not self.model_download_if_not_present:
            return
        model = self.model
        logger.info(f"Pulling ollama model '{model}")
        self.instance._client.pull(model)
        logger.info(f"Finished downloading '{model}")

    async def ainvoke(self, messages: MessagesT):
        return self._instance.ainvoke(messages)


LLMProvider = Annotated[
    Annotated[OllamaProvider, Tag("ollama")]
    | Annotated[OpenRouterProvider, Tag("openrouter")],
    Discriminator(get_provider_type),
]

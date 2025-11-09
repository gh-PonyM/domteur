import json
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import typer
import yaml
from pydantic import (
    BaseModel,
    Field,
)
from pydantic.functional_validators import AfterValidator
from pydantic_settings import BaseSettings as DefaultBaseSettings, SettingsConfigDict

from domteur import __version__
from domteur.components.llm_processor.config import (
    LLMProvider,
    OllamaProvider,
)
from domteur.components.stt.configs import WhisperSTTConfig
from domteur.components.tts.piper import PiperTTSConfig
from domteur.validators import string_or_path

APP_NAME = "domteur"
ENV_CONFIG_KEY = "DOMTEUR_CONFIG_DIR"
CONFIG_FN = "config.yml"
APP_DIR = Path(typer.get_app_dir(APP_NAME))
APP_CFG = APP_DIR / CONFIG_FN


class BaseSettings(DefaultBaseSettings):
    model_config = SettingsConfigDict(extra="forbid", env_nested_delimiter="__")


StrOrPath = Annotated[Path, AfterValidator(string_or_path)]


class DatabaseConfig(BaseModel):
    """Database configuration."""

    type: Literal["sqlite"] = "sqlite"
    path: str = "./domteur.db"


class Settings(BaseSettings):
    version: str = __version__
    config_format: ClassVar[str] = "yaml"
    model_config = SettingsConfigDict(extra="forbid")
    broker_host: str = "broker"
    broker_port: int = 1883

    # LLM providers configuration
    llm_providers: list[LLMProvider] = Field(
        default_factory=lambda: [OllamaProvider(type="ollama", model="llama2")]
    )

    # Database configuration
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # TTS configuration
    tts: PiperTTSConfig = Field(default_factory=PiperTTSConfig)

    ## SST
    sst: WhisperSTTConfig = Field(default_factory=WhisperSTTConfig)

    # not serialized
    file_path: StrOrPath | None = Field(
        description="File path where this is stored", default=None
    )

    def to_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True, exclude={"file_path"}),
            indent=2,
        )

    def to_yaml(self):
        return yaml.safe_dump(
            self.model_dump(mode="json", exclude_none=True, exclude={"file_path"})
        )

    def dump(self):
        return getattr(self, f"to_{self.config_format}")()

    def save(self, config_path: None | Path = None):
        """Save the current configuration. Secrets should be excluded as ony **** is dumped"""
        path = config_path or self.file_path
        if not path:
            raise FileNotFoundError("config file path not set")
        path.write_text(self.to_json())

    def __bool__(self):
        """A config with only the version and filepath set is considered falsy"""
        return bool(
            self.model_dump(exclude_none=True, exclude={"file_path", "version"})
        )

    def prompt_initial_config(self):
        """Function used when the cli is run initially without a config"""
        pass

    @classmethod
    def from_file(cls, file_path: Path | str):
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        data = yaml.safe_load(file_path.read_text())
        if "file_path" in cls.model_fields:
            data["file_path"] = str(file_path)
        return cls.model_validate(data)

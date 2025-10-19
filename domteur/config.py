import json
from pathlib import Path
from typing import Annotated, ClassVar

import typer
import yaml
from pydantic import (
    Field,
    PrivateAttr,
    SecretStr,
)
from pydantic.functional_validators import AfterValidator
from pydantic_settings import BaseSettings, SettingsConfigDict

from domteur import __version__
from domteur.validators import string_or_path

APP_NAME = "domteur"
ENV_CONFIG_KEY = "DOMTEUR_CONFIG_DIR"
CONFIG_FN = "config.yaml"
APP_DIR = Path(typer.get_app_dir(APP_NAME))
APP_CFG = APP_DIR / CONFIG_FN


class ExampleClient:
    def __init__(self, secret):
        self.secret = secret


class SecretsSettings(BaseSettings):
    """Some settings for your cli"""

    SECRET_KEY: SecretStr | None = None
    _client: ExampleClient | None = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = (
            ExampleClient(secret=self.SECRET_KEY.get_secret_value())
            if self.SECRET_KEY
            else None
        )

    @property
    def client(self) -> ExampleClient | None:
        return self._client

    # TODO[pydantic]: The following keys were removed: `json_encoders`.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
    model_config = SettingsConfigDict(
        json_encoders={
            SecretStr: lambda v: v.get_secret_value() if v else None,
        }
    )


StrOrPath = Annotated[Path, AfterValidator(string_or_path)]


class Settings(SecretsSettings):
    version: str = __version__
    config_format: ClassVar[str] = "yaml"

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

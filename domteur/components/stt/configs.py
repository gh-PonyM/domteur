from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, NegativeFloat

ModelSizesT = Literal[
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "small.en",
    "medium",
    "medium.en",
    "distil-medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v3",
    "distil-large-v2",
    "distil-medium.en",
    "distil-small.en",
    "large-v3-turbo",
    "turbo",
]


class WhisperSTTConfig(BaseModel):
    model_size: str = ModelSizesT
    device: Literal["cuda", "cpu"] = "cpu"
    compute_type: Literal["float16", "int8_float16", "int8"] = "int8"
    beam_size: int = 5
    word_timestamps: bool = False
    model_path: Path = Field(
        Path("/tmp/whisper_stt"), description="Download path for whisper models"
    )
    # vad_filter: bool = False
    # vad_parameters = dict(min_silence_duration_ms=500),


class AudioStreamConfig(BaseModel):
    split_after_silence_secs: float = 0.5
    limiter_threshold: NegativeFloat = -40.0
    sample_rate: int = 16000
    channels: int = 2
    block_size: int = 1024

from typing import Literal

from pydantic import BaseModel


class WhisperSTTConfig(BaseModel):
    model_size: str = "large-v3"
    device: Literal["cuda", "cpu"] = "cpu"
    compute_type: Literal["float16", "int8_float16", "int8"] = "int8"
    beam_size: int = 5
    word_timestamps: bool = False
    sample_rate: int = 44100
    channels: int = 2
    block_size: int = 1024
    # vad_filter: bool = False
    # vad_parameters = dict(min_silence_duration_ms=500),


class AudioStreamConfig(BaseModel):
    split_after_silence_secs: float = 0.75
    split_threshold_db: float = 20.0

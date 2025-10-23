"""Text-to-speech component using Piper TTS with direct audio output."""

import logging
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pyaudio
from piper import PiperVoice
from pydantic import BaseModel, Field

from domteur.components.base import MQTTClient, on_receive
from domteur.components.llm_processor.constants import TOPIC_COMPONENT_LLM_PROC_ANSWER
from domteur.components.llm_processor.contracts import LLMResponse

if TYPE_CHECKING:
    from domteur.config import Settings

logger = logging.getLogger(__name__)


class PiperTTSConfig(BaseModel):
    """Configuration for text-to-speech engine."""

    type: Literal["piper"] = "piper"
    voice_model_name: str = "de_DE-kerstin-low"
    volume: float = 1.0
    speed: float = 1.0
    use_cuda: bool = False
    auto_download_voice: bool = True
    sample_rate: int = 22050
    chunk_size: int = 1024
    model_storage_path: Path = Field(default_factory=lambda: Path("/tmp"))

    @property
    def voice_model_path(self) -> Path:
        return Path(f"{self.voice_model_name}.onnx")

    @property
    def voice_config_path(self) -> Path:
        return Path(f"{self.voice_model_name}.json")


def synthesize_to_file(voice, text: str, output_path: Path) -> None:
    """Synthesize text to a WAV file."""
    with wave.open(str(output_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)


def download_voice(voice_name: str, dl_dir: Path) -> None:
    """Download the voice model if it doesn't exist."""
    logger.info(f"Downloading voice model: {voice_name}")
    import subprocess
    import sys

    # Use piper's download utility
    subprocess.run(
        [
            sys.executable,
            "-m",
            "piper.download_voices",
            "--download-dir",
            str(dl_dir),
            voice_name,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    logger.info("Voice model downloaded successfully")


def play_wav_file(
    wav_path: Path, chunk_size: int, audio_interface: pyaudio.PyAudio | None = None
) -> None:
    """Play a WAV file using PyAudio."""
    audio_interface = audio_interface or pyaudio.PyAudio()

    # TODO: claude fucked up, it's way easier, and this only stutters
    with wave.open(str(wav_path), "rb") as wav_file:
        # Get audio parameters
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()

        # Open audio stream
        stream = audio_interface.open(
            format=audio_interface.get_format_from_width(sample_width),
            channels=channels,
            rate=sample_rate,
            output=True,
        )

        try:
            # Read and play audio data in chunks
            data = wav_file.readframes(chunk_size)
            while data:
                stream.write(data)
                data = wav_file.readframes(chunk_size)
        finally:
            stream.stop_stream()
            stream.close()


def load_voice(
    voice_name: str,
    model_path: Path,
    auto_download_voice: bool = True,
    use_cuda: bool = False,
) -> PiperVoice:
    """Load the Piper TTS voice model."""
    # Auto-download voice if configured and not exists
    if auto_download_voice and not model_path.exists():
        # TODO: Download path not specified
        download_voice(voice_name, model_path.parent)

    if not model_path.exists():
        raise FileNotFoundError(f"Voice model not found: {model_path}")
    return PiperVoice.load(model_path=model_path, use_cuda=use_cuda)


class PiperTTS(MQTTClient):
    """Text-to-speech component that converts text to speech using Piper TTS.
    This is meant to run in a separate thread if async is used"""

    def __init__(self, client, settings: "Settings", name: str | None = None):
        super().__init__(client, name)
        self.config = settings.tts
        self.voice: PiperVoice | None = None
        self.audio_interface: pyaudio.PyAudio | None = None

    def _prepare_text(self, text):
        return text

    @on_receive(TOPIC_COMPONENT_LLM_PROC_ANSWER, LLMResponse)
    async def handle_tts_request(self, event: LLMResponse) -> None:
        """Handle LLM responses."""
        text = self._prepare_text(event.content)
        logger.info(f"Processing TTS request: {text[:50]}...")

        try:
            self._synthesize_and_play(text)
            logger.info("TTS synthesis and playback completed")
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")

    def _synthesize_and_play(self, text: str) -> None:
        """Synthesize text to speech and play it directly."""
        if not self.voice:
            # maybe try except here?
            self.voice = load_voice(
                voice_name=self.config.voice_model_name,
                model_path=self.config.voice_model_path,
                auto_download_voice=self.config.auto_download_voice,
                use_cuda=self.config.use_cuda,
            )

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Synthesize speech to WAV file
            synthesize_to_file(self.voice, text, temp_path)

            # Play the WAV file
            if not self.audio_interface:
                self.audio_interface = pyaudio.PyAudio()
            play_wav_file(temp_path, self.config.chunk_size, self.audio_interface)

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.audio_interface:
            self.audio_interface.terminate()
            self.audio_interface = None

        self.voice = None
        logger.info("TTS engine cleaned up")

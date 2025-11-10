"""Audio commands for domteur CLI."""

import asyncio

import typer
from loguru import logger

from domteur.components.stt.local_streaming import (
    StreamingTranscriber,
    load_model,
)
from domteur.config import Settings

audio_cli = typer.Typer(name="audio", help="Audio functionalities")


@audio_cli.command()
def live_transcribe(
    ctx: typer.Context, dry_run: bool = False, volume_bar: bool = False
) -> None:
    """Demo of StreamingTranscriber with live audio transcription.

    Press Ctrl+C to stop recording.
    """
    cfg: Settings = ctx.meta["cfg"]

    typer.echo("🎤 Live streaming transcription demo started. Press Ctrl+C to stop.\n")

    # Load the Whisper model
    logger.info(f"Loading Whisper model: {cfg.stt.model_size}")

    # Create audio stream config
    model = load_model(cfg.stt)

    # Create the streaming transcriber
    transcriber = StreamingTranscriber(
        model=model, dry_run=dry_run, show_volume_bar=volume_bar, whisper_config=cfg.stt
    )

    # Run the async transcription
    try:
        asyncio.run(_run_stream_demo(transcriber))
    except KeyboardInterrupt:
        typer.echo("\n\n✋ Transcription stopped.")
        logger.info("Live transcription interrupted by user")


async def _run_stream_demo(transcriber: StreamingTranscriber) -> None:
    """Run the streaming transcriber demo."""
    async with transcriber.stream():
        async for info, segment in transcriber.transcribe_stream():
            typer.echo(f"📝 {segment.text if segment else 'Dry run text'}")

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from domteur.components.base import Error, MQTTClient, on_publish, on_receive
from domteur.components.llm_processor.contracts import (
    Conversation,
    HistoryEntry,
    LLMResponse,
)
from domteur.components.stt.contracts import AudioFileTranscribeRequest, STTextSegment
from domteur.components.tts.contracts import TTSControl, TTSStreamChunk
from domteur.config import Settings


class LLMTerminalChat(MQTTClient):
    def __init__(
        self,
        client,
        settings: Settings,
        name: str | None = None,
    ):
        super().__init__(client, settings, name)
        self.conversation_history: list[HistoryEntry] = []
        self.session_id = None

    @on_publish("output", TTSStreamChunk, event="play")
    async def send_tts_single_text(self, msg, query, priority: int = 2):
        return TTSStreamChunk(content=query, message_type="complete", priority=priority)

    @on_publish("tts_control", TTSControl)
    async def send_tts_control(self, msg, action):
        return TTSControl(action=action)

    @on_receive("LLMProcessor", "output", LLMResponse, "complete")
    async def handle_llm_message(self, msg, response: LLMResponse):
        self.conversation_history.extend(
            [
                HistoryEntry(role="user", content=response.original_message),
                HistoryEntry(role="assistant", content=response.content),
            ]
        )
        with patch_stdout():
            print(f"\n🤖 Assistant: {response.content}\n")

    @on_publish("audio_transcribe", AudioFileTranscribeRequest, event="request")
    async def terminal_stt_request_fp(self, msg, file_path: str):
        return AudioFileTranscribeRequest(file_path=file_path)

    @on_receive("WhisperSTT", "audio_transcribe", STTextSegment, "batch_output")
    async def show_transcription(self, msg, segment: STTextSegment):
        print(f"Transcription received: {segment.content}")

    @on_receive("+", "error", Error)
    async def show_errors(self, msg, err: Error):
        print(err.to_json())

    async def ask_questions(self):
        session = PromptSession()
        while True:
            with patch_stdout():
                query = await session.prompt_async("💬 You: ")
                query = query.strip()
            if not query:
                continue
            if query.startswith("/transcribe"):
                fp = query.replace("/transcribe", "").strip()
                await self.terminal_stt_request_fp(None, fp)
            elif query.startswith("/speak"):
                await self.send_tts_single_text(
                    None, query.replace("/speak", "", 1), priority=2
                )
            elif query.startswith("/interrupt"):
                await self.send_tts_single_text(
                    None, "You wanted me to stop", priority=1
                )
            elif query.startswith("/stop"):
                logger.info("Send audio stop")
                await self.send_tts_control(None, action="STOP")
            elif query.startswith("/pause"):
                logger.info("Send audio stop")
                await self.send_tts_control(None, action="PAUSE")
            elif query.startswith("/mute"):
                logger.info("Send audio mute")
                await self.send_tts_control(None, action="MUTE")
            elif query.startswith("/unmute"):
                logger.info("Send audio unmute")
                await self.send_tts_control(None, action="UNMUTE")
            else:
                logger.info("Send to llm")
                await self.send_to_llm(None, query)

    @on_publish("llm_chat", Conversation)
    async def send_to_llm(self, msg, query: str):
        logger.info(f"Conversation lengths: {len(self.conversation_history)}")
        c = Conversation(content=query, conversation_history=self.conversation_history)
        if not self.session_id:
            self.session_id = c.session_id
        else:
            c.session_id = self.session_id
        return c

    def start_coros(self):
        yield self.start()
        yield self.ask_questions()

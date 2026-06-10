from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts


class EdgeTTS:
    #--------[Initialize Edge TTS Voice Settings]--------#
    def __init__(self, voice: str = "ko-KR-SunHiNeural", rate: str = "+0%"):
        self.voice = voice
        self.rate = rate

    #--------[Save Text As Speech Audio]--------#
    async def _save_async(self, text: str, output_path: str) -> str:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        communicator = edge_tts.Communicate(text=text, voice=self.voice, rate=self.rate)
        await communicator.save(str(output))
        return str(output)

    #--------[Run TTS Synchronously From Agent]--------#
    def speak(self, text: str, output_path: str) -> str:
        return asyncio.run(self._save_async(text, output_path))

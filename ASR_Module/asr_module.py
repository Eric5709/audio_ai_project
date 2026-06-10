from __future__ import annotations

import whisper


class WhisperASR:
    #--------[Load OpenAI Whisper ASR Model]--------#
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str | None = None,
        compute_type: str | None = None,
    ):
        self.compute_type = compute_type
        self.model = whisper.load_model(model_size, device=device)

    #--------[Transcribe Audio To Text]--------#
    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        transcribe_options = {"language": language}

        if self.compute_type == "float16":
            transcribe_options["fp16"] = True
        elif self.compute_type in {"float32", "int8"}:
            transcribe_options["fp16"] = False

        result = self.model.transcribe(audio_path, **transcribe_options)
        return result["text"].strip()
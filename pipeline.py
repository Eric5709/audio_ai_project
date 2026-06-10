"""
pipeline.py

식재료 레시피 음성 도우미의 핵심 파이프라인.

흐름:
  🎤 음성 + 📷 사진 → [ASR] 음성을 텍스트로
                    → [Gemini] 사진 속 식재료 인식 + 레시피 생성 (VLM+LLM 한 방)
                    → [TTS] 레시피를 음성으로 🔊

설계 원칙 (이전과 동일):
  각 단계를 인터페이스 뒤에 두고, 진짜 구현과 mock을 1:1 교체 가능하게 한다.
  => API 키 없이도 mock으로 전체 흐름을 돌려볼 수 있다.
"""
from __future__ import annotations
from typing import Protocol
import os

# .env 파일이 있으면 읽어서 os.environ 에 채운다 (키를 코드 밖에 둠).
# python-dotenv 가 없어도 동작하도록 try 로 감싼다.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ===========================================================================
# 인터페이스 (계약)
# ===========================================================================
class ASR(Protocol):
    def transcribe(self, audio_path: str) -> str:
        """오디오 파일 경로 -> 한국어 텍스트."""
        ...


class RecipeVLM(Protocol):
    def suggest(self, image_path: str, user_request: str) -> str:
        """식재료 사진 + 사용자 요청 -> 레시피 텍스트(한국어)."""
        ...


class TTS(Protocol):
    def speak(self, text: str, out_path: str = "response.mp3") -> str:
        """텍스트 -> 음성 파일. 저장 경로를 반환."""
        ...


# ===========================================================================
# 진짜 구현 1: faster-whisper (로컬 ASR)
# ===========================================================================
class WhisperASR:
    """faster-whisper 로컬 추론. 무료, 오프라인, 한국어 지원."""

    def __init__(self, model_size: str = "large-v3", device: str = "auto",
                 compute_type: str = "auto"):
        from faster_whisper import WhisperModel
        # GPU 있으면 device="cuda", compute_type="float16" 권장
        self.model = WhisperModel(model_size, device=device,
                                  compute_type=compute_type)

    def transcribe(self, audio_path: str) -> str:
        segments, info = self.model.transcribe(audio_path, language="ko")
        return "".join(seg.text for seg in segments).strip()


# ===========================================================================
# 진짜 구현 2: Gemini Flash (VLM + LLM 한 방)
#   사진 속 식재료 인식과 레시피 생성을 한 번의 호출로 처리.
# ===========================================================================
RECIPE_SYSTEM = """당신은 친절한 한국어 요리 도우미입니다.
사용자가 보내준 사진 속 식재료를 보고, 그 재료들로 만들 수 있는 요리 레시피를 제안하세요.

답변 형식:
1) 사진에서 보이는 식재료를 먼저 짧게 확인해 주세요.
2) 그 재료로 만들 수 있는 요리 1~2개를 제안하세요.
3) 가장 추천하는 요리의 조리법을 단계별로 간단히 설명하세요.

음성으로 들려줄 답변이므로, 너무 길지 않고 자연스러운 구어체 한국어로 말하세요.
사진에 식재료가 잘 안 보이면 솔직히 말하고 다시 찍어달라고 하세요."""


class GeminiRecipe:
    """Google Gemini Flash 멀티모달. 무료 티어(하루 1,500 요청, 카드 불필요)."""

    def __init__(self, model: str = "gemini-2.5-flash-lite", api_key: str | None = None):
        from google import genai
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY 환경변수가 없습니다. "
                "https://aistudio.google.com 에서 무료 키를 발급받아 설정하세요."
            )
        self.client = genai.Client(api_key=key)
        self.model = model

    def suggest(self, image_path: str, user_request: str) -> str:
        from google.genai import types
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        # 확장자로 mime 추정 (간단히)
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        mime = "image/png" if ext == "png" else "image/jpeg"

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                f"{RECIPE_SYSTEM}\n\n[사용자 요청] {user_request}",
            ],
        )
        return response.text.strip()


# ===========================================================================
# 진짜 구현 3: Edge TTS (무료, 무제한, 한국어 자연스러움)
# ===========================================================================
class EdgeTTS:
    """Microsoft Edge 신경망 TTS. 무료, API 키 불필요.
    한국어 음성: ko-KR-SunHiNeural(여), ko-KR-HyunsuNeural(남), ko-KR-InJoonNeural(남)."""

    def __init__(self, voice: str = "ko-KR-SunHiNeural"):
        self.voice = voice

    def speak(self, text: str, out_path: str = "response.mp3") -> str:
        import asyncio
        import edge_tts

        async def _run():
            communicate = edge_tts.Communicate(text=text, voice=self.voice)
            await communicate.save(out_path)

        asyncio.run(_run())
        return out_path

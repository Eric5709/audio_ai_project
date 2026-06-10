from __future__ import annotations
import os
import mimetypes
from google import genai
from google.genai import types

class GeminiRecipe:
    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY 환경변수가 없습니다. "
                "https://aistudio.google.com 에서 무료 키를 발급받아 설정하세요."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def suggest(self, image_path: str, user_text: str) -> str:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                user_text,
            ],
        )
        return (response.text or "").strip()
"""
main.py

오케스트레이터. 음성 + 사진 -> 레시피 음성 파이프라인을 한 턴 실행한다.
단계별 latency를 측정해 출력한다 (음성AI 과제의 핵심 평가 지표).

실행:
  python main.py            # mock 모드 (키/마이크 불필요, 흐름 검증)
  python main.py --real     # 진짜 모드 (faster-whisper + Gemini + Edge TTS)

진짜 모드 준비물:
  - 환경변수 GEMINI_API_KEY  (https://aistudio.google.com 무료 발급)
  - 입력 음성 파일 (예: input.wav) 과 식재료 사진 (예: fridge.jpg)
"""
from __future__ import annotations
import sys
import time

from pipeline import ASR, RecipeVLM, TTS


class RecipeBot:
    def __init__(self, asr: ASR, vlm: RecipeVLM, tts: TTS):
        self.asr = asr
        self.vlm = vlm
        self.tts = tts

    def run_once(self, audio_path: str, image_path: str,
                 out_path: str = "response.mp3") -> str:
        timings: dict[str, float] = {}

        # 1) ASR: 음성 -> 텍스트
        t = time.perf_counter()
        user_text = self.asr.transcribe(audio_path)
        timings["ASR"] = time.perf_counter() - t
        print(f"   📝 인식된 요청: {user_text}")

        # 2) Gemini: 사진 속 식재료 인식 + 레시피 생성
        t = time.perf_counter()
        recipe = self.vlm.suggest(image_path, user_text)
        timings["Gemini(VLM+LLM)"] = time.perf_counter() - t

        # 3) TTS: 레시피 -> 음성
        t = time.perf_counter()
        saved = self.tts.speak(recipe, out_path)
        timings["TTS"] = time.perf_counter() - t

        # latency 리포트
        print("   " + "-" * 44)
        total = 0.0
        for stage, dt in timings.items():
            print(f"   ⏱  {stage:<18} {dt*1000:7.0f} ms")
            total += dt
        print(f"   ⏱  {'TOTAL':<18} {total*1000:7.0f} ms")
        if saved and saved != out_path:
            pass
        print(f"   💾 음성 저장: {saved}")
        return recipe


# ===========================================================================
# 조립 (Composition Root) -- mock <-> 진짜 교체 지점
# ===========================================================================
def build(real: bool) -> RecipeBot:
    if real:
        from pipeline import WhisperASR, GeminiRecipe, EdgeTTS
        return RecipeBot(
            asr=WhisperASR(model_size="large-v3"),   # GPU면 device="cuda"
            vlm=GeminiRecipe(model="gemini-2.5-flash"),
            tts=EdgeTTS(voice="ko-KR-SunHiNeural"),
        )
    else:
        from mocks import MockASR, MockRecipe, MockTTS
        return RecipeBot(
            asr=MockASR(scripted="나 너무 배고파, 이 사진에 있는 재료로 만들 수 있는 요리 알려줘"),
            vlm=MockRecipe(),
            tts=MockTTS(),
        )


if __name__ == "__main__":
    real = "--real" in sys.argv
    print("=" * 52)
    print(f"  식재료 레시피 음성 도우미  ({'REAL' if real else 'MOCK'} 모드)")
    print("=" * 52)

    bot = build(real)

    if real:
        # 실제 입력 파일 경로 (본인 환경에 맞게 수정)
        bot.run_once(audio_path="input.wav", image_path="fridge.jpg")
    else:
        # mock: 더미 경로로 흐름만 검증
        bot.run_once(audio_path="(mock)", image_path="(mock)")

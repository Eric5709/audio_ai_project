"""
run_text.py

음성 파일 없이 '사진 + 텍스트 요청'만으로 레시피를 받는 버전.
ASR(음성->텍스트) 단계를 건너뛴다. 이미 텍스트 요청을 가지고 있을 때 사용.

실행:
  python run_text.py
  python run_text.py --image fridge.jpg --text "이 재료로 매운 요리 알려줘"
"""
from __future__ import annotations
import argparse
import time

from pipeline import GeminiRecipe, EdgeTTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="fridge.jpg", help="식재료 사진 경로")
    ap.add_argument("--text",
                    default="나 너무 배고파, 이 사진에 있는 재료로 만들 수 있는 요리 알려줘",
                    help="요청 텍스트")
    ap.add_argument("--out", default="response.mp3", help="출력 음성 파일")
    ap.add_argument("--voice", default="ko-KR-SunHiNeural", help="TTS 음성")
    args = ap.parse_args()

    print("=" * 52)
    print("  레시피 도우미 (텍스트 입력 모드, ASR 생략)")
    print("=" * 52)
    print(f"   📷 사진: {args.image}")
    print(f"   📝 요청: {args.text}")

    vlm = GeminiRecipe(model="gemini-2.5-flash")
    tts = EdgeTTS(voice=args.voice)

    # 1) Gemini: 사진 속 식재료 인식 + 레시피 생성
    t = time.perf_counter()
    recipe = vlm.suggest(args.image, args.text)
    t_gemini = time.perf_counter() - t
    print("   " + "-" * 44)
    print(f"   🍳 레시피:\n   {recipe}")

    # 2) TTS: 레시피 -> 음성
    t = time.perf_counter()
    saved = tts.speak(recipe, args.out)
    t_tts = time.perf_counter() - t

    print("   " + "-" * 44)
    print(f"   ⏱  Gemini(VLM+LLM)  {t_gemini*1000:7.0f} ms")
    print(f"   ⏱  TTS              {t_tts*1000:7.0f} ms")
    print(f"   💾 음성 저장: {saved}")


if __name__ == "__main__":
    main()

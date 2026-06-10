from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from ASR_Module import WhisperASR
from LLM_Module.llm_module import GeminiRecipe
from TTS_Module import EdgeTTS

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "macgyver_config.yaml"


#--------[Resolve Project File Path]--------#
def resolve_project_path(project_root: Path, file_path: str) -> Path:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


#--------[Load MacGyver YAML Config]--------#
def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    return {
        "audio_path": config.get("audio_path", "Source/Audio/english_cook.m4a"),
        "image_path": config.get("image_path", "Source/Image/fridge.png"),
        "language": config.get("language"),
        "model_size": config.get("model_size", "large-v3"),
        "device": config.get("device"),
        "compute_type": config.get("compute_type"),
        "gemini_model": config.get("gemini_model", "gemini-3.5-flash"),
        "tts_enabled": config.get("tts_enabled", True),
        "tts_output_path": config.get("tts_output_path", "response.mp3"),
        "tts_voice": config.get("tts_voice", "ko-KR-SunHiNeural"),
        "tts_rate": config.get("tts_rate", "+0%"),
    }


#--------[Build Terminal Friendly Agent Request]--------#
def build_agent_prompt(user_text: str) -> str:
    return (
        f"{user_text}\n\n"
        "이미지 속 물건과 사용자의 요청을 함께 보고, 요청 의도에 맞는 추천을 해줘. "
        "음식 재료라면 요리를, 술이나 음료라면 칵테일 또는 음료 조합을, "
        "도구나 물건이라면 사용 방법이나 선택지를 추천해줘. "
        "터미널에 그대로 출력할 답변이니까 JSON이나 마크다운 코드블록은 쓰지 마. "
        "별표도 절대 쓰지 마. "
        "첫 줄은 TTS로 읽기 좋은 짧은 추천 멘트 1문장으로 작성해. "
        "그 다음에는 추천 2개를 메뉴/추천 이름, 재료/물건, 짧은 방법 순서로 간단히 적어줘. "
        "서론은 길게 쓰지 말고 바로 추천으로 들어가."
    )


#--------[Run Voice And Image Reasoning Agent]--------#
def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")
    config = load_config(CONFIG_PATH)

    parser = argparse.ArgumentParser(
        description="Transcribe a voice request, then reason over it with an image."
    )
    parser.add_argument("--audio", default=config["audio_path"], help="음성 파일 경로")
    parser.add_argument("--image", default=config["image_path"], help="이미지 파일 경로")
    parser.add_argument(
        "--language",
        default=config["language"],
        help='음성 언어 코드입니다. 영어는 "en", 한국어는 "ko"를 사용합니다.',
    )
    parser.add_argument("--model-size", default=config["model_size"], help="Whisper 모델 크기")
    parser.add_argument("--device", default=config["device"], help='Whisper 실행 장치: "cuda" 또는 "cpu"')
    parser.add_argument(
        "--compute-type",
        default=config["compute_type"],
        help='Whisper 정밀도 힌트입니다. GPU는 "float16", CPU는 "float32"를 권장합니다.',
    )
    parser.add_argument("--gemini-model", default=config["gemini_model"], help="Gemini 모델명")
    parser.add_argument("--tts-output", default=config["tts_output_path"], help="TTS 출력 음성 파일 경로")
    parser.add_argument("--tts-voice", default=config["tts_voice"], help="Edge TTS 음성 이름")
    parser.add_argument("--tts-rate", default=config["tts_rate"], help='Edge TTS 음성 속도입니다. 예: "+0%"')
    parser.add_argument(
        "--no-tts",
        action="store_false",
        dest="tts_enabled",
        default=config["tts_enabled"],
        help="TTS 음성 파일 생성을 끕니다.",
    )
    args = parser.parse_args()

    audio_path = resolve_project_path(project_root, args.audio)
    image_path = resolve_project_path(project_root, args.image)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print("=" * 52)
    print("  MacGyver Agent (ASR + Image Reasoning)")
    print("=" * 52)
    print(f"   Audio: {audio_path}")
    print(f"   Image: {image_path}")

    #--------[Convert Voice Request To Text]--------#
    asr_started_at = time.perf_counter()
    asr = WhisperASR(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )
    user_text = asr.transcribe(str(audio_path), language=args.language)
    asr_elapsed = time.perf_counter() - asr_started_at

    print("   " + "-" * 44)
    print(f"   ASR Text:\n   {user_text}")

    #--------[Analyze Image With Transcribed Request]--------#
    gemini_started_at = time.perf_counter()
    gemini = GeminiRecipe(model=args.gemini_model)
    answer = gemini.suggest(str(image_path), build_agent_prompt(user_text)).replace("*", "")
    gemini_elapsed = time.perf_counter() - gemini_started_at

    print("   " + "-" * 44)
    print(f"   Answer:\n{answer}")

    tts_elapsed = 0.0
    saved_audio_path = None
    if args.tts_enabled:
        #--------[Convert Concise Answer To Speech]--------#
        tts_started_at = time.perf_counter()
        tts_output_path = resolve_project_path(project_root, args.tts_output)
        tts = EdgeTTS(voice=args.tts_voice, rate=args.tts_rate)
        saved_audio_path = tts.speak(answer, str(tts_output_path))
        tts_elapsed = time.perf_counter() - tts_started_at

    print("   " + "-" * 44)
    print(f"   ASR     {asr_elapsed * 1000:7.0f} ms")
    print(f"   Gemini  {gemini_elapsed * 1000:7.0f} ms")
    if saved_audio_path is not None:
        print(f"   TTS     {tts_elapsed * 1000:7.0f} ms")
        print(f"   TTS saved: {saved_audio_path}")


if __name__ == "__main__":
    main()

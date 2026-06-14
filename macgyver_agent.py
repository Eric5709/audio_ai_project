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
        "asr_refine_enabled": config.get("asr_refine_enabled", True),
        "tts_enabled": config.get("tts_enabled", True),
        "tts_refine_enabled": config.get("tts_refine_enabled", True),
        "tts_output_path": config.get("tts_output_path", "response.mp3"),
        "tts_voice_ko": config.get("tts_voice_ko", "ko-KR-SunHiNeural"),
        "tts_voice_en": config.get("tts_voice_en", "en-US-JennyNeural"),
        "tts_rate": config.get("tts_rate", "+0%"),
    }


#--------[Detect Response Language From ASR Text]--------#
def detect_response_language(user_text: str, language_hint: str | None = None) -> str:
    if language_hint in {"ko", "en"}:
        return language_hint

    has_korean = any("가" <= letter <= "힣" for letter in user_text)
    return "ko" if has_korean else "en"


#--------[Select TTS Voice For Language]--------#
def select_tts_voice(config: dict[str, Any], response_language: str, voice_override: str | None) -> str:
    if voice_override:
        return voice_override

    if response_language == "ko":
        return config["tts_voice_ko"]
    return config["tts_voice_en"]


#--------[Build Terminal Friendly Agent Request]--------#
def build_agent_prompt(user_text: str, response_language: str) -> str:
    language_instruction = "한국어로 답변해줘." if response_language == "ko" else "Answer in English."

    return (
        f"{user_text}\n\n"
        f"{language_instruction} "
        "이미지 속 물건과 사용자의 요청을 함께 보고, 요청 의도에 맞는 추천을 해줘. "
        "음식 재료라면 요리를, 술이나 음료라면 칵테일 또는 음료 조합을, "
        "도구나 물건이라면 사용 방법이나 선택지를 추천해줘. "
        "터미널에 그대로 출력할 답변이니까 JSON이나 마크다운 코드블록은 쓰지 마. "
        "별표도 절대 쓰지 마. "
        "첫 줄은 TTS로 읽기 좋은 짧은 추천 멘트 1문장으로 작성해. "
        "그 다음에는 추천 2개를 메뉴/추천 이름, 재료/물건, 짧은 방법 순서로 간단히 적어줘. "
        "서론은 길게 쓰지 말고 바로 추천으로 들어가."
    )


#--------[Build ASR Refinement Request]--------#
def build_asr_refine_prompt(user_text: str, response_language: str) -> str:
    language_instruction = "한국어 문장으로만 답해줘." if response_language == "ko" else "Answer only in English."

    return (
        f"ASR raw text: {user_text}\n\n"
        f"{language_instruction} "
        "이 문장은 음성 인식 결과라 오타가 있을 수 있어. "
        "사용자 의도를 보존하면서 자연스럽고 짧은 요청 문장으로 교정해줘. "
        "예를 들어 음식 이미지 맥락에서 '유리'가 어색하면 '요리'처럼 고쳐줘. "
        "설명 없이 교정된 문장 하나만 출력해."
    )


#--------[Build TTS Script Refinement Request]--------#
def build_tts_refine_prompt(answer: str, response_language: str) -> str:
    language_instruction = "한국어로만 말해줘." if response_language == "ko" else "Speak only in English."

    return (
        f"Reasoning result:\n{answer}\n\n"
        f"{language_instruction} "
        "위 내용을 TTS로 읽기 좋은 자연스러운 짧은 대본으로 바꿔줘. "
        "Recommendation, Ingredients, Method 같은 라벨은 읽지 마. "
        "핵심 추천 2개만 부드럽게 이어서 말하고, 너무 길게 설명하지 마. "
        "마크다운, 별표, 번호 목록 없이 3문장 이내로 작성해."
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
    parser.add_argument("--tts-voice", default=None, help="Edge TTS 음성 이름입니다. 생략하면 언어에 맞춰 자동 선택합니다.")
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
    asr_load_started_at = time.perf_counter()
    asr = WhisperASR(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )
    asr_load_elapsed = time.perf_counter() - asr_load_started_at

    asr_transcribe_started_at = time.perf_counter()
    user_text = asr.transcribe(str(audio_path), language=args.language)
    asr_transcribe_elapsed = time.perf_counter() - asr_transcribe_started_at
    asr_elapsed = time.perf_counter() - asr_started_at

    print("   " + "-" * 44)
    print(f"   ASR Text:\n   {user_text}")
    print("   " + "-" * 44)
    print(f"   ASR load       {asr_load_elapsed * 1000:7.0f} ms")
    print(f"   ASR transcribe {asr_transcribe_elapsed * 1000:7.0f} ms")
    print(f"   ASR total      {asr_elapsed * 1000:7.0f} ms")
    response_language = detect_response_language(user_text, args.language)
    selected_tts_voice = select_tts_voice(config, response_language, args.tts_voice)
    gemini = GeminiRecipe(model=args.gemini_model)

    refined_user_text = user_text
    asr_refine_elapsed = 0.0
    if config["asr_refine_enabled"]:
        #--------[Refine ASR Text With LLM]--------#
        asr_refine_started_at = time.perf_counter()
        refined_user_text = gemini.suggest(
            str(image_path),
            build_asr_refine_prompt(user_text, response_language),
        ).replace("*", "").strip()
        asr_refine_elapsed = time.perf_counter() - asr_refine_started_at

        print("   " + "-" * 44)
        print(f"   Refined ASR Text:\n   {refined_user_text}")

    #--------[Analyze Image With Refined Request]--------#
    gemini_started_at = time.perf_counter()
    answer = gemini.suggest(str(image_path), build_agent_prompt(refined_user_text, response_language)).replace("*", "")
    gemini_elapsed = time.perf_counter() - gemini_started_at

    print("   " + "-" * 44)
    print(f"   Answer:\n{answer}")

    tts_script = answer
    tts_refine_elapsed = 0.0
    if args.tts_enabled and config["tts_refine_enabled"]:
        #--------[Refine Reasoning Output For TTS]--------#
        tts_refine_started_at = time.perf_counter()
        tts_script = gemini.suggest(
            str(image_path),
            build_tts_refine_prompt(answer, response_language),
        ).replace("*", "").strip()
        tts_refine_elapsed = time.perf_counter() - tts_refine_started_at

        print("   " + "-" * 44)
        print(f"   TTS Script:\n{tts_script}")

    tts_elapsed = 0.0
    saved_audio_path = None
    if args.tts_enabled:
        #--------[Convert Concise Answer To Speech]--------#
        tts_started_at = time.perf_counter()
        tts_output_path = resolve_project_path(project_root, args.tts_output)
        tts = EdgeTTS(voice=selected_tts_voice, rate=args.tts_rate)
        saved_audio_path = tts.speak(tts_script, str(tts_output_path))
        tts_elapsed = time.perf_counter() - tts_started_at

    print("   " + "-" * 44)
    print(f"   ASR          {asr_elapsed * 1000:7.0f} ms")
    if config["asr_refine_enabled"]:
        print(f"   ASR refine   {asr_refine_elapsed * 1000:7.0f} ms")
    print(f"   Gemini       {gemini_elapsed * 1000:7.0f} ms")
    if args.tts_enabled and config["tts_refine_enabled"]:
        print(f"   TTS refine   {tts_refine_elapsed * 1000:7.0f} ms")
    if saved_audio_path is not None:
        print(f"   TTS          {tts_elapsed * 1000:7.0f} ms")
        print(f"   TTS voice: {selected_tts_voice}")
        print(f"   TTS saved: {saved_audio_path}")


if __name__ == "__main__":
    main()

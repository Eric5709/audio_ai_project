# 맥가이버

음성 질문과 이미지를 함께 분석해서 상황에 맞는 추천을 해주는 멀티모달 에이전트입니다.

```text
음성 질문 + 이미지
   -> OpenAI Whisper ASR로 음성을 텍스트로 변환
   -> Gemini가 이미지와 ASR 텍스트를 함께 해석
   -> 터미널에 보기 좋은 추천과 짧은 방법을 출력
   -> Edge TTS로 같은 추천 내용을 음성으로 저장
```

## 설치

```bash
pip install -r requirements.txt
```

GPU로 Whisper를 실행하려면 현재 conda 환경의 PyTorch가 CUDA를 지원해야 합니다. CUDA가 불안정하면 `config/macgyver_config.yaml`에서 `device`를 `cpu`, `compute_type`을 `float32`로 바꿔 테스트할 수 있습니다.

## 환경 설정

Gemini API 키를 `.env`에 설정합니다.

```bash
cp env.example .env
```

`.env` 파일 안에 본인 키를 넣습니다.

```text
GEMINI_API_KEY=발급받은_키
```

API 키는 코드에 직접 넣거나 커밋하지 않습니다.

## 설정 파일

기본 실행값은 `config/macgyver_config.yaml`에서 관리합니다.

```yaml
audio_path: Source/Audio/english_cook.m4a
image_path: Source/Image/fridge.png
language: null
model_size: large-v3
device: cuda
compute_type: float16
gemini_model: gemini-2.5-flash
tts_enabled: true
tts_output_path: response.mp3
tts_voice: ko-KR-SunHiNeural
tts_rate: +0%
```

한국어 음성을 기본으로 쓰려면 `audio_path`와 `language`를 바꿉니다.

```yaml
audio_path: Source/Audio/korean_cook.m4a
language: ko
```

## 실행

YAML 설정값 그대로 실행합니다.

```bash
python macgyver_agent.py
```

명령어 옵션으로 설정을 임시로 덮어쓸 수도 있습니다.

```bash
python macgyver_agent.py --audio Source/Audio/korean_cook.m4a --language ko
```

이미지나 모델도 실행 시 바꿀 수 있습니다.

```bash
python macgyver_agent.py --image Source/Image/fridge.png --gemini-model gemini-3.5-flash
```

TTS를 끄고 텍스트 결과만 보고 싶으면 `--no-tts`를 사용합니다.

```bash
python macgyver_agent.py --no-tts
```

TTS 출력 파일이나 음성을 바꿀 수도 있습니다.

```bash
python macgyver_agent.py --tts-output response.mp3 --tts-voice ko-KR-SunHiNeural
```

실행 결과는 터미널에 바로 읽기 좋은 텍스트로 출력됩니다. 별표나 JSON 없이 추천 이름, 재료/물건, 짧은 방법 중심으로 나오고, 같은 내용을 TTS 음성 파일로 저장합니다.

## 프로젝트 구조

```text
ASR_Module/asr_module.py        OpenAI Whisper 기반 음성 인식 모듈
LLM_Module/llm_module.py        Gemini 이미지 + 텍스트 해석 모듈
TTS_Module/edge_tts_module.py   Edge TTS 기반 음성 출력 모듈
config/macgyver_config.yaml     기본 실행 설정
macgyver_agent.py               ASR, Gemini, TTS를 연결하는 메인 실행 파일
Source/Audio/                   입력 음성 파일
Source/Image/                   입력 이미지 파일
```

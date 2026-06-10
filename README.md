# 맥가이버

음성 질문과 식재료 이미지를 함께 분석해서 만들 수 있는 요리를 추천하는 멀티모달 에이전트입니다.

```text
음성 질문 + 식재료 이미지
   -> OpenAI Whisper ASR로 음성을 텍스트로 변환
   -> Gemini가 이미지와 ASR 텍스트를 함께 해석
   -> 추천 결과를 터미널에 출력
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
language: en
model_size: large-v3
device: cuda
compute_type: float16
gemini_model: gemini-3.5-flash
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

## 프로젝트 구조

```text
ASR_Module/asr_module.py        OpenAI Whisper 기반 음성 인식 모듈
LLM_Module/llm_module.py        Gemini 이미지 + 텍스트 해석 모듈
config/macgyver_config.yaml     기본 실행 설정
macgyver_agent.py               ASR과 Gemini를 연결하는 메인 실행 파일
Source/Audio/                   입력 음성 파일
Source/Image/                   입력 이미지 파일
```

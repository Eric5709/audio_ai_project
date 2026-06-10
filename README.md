# 맥가이버 🍳🎤

사진 속 재료를 보고 만들 수 있는 모든 걸 알려주는 시스템.

```
🎤 "나 배고파, 이 재료로 뭐 만들 수 있어?" + 📷 사진
   → faster-whisper (음성→텍스트)
   → Gemini Flash (사진 속 식재료 인식 + 레시피 생성, VLM+LLM 한 방)
   → Edge TTS (레시피→음성) 🔊
```


## 설치

```bash
pip install -r requirements.txt
```

## 실행

**0) 환경 설정**
```bash
# (1) Gemini 무료 키 발급: https://aistudio.google.com
# (2) .env 파일 만들기 (키는 코드에 넣지 않는다!)
cp .env.example .env
#    .env 를 열어 GEMINI_API_KEY=발급받은_키 채우기
```

**1) 텍스트 모드** — 음성 대신 텍스트를 사용:
```bash
python run_text.py --image fridge.png --text "what can i cook here"
```

**2) 진짜 모드** — 실제 음성·사진으로 동작:
```bash
input.wav (음성 질문) 와 fridge.jpg (식재료 사진) 준비 후
python main.py --real
```

> ⚠️ **API 키 공유 주의**: `.env` 는 `.gitignore` 에 등록되어 git 에 올라가지 않습니다.
> 키를 코드에 직접 쓰거나 커밋하지 마세요. 팀원은 각자 `.env` 를 만들어 자기 키를 넣습니다.
> 공유되는 건 키가 빈 `.env.example` 견본뿐입니다.

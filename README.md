# 식재료 레시피 음성 도우미 🍳🎤

사진 속 식재료를 보고 만들 수 있는 요리 레시피를 **음성으로** 알려주는 시스템.

```
🎤 "나 배고파, 이 재료로 뭐 만들 수 있어?" + 📷 사진
   → faster-whisper (음성→텍스트)
   → Gemini Flash (사진 속 식재료 인식 + 레시피 생성, VLM+LLM 한 방)
   → Edge TTS (레시피→음성) 🔊
```

전부 **무료** 모델/API로 구성:
| 단계 | 도구 | 비용 |
|------|------|------|
| ASR | faster-whisper (로컬) | 무료·오프라인 |
| VLM+LLM | Google Gemini 2.5 Flash | 무료 (하루 1,500 요청, 카드 불필요) |
| TTS | Microsoft Edge TTS | 무료·무제한·키 불필요 |

## 설치

```bash
pip install -r requirements.txt
```

## 실행

**1) mock 모드** — 키/마이크/네트워크 없이 흐름만 확인:
```bash
python main.py
```

**2) 진짜 모드** — 실제 음성·사진으로 동작:
```bash
# (1) Gemini 무료 키 발급: https://aistudio.google.com
# (2) .env 파일 만들기 (키는 코드에 넣지 않는다!)
cp .env.example .env
#    .env 를 열어 GEMINI_API_KEY=발급받은_키 채우기

# (3) input.wav (음성 질문) 와 fridge.jpg (식재료 사진) 준비 후
python main.py --real
```

> ⚠️ **API 키 공유 주의**: `.env` 는 `.gitignore` 에 등록되어 git 에 올라가지 않습니다.
> 키를 코드에 직접 쓰거나 커밋하지 마세요. 팀원은 각자 `.env` 를 만들어 자기 키를 넣습니다.
> 공유되는 건 키가 빈 `.env.example` 견본뿐입니다.

## 파일 구조
- `pipeline.py` — 인터페이스 + 진짜 구현 3종 (WhisperASR / GeminiRecipe / EdgeTTS)
- `mocks.py` — 가짜 구현 3종 (키·마이크 없이 테스트)
- `main.py` — 오케스트레이터 + 단계별 latency 측정 + 조립부

## 다음 단계 아이디어
- **마이크 실시간 입력**: `sounddevice`로 녹음 → faster-whisper. VAD(발화 끝 감지) 추가.
- **음성 자동 재생**: 생성된 `response.mp3`를 `playsound` 등으로 즉시 재생.
- **다중 턴**: "그럼 더 매운 버전은?" 같은 후속 대화. 대화 히스토리를 Gemini에 누적.
- **GPU 가속**: `WhisperASR(device="cuda", compute_type="float16")`로 ASR latency 단축.

## 한국어 TTS 음성 선택
- `ko-KR-SunHiNeural` (여성, 기본)
- `ko-KR-HyunsuNeural` / `ko-KR-InJoonNeural` (남성)

`main.py`의 `EdgeTTS(voice=...)`에서 변경.

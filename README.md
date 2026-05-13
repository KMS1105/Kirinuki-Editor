# Kirinuki-Editor

## 엔비디아 환경에서 torch 재설치 필요

```bash
python -m pip install torch==2.1.0 torchaudio==2.1.0 torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

## PANNs (Audio Tagging) 모델: 소리 이벤트를 감지하는 데 사용됩니다.

가중치(Cnn14): https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1

클래스 라벨(CSV): http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

---

Ollama (LLM) 관련: 문맥 분석 및 주제 추출에 사용되는 로컬 언어 모델 엔진입니다. (자동)

Ollama 설치 파일: https://ollama.com/download/OllamaSetup.exe

EEVE-Korean 모델: EEVE-Korean-10.8B:latest (Ollama를 통해 pull 명령어로 설치) (자동)

Whisper (STT) 모델: 음성 인식을 담당하며, 코드 실행 시 small 또는 base 버전을 자동으로 로드합니다.

FFmpeg (Essentials Build): 영상 인코딩 및 디코딩의 엔진입니다. (자동)

다운로드 링크: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip

---

## Today's To Do List

상관 관계 프롬프트 개선

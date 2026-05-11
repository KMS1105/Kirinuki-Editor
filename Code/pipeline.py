import os
import sys

# [WinError 1114] 및 CUDA 환경 충돌 방지 핵심 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_THREADING_LAYER'] = 'GNU'  # CUDA와 Intel MKL 간의 스레딩 충돌 방지
os.environ['DNNL_MAX_CPU_ISA'] = 'SSE41' 

# DLL 경로 우선순위 설정
if sys.platform == "win32":
    # 1. 가상환경 라이브러리 경로
    venv_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.exists(venv_bin) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(venv_bin)
    
    # 2. Torch 내부 DLL 경로 (CUDA 환경에서 c10.dll 에러 방지에 효과적)
    torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.exists(torch_lib) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(torch_lib)

import warnings
import subprocess
import shutil
import time
import json
import struct
import urllib.request
import zipfile
from pathlib import Path

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_FILE_DIR if "Kirinuki-Editor-main" in os.path.basename(CURRENT_FILE_DIR) else os.path.dirname(CURRENT_FILE_DIR)

CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
HF_CACHE = os.path.join(CACHE_DIR, "huggingface")
PANNS_DATA = os.path.join(CACHE_DIR, "panns_data")

os.makedirs(HF_CACHE, exist_ok=True)
os.makedirs(PANNS_DATA, exist_ok=True)

os.environ['USERPROFILE'] = PROJECT_ROOT
os.environ['HOME'] = PROJECT_ROOT
os.environ['PANNS_DATA_PATH'] = PANNS_DATA
os.environ['HF_HOME'] = HF_CACHE
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    os.environ["FFMPEG_BINARY"] = ffmpeg_exe
except Exception as e:
    print(f"[경고] FFmpeg 환경 설정 실패: {e}")

import numpy as np
import torch
import requests
import webrtcvad
import torchaudio

csv_path = os.path.join(PANNS_DATA, 'class_labels_indices.csv')
if not os.path.exists(csv_path):
    print(f"[LOG] PANNs 라벨 파일을 다운로드합니다...")
    url = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
    try:
        urllib.request.urlretrieve(url, csv_path)
    except Exception as e:
        print(f"[오류] 라벨 파일 다운로드 실패: {e}")

from panns_inference import AudioTagging, config
config.labels_csv_path = csv_path

from faster_whisper import WhisperModel
from moviepy import VideoFileClip, concatenate_videoclips
from proglog import ProgressBarLogger

try:
    from preset import PresetManager
except ImportError:
    PresetManager = None

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=DeprecationWarning)

print(f"[정보] 시스템 준비 완료 (루트: {PROJECT_ROOT})")
    
class GUIProgressLogger(ProgressBarLogger):
    def __init__(self):
        super().__init__()
        self.last_pct = -1

    def callback(self, **kw):
        if not self.state.get('bars'): return

        bar = list(self.state['bars'].values())[-1]
        
        if bar['total'] > 0:
            pct = int(bar['index'] / bar['total'] * 100)
            
            if pct > self.last_pct:
                print(f"[렌더링 진행] {pct}% 완료...")
                self.last_pct = pct

class StyleAnalyzer:
    def __init__(self, ffmpeg_bin_path=None):
        print("[LOG] StyleAnalyzer 초기화 중...")
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000
        self.pm = PresetManager()
        self.ffmpeg_bin_path = ffmpeg_bin_path
        self.stt_model = None
        self.sed_model = None
        
        print("[LOG] SED 모델 로드 중...")
        
        try:
            self.sed_model = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            print(f"[LOG] SED 로드 실패: {e}")
            self.sed_model = None
            
        self.interest_events = ['Sigh', 'Laughter', 'Gasp', 'Cough', 'Snicker', 'Whispering']
        
    def _get_ffmpeg_path(self):
        if self.ffmpeg_bin_path:
            return os.path.join(self.ffmpeg_bin_path, "ffmpeg.exe")
        return "ffmpeg"

    def _get_audio_data(self, video_path):
        ffmpeg_bin = self._get_ffmpeg_path()
        command = [
            ffmpeg_bin, '-i', video_path,
            '-ar', str(self.sample_rate), '-ac', '1', '-f', 's16le', '-'
        ]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            raw_audio, _ = process.communicate()
            if not raw_audio: return None
            return np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        except:
            return None

    def analyze_user_styles(self, original_paths, edited_paths, preset_name):
        all_thresholds = []
        all_silence_tolerances = []
        all_padding_i = []
        all_padding_o = []

        try:
            if self.stt_model is None:
                print("[LOG] Whisper 모델 로드 중...")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            if device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8" if torch.__version__ >= "2.0" else "float32"

            print(f"[LOG] Whisper 장치: {device}, 연산타입: {compute_type}")
            
            try:
                self.stt_model = WhisperModel("base", device=device, compute_type=compute_type)
            except Exception as e:
                print(f"[LOG] Whisper 로드 재시도 (기본값 사용): {e}")
                self.stt_model = WhisperModel("base", device="cpu", compute_type="int8")

            for orig_path, edit_path in zip(original_paths, edited_paths):
                print(f"[LOG] 대조 분석 시작: {os.path.basename(orig_path)} <-> {os.path.basename(edit_path)}")
                
                orig_audio = self._get_audio_data(orig_path)
                edit_audio = self._get_audio_data(edit_path)
                
                if orig_audio is None or edit_audio is None:
                    print(f"[LOG] 오디오 추출 실패: {edit_path}")
                    continue

                print(f"[LOG] SED 사운드 이벤트 감지 수행 중...")
                query_audio = edit_audio[None, :]
                _, _ = self.sed_model.inference(query_audio)
                
                print(f"[LOG] 원본-편집본 시간 대조(Alignment) 및 컷 지점 추론 중...")
                
                abs_orig = np.abs(orig_audio)
                abs_edit = np.abs(edit_audio)
                
                orig_avg = np.mean(abs_orig)
                speech_parts = abs_edit[abs_edit > (orig_avg * 0.3)]
                user_threshold = np.percentile(speech_parts, 3) if len(speech_parts) > 0 else 0.08
                all_thresholds.append(user_threshold)

                correlation = np.correlate(abs_orig[::1600], abs_edit[::1600], mode='valid')
                offset_idx = np.argmax(correlation)
                print(f"[LOG] 추정 오프셋: {offset_idx / 10.0}초")

                all_silence_tolerances.append(30)
                all_padding_i.append(0.4)
                all_padding_o.append(0.6)

            if not all_thresholds:
                print("[LOG] 유효한 분석 데이터가 없어 프리셋을 생성할 수 없습니다.")
                return None
        
        except Exception as e:
            print(f"[ERROR] 내용: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()

        preset_data = {
            "threshold": round(float(np.mean(all_thresholds)) * 0.85, 5),
            "padding_i": float(np.mean(all_padding_i)),
            "padding_o": float(np.mean(all_padding_o)),
            "max_silence_frames": int(np.mean(all_silence_tolerances)),
            "use_whisper": True,
            "use_sed": True,
            "interest_events": self.interest_events
        }

        self.pm.save_preset(preset_name, preset_data)
        print(f"[LOG] 분석 완료 및 프리셋 저장: {preset_name}")
        print(f"[LOG] 최종 결과 - Thres: {preset_data['threshold']}, SilenceFrames: {preset_data['max_silence_frames']}")
        
        return preset_data

class CutEngine:
    def __init__(self):
        from preset import PresetManager
        import webrtcvad
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, "output_results")
        self.temp_dir = os.path.join(self.base_dir, "temp_vocals")
        self.panns_dir = os.path.join(self.base_dir, "panns_data")
        self.ffmpeg_dir = os.path.join(self.base_dir, "ffmpeg", "bin")
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000
        self.pm = PresetManager()

        if os.path.isdir(self.ffmpeg_dir):
            os.environ["PATH"] = self.ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            os.environ["FFMPEG_BINARY"] = os.path.join(self.ffmpeg_dir, "ffmpeg.exe")

        self.sed_model = None
        self.stt_model = None
        
        self._initialize_panns_resources()
        
        for d in [self.output_dir, self.temp_dir]:
            if not os.path.exists(d): 
                os.makedirs(d)
                
    def _initialize_panns_resources(self):
        labels_csv_path = os.path.join(self.panns_dir, 'class_labels_indices.csv')
        model_path = os.path.join(self.panns_dir, 'Cnn14_mAP=0.431.pth')
        
        if not os.path.isfile(labels_csv_path):
            print(f"[LOG] 클래스 라벨 파일이 없습니다. 다운로드를 시도합니다...")
            url_csv = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
            try:
                urllib.request.urlretrieve(url_csv, labels_csv_path)
                print(f"[LOG] 클래스 라벨 다운로드 완료: {labels_csv_path}")
            except Exception as e:
                print(f"[경고] 라벨 다운로드 실패({e}). 기본 라벨 세트를 생성합니다.")
                with open(labels_csv_path, 'w', encoding='utf-8') as f:
                    f.write("index,mid,display_name\n0,/m/09x0r,Speech\n16,/m/07p6fty,Sigh\n15,/m/026t6,Laughter")
        else:
            print(f"[LOG] 클래스 라벨 파일이 이미 존재합니다: {labels_csv_path}")

        if not os.path.isfile(model_path):
            print(f"[LOG] 모델 가중치 파일이 없습니다. 다운로드를 시도합니다...")
            url_model = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                req = urllib.request.Request(url_model, headers=headers)
                with urllib.request.urlopen(req) as response, open(model_path, 'wb') as f:
                    f.write(response.read())
                print(f"[LOG] 모델 다운로드 완료.")
            except Exception as e:
                print(f"[LOG] 모델 다운로드 실패: {e}. 파일을 {model_path}에 직접 넣어주세요.")
        else:
            print(f"[LOG] PANNs 모델 가중치가 이미 존재합니다: {model_path}")
                
    def activate(self):
        try:
            if self.stt_model is None:
                from faster_whisper import WhisperModel
                print("[LOG] AI 모델(Whisper)을 메모리에 로드합니다...", flush=True)
                try:
                    self.stt_model = WhisperModel("small", device="cpu", compute_type="float32")
                    print("[LOG] AI 모델 로드 완료.", flush=True)
                except Exception as e:
                    print(f"[오류] : {str(e)}", flush=True)
                                
        except Exception as e:
            print(f"[오류] 모델 로드 중 실패: {e}")

    def _ollama_model_id(self):
        return "llama3.2:3b"

    def _ollama_cli_path(self):
        candidates = [
            os.path.join(PROJECT_ROOT, "AppData", "Local", "Programs", "Ollama", "ollama.exe"),
            shutil.which("ollama"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
        ]
        return next((c for c in candidates if c and os.path.exists(c)), None)

    def _ollama_cli_request(self, prompt, timeout=30):
        ollama_bin = self._ollama_cli_path()
        if not ollama_bin:
            return None

        cmd = [
            ollama_bin,
            "run",
            self._ollama_model_id(),
            prompt,
            "--format",
            "json",
            "--nowordwrap",
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode != 0:
                print(f"\n[LOG] Ollama CLI 실패(returncode={result.returncode}): {result.stderr.strip()}")
                return None

            raw = result.stdout.strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                json_start = raw.find("{")
                if json_start >= 0:
                    try:
                        data = json.loads(raw[json_start:])
                    except Exception:
                        print(f"\n[LOG] Ollama CLI JSON 파싱 실패: {raw}")
                        return None
                else:
                    print(f"\n[LOG] Ollama CLI 응답이 JSON이 아닙니다: {raw}")
                    return None

            if isinstance(data, dict):
                return data.get("topic") or data.get("response") or next((v for v in data.values() if isinstance(v, str) and v.strip()), None)
        except Exception as e:
            print(f"\n[LOG] Ollama CLI 호출 실패: {e}")
            return None
        return None

    def _ollama_request(self, prompt, timeout=30):
        url = "http://localhost:11434/api/generate"
        for _ in range(2):
            try:
                res = requests.post(
                    url,
                    json={"model": self._ollama_model_id(), "prompt": prompt, "stream": False},
                    timeout=timeout,
                )
                if res.status_code == 200:
                    return res.json().get("response")
            except Exception:
                time.sleep(1)
                continue

        cli_result = self._ollama_cli_request(prompt, timeout=timeout)
        if cli_result:
            print("\n[LOG] Ollama HTTP 연결 실패 - CLI로 대체합니다.")
            return cli_result

        print("\n[LOG] Ollama 분석 실패: HTTP 및 CLI 모두 사용할 수 없습니다.")
        return None

    def _extract_topic(self, full_text):
        if not full_text:
            return "General"

        prompt = (
            f"당신은 영상 편집을 위한 주제 분석기입니다. 설명이나 서론은 생략하십시오.\n"
            f"태스크: 아래 스크립트에서 비속어나 거친 표현을 포함하여 영상의 핵심 주제를 3줄 내외의 한국어로 요약하세요.\n"
            f"규칙: \n"
            f"1. '[AI 분석 주제] 는...' 같은 설명형 문장을 절대 쓰지 마십시오.\n"
            f"2. 오직 분석된 주제 키워드만 출력하십시오.\n"
            f"3. 비속어가 있다면 그 특징을 살려 '격한 반응', '거친 담화' 등으로 표현하십시오.\n\n"
            f"스크립트: {full_text[:1500]}\n\n"
            f"결과(주제 키워드만):"
        )

        response = self._ollama_request(prompt, timeout=30)
        if response:
            topic = str(response).strip()
            if topic:
                print(f"\n[AI 분석 주제] {topic}")
                return topic

        print("\n[LOG] 주제 분석 스킵 (Ollama 무응답 또는 CLI 실패)")
        return "General"

    def _check_segment_relevance(self, text, topic):
        if topic == "General" or not text.strip():
            return True

        prompt = (
            f"주제: {topic}\n"
            f"문장: {text}\n"
            f"위 문장이 설정된 주제와 맥락상 관련이 있거나, 영상의 재미를 위해 유지해야 하는 구간인가요? "
            f"비속어가 포함되어 있어도 재미있거나 맥락상 필요하다면 유지하세요.\n"
            f"반드시 'Y' 또는 'N'으로만 대답하세요."
        )

        response = self._ollama_request(prompt, timeout=20)
        if response:
            ans = str(response).upper()
            if "N" in ans and "Y" not in ans:
                return False
            return True

        return True
    def _read_wave(self, path):
        import wave
        import audioop

        with wave.open(path, 'rb') as wf:
            nchan = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if nchan > 1:
            frames = audioop.tomono(frames, sampwidth, 1, 1)

        if framerate != self.sample_rate:
            frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, self.sample_rate, None)

        return frames

    def _vad_speech_ratio(self, audio_bytes, start, end):
        import audioop

        sample_bytes = 2
        start_byte = int(start * self.sample_rate) * sample_bytes
        end_byte = int(end * self.sample_rate) * sample_bytes
        seg = audio_bytes[start_byte:end_byte]
        frame_bytes = int(0.03 * self.sample_rate) * sample_bytes
        if len(seg) < frame_bytes:
            return 0.0

        speech_frames = 0
        total_frames = 0
        for offset in range(0, len(seg) - frame_bytes + 1, frame_bytes):
            frame = seg[offset:offset + frame_bytes]
            if self.vad.is_speech(frame, self.sample_rate):
                speech_frames += 1
            total_frames += 1

        return speech_frames / total_frames if total_frames else 0.0

    def _is_silent_segment(self, wav_path, start, end, threshold=0.2):
        try:
            audio_bytes = self._read_wave(wav_path)
            ratio = self._vad_speech_ratio(audio_bytes, start, end)
            return ratio < threshold
        except Exception as e:
            print(f"[LOG] 무음 감지 실패: {e}")
            return False
        
    def _separate_vocals(self, video_path):
        print(f"[1/3] 음원 분리 중 (Demucs)...")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        video_path = os.path.abspath(video_path).replace("\\", "/")
        
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe().replace("\\", "/")
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        except Exception as e:
            print(f"[오류] FFmpeg 위치 파악 실패: {e}")
            return None

        file_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(self.temp_dir, "mdx_extra_q", file_no_ext, "vocals.wav")

        current_env = os.environ.copy()
        current_env["PATH"] = ffmpeg_dir + os.pathsep + current_env["PATH"]
        current_env["TORCHAUDIO_USE_BACKEND"] = "ffmpeg"
        current_env["FFMPEG_BINARY"] = ffmpeg_exe

        cmd = [
            sys.executable, "-u", "-m", "demucs.separate",
            "-n", "mdx_extra_q",
            "-o", self.temp_dir,
            "--two-stems", "vocals",
            video_path
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env=current_env,
        )

        log_lines = []
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            if "Processing" in line or "Separating" in line:
                sys.stdout.write(f"\r[진행] {line.strip()[:65]}...    ")
                sys.stdout.flush()

        proc.wait()
        
        if proc.returncode != 0:
            print(f"\n[오류] 분리 실패 (Code {proc.returncode})")
            if any(ord(c) > 127 for c in video_path):
                print("[알림] 파일 경로에 한글이 포함되어 있습니다. 안전한 폴더로 복사하여 재시도합니다.")
            return None

        return output_path

    def get_keep_intervals(self, vocal_wav_path, preset_data):
        """Whisper 추출 -> Gemma 주제 분석 -> 문맥 필터링"""
        if self.stt_model is None:
            sys.stdout.write("\r[LOG] Whisper AI 모델 로딩 중...")
            sys.stdout.flush()
            self.stt_model = WhisperModel("base", device="cpu", compute_type="int8")
            print("\n[LOG] 모델 로딩 완료.")

        print(f"[2/3] 음성 인식 및 문맥 분석 시작...")
        segments, _ = self.stt_model.transcribe(vocal_wav_path)
        all_segments = list(segments)

        subtitles_dir = os.path.join(self.output_dir, "subtitles")
        os.makedirs(subtitles_dir, exist_ok=True)
        file_no_ext = os.path.basename(os.path.dirname(vocal_wav_path))
        full_subtitles_file = os.path.join(subtitles_dir, f"{file_no_ext}_full_subtitles.txt")
        with open(full_subtitles_file, 'w', encoding='utf-8') as f:
            for i, s in enumerate(all_segments, 1):
                f.write(f"{i}\n{s.start:.2f} --> {s.end:.2f}\n{s.text}\n\n")
        
        if not all_segments:
            print("[경고] 인식된 음성이 없습니다.")
            return []

        full_text = " ".join([s.text for s in all_segments])
        auto_topic = self._extract_topic(full_text)
        
        padding_i = preset_data.get("padding_i", 0.4)
        padding_o = preset_data.get("padding_o", 0.6)
        
        intervals = []
        for i, s in enumerate(all_segments):
            pct = (i + 1) / len(all_segments) * 100
            
            if self._is_silent_segment(vocal_wav_path, s.start, s.end):
                status = "무음 제거"
            elif self._check_segment_relevance(s.text, auto_topic):
                intervals.append((max(0, s.start - padding_i), s.end + padding_o, s.text))
                status = "유지"
            else:
                status = "제거"
            
            sys.stdout.write(f"\r[{pct:3.0f}%] {status}: {s.text[:35]}...    ")
            sys.stdout.flush()
        
        print("\n[LOG] 구간 분석 완료.")
        return self._merge_intervals(intervals)

    def _merge_intervals(self, intervals):
        if not intervals: return []
        intervals.sort()
        merged = []
        curr_s, curr_e, curr_t = intervals[0]
        for n_s, n_e, n_t in intervals[1:]:
            if n_s < curr_e: 
                curr_e = max(curr_e, n_e)
                curr_t += " " + n_t
            else:
                merged.append((curr_s, curr_e, curr_t))
                curr_s, curr_e, curr_t = n_s, n_e, n_t
        merged.append((curr_s, curr_e, curr_t))
        return merged

    def process_video(self, input_path, preset_name):
        """전체 파이프라인 실행 및 최종 렌더링"""
        preset_file = preset_name if preset_name.endswith(".json") else preset_name + ".json"
        preset_data = self.pm.load_preset(preset_file)
   
        vocal_wav = self._separate_vocals(input_path)
        if not vocal_wav or not os.path.exists(vocal_wav):
            print("[오류] 보컬 분리 파일을 찾을 수 없습니다.")
            return

        print(f"[3/3] 최종 영상 렌더링 중...")
        clip = VideoFileClip(input_path)

        kept_segments = self.get_keep_intervals(vocal_wav, preset_data)
        intervals = [(s, e) for s, e, t in kept_segments]
        if not intervals:
            print("[LOG] 유지할 구간이 없습니다. 전체 유지합니다.")
            intervals = [(0, clip.duration)]

        subtitles_dir = os.path.join(self.output_dir, "subtitles")
        os.makedirs(subtitles_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        subtitles_file = os.path.join(subtitles_dir, f"{base_name}_subtitles.txt")
        with open(subtitles_file, 'w', encoding='utf-8') as f:
            for i, (start, end, text) in enumerate(kept_segments, 1):
                f.write(f"{i}\n{start:.2f} --> {end:.2f}\n{text}\n\n")

        try: 
            clips = [clip.subclipped(s, min(clip.duration, e)) for s, e in intervals]
        except AttributeError: 
            clips = [clip.subclip(s, min(clip.duration, e)) for s, e in intervals]
            
        if not clips:
            clip.close()
            return

        final_video = concatenate_videoclips(clips)
        final_path = os.path.join(self.output_dir, f"final_edited_{os.path.basename(input_path)}")
        custom_logger = GUIProgressLogger()
        
        final_video.write_videofile(
            final_path, 
            codec="libx264", 
            audio_codec="aac", 
            fps=clip.fps, 
            logger=custom_logger,
            threads=os.cpu_count(), 
            preset="ultrafast"      
        )
        
        final_video.close()
        clip.close()
        print(f"\n[성공] 작업 완료: {final_path}")
 

from PySide6.QtCore import QObject, Signal, QThread

class OllamaWorker(QObject):
    progress_signal = Signal(str)
    finished_signal = Signal(bool)

    def run(self):
        success = False
        try:
            success = self._bootstrap_ollama()
        except Exception as e:
            self.progress_signal.emit(f"[에러] {e}")
        finally:
            self.finished_signal.emit(success)

    def _bootstrap_ollama(self):
        model_name = "anpigon/eeve-korean-10.8b"
        candidates = [
            os.path.join(PROJECT_ROOT, "AppData", "Local", "Programs", "Ollama", "ollama.exe"),
            shutil.which("ollama"), 
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
        ]
        
        ollama_bin = next((c for c in candidates if c and os.path.exists(c)), None)

        if not ollama_bin:
            self.progress_signal.emit("Ollama 미설치 상태입니다. 자동 설치를 시작합니다...")
            installer_path = os.path.join(PROJECT_ROOT, "OllamaSetup.exe")
            url = "https://ollama.com/download/OllamaSetup.exe"
            try:
                urllib.request.urlretrieve(url, installer_path)
                self.progress_signal.emit("설치 프로그램 실행 중... 잠시만 기다려주세요.")
                subprocess.run([installer_path, "/S"], check=True)
                os.remove(installer_path)
                time.sleep(10)
                ollama_bin = next((c for c in candidates if c and os.path.exists(c)), None)
            except Exception as e:
                self.progress_signal.emit(f"자동 설치 중 오류: {e}")
                return False

        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            models = [m['name'] for m in response.json().get('models', [])]
            
            if not any(model_name in m for m in models):
                self.progress_signal.emit(f"{model_name} 모델이 없어 다운로드를 시작합니다 (약 7.7GB).")
              
                process = subprocess.Popen(
                    [ollama_bin, "pull", model_name], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    encoding="utf-8",     
                    errors="replace",     
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                if process.stdout:
                    for line in process.stdout:
                        if "downloading" in line.lower():
                            self.progress_signal.emit(f"모델 다운로드 중: {line.strip()[-20:]}")
                process.wait()
                
            try:
                requests.get("http://localhost:11434/api/tags", timeout=2)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                self.progress_signal.emit("Ollama 서버를 백그라운드에서 시작합니다...")
                subprocess.Popen(
                    [ollama_bin, "serve"],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(5) 
                    
            self.progress_signal.emit("AI 분석 엔진 준비 완료.")
            return True
    
        except Exception as e:
            self.progress_signal.emit(f"모델 확인 중 오류 발생: {e}")
            return False
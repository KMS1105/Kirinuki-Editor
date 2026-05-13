import csv
import os
import warnings

os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1' 
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HOME'] = os.path.join(os.getcwd(), ".cache", "huggingface")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*bytes wanted but 0 bytes read.*")

import sys
import subprocess
import numpy as np
import torch
import webrtcvad
import urllib.request
from pathlib import Path

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_DIR)
os.environ['USERPROFILE'] = PROJECT_ROOT
os.environ['HOME'] = PROJECT_ROOT
os.environ['HF_HOME'] = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from faster_whisper import WhisperModel
import torchaudio

try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    import torchaudio
    if "ffmpeg" in torchaudio.list_audio_backends():
        torchaudio.set_audio_backend("ffmpeg")
except Exception as e:
    print(f"[경고] FFmpeg 경로 설정 중 오류: {e}")

import struct
import webrtcvad
import json
from moviepy import VideoFileClip, concatenate_videoclips
from panns_inference import AudioTagging
from preset import PresetManager
import zipfile
import time
import shutil
from proglog import ProgressBarLogger
import requests

class ResourceInitializer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.panns_dir = os.path.join(self.base_dir, "panns_data")
        self.temp_dir = os.path.join(self.base_dir, "temp_vocals")
        self.output_dir = os.path.join(self.base_dir, "output_results")
        self.label_map = {}

    def setup_all(self):
        for d in [self.panns_dir, self.temp_dir, self.output_dir]:
            os.makedirs(d, exist_ok=True)

        if not self.prepare_ffmpeg(self.base_dir, log_func=lambda msg: print(f"[LOG] {msg}")):
            print("[경고] FFmpeg를 찾을 수 없습니다. 정상 동작하지 않을 수 있습니다.")

        self._prepare_panns_resources()
        return self

    def _prepare_panns_resources(self):
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
        
        try:
            with open(labels_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 3:
                        self.label_map[int(row[0])] = row[2].strip()
        except Exception as e:
            print(f"[경고] 라벨 CSV 파싱 실패: {e}")
            self.label_map = {0: 'Speech', 15: 'Laughter', 16: 'Sigh'}

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

    def prepare_ffmpeg(self, base_dir, log_func=None):
        ffmpeg_dir = os.path.join(base_dir, "ffmpeg")
        ffmpeg_exe = os.path.join(ffmpeg_dir, "bin", "ffmpeg.exe")

        if os.path.exists(ffmpeg_exe):
            try:
                result = subprocess.run(
                    [ffmpeg_exe, "-version"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=5
                )
                if "ffmpeg version" in result.stdout:
                    if log_func: log_func("FFmpeg is already installed and verified.")
                    return True
                else:
                    shutil.rmtree(ffmpeg_dir, ignore_errors=True)
            except Exception:
                shutil.rmtree(ffmpeg_dir, ignore_errors=True)

        zip_path = os.path.join(base_dir, "ffmpeg.zip")
        tmp_zip = zip_path + ".tmp"
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        
        try:
            if log_func: log_func("FFmpeg downloading (Essentials)...")
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            if os.path.exists(zip_path): os.remove(zip_path)
            if os.path.exists(tmp_zip): os.remove(tmp_zip)

            urllib.request.urlretrieve(url, tmp_zip)
            os.rename(tmp_zip, zip_path)

            if log_func: log_func("Extracting FFmpeg...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                if os.path.exists(ffmpeg_dir):
                    shutil.rmtree(ffmpeg_dir, ignore_errors=True)
                zip_ref.extractall(base_dir)
                
                extracted_folder = [f for f in os.listdir(base_dir) if f.startswith("ffmpeg-") and os.path.isdir(os.path.join(base_dir, f))]
                if extracted_folder:
                    source_path = os.path.join(base_dir, extracted_folder[0])
                    os.rename(source_path, ffmpeg_dir)

            if os.path.exists(zip_path):
                try: os.remove(zip_path)
                except: pass
                
            if os.path.exists(ffmpeg_exe):
                final_check = subprocess.run([ffmpeg_exe, "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW, timeout=5)
                if "ffmpeg version" in final_check.stdout:
                    if log_func: log_func("FFmpeg is ready and verified.")
                    time.sleep(0.2) 
                    return True
            return False
        except Exception as e:
            if log_func: log_func(f"FFmpeg error: {str(e)}")
            if os.path.exists(tmp_zip):
                try: os.remove(tmp_zip)
                except: pass
            return False
    
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
                print(f"\r[렌더링 진행] {pct}% 완료...")
                self.last_pct = pct
                
class StyleAnalyzer:
    def __init__(self, ffmpeg_bin_path=None):
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000
        self.pm = PresetManager()
        self.ffmpeg_bin_path = ffmpeg_bin_path
        self.stt_model = None
        try:
            self.sed_model = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')
        except:
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
        label_map = {0: 'Speech', 15: 'Laughter', 16: 'Sigh', 17: 'Snicker', 18: 'Gasp', 22: 'Cough', 25: 'Whispering'}
        collected_metrics = {
            event: {"thresholds": [], "paddings_i": [], "paddings_o": []} 
            for event in self.interest_events
        }
        collected_metrics["Default"] = {"thresholds": [], "paddings_i": [], "paddings_o": []}

        if self.stt_model is None:
            print("[LOG] Whisper 모델 로드 중...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            try:
                self.stt_model = WhisperModel("base", device=device, compute_type=compute_type)
            except Exception as e:
                self.stt_model = WhisperModel("base", device="cpu", compute_type="int8")

        for orig_path, edit_path in zip(original_paths, edited_paths):
            print(f"[LOG] 대조 스타일 분석 시작: {os.path.basename(orig_path)} <-> {os.path.basename(edit_path)}")
            
            orig_audio = self._get_audio_data(orig_path)
            edit_audio = self._get_audio_data(edit_path)
            if orig_audio is None or edit_audio is None:
                continue

            correlation = np.correlate(orig_audio[::1600], edit_audio[::1600], mode='valid')
            offset_seconds = np.argmax(correlation) / 10.0
            
            segments, _ = self.stt_model.transcribe(orig_path)
            all_segments = list(segments)
            
            for s in all_segments:
                s_start_idx = int(s.start * self.sample_rate)
                s_end_idx = int(s.end * self.sample_rate)
                orig_segment_audio = orig_audio[s_start_idx:s_end_idx]
                
                detected_event = "Default"
                if self.sed_model is not None and len(orig_segment_audio) > int(0.2 * self.sample_rate):
                    try:
                        clip_output, _ = self.sed_model.inference(orig_segment_audio[None, :])
                        top_indices = np.argsort(clip_output[0])[::-1][:5]
                        for idx in top_indices:
                            ev_name = label_map.get(idx, "Unknown")
                            if ev_name in self.interest_events and clip_output[0][idx] > 0.15:
                                detected_event = ev_name
                                break 
                    except:
                        pass

                mapped_start = s.start - offset_seconds
                mapped_end = s.end - offset_seconds
                
                search_margin = 1.5 
                search_start = max(0, mapped_start - search_margin)
                search_end = min(len(edit_audio) / self.sample_rate, mapped_end + search_margin)
                
                search_start_idx = int(search_start * self.sample_rate)
                search_end_idx = int(search_end * self.sample_rate)
                edit_search_zone = edit_audio[search_start_idx:search_end_idx]
                
                if len(edit_search_zone) > int(0.1 * self.sample_rate):
                    rms_profile = np.array([
                        np.sqrt(np.mean(edit_search_zone[max(0, j-160):j+160]**2)) 
                        for j in range(0, len(edit_search_zone), 160)
                    ])
                    
                    noise_floor = np.percentile(rms_profile, 15) if len(rms_profile) > 0 else 0.001
                    active_indices = np.where(rms_profile > (noise_floor * 2.0))[0]
                    
                    if len(active_indices) > 0:
                        actual_start_sec = search_start + (active_indices[0] * 160 / self.sample_rate)
                        actual_end_sec = search_start + (active_indices[-1] * 160 / self.sample_rate)
                        
                        calc_padding_i = max(0.05, round(mapped_start - actual_start_sec, 2))
                        calc_padding_o = max(0.05, round(actual_end_sec - mapped_end, 2))
                        
                        if calc_padding_i > search_margin: calc_padding_i = 0.4
                        if calc_padding_o > search_margin: calc_padding_o = 0.6
                        
                        actual_speech_signal = edit_search_zone[int(max(0, actual_start_sec - search_start)*self.sample_rate):int(min(len(edit_search_zone)/self.sample_rate, actual_end_sec - search_start)*self.sample_rate)]
                        if len(actual_speech_signal) > 0:
                            calc_threshold = float(np.percentile(np.abs(actual_speech_signal), 5))
                            if calc_threshold < 0.0005: calc_threshold = 0.00427
                        else:
                            calc_threshold = 0.00427
                        
                        collected_metrics[detected_event]["thresholds"].append(calc_threshold)
                        collected_metrics[detected_event]["paddings_i"].append(calc_padding_i)
                        collected_metrics[detected_event]["paddings_o"].append(calc_padding_o)

        preset_data = {
            "max_silence_frames": 30,
            "use_sed": True,
            "use_whisper": True,
            "interest_events": self.interest_events,
            "event_specific_settings": {}
        }

        for event in self.interest_events:
            metrics = collected_metrics[event]
            if metrics["thresholds"]: 
                preset_data["event_specific_settings"][event] = {
                    "threshold": round(float(np.mean(metrics["thresholds"])), 5),
                    "padding_i": round(float(np.mean(metrics["paddings_i"])), 2),
                    "padding_o": round(float(np.mean(metrics["paddings_o"])), 2)
                }
            else: 
                seed_map = {
                    "Cough": (0.35, 0.55, 0.0035), "Gasp": (0.25, 0.45, 0.0028),
                    "Laughter": (0.50, 0.70, 0.0052), "Sigh": (0.30, 0.50, 0.0031),
                    "Snicker": (0.40, 0.60, 0.0040), "Whispering": (0.20, 0.40, 0.0022)
                }
                pi, po, th = seed_map.get(event, (0.4, 0.6, 0.00427))
                preset_data["event_specific_settings"][event] = {
                    "threshold": th, "padding_i": pi, "padding_o": po
                }

        default_metrics = collected_metrics["Default"]
        preset_data["threshold"] = round(float(np.mean(default_metrics["thresholds"])), 5) if default_metrics["thresholds"] else 0.00427
        preset_data["padding_i"] = round(float(np.mean(default_metrics["paddings_i"])), 2) if default_metrics["paddings_i"] else 0.4
        preset_data["padding_o"] = round(float(np.mean(default_metrics["paddings_o"])), 2) if default_metrics["paddings_o"] else 0.6

        self.pm.save_preset(preset_name, preset_data)
        print(f"[LOG] 사운드 분류 학습 기반 맞춤형 프리셋 생성 완료: {preset_name}")
        return preset_data

class CutEngine:
    def __init__(self, initializer=None):
        from preset import PresetManager
        import webrtcvad
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.initializer = initializer
        
        if self.initializer:
            self.output_dir = self.initializer.output_dir
            self.temp_dir = self.initializer.temp_dir
            self.panns_dir = self.initializer.panns_dir
        else:
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

    def activate(self):
        try:
            if self.stt_model is None:
                from faster_whisper import WhisperModel
                print("[LOG] Whisper 분리 전사 엔진 초기화...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                compute_type = "float16" if device == "cuda" else "int8"
                self.stt_model = WhisperModel("small", device=device, compute_type=compute_type)
            
            if self.sed_model is None:
                print("[LOG] PANNs 오디오 감지 엔진 활성화...")
                self.sed_model = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            print(f"[오류] 엔진 가동 Fallback 스위칭: {e}")
            if self.stt_model is None:
                self.stt_model = WhisperModel("small", device="cpu", compute_type="int8")

    def _ollama_model_id(self):
        return "anpigon/eeve-korean-10.8b"

    def _ollama_cli_path(self):
        candidates = [
            os.path.join(PROJECT_ROOT, "AppData", "Local", "Programs", "Ollama", "ollama.exe"),
            shutil.which("ollama"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
        ]
        return next((c for c in candidates if c and os.path.exists(c)), None)

    def _ollama_request(self, prompt, timeout=30):
        url = "http://localhost:11434/api/generate"
        for _ in range(2):
            try:
                res = requests.post(
                    url,
                    json={
                        "model": self._ollama_model_id(),
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "top_p": 0.3, "num_predict": 2, "num_ctx": 1024}
                    },
                    timeout=timeout,
                )
                if res.status_code == 200:
                    return res.json().get("response")
            except Exception:
                time.sleep(1)
                continue
        return None

    def _extract_topic(self, full_text):
        if not full_text: return "General"
        prompt = (
            f"영상의 스크립트를 분석하여 핵심 키워드 10~20개를 추출하세요.\n"
            f"규칙:\n"
            f"1. 설명이나 인사말 없이 오직 키워드만 쉼표(,)로 구분하여 나열하십시오.\n"
            f"2. 주제, 등장인물, 주요 사건, 감정 상태를 모두 포함하세요.\n"
            f"3. 비속어나 거친 표현이 있다면 순화하지 말고 그 분위기를 살린 키워드로 만드세요.\n"
            f"4. 반드시 10개 이상의 키워드를 생성하세요.\n\n"
            f"스크립트: {full_text[:1500]}\n\n"
            f"결과(키워드만):"

        )
        response = self._ollama_request(prompt, timeout=300)
        return str(response).strip() if response else "General"

    def _check_segment_relevance_with_sound(self, text, topic, sound_event, previous_events):
        if topic == "General" or not text.strip(): return True
        
        repeat_context = "없음"
        if len(previous_events) >= 2 and all(ev == sound_event for ev in previous_events[-2:]) and sound_event in ['Sigh', 'Cough']:
            repeat_context = f"현재 '{sound_event}' 사운드가 연속 3회 이상 중복 발생하여 흐름이 완전히 끊겼습니다."

        clean_topic = str(topic).strip()

        prompt = (
            f"당신은 동영상 컷편집 여부를 결정하는 맥락 및 사운드 분석 AI 시스템입니다.\n\n"
            f"[판단 기준]\n"
            f"- 입력된 [대상 문장] 내용이 [주제 키워드]에 속한 단어들과 조금이라도 관련이 있다면 반드시 'Y'입니다.\n"
            f"- [대상 문장]에 키워드가 직접 없더라도 문맥상 자연스러운 대화 흐름이거나, 감정 표현, 리액션(귀엽다, 쳐다보다 등)을 담고 있다면 무조건 'Y'입니다.\n"
            f"- [감지된 사운드]가 웃음소리('Laughter'), 속삭임('Whispering') 등의 유의미한 리액션 사운드이거나 감정을 풍부하게 해주는 소리인 경우에도 무조건 'Y'입니다.\n"
            f"- 오직 주제와 완전히 무관한 소음, 오인식된 엉뚱한 단어, 무의미한 웅얼거림('어..', '음..')이거나, 문장 내용이 없는 상태에서 'Sigh(한숨)'나 'Cough(기침)'가 연속 누적(특이 사항 참조)되어 흐름을 방해하는 경우에만 'N'입니다.\n\n"
            f"[참고 메커니즘]\n"
            f"- 키워드에 '귀엽다, 아기'가 있고 대상 문장이 '귀엽다 이랬더니' -> 주제 범위 내이므로 결과는 'Y'\n"
            f"- 키워드에 '유모차, 아기'가 있고 대상 문장이 '유모차에 애기가 많기 있어' -> 맥락이 일치하므로 결과는 'Y'\n"
            f"- 키워드에 관련 정보가 있고 대상 문장이 흐릿하지만 감지된 사운드가 'Laughter(웃음)'인 경우 -> 유의미한 감정 표현이므로 결과는 'Y'\n\n"
            f"--- 실제 분석할 데이터 ---\n"
            f"[주제 키워드]: {clean_topic}\n"
            f"[대상 문장]: \"{text}\"\n"
            f"[감지된 사운드]: {sound_event}\n"
            f"[특이 사항 연속 노이즈 컨텍스트]: {repeat_context}\n"
            f"---------------------------\n\n"
            f"[절대 규칙] 다른 설명이나 부가적인 말은 일절 금지하며 오직 대문자 'Y' 또는 'N' 중 딱 한 글자만 출력하세요.\n"
            f"결과: "
        )
        
        response = self._ollama_request(prompt, timeout=40)
        return False if response and "N" in str(response).upper() else True
    
    def _is_silent_segment(self, wav_path, start, end, threshold=0.00427):
        try:
            import torchaudio
            waveform, sr = torchaudio.load(wav_path)
            start_frame = int(start * sr)
            end_frame = int(end * sr)
            segment = waveform[:, start_frame:end_frame]
            if segment.numel() == 0: return True
            rms = torch.sqrt(torch.mean(segment ** 2)).item()
            return rms < threshold
        except Exception as e:
            print(f"[LOG] 가변 임계값 검사 예외: {e}")
            return False

    def _separate_vocals(self, video_path):
        print(f"[1/3] 음원 분리 중 (Demucs)...")
        os.makedirs(self.temp_dir, exist_ok=True)
        file_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(self.temp_dir, "mdx_extra_q", file_no_ext, "vocals.wav")
        cmd = [sys.executable, "-u", "-m", "demucs.separate", "-n", "mdx_extra_q", "-o", self.temp_dir, "--two-stems", "vocals", video_path]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW)
        
        for line in proc.stdout:
            if "Processing" in line or "Separating" in line:
                sys.stdout.write(f"\r[진행] {line.strip()[:65]}...    ")
                sys.stdout.flush()
        proc.wait()
        return output_path if os.path.exists(output_path) else None

    def get_keep_intervals(self, vocal_wav_path, preset_data):
        self.activate()
        print(f"[2/3] 구간 분석 및 매칭 루프 가동...")
        
        import torchaudio
        audio_data = None
        try:
            waveform, sr = torchaudio.load(vocal_wav_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            audio_data = waveform.mean(dim=0).numpy()
        except Exception as e:
            print(f"[경고] SED 전처리 버퍼 변환 실패: {e}")

        segments, _ = self.stt_model.transcribe(vocal_wav_path)
        all_segments = list(segments)
        if not all_segments: return []

        full_text = " ".join([s.text for s in all_segments])
        auto_topic = self._extract_topic(full_text)
        is_ollama_active = (auto_topic != "General")

        event_configs = preset_data.get("event_specific_settings", {})
        interest_events = preset_data.get("interest_events", [])
        label_map = self.initializer.label_map if self.initializer else {0: 'Speech', 15: 'Laughter', 16: 'Sigh'}
        
        intervals = []
        log_records = []
        historical_detected_events = [] 
        
        for i, s in enumerate(all_segments):
            pct = (i + 1) / len(all_segments) * 100
            
            detected_event_name = "None"
            is_event_detected = False

            if audio_data is not None and self.sed_model is not None:
                start_idx = int(s.start * self.sample_rate)
                end_idx = int(s.end * self.sample_rate)
                segment_audio = audio_data[start_idx:end_idx]
                
                if len(segment_audio) > int(0.2 * self.sample_rate):
                    try:
                        clip_output, _ = self.sed_model.inference(segment_audio[None, :])
                        top_indices = np.argsort(clip_output[0])[::-1][:5]
                        
                        for idx in top_indices:
                            event_name = label_map.get(idx, "Unknown")
                            if event_name in interest_events and clip_output[0][idx] > 0.15:
                                is_event_detected = True
                                detected_event_name = event_name
                                break
                    except:
                        pass

            historical_detected_events.append(detected_event_name)

            if is_ollama_active:
                is_relevant = self._check_segment_relevance_with_sound(
                    s.text, auto_topic, detected_event_name, historical_detected_events
                )
            else:
                is_relevant = True

            p_i = preset_data.get("padding_i", 0.4)
            p_o = preset_data.get("padding_o", 0.6)
            v_threshold = preset_data.get("threshold", 0.00427)

            if detected_event_name in event_configs:
                p_i = event_configs[detected_event_name].get("padding_i", p_i)
                p_o = event_configs[detected_event_name].get("padding_o", p_o)
                v_threshold = event_configs[detected_event_name].get("threshold", v_threshold)

            is_silent = self._is_silent_segment(vocal_wav_path, s.start, s.end, threshold=v_threshold)
            
            if is_silent and not is_event_detected:
                status = "무음 제거"
            elif not is_relevant:
                status = "AI 판단 제거"
            else:
                intervals.append((max(0, s.start - p_i), s.end + p_o, s.text))
                status = f"유지({detected_event_name})" if is_event_detected else "유지"
            
            log_records.append((i, pct, status, s.text))
            sys.stdout.write(f"\r[분석 진행 중] {pct:3.0f}% 완료...")
            sys.stdout.flush()
        
        print("\n\n--- [사운드-문맥 융합 분석 결과 목록] ---")
        for _, pct, status, text in log_records:
            print(f"[{pct:3.0f}%] {status}: {text[:45]}...")
            
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
        preset_file = preset_name if preset_name.endswith(".json") else preset_name + ".json"
        preset_data = self.pm.load_preset(preset_file)
        vocal_wav = self._separate_vocals(input_path)
        if not vocal_wav: return

        clip = VideoFileClip(input_path)
        kept_segments = self.get_keep_intervals(vocal_wav, preset_data)
        intervals = [(s, e) for s, e, t in kept_segments]
        if not intervals: intervals = [(0, clip.duration)]

        clips = []
        for s, e in intervals:
            end_t = min(clip.duration, e)
            if hasattr(clip, "subclipped"):
                clips.append(clip.subclipped(s, end_t)) 
            else:
                clips.append(clip.subclip(s, end_t))     
        if not clips: return

        final_video = concatenate_videoclips(clips)
        final_path = os.path.join(self.output_dir, f"final_edited_{os.path.basename(input_path)}")
        final_video.write_videofile(final_path, codec="libx264", audio_codec="aac", fps=clip.fps, logger=GUIProgressLogger(), threads=os.cpu_count(), preset="ultrafast")
        final_video.close()
        clip.close()
        print(f"\n[성공] 작업 완료: {final_path}")
 

from PySide6.QtCore import QObject, Signal, QThread

class OllamaWorker(QObject):
    progress_signal = Signal(str)
    finished_signal = Signal(bool)

    def run(self):
        success = False
        self.ollama_bootstrapped = False
        
        try:
            success = self._bootstrap_ollama()
        except Exception as e:
            self.progress_signal.emit(f"[에러] {e}")
        finally:
            self.finished_signal.emit(success)

    def _wait_for_server(self, timeout=30):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(2)
        return False

    def _bootstrap_ollama(self):
        if getattr(self, 'ollama_bootstrapped', False):
            try:
                if requests.get("http://localhost:11434/api/tags", timeout=0.5).status_code == 200:
                    return True
            except:
                print("[Ollama 재연결] 백그라운드 서버 세션 재확보를 시도합니다...")
                
        model_name = "anpigon/eeve-korean-10.8b"
        fallback_model = "teddylee777/llama-3-korean-8b-instruct"
        
        candidates = [
            shutil.which("ollama"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
            os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Programs", "Ollama", "ollama.exe"),
            "C:\\Program Files\\Ollama\\ollama.exe"
        ]
        ollama_bin = next((c for c in candidates if c and os.path.exists(c)), None)
        
        print("\n[Ollama 자원 점검] 백그라운드 프로세스 세션을 최적화합니다...")

        if not ollama_bin:
            self.progress_signal.emit("Ollama를 찾을 수 없어 설치를 시도합니다...")
            installer_path = os.path.join(PROJECT_ROOT, "OllamaSetup.exe")
            url = "https://ollama.com/download/OllamaSetup.exe"
            try:
                urllib.request.urlretrieve(url, installer_path)
                subprocess.run([installer_path, "/S"], check=True)
                os.remove(installer_path)
                for _ in range(15):
                    time.sleep(2)
                    ollama_bin = next((c for c in candidates if c and os.path.exists(c)), None)
                    if ollama_bin: break
            except Exception as e:
                self.progress_signal.emit(f"자동 설치 실패: {e}")
                return False

        subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        time.sleep(1.2)

        ollama_log_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Ollama")
        os.makedirs(ollama_log_dir, exist_ok=True)
        ollama_log_file = os.path.join(ollama_log_dir, "server.log")
        if os.path.exists(ollama_log_file):
            try: os.remove(ollama_log_file)
            except: pass

        env = os.environ.copy()
        env['OLLAMA_NUM_PARALLEL'] = '1'
        env['OLLAMA_NOPRUNE'] = '1'
        env['CUDA_MODULE_LOADING'] = 'LAZY'
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

        self.progress_signal.emit("Ollama 서버를 시작합니다...")
        with open(ollama_log_file, "w", encoding="utf-8", errors="replace") as log_f:
            subprocess.Popen(
                [ollama_bin, "serve"],
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW,
                stdout=log_f,
                stderr=log_f
            )
        
        server_ready = False
        for _ in range(20):
            try:
                if requests.get("http://localhost:11434/api/tags", timeout=1.5).status_code == 200:
                    server_ready = True
                    break
            except:
                time.sleep(1.0)

        if not server_ready:
            if not self._wait_for_server(timeout=10):
                print("[Ollama 부트스트랩 장치 확인] 구동 장치: CPU")
                self.progress_signal.emit("AI 분석 엔진 초기화 완료 (CPU)")
                return False

        def ensure_model_pulled(target_model):
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=10)
                models = [m['name'] for m in response.json().get('models', [])]
                
                if not any(target_model in m for m in models):
                    self.progress_signal.emit(f"모델 다운로드 중 ({target_model})...")
                    print(f"[Ollama 초기화] 시스템에 {target_model} 모델이 없어 다운로드를 시작합니다.")
                    process = subprocess.Popen(
                        [ollama_bin, "pull", target_model], 
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                        text=True, encoding="utf-8", errors="replace",
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    if process.stdout:
                        for line in process.stdout:
                            if "downloading" in line.lower():
                                clean_line = line.strip().split('\r')[-1]
                                if "%" in clean_line:
                                    self.progress_signal.emit(f"다운로드 {clean_line[-20:]}")
                    process.wait()
                return True
            except Exception as e:
                self.progress_signal.emit(f"모델 확인 중 오류: {e}")
                return False

        if not ensure_model_pulled(model_name):
            return False

        self.progress_signal.emit("LLM 모델을 외장 GPU VRAM에 로드하는 중입니다...")
        use_fallback = False

        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name, 
                    "prompt": "hi", 
                    "stream": False, 
                    "keep_alive": -1,
                    "options": {"num_predict": 1}
                },
                timeout=25 
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout):
            print(f"[Ollama 타임아웃 감지] 기본 모델 연산이 25초를 초과하여 경량 가속 모델({fallback_model})로 전환합니다.")
            self.progress_signal.emit("VRAM 용량 한계 감지: 경량 가속 모델로 전환 및 다운로드 시작...")
            use_fallback = True
        except Exception as e:
            print(f"[Ollama 부트스트랩] 웜업 중 예외 발생: {e}")

        if use_fallback:
            if not ensure_model_pulled(fallback_model):
                return False
                
            self._ollama_model_id = lambda: fallback_model
            self.progress_signal.emit("경량 모델을 외장 GPU VRAM에 다시 로드 중입니다...")
            
            try:
                requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": fallback_model, 
                        "prompt": "hi", 
                        "stream": False, 
                        "keep_alive": -1,
                        "options": {"num_predict": 1}
                    },
                    timeout=25 
                )
            except Exception as e:
                print(f"[Ollama 부트스트랩] 대체 모델 웜업 중 예외 발생: {e}")

        final_device = "CPU"
        try:
            time.sleep(1.5)
            ps_res = requests.get("http://localhost:11434/api/ps", timeout=3)
            if ps_res.status_code == 200:
                models_active = ps_res.json().get("models", [])
                if models_active:
                    processor = models_active[0].get("processor", "Unknown")
                    if "GPU" in processor or "CUDA" in processor or "ROCm" in processor:
                        final_device = "GPU"
                        
            if final_device == "CPU" and os.path.exists(ollama_log_file):
                with open(ollama_log_file, "r", encoding="utf-8", errors="replace") as f:
                    log_content = f.read()
                    if "cuda" in log_content.lower() or "gpu" in log_content.lower() or "llm_load_tensors: offloaded" in log_content.lower():
                        final_device = "GPU"
        except:
            pass

        print(f"[Ollama 부트스트랩 장치 확인] 구동 장치: {final_device}")
        self.progress_signal.emit(f"AI 분석 엔진 초기화 완료 ({final_device})")
        self.ollama_bootstrapped = True

        return True
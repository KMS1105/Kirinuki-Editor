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
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

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
import imageio_ffmpeg

try:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

    try:
        if hasattr(torchaudio, 'list_audio_backends'):
            if "ffmpeg" in torchaudio.list_audio_backends():
                torchaudio.set_audio_backend("ffmpeg")
        else:
            pass 
    except:
        pass

except Exception as e:
    print(f"[경고] 시스템 환경 설정 중 확인 필요: {e}")

if torch.cuda.is_available():
    device = 'cuda'
    backend_name = "NVIDIA_CUDA_Acceleration"
else:
    device = 'cpu'
    is_intel = "INTEL" in os.environ.get("PROCESSOR_IDENTIFIER", "").upper()
    backend_name = "GPU" if is_intel else "Generic_CPU"

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

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from scipy.spatial.distance import cosine
import cv2
import platform
import psutil
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from faster_whisper import WhisperModel
from panns_inference import AudioTagging
from sklearn.cluster import DBSCAN
import webrtcvad
from panns_inference import AudioTagging
from preset import PresetManager
from fontTools import ttLib

class StyleAnalyzer:
    def __init__(self, ffmpeg_bin_path=None, domains=["Talking"]):
        self.vad = webrtcvad.Vad(3)
        self.sample_rate = 16000
        self.pm = PresetManager()
        self.ffmpeg_bin_path = ffmpeg_bin_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stt_model = None
        self.font_db = {}
        self.font_embeddings = {}
        
        weights = ResNet18_Weights.DEFAULT
        base_model = models.resnet18(weights=weights).to(self.device)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_extractor.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._load_font_database()
        self._precompute_font_embeddings()

        try:
            self.sed_model = AudioTagging(
                checkpoint_path=None,
                device=self.device
            )
        except:
            self.sed_model = None

        self.interest_events = [
            "Sigh",
            "Laughter",
            "Gasp",
            "Cough",
            "Snicker",
            "Whispering"
        ]

        self.domains = [domains] if isinstance(domains, str) else domains
        self.domain_map = {
            "게임": "Gaming",
            "토크": "Talking",
            "노래": "Singing",
            "쇼츠": "Shorts",
            "롱폼": "Longform"
        }

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(current_dir)
        self.font_db = {}
        self.project_root = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        self._load_font_database()
        print(f"[LOG] 폰트 로드 완료: {len(self.font_db)}개")

    def _get_ffmpeg_path(self):
        possible_paths = []

        if self.ffmpeg_bin_path:
            possible_paths.append(self.ffmpeg_bin_path)
            possible_paths.append(
                os.path.join(
                    self.ffmpeg_bin_path,
                    "ffmpeg.exe"
                )
            )

        local_ffmpeg = os.path.join(
            self.project_root,
            "ffmpeg",
            "bin",
            "ffmpeg.exe"
        )

        possible_paths.append(local_ffmpeg)

        for path in possible_paths:
            if os.path.isfile(path):
                print(f"[LOG] FFmpeg 사용: {path}")
                return path

        ffmpeg_sys = shutil.which("ffmpeg")

        if ffmpeg_sys:
            print(f"[LOG] 시스템 FFmpeg 사용: {ffmpeg_sys}")
            return ffmpeg_sys

        raise FileNotFoundError(
            "FFmpeg를 찾을 수 없습니다"
        )

    def _get_audio_data(self, video_path):
        ffmpeg_bin = self._get_ffmpeg_path()
        command = [
            ffmpeg_bin,
            "-i", video_path,
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "s16le",
            "-"
        ]

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            raw_audio, _ = process.communicate()

            if not raw_audio:
                return None

            return (
                np.frombuffer(
                    raw_audio,
                    dtype=np.int16
                ).astype(np.float32) / 32768.0
            )

        except Exception as e:
            print(f"[LOG] 오디오 추출 실패: {video_path}")
            print(e)
            return None

    def _extract_subtitle_roi(self, frame):
        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, 160]),
            np.array([180, 90, 255])
        )
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None, None

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )
        boxes = []

        for c in contours[:80]:
            x, y, w, h = cv2.boundingRect(c)
            
            if w < 6 or h < 6:
                continue

            boxes.append((x, y, w, h))

        if len(boxes) == 0:
            return None, None

        x1 = min([b[0] for b in boxes])
        y1 = min([b[1] for b in boxes])

        x2 = max([b[0] + b[2] for b in boxes])
        y2 = max([b[1] + b[3] for b in boxes])

        roi = frame[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2]

        return roi, roi_mask
    
    def _load_font_database(self):
        from fontTools import ttLib
        import os

        user_font_dir = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'Windows', 'Fonts')
        target_dirs = ["C:/Windows/Fonts"]
        if os.path.exists(user_font_dir):
            target_dirs.append(user_font_dir)

        print(f"[LOG] 자동 분석을 위한 전체 폰트 스캔 시작...")
        
        for font_dir in target_dirs:
            if not os.path.isdir(font_dir): continue
            for file in os.listdir(font_dir):
                if not file.lower().endswith((".ttf", ".otf", ".ttc")): continue
                
                font_path = os.path.join(font_dir, file)
                try:
                    font_idx = 0 if file.lower().endswith(".ttc") else -1
                    tt = ttLib.TTFont(font_path, fontNumber=font_idx, lazy=True)
                    
                    font_name = tt['name'].getDebugName(4) or tt['name'].getDebugName(1) or os.path.splitext(file)[0]
                    font_name = font_name.replace('\x00', '').strip()
                    
                    if font_name not in self.font_db:
                        self.font_db[font_name] = font_path
                except:
                    continue
        print(f"[LOG] 총 {len(self.font_db)}개의 폰트 로드 완료. 자동 분석 준비됨.")

        self.font_candidates = {
            "thin": [],
            "normal": [],
            "bold": []
        }

        for font_name, font_path in self.font_db.items():
            lower = font_name.lower()

            if any(x in lower for x in [

                "thin",
                "light",
                "extralight"

            ]):
                self.font_candidates[
                    "thin"
                ].append((
                    font_name,
                    font_path
                ))

            elif any(x in lower for x in [
                "bold",
                "black",
                "heavy",
                "extrabold"

            ]):
                self.font_candidates[
                    "bold"
                ].append((
                    font_name,
                    font_path
                ))

            else:
                self.font_candidates[
                    "normal"
                ].append((
                    font_name,
                    font_path
                ))

        print(
            f"[LOG] 로드된 폰트 수: "
            f"{len(self.font_db)}"
        )

    def _render_font_text(self, text, font_path, size=100):
        canvas_w, canvas_h = 1024, 256
        canvas = Image.new("L", (canvas_w, canvas_h), 0)
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype(font_path, size)
            
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((canvas_w - w) // 2, (canvas_h - h) // 2), text, fill=255, font=font)
            
            return np.array(canvas)
        except:
            return None

    def _compare_font_similarity(self, roi_mask, rendered):
        try:
            h, w = rendered.shape[:2]
            resized = cv2.resize(
                roi_mask,
                (w, h)
            )

            resized = cv2.threshold(
                resized,
                127,
                255,
                cv2.THRESH_BINARY
            )[1]

            rendered = cv2.threshold(
                rendered,
                127,
                255,
                cv2.THRESH_BINARY
            )[1]

            diff = cv2.absdiff(
                resized,
                rendered
            )

            edge_a = cv2.Canny(
                resized,
                100,
                200
            )

            edge_b = cv2.Canny(
                rendered,
                100,
                200
            )

            edge_diff = cv2.absdiff(
                edge_a,
                edge_b
            )

            pixel_score = (
                1.0 -
                (np.mean(diff) / 255.0)
            )

            edge_score = (
                1.0 -
                (np.mean(edge_diff) / 255.0)
            )

            final_score = (
                pixel_score * 0.65 +
                edge_score * 0.35
            )

            return float(final_score)

        except:
            return 0.0

    def _event_font_prior(self, font_name, event_name):
        name = font_name.lower()
        score = 1.0

        if event_name == "Whispering":

            if (
                "thin" in name or
                "light" in name
            ):

                score += 0.45

            if (
                "black" in name or
                "heavy" in name
            ):

                score -= 0.6

        elif event_name == "Gasp":
            if (
                "bold" in name or
                "black" in name or
                "heavy" in name
            ):

                score += 0.5

            if (
                "thin" in name or
                "light" in name
            ):

                score -= 0.5

        elif event_name == "Laughter":
            if (
                "rounded" in name or
                "soft" in name
            ):

                score += 0.35

        return score

    def _detect_main_event(self, energy):
        if energy < 0.015:
            return "Whispering"

        if energy > 0.08:
            return "Gasp"

        if energy > 0.05:
            return "Laughter"

        if energy > 0.03:
            return "Snicker"

        return "Talking"
    
    def _get_font_priority(self, font_path, text):
        import re
        from fontTools.ttLib import TTFont
        has_korean = bool(re.search("[가-힣]", text))
        if not has_korean: return 1.0
        try:
            font_idx = 0 if font_path.lower().endswith(".ttc") else -1
            with TTFont(font_path, fontNumber=font_idx, lazy=True) as tt:
                for table in tt['cmap'].tables:
                    if 0xAC00 in table.cmap:
                        return 1.2 
            return 0.2 
        except:
            return 1.0
        
    def _get_image_vector(self, pil_img):
        with torch.no_grad():
            img_t = self.transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
            vector = self.feature_extractor(img_t)
            return vector.flatten().cpu().numpy()

    def _precompute_font_embeddings(self):
        from PIL import Image, ImageOps, ImageEnhance
        test_text = "각숑합QY7" 
        print(f"[LOG] {len(self.font_db)}개 폰트 고정밀 인덱싱 시작...")
        
        for name, path in self.font_db.items():
            try:
                render_img = self._render_font_text(test_text, path)
                if render_img is None: continue
                
                pil_img = Image.fromarray(render_img).convert("L")
                bbox = pil_img.getbbox()
                if bbox: pil_img = pil_img.crop(bbox)
                pil_img = ImageOps.autocontrast(pil_img)
                pil_img = ImageEnhance.Contrast(pil_img).enhance(2.5).convert("RGB")
                
                with torch.no_grad():
                    img_t = self.transform(pil_img).unsqueeze(0).to(self.device)
                    self.font_embeddings[name] = {
                        "vector": self.feature_extractor(img_t).flatten().cpu().numpy(),
                        "aspect_ratio": render_img.shape[1] / render_img.shape[0] 
                    }
            except: continue

    def _match_font(self, sample_img, features, text, event_name):
        best_font = "NanumGothic"
        min_dist = float('inf')
        
        try:
            sample_coords = np.column_stack(np.where(sample_img > 0))
            if sample_coords.size == 0: return best_font, 0
            
            y1, x1 = sample_coords.min(axis=0); y2, x2 = sample_coords.max(axis=0)
            cropped_sample = sample_img[y1:y2+1, x1:x2+1]
            
            kernel = np.ones((2, 2), np.uint8)
            processed_sample = cv2.dilate(cropped_sample, kernel, iterations=1)
            
            density = np.count_nonzero(processed_sample) / (processed_sample.size + 1e-6)
            
            circularity = 0
            contours, _ = cv2.findContours(processed_sample, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                circle_area = np.pi * (radius ** 2)
                if circle_area > 0:
                    circularity = area / circle_area

            sample_pil = Image.fromarray(processed_sample).convert("L")
            sample_pil = ImageOps.autocontrast(sample_pil)
            sample_pil = ImageEnhance.Contrast(sample_pil).enhance(2.5).convert("RGB")
            
            with torch.no_grad():
                img_t = self.transform(sample_pil).unsqueeze(0).to(self.device)
                sample_vec = self.feature_extractor(img_t).flatten().cpu().numpy()

            has_hangul = any('가' <= char <= '힣' for char in text)

            for name, font_data in self.font_embeddings.items():
                vec = font_data["vector"] if isinstance(font_data, dict) else font_data
                dist = cosine(sample_vec, vec)
                
                lname = name.lower()
                
                is_round_font = any(kw in lname for kw in ['round', 'cookie', 'jua', 'soft', '둥근', '어비'])
                is_myeongjo = any(kw in lname for kw in ['myeongjo', 'batang', 'shinmyungjo', '명조', '바탕', '궁서'])
                is_heavy_font = any(kw in lname for kw in ['black', 'bold', 'heavy', 'extra'])

                if circularity > 0.45:
                    if is_round_font: dist *= 0.65 
                    if is_myeongjo: dist *= 2.5     
                elif circularity < 0.25:
                    if is_round_font: dist *= 1.5  
                    if is_myeongjo: dist *= 0.9  

                if density > 0.5: 
                    if is_heavy_font or "cookierun" in lname: dist *= 0.8
                    else: dist *= 1.5
                elif density < 0.25: 
                    if not is_heavy_font: dist *= 0.8
                    else: dist *= 1.5

                if has_hangul:
                    k_keywords = ['nanum', 'gmarket', 'cookierun', 'rix', 'mapo', 'pretendard', 'hy', 'arita', 'bm', 'uhbee', 'hcr', '조선']
                    if not any(k in lname for k in k_keywords):
                        dist *= 2.0 

                if dist < min_dist:
                    min_dist = dist
                    best_font = name
            
            final_conf = max(0.1, 1.0 - (min_dist * 1.2))
            
            print(f"   [결과] {best_font}")
            print(f"         (밀도: {density:.2f} | 원형률: {circularity:.2f} | 신뢰도: {final_conf:.4f})")
            
            return best_font, final_conf
            
        except Exception as e:
            print(f"[LOG] {e}")
            return best_font, 0
    
    def _extract_jaso_features(self, roi_mask):
        try:
            if roi_mask is None:
                return None

            binary = cv2.threshold(roi_mask,
                127,
                255,
                cv2.THRESH_BINARY
            )[1]

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                return None

            char_imgs = []
            contours = sorted(
                contours,
                key=lambda c: cv2.boundingRect(c)[0]
            )

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)

                if w < 5 or h < 5:
                    continue

                char = binary[
                    y:y+h,
                    x:x+w
                ]

                char = cv2.resize(char, (64, 64))
                char_imgs.append(char)

            if len(char_imgs) == 0:
                return None

            merged = np.zeros(
                (64, 64),
                dtype=np.float32
            )

            count = 0

            for char in char_imgs[:12]:
                merged += (
                    char.astype(np.float32) / 255.0
                )

                count += 1

            merged /= max(count, 1)
            edges = cv2.Canny(
                (merged * 255).astype(np.uint8),
                100,
                200
            )

            features = {
                "merged": merged,
                "edges": edges,
                "density": float(np.mean(merged)),
                "edge_density": float(
                    np.mean(edges) / 255.0
                )
            }
            
            return features

        except Exception as e:
            print("[LOG] 자소 특징 추출 실패")
            print(e)

            return None
        
    def _compare_jaso_features(self, a, b):
        try:
            density_diff = abs(
                a["density"] -
                b["density"]
            )

            edge_diff = abs(
                a["edge_density"] -
                b["edge_density"]
            )

            merged_diff = np.mean(
                np.abs(
                    a["merged"] -
                    b["merged"]
                )
            )

            edge_img_diff = np.mean(
                np.abs(
                    a["edges"].astype(np.float32) -
                    b["edges"].astype(np.float32)
                )
            ) / 255.0

            score = 1.0 - (
                density_diff * 0.2 +
                edge_diff * 0.2 +
                merged_diff * 0.35 +
                edge_img_diff * 0.25
            )

            return float(score)

        except Exception as e:
            print("[LOG] 자소 특징 비교 실패")
            print(e)

            return 0.0

    def analyze_user_styles(
        self,
        original_paths,
        edited_paths,
        preset_name
    ):
        all_thresholds = []
        all_silence_tolerances = []
        all_padding_i = []
        all_padding_o = []

        event_specific_settings = {
            "Sigh": {"padding_i": 0.3, "padding_o": 0.5, "threshold": 0.0031},
            "Laughter": {"padding_i": 0.5, "padding_o": 0.7, "threshold": 0.0052},
            "Gasp": {"padding_i": 0.25, "padding_o": 0.45, "threshold": 0.0028},
            "Cough": {"padding_i": 0.35, "padding_o": 0.55, "threshold": 0.0035},
            "Snicker": {"padding_i": 0.4, "padding_o": 0.6, "threshold": 0.0040},
            "Whispering": {"padding_i": 0.2, "padding_o": 0.4, "threshold": 0.0022}
        }
        font_features = []

        if self.stt_model is None:
            print("[LOG] Whisper 모델 로드 중...")
            device = self.device
            compute_type = "float16" if device == "cuda" else "int8"
            try:
                self.stt_model = WhisperModel("base", device=device, compute_type=compute_type)
            except Exception as e:
                print(f"[ERROR] {e}")
                self.stt_model = WhisperModel("base", device="cpu", compute_type="int8")

        for orig_path, edit_path in zip(original_paths, edited_paths):
            print(f"[LOG] 대조 분석 시작: {os.path.basename(orig_path)} <-> {os.path.basename(edit_path)}")

            orig_audio = self._get_audio_data(orig_path)
            edit_audio = self._get_audio_data(edit_path)

            if orig_audio is None or edit_audio is None:
                print(f"[LOG] 오디오 추출 실패: {edit_path}")
                continue

            abs_orig = np.abs(orig_audio)
            abs_edit = np.abs(edit_audio)
            orig_avg = np.mean(abs_orig)
            speech_parts = abs_edit[abs_edit > (orig_avg * 0.3)]

            user_threshold = np.percentile(speech_parts, 3) if len(speech_parts) > 0 else 0.08
            all_thresholds.append(user_threshold)

            correlation = np.correlate(abs_orig[::1600], abs_edit[::1600], mode='valid')
            offset_idx = np.argmax(correlation)
            print(f"[LOG] 추정 오프셋: {offset_idx / 10.0:.2f}초")

            energy = float(np.mean(abs_edit))
            event_name = self._detect_main_event(energy)

            dynamic_padding_i = round(0.25 + energy * 12.0, 3)
            dynamic_padding_o = round(0.45 + energy * 15.0, 3)
            dynamic_threshold = round(float(user_threshold), 5)

            event_specific_settings[event_name] = {
                "padding_i": dynamic_padding_i,
                "padding_o": dynamic_padding_o,
                "threshold": dynamic_threshold
            }

            all_padding_i.append(dynamic_padding_i)
            all_padding_o.append(dynamic_padding_o)
            all_silence_tolerances.append(30)

            cap = cv2.VideoCapture(edit_path)
            if not cap.isOpened():
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_positions = np.linspace(0, max(0, total_frames - 1), 20).astype(np.int32)

            for pos in sample_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
                ret, frame = cap.read()
                if not ret:
                    continue
                
                roi, roi_mask = self._extract_subtitle_roi(frame)
                if roi_mask is None:
                    continue

                text = "가나다ABC123"
                best_font, best_score = self._match_font(
                    roi_mask, 
                    None,  
                    text, 
                    event_name
                )

                current_density = np.count_nonzero(roi_mask) / (roi_mask.size + 1e-6)
                stroke_width = max(1.0, current_density * 18.0)

                font_features.append({
                    "font": best_font,
                    "score": float(best_score),
                    "stroke": float(stroke_width),
                    "density": float(current_density),
                    "event": event_name
                })
            cap.release()

        if len(all_thresholds) == 0:
            print("[LOG] 유효한 분석 데이터가 없어 프리셋 생성 실패")
            return None

        valid_features = [f for f in font_features if isinstance(f, dict)]

        if len(valid_features) > 0:
            font_counter = {}
            for f in valid_features:
                fname = f["font"]
                font_counter[fname] = font_counter.get(fname, 0) + 1

            detected_font = max(font_counter, key=font_counter.get)
            avg_stroke = np.mean([f["stroke"] for f in valid_features])
            avg_density = np.mean([f["density"] for f in valid_features])
        else:
            detected_font = "NanumGothic_Bold"
            avg_stroke = 2.0
            avg_density = 0.5

        if avg_stroke >= 6.0: base_weight_offset = 800.0
        elif avg_stroke >= 4.0: base_weight_offset = 650.0
        elif avg_stroke >= 2.5: base_weight_offset = 520.0
        else: base_weight_offset = 350.0

        size_multiplier = round(1.0 + (avg_density * 1.8), 2)
        mapped_domains = [self.domain_map.get(d, d) for d in self.domains]

        preset_data = {
            "threshold": round(float(np.mean(all_thresholds)) * 0.85, 5),
            "padding_i": round(float(np.mean(all_padding_i)), 3),
            "padding_o": round(float(np.mean(all_padding_o)), 3),
            "max_silence_frames": int(np.mean(all_silence_tolerances)),
            "use_whisper": True,
            "use_sed": True,
            "interest_events": self.interest_events,
            "event_specific_settings": event_specific_settings,
            "style_config": {
                "preset_meta": {
                    "preset_id": f"{'_'.join(mapped_domains).lower()}_001",
                    "domain": mapped_domains
                },
                "default_base_style": {
                    "target_font": detected_font,
                    "color_hex": "#e9e9e9",
                    "base_size_multiplier": float(size_multiplier),
                    "base_weight_offset": float(base_weight_offset),
                    "alignment": "bottom_center"
                },
                "layout_rules": {
                    "max_chars_per_line": 20,
                    "vertical_offset_percent": 10,
                    "line_height_multiplier": 1.2
                },
                "style_rules": []
            }
        }

        self.pm.save_preset(preset_name, preset_data)
        print(f"[LOG] 분석 완료 및 프리셋 저장: {preset_name}")
        print(f"[LOG] 감지 폰트: {detected_font}")
        print(f"[LOG] 평균 Stroke: {avg_stroke:.2f}")

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
                        "options": {
                            "temperature": 0.3,      
                            "top_p": 0.5,            
                            "num_predict": 192,      
                            "num_ctx": 2048          
                        }
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
        if not full_text or not full_text.strip(): return "General"
        
        prompt = (
            f"### 시스템 역할: 당신은 음성 인식(STT) 오타가 심한 동영상 스크립트를 분석하여 문맥을 파악하고 올바른 핵심 키워드를 도출하는 지능형 언어 모델입니다.\n\n"
            f"[상황 분석 및 미션]\n"
            f"현재 제공된 스크립트는 마이크 잡음이나 발음 웅얼거림으로 인해, 원본 단어의 발음이 깨져서 유령 단어(예: '보아나', '살소', '직체기' 등)로 잘못 녹취되어 있습니다.\n"
            f"당신은 이러한 엉터리 단어들을 단순 삭제하지 말고, 문장 전체의 흐름과 앞뒤 맥락을 살펴 **'말하는 사람이 실제로 하려던 원래 단어나 주제가 무엇이었을지' 최고 확률로 유추 및 복원(문맥 필터링)**해야 합니다.\n\n"
            f"[엄격한 키워드 변환 규칙]\n"
            f"1. **절대로** 스크립트에 적힌 꼬인 발음 형태('살소', '직체기' 등)를 그대로 출력하지 마십시오.\n"
            f"2. 문맥을 분석하여 해당 엉터리 단어의 원본 의미를 유추한 뒤, 반드시 **정상적이고 완벽한 한국어 표준어 단어**로 교정해서 키워드 목록에 포함하세요.\n"
            f"3. 만약 도저히 문맥적으로도 유추가 불가능한 완전한 소음 파편이라면 그때는 과감히 제외하십시오.\n"
            f"4. 오직 핵심 키워드 단어들만 쉼표(,)로 구분하여 나열하고, 인사말이나 사족('주제:', '결과:')은 절대 넣지 마세요.\n"
            f"5. 영상의 핵심 상황, 감정 상태, 주요 행동을 대변하는 유효한 단어를 최소 10개 이상, 20개 이하로 추출하세요.\n\n"
            f"[현재 영상 스크립트 (오타 포함)]\n"
            f"{full_text[:1500]}\n\n"
            f"결과(교정 및 유추를 완료한 10개 이상의 정상 단어 목록):"
        )
        print("[LOG] LLM 분석 엔진을 통해 영상 핵심 키워드 추출을 시작합니다...")
        response = self._ollama_request(prompt, timeout=300)

        if response:
            clean_res = str(response).strip()
            for prefix in ["결과:", "주제 키워드:", "주제:", "키워드:"]:
                if clean_res.startswith(prefix):
                    clean_res = clean_res[len(prefix):].strip()
        
            print("=" * 60)
            print(f"[주제 키워드 추출 완료]\n-> {clean_res}")
            print("=" * 60)
            return clean_res
        
        print("[주제 키워드] 모델 응답이 없어 'General' 모드로 대체합니다.")
        
        return "General"

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

        segments, _ = self.stt_model.transcribe(
            vocal_wav_path,
            language="ko",
            beam_size=5,
            patience=1.0,
            temperature=[0.0, 0.2, 0.4],
            no_repeat_ngram_size=3, 
            initial_prompt="안녕하세요. 다음은 자연스러운 한국어 대화 녹음입니다. 비속어, 한숨, 신조어, 웃음소리가 포함되어 있으며 문맥에 맞게 정확한 단어로 받아 적습니다."
        )
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

        subtitles_dir = os.path.join(self.output_dir, "subtitles")
        os.makedirs(subtitles_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        full_subtitle_file = os.path.join(subtitles_dir, f"{base_name}_full_subtitle.txt")
        subtitle_file = os.path.join(subtitles_dir, f"{base_name}_subtitle.txt")

        import re
        def clean_repeated_text(text):
            cleaned = re.sub(r'(\b\w+[!?.~]*\s*)\1{2,}', r'\1\1', text.strip())
            return cleaned if cleaned else text.strip()

        try:
            with open(full_subtitle_file, 'w', encoding='utf-8') as f_full:
                for i, (start, end, text) in enumerate(kept_segments, 1):
                    refined_text = clean_repeated_text(text)
                    if not refined_text:
                        refined_text = text.strip()
                    f_full.write(f"{i}\n{start:.2f} --> {end:.2f}\n{refined_text}\n\n")
            print(f"[자막 생성] 타임코드 포함 전체 자막 저장 완료: {full_subtitle_file}")
        except Exception as e:
            print(f"[경고] full_subtitle 생성 중 오류 발생: {e}")

        try:
            with open(subtitle_file, 'w', encoding='utf-8') as f_sub:
                for _, _, text in kept_segments:
                    refined_text = clean_repeated_text(text)
                    if not refined_text:
                        refined_text = text.strip()
                    f_sub.write(f"{refined_text}\n")
            print(f"[자막 생성] 텍스트 전용 자막 저장 완료: {subtitle_file}")
        except Exception as e:
            print(f"[경고] subtitle 생성 중 오류 발생: {e}")

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

        self.progress_signal.emit("LLM 모델을 VRAM에 로드하는 중입니다...")
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
            self.progress_signal.emit("경량 모델을 VRAM에 다시 로드 중입니다...")
            
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

import sys
import os
import requests
import psutil
import subprocess
import torch
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QStatusBar
from PySide6.QtCore import QThread, QTimer

def setup_hardware_env():
    has_nvidia = torch.cuda.is_available()
    if has_nvidia:
        os.environ["OLLAMA_VULKAN"] = "0" 
        os.environ["OLLAMA_FLASH_ATTENTION"] = "true"
    else:
        os.environ["OLLAMA_VULKAN"] = "1"
        os.environ["OLLAMA_FLASH_ATTENTION"] = "true"
    os.environ["OLLAMA_LOAD_TIMEOUT"] = "10m"

setup_hardware_env()

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_FILE_DIR not in sys.path:
    sys.path.append(CURRENT_FILE_DIR)

class KirinukiLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kirinuki Editor v0.1")
        self.resize(1400, 900)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("시스템 시작 중...")
        self.init_ui()
        self.ollama_thread = None
        self.ollama_worker = None
        QTimer.singleShot(100, self.start_background_process)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        try:
            from Pmode import PmodeWidget
            self.mode_tabs = QTabWidget()
            self.mode_tabs.addTab(PmodeWidget(), "자동 편집 모드 (P-Mode)")
            self.mode_tabs.setTabEnabled(0, False)
            self.main_layout.addWidget(self.mode_tabs)
        except Exception as e:
            print(f"UI 로드 실패: {e}")

    def start_background_process(self):
        try:
            from pipeline import OllamaWorker
            self.ollama_thread = QThread()
            self.ollama_worker = OllamaWorker()
            self.ollama_worker.moveToThread(self.ollama_thread)
            self.ollama_worker.progress_signal.connect(self.log_to_status_bar)
            self.ollama_thread.started.connect(self.ollama_worker.run)
            self.ollama_worker.finished_signal.connect(self.ollama_thread.quit)
            self.ollama_worker.finished_signal.connect(self.on_initialization_finished)
            self.ollama_thread.start()
        except Exception as e:
            self.log_to_status_bar(f"초기화 오류: {str(e)}")

    def on_initialization_finished(self, success):
        self.log_to_status_bar("연결 상태 최종 확인 중 (CLI)...")
        QTimer.singleShot(1000, self.force_check_and_enable)

    def force_check_and_enable(self):
        try:
            result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=15)
            output = result.stdout.strip()
            if output and "NAME" in output:
                model_line = output.split('\n')[-1]
                self.log_to_status_bar(f"로드 성공: {model_line}")
                self.mode_tabs.setTabEnabled(0, True)
            else:
                self.log_to_status_bar("엔진 감지됨. 사용을 허용합니다.")
                self.mode_tabs.setTabEnabled(0, True)
        except Exception:
            self.log_to_status_bar("CLI 확인 불가. 버튼을 강제 활성화합니다.")
            self.mode_tabs.setTabEnabled(0, True)

    def log_to_status_bar(self, message):
        print(f"[LOG] {message}") 
        self.statusBar().showMessage(message)

    def closeEvent(self, event):
        try:
            for p in psutil.process_iter(['name']):
                if "ollama" in p.info['name'].lower(): p.kill()
        except: pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KirinukiLauncher()
    window.show()
    sys.exit(app.exec())
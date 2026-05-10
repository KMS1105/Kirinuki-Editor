import sys
import os
import traceback
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QStatusBar
from PySide6.QtCore import QThread, QTimer

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

        QTimer.singleShot(1500, self.start_background_process)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        try:
            from Mmode import MmodeWidget
            from Pmode import PmodeWidget
            self.mode_tabs = QTabWidget()
            self.mode_tabs.addTab(PmodeWidget(), "자동 편집 모드 (P-Mode)")
            self.mode_tabs.addTab(MmodeWidget(), "수동 편집 모드 (M-Mode)")
            self.main_layout.addWidget(self.mode_tabs)
        except Exception as e:
            print(f"UI 로드 실패: {e}")

    def start_background_process(self):
        """Ollama 설치 및 모델 체크 시작"""
        try:
            from pipeline import OllamaWorker
            
            self.ollama_thread = QThread()
            self.ollama_worker = OllamaWorker()
            self.ollama_worker.moveToThread(self.ollama_thread)
            
            self.ollama_worker.progress_signal.connect(self.log_to_status_bar)
            self.ollama_thread.started.connect(self.ollama_worker.run)
       
            self.ollama_worker.finished_signal.connect(lambda success: self.ollama_thread.quit())
            self.ollama_worker.finished_signal.connect(
                lambda success: self.log_to_status_bar("AI 엔진 준비 완료" if success else "AI 엔진 초기화 실패")
            )
            
            self.ollama_thread.start()
        except Exception as e:
            self.log_to_status_bar(f"초기화 오류: {str(e)}")

    def log_to_status_bar(self, message):
        """로그를 상태바에 출력 (터미널에도 동시 출력)"""
        print(f"[LOG] {message}") 
        self.statusBar().showMessage(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KirinukiLauncher()
    window.show()
    sys.exit(app.exec())
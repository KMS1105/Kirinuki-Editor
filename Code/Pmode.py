import os
import sys
from PySide6.QtCore import Qt, QUrl, QThread, Signal, Slot, QObject
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QTextCursor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QProgressBar, QGroupBox, QCheckBox, QComboBox, QFrame, 
    QSlider, QListWidget, QListWidgetItem, QSizePolicy, 
    QTabWidget, QMenu, QInputDialog, QMessageBox, QTextEdit
)

from preset import PresetManager

class LogStream(QObject):
    new_log = Signal(str, bool)

    def write(self, text):
        if not text: return
        if '\r' in text:
            parts = text.split('\r')
            self.new_log.emit(parts[-1], True)
        else:
            self.new_log.emit(text, False)

    def flush(self):
        pass

class AnalysisThread(QThread):
    finished_signal = Signal(dict)

    def __init__(self, origins, editeds, preset_name):
        super().__init__()
        self.origins = origins
        self.editeds = editeds
        self.preset_name = preset_name

    def run(self):
        from pipeline import StyleAnalyzer, CutEngine
        
        engine = CutEngine()
        analyzer = StyleAnalyzer(ffmpeg_bin_path=engine.ffmpeg_bin_path)
            
        style_result = self.analyzer.analyze_user_styles(
            self.origins, 
            self.editeds, 
            self.preset_name, 
        )
        self.finished_signal.emit(style_result)

class EditThread(QThread):
    finished_signal = Signal()

    def __init__(self, target_files, preset_name):
        super().__init__()
        self.target_files = target_files
        self.preset_name = preset_name
        
    def run(self):
        try:
            print("[LOG] 편집 엔진 초기화 중...")
            from pipeline import CutEngine
            self.engine = CutEngine()

            print("[LOG] 편집 엔진 준비 중...")
            self.engine.activate()

            for file_path in self.target_files:
                print(f"[LOG] 작업 시작: {os.path.basename(file_path)}")
                self.engine.process_video(file_path, self.preset_name)
                    
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[시스템] 편집 중단됨: {str(e)}\n{error_details}")
                
        self.finished_signal.emit()

class DropListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DropOnly)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; color: #ffffff;")
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.open_menu)
        
        self.del_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        self.del_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.del_shortcut.activated.connect(self.delete_selected)
        self.data_store = []

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and path not in self.data_store:
                    self.data_store.append(path)
                    self.addItem(QListWidgetItem(path))
            event.acceptProposedAction()
        else: event.ignore()

    def open_menu(self, pos):
        item = self.itemAt(pos)
        if not item: return
        menu = QMenu(self)
        act = QAction("목록에서 삭제", self)
        act.triggered.connect(lambda: self.delete_item(item))
        menu.addAction(act)
        menu.exec(self.mapToGlobal(pos))

    def delete_item(self, item):
        if not item: return
        path = item.text()
        if path in self.data_store: self.data_store.remove(path)
        self.takeItem(self.row(item))

    def delete_selected(self):
        for item in self.selectedItems(): self.delete_item(item)

class PmodeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pm = PresetManager()
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.current_preset_data = None
        self.seeking = False
        
        self.init_ui()
        self.space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_shortcut.activated.connect(self.toggle_play)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        top_h_layout = QHBoxLayout()

        preview_area = QWidget()
        preview_layout = QVBoxLayout(preview_area)
        
        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.video_frame.setMinimumHeight(360)
        
        vf_inner_layout = QVBoxLayout(self.video_frame)
        vf_inner_layout.setContentsMargins(0, 0, 0, 0)
        self.video_widget = QVideoWidget()
        self.player.setVideoOutput(self.video_widget)
        vf_inner_layout.addWidget(self.video_widget)
        preview_layout.addWidget(self.video_frame)

        self.time_slider = QSlider(Qt.Horizontal)
        self.player.positionChanged.connect(self.update_position)
        self.player.durationChanged.connect(self.update_duration)
        self.time_slider.sliderPressed.connect(self.slider_pressed)
        self.time_slider.sliderReleased.connect(self.slider_released)
        
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(40)
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.time_slider)
        preview_layout.addLayout(ctrl_layout)

        preview_layout.addWidget(QLabel("<b>편집 대상 파일 (드래그 앤 드롭)</b>"))
        self.target_file_list = DropListWidget(self)
        self.target_file_list.setMinimumHeight(150)
        self.target_file_list.itemClicked.connect(self.load_video_to_player)
        preview_layout.addWidget(self.target_file_list)
        
        top_h_layout.addWidget(preview_area, 2)

        self.tab_widget = QTabWidget()
        self.tab_widget.setMinimumWidth(380)

        self.init_log_tab()
        self.init_analysis_tab()
        self.init_automation_tab()
        
        top_h_layout.addWidget(self.tab_widget, 1)
        main_layout.addLayout(top_h_layout)
        
        self.log_stream = LogStream()
        self.log_stream.new_log.connect(self.update_log_display)
        sys.stdout = self.log_stream
        sys.stderr = self.log_stream
        
        self.btn_main_run = QPushButton("일괄 자동 편집 시작")
        self.btn_main_run.setFixedHeight(50)
        self.btn_main_run.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-weight: bold; font-size: 14px; border-radius: 4px; } QPushButton:disabled { background-color: #7f8c8d; }")
        self.btn_main_run.clicked.connect(self.run_batch_processing)
        main_layout.addWidget(self.btn_main_run)

    def init_analysis_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<b>[1] 원본 영상 리스트</b>"))
        self.list_analysis_origin = DropListWidget(self); layout.addWidget(self.list_analysis_origin)
        layout.addWidget(QLabel("<b>[2] 결과 영상 리스트 (편집본)</b>"))
        self.list_analysis_edited = DropListWidget(self); layout.addWidget(self.list_analysis_edited)
        
        self.btn_do_analysis = QPushButton("편집 스타일 분석 및 프리셋 저장")
        self.btn_do_analysis.setFixedHeight(45)
        self.btn_do_analysis.setStyleSheet("QPushButton { background-color: #2980b9; color: white; border-radius: 4px; }")
        self.btn_do_analysis.clicked.connect(self.on_click_analysis)
        layout.addWidget(self.btn_do_analysis)
        self.tab_widget.addTab(tab, "스타일 분석")
        
    def init_log_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.log_display = QTextEdit()
        
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText("작업이 시작되면 로그가 여기에 표시됩니다...")

        self.log_display.setStyleSheet("""
            background-color: #1e1e1e; 
            color: #d4d4d4; 
            font-family: 'Consolas', 'Monaco', monospace;
        """)
        
        layout.addWidget(self.log_display)
        self.tab_widget.addTab(tab, "실시간 로그")

    def init_automation_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("<b>프리셋 선택</b>"))
        self.combo_presets = QComboBox()
        self.refresh_preset_combo(); layout.addWidget(self.combo_presets)
        
        self.btn_apply_preset = QPushButton("프리셋 적용")
        self.btn_apply_preset.setFixedHeight(35)
        self.btn_apply_preset.setStyleSheet("background-color: #8e44ad; color: white;")
        self.btn_apply_preset.clicked.connect(self.on_click_apply_preset)
        layout.addWidget(self.btn_apply_preset)
        
        line = QFrame(); line.setFrameShape(QFrame.HLine); layout.addWidget(line)
        
        opt_group = QGroupBox("자동화 옵션")
        opt_layout = QVBoxLayout(opt_group)
        self.chk_use_cut = QCheckBox("자동 컷 편집 적용", checked=True)
        self.chk_use_fx = QCheckBox("전환 효과 자동 추가")
        self.chk_use_audio = QCheckBox("오디오 노이즈 제거")
        opt_layout.addWidget(self.chk_use_cut)
        opt_layout.addWidget(self.chk_use_fx)
        opt_layout.addWidget(self.chk_use_audio)
        layout.addWidget(opt_group)
        
        layout.addStretch()
        self.tab_widget.addTab(tab, "자동화 프로세스")

    def refresh_preset_combo(self):
        self.combo_presets.clear()
        presets = self.pm.get_preset_list()
        if presets: self.combo_presets.addItems(presets)
        else: self.combo_presets.addItem("저장된 프리셋 없음")

    def on_click_analysis(self):
        origins = self.list_analysis_origin.data_store
        editeds = self.list_analysis_edited.data_store
        if not origins or not editeds:
            QMessageBox.warning(self, "알림", "분석할 파일을 추가해주세요.")
            return

        preset_name, ok = QInputDialog.getText(self, "스타일 저장", "프리셋 이름을 입력하세요:")
        if not (ok and preset_name): return

        self.target_preset_name = preset_name
        self.btn_do_analysis.setEnabled(False)

        self.analysis_worker = AnalysisThread(origins, editeds, preset_name)
        self.analysis_worker.finished_signal.connect(self.on_analysis_finished)
        self.analysis_worker.start()
        
    def update_log_display(self, text, overwrite):
        """오류를 수정한 로그 갱신 함수"""
        cursor = self.log_display.textCursor()
        
        if overwrite:
            cursor.movePosition(QTextCursor.End) 
            cursor.select(QTextCursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.insertText(text)
        
        else:
            cursor.movePosition(QTextCursor.End)
            self.log_display.insertPlainText(text)

        self.log_display.ensureCursorVisible()

    @Slot(dict)
    def on_analysis_finished(self, result):
        self.pm.save_preset(self.target_preset_name, result)
        self.refresh_preset_combo()
        self.btn_do_analysis.setEnabled(True)
        QMessageBox.information(self, "완료", f"'{self.target_preset_name}' 프리셋 저장 완료!")
        
        self.refresh_preset_combo()
        self.btn_do_analysis.setEnabled(True)
        QMessageBox.information(self, "분석 완료", 
            f"프리셋 '{self.target_preset_name}' 저장 및\n분석 디버그 JSON 파일이 생성되었습니다.")

    def on_click_apply_preset(self):
        selected = self.combo_presets.currentText()
        if selected == "저장된 프리셋 없음": return
        self.current_preset_data = self.pm.load_preset(selected)
        QMessageBox.information(self, "알림", f"'{selected}' 프리셋이 로드되었습니다.")

    def run_batch_processing(self):
        target_files = self.target_file_list.data_store
        selected_preset = self.combo_presets.currentText() 
        
        if not target_files or selected_preset == "저장된 프리셋 없음":
            QMessageBox.warning(self, "알림", "대상 파일과 프리셋을 확인해주세요.")
            return

        self.btn_main_run.setEnabled(False)
        self.tab_widget.setCurrentIndex(0) 
        
        from pipeline import OllamaWorker
        self.ollama_thread = QThread()
        self.ollama_worker = OllamaWorker()
        self.ollama_worker.moveToThread(self.ollama_thread)

        self.ollama_worker.progress_signal.connect(self.log_stream.write)
        self.ollama_worker.finished_signal.connect(lambda success: self.ollama_thread.quit())
        self.ollama_worker.finished_signal.connect(
            lambda success, tf=target_files, sp=selected_preset: self.on_ollama_bootstrap_finished(success, tf, sp)
        )
        
        self.ollama_thread.finished.connect(self.ollama_worker.deleteLater)
        self.ollama_thread.started.connect(self.ollama_worker.run)
        self.ollama_thread.start()

    def on_ollama_bootstrap_finished(self, success, target_files, selected_preset):
        if not success:
            self.btn_main_run.setEnabled(True)
            QMessageBox.critical(self, "오류", "AI 모델 초기화에 실패했습니다. 로그를 확인해주세요.")
            return

        print("\n[시스템] 모든 준비 완료. 편집 프로세스를 시작합니다.")
        self.start_real_editing(target_files, selected_preset)

    def start_real_editing(self, target_files, selected_preset):
        self.edit_worker = EditThread(target_files, selected_preset)
        self.edit_worker.finished_signal.connect(self.on_processing_finished)
        self.edit_worker.finished.connect(self.edit_worker.deleteLater)
        self.edit_worker.start()

    def on_processing_finished(self):
        self.btn_main_run.setEnabled(True)
        QMessageBox.information(self, "작업 완료", "일괄 편집이 완료되었습니다.")

    def load_video_to_player(self, item):
        self.player.setSource(QUrl.fromLocalFile(item.text()))
        self.player.play()
        self.btn_play.setText("⏸")

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause(); self.btn_play.setText("▶")
        else:
            if not self.player.source().isEmpty(): self.player.play(); self.btn_play.setText("⏸")

    def update_position(self, pos):
        if not self.seeking: self.time_slider.setValue(pos)

    def update_duration(self, dur): self.time_slider.setRange(0, dur)
    def slider_pressed(self): self.seeking = True
    def slider_released(self):
        self.player.setPosition(self.time_slider.value())
        self.seeking = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.video_frame.setFixedHeight(int(self.video_frame.width() * 9 / 16))
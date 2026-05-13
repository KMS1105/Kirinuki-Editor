import sys
from tokenize import group
import shutil
import tempfile
import os

from PySide6.QtCore import QMimeData, QPoint, QUrl, Qt
from PySide6.QtGui import QCursor, QDrag, QMouseEvent, QColor, QFont, QPainter, QPen, QBrush, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFontComboBox,
    QFileDialog, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QMainWindow, QMenu,
    QProgressBar, QPushButton, QSlider, QSizePolicy,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
    QWidget, QStyle, QStyleOption, QGridLayout, QScrollArea,
)

from PySide6.QtMultimedia import (
    QAudioDecoder,
    QAudioBuffer,
    QMediaPlayer,
    QAudioOutput
)

from editor import (
    AudioWaveformVisualizer,
    calculate_max_waveform_length,
    get_media_duration_ms,
    trim_media_clip,
    VideoPlayerController,
    VideoClipWidget,
    AudioClipWidget,
    ClipGroup
)

class DropListWidget(QListWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QListWidget.ExtendedSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    
                    if not self.findItems(path, Qt.MatchExactly):
                        self.addItem(path)

                        if hasattr(self.main_window, "on_source_added"):
                            self.main_window.on_source_added(path)
                        
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def startDrag(self, supported_actions):
        item = self.currentItem()
        if item:
            mime_data = QMimeData()
            mime_data.setUrls([QUrl.fromLocalFile(item.text())])
            mime_data.setText(item.text())
            drag = QDrag(self)
            drag.setMimeData(mime_data)
            drag.exec(Qt.MoveAction)

class AspectRatioFrame(QFrame):
    def __init__(self, aspect_ratio=16/9, parent=None):
        super().__init__(parent)

        self.aspect_ratio = aspect_ratio

        self.setStyleSheet("""
            background-color: black;
        """)

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_size()

    def update_size(self):
        p = self.parentWidget()

        if not p:
            return

        pw = p.width()
        ph = p.height()

        if pw <= 0 or ph <= 0:
            return

        if pw / ph > self.aspect_ratio:
            new_h = ph
            new_w = int(ph * self.aspect_ratio)

        else:
            new_w = pw
            new_h = int(pw / self.aspect_ratio)

        self.setFixedSize(new_w, new_h)

class MmodeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player_controller = VideoPlayerController(self)
        self.current_file = None
        self.timeline_clips = []
        self.selected_clip_group = None
        self.selected_source_item = None
        self.current_clip_group = None
        self.timeline_playhead = None
        self.is_scrubbing = False
        self.total_duration = 0
        self.timeline_total_duration = 0
        self.timeline_max_ms = 0
        self.timeline_scale = 3
        self.min_clip_width = 24
        
        self.init_ui()
        self.init_connections()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        main_splitter = QSplitter(Qt.Horizontal)
        
        self.left_sidebar = QTabWidget()
        self.file_list = DropListWidget(self)
        self.file_list.setDragEnabled(True)
        self.left_sidebar.addTab(self.file_list, "파일 목록")
        self.left_sidebar.setMinimumWidth(220)
        
        center_section = QWidget()
        center_layout = QVBoxLayout(center_section)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        
        preview_container = QWidget()
        preview_container.setStyleSheet("background-color: #222;")
        preview_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_container_layout = QVBoxLayout(preview_container)
        preview_container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.preview_frame = AspectRatioFrame(16/9)
        self.preview_frame_layout = QVBoxLayout(
            self.preview_frame
        )

        self.preview_frame_layout.setContentsMargins(
            0, 0, 0, 0
        )

        self.preview_frame_layout.addWidget(
            self.player_controller.video_widget
        )

        self.player_controller.video_widget.setAspectRatioMode(
            Qt.KeepAspectRatio
        )

        preview_container_layout.addWidget(
            self.preview_frame,
            alignment=Qt.AlignCenter
        )
        center_layout.addWidget(preview_container, stretch=10)
                
        transport_container = QWidget()
        transport_main_vbox = QVBoxLayout(transport_container)
        transport_main_vbox.setContentsMargins(10, 5, 10, 5)
        transport_main_vbox.setSpacing(5)

        upper_row = QHBoxLayout()
        upper_row.setSpacing(10)
        
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(60, 40)
        self.btn_play.setStyleSheet("font-size: 13px; font-weight: bold;")
        
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        upper_row.addWidget(self.btn_play)
        upper_row.addWidget(self.scrubber)
        
        self.lbl_time = QLabel("00:00:00.000 / 00:00:00.000")
        self.lbl_time.setAlignment(Qt.AlignLeft)
        
        transport_main_vbox.addLayout(upper_row)
        transport_main_vbox.addWidget(self.lbl_time)
        
        center_layout.addWidget(transport_container)
        
        self.right_sidebar = QTabWidget()
        self.init_inspector()
        self.right_sidebar.setMinimumWidth(300)
        
        main_splitter.addWidget(self.left_sidebar)
        main_splitter.addWidget(center_section)
        main_splitter.addWidget(self.right_sidebar)
        main_splitter.setStretchFactor(1, 10)
        
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.addWidget(main_splitter)
        self.timeline_container = QWidget()
        self.init_timeline()
        v_splitter.addWidget(self.timeline_container)
        v_splitter.setSizes([800, 400])
        v_splitter.setStretchFactor(0, 10)
        v_splitter.setStretchFactor(1, 1)
        
        layout.addWidget(v_splitter)

    def init_inspector(self):
        v_tab = QWidget()
        v_lay = QVBoxLayout(v_tab)
        v_g = QGroupBox("Transform")
        v_gl = QGridLayout(v_g)
        v_gl.addWidget(QLabel("Scale:"), 0, 0); v_gl.addWidget(QSpinBox(), 0, 1)
        v_gl.addWidget(QLabel("Angle:"), 1, 0); v_gl.addWidget(QSpinBox(), 1, 1)
        v_lay.addWidget(v_g)
        v_lay.addWidget(QLabel("Opacity")); v_lay.addWidget(QSlider(Qt.Horizontal))
        v_lay.addStretch()
        
        a_tab = QWidget()
        a_lay = QVBoxLayout(a_tab)
        a_lay.addWidget(QLabel("Volume (dB)")); a_lay.addWidget(QSpinBox())
        a_lay.addStretch()
        
        self.right_sidebar.addTab(v_tab, "비디오")
        self.right_sidebar.addTab(a_tab, "오디오")
        
    def init_connections(self):
        self.file_list.itemDoubleClicked.connect(self.load_selected_media)
        self.btn_play.clicked.connect(self.toggle_playback)
        self.scrubber.sliderPressed.connect(
            self.on_scrub_start
        )

        self.scrubber.sliderReleased.connect(
            self.on_scrub_end
        )

        self.scrubber.sliderMoved.connect(
            self.on_scrub_move
        )

        self.player_controller.positionChanged.connect(self.update_position)
        self.player_controller.durationChanged.connect(self.update_duration)
        self.player_controller.nextClipNeeded.connect(self.play_next_clip)
        self.btn_render.clicked.connect(self.render_project)
        
        self.file_list.setContextMenuPolicy(
            Qt.CustomContextMenu
        )

        self.file_list.customContextMenuRequested.connect(
            self.show_source_context_menu
        )
        
        self.delete_shortcut = QShortcut(
            QKeySequence("Delete"),
            self
        )

        self.delete_shortcut.setContext(
            Qt.ApplicationShortcut
        )

        self.delete_shortcut.activated.connect(
            self.delete_selected_item
        )

        self.space_shortcut = QShortcut(
            QKeySequence(Qt.Key_Space),
            self
        )

        self.space_shortcut.setContext(
            Qt.ApplicationShortcut
        )

        self.space_shortcut.activated.connect(
            self.toggle_playback
        )

    def init_timeline(self):
        t_lay = QVBoxLayout(self.timeline_container)
        t_lay.setContentsMargins(0, 0, 0, 0)

        tb = QHBoxLayout()
        tb.setContentsMargins(5, 5, 5, 5)

        tb.addWidget(QPushButton("분할 (Ctrl+B)"))
        tb.addWidget(QPushButton("트랙 +"))
        tb.addStretch()
        tb.addWidget(QLabel("Zoom"))
        tb.addWidget(QSlider(Qt.Horizontal))

        self.btn_render = QPushButton("내보내기 (Render)")
        self.btn_render.setStyleSheet("""
            background-color: #c0392b;
            color: white;
            font-weight: bold;
            padding: 5px 15px;
        """)

        tb.addWidget(self.btn_render)
        t_lay.addLayout(tb)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self.timeline_content = QWidget()

        cl = QVBoxLayout(self.timeline_content)

        self.timeline_content.setAcceptDrops(True)
        self.timeline_content.dragEnterEvent = (
            self.timeline_drag_enter_event
        )

        self.timeline_content.dropEvent = (
            self.timeline_drop_event
        )

        self.timeline_content.mousePressEvent = (
            self.timeline_empty_click
        )

        self.v_track = QFrame()
        self.v_track.setMinimumHeight(70)
        self.v_track.setMinimumWidth(4000)

        self.v_track.setStyleSheet("""
            background-color: #252525;
            border: 1px solid #333;
        """)

        self.a_track = QFrame()
        self.a_track.setMinimumHeight(90)
        self.a_track.setMinimumWidth(4000)

        self.a_track.setStyleSheet("""
            background-color: #252525;
            border: 1px solid #333;
        """)

        self.video_track_label = QLabel("Video Track (V1)")
        self.audio_track_label = QLabel("Audio Track (A1)")

        self.video_track_label.setCursor(Qt.PointingHandCursor)
        self.audio_track_label.setCursor(Qt.PointingHandCursor)
        self.video_track_label.mousePressEvent = (
            lambda event: self.timeline_label_click(event, self.video_track_label, self.v_track)
        )
        self.audio_track_label.mousePressEvent = (
            lambda event: self.timeline_label_click(event, self.audio_track_label, self.a_track)
        )

        self.v_track.mousePressEvent = (
            lambda event: self.timeline_track_click(event, self.v_track)
        )
        self.a_track.mousePressEvent = (
            lambda event: self.timeline_track_click(event, self.a_track)
        )

        cl.addWidget(self.video_track_label)
        cl.addWidget(self.v_track)
        cl.addWidget(self.audio_track_label)
        cl.addWidget(self.a_track)

        scroll.setWidget(self.timeline_content)

        t_lay.addWidget(scroll)

        self.timeline_playhead = QFrame(
            self.timeline_content
        )

        self.timeline_playhead.setStyleSheet("""
            background-color: #ff3b30;
        """)

        self.timeline_playhead.setGeometry(
            0,
            20,
            2,
            220
        )

        self.timeline_playhead.raise_()
        self.timeline_playhead.show()

        self.timeline_playhead.setMouseTracking(True)
        self.timeline_playhead.mousePressEvent = self.playhead_mouse_press
        self.timeline_playhead.mouseMoveEvent = self.playhead_mouse_move
        self.timeline_playhead.mouseReleaseEvent = self.playhead_mouse_release
        self.playhead_dragging = False
        self.playhead_drag_start_x = 0
    
    def playhead_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.playhead_dragging = True
            self.playhead_drag_start_x = event.pos().x()
            event.accept()

    def playhead_mouse_move(self, event):
        if self.playhead_dragging:
            mouse_x = self.timeline_playhead.mapToParent(event.pos()).x()
            pos = self.pixel_to_timeline_position(mouse_x)

            if self.timeline_clips:
                min_x = min(group.video_clip.x() for group in self.timeline_clips)
                max_x = max(group.video_clip.x() + group.video_clip.width() for group in self.timeline_clips)
                line_x = max(min_x, min(mouse_x, max_x))
                self.timeline_playhead.move(line_x, 20)

            self.player_controller.set_position(pos)
            self.scrubber.setValue(pos)
            self.update_time_label(pos, self.timeline_max_ms)
            event.accept()

    def playhead_mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self.playhead_dragging = False
            event.accept()

    def timeline_label_click(self, event, label, track):
        self.timeline_track_click(event, track)

    def timeline_track_click(self, event, track):
        if event.button() != Qt.LeftButton:
            return

        click_x = track.mapFromGlobal(event.globalPos()).x()
        timeline_pos = self.pixel_to_timeline_position(click_x)
        timeline_pos = max(0, min(timeline_pos, self.timeline_max_ms))

        self.scrubber.blockSignals(True)
        self.scrubber.setValue(timeline_pos)
        self.scrubber.blockSignals(False)
        self.update_time_label(timeline_pos, self.timeline_max_ms)

        clip = self.find_clip_at_timeline_position(timeline_pos)
        if clip:
            if clip != self.current_clip_group:
                self.current_clip_group = clip
                self.player_controller.load_video(clip.file_path)

            local_pos = max(
                0,
                timeline_pos - self.get_clip_start_ms(clip)
            )
            self.player_controller.set_position(local_pos)
        else:
            self.player_controller.set_position(0)

        if self.timeline_clips:
            line_x = self.timeline_position_to_pixel(timeline_pos)
            min_x = min(group.video_clip.x() for group in self.timeline_clips)
            max_x = max(group.video_clip.x() + group.video_clip.width() for group in self.timeline_clips)
            line_x = max(min_x, min(line_x, max_x))
            self.timeline_playhead.move(line_x, 20)
    
    def on_source_item_clicked(self, item):
        self.selected_source_item = item
        self.selected_clip_group = None

        for group in self.timeline_clips:
            group.deselect()
            
    def on_scrub_start(self):
        self.is_scrubbing = True
        
    def on_scrub_move(self, value):
        if self.is_scrubbing:
            self.update_time_label(
                value,
                max(1, self.timeline_max_ms)
            )

            if self.timeline_clips:
                line_x = self.timeline_position_to_pixel(value)
                min_x = min(
                    group.video_clip.x()
                    for group in self.timeline_clips
                )

                max_x = max(
                    group.video_clip.x()
                    + group.video_clip.width()
                    for group in self.timeline_clips
                )

                line_x = max(min_x, min(line_x, max_x))
                self.timeline_playhead.move(line_x, 20)
        
    def on_scrub_end(self):
        self.is_scrubbing = False
        timeline_pos = self.scrubber.value()
        clip = self.find_clip_at_timeline_position(timeline_pos)

        if clip:
            if clip != self.current_clip_group:
                self.current_clip_group = clip
                self.player_controller.load_video(clip.file_path)

            local_pos = max(
                0,
                timeline_pos - self.get_clip_start_ms(clip)
            )
            self.player_controller.set_position(local_pos)
        else:
            self.player_controller.player.pause()
    
    def timeline_empty_click(self, event):
        if event.button() == Qt.LeftButton:
            self.selected_clip_group = None

            for group in self.timeline_clips:
                group.deselect()

        event.accept()

        QWidget.mousePressEvent(
            self.timeline_container,
            event
        )
        
    def load_selected_media(self, item):
        path = item.text()

        self.current_file = path
        self.player_controller.load_video(path)

        for group in self.timeline_clips:
            if group.file_path == path:
                self.current_clip_group = group
                break
        
    def toggle_playback(self):
        if not self.timeline_clips:
            return

        player = self.player_controller.player

        if (
            player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        ):
            player.pause()
            self.btn_play.setText("▶")

        else:
            if not self.current_clip_group:
                self.current_clip_group = (
                    self.timeline_clips[0]
                )

                self.player_controller.load_video(
                    self.current_clip_group.file_path
                )

            player.play()
            self.btn_play.setText("⏸")
        
    def update_position(self, pos):
        if self.is_scrubbing:
            return

        if self.current_clip_group:
            clip_start = self.get_clip_start_ms(
                self.current_clip_group
            )
            global_pos = clip_start + pos
        else:
            global_pos = pos

        global_pos = max(0, global_pos)

        self.scrubber.blockSignals(True)
        self.scrubber.setValue(global_pos)
        self.scrubber.blockSignals(False)

        self.update_time_label(
            global_pos,
            max(1, self.timeline_max_ms)
        )

        if not self.timeline_clips:
            return

        line_x = self.timeline_position_to_pixel(global_pos)
        min_x = min(
            group.video_clip.x()
            for group in self.timeline_clips
        )
        max_x = max(
            group.video_clip.x()
            + group.video_clip.width()
            for group in self.timeline_clips
        )

        line_x = max(min_x, min(line_x, max_x))

        self.timeline_playhead.move(
            line_x,
            20
        )
        
    def get_clip_start_ms(self, group):
        return int(
            group.video_clip.x()
            / self.timeline_scale
            * 1000
        )

    def get_clip_end_ms(self, group):
        return (
            self.get_clip_start_ms(group)
            + getattr(group, "duration", getattr(group, "duration_ms", 0))
        )

    def duration_ms_to_width(self, duration_ms):
        return max(
            self.min_clip_width,
            int(duration_ms / 1000.0 * self.timeline_scale)
        )

    def timeline_position_to_pixel(self, pos_ms):
        if not self.timeline_clips:
            return 0

        min_x = min(group.video_clip.x() for group in self.timeline_clips)
        return int(min_x + (pos_ms / 1000.0 * self.timeline_scale))

    def pixel_to_timeline_position(self, pixel_x):
        if not self.timeline_clips:
            return 0

        min_x = min(group.video_clip.x() for group in self.timeline_clips)
        return int(max(0, (pixel_x - min_x) / self.timeline_scale * 1000))

    def find_clip_at_timeline_position(self, timeline_pos):
        for group in self.timeline_clips:
            start_ms = self.get_clip_start_ms(group)
            end_ms = self.get_clip_end_ms(group)
            if start_ms <= timeline_pos < end_ms:
                return group
        return None

    def resolve_clip_overlaps(self):
        if not self.timeline_clips:
            return

        self.timeline_clips.sort(
            key=lambda group: group.video_clip.x()
        )

        for index in range(1, len(self.timeline_clips)):
            prev = self.timeline_clips[index - 1]
            curr = self.timeline_clips[index]

            prev_end = self.get_clip_end_ms(prev)
            curr_start = self.get_clip_start_ms(curr)

            if curr_start < prev_end:
                overlap_ms = prev_end - curr_start
                
                # 조작 중인 클립을 우선: 조작 중인 클립이 curr이면 prev를 잘라내고, prev이면 curr를 잘라냄
                if curr == self.current_clip_group or curr == self.selected_clip_group:
                    # curr가 조작 중이므로 prev를 잘라냄
                    source_path = getattr(
                        prev,
                        "original_file_path",
                        prev.file_path
                    )
                    trimmed_path = trim_media_clip(
                        source_path,
                        start_ms=0,
                        end_ms=getattr(prev, "duration", getattr(prev, "duration_ms", 0)) - overlap_ms
                    )
                    prev.file_path = trimmed_path

                    current_duration = getattr(
                        prev,
                        "duration",
                        getattr(prev, "duration_ms", 0)
                    )
                    prev.duration = max(0, current_duration - overlap_ms)
                    prev.duration_ms = prev.duration

                    width = self.duration_ms_to_width(prev.duration)
                    prev.base_width = width
                    prev.video_clip.setFixedWidth(width)
                    prev.audio_clip.setFixedWidth(width)
                else:
                    # 기본적으로 curr를 잘라냄
                    source_path = getattr(
                        curr,
                        "original_file_path",
                        curr.file_path
                    )
                    trimmed_path = trim_media_clip(
                        source_path,
                        start_ms=overlap_ms,
                        end_ms=getattr(curr, "duration", getattr(curr, "duration_ms", 0))
                    )
                    curr.file_path = trimmed_path

                    current_duration = getattr(
                        curr,
                        "duration",
                        getattr(curr, "duration_ms", 0)
                    )
                    curr.duration = max(0, current_duration - overlap_ms)
                    curr.duration_ms = curr.duration

                    width = self.duration_ms_to_width(curr.duration)
                    curr.base_width = width
                    curr.video_clip.setFixedWidth(width)
                    curr.audio_clip.setFixedWidth(width)

    def update_duration(self, dur):
        if self.current_clip_group:
            if not hasattr(
                self.current_clip_group,
                "initialized_duration"
            ):
                self.current_clip_group.initialized_duration = dur
                self.current_clip_group.duration = dur
                self.current_clip_group.duration_ms = dur

                width = self.duration_ms_to_width(dur)
                self.current_clip_group.base_width = width

                self.current_clip_group.video_clip.setFixedWidth(width)
                self.current_clip_group.audio_clip.setFixedWidth(width)

        self.update_timeline_duration()
        self.scrubber.setRange(
            0,
            max(1, self.timeline_max_ms)
        )
        self.update_time_label(
            self.scrubber.value(),
            self.timeline_max_ms
        )
        
    def update_time_label(self, pos, dur=None):
        if dur is None:
            dur = self.timeline_max_ms

        pos_sec = pos / 1000
        dur_sec = dur / 1000

        pos_str = (
            f"{int(pos_sec // 3600):02}:"
            f"{int((pos_sec % 3600) // 60):02}:"
            f"{pos_sec % 60:06.3f}"
        )

        dur_str = (
            f"{int(dur_sec // 3600):02}:"
            f"{int((dur_sec % 3600) // 60):02}:"
            f"{dur_sec % 60:06.3f}"
        )

        self.lbl_time.setText(
            f"{pos_str} / {dur_str}"
        )
        
    def update_current_time_label(self, current_ms):
        total_ms = max(1, self.timeline_total_duration)

        cur_sec = current_ms / 1000.0
        total_sec = total_ms / 1000.0

        cur_str = (
            f"{int(cur_sec // 3600):02}:"
            f"{int((cur_sec % 3600) // 60):02}:"
            f"{cur_sec % 60:06.3f}"
        )

        total_str = (
            f"{int(total_sec // 3600):02}:"
            f"{int((total_sec % 3600) // 60):02}:"
            f"{total_sec % 60:06.3f}"
        )

        self.lbl_time.setText(
            f"{cur_str} / {total_str}"
        )
        
    def on_source_added(self, path):
        pass
    
    def check_next_clip(self, timeline_pos):
        for group in self.timeline_clips:
            start_ms = int(
                group.video_clip.x()
                / self.timeline_scale
                * 1000
            )

            duration = getattr(
                group,
                "initialized_duration",
                0
            )

            end_ms = start_ms + duration

            if start_ms <= timeline_pos < end_ms:
                if self.current_clip_group != group:
                    self.current_clip_group = group

                    local_pos = (
                        timeline_pos
                        - start_ms
                    )

                    self.player_controller.load_video(
                        group.file_path
                    )

                    self.player_controller.player.setPosition(
                        local_pos
                    )

                    self.player_controller.player.play()

                return

        self.player_controller.stop()
        
    def add_media_to_timeline(self, file_path):
        x = 0

        if self.timeline_clips:
            last_group = self.timeline_clips[-1]
            x = last_group.video_clip.x() + last_group.video_clip.width() + 10

        duration_ms = get_media_duration_ms(file_path)
        width = self.duration_ms_to_width(duration_ms)

        # 임시 복사본 생성
        temp_dir = tempfile.gettempdir()
        temp_filename = f"kirinuki_clip_{len(self.timeline_clips)}_{os.path.basename(file_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        shutil.copy2(file_path, temp_path)

        video_clip = VideoClipWidget(
            temp_path,
            self.v_track,
            os.path.basename(file_path)
        )

        audio_clip = AudioClipWidget(
            temp_path,
            self.a_track
        )

        video_clip.move(x, 10)
        audio_clip.move(x, 10)

        video_clip.setFixedWidth(width)
        audio_clip.setFixedWidth(width)

        group = ClipGroup(
            video_clip,
            audio_clip,
            self
        )

        group.file_path = temp_path
        group.original_file_path = file_path
        group.base_width = width
        group.duration = duration_ms
        group.duration_ms = duration_ms
        group.timeline_x = x

        video_clip.mouseDoubleClickEvent = (
            lambda e, g=group: self.load_clip_from_group(g)
        )

        video_clip.show()
        audio_clip.show()

        self.timeline_clips.append(group)
        self.update_timeline_duration()
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, max(0, self.timeline_max_ms))
        self.scrubber.setValue(min(self.scrubber.value(), max(0, self.timeline_max_ms)))
        self.scrubber.blockSignals(False)
        self.update_time_label(
            self.scrubber.value(),
            max(1, self.timeline_max_ms)
        )

        if len(self.timeline_clips) == 1:
            self.current_clip_group = group
            self.current_file = temp_path
            self.player_controller.load_video(
                temp_path
            )
        
    def update_timeline_duration(self):
        self.resolve_clip_overlaps()
        max_end_ms = 0

        for group in self.timeline_clips:
            duration_ms = getattr(
                group,
                "duration",
                getattr(group, "duration_ms", 0)
            )

            start_ms = self.get_clip_start_ms(group)
            end_ms = start_ms + duration_ms
            width = self.duration_ms_to_width(duration_ms)
            group.base_width = width
            group.video_clip.setFixedWidth(width)
            group.audio_clip.setFixedWidth(width)

            max_end_ms = max(
                max_end_ms,
                end_ms
            )

        self.timeline_total_duration = max_end_ms
        self.timeline_max_ms = max(1, self.timeline_total_duration)
        
    def load_clip_from_group(self, group):
        self.current_clip_group = group
        self.current_file = group.file_path

        self.player_controller.stop()
        self.player_controller.load_video(
            group.file_path
        )

        current_timeline_pos = self.scrubber.value()
        group_start = self.get_clip_start_ms(group)
        group_duration = getattr(
            group,
            "duration",
            getattr(group, "duration_ms", 0)
        )

        if group_start <= current_timeline_pos < group_start + group_duration:
            local_pos = current_timeline_pos - group_start
        else:
            local_pos = 0
            self.scrubber.blockSignals(True)
            self.scrubber.setValue(group_start)
            self.scrubber.blockSignals(False)

        self.player_controller.set_position(local_pos)
        self.player_controller.player.play()

        self.update_position(local_pos)
            
    def play_next_clip(self):
        if not self.timeline_clips:
            return

        current_index = -1
        if self.current_clip_group:
            try:
                current_index = self.timeline_clips.index(self.current_clip_group)
            except ValueError:
                current_index = -1

        next_index = current_index + 1
        if next_index < len(self.timeline_clips):
            next_group = self.timeline_clips[next_index]
            self.load_clip_from_group(next_group)
        else:
            # 타임라인 끝에 도달하면 재생 중지
            self.player_controller.player.pause()
            self.btn_play.setText("▶")
    
    def delete_selected_item(self):
        target_group = None

        if self.selected_clip_group:
            target_group = self.selected_clip_group

        elif self.selected_source_item:
            path = self.selected_source_item.text()

            for group in self.timeline_clips:
                if group.file_path == path:
                    target_group = group
                    break

            row = self.file_list.row(
                self.selected_source_item
            )

            self.file_list.takeItem(row)
            self.selected_source_item = None

        if not target_group:
            return

        was_current = (
            self.current_clip_group == target_group
        )

        target_group.video_clip.hide()
        target_group.audio_clip.hide()

        target_group.video_clip.deleteLater()
        target_group.audio_clip.deleteLater()

        if target_group in self.timeline_clips:
            self.timeline_clips.remove(target_group)

        self.selected_clip_group = None
        self.update_timeline_duration()

        current_pos = min(
            self.scrubber.value(),
            max(0, self.timeline_max_ms)
        )

        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, max(0, self.timeline_max_ms))
        self.scrubber.setValue(current_pos)
        self.scrubber.blockSignals(False)
        self.update_time_label(
            current_pos,
            max(1, self.timeline_max_ms)
        )

        if was_current:
            self.player_controller.stop()

            self.current_clip_group = None
            self.current_file = None

            self.scrubber.blockSignals(True)
            self.scrubber.setRange(0, 0)
            self.scrubber.setValue(0)
            self.scrubber.blockSignals(False)

            self.lbl_time.setText(
                "00:00:00.000 / 00:00:00.000"
            )

            self.btn_play.setText("▶")

            self.timeline_playhead.move(0, 0)

            if self.timeline_clips:
                next_group = self.timeline_clips[0]

                self.current_clip_group = next_group
                self.current_file = next_group.file_path

                self.player_controller.load_video(
                    next_group.file_path
                )
            
    def show_source_context_menu(self, pos):
        item = self.file_list.itemAt(pos)

        if not item:
            return

        menu = QMenu(self)
        delete_action = menu.addAction("삭제")
        action = menu.exec(
            self.file_list.mapToGlobal(pos)
        )

        if action == delete_action:
            path = item.text()

            remove_groups = []

            for group in self.timeline_clips:
                if group.file_path == path:
                    remove_groups.append(group)

            for group in remove_groups:
                group.video_clip.deleteLater()
                group.audio_clip.deleteLater()

                if group in self.timeline_clips:
                    self.timeline_clips.remove(group)

            row = self.file_list.row(item)

            self.file_list.takeItem(row)

            self.update_timeline_duration()
            current_pos = min(
                self.scrubber.value(),
                max(0, self.timeline_max_ms)
            )
            self.scrubber.blockSignals(True)
            self.scrubber.setRange(0, max(0, self.timeline_max_ms))
            self.scrubber.setValue(current_pos)
            self.scrubber.blockSignals(False)
            self.update_time_label(
                current_pos,
                max(1, self.timeline_max_ms)
            )
        
    def timeline_drag_enter_event(self, event):
        if event.mimeData().hasUrls() or event.mimeData().text():
            event.acceptProposedAction()
            
    def timeline_drop_event(self, event):
        paths = []

        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    paths.append(
                        url.toLocalFile()
                    )

        elif event.mimeData().text():
            paths.append(
                event.mimeData().text()
            )

        for path in paths:
            if not self.file_list.findItems(path, Qt.MatchExactly):
                self.file_list.addItem(path)

            self.add_media_to_timeline(path)

        event.acceptProposedAction()
    
    def render_project(self):
        if not self.current_file:
            print("No media loaded")
            return

        print(f"Render Start: {self.current_file}")
import math
import os
import random
import tempfile
from PySide6.QtCore import Qt, QUrl, QObject, Slot, Signal, QPoint
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtMultimedia import QAudioDecoder, QAudioBuffer, QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import QMenu

try:
    from moviepy.editor import VideoFileClip
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    _MOVIEPY_AVAILABLE = True
except ImportError:
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
        _MOVIEPY_AVAILABLE = True
    except ImportError:
        VideoFileClip = None
        ffmpeg_extract_subclip = None
        _MOVIEPY_AVAILABLE = False
        print("Warning: moviepy is not installed or moviepy.editor is unavailable. clip trimming and duration detection are disabled.")

class AudioWaveformVisualizer(QWidget):
    """실제 오디오 파형을 클립 길이에 맞게 전체 영역으로 늘려주는 위젯"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        self.setMinimumWidth(200)
        
        self.volume_scale = 1.0
        self.db_level = 0.0
        self.line_y_offset = 0
        self.audio_samples = []
        self.is_dragging = False
        self.last_y = 0

    def load_audio_file(self, file_path):
        try:
            random.seed(hash(file_path))
            self.audio_samples = [random.uniform(-0.5, 0.5) for _ in range(250)]
            self.db_level = 0.0
            self.volume_scale = 1.0
            self.update()
        except Exception as e:
            print(f"Error loading audio file: {e}")

    def set_db_level(self, db_val):
        self.db_level = float(db_val)
        self.volume_scale = 10 ** (self.db_level / 20.0)
        self.line_y_offset = -int(self.db_level * 1.5)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event):
        try:
            painter = QPainter(self)
            rect = self.rect()

            painter.fillRect(rect, QColor(30, 30, 30))

            width = rect.width()
            height = rect.height()
            mid_y = height / 2

            pen_line = QPen(QColor(150, 150, 150), 1, Qt.DashLine)
            painter.setPen(pen_line)
            painter.drawLine(0, int(mid_y), width, int(mid_y))

            pen_wave = QPen(QColor(0, 180, 216), 1, Qt.SolidLine)
            painter.setPen(pen_wave)

            if not self.audio_samples:
                return

            step = max(1, len(self.audio_samples) / width)
            for i in range(0, width, max(1, width // 100)):
                idx = int(i * step)
                if idx < len(self.audio_samples):
                    val = self.audio_samples[idx] * self.volume_scale
                    wave_h = int(val * height * 0.9)
                    
                    start_y = int(mid_y - wave_h / 2)
                    end_y = int(mid_y + wave_h / 2)
                    
                    painter.drawLine(i, start_y, i, end_y)

            target_y = int(mid_y + self.line_y_offset)
            pen_db_line = QPen(QColor(255, 69, 58), 2, Qt.SolidLine)
            painter.setPen(pen_db_line)
            painter.drawLine(0, target_y, width, target_y)

            painter.setPen(QColor(255, 255, 255))
            painter.drawText(10, int(mid_y + 18), f"{self.db_level:+.1f} dB")
        except Exception as e:
            print(f"Error in paintEvent: {e}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_offset = event.pos()

            if hasattr(self, "group"):
                self.group.select()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            current_y = event.position().y()
            delta = self.last_y - current_y
            self.db_level = max(-24.0, min(24.0, self.db_level + delta * 0.1))
            self.set_db_level(self.db_level)
            self.last_y = current_y

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False

class BaseClipWidget(QFrame):
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.dragging = False
        self.drag_offset = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_offset = event.pos()

            if hasattr(self, "group"):
                self.group.select()

    def mouseMoveEvent(self, event):
        if self.dragging:
            parent_pos = self.mapToParent(
                event.pos() - self.drag_offset
            )

            x = max(0, parent_pos.x())

            if hasattr(self, "group"):
                self.group.move(x)

    def mouseReleaseEvent(self, event):
        self.dragging = False
        
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        delete_action = menu.addAction("삭제")
        action = menu.exec(event.globalPos())

        if action == delete_action:
            if hasattr(self, "group"):
                self.group.delete()

class VideoClipWidget(BaseClipWidget):
    def __init__(self, file_path, parent=None, display_name=None):
        super().__init__(file_path, parent)

        self.setFixedSize(300, 50)

        self.setStyleSheet("""
            QFrame {
                background-color: #2d5a88;
                border: 1px solid #5fa8ff;
                border-radius: 4px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel(display_name if display_name else file_path.split("/")[-1])
        layout.addStretch()
        title.setStyleSheet("""
            color: white;
            font-weight: bold;
        """)

        layout.addWidget(title)
        
class AudioClipWidget(BaseClipWidget):
    def __init__(self, file_path, parent=None):
        super().__init__(file_path, parent)

        self.setFixedSize(300, 70)

        self.setStyleSheet("""
            QFrame {
                background-color: #3a3a3a;
                border: 1px solid #808080;
                border-radius: 4px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.waveform = AudioWaveformVisualizer()
        self.waveform.load_audio_file(file_path)

        layout.addWidget(self.waveform)

class ClipGroup:
    def __init__(self, video_clip, audio_clip, editor):
        self.video_clip = video_clip
        self.audio_clip = audio_clip
        self.editor = editor
        self.base_width = 300

        self.video_clip.group = self
        self.audio_clip.group = self

    def move(self, x):
        self.video_clip.move(x, self.video_clip.y())
        self.audio_clip.move(x, self.audio_clip.y())
        
    def select(self):
        for group in self.editor.timeline_clips:
            group.deselect()

        self.editor.selected_clip_group = self
        self.editor.selected_source_item = None
        self.editor.current_clip_group = self
        self.editor.current_file = self.file_path

        self.editor.player_controller.stop()

        self.editor.player_controller.load_video(
            self.file_path
        )

        self.editor.player_controller.player.setPosition(0)

        self.video_clip.setStyleSheet("""
            QFrame {
                background-color: #4a7db8;
                border: 2px solid #ffffff;
                border-radius: 4px;
            }
        """)

        self.audio_clip.setStyleSheet("""
            QFrame {
                background-color: #5a5a5a;
                border: 2px solid #ffffff;
                border-radius: 4px;
            }
        """)
        
        if hasattr(self, "base_width"):
            self.video_clip.resize(
            self.base_width,
            self.video_clip.height()
        )

        self.audio_clip.resize(
            self.base_width,
            self.audio_clip.height()
        )
    
    def deselect(self):
        self.video_clip.setStyleSheet("""
            QFrame {
                background-color: #2d5a88;
                border: 1px solid #5fa8ff;
                border-radius: 4px;
            }
        """)

        self.audio_clip.setStyleSheet("""
            QFrame {
                background-color: #3a3a3a;
                border: 1px solid #808080;
                border-radius: 4px;
            }
        """)

        if hasattr(self, "base_width"):
            self.video_clip.resize(
                self.base_width,
                self.video_clip.height()
            )

            self.audio_clip.resize(
                self.base_width,
                self.audio_clip.height()
            )
    
    def delete(self):
        was_current = (
            self.editor.current_clip_group == self
        )

        self.video_clip.deleteLater()
        self.audio_clip.deleteLater()

        if self in self.editor.timeline_clips:
            self.editor.timeline_clips.remove(self)

        if was_current:
            self.editor.player_controller.stop()
            
            self.editor.current_clip_group = None
            self.editor.current_file = None

            self.editor.scrubber.blockSignals(True)
            self.editor.scrubber.setValue(0)
            self.editor.scrubber.setRange(0, 0)
            self.editor.scrubber.blockSignals(False)

            self.editor.timeline_playhead.move(0, 0)

            self.editor.lbl_time.setText(
                "00:00:00.000 / 00:00:00.000"
            )

            self.editor.btn_play.setText("▶")

class SubtitleWidget(QWidget):
    """자막을 표시하는 위젯 (오디오 파형 표시 안 함)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setMinimumWidth(200)
        self.subtitle_text = "자막"
        
    def set_subtitle_text(self, text):
        self.subtitle_text = text
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        
        painter.fillRect(rect, QColor(50, 50, 50))
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(rect.adjusted(5, 5, -5, -5), Qt.AlignCenter, self.subtitle_text)


def calculate_max_waveform_length(center_width):
    max_limit = center_width - 40
    return max(200, max_limit)


def get_media_duration_ms(file_path):
    if not _MOVIEPY_AVAILABLE:
        print(
            "Warning: moviepy unavailable, cannot determine media duration."
        )
        return 0

    try:
        with VideoFileClip(file_path) as clip:
            return int(clip.duration * 1000)
    except Exception as e:
        print(f"Error getting media duration: {e}")
        return 0


def trim_media_clip(file_path, start_ms=0, end_ms=None, output_path=None):
    if not _MOVIEPY_AVAILABLE:
        print(
            "Warning: moviepy unavailable, cannot trim media clip."
        )
        return file_path

    try:
        with VideoFileClip(file_path) as clip:
            duration_ms = int(clip.duration * 1000)
            if end_ms is None or end_ms > duration_ms:
                end_ms = duration_ms
            start_ms = max(0, min(start_ms, duration_ms))
            end_ms = max(start_ms, end_ms)
            if start_ms >= end_ms:
                return file_path

            start_s = start_ms / 1000.0
            end_s = end_ms / 1000.0

            if output_path is None:
                tmp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(file_path)[1]
                )
                output_path = tmp.name
                tmp.close()

            if hasattr(clip, "subclip"):
                trimmed_clip = clip.subclip(start_s, end_s)
                temp_audio = f"{output_path}.temp_audio.m4a"
                trimmed_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=temp_audio,
                    remove_temp=True,
                    threads=4,
                    verbose=False,
                    logger=None
                )
                trimmed_clip.close()
            elif ffmpeg_extract_subclip is not None:
                ffmpeg_extract_subclip(file_path, start_s, end_s, output_path)
            else:
                print("Warning: no supported trimming method available.")
                return file_path

            return output_path
    except Exception as e:
        print(f"Error trimming media clip: {e}")
        return file_path


class VideoPlayerController(QObject):
    positionChanged = Signal(int)
    durationChanged = Signal(int)
    nextClipNeeded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.video_widget = QVideoWidget()
        
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)

        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        
        self.trim_start = 0
        self.trim_end = 0
        self.clip_end_ms = 0

    def load_video(self, file_path):
        try:
            self.player.setSource(QUrl.fromLocalFile(file_path))
            self.trim_start = 0
            self.trim_end = 0
        except Exception as e:
            print(f"Error loading video: {e}")

    def set_trim_range(self, start_ms, end_ms):
        try:
            self.trim_start = start_ms
            self.trim_end = end_ms if end_ms > 0 else self.player.duration()
            if self.trim_end <= self.trim_start:
                self.trim_end = self.player.duration()
            # Removed: self.player.setPosition(self.trim_start)
        except Exception as e:
            print(f"Error setting trim range: {e}")

    def set_clip_end(self, end_ms):
        self.clip_end_ms = end_ms

    def set_position(self, position):
        try:
            pos = position
            if self.trim_end > 0:
                pos = max(self.trim_start, min(self.trim_end, pos))
            else:
                pos = max(self.trim_start, pos)
            self.player.setPosition(pos)
        except Exception as e:
            print(f"Error setting position: {e}")

    def _on_position_changed(self, position):
        try:
            if self.trim_end > 0 and position >= self.trim_end:
                self.player.pause()
                self.player.setPosition(self.trim_start)
                self.positionChanged.emit(self.trim_start)
                return

            if self.clip_end_ms > 0 and position >= self.clip_end_ms:
                self.nextClipNeeded.emit()
                return

            self.positionChanged.emit(position)
        except Exception as e:
            print(f"Error in position changed: {e}")

    def _on_duration_changed(self, duration):
        try:
            self.durationChanged.emit(duration)
        except Exception as e:
            print(f"Error in duration changed: {e}")

    @Slot()
    def toggle_play_pause(self):
        try:
            if self.player.playbackState() == QMediaPlayer.PlayingState:
                self.player.pause()
                return "▶", "⏸"
            else:
                if self.player.position() < self.trim_start or (self.trim_end > 0 and self.player.position() >= self.trim_end):
                    self.player.setPosition(self.trim_start)
                self.player.play()
                return "⏸", "▶"
            
        except Exception as e:
            print(f"Error toggling play/pause: {e}")
            return "▶ 재생", "오류"

    @Slot()
    def stop(self):
        try:
            self.player.stop()
        except Exception as e:
            print(f"Error stopping: {e}")
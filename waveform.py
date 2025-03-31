from pydub import AudioSegment
from PyQt5.QtWidgets import QWidget, QToolTip
from PyQt5.QtCore import Qt, QRect, QPoint, QSize, QEvent
from PyQt5.QtGui import QPainter, QPen, QColor, QMouseEvent, QPixmap
from utils import seconds_to_formatted
import struct

class WaveformProgressBar(QWidget):
    """
    A waveform + progress slider that shows the waveform, a red playhead, 
    and now a translucent overlay indicating the loop boundaries.
    """
    def __init__(self, media_path, parent=None):
        super().__init__(parent)
        self.media_path = media_path
        self.samples = []
        self.duration_ms = 0

        # Current playback position in ms.
        self.current_position_ms = 0
        # Callback for seeking.
        self.seekRequestedCallback = None
        # Cached waveform pixmap.
        self.wave_pixmap = QPixmap()

        # For loop boundaries (in ms); if not set, they are None.
        self.loop_start_ms = None
        self.loop_end_ms = None

        self.setMinimumHeight(100)
        self.setAutoFillBackground(True)
        self.setMouseTracking(True)  # Ensure hover events fire without clicks.

        self.extract_audio_samples()
        self.precompute_min_max()
        self.build_waveform_pixmap()

        # Flag for click-drag seeking.
        self.dragging = False

    def extract_audio_samples(self):
        audio = AudioSegment.from_file(self.media_path, format=None)
        audio = audio.set_channels(1)
        self.duration_ms = len(audio)
        if audio.sample_width != 2:
            audio = audio.set_sample_width(2)
        raw_data = audio.raw_data
        total_samples = len(raw_data) // 2
        self.samples = struct.unpack("<" + "h" * total_samples, raw_data)

    def precompute_min_max(self):
        if not self.samples or self.duration_ms == 0:
            self.min_array = []
            self.max_array = []
            return

        self.min_array = []
        self.max_array = []
        chunk_size = 400  # number of samples per chunk
        idx = 0
        tmp_min = None
        tmp_max = None
        for s in self.samples:
            if tmp_min is None or s < tmp_min:
                tmp_min = s
            if tmp_max is None or s > tmp_max:
                tmp_max = s
            idx += 1
            if idx >= chunk_size:
                self.min_array.append(tmp_min)
                self.max_array.append(tmp_max)
                tmp_min, tmp_max = None, None
                idx = 0
        if tmp_min is not None and tmp_max is not None:
            self.min_array.append(tmp_min)
            self.max_array.append(tmp_max)

    def build_waveform_pixmap(self):
        w = max(1, self.width())
        h = max(1, self.height())
        self.wave_pixmap = QPixmap(w, h)
        self.wave_pixmap.fill(Qt.black)

        if not self.samples or not self.min_array or not self.max_array:
            return

        painter = QPainter(self.wave_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        total_chunks = len(self.min_array)
        pen_wave = QPen(QColor(0, 255, 0))
        pen_wave.setWidth(1)
        painter.setPen(pen_wave)

        global_min = min(self.min_array)
        global_max = max(self.max_array)
        global_range = float(global_max - global_min) if global_max != global_min else 1

        for x in range(w):
            idx = int(x / float(w) * total_chunks)
            if idx >= total_chunks:
                idx = total_chunks - 1
            c_min = self.min_array[idx]
            c_max = self.max_array[idx]
            min_norm = (c_min - global_min) / global_range
            max_norm = (c_max - global_min) / global_range
            y_min = int(min_norm * h)
            y_max = int(max_norm * h)
            painter.drawLine(x, y_min, x, y_max)

        painter.end()

    def paintEvent(self, event):
        painter = QPainter(self)
        # Draw cached waveform pixmap.
        painter.drawPixmap(0, 0, self.wave_pixmap)
        w = self.width()
        h = self.height()

        # If loop boundaries are set, draw a translucent overlay.
        if self.loop_start_ms is not None and self.loop_end_ms is not None:
            x1 = int((self.loop_start_ms / float(self.duration_ms)) * w)
            x2 = int((self.loop_end_ms / float(self.duration_ms)) * w)
            # Ensure x1 <= x2.
            if x1 > x2:
                x1, x2 = x2, x1
            loop_color = QColor(255, 255, 0, 150)
            painter.fillRect(x1, 0, x2 - x1, h, loop_color)

        # Draw red playhead.
        if self.duration_ms > 0:
            ratio = self.current_position_ms / float(self.duration_ms)
            x_pos = int(ratio * w)
            pen_head = QPen(QColor(255, 0, 0))
            pen_head.setWidth(2)
            painter.setPen(pen_head)
            painter.drawLine(x_pos, 0, x_pos, h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.build_waveform_pixmap()
        self.update()

    def set_current_position(self, ms):
        if self.duration_ms > 0:
            ms = max(0, min(ms, self.duration_ms))
        self.current_position_ms = ms
        self.update()

    def set_loop_boundaries(self, loop_start_seconds, loop_end_seconds):
        """Call this method to update loop boundaries (in seconds)."""
        self.loop_start_ms = int(loop_start_seconds * 1000)
        self.loop_end_ms = int(loop_end_seconds * 1000)
        self.update()

    def mouseMoveEvent(self, event):
        if self.duration_ms > 0:
            x = event.x()
            width = self.width()
            hovered_ms = (x / width) * self.duration_ms
            QToolTip.showText(event.globalPos(), seconds_to_formatted(hovered_ms / 1000.0))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.duration_ms > 0:
            self.dragging = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.duration_ms > 0 and self.dragging:
            ratio = event.x() / float(self.width())
            new_time = int(ratio * self.duration_ms)
            if self.seekRequestedCallback:
                self.seekRequestedCallback(new_time)
            self.dragging = False
        super().mouseReleaseEvent(event)
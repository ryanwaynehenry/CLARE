from pydub import AudioSegment
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRect, QPoint, QSize
from PyQt5.QtGui import QPainter, QPen, QColor, QMouseEvent, QPixmap
import struct

class WaveformProgressBar(QWidget):
    """
    A combined waveform + progress "slider" widget that:
      - Precomputes min/max data for the entire audio file.
      - Renders the waveform once to a QPixmap (heavy operation).
      - On each paint, draws the cached pixmap plus a quick red line for the playhead.
      - Allows click-to-seek by letting the user click on the widget.
    """
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.samples = []
        self.duration_ms = 0

        # The current playback position in ms
        self.current_position_ms = 0

        # Callback for seeking
        self.seekRequestedCallback = None

        # We'll store the waveform in a QPixmap for quick repaint
        self.wave_pixmap = QPixmap()

        self.setMinimumHeight(100)
        self.setAutoFillBackground(True)

        self.extract_audio_samples()
        # Build the precomputed min/max arrays
        self.precompute_min_max()
        # Render once to pixmap
        self.build_waveform_pixmap()

    # ---------------------
    # Step 1: Extract Audio
    # ---------------------
    def extract_audio_samples(self):
        audio = AudioSegment.from_file(self.video_path, format=None)
        audio = audio.set_channels(1)
        self.duration_ms = len(audio)

        if audio.sample_width != 2:
            audio = audio.set_sample_width(2)

        raw_data = audio.raw_data
        total_samples = len(raw_data) // 2
        self.samples = struct.unpack("<" + "h" * total_samples, raw_data)

    # ---------------------
    # Step 2: Precompute Min/Max
    # ---------------------
    def precompute_min_max(self):
        """Compute min/max arrays for the entire audio, 
           to quickly build the waveform at any width."""
        if not self.samples or self.duration_ms == 0:
            self.min_array = []
            self.max_array = []
            return

        self.min_array = []
        self.max_array = []

        # We'll store these in a "reasonable resolution" so we don't store huge arrays.
        # E.g., store one min/max for each ~2ms or so. 
        # Or you can store for each sample if file is not too large.
        # This is a design choice. Let's pick a sample chunk size:
        chunk_size = 400  # e.g., # of samples per chunk
        s_min, s_max = 0, 0
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

        # Final chunk if leftover
        if tmp_min is not None and tmp_max is not None:
            self.min_array.append(tmp_min)
            self.max_array.append(tmp_max)

    # ---------------------
    # Step 3: Build QPixmap
    # ---------------------
    def build_waveform_pixmap(self):
        """
        Render the waveform once to a QPixmap.
        We'll adapt to self.width() and self.height() 
        each time we call this (e.g., on resize).
        """
        w = max(1, self.width())
        h = max(1, self.height())

        # Create pixmap + painter
        self.wave_pixmap = QPixmap(w, h)
        self.wave_pixmap.fill(Qt.black)

        if not self.samples or not self.min_array or not self.max_array:
            # nothing to draw
            return

        # We'll step through width w, map to min_array / max_array
        # We want 1 vertical line per x pixel
        painter = QPainter(self.wave_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        total_chunks = len(self.min_array)  # number of stored min/max
        pen_wave = QPen(QColor(0, 255, 0))
        pen_wave.setWidth(1)
        painter.setPen(pen_wave)

        # Find global min/max in these arrays to fully scale
        global_min = min(self.min_array)
        global_max = max(self.max_array)
        global_range = float(global_max - global_min) if global_max != global_min else 1

        for x in range(w):
            # map x => index in [0..total_chunks)
            idx = int(x / float(w) * total_chunks)
            if idx >= total_chunks:
                idx = total_chunks - 1

            c_min = self.min_array[idx]
            c_max = self.max_array[idx]

            # normalize [global_min..global_max] => 0..1
            min_norm = (c_min - global_min) / global_range
            max_norm = (c_max - global_min) / global_range
            y_min = int(min_norm * h)
            y_max = int(max_norm * h)

            # draw vertical line
            painter.drawLine(x, y_min, x, y_max)

        painter.end()

    # ------------------------------------
    # Step 4: PaintEvent => draw QPixmap + red line
    # ------------------------------------
    def paintEvent(self, event):
        # 1) Draw the cached pixmap
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.wave_pixmap)

        # 2) Draw a red vertical line for the playhead
        w = self.width()
        h = self.height()

        if self.duration_ms > 0:
            ratio = self.current_position_ms / float(self.duration_ms)
            x_pos = int(ratio * w)
            pen_head = QPen(QColor(255, 0, 0))
            pen_head.setWidth(2)
            painter.setPen(pen_head)
            painter.drawLine(x_pos, 0, x_pos, h)

    # ------------------------------------
    # Step 5: Update Layout / Resize
    # ------------------------------------
    def resizeEvent(self, event):
        """
        Rebuild the waveform pixmap at the new size 
        so we keep a crisp image.
        """
        super().resizeEvent(event)
        self.build_waveform_pixmap()
        self.update()

    # ------------------------------------
    # Step 6: Playback Position + Click-to-Seek
    # ------------------------------------
    def set_current_position(self, ms):
        """Just move the red line. No re-render of waveform needed."""
        if self.duration_ms > 0:
            ms = max(0, min(ms, self.duration_ms))
        self.current_position_ms = ms
        self.update()  # Only a quick paint: draws pixmap + line

    def mousePressEvent(self, event: QMouseEvent):
        """Seek on left-click."""
        if event.button() == Qt.LeftButton and self.duration_ms > 0:
            ratio = event.x() / float(self.width())
            new_time = int(ratio * self.duration_ms)
            if self.seekRequestedCallback:
                self.seekRequestedCallback(new_time)
        super().mousePressEvent(event)

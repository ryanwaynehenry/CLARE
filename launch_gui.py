import json
import sys
import os
import base64

from PyQt5.QtCore import Qt, QUrl, QEvent, QTimer, QBuffer, QByteArray, QMimeData
from PyQt5.QtGui import QDrag, QKeySequence, QPainter, QPen, QColor, QFont, QMovie
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QLabel, QLineEdit, QDoubleSpinBox,
    QCheckBox, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QShortcut, 
    QGraphicsRectItem, QHeaderView, QFileDialog, QMessageBox, QSplitter
)

from waveform import WaveformProgressBar  # The optimized waveform widget

# A tiny spinner GIF in base64 format.
spinner_gif_data = b'R0lGODlhEAAQAPIAAP///wAAAMLCwkJCQgAAAAAAACH5BAEAAAIALAAAAAAQABAAAAM5SLrc/jDKSau9OOvNu/9gKI5kaZ5oqubL7D0b00Zp3f37gQA7'

def create_spinner():
    spinner_label = QLabel()
    spinner_label.setFixedSize(24, 24)
    movie = QMovie()
    buffer = QBuffer()
    buffer.setData(QByteArray(spinner_gif_data))
    buffer.open(QBuffer.ReadOnly)
    movie.setDevice(buffer)
    spinner_label.setMovie(movie)
    movie.start()
    return spinner_label

class TimeSpinBox(QDoubleSpinBox):
    def textFromValue(self, value):
        if value < 3600:
            minutes = int(value // 60)
            secs = value - minutes * 60
            return f"{minutes}:{secs:04.1f}"
        else:
            hours = int(value // 3600)
            remainder = value % 3600
            minutes = int(remainder // 60)
            secs = remainder - minutes * 60
            return f"{hours}:{minutes:02d}:{secs:04.1f}"

def seconds_to_formatted(seconds):
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds - minutes * 60
        return f"{minutes}:{secs:04.1f}"
    else:
        hours = int(seconds // 3600)
        remainder = seconds % 3600
        minutes = int(remainder // 60)
        secs = remainder - minutes * 60
        return f"{hours}:{minutes:02d}:{secs:04.1f}"

def formatted_to_seconds(time_str):
    parts = time_str.split(":")
    if len(parts) == 2:
        m = int(parts[0])
        s = float(parts[1])
        return m * 60 + s
    elif len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    else:
        raise ValueError("Invalid time format")

class DraggableTableWidget(QTableWidget):
    """A QTableWidget subclass that supports dragging and dropping entire rows with a drop indicator."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragDropMode(QTableWidget.DragDrop)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(False)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.drag_row = -1
        self.dropIndicatorRow = None

    def supportedDropActions(self):
        return Qt.MoveAction

    def startDrag(self, supportedActions):
        selected_items = self.selectedItems()
        if not selected_items:
            return
        self.drag_row = self.currentRow()
        drag = QDrag(self)
        mimeData = QMimeData()
        mimeData.setText("rowdrag")
        drag.setMimeData(mimeData)
        drag.exec_(Qt.MoveAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            pos = event.pos()
            row = self.rowAt(pos.y())
            if row < 0:
                row = self.rowCount()
            self.dropIndicatorRow = row
            self.viewport().update()
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dragLeaveEvent(self, event):
        self.dropIndicatorRow = None
        self.viewport().update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasText() and self.drag_row != -1:
            self.blockSignals(True)  # Block signals during reordering to prevent cellChanged errors.
            pos = event.pos()
            drop_row = self.rowAt(pos.y())
            if drop_row < 0:
                drop_row = self.rowCount()
            if drop_row == self.drag_row or drop_row == self.drag_row + 1:
                self.drag_row = -1
                self.dropIndicatorRow = None
                self.viewport().update()
                self.blockSignals(False)
                return

            row_data = self.get_row_data(self.drag_row)
            old_row = self.drag_row
            if old_row < drop_row:
                drop_row -= 1

            self.removeRow(old_row)
            self.insertRow(drop_row)
            for col, item_text in enumerate(row_data):
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
                self.setItem(drop_row, col, item)

            self.selectRow(drop_row)
            self.drag_row = -1
            self.dropIndicatorRow = None
            self.resizeRowsToContents()
            self.viewport().update()
            self.blockSignals(False)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def get_row_data(self, row):
        data = []
        for col in range(self.columnCount()):
            item = self.item(row, col)
            data.append(item.text() if item else "")
        return data

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.dropIndicatorRow is not None:
            painter = QPainter(self.viewport())
            pen = QPen(QColor(0, 0, 255), 2)
            painter.setPen(pen)
            if self.dropIndicatorRow < self.rowCount():
                rect = self.visualRect(self.model().index(self.dropIndicatorRow, 0))
                y = rect.top()
            else:
                if self.rowCount() > 0:
                    rect = self.visualRect(self.model().index(self.rowCount() - 1, 0))
                    y = rect.bottom()
                else:
                    y = 0
            painter.drawLine(0, y, self.viewport().width(), y)

class VideoView(QGraphicsView):
    def __init__(self, scene, video_item, parent=None):
        super().__init__(scene, parent)
        self.video_item = video_item

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.video_item.boundingRect().isEmpty():
            self.fitInView(self.video_item, Qt.KeepAspectRatio)

class TranscriptEditor(QWidget):
    def __init__(self):
        super().__init__()
        # Start with no file paths and an empty transcript.
        self.video_path = None
        self.json_path = None
        self.transcript = []

        self.current_row = None
        self.loop_start = None
        self.loop_end = None
        self.loop_enabled = True
        self.overlay_visible = True
        self.auto_seek_enabled = True  # Flag for auto seeking

        self.normalized_positions = [
            (0.45, 0.65, 0.1, 0.1),
            (0, 0.3, 0.1, 0.1),
            (0.125, 0.3, 0.1, 0.1),
            (0.55, 0.55, 0.1, 0.1),
            (0.825, 0.55, 0.1, 0.1),
            (0.0625, 1.05, 0.1, 0.1),
            (0.325, 1.05, 0.1, 0.1)
        ]

        self.init_ui()

        self.global_space_shortcut = QShortcut(QKeySequence("Space"), self)
        self.global_space_shortcut.setContext(Qt.ApplicationShortcut)
        self.global_space_shortcut.activated.connect(self.handle_global_space)

    def disable_all_buttons(self):
        for btn in self.findChildren(QPushButton):
            btn.setEnabled(False)

    def enable_all_buttons(self):
        for btn in self.findChildren(QPushButton):
            btn.setEnabled(True)

    def handle_global_space(self):
        if not isinstance(self.focusWidget(), QLineEdit):
            self.toggle_play()

    def eventFilter(self, source, event):
        if source == self.table and event.type() == QEvent.KeyPress:
            if not isinstance(self.focusWidget(), QLineEdit):
                if event.key() == Qt.Key_Space:
                    self.toggle_play()
                    return True
                elif event.key() in (Qt.Key_Enter, Qt.Key_Return):
                    current_index = self.table.currentIndex()
                    if current_index.isValid():
                        self.table.edit(current_index)
                        return True
                elif event.key() == Qt.Key_Up:
                    current_row = self.table.currentRow()
                    if current_row > 0:
                        new_row = current_row - 1
                        self.table.selectRow(new_row)
                        self.on_cell_clicked(new_row, 0)
                    return True
                elif event.key() == Qt.Key_Down:
                    current_row = self.table.currentRow()
                    if current_row < self.table.rowCount() - 1:
                        new_row = current_row + 1
                        self.table.selectRow(new_row)
                        self.on_cell_clicked(new_row, 0)
                    return True
        return super().eventFilter(source, event)

    def load_transcript(self):
        self.disable_all_buttons()
        self.transcript_btn.setText("Loading Transcript...")
        file, _ = QFileDialog.getOpenFileName(self, "Open Transcript File", "", "JSON Files (*.json);;All Files (*)")
        if file:
            try:
                with open(file, 'r') as f:
                    self.transcript = json.load(f)
                self.json_path = file
                self.populate_transcript_table()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load transcript: {e}")
        self.enable_all_buttons()
        self.transcript_btn.setText("Upload Transcript")

    def populate_transcript_table(self):
        self.table.setRowCount(len(self.transcript))
        for row, entry in enumerate(self.transcript):
            start_item = QTableWidgetItem(seconds_to_formatted(entry.get("start", 0)))
            end_item = QTableWidgetItem(seconds_to_formatted(entry.get("end", 0)))
            speaker_item = QTableWidgetItem(entry.get("speaker", "speaker_0"))
            text_item = QTableWidgetItem(entry.get("text", ""))
            start_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
            end_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
            speaker_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
            text_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
            self.table.setItem(row, 0, start_item)
            self.table.setItem(row, 1, end_item)
            self.table.setItem(row, 2, speaker_item)
            self.table.setItem(row, 3, text_item)
        self.table.resizeRowsToContents()

    def load_media(self):
        self.disable_all_buttons()
        self.media_btn.setText("Loading Media...")
        file, _ = QFileDialog.getOpenFileName(self, "Open Media File", "",
                                              "Video/Audio Files (*.wmv *.mp4 *.mov *.avi *.wav *.mp3);;All Files (*)")
        if file:
            self.video_path = file
            self.loading_text.setText("Loading")
            self.spinner_label.setVisible(True)
            self.media_placeholder.show()
            media = QMediaContent(QUrl.fromLocalFile(self.video_path))
            self.player.setMedia(media)
            if self.waveformProgress is not None:
                self.waveformProgress.setParent(None)
                self.waveformProgress.deleteLater()
            try:
                self.waveformProgress = WaveformProgressBar(self.video_path)
                def on_seek_requested(ms):
                    if self.player.duration() > 0 and ms <= self.player.duration():
                        self.player.setPosition(ms)
                    else:
                        QMessageBox.warning(self, "Invalid Seek", "The requested time is beyond media duration.")
                self.waveformProgress.seekRequestedCallback = on_seek_requested
                self.video_layout.addWidget(self.waveformProgress)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load waveform: {e}")
        # Buttons are re-enabled in on_media_status_changed.

    def sort_transcript_by_start(self):
        rows = []
        for row in range(self.table.rowCount()):
            start_item = self.table.item(row, 0)
            end_item = self.table.item(row, 1)
            speaker_item = self.table.item(row, 2)
            text_item = self.table.item(row, 3)
            if not (start_item and end_item and speaker_item and text_item):
                continue
            try:
                start = formatted_to_seconds(start_item.text())
            except ValueError:
                start = 0.0
            try:
                end = formatted_to_seconds(end_item.text())
            except ValueError:
                end = 0.0
            speaker = speaker_item.text()
            text = text_item.text()
            rows.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text
            })
        rows.sort(key=lambda r: r["start"])
        self.transcript = rows
        self.populate_transcript_table()

    def init_ui(self):
        self.setWindowTitle("Manual Transcription Editor")
        main_v_layout = QVBoxLayout(self)

        banner_layout = QHBoxLayout()
        self.media_btn = QPushButton("Upload Media")
        self.media_btn.clicked.connect(self.load_media)
        self.transcript_btn = QPushButton("Upload Transcript")
        self.transcript_btn.clicked.connect(self.load_transcript)
        banner_layout.addWidget(self.media_btn)
        banner_layout.addWidget(self.transcript_btn)
        main_v_layout.addLayout(banner_layout)

        splitter = QSplitter(Qt.Horizontal)

        self.video_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(self.video_layout)

        self.scene = QGraphicsScene(self)
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        self.view = VideoView(self.scene, self.video_item)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setAlignment(Qt.AlignCenter)
        self.video_layout.addWidget(self.view)

        self.player = QMediaPlayer(None)
        self.player.setNotifyInterval(100)
        self.player.setVideoOutput(self.video_item)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)

        self.media_placeholder = QWidget()
        ph_layout = QHBoxLayout()
        ph_layout.setContentsMargins(0, 0, 0, 0)
        ph_layout.setAlignment(Qt.AlignCenter)
        self.spinner_label = create_spinner()
        self.spinner_label.setVisible(False)
        self.loading_text = QLabel("No media loaded")
        ph_layout.addWidget(self.spinner_label)
        ph_layout.addWidget(self.loading_text)
        self.media_placeholder.setLayout(ph_layout)
        self.media_placeholder.setFixedHeight(50)
        self.video_layout.addWidget(self.media_placeholder)

        if self.video_path:
            self.waveformProgress = WaveformProgressBar(self.video_path)
            def on_seek_requested(ms):
                if self.player.duration() > 0 and ms <= self.player.duration():
                    self.player.setPosition(ms)
                else:
                    QMessageBox.warning(self, "Invalid Seek", "The requested time is beyond media duration.")
            self.waveformProgress.seekRequestedCallback = on_seek_requested
            self.video_layout.addWidget(self.waveformProgress)
        else:
            self.waveformProgress = None

        controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play/Pause")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)
        self.current_time_label = QLabel("0:00.0")
        controls_layout.addWidget(self.current_time_label)
        self.video_layout.addLayout(controls_layout)

        self.overlay_bg = QGraphicsRectItem()
        self.overlay_bg.setBrush(Qt.black)
        self.overlay_bg.setOpacity(0.3)
        self.overlay_bg.setVisible(self.overlay_visible)
        self.overlay_bg.setZValue(10)
        self.scene.addItem(self.overlay_bg)
        self.speaker_buttons = []
        for i, (x_ratio, y_ratio, w_ratio, h_ratio) in enumerate(self.normalized_positions):
            btn = QPushButton(f"Speaker {i}")
            btn.setStyleSheet(
                "background-color: rgba(0, 0, 255, 50); "
                "color: rgba(255,255,255,255); border: none; font-size: 6px;"
            )
            proxy = QGraphicsProxyWidget()
            proxy.setWidget(btn)
            proxy.setZValue(11)
            self.scene.addItem(proxy)
            proxy.setVisible(self.overlay_visible)
            btn.clicked.connect(lambda checked, spk=i: self.assign_speaker(spk))
            self.speaker_buttons.append(proxy)
        loop_layout = QHBoxLayout()
        self.toggle_loop_button = QPushButton("Toggle Loop (On)")
        self.toggle_loop_button.clicked.connect(self.toggle_loop)
        loop_layout.addWidget(self.toggle_loop_button)
        loop_layout.addWidget(QLabel("Loop Start:"))
        self.loop_start_spin = TimeSpinBox()
        self.loop_start_spin.setRange(0, 9999999)
        self.loop_start_spin.setSingleStep(0.1)
        self.loop_start_spin.valueChanged.connect(self.loop_start_changed)
        loop_layout.addWidget(self.loop_start_spin)
        loop_layout.addWidget(QLabel("Loop End:"))
        self.loop_end_spin = TimeSpinBox()
        self.loop_end_spin.setRange(0, 9999999)
        self.loop_end_spin.setSingleStep(0.1)
        self.loop_end_spin.valueChanged.connect(self.loop_end_changed)
        loop_layout.addWidget(self.loop_end_spin)
        self.auto_seek_btn = QPushButton("Auto Seek: On")
        self.auto_seek_btn.clicked.connect(self.toggle_auto_seek)
        loop_layout.addWidget(self.auto_seek_btn)
        self.video_layout.addLayout(loop_layout)
        overlay_checkbox_layout = QHBoxLayout()
        self.overlay_checkbox = QCheckBox("Show Speaker Overlay")
        self.overlay_checkbox.setChecked(True)
        self.overlay_checkbox.stateChanged.connect(self.toggle_overlay)
        overlay_checkbox_layout.addWidget(self.overlay_checkbox)
        self.video_layout.addLayout(overlay_checkbox_layout)

        splitter.addWidget(left_widget)

        right_layout = QVBoxLayout()
        self.table = DraggableTableWidget(len(self.transcript), 4)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Speaker", "Text"])
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        self.table.installEventFilter(self)
        self.table.setWordWrap(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        if self.transcript:
            for row, entry in enumerate(self.transcript):
                start_item = QTableWidgetItem(seconds_to_formatted(entry.get("start", 0)))
                end_item = QTableWidgetItem(seconds_to_formatted(entry.get("end", 0)))
                speaker_item = QTableWidgetItem(entry.get("speaker", "speaker_0"))
                text_item = QTableWidgetItem(entry.get("text", ""))
                start_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
                end_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
                speaker_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
                text_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
                self.table.setItem(row, 0, start_item)
                self.table.setItem(row, 1, end_item)
                self.table.setItem(row, 2, speaker_item)
                self.table.setItem(row, 3, text_item)
        self.table.resizeRowsToContents()
        self.table.cellClicked.connect(self.on_cell_clicked)
        self.table.cellChanged.connect(self.on_cell_changed)
        right_layout.addWidget(self.table)
        control_layout = QHBoxLayout()
        add_btn = QPushButton("Add Line")
        add_btn.clicked.connect(self.add_line)
        del_btn = QPushButton("Delete Line")
        del_btn.clicked.connect(self.delete_line)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_transcript)
        sort_btn = QPushButton("Sort by Start")
        sort_btn.clicked.connect(self.sort_transcript_by_start)
        control_layout.addWidget(add_btn)
        control_layout.addWidget(del_btn)
        control_layout.addWidget(save_btn)
        control_layout.addWidget(sort_btn)
        right_layout.addLayout(control_layout)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_v_layout.addWidget(splitter)
        self.setLayout(main_v_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.video_item.boundingRect().isEmpty():
            self.view.fitInView(self.video_item, Qt.KeepAspectRatio)
        self.update_button_positions()
        if self.waveformProgress:
            self.waveformProgress.resizeEvent(event)

    def on_media_status_changed(self, status):
        if status in (QMediaPlayer.BufferedMedia, QMediaPlayer.LoadedMedia):
            self.media_placeholder.hide()
            QTimer.singleShot(100, lambda: self.view.fitInView(self.video_item, Qt.KeepAspectRatio))
            QTimer.singleShot(100, self.update_button_positions)
            self.enable_all_buttons()
            self.media_btn.setText("Upload Media")

    def assign_speaker(self, speaker_id):
        if self.current_row is not None:
            self.table.setItem(self.current_row, 2, QTableWidgetItem(f"speaker_{speaker_id}"))
            self.table.resizeRowsToContents()

    def on_cell_clicked(self, row, col):
        if row < 0 or row >= self.table.rowCount():
            return
        start_item = self.table.item(row, 0)
        end_item = self.table.item(row, 1)
        if not (start_item and end_item):
            return
        try:
            start_time = formatted_to_seconds(start_item.text())
            if self.auto_seek_enabled and self.video_path and self.player.duration() > 0:
                if start_time * 1000 > self.player.duration():
                    QMessageBox.warning(self, "Invalid Time", "The specified start time exceeds media duration.")
                    return
                self.player.setPosition(int(start_time * 1000))
                if col < 2:
                    self.player.play()
            self.current_row = row
            self.loop_start = start_time
            try:
                end_time = formatted_to_seconds(end_item.text())
            except ValueError:
                end_time = start_time
            self.loop_end = end_time
            self.loop_start_spin.setValue(self.loop_start)
            self.loop_end_spin.setValue(self.loop_end)
        except ValueError:
            pass

    def on_cell_changed(self, row, col):
        if row < 0 or row >= self.table.rowCount():
            return
        start_item = self.table.item(row, 0)
        end_item = self.table.item(row, 1)
        if not (start_item and end_item):
            return
        try:
            new_start = formatted_to_seconds(start_item.text())
            if self.current_row == row:
                self.loop_start = new_start
                self.loop_start_spin.setValue(new_start)
        except ValueError:
            pass
        try:
            new_end = formatted_to_seconds(end_item.text())
            if self.current_row == row:
                self.loop_end = new_end
                self.loop_end_spin.setValue(new_end)
        except ValueError:
            pass
        self.table.resizeRowsToContents()

    def loop_start_changed(self, val):
        self.loop_start = val
        if self.current_row is not None:
            self.table.setItem(self.current_row, 0, QTableWidgetItem(seconds_to_formatted(val)))
        self.table.resizeRowsToContents()

    def loop_end_changed(self, val):
        if self.loop_start is not None and val < self.loop_start:
            QMessageBox.warning(self, "Invalid Loop", "Loop end must be after loop start.")
            self.loop_end_spin.setValue(self.loop_start)
            return
        self.loop_end = val
        if self.current_row is not None:
            self.table.setItem(self.current_row, 1, QTableWidgetItem(seconds_to_formatted(val)))
        self.table.resizeRowsToContents()

    def toggle_loop(self):
        self.loop_enabled = not self.loop_enabled
        txt = "(On)" if self.loop_enabled else "(Off)"
        self.toggle_loop_button.setText("Toggle Loop " + txt)

    def toggle_auto_seek(self):
        self.auto_seek_enabled = not self.auto_seek_enabled
        txt = "On" if self.auto_seek_enabled else "Off"
        self.auto_seek_btn.setText(f"Auto Seek: {txt}")

    def on_position_changed(self, position):
        if self.waveformProgress:
            self.waveformProgress.set_current_position(position)
        secs = position / 1000.0
        self.current_time_label.setText(seconds_to_formatted(secs))
        if self.loop_enabled and self.loop_start is not None and self.loop_end is not None:
            if secs > self.loop_end:
                self.player.setPosition(int(self.loop_start * 1000))
        br = self.video_item.boundingRect()
        self.overlay_bg.setRect(br)

    def on_duration_changed(self, duration):
        pass

    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def toggle_overlay(self, state):
        self.overlay_visible = (state == Qt.Checked)
        self.overlay_bg.setVisible(self.overlay_visible)
        for btn in self.speaker_buttons:
            btn.setVisible(self.overlay_visible)

    def update_button_positions(self):
        br = self.video_item.boundingRect()
        video_width = br.width()
        video_height = br.height()
        for (x_ratio, y_ratio, w_ratio, h_ratio), proxy in zip(
            self.normalized_positions, self.speaker_buttons
        ):
            x = round(video_width * x_ratio)
            y = round(video_height * y_ratio)
            w = round(video_width * w_ratio)
            h = round(video_height * h_ratio)
            proxy.setPos(x, y)
            proxy.widget().resize(w, h)

    def add_line(self):
        current = self.table.currentRow()
        if current == -1:
            insert_position = self.table.rowCount()
        else:
            insert_position = current + 1
        self.table.insertRow(insert_position)
        self.table.setItem(insert_position, 0, QTableWidgetItem(seconds_to_formatted(0.0)))
        self.table.setItem(insert_position, 1, QTableWidgetItem(seconds_to_formatted(0.0)))
        self.table.setItem(insert_position, 2, QTableWidgetItem("speaker_0"))
        new_text_item = QTableWidgetItem("")
        new_text_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.table.setItem(insert_position, 3, new_text_item)
        self.table.resizeRowsToContents()

    def delete_line(self):
        if self.current_row is not None and self.current_row < self.table.rowCount():
            self.table.removeRow(self.current_row)
            self.current_row = None
            self.loop_start = None
            self.loop_end = None
            self.loop_start_spin.setValue(0.0)
            self.loop_end_spin.setValue(0.0)
            self.table.resizeRowsToContents()

    def save_transcript(self):
        new_data = []
        for row in range(self.table.rowCount()):
            start_item = self.table.item(row, 0)
            end_item = self.table.item(row, 1)
            speaker_item = self.table.item(row, 2)
            text_item = self.table.item(row, 3)
            if not (start_item and end_item and speaker_item and text_item):
                continue
            try:
                start = formatted_to_seconds(start_item.text())
            except ValueError:
                start = 0.0
            try:
                end = formatted_to_seconds(end_item.text())
            except ValueError:
                end = 0.0
            speaker = speaker_item.text()
            text = text_item.text()
            new_data.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text
            })
        if not new_data:
            QMessageBox.warning(self, "No Data", "No transcript data to save.")
            return
        if self.json_path:
            base, ext = os.path.splitext(self.json_path)
            default_path = base + "_manual_edit.json"
        else:
            default_path = "transcript_manual_edit.json"
        save_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcript",
            default_path,
            "JSON Files (*.json);;All Files (*)"
        )
        if not save_file:
            return
        with open(save_file, 'w') as f:
            json.dump(new_data, f, indent=2)
        QMessageBox.information(self, "Saved", f"Transcript saved to {save_file}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = TranscriptEditor()
    editor.show()
    sys.exit(app.exec_())

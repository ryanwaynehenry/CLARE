import csv
import json
import os
import struct
import sys
from PyQt5.QtCore import Qt, QUrl, QEvent, QTimer, QSize
from PyQt5.QtGui import QKeySequence, QFont, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
    QLabel, QCheckBox, QShortcut, QFileDialog, QMessageBox, QSplitter, QGraphicsRectItem,
    QToolButton
)
from custom_widgets import DraggableTableWidget, VideoView
from utils import create_spinner, seconds_to_formatted, formatted_to_seconds, TimeSpinBox, TimeTableWidgetItem
from waveform import WaveformProgressBar

class TranscriptEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.json_path = None
        self.transcript = []

        self.current_row = None
        self.loop_start = None
        self.loop_end = None
        self.loop_enabled = True
        self.overlay_visible = False
        self.auto_seek_enabled = True

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

    def init_ui(self):
        self.setWindowTitle("Manual Transcription Editor")
        main_v_layout = QVBoxLayout(self)

        # Top banner with upload buttons.
        ribbon_layout = QHBoxLayout()
        self.media_btn = QPushButton()
        upload_media_path = os.path.join("imgs", "upload_media.png")
        self.media_btn.setIcon(QIcon(upload_media_path))
        self.media_btn.setIconSize(QSize(48, 32))
        self.media_btn.setFixedSize(54, 36)
        self.media_btn.setToolTip("Load .wmv or .wav file")
        self.media_btn.clicked.connect(self.load_media)
        ribbon_layout.addWidget(self.media_btn)

        self.transcript_btn = QPushButton()
        upload_json_path = os.path.join("imgs", "upload_json.png")
        self.transcript_btn.setIcon(QIcon(upload_json_path))
        self.transcript_btn.setIconSize(QSize(48, 32))
        self.transcript_btn.setFixedSize(54, 36)
        self.transcript_btn.setToolTip("Load .json transcript")
        self.transcript_btn.clicked.connect(self.load_transcript)
        ribbon_layout.addWidget(self.transcript_btn)

        self.save_btn = QPushButton()
        save_json_path = os.path.join("imgs", "diskette.png")
        self.save_btn.setIcon(QIcon(save_json_path))
        self.save_btn.setIconSize(QSize(32, 32))
        self.save_btn.setFixedSize(36, 36)
        self.save_btn.setToolTip("Save .json transcript")
        self.save_btn.clicked.connect(self.save_transcript)
        ribbon_layout.addWidget(self.save_btn)

        ribbon_layout.addStretch()
        main_v_layout.addLayout(ribbon_layout)


        splitter = QSplitter(Qt.Horizontal)

        # Left side: video and waveform area.
        self.video_layout = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(self.video_layout)

        from PyQt5.QtWidgets import QGraphicsScene
        self.scene = QGraphicsScene(self)
        from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        self.view = VideoView(self.scene, self.video_item)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setAlignment(Qt.AlignCenter)
        self.video_layout.addWidget(self.view)

        overlay_checkbox_layout = QHBoxLayout()
        self.overlay_checkbox = QCheckBox("Show Speaker Overlay")
        self.overlay_checkbox.setChecked(True)
        self.overlay_checkbox.stateChanged.connect(self.toggle_overlay)
        overlay_checkbox_layout.addWidget(self.overlay_checkbox)
        self.video_layout.addLayout(overlay_checkbox_layout)

        self.player = QMediaPlayer(None)
        self.player.setNotifyInterval(100)
        self.player.setVideoOutput(self.video_item)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)

        # Placeholder widget below the video.
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

        self.waveformProgress = None
        if self.video_path:
            self.waveformProgress = WaveformProgressBar(self.video_path)
            def on_seek_requested(ms):
                if self.player.duration() > 0 and ms <= self.player.duration():
                    self.player.setPosition(int(ms))
                else:
                    QMessageBox.warning(self, "Invalid Seek", "The requested time is beyond media duration.")
            self.waveformProgress.seekRequestedCallback = on_seek_requested
            self.video_layout.addWidget(self.waveformProgress)

        controls_layout = QHBoxLayout()

        # Jump Backward Button
        self.jump_backward_btn = QToolButton()
        self.jump_backward_btn.setIcon(
            QIcon(os.path.join("imgs", "backward10.png")))
        self.jump_backward_btn.setIconSize(QSize(48, 32))
        self.jump_backward_btn.setFixedSize(54, 36)
        self.jump_backward_btn.setToolTip("Jump Backward 10 Seconds")
        self.jump_backward_btn.clicked.connect(self.jump_backward)
        controls_layout.addStretch()
        controls_layout.addWidget(self.jump_backward_btn)

        # Play/Pause Button (with small fixed size and icon swapping)
        self.play_button = QToolButton()
        # Define your play and pause icons – make sure the image files exist.
        self.play_icon = QIcon(os.path.join("imgs", "play.png"))
        self.pause_icon = QIcon(os.path.join("imgs", "pause.png"))
        self.play_button.setIcon(self.play_icon)
        self.play_button.setIconSize(QSize(48, 32))
        self.play_button.setFixedSize(54, 36)
        self.play_button.setToolTip("Play/Pause")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        # Jump Forward Button
        self.jump_forward_btn = QToolButton()
        self.jump_forward_btn.setIcon(
            QIcon(os.path.join("imgs", "forward10.png")))
        self.jump_forward_btn.setIconSize(QSize(48, 32))
        self.jump_forward_btn.setFixedSize(54, 36)
        self.jump_forward_btn.setToolTip("Jump Forward 10 Seconds")
        self.jump_forward_btn.clicked.connect(self.jump_forward)
        controls_layout.addWidget(self.jump_forward_btn)

        # Add the current time label alongside the controls.
        self.current_time_label = QLabel("0:00.0")
        controls_layout.addWidget(self.current_time_label)
        controls_layout.addStretch()

        self.video_layout.addLayout(controls_layout)

        # Overlay background and speaker buttons.
        self.overlay_bg = QGraphicsRectItem()
        self.overlay_bg.setBrush(Qt.black)
        self.overlay_bg.setOpacity(0.3)
        self.overlay_bg.setVisible(self.overlay_visible)
        self.overlay_bg.setZValue(10)
        self.scene.addItem(self.overlay_bg)
        self.speaker_buttons = []
        for i, (x_ratio, y_ratio, w_ratio, h_ratio) in enumerate(self.normalized_positions):
            btn = QPushButton(f"Speaker {i}")
            btn.setStyleSheet("background-color: rgba(0, 0, 255, 50); color: rgba(255,255,255,255); border: none; font-size: 6px;")
            from PyQt5.QtWidgets import QGraphicsProxyWidget
            proxy = QGraphicsProxyWidget()
            proxy.setWidget(btn)
            proxy.setZValue(11)
            self.scene.addItem(proxy)
            proxy.setVisible(self.overlay_visible)
            btn.clicked.connect(lambda checked, spk=i: self.assign_speaker(spk))
            self.speaker_buttons.append(proxy)

        # Loop controls and auto-seek toggle.
        loop_layout = QHBoxLayout()
        self.toggle_loop_button = QPushButton("Looping: Enabled")
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
        loop_layout.addStretch()

        self.video_layout.addLayout(loop_layout)

        splitter.addWidget(left_widget)

        # Right side: transcript table and controls.
        right_layout = QVBoxLayout()
        self.table = DraggableTableWidget(len(self.transcript), 4)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Speaker", "Text"])
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
        self.table.installEventFilter(self)
        self.table.setWordWrap(True)
        from PyQt5.QtWidgets import QHeaderView
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
        self.auto_seek_btn = QPushButton("Jump to: Enabled")
        self.auto_seek_btn.clicked.connect(self.toggle_auto_seek)
        self.add_btn = QPushButton("Add Line")
        self.add_btn.clicked.connect(self.add_line)
        self.del_btn = QPushButton("Delete Line")
        self.del_btn.clicked.connect(self.delete_line)
        self.sort_btn = QPushButton("Sort by Start")
        self.sort_btn.clicked.connect(lambda: self.table.sortItems(0, Qt.AscendingOrder))
        control_layout.addWidget(self.auto_seek_btn)
        control_layout.addWidget(self.sort_btn)
        control_layout.addWidget(self.add_btn)
        control_layout.addWidget(self.del_btn)

        right_layout.addLayout(control_layout)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_v_layout.addWidget(splitter)
        self.setLayout(main_v_layout)

    def disable_all_buttons(self):
        for btn in self.findChildren(QPushButton):
            btn.setEnabled(False)
        for btn in self.findChildren(QToolButton):
            btn.setEnabled(False)

    def enable_all_buttons(self):
        for btn in self.findChildren(QPushButton):
            btn.setEnabled(True)
        for btn in self.findChildren(QToolButton):
            btn.setEnabled(True)

    def handle_global_space(self):
        if not isinstance(self.focusWidget(), QLabel):
            self.toggle_play()

    def assign_speaker(self, speaker_id):
        if self.current_row is not None:
            self.table.setItem(self.current_row, 2, QTableWidgetItem(f"speaker_{speaker_id}"))
            self.table.resizeRowsToContents()

    def eventFilter(self, source, event):
        if source == self.table and event.type() == QEvent.KeyPress:
            if not isinstance(self.focusWidget(), QLabel):
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
        file, _ = QFileDialog.getOpenFileName(self, "Open Transcript File", "", "JSON Files (*.json);;All Files (*)")
        if file:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                # Check if the JSON has a "segments" key (new format)
                if isinstance(data, dict) and "segments" in data:
                    self.transcript = []
                    for seg in data["segments"]:
                        entry = {
                            "start": seg.get("start", 0.0),
                            "end": seg.get("end", 0.0),
                            "text": seg.get("text", "").strip(),
                            "speaker": seg.get("speaker", "speaker_0")
                        }
                        self.transcript.append(entry)
                else:
                    self.transcript = data
                self.json_path = file
                self.populate_transcript_table()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load transcript: {e}")
        self.enable_all_buttons()

    def populate_transcript_table(self):
        self.table.setRowCount(len(self.transcript))
        for row, entry in enumerate(self.transcript):
            start_item = TimeTableWidgetItem(seconds_to_formatted(entry.get("start", 0)))
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
        file, _ = QFileDialog.getOpenFileName(self, "Open Media File", "",
                                              "Video/Audio Files (*.wmv *.wav);;All Files (*)")
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
                        if self.loop_enabled:
                            loop_start_ms = self.loop_start * 1000 if self.loop_start is not None else 0
                            loop_end_ms = self.loop_end * 1000 if self.loop_end is not None else self.player.duration()
                            if ms < loop_start_ms:
                                self.player.setPosition(int(loop_start_ms))
                                return
                            elif ms > loop_end_ms:
                                return
                        self.player.setPosition(int(ms))
                    else:
                        QMessageBox.warning(self, "Invalid Seek", "The requested time is beyond media duration.")
                self.waveformProgress.seekRequestedCallback = on_seek_requested
                placeholder_index = self.video_layout.indexOf(
                    self.media_placeholder)
                self.video_layout.removeWidget(self.media_placeholder)
                # self.media_placeholder.hide()
                self.media_placeholder.deleteLater()
                self.video_layout.insertWidget(placeholder_index,
                                               self.waveformProgress)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load waveform: {e}")
        # Buttons re-enabled in on_media_status_changed.

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
            rows.append({
                "start": start,
                "end": end,
                "speaker": speaker_item.text(),
                "text": text_item.text()
            })
        rows.sort(key=lambda r: r["start"])
        self.transcript = rows
        self.populate_transcript_table()

    def on_cell_clicked(self, row, col):
        if row < 0 or row >= self.table.rowCount():
            return
        start_item = self.table.item(row, 0)
        end_item = self.table.item(row, 1)
        if not (start_item and end_item):
            return
        try:
            self.current_row = row
            start_time = formatted_to_seconds(start_item.text())
            if self.auto_seek_enabled and self.video_path and self.player.duration() > 0:
                if start_time * 1000 > self.player.duration():
                    QMessageBox.warning(self, "Invalid Time", "The specified start time exceeds media duration.")
                    return
                self.player.setPosition(int(start_time * 1000))
                if col < 2:
                    self.player.play()

                self.loop_start = start_time
                try:
                    end_time = formatted_to_seconds(end_item.text())
                except ValueError:
                    end_time = start_time
                self.loop_end = end_time
                self.loop_start_spin.setValue(self.loop_start)
                self.loop_end_spin.setValue(self.loop_end)
                if self.waveformProgress is not None:
                    self.waveformProgress.set_loop_boundaries(self.loop_start, self.loop_end)
        except ValueError:
            pass

    # --- Added helper and modified cell change handling ---
    def normalize_time_entry(self, time_string):
        """
        Convert the user's input into a properly formatted time string.
        If the input is a plain number, treat it as seconds.
        If it already contains a separator, parse it accordingly.
        The returned string is produced by seconds_to_formatted,
        which handles conversion into proper minute:second (or hour:minute:second)
        notation.
        """
        try:
            # Try converting the input directly to a float (seconds)
            time_value = float(time_string)
        except ValueError:
            # Otherwise, assume the value is given in a separated format.
            time_value = formatted_to_seconds(time_string)
        return seconds_to_formatted(time_value)

    def on_cell_changed(self, row, col):
        if row < 0 or row >= self.table.rowCount():
            return
        start_item = self.table.item(row, 0)
        end_item = self.table.item(row, 1)
        if not (start_item and end_item):
            return

        # Block signals during normalization to avoid recursive calls.
        self.table.blockSignals(True)
        try:
            normalized_start = self.normalize_time_entry(start_item.text())
            start_item.setText(normalized_start)
            new_start = formatted_to_seconds(normalized_start)
            if self.current_row == row:
                self.loop_start = new_start
                self.loop_start_spin.setValue(new_start)
        except ValueError:
            pass
        try:
            normalized_end = self.normalize_time_entry(end_item.text())
            end_item.setText(normalized_end)
            new_end = formatted_to_seconds(normalized_end)
            if self.current_row == row:
                self.loop_end = new_end
                self.loop_end_spin.setValue(new_end)
        except ValueError:
            pass
        self.table.blockSignals(False)
        self.table.resizeRowsToContents()
    # --- End modifications ---

    def loop_start_changed(self, val):
        if self.loop_end is not None and val > self.loop_end:
            # QMessageBox.warning(self, "Invalid Loop", "Loop start cannot be greater than loop end.")
            self.loop_start_spin.blockSignals(True)
            self.loop_start_spin.setValue(self.prev_loop_start)
            self.loop_start_spin.blockSignals(False)
            return
        self.loop_start = val
        self.prev_loop_start = val
        if self.current_row is not None:
            self.table.setItem(self.current_row, 0, QTableWidgetItem(seconds_to_formatted(val)))
        self.table.resizeRowsToContents()
        if self.waveformProgress is not None:
            self.waveformProgress.set_loop_boundaries(self.loop_start, self.loop_end if self.loop_end is not None else self.loop_start)

    def loop_end_changed(self, val):
        if self.loop_start is not None and val < self.loop_start:
            # QMessageBox.warning(self, "Invalid Loop", "Loop end must be after loop start.")
            self.loop_end_spin.blockSignals(True)
            self.loop_end_spin.setValue(self.prev_loop_end)
            self.loop_end_spin.blockSignals(False)
            return
        self.loop_end = val
        self.prev_loop_end = val
        if self.current_row is not None:
            self.table.setItem(self.current_row, 1, QTableWidgetItem(seconds_to_formatted(val)))
        self.table.resizeRowsToContents()
        if self.waveformProgress is not None:
            self.waveformProgress.set_loop_boundaries(self.loop_start if self.loop_start is not None else val, val)

    def toggle_loop(self):
        self.loop_enabled = not self.loop_enabled
        txt = "Enabled" if self.loop_enabled else "Disabled"
        self.toggle_loop_button.setText("Looping: " + txt)

        # If looping is now enabled and loop boundaries exist, check the current position.
        if self.loop_enabled and self.loop_start is not None and self.loop_end is not None:
            current_position = self.player.position()  # current position in milliseconds
            loop_start_ms = int(self.loop_start * 1000)
            loop_end_ms = int(self.loop_end * 1000)
            # If the current position is before the loop start or after the loop end, seek to loop start.
            if current_position < loop_start_ms or current_position > loop_end_ms:
                self.player.setPosition(loop_start_ms)

    def toggle_auto_seek(self):
        self.auto_seek_enabled = not self.auto_seek_enabled
        txt = "Enabled" if self.auto_seek_enabled else "Disabled"
        self.auto_seek_btn.setText(f"Jump to: {txt}")

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

    def on_media_status_changed(self, status):
        if status in (QMediaPlayer.BufferedMedia, QMediaPlayer.LoadedMedia):
            # self.media_placeholder.hide()
            QTimer.singleShot(100, lambda: self.view.fitInView(self.video_item, Qt.KeepAspectRatio))
            QTimer.singleShot(100, self.update_button_positions)
            self.enable_all_buttons()
            if self.overlay_checkbox.isChecked():
                self.overlay_visible = True
                for proxy in self.speaker_buttons:
                    proxy.setVisible(True)

    def jump_backward(self):
        # Jump 10 seconds backward.
        new_pos = self.player.position() - 10000
        if new_pos < 0:
            new_pos = 0
        self.player.setPosition(int(new_pos))

    def jump_forward(self):
        # Jump 10 seconds forward.
        new_pos = self.player.position() + 10000
        if new_pos > self.player.duration():
            new_pos = self.player.duration()
        self.player.setPosition(int(new_pos))

    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.play_button.setIcon(self.play_icon)
        else:
            self.player.play()
            self.play_button.setIcon(self.pause_icon)

    def toggle_overlay(self, state):
        self.overlay_visible = (state == Qt.Checked)
        self.overlay_bg.setVisible(self.overlay_visible)
        for btn in self.speaker_buttons:
            btn.setVisible(self.overlay_visible)

    def update_button_positions(self):
        br = self.video_item.boundingRect()
        video_width = br.width()
        video_height = br.height()
        for (x_ratio, y_ratio, w_ratio, h_ratio), proxy in zip(self.normalized_positions, self.speaker_buttons):
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
            new_data.append({
                "start": start,
                "end": end,
                "speaker": speaker_item.text(),
                "text": text_item.text()
            })
        if not new_data:
            QMessageBox.warning(self, "No Data", "No transcript data to save.")
            return
        if self.json_path:
            base, ext = os.path.splitext(self.json_path)
            default_path = base + "_manual_edit.json"
        else:
            default_path = "transcript_manual_edit.json"
        save_file, _ = QFileDialog.getSaveFileName(self, "Save Transcript", default_path, "JSON Files (*.json);;All Files (*)")
        if not save_file:
            return
        with open(save_file, 'w') as f:
            json.dump(new_data, f, indent=2)
        QMessageBox.information(self, "Saved", f"Transcript saved to {save_file}")

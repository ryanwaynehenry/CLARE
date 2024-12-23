import json
import sys
import os

from PyQt5.QtCore import Qt, QUrl, QMimeData
from PyQt5.QtGui import QDrag
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QLabel, QDoubleSpinBox,
    QCheckBox, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget,
    QGraphicsRectItem, QHeaderView, QFileDialog, QMessageBox
)

from waveform import WaveformProgressBar  # The optimized waveform widget


class DraggableTableWidget(QTableWidget):
    """A QTableWidget subclass that supports dragging and dropping entire rows."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragDropMode(QTableWidget.DragDrop)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.drag_row = -1

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
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasText() and self.drag_row != -1:
            drop_position = event.pos()
            drop_row = self.rowAt(drop_position.y())
            if drop_row == -1:
                drop_row = self.rowCount()

            if drop_row == self.drag_row or drop_row == self.drag_row + 1:
                self.drag_row = -1
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
            self.resizeRowsToContents()
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def get_row_data(self, row):
        data = []
        for col in range(self.columnCount()):
            item = self.item(row, col)
            data.append(item.text() if item else "")
        return data


class TranscriptEditor(QWidget):
    def __init__(self):
        super().__init__()

        # We'll store paths after prompting the user
        self.video_path = None
        self.json_path = None

        self.current_row = None
        self.loop_start = None
        self.loop_end = None
        self.loop_enabled = True
        self.overlay_visible = True

        self.normalized_positions = [
            (0.45, 0.65, 0.1, 0.1),
            (0, 0.3, 0.1, 0.1),
            (0.125, 0.3, 0.1, 0.1),
            (0.55, 0.55, 0.1, 0.1),
            (0.825, 0.55, 0.1, 0.1),
            (0.0625, 1.05, 0.1, 0.1),
            (0.325, 1.05, 0.1, 0.1)
        ]

        # Prompt user for files
        if not self.prompt_for_files():
            # If user canceled either prompt, we won't proceed
            QMessageBox.warning(self, "No Files", "No valid file paths selected. Exiting.")
            sys.exit(0)

        # Load transcript data
        with open(self.json_path, 'r') as f:
            self.transcript = json.load(f)

        self.init_ui()

    def prompt_for_files(self):
        """
        Prompt user for a video file and a JSON file.
        Return True if both selected, False if user cancels.
        """
        # 1) Prompt for JSON transcript
        json_file, _ = QFileDialog.getOpenFileName(self, "Open Transcript JSON", "",
                                                  "JSON Files (*.json);;All Files (*.*)")
        if not json_file:
            return False

        # 2) Prompt for video
        video_file, _ = QFileDialog.getOpenFileName(self, "Open Video File", "",
                                                    "Video Files (*.wmv *.mp4 *.mov *.avi);;All Files (*.*)")
        if not video_file:
            return False

        self.json_path = json_file
        self.video_path = video_file
        return True

    def init_ui(self):
        self.setWindowTitle("Manual Transcription Editor")

        main_layout = QHBoxLayout(self)

        # Left side: video + waveform
        video_layout = QVBoxLayout()

        # Graphics scene for the video
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setAlignment(Qt.AlignCenter)

        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)

        self.player = QMediaPlayer(None)
        self.player.setNotifyInterval(100)
        self.player.setVideoOutput(self.video_item)
        media = QMediaContent(QUrl.fromLocalFile(self.video_path))
        self.player.setMedia(media)

        # Connect signals
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)

        video_layout.addWidget(self.view)

        # Combined waveform widget
        self.waveformProgress = WaveformProgressBar(self.video_path)

        def on_seek_requested(ms):
            self.player.setPosition(ms)
        self.waveformProgress.seekRequestedCallback = on_seek_requested

        video_layout.addWidget(self.waveformProgress)

        # Playback controls
        controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play/Pause")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        self.current_time_label = QLabel("00:00:00.0")
        controls_layout.addWidget(self.current_time_label)

        video_layout.addLayout(controls_layout)

        # Overlay background
        self.overlay_bg = QGraphicsRectItem()
        self.overlay_bg.setBrush(Qt.black)
        self.overlay_bg.setOpacity(0.3)
        self.overlay_bg.setVisible(self.overlay_visible)
        self.overlay_bg.setZValue(10)
        self.scene.addItem(self.overlay_bg)

        # Speaker overlay buttons
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

        # Loop controls
        loop_layout = QHBoxLayout()
        self.toggle_loop_button = QPushButton("Toggle Loop (On)")
        self.toggle_loop_button.clicked.connect(self.toggle_loop)
        loop_layout.addWidget(self.toggle_loop_button)

        loop_layout.addWidget(QLabel("Loop Start:"))
        self.loop_start_spin = QDoubleSpinBox()
        self.loop_start_spin.setRange(0, 9999999)
        self.loop_start_spin.setSingleStep(0.1)
        self.loop_start_spin.valueChanged.connect(self.loop_start_changed)
        loop_layout.addWidget(self.loop_start_spin)

        loop_layout.addWidget(QLabel("Loop End:"))
        self.loop_end_spin = QDoubleSpinBox()
        self.loop_end_spin.setRange(0, 9999999)
        self.loop_end_spin.setSingleStep(0.1)
        self.loop_end_spin.valueChanged.connect(self.loop_end_changed)
        loop_layout.addWidget(self.loop_end_spin)

        video_layout.addLayout(loop_layout)

        # Overlay checkbox
        overlay_checkbox_layout = QHBoxLayout()
        self.overlay_checkbox = QCheckBox("Show Speaker Overlay")
        self.overlay_checkbox.setChecked(True)
        self.overlay_checkbox.stateChanged.connect(self.toggle_overlay)
        overlay_checkbox_layout.addWidget(self.overlay_checkbox)
        video_layout.addLayout(overlay_checkbox_layout)

        # Right side: transcript table
        right_layout = QVBoxLayout()

        self.table = DraggableTableWidget(len(self.transcript), 4)
        self.table.setHorizontalHeaderLabels(["Start", "End", "Speaker", "Text"])
        self.table.setWordWrap(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

        for row, entry in enumerate(self.transcript):
            start_item = QTableWidgetItem(str(entry["start"]))
            end_item = QTableWidgetItem(str(entry["end"]))
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
        # If user clicks on first two columns => jump to start_time, set loop
        self.table.cellClicked.connect(self.on_cell_clicked)
        self.table.cellChanged.connect(self.on_cell_changed)
        right_layout.addWidget(self.table)

        # Add/Remove/Save buttons
        control_layout = QHBoxLayout()
        add_btn = QPushButton("Add Line")
        add_btn.clicked.connect(self.add_line)
        del_btn = QPushButton("Delete Line")
        del_btn.clicked.connect(self.delete_line)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_transcript)
        control_layout.addWidget(add_btn)
        control_layout.addWidget(del_btn)
        control_layout.addWidget(save_btn)
        right_layout.addLayout(control_layout)

        main_layout.addLayout(video_layout, 3)
        main_layout.addLayout(right_layout, 2)
        self.setLayout(main_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.video_item.boundingRect().isEmpty():
            self.view.fitInView(self.video_item, Qt.KeepAspectRatio)
        self.update_button_positions()
        # Force a waveform repaint if size changed
        self.waveformProgress.resizeEvent(event)

    # -----------
    # Speaker
    # -----------
    def assign_speaker(self, speaker_id):
        if self.current_row is not None:
            self.table.setItem(self.current_row, 2, QTableWidgetItem(f"speaker_{speaker_id}"))
            self.table.resizeRowsToContents()

    # -----------
    # Table Clicking => Automated Loop
    # -----------
    def on_cell_clicked(self, row, col):
        if row < 0 or row >= self.table.rowCount():
            return
        start_item = self.table.item(row, 0)
        end_item = self.table.item(row, 1)
        if not (start_item and end_item):
            return
        try:
            start_time = float(start_item.text())
            end_time = float(end_item.text())
            # Jump to start_time
            self.player.setPosition(int(start_time * 1000))
            # If user clicked col < 2 => auto-play
            if col < 2:
                self.player.play()

            self.current_row = row
            self.loop_start = start_time
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
            new_start = float(start_item.text())
            if self.current_row == row:
                self.loop_start = new_start
                self.loop_start_spin.setValue(new_start)
        except ValueError:
            pass

        try:
            new_end = float(end_item.text())
            if self.current_row == row:
                self.loop_end = new_end
                self.loop_end_spin.setValue(new_end)
        except ValueError:
            pass

        self.table.resizeRowsToContents()

    # -----------
    # Loop Methods
    # -----------
    def loop_start_changed(self, val):
        self.loop_start = val
        if self.current_row is not None:
            self.table.setItem(self.current_row, 0, QTableWidgetItem(str(val)))
        self.table.resizeRowsToContents()

    def loop_end_changed(self, val):
        self.loop_end = val
        if self.current_row is not None:
            self.table.setItem(self.current_row, 1, QTableWidgetItem(str(val)))
        self.table.resizeRowsToContents()

    def toggle_loop(self):
        self.loop_enabled = not self.loop_enabled
        txt = "(On)" if self.loop_enabled else "(Off)"
        self.toggle_loop_button.setText("Toggle Loop " + txt)

    # -----------
    # Player
    # -----------
    def on_position_changed(self, position):
        self.waveformProgress.set_current_position(position)
        secs = position / 1000.0
        self.current_time_label.setText(self.format_time(secs))

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

    # -----------
    # Overlay
    # -----------
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

    # -----------
    # Transcript
    # -----------
    def add_line(self):
        current = self.table.currentRow()
        if current == -1:
            insert_position = self.table.rowCount()
        else:
            insert_position = current + 1

        self.table.insertRow(insert_position)
        self.table.setItem(insert_position, 0, QTableWidgetItem("0.0"))
        self.table.setItem(insert_position, 1, QTableWidgetItem("0.0"))
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
        """
        Prompt user for a location to save the transcript.
        The default name is "<original_basename>_manual_edit.json"
        """
        new_data = []
        for row in range(self.table.rowCount()):
            start_item = self.table.item(row, 0)
            end_item = self.table.item(row, 1)
            speaker_item = self.table.item(row, 2)
            text_item = self.table.item(row, 3)

            if not (start_item and end_item and speaker_item and text_item):
                continue

            try:
                start = float(start_item.text())
            except ValueError:
                start = 0.0
            try:
                end = float(end_item.text())
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

        # Build default path: e.g. transcript.json => transcript_manual_edit.json
        base, ext = os.path.splitext(self.json_path)
        default_path = base + "_manual_edit.json"

        # Prompt user for save location
        save_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcript",
            default_path,
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not save_file:
            return  # user canceled

        # Write new_data to save_file
        with open(save_file, 'w') as f:
            json.dump(new_data, f, indent=2)

        QMessageBox.information(self, "Saved", f"Transcript saved to {save_file}")

    def format_time(self, seconds):
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = (seconds % 60)
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:04.1f}"
        else:
            return f"{mins:02d}:{secs:04.1f}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = TranscriptEditor()
    editor.show()
    sys.exit(app.exec_())

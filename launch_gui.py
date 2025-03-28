import json
import sys
import os

from PyQt5.QtCore import Qt, QUrl, QMimeData, QEvent
from PyQt5.QtGui import QDrag, QKeySequence, QPainter, QPen, QColor
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QLabel, QLineEdit, QDoubleSpinBox,
    QCheckBox, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QShortcut, 
    QGraphicsRectItem, QHeaderView, QFileDialog, QMessageBox, QSplitter
)

from waveform import WaveformProgressBar  # The optimized waveform widget

# --- New Custom Spin Box for Time Display ---
class TimeSpinBox(QDoubleSpinBox):
    def textFromValue(self, value):
        if value < 3600:
            minutes = int(value // 60)
            secs = value - minutes * 60
            # Format as m:ss.d (e.g., "0:07.0", "27:01.5")
            return f"{minutes}:{secs:04.1f}"
        else:
            hours = int(value // 3600)
            remainder = value % 3600
            minutes = int(remainder // 60)
            secs = remainder - minutes * 60
            # Format as h:mm:ss.d (e.g., "1:00:00.0", "2:34:05.9")
            return f"{hours}:{minutes:02d}:{secs:04.1f}"

# --- Updated Helper Methods for Time Conversion ---
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

# --- Rest of the Code (with modifications highlighted) ---

class DraggableTableWidget(QTableWidget):
    """A QTableWidget subclass that supports dragging and dropping entire rows with a drop indicator."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragDropMode(QTableWidget.DragDrop)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(False)  # We'll draw our own indicator
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
            self.viewport().update()  # Trigger repaint to show the indicator
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dragLeaveEvent(self, event):
        self.dropIndicatorRow = None
        self.viewport().update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasText() and self.drag_row != -1:
            pos = event.pos()
            drop_row = self.rowAt(pos.y())
            if drop_row < 0:
                drop_row = self.rowCount()
            # Avoid inserting in the same location
            if drop_row == self.drag_row or drop_row == self.drag_row + 1:
                self.drag_row = -1
                self.dropIndicatorRow = None
                self.viewport().update()
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
            pen = QPen(QColor(0, 0, 255), 2)  # Blue line, 2 pixels wide
            painter.setPen(pen)
            if self.dropIndicatorRow < self.rowCount():
                # Draw at the top of the indicated row.
                rect = self.visualRect(self.model().index(self.dropIndicatorRow, 0))
                y = rect.top()
            else:
                # If below the last row, draw at the bottom.
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

        if not self.prompt_for_files():
            QMessageBox.warning(self, "No Files", "No valid file paths selected. Exiting.")
            sys.exit(0)

        with open(self.json_path, 'r') as f:
            self.transcript = json.load(f)

        self.init_ui()

        self.global_space_shortcut = QShortcut(QKeySequence("Space"), self)
        self.global_space_shortcut.setContext(Qt.ApplicationShortcut)
        self.global_space_shortcut.activated.connect(self.handle_global_space)

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

    def prompt_for_files(self):
        json_file, _ = QFileDialog.getOpenFileName(self, "Open Transcript JSON", "",
                                                  "JSON Files (*.json);;All Files (*.*)")
        if not json_file:
            return False

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
        self.scene = QGraphicsScene(self)
        self.video_item = QGraphicsVideoItem()
        self.scene.addItem(self.video_item)
        self.view = VideoView(self.scene, self.video_item)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setAlignment(Qt.AlignCenter)
        self.player = QMediaPlayer(None)
        self.player.setNotifyInterval(100)
        self.player.setVideoOutput(self.video_item)
        media = QMediaContent(QUrl.fromLocalFile(self.video_path))
        self.player.setMedia(media)
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        video_layout.addWidget(self.view)
        self.waveformProgress = WaveformProgressBar(self.video_path)
        def on_seek_requested(ms):
            self.player.setPosition(ms)
        self.waveformProgress.seekRequestedCallback = on_seek_requested
        video_layout.addWidget(self.waveformProgress)
        controls_layout = QHBoxLayout()
        self.play_button = QPushButton("Play/Pause")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)
        # Update current time label to use new format:
        self.current_time_label = QLabel("0:00.0")
        controls_layout.addWidget(self.current_time_label)
        video_layout.addLayout(controls_layout)

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
        # Use custom TimeSpinBox for formatted display.
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
        video_layout.addLayout(loop_layout)
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
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.table.installEventFilter(self)
        self.table.setWordWrap(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, entry in enumerate(self.transcript):
            start_item = QTableWidgetItem(seconds_to_formatted(entry["start"]))
            end_item = QTableWidgetItem(seconds_to_formatted(entry["end"]))
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
        control_layout.addWidget(add_btn)
        control_layout.addWidget(del_btn)
        control_layout.addWidget(save_btn)
        right_layout.addLayout(control_layout)

        left_container = QWidget()
        left_container.setLayout(video_layout)
        right_container = QWidget()
        right_container.setLayout(right_layout)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.video_item.boundingRect().isEmpty():
            self.view.fitInView(self.video_item, Qt.KeepAspectRatio)
        self.update_button_positions()
        self.waveformProgress.resizeEvent(event)

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
            end_time = formatted_to_seconds(end_item.text())
            self.player.setPosition(int(start_time * 1000))
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
        self.loop_end = val
        if self.current_row is not None:
            self.table.setItem(self.current_row, 1, QTableWidgetItem(seconds_to_formatted(val)))
        self.table.resizeRowsToContents()

    def toggle_loop(self):
        self.loop_enabled = not self.loop_enabled
        txt = "(On)" if self.loop_enabled else "(Off)"
        self.toggle_loop_button.setText("Toggle Loop " + txt)

    def on_position_changed(self, position):
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

        base, ext = os.path.splitext(self.json_path)
        default_path = base + "_manual_edit.json"

        save_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcript",
            default_path,
            "JSON Files (*.json);;All Files (*.*)"
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

import csv
import json
import os
import struct
import sys
from PyQt5.QtCore import (
    Qt, QUrl, QEvent, QTimer, QSize, QModelIndex, QAbstractTableModel, QVariant,
    QItemSelectionModel
)
from PyQt5.QtGui import QKeySequence, QFont, QIcon, QPalette, QColor, QPainter, QPen
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
    QLabel, QCheckBox, QShortcut, QFileDialog, QMessageBox, QSplitter, QGraphicsRectItem,
    QToolButton, QTableView, QHeaderView, QAbstractItemView, QAbstractItemView
)
from custom_widgets import DraggableTableWidget, VideoView
from utils import create_spinner, seconds_to_formatted, formatted_to_seconds, TimeSpinBox, TimeTableWidgetItem
from waveform import WaveformProgressBar

COL_CUSHION = 16

from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtWidgets import QTableView, QAbstractItemView

class DragTableView(QTableView):
    """
    QTableView with drag-and-drop row reordering and a visual drop indicator line.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(False)  # we'll draw our own line
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)

        self.dropIndicatorRow = None

    def dragEnterEvent(self, event):
        if event.source() == self:
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.source() == self:
            self.dropIndicatorRow = self.drop_on(event)
            self.viewport().update()  # trigger repaint to draw indicator
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dragLeaveEvent(self, event):
        self.dropIndicatorRow = None
        self.viewport().update()
        super().dragLeaveEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.startDrag(Qt.MoveAction)
        super().mouseMoveEvent(event)

    def dropEvent(self, event):
        if event.source() == self and event.dropAction() == Qt.MoveAction:
            source_row = self.currentIndex().row()
            drop_row = self.drop_on(event)

            if source_row < drop_row:
                drop_row -= 1
            if source_row != drop_row:
                self.model().moveRows(QModelIndex(), source_row, 1, QModelIndex(), drop_row)
                self.selectRow(drop_row)

            self.dropIndicatorRow = None
            self.viewport().update()
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def drop_on(self, event):
        idx = self.indexAt(event.pos())
        if not idx.isValid():
            return self.model().rowCount()
        rect = self.visualRect(idx)
        y = event.pos().y()
        margin = 4
        if y - rect.top() < margin:
            return idx.row()
        if rect.bottom() - y < margin:
            return idx.row() + 1
        return idx.row()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.dropIndicatorRow is not None:
            painter = QPainter(self.viewport())
            pen = QPen(QColor(0, 120, 215), 2)  # blue line
            painter.setPen(pen)

            if self.dropIndicatorRow < self.model().rowCount():
                rect = self.visualRect(self.model().index(self.dropIndicatorRow, 0))
                y = rect.top()
            else:
                # draw at the bottom of the last row
                if self.model().rowCount() == 0:
                    y = 0
                else:
                    rect = self.visualRect(self.model().index(self.model().rowCount() - 1, 0))
                    y = rect.bottom()

            painter.drawLine(0, y, self.viewport().width(), y)

class TranscriptModel(QAbstractTableModel):
    def __init__(self, transcript: list[dict], parent=None):
        super().__init__(parent)
        self._data = transcript
        self._headers = ["Start", "End", "Speaker", "Text"]

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
        # For both display _and_ in-place editing, return the same text
        if role not in (Qt.DisplayRole, Qt.EditRole):
            return QVariant()
        entry = self._data[index.row()]
        col = index.column()
        if col == 0:
            return seconds_to_formatted(entry.get("start", 0))
        elif col == 1:
            return seconds_to_formatted(entry.get("end", 0))
        elif col == 2:
            return entry.get("speaker", "speaker_0")
        elif col == 3:
            return entry.get("text", "")
        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]
        return QVariant()

    def updateTranscript(self, new_transcript: list[dict]):
        self.beginResetModel()
        self._data = new_transcript
        self.endResetModel()

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsDropEnabled
        base = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        # allow editing on all four columns
        return base | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False

        row = index.row()
        col = index.column()
        txt = value if isinstance(value, str) else value.toString()

        # Time‐columns: allow raw seconds or formatted strings
        if col in (0, 1):
            try:
                # 1) try raw float parse
                secs = float(txt)
            except ValueError:
                # 2) fallback to your formatted parser
                secs = formatted_to_seconds(txt)
            # store back into the right key
            key = 'start' if col == 0 else 'end'
            self._data[row][key] = secs

        elif col == 2:
            self._data[row]['speaker'] = txt

        else:  # col == 3
            self._data[row]['text'] = txt

        # let the view repaint that one cell (it will call data() → seconds_to_formatted)
        self.dataChanged.emit(index, index, [Qt.DisplayRole])
        return True
    
    def supportedDropActions(self):
        # internal move only – no copy
        return Qt.MoveAction

    def moveRows(self, parent, sourceRow, count, destinationParent, destinationChild):
        """Re-order self._data so InternalMove works."""
        if count != 1 or parent != destinationParent:
            return False  # only single-row moves in same parent (flat table)

        if sourceRow < destinationChild:
            destinationChild -= 1  # Qt’s docs: adjust when moving downward

        if sourceRow == destinationChild:
            return False  # no‐op

        self.beginMoveRows(parent, sourceRow, sourceRow,
                           destinationParent, destinationChild)
        row = self._data.pop(sourceRow)
        self._data.insert(destinationChild, row)
        self.endMoveRows()
        return True


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
        self.prev_loop_start = 0.0
        self.prev_loop_end   = 0.0

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
        self.highlight_timer = QTimer(self)
        self.highlight_timer.setInterval(200)  # check every 200 ms
        self.highlight_timer.timeout.connect(self.update_highlighting)

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
        self.table = DragTableView()
        self.table.setDragDropMode(QAbstractItemView.InternalMove)
        self.table.setDefaultDropAction(Qt.MoveAction)
        pal = self.table.palette()
        blue = QColor(51, 153, 255)
        pal.setColor(QPalette.Highlight, blue)
        pal.setColor(QPalette.Inactive, QPalette.Highlight, blue)
        pal.setColor(QPalette.HighlightedText, Qt.white)
        pal.setColor(QPalette.Inactive, QPalette.HighlightedText, Qt.white)
        self.table.setPalette(pal)
        self.model = TranscriptModel(self.transcript, parent=self)
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.installEventFilter(self)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked |
            QAbstractItemView.EditKeyPressed
        )
        self.table.setDropIndicatorShown(True)
        self.model.dataChanged.connect(self.on_cell_changed)
        header = self.table.horizontalHeader()
        for col in (0, 1, 2):
            header.setSectionResizeMode(col, QHeaderView.Interactive)
            ideal = header.sectionSizeHint(col)
            header.resizeSection(col, ideal + COL_CUSHION)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        v = self.table.verticalHeader()
        v.setDefaultSectionSize(24)
        self.table.setWordWrap(True)
        self.table.setSortingEnabled(False)
        self.table.clicked.connect(
            lambda idx: self.on_cell_clicked(idx.row(), idx.column())
        )

        right_layout.addWidget(self.table)

        control_layout = QHBoxLayout()
        self.auto_seek_btn = QPushButton("Jump to: Enabled")
        self.auto_seek_btn.clicked.connect(self.toggle_auto_seek)
        self.add_btn = QPushButton("Add Line")
        self.add_btn.clicked.connect(self.add_line)
        self.del_btn = QPushButton("Delete Line")
        self.del_btn.clicked.connect(self.delete_line)
        self.sort_btn = QPushButton("Sort by Start")
        self.sort_btn.clicked.connect(self.sort_transcript_by_start)
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
        if self.current_row is not None and 0 <= self.current_row < len(
                self.transcript):
            # 1) Update the underlying data
            self.transcript[self.current_row][
                'speaker'] = f"speaker_{speaker_id}"
            # 2) Emit a dataChanged signal so the view refreshes that one cell
            idx = self.model.index(self.current_row, 2)  # column 2 is “Speaker”
            self.model.dataChanged.emit(idx, idx, [Qt.DisplayRole])

    def eventFilter(self, source, event):
        if source == self.table and event.type() == QEvent.KeyPress:
            if not isinstance(self.focusWidget(), QLabel):
                idx = self.table.currentIndex()
                row = idx.row()

                if event.key() == Qt.Key_Space:
                    self.toggle_play()
                    return True

                elif event.key() in (Qt.Key_Enter, Qt.Key_Return):
                    if idx.isValid():
                        self.table.edit(idx)
                    return True

                elif event.key() == Qt.Key_Up:
                    if row > 0:
                        new_row = row - 1
                        self.table.selectRow(new_row)
                        self.on_cell_clicked(new_row, 0)
                    return True

                elif event.key() == Qt.Key_Down:
                    # use model.rowCount() instead of table.rowCount()
                    if row < self.model.rowCount() - 1:
                        new_row = row + 1
                        self.table.selectRow(new_row)
                        self.on_cell_clicked(new_row, 0)
                    return True

        return super().eventFilter(source, event)

    def load_transcript(self):
        self.disable_all_buttons()
        try:
            file, _ = QFileDialog.getOpenFileName(self, "Open Transcript File", "", "JSON Files (*.json);;All Files (*)")
            if not file:
                return

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
            self.model.updateTranscript(self.transcript)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load transcript: {e}")
        finally:
            self.enable_all_buttons()

    def load_media(self):
        # disable up front
        self.disable_all_buttons()

        # 1) pick file
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Open Media File",
            "",
            "Video/Audio Files (*.wmv *.wav);;All Files (*)"
        )
        # if user cancels, re-enable and bail
        if not file:
            self.enable_all_buttons()
            return

        # 2) set up the media and spinner
        self.video_path = file
        self.loading_text.setText("Loading")
        self.spinner_label.setVisible(True)
        self.media_placeholder.show()

        # 3) attempt to load
        try:
            media = QMediaContent(QUrl.fromLocalFile(self.video_path))
            self.player.setMedia(media)

            # clean up any old waveform
            if self.waveformProgress is not None:
                self.waveformProgress.setParent(None)
                self.waveformProgress.deleteLater()

            # build new waveform view
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

                idx = self.video_layout.indexOf(self.media_placeholder)
                self.video_layout.removeWidget(self.media_placeholder)
                self.media_placeholder.deleteLater()
                self.video_layout.insertWidget(idx, self.waveformProgress)

            except Exception as e_wave:
                QMessageBox.warning(self, "Error",
                                    f"Failed to load waveform: {e_wave}")
                # if waveform fails, re-enable so the user can try again
                self.enable_all_buttons()

        except Exception as e_media:
            QMessageBox.warning(self, "Error",
                                f"Failed to load media: {e_media}")
            # media itself failed—re-enable immediately
            self.enable_all_buttons()

    def sort_transcript_by_start(self):
        self.transcript.sort(key=lambda r: r.get("start", 0.0))
        self.model.updateTranscript(self.transcript)

    def on_cell_clicked(self, row, col):
        # guard against out-of-range
        if row < 0 or row >= self.model.rowCount():
            return

        # pull the display text
        start_text = self.model.data(self.model.index(row, 0), Qt.DisplayRole)
        end_text = self.model.data(self.model.index(row, 1), Qt.DisplayRole)
        if start_text is None or end_text is None:
            return

        try:
            prev_row = self.current_row
            # update current_row first so spin callbacks use the right index
            self.current_row = row

            # only auto-seek when clicking a *different* row
            if (row != prev_row and self.auto_seek_enabled
                    and self.video_path and self.player.duration() > 0):

                start_secs = formatted_to_seconds(start_text)
                if start_secs * 1000 > self.player.duration():
                    QMessageBox.warning(self, "Invalid Time",
                                        "The specified start time exceeds media duration.")
                    return

                # jump to that time
                self.player.setPosition(int(start_secs * 1000))

                # update loop bounds
                self.loop_start = start_secs
                try:
                    end_secs = formatted_to_seconds(end_text)
                except ValueError:
                    end_secs = start_secs
                self.loop_end = end_secs

                # block spin-box signals so we don't re-enter loop_*_changed
                self.loop_start_spin.blockSignals(True)
                self.loop_end_spin.blockSignals(True)

                self.loop_start_spin.setValue(self.loop_start)
                self.loop_end_spin.setValue(self.loop_end)

                self.loop_start_spin.blockSignals(False)
                self.loop_end_spin.blockSignals(False)

                # update the waveform overlay if you have one
                if self.waveformProgress is not None:
                    self.waveformProgress.set_loop_boundaries(
                        self.loop_start, self.loop_end)

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

    def on_cell_changed(self, topLeft: QModelIndex, bottomRight: QModelIndex,
                        roles):
        # we only care about display edits
        if Qt.DisplayRole not in roles:
            return

        row = topLeft.row()
        col = topLeft.column()
        # only “Start” (col 0) or “End” (col 1) need normalization/spin-box sync
        if col not in (0, 1):
            return

        # 1) grab the raw text the user just typed
        raw = self.model.data(topLeft, Qt.DisplayRole)

        # 2) normalize it (this will raise if it’s totally invalid)
        try:
            normalized = self.normalize_time_entry(raw)
            seconds = formatted_to_seconds(normalized)
        except ValueError:
            return

        # 3) update your underlying transcript
        key = "start" if col == 0 else "end"
        self.transcript[row][key] = seconds

        # 4) push the normalized text back into the view
        #    block the model’s own signals briefly to avoid recursion
        self.model.blockSignals(True)
        #    you can directly mutate the model’s data store since you’ll emit yourself
        self.model._data[row][key] = seconds
        self.model.dataChanged.emit(topLeft, topLeft, [Qt.DisplayRole])
        self.model.blockSignals(False)

        # 5) if this is the current row, update your loop spin-boxes
        if self.current_row == row:
            if col == 0:
                self.loop_start = seconds
                self.loop_start_spin.setValue(seconds)
            else:
                self.loop_end = seconds
                self.loop_end_spin.setValue(seconds)

    def loop_start_changed(self, val):
        # Prevent invalid ranges
        if self.loop_end is not None and val > self.loop_end:
            self.loop_start_spin.blockSignals(True)
            self.loop_start_spin.setValue(self.prev_loop_start)
            self.loop_start_spin.blockSignals(False)
            return

        # 1) update your loop bounds
        self.loop_start = val
        self.prev_loop_start = val

        # 2) if a row is selected, update its 'start' value in your transcript list
        if self.current_row is not None and 0 <= self.current_row < len(
                self.transcript):
            self.transcript[self.current_row]['start'] = val

            # 3) notify the model/view that one cell changed
            idx: QModelIndex = self.model.index(self.current_row, 0)
            self.model.dataChanged.emit(idx, idx, [Qt.DisplayRole])

        # 4) update the waveform if needed
        if self.waveformProgress is not None:
            end = self.loop_end if self.loop_end is not None else self.loop_start
            self.waveformProgress.set_loop_boundaries(self.loop_start, end)

    def loop_end_changed(self, val):
        # Prevent invalid ranges
        if self.loop_start is not None and val < self.loop_start:
            self.loop_end_spin.blockSignals(True)
            self.loop_end_spin.setValue(self.prev_loop_end)
            self.loop_end_spin.blockSignals(False)
            return

        # 1) Update your loop bounds
        self.loop_end = val
        self.prev_loop_end = val

        # 2) If a row is selected, update its 'end' value in your transcript list
        if self.current_row is not None and 0 <= self.current_row < len(
                self.transcript):
            self.transcript[self.current_row]['end'] = val

            # 3) Notify the model/view that this one cell changed
            idx: QModelIndex = self.model.index(self.current_row, 1)
            self.model.dataChanged.emit(idx, idx, [Qt.DisplayRole])

        # 4) Update the waveform loop boundaries
        start = self.loop_start if self.loop_start is not None else val
        if self.waveformProgress is not None:
            self.waveformProgress.set_loop_boundaries(start, self.loop_end)

    def toggle_loop(self):
        self.loop_enabled = not self.loop_enabled
        state = "Enabled" if self.loop_enabled else "Disabled"
        self.toggle_loop_button.setText(f"Looping: {state}")

        # if looping just got turned off *and* we're already playing, start the highlight timer
        if not self.loop_enabled and self.player.state() == QMediaPlayer.PlayingState:
            self.highlight_timer.start()
        # if looping just got turned on, we don't want highlighting
        elif self.loop_enabled:
            self.highlight_timer.stop()

        # existing behavior: if loop now enabled, force seek to loop start
        if self.loop_enabled and self.loop_start is not None and self.loop_end is not None:
            pos = self.player.position()
            start_ms = int(self.loop_start * 1000)
            end_ms   = int(self.loop_end * 1000)
            if pos < start_ms or pos > end_ms:
                self.player.setPosition(start_ms)

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
            self.highlight_timer.stop()
        else:
            self.player.play()
            self.play_button.setIcon(self.pause_icon)
            if not self.loop_enabled:
                self.highlight_timer.start()

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
        # 1) Figure out where to insert
        current = self.table.currentIndex().row()
        pos = current + 1 if current != -1 else len(self.transcript)

        # 2) Prepare a blank entry
        new_entry = {
            "start": 0.0,
            "end": 0.0,
            "speaker": "speaker_0",
            "text": ""
        }

        # 3) Tell Qt you’re inserting a row
        self.model.beginInsertRows(QModelIndex(), pos, pos)
        self.transcript.insert(pos, new_entry)
        self.model.endInsertRows()

        # 4) (Optional) select the new row
        idx = self.model.index(pos, 0)
        self.table.setCurrentIndex(idx)

    def delete_line(self):
        # only if a row is selected
        row = self.table.currentIndex().row()
        if row is None or row < 0 or row >= len(self.transcript):
            return

        # 1) Tell Qt you’re removing a row
        self.model.beginRemoveRows(QModelIndex(), row, row)
        self.transcript.pop(row)
        self.model.endRemoveRows()

        # 2) reset any loop state
        self.current_row = None
        self.loop_start = None
        self.loop_end = None
        self.loop_start_spin.setValue(0.0)
        self.loop_end_spin.setValue(0.0)

    def save_transcript(self):
        # if there's nothing to write, warn
        if not self.transcript:
            QMessageBox.warning(self, "No Data", "No transcript data to save.")
            return

        # suggest a default file name
        if self.json_path:
            base, _ = os.path.splitext(self.json_path)
            default_path = base + "_manual_edit.json"
        else:
            default_path = "transcript_manual_edit.json"

        # ask where to save
        save_file, _ = QFileDialog.getSaveFileName(
            self, "Save Transcript", default_path,
            "JSON Files (*.json);;All Files (*)"
        )
        if not save_file:
            return

        # dump the transcript list directly
        with open(save_file, "w") as f:
            json.dump(self.transcript, f, indent=2)

        QMessageBox.information(self, "Saved",
                                f"Transcript saved to {save_file}")
    
    def update_highlighting(self):
        """Called by timer—find which rows cover the current time and highlight them."""
        # don’t highlight if looping is on or if paused
        if self.loop_enabled or self.player.state() != QMediaPlayer.PlayingState:
            return

        current_secs = self.player.position() / 1000.0
        active_rows = []
        for i, entry in enumerate(self.transcript):
            if entry["start"] <= current_secs <= entry["end"]:
                active_rows.append(i)

        self.highlight_rows(active_rows)

    def highlight_rows(self, rows: list[int]):
        sel = self.table.selectionModel()
        sel.clearSelection()
        first_idx = None
        for r in rows:
            idx = self.model.index(r, 0)
            if first_idx is None:
                first_idx = idx
                flags = QItemSelectionModel.Rows | QItemSelectionModel.ClearAndSelect
            else:
                flags = QItemSelectionModel.Rows | QItemSelectionModel.Select
            sel.select(idx, flags)
        if first_idx:
            self.table.scrollTo(first_idx, QAbstractItemView.PositionAtCenter)

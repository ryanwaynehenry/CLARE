from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QGraphicsView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor

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
        from PyQt5.QtGui import QDrag
        from PyQt5.QtCore import QMimeData
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
            self.blockSignals(True)
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

import re
from PyQt5.QtCore import QBuffer, QByteArray, Qt
from PyQt5.QtGui import QMovie, QValidator
from PyQt5.QtWidgets import QLabel, QDoubleSpinBox, QTableWidgetItem

# Spinner GIF data
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

class TimeSpinBox(QDoubleSpinBox):
    """
    A custom spin box that displays and accepts time in m:ss.d or h:mm:ss.d format.
    The user may type anything; the value (in seconds) is only updated when the user
    clicks outside the box or presses Enter. If the entered text is invalid, it reverts
    to the last valid value.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Disable keyboard tracking so that value isn't updated on every key press.
        self.setKeyboardTracking(False)
        self.lineEdit().setValidator(None)
        # Store the last valid value (in seconds)
        self._lastValidValue = 0.0
        # Optionally, set a default value:
        self.setValue(0.0)
    
    def textFromValue(self, value):
        # If less than one hour, format as m:ss.d; otherwise, h:mm:ss.d.
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
    
    def focusOutEvent(self, event):
        # When focus is lost, validate the text.
        self.validate_and_update()
        super().focusOutEvent(event)
    
    def keyPressEvent(self, event):
        # If the user presses Enter, validate and update.
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.validate_and_update()
        else:
            super().keyPressEvent(event)
    
    def validate_and_update(self):
        text = self.lineEdit().text().strip()
        # Only update if there is something typed.
        if text:
            try:
                new_val = formatted_to_seconds(text)
                self.setValue(new_val)
                self._lastValidValue = new_val
            except Exception:
                # If invalid, revert to the last valid value.
                self.lineEdit().setText(self.textFromValue(self._lastValidValue))
        else:
            # If the field is empty, revert to the last valid text.
            self.lineEdit().setText(self.textFromValue(self._lastValidValue))        
class TimeTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        try:
            self_time = formatted_to_seconds(self.text())
        except Exception:
            self_time = 0
        try:
            other_time = formatted_to_seconds(other.text())
        except Exception:
            other_time = 0
        return self_time < other_time

import sys
from PyQt5.QtWidgets import QApplication
from transcript_editor import TranscriptEditor

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = TranscriptEditor()
    editor.show()
    sys.exit(app.exec_())

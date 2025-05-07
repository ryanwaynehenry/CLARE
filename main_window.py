import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget, QPushButton, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon
from transcript_editor import TranscriptEditor
from knowledge_graph_tab import KnowledgeGraphTab

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manual Transcription Editor and AI-Enabled Knowledge Graph Constructor")
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create the transcript editor tab.
        self.transcript_editor = TranscriptEditor()
        # Create the knowledge graph tab using a callback to fetch transcript text.
        self.knowledge_graph_tab = KnowledgeGraphTab(transcript_provider=lambda: self.transcript_editor.transcript)

        self.tabs.addTab(self.transcript_editor, "Transcript Editor")
        self.tabs.addTab(self.knowledge_graph_tab, "Knowledge Graph")

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)

    # Point-sized font goes DPI-aware
    font = QFont("Segoe UI", pointSize=9)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

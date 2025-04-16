import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget
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
        self.knowledge_graph_tab = KnowledgeGraphTab(self.get_transcript_text)

        self.tabs.addTab(self.transcript_editor, "Transcript Editor")
        self.tabs.addTab(self.knowledge_graph_tab, "Knowledge Graph")

    def get_transcript_text(self):
        texts = []
        for entry in self.transcript_editor.transcript:
            texts.append(entry.get("text", ""))
        return "\n".join(texts)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

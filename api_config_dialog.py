# api_config_dialog.py
import os
import yaml
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QWidget,
    QGroupBox, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt
import utils

class ApiConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure API Keys")

        # --- Tighter margins and larger text ---
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)
        # global stylesheet for tighter margins
        self.setStyleSheet("""
            QLabel { margin: 2px; font-size: 12pt; }
            QGroupBox {
                margin-top: 4px;
                margin-bottom: 4px;
                font-size: 13pt;
                font-weight: bold;
            }
            QLineEdit { margin: 2px; padding: 2px; font-size: 12pt; }
            QPushButton { font-size: 12pt }
        """)
        self.resize(600, 500)

        # load or init config
        cfg_path = "config.yaml"
        try:
            with open(cfg_path) as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.config = {}

        self.fields = {}
        layout = QVBoxLayout(self)

        # Column headers
        header_layout = QHBoxLayout()
        left_header = QLabel("LLM Providers")
        left_header.setAlignment(Qt.AlignCenter)
        right_header = QLabel("Embedding Providers")
        right_header.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(left_header)
        header_layout.addWidget(right_header)
        layout.addLayout(header_layout)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(10, 10, 10, 10)

        # determine provider lists
        llm_providers = [
            name.replace("_llm_models", "").title()
            for name in dir(utils)
            if name.endswith("llm_models")
        ]
        embed_providers = [
            name.replace("_embedding_models", "").title()
            for name in dir(utils)
            if name.endswith("embedding_models")
        ]

        # Left column: LLM providers
        llm_widget = QWidget()
        llm_layout = QVBoxLayout(llm_widget)
        llm_layout.setSpacing(10)
        llm_layout.setContentsMargins(0, 0, 0, 0)

        for prov in llm_providers:
            key = prov.lower()
            group = QGroupBox(prov)
            group.setStyleSheet("""
                                QGroupBox {
                                    padding-top: 20px;     /* push contents down */
                                }
                            """)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(5)
            group_layout.setContentsMargins(5, 5, 5, 5)

            if key == "bedrock":
                for envname in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"):
                    edit = QLineEdit()
                    edit.setMinimumHeight(28)
                    edit.setPlaceholderText(envname)
                    existing = (
                        self.config.get(key, {}).get(envname)
                        or os.environ.get(envname, "")
                    )
                    edit.setText(existing)
                    group_layout.addWidget(edit)
                    self.fields[f"{key}.{envname}"] = edit

            elif key == "ollama":
                edit = QLineEdit()
                edit.setMinimumHeight(28)
                edit.setPlaceholderText("API Base URL (e.g. http://localhost:11434)")
                existing = (
                    self.config.get(key, {}).get("api_base")
                    or os.environ.get("OLLAMA_API_BASE", "")
                )
                edit.setText(existing)
                group_layout.addWidget(edit)
                self.fields[f"{key}.api_base"] = edit

            else:
                edit = QLineEdit()
                edit.setMinimumHeight(28)
                edit.setPlaceholderText("API Key")
                envkey = f"{key.upper()}_API_KEY"
                existing = (
                    self.config.get(key, {}).get("api_key")
                    or os.environ.get(envkey, "")
                )
                edit.setText(existing)
                group_layout.addWidget(edit)
                self.fields[f"{key}.api_key"] = edit

            llm_layout.addWidget(group)

        llm_widget.setSizePolicy(llm_widget.sizePolicy().horizontalPolicy(),
                                   QSizePolicy.Maximum)
        content_layout.addWidget(llm_widget, 0, Qt.AlignTop)

        # Right column: embedding providers
        embed_widget = QWidget()
        embed_layout = QVBoxLayout(embed_widget)
        embed_layout.setSpacing(10)
        embed_layout.setContentsMargins(0, 0, 0, 0)

        for prov in embed_providers:
            key = prov.lower()
            if key == "bedrock":
                continue  # skip, already handled

            group = QGroupBox(prov)
            group.setStyleSheet("""
                    QGroupBox {
                        padding-top: 20px;     /* push contents down */
                    }
                """)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(5)
            group_layout.setContentsMargins(5, 5, 5, 5)

            if key == "local":
                # Local Sentence‐Transformer models need no key
                lbl = QLabel(
                    "Uses local Sentence Transformer models\nNo API key required")
                lbl.setWordWrap(True)
                group_layout.addWidget(lbl)
            else:
                edit = QLineEdit()
                edit.setMinimumHeight(28)
                edit.setPlaceholderText("API Key")
                section = f"{key}_embedding"
                envkey = f"{key.upper()}_EMBEDDING_API_KEY"
                existing = (
                        self.config.get(section, {}).get("api_key")
                        or os.environ.get(envkey, "")
                )
                edit.setText(existing)
                group_layout.addWidget(edit)
                self.fields[f"{section}.api_key"] = edit

            embed_layout.addWidget(group)

        embed_widget.setSizePolicy(embed_widget.sizePolicy().horizontalPolicy(), QSizePolicy.Maximum)
        content_layout.addWidget(embed_widget, 0, Qt.AlignTop)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Save / Cancel
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _save(self):
        newcfg = {}
        for field, edt in self.fields.items():
            section, subkey = field.split(".", 1)
            val = edt.text().strip()
            secdict = newcfg.setdefault(section, {})
            secdict[subkey] = val

        # write config.yaml
        try:
            with open("config.yaml", "w") as f:
                yaml.safe_dump(newcfg, f)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not write config.yaml:\n{e}")
            return

        # update in-memory copy
        self.config = newcfg
        self.accept()

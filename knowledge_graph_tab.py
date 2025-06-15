import copy
import os
from typing import List, Dict
import bisect
import json
import re
import csv
from PyQt5.QtCore import QUrl, QObject, pyqtSlot, Qt, QPropertyAnimation, QSize, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QMenu,
    QFileDialog, QLineEdit, QLabel, QFormLayout, QGroupBox, QSplitter, QDialog,
    QToolButton, QSizePolicy, QScrollArea, QShortcut, QFrame, QProgressDialog, QApplication,
    QSpinBox
)
from PyQt5.QtGui import QKeySequence, QIcon, QPixmap, QPainter, QFontMetrics
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from pyvis.network import Network
import utils
import yaml
from api_config_dialog import ApiConfigDialog
from clustering_options_dialog import ClusteringOptionsDialog

# Import your autoKG class
# Make sure that autoKG.py is in a location where Python can find it (same folder or properly installed).
from autokg import autoKG

MIN_LABEL_WIDTH = 40  # pixels
MAX_LABEL_WIDTH = 180  # pixels
MAX_WORDS = 100

class GraphGenerationThread(QThread):
    finished     = pyqtSignal(list, list)
    errorOccurred= pyqtSignal(str)

    def __init__(
        self,
        texts: List[str],
        sources: List[str],
        llm_model: str,
        embedding_model: str,
        llm_key1: str, llm_key2: str, llm_key3: str,
        emb_key1: str, emb_key2: str, emb_key3: str,
        main_topic: str,
        n_clusters: int = None, 
        num_topics: int = 10,
        dynamic_threshold=50,
        parent: QObject = None,
    ):
        super().__init__(parent)
        self.texts = texts
        self.sources = sources
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.llm_key1, self.llm_key2, self.llm_key3 = llm_key1, llm_key2, llm_key3
        self.emb_key1, self.emb_key2, self.emb_key3 = emb_key1, emb_key2, emb_key3
        self.main_topic = main_topic
        self.n_clusters = None if n_clusters == 0 else n_clusters  # Convert 0 to None for auto
        self.num_topics = num_topics
        self.dynamic_threshold = dynamic_threshold

    def run(self):
        try:
            if not self.texts:
                self.errorOccurred.emit("Transcript is empty.")
                return

            auto_kg = autoKG(
                texts=self.texts,
                source=self.sources,
                embedding_model=self.embedding_model,
                llm_model=self.llm_model,
                embedding_api_key=self.emb_key1,
                llm_api_key=self.llm_key1,
                main_topic=self.main_topic,
                embed=True,
                embedding_key2=self.emb_key2,
                embedding_key3=self.emb_key3,
                llm_key2=self.llm_key2,
                llm_key3=self.llm_key3,
            )

            auto_kg.make_graph(k=5, method='annoy', similarity='angular', kernel='gaussian')
            auto_kg.remove_same_text(use_nn=True, n_neighbors=3, thresh=1e-6, update=True)
            auto_kg.cluster(
                n_clusters=self.n_clusters,
                clustering_method='k_means',
                max_texts=8,
                select_mtd='similarity',
                prompt_language='English',
                num_topics=self.num_topics,
                max_length=3,
                post_process=True,
                add_keywords=True,
                verbose=False
            )
            auto_kg.coretexts_seg_individual(
                trust_num=5,
                core_labels=None,
                dist_metric='cosine',
                negative_multiplier=3,
                seg_mtd='laplace',
                return_mat=True,
                connect_threshold=1
            )

            threshold = auto_kg.apply_dynamic_threshold(self.dynamic_threshold)

            edges = auto_kg.build_entity_relationships(transcript_str=' '.join(self.texts))

            nodes = auto_kg.keywords.copy()
            triples = []
            for kw1, relation, kw2, direction in edges:
                if direction == "forward":
                    triples.append({"subject": kw1, "relation": relation, "object": kw2})
                else:
                    triples.append({"subject": kw2, "relation": relation, "object": kw1})

            self.finished.emit(nodes, triples)
        except Exception as e:
            self.errorOccurred.emit(str(e))

class GraphEditorBridge(QObject):
    """
    Bridge for communication between JavaScript and Python.
    Tracks selected node or edge for deletion or for pick actions.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selectedNode = None
        self.selectedEdge = None
        self.parentTab = parent

    @pyqtSlot(str)
    def nodeClicked(self, nodeId):
        print("Bridge: Node clicked:", nodeId)
        self.selectedNode = nodeId
        self.selectedEdge = None
        self.parentTab.on_node_selected(nodeId)

    @pyqtSlot(str)
    def edgeClicked(self, edgeId):
        print("Bridge: Edge clicked:", edgeId)
        self.selectedEdge = edgeId
        self.selectedNode = None
        self.parentTab.on_edge_selected(edgeId)

    @pyqtSlot()
    def deleteSelected(self):
        print("Bridge: Delete key pressed")
        if self.selectedNode:
            print("Bridge: Deleting selected node:", self.selectedNode)
            self.parentTab.delete_node_via_bridge(self.selectedNode)
            self.selectedNode = None
        elif self.selectedEdge:
            print("Bridge: Deleting selected edge:", self.selectedEdge)
            self.parentTab.delete_edge_via_bridge(self.selectedEdge)
            self.selectedEdge = None
        else:
            print("Bridge: No element selected for deletion.")
        self.parentTab.build_graph()

class CollapsibleWidget(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QToolButton(text=title, checkable=True, checked=True)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.clicked.connect(self.on_toggle)

        self.content_area = QWidget()
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_area)

        self.toggle_animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.toggle_animation.setDuration(150)

    def setContentLayout(self, layout):
        self.content_area.setLayout(layout)
        content_height = layout.sizeHint().height()
        self.toggle_animation.setStartValue(0)
        self.toggle_animation.setEndValue(content_height)
        self.content_area.setMaximumHeight(content_height)

    def on_toggle(self):
        try:
            self.toggle_animation.finished.disconnect()  # disconnect any previous connections
        except Exception:
            pass

        if self.toggle_button.isChecked():
            # Expand: show the content_area and animate to full height.
            self.toggle_button.setArrowType(Qt.DownArrow)
            self.content_area.setVisible(True)
            self.toggle_animation.setDirection(QPropertyAnimation.Forward)
        else:
            # Collapse: animate backward and, when finished, hide the content area.
            self.toggle_button.setArrowType(Qt.RightArrow)
            self.toggle_animation.setDirection(QPropertyAnimation.Backward)
            self.toggle_animation.finished.connect(
                lambda: self.content_area.setVisible(False))
        self.toggle_animation.start()

class KnowledgeGraphTab(QWidget):
    def __init__(self, transcript_provider, parent=None):
        super().__init__(parent)
        self.transcript_provider = transcript_provider
        self.undo_stack = []
        self.max_undo = 10
        self.current_search_results = []
        self.current_search_index = 0

        self.triples = []
        self.nodes = []
        self.pyvis_net = None
        self.currentFieldSelection = None

        main_layout = QVBoxLayout(self)

        # Create a vertical separator widget
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setLineWidth(1)

        # --- Top Ribbon ---
        ribbon_widget = QWidget()
        ribbon_layout = QHBoxLayout(ribbon_widget)
        ribbon_layout.setContentsMargins(0, 0, 0, 0)
        ribbon_layout.setSpacing(5)

        # Create container widgets for each section:
        graph_generation_widget = QWidget()
        graph_generation_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        graph_generation_layout = QHBoxLayout()
        graph_generation_layout.setContentsMargins(0, 0, 0, 0)
        graph_generation_layout.setSpacing(5)

        # Generate button with a symbol
        self.generate_btn = QPushButton()
        generate_from_transcript_path = os.path.join("imgs", "generate_from_transcript.png")
        self.generate_btn.setIcon(QIcon(generate_from_transcript_path))
        self.generate_btn.setIconSize(QSize(28, 28))  # or set a custom size
        self.generate_btn.setFixedSize(36, 36)
        self.generate_btn.setToolTip("Generate Graph from Transcript")
        self.generate_btn.clicked.connect(self.generate_graph)
        graph_generation_layout.addWidget(self.generate_btn)

        # Main Topic input (no preceding text label); update placeholder text.
        self.main_topic_field = QLineEdit()
        self.main_topic_field.setPlaceholderText(
            "Optional: Specify central topic for entities (e.g. chemistry)"
        )
        self.main_topic_field.setFixedSize(310, 40)
        graph_generation_layout.addWidget(self.main_topic_field)

        # Add Advanced Clustering Options button
        self.clustering_btn = QPushButton("Clustering Options")
        self.clustering_btn.setToolTip("Configure advanced clustering options")
        self.clustering_btn.setFixedHeight(40)
        self.clustering_btn.clicked.connect(self.open_clustering_options)
        graph_generation_layout.addWidget(self.clustering_btn)

        # Initialize default clustering values
        self.n_clusters = None  # Auto
        self.num_topics = 10    # Default value
        self.dynamic_threshold = 50

        # Select LLM model drop-down button
        self.llm_btn = QToolButton()
        self.llm_btn.setText("LLM Model")
        self.llm_btn.setPopupMode(QToolButton.InstantPopup)
        self.llm_btn.setMinimumSize(MIN_LABEL_WIDTH, 40)
        self.llm_btn.setStyleSheet("""
                   QToolButton { padding-right:20px; }
                   QToolButton::menu-indicator {
                     subcontrol-origin: padding;
                     subcontrol-position: right center;
                     margin-left: 5px;
                   }
               """)
        self.llm_menu = QMenu()
        for name in dir(utils):
            if name.endswith("llm_models"):
                provider = name.replace("_llm_models", "").title()
                sub = self.llm_menu.addMenu(provider)
                sub_action = sub.menuAction()
                sub_action.setCheckable(True)
                for model in getattr(utils, name):
                    act = sub.addAction(model)
                    act.setCheckable(True)
                    act.triggered.connect(lambda _, m=model: self.set_llm_model(m))
        self.llm_btn.setMenu(self.llm_menu)
        self.llm_btn.setMaximumWidth(MAX_LABEL_WIDTH + 40)
        graph_generation_layout.addWidget(self.llm_btn)

        # Embedder-selector button
        self.embed_btn = QToolButton()
        self.embed_btn.setText("Embedding Model")
        self.embed_btn.setPopupMode(QToolButton.InstantPopup)
        self.embed_btn.setMinimumSize(MIN_LABEL_WIDTH, 40)
        self.embed_btn.setStyleSheet("""
                   QToolButton { padding-right:20px; }
                   QToolButton::menu-indicator {
                     subcontrol-origin: padding;
                     subcontrol-position: right center;
                     margin-left: 5px;
                   }
               """)
        self.embed_menu = QMenu()
        # find all lists in utils that end with "embedding_models"
        for name in dir(utils):
            if name.endswith("embedding_models"):
                provider = name.replace("_embedding_models", "").title()
                sub = self.embed_menu.addMenu(provider)
                sub_action = sub.menuAction()
                sub_action.setCheckable(True)
                for model in getattr(utils, name):
                    act = sub.addAction(model)
                    act.setCheckable(True)
                    act.triggered.connect(
                        lambda _, m=model: self.set_embedding_model(m))
        self.embed_btn.setMenu(self.embed_menu)
        self.embed_btn.setMaximumWidth(MAX_LABEL_WIDTH + 40)
        graph_generation_layout.addWidget(self.embed_btn)

        # — API Keys button —
        self.api_btn = QPushButton("API Keys")
        self.api_btn.setMinimumSize(MIN_LABEL_WIDTH, 40)
        self.api_btn.setToolTip("Configure provider API keys")
        self.api_btn.clicked.connect(self.open_api_config)
        graph_generation_layout.addWidget(self.api_btn)

        graph_generation_widget.setLayout(graph_generation_layout)
        graph_generation_section = QWidget()
        graph_generation_section_layout = QVBoxLayout()
        graph_generation_section_layout.setContentsMargins(0, 0, 0, 0)
        graph_generation_section_layout.setSpacing(2)
        graph_generation_label = QLabel("Graph Generation")
        graph_generation_label.setStyleSheet(
            "font-size: 14pt; color: black;")  # adjust style as needed
        graph_generation_section_layout.addWidget(graph_generation_label)
        graph_generation_section_layout.addWidget(graph_generation_widget)
        graph_generation_section.setLayout(graph_generation_section_layout)

        # Graph IO Section
        graph_io_widget = QWidget()
        graph_io_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        graph_io_layout = QHBoxLayout()
        graph_io_layout.setContentsMargins(0, 0, 0, 0)
        graph_io_layout.setSpacing(5)

        # Import button: uses an open folder icon
        self.import_csv_btn = QPushButton()
        open_icon_path = os.path.join("imgs", "folder.png")
        self.import_csv_btn.setIcon(QIcon(open_icon_path))
        self.import_csv_btn.setIconSize(QSize(28, 28))
        self.import_csv_btn.setFixedSize(36, 36)
        self.import_csv_btn.setToolTip("Import Graph from CSV Files")
        self.import_csv_btn.clicked.connect(self.import_graph_from_csv)
        graph_io_layout.addWidget(self.import_csv_btn)

        # Export button: uses a save icon
        self.export_btn = QPushButton()
        save_icon_path = os.path.join("imgs", "diskette.png")
        self.export_btn.setIcon(QIcon(save_icon_path))
        self.export_btn.setIconSize(QSize(28, 28))
        self.export_btn.setFixedSize(36, 36)
        self.export_btn.setToolTip("Export CSVs for Neo4j")
        self.export_btn.clicked.connect(self.export_graph)
        graph_io_layout.addWidget(self.export_btn)
        graph_io_widget.setLayout(graph_io_layout)

        graph_io_section = QWidget()
        graph_io_section_layout = QVBoxLayout()
        graph_io_section_layout.setContentsMargins(0, 0, 0, 0)
        graph_io_section_layout.setSpacing(2)
        graph_io_label = QLabel("Graph I/O")
        graph_io_label.setStyleSheet(
            "font-size: 14pt; color: black;")  # adjust style as needed
        graph_io_section_layout.addWidget(graph_io_label)
        graph_io_section_layout.addWidget(graph_io_widget)
        graph_io_section.setLayout(graph_io_section_layout)

        graph_editing_widget = QWidget()
        graph_editing_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        graph_editing_layout = QHBoxLayout()
        graph_editing_layout.setContentsMargins(0, 0, 0, 0)
        graph_editing_layout.setSpacing(5)

        # Undo Button with an undo arrow icon (using Unicode character)
        self.undo_btn = QPushButton()
        undo_icon_path = os.path.join("imgs", "undo-circular-arrow.png")
        self.undo_btn.setIcon(QIcon(undo_icon_path))
        self.undo_btn.setIconSize(QSize(28, 28))  # set your preferred icon size
        self.undo_btn.setFixedSize(36, 36)
        self.undo_btn.setToolTip("Undo last action")
        self.undo_btn.clicked.connect(self.undo_action)
        graph_editing_layout.addWidget(self.undo_btn)

        # Delete Node
        self.delete_node_mode_btn = QPushButton()
        delete_node_icon_path = os.path.join("imgs", "delete-node.png")
        self.delete_node_mode_btn.setIcon(QIcon(delete_node_icon_path))
        self.delete_node_mode_btn.setIconSize(QSize(28, 28))
        self.delete_node_mode_btn.setFixedSize(36, 36)
        self.delete_node_mode_btn.setCheckable(True)
        self.delete_node_mode_btn.setToolTip(
            "Delete Nodes. Click on nodes to delete them.")
        self.delete_node_mode_btn.clicked.connect(
            lambda: self.set_pick_mode("delete_node", "multiple", self.delete_node_mode_btn))
        graph_editing_layout.addWidget(self.delete_node_mode_btn)

        # Delete Relationship
        self.delete_relationship_mode_btn = QPushButton()
        delete_relationship_icon_path = os.path.join("imgs", "delete-relationship.png")
        self.delete_relationship_mode_btn.setIcon(QIcon(delete_relationship_icon_path))
        self.delete_relationship_mode_btn.setIconSize(QSize(28, 28))
        self.delete_relationship_mode_btn.setFixedSize(36, 36)
        self.delete_relationship_mode_btn.setCheckable(True)
        self.delete_relationship_mode_btn.setToolTip(
            "Delete Relationships. Click on relationships to delete them.")
        self.delete_relationship_mode_btn.clicked.connect(
            lambda: self.set_pick_mode("delete_relationship", "multiple", self.delete_relationship_mode_btn))
        graph_editing_layout.addWidget(self.delete_relationship_mode_btn)

        self.reverse_relationship_mode_btn = QPushButton()
        reverse_relationship_icon_path = os.path.join("imgs", "realtionship_reverse.png")
        self.reverse_relationship_mode_btn.setIcon(QIcon(reverse_relationship_icon_path))
        self.reverse_relationship_mode_btn.setIconSize(QSize(28, 28))
        self.reverse_relationship_mode_btn.setFixedSize(36, 36)
        self.reverse_relationship_mode_btn.setCheckable(True)
        self.reverse_relationship_mode_btn.setToolTip(
            "Toggle relationship. Click on relationships to flip them.")
        self.reverse_relationship_mode_btn.clicked.connect(
            lambda: self.set_pick_mode("reverse_relationship", "multiple",
                                       self.reverse_relationship_mode_btn))
        graph_editing_layout.addWidget(self.reverse_relationship_mode_btn)
        graph_editing_widget.setLayout(graph_editing_layout)

        graph_editing_section = QWidget()
        graph_editing_section_layout = QVBoxLayout()
        graph_editing_section_layout.setContentsMargins(0, 0, 0, 0)
        graph_editing_section_layout.setSpacing(2)
        graph_editing_label = QLabel("Graph Editing")
        graph_editing_label.setStyleSheet("font-size: 14pt; color: black;")
        graph_editing_section_layout.addWidget(graph_editing_label)
        graph_editing_section_layout.addWidget(graph_editing_widget)
        graph_editing_section.setLayout(graph_editing_section_layout)


        search_widget = QWidget()
        search_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(5)

        # Search: No label before the box.
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search")
        self.search_field.setFixedSize(300, 40)
        # Enable search on return key pressed:
        self.search_field.returnPressed.connect(self.search_node)
        search_layout.addWidget(self.search_field)
        self.search_btn = QPushButton()
        search_icon_path = os.path.join("imgs", "magnifying-glass.png")
        self.search_btn.setIcon(QIcon(search_icon_path))
        self.search_btn.setIconSize(QSize(28, 28))
        self.search_btn.setFixedSize(36, 36)
        self.search_btn.setToolTip("Search Node")
        self.search_btn.clicked.connect(self.search_node)
        search_layout.addWidget(self.search_btn)
        self.search_status_label = QLabel("")
        self.search_status_label.setStyleSheet("font-size: 8pt; color: black;")
        search_layout.addWidget(self.search_status_label)
        search_widget.setLayout(search_layout)

        search_section = QWidget()
        search_section_layout = QVBoxLayout()
        search_section_layout.setContentsMargins(0, 0, 0, 0)
        search_section_layout.setSpacing(2)
        # search_label = QLabel("Search")
        search_label = QLabel("")
        search_label.setStyleSheet("font-size: 14pt; color: black;")
        search_section_layout.addWidget(search_label)
        search_section_layout.addWidget(search_widget)

        search_section.setLayout(search_section_layout)

        ribbon_layout.addWidget(graph_generation_section)
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.VLine)
        separator1.setFrameShadow(QFrame.Sunken)
        ribbon_layout.addWidget(separator1)

        ribbon_layout.addWidget(graph_io_section)
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setFrameShadow(QFrame.Sunken)
        ribbon_layout.addWidget(separator2)

        ribbon_layout.addWidget(graph_editing_section)
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.VLine)
        separator3.setFrameShadow(QFrame.Sunken)
        ribbon_layout.addWidget(separator3)

        ribbon_layout.addStretch()
        ribbon_layout.addWidget(search_section)

        main_layout.addWidget(ribbon_widget)

        # --- Filters Panel (renamed as "Entity Filters" and starting collapsed) ---
        self.filter_box = CollapsibleWidget("Filters")
        filter_layout = QVBoxLayout()

        # Inclusive Filter section → "Include Links to:"
        inclusive_layout = QHBoxLayout()
        inclusive_label = QLabel("Direct Connections:")
        inclusive_pick_multi_btn = QPushButton("Pick")
        inclusive_pick_multi_btn.clicked.connect(
            lambda: self.set_pick_mode("filter_inclusive_multi", "inclusive"))
        self.inclusive_filter_field = QLineEdit()
        self.inclusive_filter_field.setPlaceholderText("entity1; entity2; …")
        inclusive_apply_btn = QPushButton("Apply")
        inclusive_apply_btn.clicked.connect(self.apply_inclusive_filter)
        inclusive_layout.addWidget(inclusive_label)
        inclusive_layout.addWidget(inclusive_pick_multi_btn)
        inclusive_layout.addWidget(self.inclusive_filter_field)
        inclusive_layout.addWidget(inclusive_apply_btn)
        filter_layout.addLayout(inclusive_layout)

        # Exclusive Filter section → "Between Only:"
        exclusive_layout = QHBoxLayout()
        exclusive_label = QLabel("Overlapping Connections:")
        exclusive_pick_multi_btn = QPushButton("Pick")
        exclusive_pick_multi_btn.clicked.connect(
            lambda: self.set_pick_mode("filter_exclusive_multi", "exclusive"))
        self.exclusive_filter_field = QLineEdit()
        self.exclusive_filter_field.setPlaceholderText("entity1; entity2; …")
        exclusive_apply_btn = QPushButton("Apply")
        exclusive_apply_btn.clicked.connect(self.apply_exclusive_filter)
        exclusive_layout.addWidget(exclusive_label)
        exclusive_layout.addWidget(exclusive_pick_multi_btn)
        exclusive_layout.addWidget(self.exclusive_filter_field)
        exclusive_layout.addWidget(exclusive_apply_btn)
        filter_layout.addLayout(exclusive_layout)

        self.clear_filter_btn = QPushButton("Clear Filters")
        self.clear_filter_btn.clicked.connect(self.clear_filters)
        filter_layout.addWidget(self.clear_filter_btn)

        self.filter_box.setContentLayout(filter_layout)
        # Start the filters panel collapsed.
        self.filter_box.toggle_button.setChecked(False)
        self.filter_box.toggle_button.setArrowType(Qt.RightArrow)
        self.filter_box.content_area.setVisible(False)
        main_layout.addWidget(self.filter_box)

        toggle_layout = QHBoxLayout()
        toggle_layout.addStretch()
        self.adv_toggle_btn = QToolButton()
        toggle_icon_path = os.path.join("imgs",
                                        "more.png")  # ensure you have an appropriate icon.
        self.adv_toggle_btn.setIcon(QIcon(toggle_icon_path))
        self.adv_toggle_btn.setIconSize(QSize(24, 24))
        self.adv_toggle_btn.setToolTip("Toggle Advanced Controls")
        self.adv_toggle_btn.clicked.connect(self.toggle_advanced_controls)
        toggle_layout.addWidget(self.adv_toggle_btn)
        # Add this toggle layout to the main layout.
        main_layout.addLayout(toggle_layout)

        splitter = QSplitter(Qt.Horizontal)

        graph_container = QWidget()
        graph_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        graph_layout = QVBoxLayout(graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        self.webview = QWebEngineView()
        self.webview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.webview.setContextMenuPolicy(Qt.NoContextMenu)
        graph_layout.addWidget(self.webview)
        splitter.addWidget(graph_container)

        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(5, 5, 5, 5)
        advanced_layout.setSpacing(10)

        # Node Operations Group
        node_ops_box = QGroupBox("Node Operations")
        node_ops_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        node_ops_layout = QVBoxLayout()
        node_ops_layout.setSpacing(10)

        # -- Add Node Operation --
        add_node_box = QGroupBox("Add Node")
        add_node_box.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 10pt;}")
        add_node_layout = QHBoxLayout()
        self.add_node_field = QLineEdit()
        self.add_node_field.setPlaceholderText("New Node Name")
        self.add_node_btn = QPushButton("Add Node")
        self.add_node_btn.clicked.connect(self.add_new_node)
        add_node_layout.addWidget(self.add_node_field)
        add_node_layout.addWidget(self.add_node_btn)
        add_node_box.setLayout(add_node_layout)
        node_ops_layout.addWidget(add_node_box)

        # -- Merge Operations (using a QFormLayout) --
        merge_box = QGroupBox("Merge Nodes")
        merge_box.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 10pt;}")
        merge_layout = QFormLayout()
        merge_layout.setSpacing(5)
        merge_keep_layout = QHBoxLayout()
        pick_merge_keep_btn = QPushButton("Pick")
        pick_merge_keep_btn.clicked.connect(
            lambda: self.set_pick_mode("node", "keep"))
        self.merge_keep_field = QLineEdit()
        self.merge_keep_field.setPlaceholderText("Keep Node")
        merge_keep_layout.addWidget(pick_merge_keep_btn)
        merge_keep_layout.addWidget(self.merge_keep_field)
        merge_layout.addRow("Keep Node:", merge_keep_layout)

        merge_merge_layout = QHBoxLayout()
        pick_merge_merge_btn = QPushButton("Pick")
        pick_merge_merge_btn.clicked.connect(
            lambda: self.set_pick_mode("node", "merge"))
        self.merge_merge_field = QLineEdit()
        self.merge_merge_field.setPlaceholderText("Node to Merge")
        merge_merge_layout.addWidget(pick_merge_merge_btn)
        merge_merge_layout.addWidget(self.merge_merge_field)
        merge_layout.addRow("Merge Node:", merge_merge_layout)

        self.merge_btn = QPushButton("Merge Nodes")
        self.merge_btn.clicked.connect(self.merge_nodes)
        merge_layout.addRow(self.merge_btn)
        merge_box.setLayout(merge_layout)
        node_ops_layout.addWidget(merge_box)

        # -- Rename Node Operation --
        rename_box = QGroupBox("Rename Node")
        rename_box.setStyleSheet(
            "QGroupBox { font-weight: bold; font-size: 10pt;}")
        rename_layout = QFormLayout()
        rename_layout.setSpacing(5)

        rename_old_layout = QHBoxLayout()
        pick_rename_old_btn = QPushButton("Pick")
        pick_rename_old_btn.clicked.connect(
            lambda: self.set_pick_mode("node", "old"))
        self.rename_old_field = QLineEdit()
        self.rename_old_field.setPlaceholderText("Old Name")
        rename_old_layout.addWidget(pick_rename_old_btn)
        rename_old_layout.addWidget(self.rename_old_field)
        rename_layout.addRow("Old Name:", rename_old_layout)

        self.rename_new_field = QLineEdit()
        self.rename_new_field.setPlaceholderText("New Name")
        rename_layout.addRow("New Name:", self.rename_new_field)
        self.rename_btn = QPushButton("Rename Node")
        self.rename_btn.clicked.connect(self.rename_node_btn)
        rename_layout.addRow(self.rename_btn)
        rename_box.setLayout(rename_layout)
        node_ops_layout.addWidget(rename_box)

        node_ops_box.setLayout(node_ops_layout)
        advanced_layout.addWidget(node_ops_box)

        # Relationship Operations Group
        rel_ops_box = QGroupBox("Relationship Operations")
        rel_ops_box.setStyleSheet("QGroupBox {font-weight: bold; font-size: "
                                  "12pt;}")
        rel_ops_layout = QVBoxLayout()
        rel_ops_layout.setSpacing(10)

        # Add Relationship Operation
        add_rel_box = QGroupBox("Add Relationship")
        add_rel_box.setStyleSheet("QGroupBox {font-weight: bold; font-size: "
                                  "10pt;}")
        add_rel_form = QFormLayout()
        add_rel_form.setSpacing(5)

        subject_layout = QHBoxLayout()
        pick_add_subj_btn = QPushButton("Pick")
        pick_add_subj_btn.clicked.connect(
            lambda: self.set_pick_mode("add", "subject"))
        self.rel_subject_field = QLineEdit()
        self.rel_subject_field.setPlaceholderText("Subject node")
        subject_layout.addWidget(pick_add_subj_btn)
        subject_layout.addWidget(self.rel_subject_field)
        add_rel_form.addRow("Subject:", subject_layout)

        relation_layout = QHBoxLayout()
        pick_add_rel_btn = QPushButton("Pick")
        pick_add_rel_btn.clicked.connect(
            lambda: self.set_pick_mode("add", "relation"))
        self.rel_relation_field = QLineEdit()
        self.rel_relation_field.setPlaceholderText("Relation")
        relation_layout.addWidget(pick_add_rel_btn)
        relation_layout.addWidget(self.rel_relation_field)
        add_rel_form.addRow("Relation:", relation_layout)

        object_layout = QHBoxLayout()
        pick_add_obj_btn = QPushButton("Pick")
        pick_add_obj_btn.clicked.connect(
            lambda: self.set_pick_mode("add", "object"))
        self.rel_object_field = QLineEdit()
        self.rel_object_field.setPlaceholderText("Object node")
        object_layout.addWidget(pick_add_obj_btn)
        object_layout.addWidget(self.rel_object_field)
        add_rel_form.addRow("Object:", object_layout)

        self.add_rel_btn = QPushButton("Add Relationship")
        self.add_rel_btn.clicked.connect(self.add_relationship)
        add_rel_form.addRow(self.add_rel_btn)
        add_rel_box.setLayout(add_rel_form)
        rel_ops_layout.addWidget(add_rel_box)

        # Edit Relationship Group within Relationship Operations
        edit_rel_box = QGroupBox("Edit Relationship")
        edit_rel_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 10pt;}")
        edit_rel_form = QFormLayout()
        edit_rel_form.setSpacing(5)

        edit_subj_layout = QHBoxLayout()
        pick_edit_subj_btn = QPushButton("Pick")
        pick_edit_subj_btn.clicked.connect(
            lambda: self.set_pick_mode("edit", "subject"))
        self.edit_rel_subject_field = QLineEdit()
        self.edit_rel_subject_field.setPlaceholderText("Subject node")
        edit_subj_layout.addWidget(pick_edit_subj_btn)
        edit_subj_layout.addWidget(self.edit_rel_subject_field)
        edit_rel_form.addRow("Subject:", edit_subj_layout)

        edit_old_layout = QHBoxLayout()
        pick_edit_old_btn = QPushButton("Pick")
        pick_edit_old_btn.clicked.connect(
            lambda: self.set_pick_mode("edit", "old_relation"))
        self.edit_rel_old_field = QLineEdit()
        self.edit_rel_old_field.setPlaceholderText("Old Relation")
        edit_old_layout.addWidget(pick_edit_old_btn)
        edit_old_layout.addWidget(self.edit_rel_old_field)
        edit_rel_form.addRow("Old Relation:", edit_old_layout)

        edit_new_layout = QHBoxLayout()
        pick_edit_new_btn = QPushButton("Pick")
        pick_edit_new_btn.clicked.connect(
            lambda: self.set_pick_mode("edit", "new_relation"))
        self.edit_rel_new_field = QLineEdit()
        self.edit_rel_new_field.setPlaceholderText("New Relation")
        edit_new_layout.addWidget(pick_edit_new_btn)
        edit_new_layout.addWidget(self.edit_rel_new_field)
        edit_rel_form.addRow("New Relation:", edit_new_layout)

        edit_obj_layout = QHBoxLayout()
        pick_edit_obj_btn = QPushButton("Pick")
        pick_edit_obj_btn.clicked.connect(
            lambda: self.set_pick_mode("edit", "object"))
        self.edit_rel_object_field = QLineEdit()
        self.edit_rel_object_field.setPlaceholderText("Object node")
        edit_obj_layout.addWidget(pick_edit_obj_btn)
        edit_obj_layout.addWidget(self.edit_rel_object_field)
        edit_rel_form.addRow("Object:", edit_obj_layout)

        edit_rel_btn = QPushButton("Edit Relationship")
        edit_rel_btn.clicked.connect(self.edit_relationship)
        edit_rel_form.addRow(edit_rel_btn)
        edit_rel_box.setLayout(edit_rel_form)
        rel_ops_layout.addWidget(edit_rel_box)

        rel_ops_box.setLayout(rel_ops_layout)
        advanced_layout.addWidget(rel_ops_box)

        advanced_layout.addStretch()

        advanced_controls_widget = QWidget()
        advanced_controls_widget.setLayout(advanced_layout)

        self.advanced_container = QWidget()
        advanced_container_layout = QVBoxLayout(self.advanced_container)
        advanced_container_layout.setContentsMargins(0, 0, 0, 0)
        advanced_container_layout.setSpacing(0)

        # Wrap the advanced controls content in a QScrollArea to provide vertical scrolling.
        self.advanced_scroll_area = QScrollArea()
        self.advanced_scroll_area.setWidgetResizable(True)
        self.advanced_scroll_area.setWidget(advanced_controls_widget)
        advanced_container_layout.addWidget(self.advanced_scroll_area)
        # Create a widget to serve as the scrollable container for the advanced controls.
        advanced_content = QWidget()
        # Using your existing advanced_layout.
        advanced_content.setLayout(advanced_layout)
        self.advanced_scroll_area.setWidget(advanced_content)
        # self.advanced_scroll_area.viewport().setStyleSheet("background-color: #c9c9c9;")
        advanced_container_layout.addWidget(self.advanced_scroll_area)

        # Add this container to the splitter instead of the original advanced_controls_widget.
        splitter.addWidget(self.advanced_container)

        # splitter.setStretchFactor(0, 3)
        # splitter.setStretchFactor(1, 1)
        splitter.setSizes([750, 250])
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(splitter)

        self.splitter = splitter
        self.savedAdvWidth = None

        self.bridge = GraphEditorBridge(self)
        self.channel = QWebChannel(self.webview.page())
        self.channel.registerObject("pyBridge", self.bridge)
        self.webview.page().setWebChannel(self.channel)

        self.setLayout(main_layout)

        self.escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.escape_shortcut.activated.connect(self.cancel_pick_mode)

        self.api_config = {}
        self.load_api_config()

    def load_api_config(self):
        try:
            with open("config.yaml", "r") as f:
                self.api_config = yaml.safe_load(f) or {}
                
                # Load saved model selections if they exist
                if "model_selections" in self.api_config:
                    saved_llm = self.api_config["model_selections"].get("llm_model")
                    saved_embedding = self.api_config["model_selections"].get("embedding_model")
                    
                    if saved_llm:
                        self.set_llm_model(saved_llm)
                    if saved_embedding:
                        self.set_embedding_model(saved_embedding)
        except FileNotFoundError:
            self.api_config = {}

    def open_api_config(self):
        dlg = ApiConfigDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.load_api_config()

    def toggle_advanced_controls(self):
        totalWidth = self.splitter.width()
        minimalAdvWidth = 0  # when hidden, give 0 width to advanced controls.
        # Get current sizes; assume sizes[0] is graph and sizes[1] is advanced.
        sizes = self.splitter.sizes()

        # If the advanced container is visible, collapse it
        if self.advanced_container.isVisible():
            # Save the current advanced width (if greater than zero).
            if sizes[1] > 0:
                self.savedAdvWidth = sizes[1]
            # Hide the advanced container; it will now vanish completely.
            self.advanced_container.hide()
            self.splitter.setSizes([totalWidth, minimalAdvWidth])
        else:
            # Restore the advanced container.
            self.advanced_container.show()
            # Use the saved width if available; otherwise, use a default.
            restoredWidth = self.savedAdvWidth if self.savedAdvWidth is not None else 250
            self.splitter.setSizes([totalWidth - restoredWidth, restoredWidth])

    def animate_advanced_container_width(self, start, end):
        # Animate the maximumWidth property of the container.
        self.width_animation = QPropertyAnimation(self.advanced_container,
                                                  b"maximumWidth")
        self.width_animation.setDuration(
            200)  # Duration in milliseconds (adjust as preferred)
        self.width_animation.setStartValue(start)
        self.width_animation.setEndValue(end)
        self.width_animation.start()

    def open_api_config(self):
        dlg = ApiConfigDialog(self)
        dlg.exec_()

    def save_state(self):
        state = {
            'nodes': copy.deepcopy(self.nodes),
            'triples': copy.deepcopy(self.triples)
        }
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo_action(self):
        self.cancel_pick_mode()
        if not self.undo_stack:
            QMessageBox.information(self, "Undo", "No actions to undo.")
            return
        state = self.undo_stack.pop()
        self.nodes = state['nodes']
        self.triples = state['triples']
        self.build_graph()  # Rebuild the graph from the restored state

    def set_filter_contents_visible(self, visible):
        layout = self.filter_box.layout()
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(visible)

    def set_pick_mode(self, section, field, sender_btn=None):
        if sender_btn is None:
            sender_btn = self.sender()
        # If a different mode is already active, uncheck that button.
        print(f"pick mode settings: section: {section}, field: {field}, {sender_btn}")
        if self.currentFieldSelection is not None:
            current_mode, _ = self.currentFieldSelection
            if current_mode != section and hasattr(self, 'active_mode_btn'):
                self.active_mode_btn.setChecked(False)
        # If the same mode is active, toggle it off.
        if self.currentFieldSelection == (section, field):
            self.cancel_pick_mode()
            print(f"Pick mode for {section} - {field} canceled.")
            return
        else:
            self.currentFieldSelection = (section, field)
            self.active_mode_btn = sender_btn  # store the currently active button
            print(f"Pick mode set for {section} - {field}.")

    def toggle_reverse_mode(self):
        if self.reverse_rel_btn.isChecked():
            self.set_pick_mode("relationship", "reverse")
            print("Reverse mode activated.")
        else:
            self.currentFieldSelection = None
            print("Reverse mode deactivated.")

    def search_node(self):
        # Cancel any active mode before starting search.
        self.cancel_pick_mode()

        search_text = self.search_field.text().strip()
        if not search_text:
            QMessageBox.warning(self, "Warning",
                                "Please enter a node name to search.")
            return

        # Check if the search text is the same as last time.
        # (You could add an instance variable self.last_search_text if desired.)
        # For simplicity, here we'll always refresh the search:
        import json
        matching_nodes = [node for node in self.nodes if
                          search_text.lower() in node.lower()]

        if not matching_nodes:
            QMessageBox.warning(self, "Not Found",
                                f"No nodes matching '{search_text}' found in the graph.")
            self.current_search_results = []
            self.current_search_index = 0
            self.search_status_label.setText("")
            return

        # If new search text, restart the index; or if same, cycle.
        if (not hasattr(self, 'last_search_text')) or (
                self.last_search_text != search_text):
            self.current_search_results = matching_nodes
            self.current_search_index = 0
            self.last_search_text = search_text
        else:
            # Cycle to the next result
            self.current_search_index = (self.current_search_index + 1) % len(
                self.current_search_results)

        # Get the current match
        current_match = self.current_search_results[self.current_search_index]

        # Update the search status label, e.g. "2 of 5"
        self.search_status_label.setText(
            f"{self.current_search_index + 1} of {len(self.current_search_results)}")

        # Optionally, clear previous highlights by resetting styles of all nodes.
        # Then, update the current node's style.
        # Here we assume the vis.js node data accepts a 'color' property.
        # To highlight, we update its background color to, e.g., red.
        js_code = f"""
            // Clear previous highlight:
            var allNodes = network.body.data.nodes.get();
            allNodes.forEach(function(node) {{
                // Reset the background; adjust this as needed.
                network.body.data.nodes.update({{id: node.id, color: {{background: 'lightblue'}}}});
            }});
            // Highlight the current match:
            network.body.data.nodes.update({{
                id: "{current_match}",
                color: {{
                    background: "#ffeb3b", 
                    border: "#f57c00"
                }}
            }});
            // Optionally, center on the highlighted node:
            network.focus("{current_match}", {{ scale: 1.5 }});
        """
        self.webview.page().runJavaScript(js_code)

    def on_node_selected(self, node_id):
        print("on_node_selected:", node_id)
        if self.currentFieldSelection is not None:
            mode, field = self.currentFieldSelection
            if mode == "delete_node":
                self.delete_node_immediate(node_id)
                return
            elif mode == "relationship" and field == "reverse":
                return
            elif mode in ["filter_inclusive", "filter_inclusive_multi"]:
                current_text = self.inclusive_filter_field.text().strip()
                if current_text:
                    if not current_text.endswith(";"):
                        current_text += ";"
                    new_text = current_text + node_id
                else:
                    new_text = node_id
                self.inclusive_filter_field.setText(new_text)
                if mode == "filter_inclusive":
                    self.currentFieldSelection = None
            elif mode in ["filter_exclusive", "filter_exclusive_multi"]:
                current_text = self.exclusive_filter_field.text().strip()
                if current_text:
                    if not current_text.endswith(";"):
                        current_text += ";"
                    new_text = current_text + node_id
                else:
                    new_text = node_id
                self.exclusive_filter_field.setText(new_text)
                if mode == "filter_exclusive":
                    self.currentFieldSelection = None
            elif mode == "node":
                # Handle node operations: delete, keep, merge, rename old
                if field == "delete":
                    self.del_node_field.setText(node_id)
                elif field == "keep":
                    self.merge_keep_field.setText(node_id)
                elif field == "merge":
                    self.merge_merge_field.setText(node_id)
                elif field == "old":
                    self.rename_old_field.setText(node_id)
                self.currentFieldSelection = None
            elif mode in ["add", "edit", "delete"]:
                if field in ["subject", "object"]:
                    if mode == "add":
                        if field == "subject":
                            self.rel_subject_field.setText(node_id)
                        else:
                            self.rel_object_field.setText(node_id)
                    elif mode == "delete":
                        if field == "subject":
                            self.del_rel_subj_field.setText(node_id)
                        else:
                            self.del_rel_object_field.setText(node_id)
                    elif mode == "edit":
                        if field == "subject":
                            self.edit_rel_subject_field.setText(node_id)
                        else:
                            self.edit_rel_object_field.setText(node_id)
                    self.currentFieldSelection = None

    def on_edge_selected(self, edge_id):
        print("on_edge_selected:", edge_id)
        try:
            # Check if reverse mode is active.
            if self.currentFieldSelection is not None:
                subj, rel, obj_val = edge_id.split("|")
                section, field = self.currentFieldSelection
                if section == "delete_relationship":
                    self.delete_relationship_immediate(edge_id)
                    return
                elif section == "reverse_relationship":
                    self.reverse_relationship(edge_id)
                    # Remain in reverse mode; do not clear the current selection.
                    return
                elif field in ["relation", "old_relation", "new_relation"]:
                    if section == "add" and field == "relation":
                        self.rel_relation_field.setText(rel)
                    elif section == "delete" and field == "relation":
                        self.del_rel_relation_field.setText(rel)
                    elif section == "edit":
                        if field == "old_relation":
                            self.edit_rel_old_field.setText(rel)
                        elif field == "new_relation":
                            self.edit_rel_new_field.setText(rel)
                    self.currentFieldSelection = None
        except Exception as e:
            print("Error parsing edge_id:", e)

    def reverse_relationship(self, edge_id):
        try:
            # Assume the edge identifier is in the format "A|relation|B".
            original_subject, relation, original_object = edge_id.split("|")
            new_from = None
            new_to = None
            # Look for the corresponding relationship in the stored list,
            # matching either the original order or the reversed order.
            for triple in self.triples:
                if (triple.get("subject") == original_subject and triple.get(
                        "object") == original_object) or \
                        (triple.get(
                            "subject") == original_object and triple.get(
                            "object") == original_subject):
                    self.save_state()
                    # Swap the values so that each click toggles the orientation.
                    triple["subject"], triple["object"] = triple["object"], \
                    triple["subject"]
                    new_from = triple["subject"]
                    new_to = triple["object"]
                    print(f"Reversed relationship for edge: {edge_id}")
                    break
            if new_from is not None and new_to is not None:
                # Option A: Update the existing edge by using the DataSet's update method.
                # The edge identifier remains unchanged.
                js_code = f"""
                    network.body.data.edges.update({{
                        id: "{edge_id}",
                        from: "{new_from}",
                        to: "{new_to}"
                    }});
                """
                self.webview.page().runJavaScript(js_code)
            else:
                print("No matching triple found for edge:", edge_id)
        except Exception as e:
            print("Error reversing relationship:", e)

    def apply_inclusive_filter(self):
        self.cancel_pick_mode()
        filter_text = self.inclusive_filter_field.text().strip()
        if not filter_text:
            return
        nodes_list = [n.strip() for n in filter_text.split(";") if n.strip()]
        import json
        js_code = f"""
        (function(){{
            var filterNodes = {json.dumps(nodes_list)};
            var allNodes = network.body.data.nodes.get();
            var allEdges = network.body.data.edges.get();
            var filteredEdges = [];
            var filteredNodeIds = new Set();
            for(var i = 0; i < allEdges.length; i++){{
                var edge = allEdges[i];
                if(filterNodes.indexOf(edge.from) !== -1 || filterNodes.indexOf(edge.to) !== -1){{
                    filteredEdges.push(edge);
                    filteredNodeIds.add(edge.from);
                    filteredNodeIds.add(edge.to);
                }}
            }}
            var filteredNodes = [];
            for(var i = 0; i < allNodes.length; i++){{
                var node = allNodes[i];
                if(filteredNodeIds.has(node.id)){{
                    filteredNodes.push(node);
                }}
            }}
            network.setData({{
                nodes: new vis.DataSet(filteredNodes),
                edges: new vis.DataSet(filteredEdges)
            }});
            network.stabilize();
        }})();
        """
        self.webview.page().runJavaScript(js_code)

    def apply_exclusive_filter(self):
        self.cancel_pick_mode()
        filter_text = self.exclusive_filter_field.text().strip()
        if not filter_text:
            return
        nodes_list = [n.strip() for n in filter_text.split(";") if n.strip()]
        import json
        js_code = f"""
        (function(){{
            var filterNodes = {json.dumps(nodes_list)};
            var allNodes = network.body.data.nodes.get();
            var allEdges = network.body.data.edges.get();
            var filteredEdges = [];
            for(var i = 0; i < allEdges.length; i++){{
                var edge = allEdges[i];
                if(filterNodes.indexOf(edge.from) !== -1 && filterNodes.indexOf(edge.to) !== -1){{
                    filteredEdges.push(edge);
                }}
            }}
            var filteredNodes = [];
            for(var i = 0; i < allNodes.length; i++){{
                var node = allNodes[i];
                if(filterNodes.indexOf(node.id) !== -1){{
                    filteredNodes.push(node);
                }}
            }}
            network.setData({{
                nodes: new vis.DataSet(filteredNodes),
                edges: new vis.DataSet(filteredEdges)
            }});
            network.stabilize();
        }})();
        """
        self.webview.page().runJavaScript(js_code)

    def import_graph_from_csv(self):
        self.cancel_pick_mode()
        directory = QFileDialog.getExistingDirectory(self, "Select directory containing nodes.csv and edges.csv", "")
        if not directory:
            return
        nodes_path = os.path.join(directory, "nodes.csv")
        edges_path = os.path.join(directory, "edges.csv")
        if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
            QMessageBox.warning(self, "Error", "nodes.csv or edges.csv not found in the selected directory.")
            return
        self.nodes = []
        self.triples = []
        try:
            with open(nodes_path, "r", encoding="utf-8") as nf:
                reader = csv.DictReader(nf)
                for row in reader:
                    node_id = row.get("id")
                    if node_id and node_id not in self.nodes:
                        self.nodes.append(node_id)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to parse nodes.csv: {e}")
            return
        try:
            with open(edges_path, "r", encoding="utf-8") as ef:
                reader = csv.DictReader(ef)
                for row in reader:
                    subj = row.get("source")
                    rel = row.get("relation")
                    obj = row.get("target")
                    if subj and subj not in self.nodes:
                        self.nodes.append(subj)
                    if obj and obj not in self.nodes:
                        self.nodes.append(obj)
                    if subj and rel and obj:
                        self.triples.append({"subject": subj, "relation": rel, "object": obj})
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to parse edges.csv: {e}")
            return
        self.build_graph()
        QMessageBox.information(self, "CSV Loaded", "Graph loaded from CSV successfully.")

    def cancel_pick_mode(self):
        self.currentFieldSelection = None
        # Uncheck all toggle mode buttons.
        for btn in [self.delete_node_mode_btn,
                    self.delete_relationship_mode_btn,
                    self.reverse_relationship_mode_btn]:
            btn.setChecked(False)
        self.active_mode_btn = None
        # Reset node colors back to default.
        # Adjust these values to match your default node styling.
        js_reset = """
            var defaultColor = {background: '#97C2FC', border: '#2B7CE9'};
            var allNodes = network.body.data.nodes.get();
            allNodes.forEach(function(node) {
                network.body.data.nodes.update({ id: node.id, color: defaultColor });
            });
            """
        self.webview.page().runJavaScript(js_reset);

    def clear_filters(self):
        self.cancel_pick_mode()
        self.inclusive_filter_field.clear()
        self.exclusive_filter_field.clear()
        js_code = """
        (function(){
            var allNodes = network.body.data.nodes.get();
            var allEdges = network.body.data.edges.get();
            network.setData({
                nodes: new vis.DataSet(allNodes),
                edges: new vis.DataSet(allEdges)
            });
            network.stabilize();
        })();
        """
        self.webview.page().runJavaScript(js_code)
        self.build_graph()

    def set_llm_model(self, model_name):
        # 1) clear and re-check the exact model
        for prov_action in self.llm_menu.actions():
            submenu = prov_action.menu()
            if submenu:
                any_checked = False
                for act in submenu.actions():
                    checked = (act.text() == model_name)
                    act.setChecked(checked)
                    if checked:
                        any_checked = True
                # now toggle the provider itself
                prov_action.setChecked(any_checked)

        self.current_llm_model = model_name

        # 2) elide long text to fit MAX_LABEL_WIDTH
        full = f"LLM: {model_name}"
        fm = QFontMetrics(self.llm_btn.font())
        elided = fm.elidedText(full, Qt.ElideRight, MAX_LABEL_WIDTH)
        self.llm_btn.setText(elided)
        self.llm_btn.setToolTip(full)

        # 3) Save to config
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f) or {}
            if "model_selections" not in config:
                config["model_selections"] = {}
            config["model_selections"]["llm_model"] = model_name
            with open("config.yaml", "w") as f:
                yaml.safe_dump(config, f)
        except Exception as e:
            print(f"Failed to save LLM model selection: {e}")

    def set_embedding_model(self, model_name):
        for prov_action in self.embed_menu.actions():
            submenu = prov_action.menu()
            if submenu:
                any_checked = False
                for act in submenu.actions():
                    checked = (act.text() == model_name)
                    act.setChecked(checked)
                    if checked:
                        any_checked = True
                prov_action.setChecked(any_checked)

        self.current_embedding_model = model_name

        full = f"Encoder: {model_name}"
        fm = QFontMetrics(self.embed_btn.font())
        elided = fm.elidedText(full, Qt.ElideRight, MAX_LABEL_WIDTH)
        self.embed_btn.setText(elided)
        self.embed_btn.setToolTip(full)

        # Save to config
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f) or {}
            if "model_selections" not in config:
                config["model_selections"] = {}
            config["model_selections"]["embedding_model"] = model_name
            with open("config.yaml", "w") as f:
                yaml.safe_dump(config, f)
        except Exception as e:
            print(f"Failed to save embedding model selection: {e}")

    def _process_entries(self, entries: list[dict]) -> list[dict]:
        """
            1) Merge adjacent segments from the same speaker
            2) Recursively split any merged block > max_words words:
               a) If >1 sentence, find the sentence boundary closest to half the words and split there
               b) If exactly 1 sentence, split that sentence in half at the word level
            Returns a flat list of {"speaker": ..., "text": ...} dicts, each <= max_words words.
            """
        # 1) Merge
        segments: list[tuple[str, str]] = []

        for seg in entries:
            spk = seg.get("speaker", "")
            txt = seg.get("text", "").strip()
            if not segments or segments[-1][0] != spk:
                segments.append([spk, txt])
            else:
                segments[-1][1] += " " + txt

        # 2) Split each merged block
        out: list[dict] = []
        for spk, text in segments:
            out.extend(self._split_block(spk, text))
        return out

    def _split_block(self, spk: str, text: str) -> list[dict]:
        """
        Recursively split `text` into chunks of ≤ MAX_WORDS words,
        preserving sentence boundaries when possible, otherwise
        bisecting the single too-long sentence in half by word count.
        Returns a list of {"speaker": spk, "text": chunk} dicts.
        """
        words = text.split()
        # Base case: already small enough
        if len(words) <= MAX_WORDS:
            return [{"speaker": spk, "text": text.strip()}]

        # Try to split on sentence boundaries first
        sentences = re.split(r'(?<=[\.?!])\s+', text)
        if len(sentences) > 1:
            # Compute word counts per sentence
            counts = [len(s.split()) for s in sentences]
            # Build prefix‐sum array
            prefix = [0]
            for c in counts:
                prefix.append(prefix[-1] + c)
            total = prefix[-1]
            half = total / 2

            # Find the split index where prefix >= half
            i0 = bisect.bisect_left(prefix, half)
            # Clamp to valid sentence‐boundary splits
            best_i = min(max(i0, 1), len(sentences) - 1)
            best_diff = abs(half - prefix[best_i])
            # Check neighbors for a better balance
            for cand in (i0 - 1, i0 + 1):
                if 1 <= cand < len(prefix):
                    diff = abs(half - prefix[cand])
                    if diff < best_diff:
                        best_i, best_diff = cand, diff

            # Recurse on the two halves
            left_text  = " ".join(sentences[:best_i])
            right_text = " ".join(sentences[best_i:])
            return (
                self._split_block(spk, left_text)
                + self._split_block(spk, right_text)
            )

        # Fallback: single sentence that's still too long → split in half by word count
        n = len(words)
        mid = n // 2
        left_text  = " ".join(words[:mid])
        right_text = " ".join(words[mid:])
        return (
            self._split_block(spk, left_text)
            + self._split_block(spk, right_text)
        )

    def generate_graph(self):
        self.cancel_pick_mode()

        # 1) Ensure models have been picked
        if not getattr(self, "current_llm_model", None):
            QMessageBox.warning(self, "Error", "Please select an LLM model.")
            return
        if not getattr(self, "current_embedding_model", None):
            QMessageBox.warning(self, "Error", "Please select an embedding model.")
            return

        # 2) Extract keys from self.api_config
        cfg = self.api_config

        # --- LLM keys ---
        parent = utils.determine_llm_parent(self.current_llm_model)
        if parent == "bedrock_llm":
            sec = cfg.get("bedrock", {})
            llm_keys = [
                sec.get("AWS_ACCESS_KEY_ID", "").strip(),
                sec.get("AWS_SECRET_ACCESS_KEY", "").strip(),
                sec.get("AWS_REGION_NAME", "").strip()
            ]
            if any(not k for k in llm_keys):
                QMessageBox.warning(self, "Error", "Please fill all three Bedrock credentials.")
                return
            llm_key1, llm_key2, llm_key3 = llm_keys

        elif parent == "ollama_llm":
            base = cfg.get("ollama", {}).get("api_base", "").strip()
            if not base:
                QMessageBox.warning(self, "Error", "Please specify Ollama API Base URL.")
                return
            llm_key1, llm_key2, llm_key3 = base, "", ""

        else:
            prov = parent.replace("_llm", "")
            key = cfg.get(prov, {}).get("api_key", "").strip()
            if not key:
                QMessageBox.warning(self, "Error", f"Please enter API key for {prov.title()}.")
                return
            llm_key1, llm_key2, llm_key3 = key, "", ""

        # --- Embedding keys ---
        parent_e = utils.determine_embedding_parent(self.current_embedding_model)
        if parent_e == "bedrock_embedding":
            sec = cfg.get("bedrock", {})
            emb_keys = [
                sec.get("AWS_ACCESS_KEY_ID", "").strip(),
                sec.get("AWS_SECRET_ACCESS_KEY", "").strip(),
                sec.get("AWS_REGION_NAME", "").strip()
            ]
            if any(not k for k in emb_keys):
                QMessageBox.warning(self, "Error", "Please fill all three Bedrock embedding credentials.")
                return
            emb_key1, emb_key2, emb_key3 = emb_keys

        elif parent_e == "local_embedding":
            emb_key1, emb_key2, emb_key3 = "", "", ""

        else:
            prov = parent_e.replace("_embedding", "")
            section = f"{prov}_embedding"
            key = cfg.get(section, {}).get("api_key", "").strip()
            if not key:
                QMessageBox.warning(self, "Error", f"Please enter embedding API key for {prov.title()}.")
                return
            emb_key1, emb_key2, emb_key3 = key, "", ""

        # 3) rest of your existing checks (transcript, main_topic, etc.)
        raw = self.transcript_provider()  # now gives List[dict]
        if not raw:
            QMessageBox.warning(self, "Error", "Transcript not loaded.")
            return

        # apply our new segmentation
        processed = self._process_entries(raw)
        texts = [e["text"] for e in processed]
        sources = [e["speaker"] for e in processed]
        main_topic_text = self.main_topic_field.text().strip()

        # Get clustering parameters from stored values instead of spinboxes
        n_clusters = self.n_clusters
        num_topics = self.num_topics

        # 4) show progress, kick off thread with processed_text…
        progress = QProgressDialog("Generating knowledge graph…", None, 0, 0,
                                   self)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()

        self.graph_worker = GraphGenerationThread(
            texts=texts,
            sources=sources,
            llm_model=self.current_llm_model,
            embedding_model=self.current_embedding_model,
            llm_key1=llm_key1, llm_key2=llm_key2, llm_key3=llm_key3,
            emb_key1=emb_key1, emb_key2=emb_key2, emb_key3=emb_key3,
            main_topic=main_topic_text,
            n_clusters=n_clusters,
            num_topics=num_topics,
            dynamic_threshold=self.dynamic_threshold,
            parent=self,
        )
        self.graph_worker.finished.connect(
            self.handle_graph_generation_finished)
        self.graph_worker.errorOccurred.connect(
            lambda msg: QMessageBox.warning(self, "Error", msg))
        self.graph_worker.finished.connect(progress.cancel)
        self.graph_worker.errorOccurred.connect(progress.cancel)
        self.graph_worker.start()

    def handle_graph_generation_finished(self, nodes, triples):
        self.nodes = nodes
        self.triples = triples
        self.build_graph()

    def build_graph(self):
        self.cancel_pick_mode()
        print(f"Building graph with {len(self.nodes)} nodes and {len(self.triples)} triples.")
        self.pyvis_net = Network(height="100vh", width="100%", directed=True)
        for node in self.nodes:
            node_str = str(node)
            self.pyvis_net.add_node(node_str, label=node_str)
        added_edge_ids = set()
        for triple in self.triples:
            subj = triple.get("subject")
            rel = triple.get("relation")
            obj = triple.get("object")
            if subj and obj and rel:
                if isinstance(obj, list):
                    for item in obj:
                        edge_id = f"{subj}|{rel}|{item}"
                        if edge_id not in added_edge_ids:
                            self.pyvis_net.add_edge(subj, item, label=rel, id=edge_id)
                            added_edge_ids.add(edge_id)
                else:
                    edge_id = f"{subj}|{rel}|{obj}"
                    if edge_id not in added_edge_ids:
                        self.pyvis_net.add_edge(subj, obj, label=rel, id=edge_id)
                        added_edge_ids.add(edge_id)

        self.pyvis_net.set_options("""
        {
          "nodes": {
            "color": {
              "background": "#97C2FC",
              "border": "#2B7CE9",
              "highlight": { "background": "#28d75c", "border": "#28d75c" },
              "hover": { "background": "#28d75c", "border": "#28d75c" }
            }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.1
            },
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 25
            }
          }
        }
        """)
        html_file = os.path.abspath("knowledge_graph.html")
        self.pyvis_net.write_html(html_file, open_browser=False, notebook=False)
        self.inject_js(html_file)
        self.webview.load(QUrl.fromLocalFile(html_file))

    def inject_js(self, html_file):
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html = f.read()
            js_code = """
                    <script type="text/javascript" src="qrc:///qtwebchannel/qwebchannel.js"></script>
                    <script type="text/javascript">
                      // This function resets all nodes and edges to their default colors.
                      function clearHighlights() {
                        var defaultNodeColor = {
                          background: '#97C2FC',
                          border: '#2B7CE9',
                          highlight: { background: '#97C2FC', border: '#2B7CE9' },
                          hover: { background: '#97C2FC', border: '#2B7CE9' }
                        };
                        var defaultEdgeColor = {
                          color: '#848484',
                          highlight: '#848484',
                          hover: '#848484',
                          inherit: false
                        };
                        var allNodes = network.body.data.nodes.get();
                        for (var i = 0; i < allNodes.length; i++) {
                          network.body.data.nodes.update({
                            id: allNodes[i].id,
                            color: defaultNodeColor
                          });
                        }
                        var allEdges = network.body.data.edges.get();
                        for (var i = 0; i < allEdges.length; i++) {
                          network.body.data.edges.update({
                            id: allEdges[i].id,
                            color: defaultEdgeColor
                          });
                        }
                      }
                    
                      // Set up the web channel.
                      window.addEventListener("load", function() {
                        new QWebChannel(qt.webChannelTransport, function(channel) {
                          window.pyBridge = channel.objects.pyBridge;
                        });
                        // When the network stabilizes, disable physics.
                        network.once("stabilizationIterationsDone", function() {
                          network.setOptions({ physics: { enabled: false } });
                        });
                    
                        // Attach click listeners.
                        function attachListeners() {
                          if (typeof network !== "undefined") {
                            network.on("click", function(params) {
                              clearHighlights();
                              if (params.nodes.length > 0) {
                                var nodeId = params.nodes[0];
                                network.body.data.nodes.update({
                                  id: nodeId,
                                  color: {
                                    background: "#28d75c",
                                    border: "#28d75c",
                                    highlight: { background: "#28d75c", border: "#1c9c42" },
                                    hover: { background: "#28d75c", border: "#28d75c" }
                                  }
                                });
                                var allEdges = network.body.data.edges.get();
                                for (var i = 0; i < allEdges.length; i++) {
                                  var edge = allEdges[i];
                                  if (edge.from === nodeId || edge.to === nodeId) {
                                    network.body.data.edges.update({
                                      id: edge.id,
                                      color: {
                                        color: "#1c9c42",
                                        highlight: "#1c9c42",
                                        hover: "#1c9c42",
                                        inherit: false
                                      }
                                    });
                                  }
                                }
                                if (window.pyBridge) {
                                  window.pyBridge.nodeClicked(nodeId);
                                }
                              } else if (params.edges.length > 0) {
                                var edgeId = params.edges[0];
                                network.body.data.edges.update({
                                  id: edgeId,
                                  color: {
                                    color: "#28d75c",
                                    highlight: "#28d75c",
                                    hover: "#28d75c",
                                    inherit: false
                                  }
                                });
                                if (window.pyBridge) {
                                  window.pyBridge.edgeClicked(edgeId);
                                }
                              }
                            });
                    
                            // Add an event listener for when a node dragging ends.
                            network.on("dragEnd", function(params) {
                              if (params.nodes.length > 0) {
                                // If a node was dragged, clear previous highlights and update the node immediately.
                                clearHighlights();
                                var nodeId = params.nodes[0];
                                network.body.data.nodes.update({
                                  id: nodeId,
                                  color: {
                                    background: "#28d75c",
                                    border: "#28d75c",
                                    highlight: { background: "#28d75c", border: "#1c9c42" },
                                    hover: { background: "#28d75c", border: "#28d75c" }
                                  }
                                });
                                var allEdges = network.body.data.edges.get();
                                for (var i = 0; i < allEdges.length; i++) {
                                  var edge = allEdges[i];
                                  if (edge.from === nodeId || edge.to === nodeId) {
                                    network.body.data.edges.update({
                                      id: edge.id,
                                      color: {
                                        color: "#1c9c42",
                                        highlight: "#1c9c42",
                                        hover: "#1c9c42",
                                        inherit: false
                                      }
                                    });
                                  }
                                }
                              }
                            });
                    
                            // Listen for deselection events.
                            network.on("deselectNode", function(params) {
                              clearHighlights();
                            });
                            network.on("deselectEdge", function(params) {
                              clearHighlights();
                            });
                    
                            // Listen for Escape key press to clear highlights.
                            document.addEventListener("keydown", function(event) {
                              if (event.key === "Escape") {
                                clearHighlights();
                              }
                            });
                          } else {
                            setTimeout(attachListeners, 100);
                          }
                        }
                        attachListeners();
                      });
                    </script>
                    </body>
                    """
            new_html = html.replace("</body>", js_code)
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(new_html)
            print("Injected custom JavaScript for node/edge highlighting.")
        except Exception as e:
            print("Error injecting JS:", e)

    def export_graph(self):
        self.cancel_pick_mode()
        if not self.triples:
            QMessageBox.warning(self, "Error", "No graph data to export.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not directory:
            return
        connected_nodes = set()
        for triple in self.triples:
            subj = triple.get("subject")
            obj = triple.get("object")
            if subj:
                connected_nodes.add(subj)
            if isinstance(obj, list):
                for item in obj:
                    connected_nodes.add(item)
            elif obj:
                connected_nodes.add(obj)
        nodes = {node for node in self.nodes if node in connected_nodes}
        edges = []
        for triple in self.triples:
            subj = triple.get("subject")
            rel = triple.get("relation")
            obj = triple.get("object")
            if subj and obj and rel:
                edges.append({"source": subj, "relation": rel, "target": obj})
        nodes_file = os.path.join(directory, "nodes.csv")
        edges_file = os.path.join(directory, "edges.csv")
        try:
            with open(nodes_file, "w", newline='', encoding="utf-8") as nf:
                writer = csv.writer(nf)
                writer.writerow(["id", "label"])
                for node in sorted(nodes):
                    writer.writerow([node, node])
            with open(edges_file, "w", newline='', encoding="utf-8") as ef:
                writer = csv.writer(ef)
                writer.writerow(["source", "relation", "target"])
                for edge in edges:
                    subj = edge["source"]
                    rel = edge["relation"]
                    obj = edge["target"]
                    if isinstance(obj, list):
                        for item in obj:
                            writer.writerow([subj, rel, item])
                    else:
                        writer.writerow([subj, rel, obj])
            QMessageBox.information(self, "Exported",
                                    f"Graph exported as nodes.csv and edges.csv in {directory}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Export failed: {e}")

    def add_new_node(self):
        self.cancel_pick_mode()
        node_name = self.add_node_field.text().strip()
        if not node_name:
            QMessageBox.warning(self, "Error", "Node name cannot be empty.")
            return
        if node_name in self.nodes:
            QMessageBox.warning(self, "Duplicate", f"Node '{node_name}' already exists.")
            return
        self.save_state()
        self.nodes.append(node_name)
        self.build_graph()
        self.add_node_field.clear()

    def delete_node_immediate(self, node_id):
        if node_id not in self.nodes:
            QMessageBox.warning(self, "Error", f"Node '{node_id}' not found.")
            return
        self.save_state()
        self.nodes.remove(node_id)
        # Remove any relationships from internal state
        self.triples = [t for t in self.triples if
                        t.get("subject") != node_id and not (
                                    isinstance(t.get("object"), str) and t.get(
                                "object") == node_id)]
        # Remove the node from the vis.js network without reloading the whole graph.
        js_code = f'network.body.data.nodes.remove("{node_id}");'
        self.webview.page().runJavaScript(js_code)

    def delete_node(self, nodeId):
        if nodeId not in self.nodes:
            QMessageBox.warning(self, "Error", f"Node '{nodeId}' not found.")
            return
        self.save_state()
        self.nodes.remove(nodeId)
        self.triples = [
            t for t in self.triples
            if t.get("subject") != nodeId
            and (t.get("object") != nodeId if isinstance(t.get("object"), str) else True)
            and (nodeId not in t["object"] if isinstance(t.get("object"), list) else True)
        ]

    def add_relationship(self):
        subj = self.rel_subject_field.text().strip()
        rel = self.rel_relation_field.text().strip()
        obj = self.rel_object_field.text().strip()
        if not subj or not rel or not obj:
            QMessageBox.warning(self, "Error", "Subject, Relation, and Object cannot be empty.")
            return
        if subj not in self.nodes:
            create_subj = QMessageBox.question(
                self, "Create Subject?",
                f"Node '{subj}' does not exist. Create it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if create_subj == QMessageBox.Yes:
                self.nodes.append(subj)
            else:
                return
        if obj not in self.nodes:
            create_obj = QMessageBox.question(
                self, "Create Object?",
                f"Node '{obj}' does not exist. Create it?",
                QMessageBox.Yes | QMessageBox.No
            )
            if create_obj == QMessageBox.Yes:
                self.nodes.append(obj)
            else:
                return
        self.save_state()
        self.triples.append({"subject": subj, "relation": rel, "object": obj})
        self.build_graph()

    def delete_relationship_immediate(self, edge_id):
        self.save_state()
        # Remove from internal model:
        new_triples = []
        for t in self.triples:
            subj, rel, obj_val = edge_id.split("|")
            if t.get("subject") == subj and t.get("relation") == rel and t.get("object") == obj_val:
                # Skip this triple (i.e., delete it)
                continue
            new_triples.append(t)
        self.triples = new_triples
        # Remove the edge from the vis.js network directly.
        js_code = f'network.body.data.edges.remove("{edge_id}");'
        self.webview.page().runJavaScript(js_code)

    def delete_relationship(self):
        subj = self.del_rel_subj_field.text().strip()
        rel = self.del_rel_relation_field.text().strip()
        obj = self.del_rel_object_field.text().strip()
        if not subj or not rel or not obj:
            QMessageBox.warning(self, "Error", "Please fill in Subject, Relation, and Object.")
            return
        new_triples = []
        for t in self.triples:
            if t.get("subject") == subj and t.get("relation") == rel:
                current_obj = t.get("object")
                if isinstance(current_obj, list):
                    filtered_obj = [o for o in current_obj if o != obj]
                    if filtered_obj:
                        t["object"] = filtered_obj
                        new_triples.append(t)
                else:
                    if current_obj == obj:
                        continue
                    else:
                        new_triples.append(t)
            else:
                new_triples.append(t)
        self.save_state()
        self.triples = new_triples
        self.build_graph()

    def edit_relationship(self):
        self.cancel_pick_mode()
        subj = self.edit_rel_subject_field.text().strip()
        old_rel = self.edit_rel_old_field.text().strip()
        new_rel = self.edit_rel_new_field.text().strip()
        obj = self.edit_rel_object_field.text().strip()
        if not subj or not old_rel or not new_rel or not obj:
            QMessageBox.warning(self, "Error", "Please fill in all fields for editing.")
            return
        updated = False
        for t in self.triples:
            if t.get("subject") == subj and t.get("relation") == old_rel:
                current_obj = t.get("object")
                self.save_state()
                if isinstance(current_obj, list):
                    if obj in current_obj:
                        t["relation"] = new_rel
                        updated = True
                else:
                    if current_obj == obj:
                        t["relation"] = new_rel
                        updated = True
        if updated:
            QMessageBox.information(self, "Success", "Relationship updated.")
        else:
            QMessageBox.warning(self, "Not Found", "No matching relationship found.")
        self.build_graph()

    def merge_nodes(self):
        self.cancel_pick_mode()
        keep_node = self.merge_keep_field.text().strip()
        merge_node = self.merge_merge_field.text().strip()
        if not keep_node or not merge_node:
            QMessageBox.warning(self, "Error", "Please specify both Keep Node and Merge Node.")
            return
        if keep_node == merge_node:
            QMessageBox.warning(self, "Error", "Cannot merge a node into itself.")
            return
        if keep_node not in self.nodes:
            QMessageBox.warning(self, "Error", f"Keep Node '{keep_node}' not found.")
            return
        if merge_node not in self.nodes:
            QMessageBox.warning(self, "Error", f"Merge Node '{merge_node}' not found.")
            return
        self.save_state()
        for t in self.triples:
            if t["subject"] == merge_node:
                t["subject"] = keep_node
            obj = t["object"]
            if isinstance(obj, str) and obj == merge_node:
                t["object"] = keep_node
            elif isinstance(obj, list):
                t["object"] = [keep_node if x == merge_node else x for x in obj]
        self.nodes.remove(merge_node)
        self.build_graph()

    def rename_node_btn(self):
        self.cancel_pick_mode()
        old_name = self.rename_old_field.text().strip()
        new_name = self.rename_new_field.text().strip()
        if not old_name or not new_name:
            QMessageBox.warning(self, "Error", "Please specify Old Name and New Name.")
            return
        if old_name not in self.nodes:
            QMessageBox.warning(self, "Error", f"Node '{old_name}' not found.")
            return
        if new_name in self.nodes:
            QMessageBox.warning(self, "Error", f"Node '{new_name}' already exists.")
            return
        for t in self.triples:
            if t["subject"] == old_name:
                t["subject"] = new_name
            obj_val = t["object"]
            if isinstance(obj_val, str) and obj_val == old_name:
                t["object"] = new_name
            elif isinstance(obj_val, list):
                t["object"] = [new_name if x == old_name else x for x in obj_val]
        self.save_state()
        self.nodes.remove(old_name)
        self.nodes.append(new_name)
        self.build_graph()

    def delete_node_via_bridge(self, nodeId):
        if nodeId in self.nodes:
            self.save_state()
            self.delete_node(nodeId)

    def delete_edge_via_bridge(self, edgeId):
        try:
            subj, rel, obj_val = edgeId.split("|")
        except Exception as e:
            print("Invalid edge id format:", e)
            return
        new_triples = []
        for t in self.triples:
            if t.get("subject") == subj and t.get("relation") == rel:
                current_obj = t.get("object")
                if isinstance(current_obj, list):
                    filtered = [o for o in current_obj if o != obj_val]
                    if filtered:
                        t["object"] = filtered
                        new_triples.append(t)
                else:
                    if current_obj == obj_val:
                        continue
                    else:
                        new_triples.append(t)
            else:
                new_triples.append(t)
        self.save_state()
        self.triples = new_triples
        self.build_graph()

    def open_clustering_options(self):
        dlg = ClusteringOptionsDialog(self)
        dlg.set_values(self.n_clusters, self.num_topics, self.dynamic_threshold)
        if dlg.exec_() == QDialog.Accepted:
            (self.n_clusters,
             self.num_topics,
             self.dynamic_threshold) = dlg.get_values()

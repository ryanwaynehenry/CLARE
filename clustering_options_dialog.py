from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QRadioButton, QButtonGroup, QFrame
)
from PyQt5.QtCore import Qt

class ClusteringOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Clustering Options")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Cluster Selection Group
        cluster_group = QFrame()
        cluster_group.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        cluster_layout = QVBoxLayout(cluster_group)
        
        cluster_label = QLabel("Number of Clusters:")
        cluster_label.setToolTip(
            "Controls how the text is grouped into clusters before extracting topics.\n"
            "Auto: Automatically determines optimal number based on content.\n"
            "Manual: Specify exact number of clusters (minimum 2)."
        )
        cluster_layout.addWidget(cluster_label)
        
        # Radio buttons for auto/manual clusters
        self.auto_clusters_radio = QRadioButton("Auto")
        self.manual_clusters_radio = QRadioButton("Manual:")
        cluster_layout.addWidget(self.auto_clusters_radio)
        
        # Manual cluster selection
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(self.manual_clusters_radio)
        self.clusters_spinbox = QSpinBox()
        self.clusters_spinbox.setMinimum(2)
        self.clusters_spinbox.setMaximum(100)
        self.clusters_spinbox.setValue(8)
        self.clusters_spinbox.setEnabled(False)
        manual_layout.addWidget(self.clusters_spinbox)
        manual_layout.addStretch()
        cluster_layout.addLayout(manual_layout)
        
        # Button group for radio buttons
        cluster_group_btns = QButtonGroup(self)
        cluster_group_btns.addButton(self.auto_clusters_radio)
        cluster_group_btns.addButton(self.manual_clusters_radio)
        self.auto_clusters_radio.setChecked(True)
        
        # Connect radio buttons
        self.auto_clusters_radio.toggled.connect(lambda checked: self.clusters_spinbox.setEnabled(not checked))
        
        layout.addWidget(cluster_group)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Topics per Cluster
        topics_group = QFrame()
        topics_group.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        topics_layout = QVBoxLayout(topics_group)
        
        topics_label = QLabel("Topics per Cluster:")
        topics_label.setToolTip(
            "Maximum number of topics/keywords to extract from each cluster.\n"
            "Higher values will result in more detailed but potentially noisier results."
        )
        topics_layout.addWidget(topics_label)
        
        self.topics_spinbox = QSpinBox()
        self.topics_spinbox.setMinimum(1)
        self.topics_spinbox.setMaximum(50)
        self.topics_spinbox.setValue(10)
        topics_layout.addWidget(self.topics_spinbox)
        
        layout.addWidget(topics_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setMinimumWidth(300)
    
    def get_values(self):
        """Returns (n_clusters, num_topics) where n_clusters is None for auto"""
        n_clusters = None if self.auto_clusters_radio.isChecked() else self.clusters_spinbox.value()
        num_topics = self.topics_spinbox.value()
        return n_clusters, num_topics
    
    def set_values(self, n_clusters, num_topics):
        """Sets the dialog values. n_clusters=None means auto"""
        if n_clusters is None:
            self.auto_clusters_radio.setChecked(True)
        else:
            self.manual_clusters_radio.setChecked(True)
            self.clusters_spinbox.setValue(n_clusters)
        self.topics_spinbox.setValue(num_topics) 
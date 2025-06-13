from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QPushButton, QRadioButton, QButtonGroup, QFrame, QToolButton
)
from PyQt5.QtCore import Qt


class ClusteringOptionsDialog(QDialog):
    """
    Advanced clustering options, including a dynamic-threshold
    percentage for filtering relation scores.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Clustering Options")
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Cluster selection --------------------------------------------------
        cluster_group = QFrame()
        cluster_group.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        cluster_layout = QVBoxLayout(cluster_group)

        cluster_label = QLabel("Number of Clusters:")
        cluster_label.setToolTip(
            "Controls how the transcript is grouped before extracting topics.\n"
            "• Auto – determine an optimal value.\n"
            "• Manual – supply an exact integer (min 2)."
        )
        cluster_layout.addWidget(cluster_label)

        self.auto_clusters_radio = QRadioButton("Auto")
        self.manual_clusters_radio = QRadioButton("Manual:")
        cluster_layout.addWidget(self.auto_clusters_radio)

        manual_layout = QHBoxLayout()
        manual_layout.addWidget(self.manual_clusters_radio)
        self.clusters_spinbox = QSpinBox()
        self.clusters_spinbox.setRange(2, 100)
        self.clusters_spinbox.setValue(8)
        self.clusters_spinbox.setEnabled(False)
        manual_layout.addWidget(self.clusters_spinbox)
        manual_layout.addStretch()
        cluster_layout.addLayout(manual_layout)

        btn_grp = QButtonGroup(self)
        btn_grp.addButton(self.auto_clusters_radio)
        btn_grp.addButton(self.manual_clusters_radio)
        self.auto_clusters_radio.setChecked(True)
        self.auto_clusters_radio.toggled.connect(
            lambda checked: self.clusters_spinbox.setEnabled(not checked)
        )

        layout.addWidget(cluster_group)

        # Separator ----------------------------------------------------------
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # Topics per cluster ------------------------------------------------
        topics_group = QFrame()
        topics_group.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        topics_layout = QVBoxLayout(topics_group)

        topics_label = QLabel("Topics per Cluster:")
        topics_label.setToolTip(
            "Max keywords per cluster; more adds detail but may add noise."
        )
        topics_layout.addWidget(topics_label)

        self.topics_spinbox = QSpinBox()
        self.topics_spinbox.setRange(1, 50)
        self.topics_spinbox.setValue(10)
        topics_layout.addWidget(self.topics_spinbox)

        layout.addWidget(topics_group)

        # Dynamic threshold (%) ---------------------------------------------
        thresh_group = QFrame()
        thresh_group.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        thresh_layout = QHBoxLayout(thresh_group)

        thresh_label = QLabel("Dynamic Threshold (%):")
        thresh_layout.addWidget(thresh_label)

        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setRange(1, 100)
        self.threshold_spinbox.setValue(50)
        thresh_layout.addWidget(self.threshold_spinbox)

        thresh_layout.addStretch()

        help_btn = QToolButton()
        help_btn.setText("?")
        help_btn.setAutoRaise(True)
        help_btn.setEnabled(False)
        help_btn.setToolTip(
            "Keep only the top X% of relation scores when building edges. 1%=strict, 100%=all."
        )
        thresh_layout.addWidget(help_btn)

        layout.addWidget(thresh_group)

        # Dialog buttons ----------------------------------------------------
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self.setMinimumWidth(320)

    def get_values(self):
        """
        Returns (n_clusters, num_topics, dyn_thresh_pct).
        n_clusters=None when Auto.
        """
        n = None if self.auto_clusters_radio.isChecked() else self.clusters_spinbox.value()
        return n, self.topics_spinbox.value(), self.threshold_spinbox.value()

    def set_values(self, n_clusters, num_topics, dyn_thresh_pct):
        if n_clusters is None:
            self.auto_clusters_radio.setChecked(True)
        else:
            self.manual_clusters_radio.setChecked(True)
            self.clusters_spinbox.setValue(n_clusters)
        self.topics_spinbox.setValue(num_topics)
        self.threshold_spinbox.setValue(dyn_thresh_pct)

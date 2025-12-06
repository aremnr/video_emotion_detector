import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QFileDialog, QLabel
)
from PyQt6.QtCore import Qt
from video_widget import VideoWidget
from engagement_graph import EngagementGraph
from PyQt6.QtWidgets import QApplication
#from theme import apply_light_theme


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BrainBit Engagement Tracker")
        self.setMinimumSize(1300, 720)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # -------- LEFT PANEL (VIDEO + HEATMAP + CONTROLS) --------
        left = QVBoxLayout()
        left.setSpacing(8)
        left.setContentsMargins(0, 0, 0, 0)

        # Кнопка загрузки видео
        self.load_btn = QPushButton("Load Video")
        self.load_btn.setFixedHeight(32)
        self.load_btn.setFixedWidth(120)
        self.load_btn.clicked.connect(self.load_video_dialog)
        left.addWidget(self.load_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # Название видео + статус EEG
        self.title_label = QLabel("No video loaded")
        self.status_label = QLabel("EEG: OFFLINE")

        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.status_label.setStyleSheet("color: green; font-size: 12px;")

        left.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignLeft)
        left.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignLeft)

        # Видео + heatmap (всё внутри VideoWidget)
        self.video = VideoWidget()
        left.addWidget(self.video, stretch=1)

        # -------- RIGHT PANEL (GRAPH) --------
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(5)

        self.graph = EngagementGraph()
        right.addWidget(self.graph, stretch=1)

        # Глобальный layout
        layout.addLayout(left, stretch=2)
        layout.addLayout(right, stretch=1)

        # Привязываем график к видеовиджету
        self.video.graph = self.graph

    # -------------------------------
    # LOAD VIDEO
    # -------------------------------
    def load_video_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.mkv *.avi *.mov)"
        )
        if file_path:
            self.video.load(file_path)

            # Имя файла
            import os
            self.title_label.setText(os.path.basename(file_path))

            # Статус EEG (пока прототип)
            self.status_label.setText("EEG: ONLINE")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    #apply_light_theme(app)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

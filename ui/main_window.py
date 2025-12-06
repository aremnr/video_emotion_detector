import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QFileDialog, QLabel
)
from PyQt6.QtCore import Qt
from ui.video_widget import VideoWidget
from ui.engagement_graph import EngagementGraph
from PyQt6.QtWidgets import QApplication
from backend.src.sensor_event_handler import SensorEventHandler
from backend.src.sensor_connector import SensorConnector
from backend.src.emotion_math_manager import EmotionMathManager
#from theme import apply_light_theme


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BrainBit Engagement Tracker")
        self.setMinimumSize(1300, 720)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        self.math = EmotionMathManager()
        self.handler = SensorEventHandler(math_manager=self.math)

        # Передаём callback для UI
        self.handler.on_engagement_update = self.on_engagement_update

        self.connector = SensorConnector(
            event_handler=self.handler,
            math_manager=self.math
        )

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
        self.status_label.setStyleSheet("color: red; font-size: 12px;")

        left.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignLeft)
        left.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignLeft)

        # Видео + heatmap (всё внутри VideoWidget)
        self.video = VideoWidget()
        left.addWidget(self.video, stretch=1)

        # Кнопка старта/остановки EEG сессии
        self.session_btn = QPushButton("Start EEG Session")
        self.session_btn.setFixedHeight(32)
        self.session_btn.clicked.connect(self.toggle_session)
        left.addWidget(self.session_btn, alignment=Qt.AlignmentFlag.AlignLeft)

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

        # Флаг состояния сессии
        self.session_active = False

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

    def on_engagement_update(self, value):
        # value = от 0 до 1
        self.graph.add_external_value(value)
        self.video.update_heatmap_from_value(value)

    def toggle_session(self):
        if not self.session_active:
            # Старт сессии
            try:
                self.connector.start_signal_from_ui()
                self.session_btn.setText("Stop EEG Session")
                self.session_active = True
                self.status_label.setText("EEG: ONLINE")
                self.status_label.setStyleSheet("color: green; font-size: 12px;")
            except Exception as e:
                print(f"Cannot start EEG session: {e}")
        else:
            # Стоп сессии
            try:
                self.connector.stop_signal_from_ui()
                self.session_btn.setText("Start EEG Session")
                self.session_active = False
                self.status_label.setText("EEG: OFFLINE")
                self.status_label.setStyleSheet("color: red; font-size: 12px;")
            except Exception as e:
                print(f"Cannot stop EEG session: {e}")



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    #apply_light_theme(app)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

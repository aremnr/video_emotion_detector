import sys
import datetime
import os
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

        self.global_min = 999
        self.global_max = -999
        self.log_path = None

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

        if not self.session_active:
            return

        current = self.graph.get_latest_value() * 100
        timestamp = self.get_video_timestamp()

        if current < 10:
            self.log_event(f"[LOW] {timestamp}s  → {current:.1f}%\n")

        if current > 50:
            self.log_event(f"[HIGH] {timestamp}s → {current:.1f}%\n")

        if current < self.global_min:
            self.global_min = current
            self.log_event(f"[NEW GLOBAL MIN] {timestamp}s → {current:.1f}%\n")

        if current > self.global_max:
            self.global_max = current
            self.log_event(f"[NEW GLOBAL MAX] {timestamp}s → {current:.1f}%\n")


    def log_event(self, text: str):
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"LOG ERROR: {e}")


    def toggle_session(self):
        if not self.session_active:
            # Старт сессии
            try:
                self.connector.start_signal_from_ui()
                self.session_btn.setText("Stop EEG Session")
                self.session_active = True
                self.status_label.setText("EEG: ONLINE")
                self.status_label.setStyleSheet("color: green; font-size: 12px;")

                # --- СОЗДАТЬ НОВЫЙ ЛОГ-ФАЙЛ ---
                ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                logs_dir = "session_logs"
                os.makedirs(logs_dir, exist_ok=True)
                self.log_path = os.path.join(logs_dir, f"session_{ts}.txt")

                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.write("=== EEG SESSION STARTED ===\n")
                    f.write(f"Created: {ts}\n\n")

                # сброс глобальных минимумов/максимумов
                self.global_min = 999
                self.global_max = -999

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

                # финальный лог
                if self.log_path:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write("=== SESSION ENDED ===\n")

            except Exception as e:
                print(f"Cannot stop EEG session: {e}")


    def get_video_timestamp(self):
        try:
            if not self.video or not self.video.player:
                return 0
            # проверяем, есть ли свойство position
            if not hasattr(self.video.player, "position"):
                return 0
            return round(self.video.player.position() / 1000, 2)
        except Exception:
            return 0



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    #apply_light_theme(app)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

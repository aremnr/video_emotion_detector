import sys
import datetime
import os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QFileDialog, QLabel
)
from PyQt6.QtCore import Qt

# --- ИМПОРТЫ ВАШИХ МОДУЛЕЙ ---
from ui.video_widget import VideoWidget
from ui.engagement_graph import EngagementGraph
from backend.src.sensor_event_handler import SensorEventHandler
from backend.src.sensor_connector import SensorConnector
from backend.src.emotion_math_manager import EmotionMathManager

# --- ИМПОРТ АНАЛИЗАТОРА ИЗ СОСЕДНЕГО ФАЙЛА (analyzer.py) ---
from ui.ui2 import run_with_params


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.low_period = None
        self.high_period = None
        self.global_min = 999
        self.global_max = -999
        self.global_min_timestamp = None
        self.global_max_timestamp = None

        self.session_start_time = None
        self.log_path = None
        self.current_video_path = None  # Для хранения пути к видео
        
        self.log_interval_ms = 2000
        self.next_log_timestamp = 0
        self.buffer_relaxation = []
        self.buffer_focus = []
        self.buffer_artifact = []
        
        self.is_calibrated = False

        self.setWindowTitle("BrainBit Engagement Tracker")
        self.setMinimumSize(1300, 720)
        self.setWindowState(Qt.WindowState.WindowMaximized)
        # Главный Layout
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(10)

        self.math = EmotionMathManager()
        self.handler = SensorEventHandler(math_manager=self.math)
        self.handler.on_engagement_update = self.on_engagement_update

        self.connector = SensorConnector(
            event_handler=self.handler,
            math_manager=self.math
        )

        # ==========================================
        # 1. ЛЕВАЯ ПАНЕЛЬ (Контейнер)
        # ==========================================
        self.left_container = QWidget()
        self.left_layout = QVBoxLayout(self.left_container)
        self.left_layout.setSpacing(8)
        self.left_layout.setContentsMargins(0, 0, 0, 0)

        self.load_btn = QPushButton("Load Video")
        self.load_btn.setFixedHeight(32)
        self.load_btn.setFixedWidth(120)
        self.load_btn.clicked.connect(self.load_video_dialog)
        self.left_layout.addWidget(self.load_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self.title_label = QLabel("No video loaded")
        self.status_label = QLabel("EEG: OFFLINE")
        self.calib_label = QLabel("Calibration: 0%")
        
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.status_label.setStyleSheet("color: red; font-size: 12px;")
        self.calib_label.setStyleSheet("color: #0078d7; font-size: 12px;")

        self.left_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignLeft)
        self.left_layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignLeft)
        self.left_layout.addWidget(self.calib_label, alignment=Qt.AlignmentFlag.AlignLeft)

        self.video = VideoWidget()
        self.video.fullscreen_changed.connect(self.toggle_video_fullscreen)
        self.left_layout.addWidget(self.video, stretch=1)

        # Кнопка старта/стопа сессии
        self.session_btn = QPushButton("Start EEG Session")
        self.session_btn.setFixedHeight(32)
        self.session_btn.clicked.connect(self.toggle_session)
        self.session_btn.setEnabled(False) 
        self.left_layout.addWidget(self.session_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # --- НОВАЯ КНОПКА ДЛЯ АНАЛИЗА ---
        self.analyze_btn = QPushButton("ANALYZE RESULTS")
        self.analyze_btn.setFixedHeight(40)
        # Делаем её заметной (синей)
        self.analyze_btn.setStyleSheet("background-color: #3ea6ff; color: white; font-weight: bold; border-radius: 4px;")
        self.analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_btn.clicked.connect(self.launch_analyzer)
        self.analyze_btn.setVisible(False) # Скрыта по умолчанию
        self.left_layout.addWidget(self.analyze_btn)

        # ==========================================
        # 2. ПРАВАЯ ПАНЕЛЬ (Контейнер)
        # ==========================================
        self.right_container = QWidget()
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(5)

        self.graph = EngagementGraph()
        self.right_layout.addWidget(self.graph, stretch=1)

        # ==========================================
        # СБОРКА MAIN LAYOUT
        # ==========================================
        self.main_layout.addWidget(self.left_container, stretch=2)
        self.main_layout.addWidget(self.right_container, stretch=1)

        self.video.graph = self.graph
        self.session_active = False
        self.video.set_play_allowed(False)

        # Список того, что скрываем при фулскрине
        self.widgets_to_hide = [
            self.load_btn,
            self.title_label,
            self.status_label,
            self.calib_label,
            self.session_btn,
            self.analyze_btn, # Добавили кнопку анализа в список скрываемых
            self.right_container 
        ]

        # Ссылка на окно анализатора, чтобы сборщик мусора не убил его сразу
        self.analyzer_window = None

    def toggle_video_fullscreen(self, enable: bool):
        """Режим 'Кинотеатр': скрываем всё кроме видео."""
        if enable:
            for w in self.widgets_to_hide:
                w.setVisible(False)
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            self.video.enter_fullscreen_mode()
            self.showFullScreen()
        else:
            self.showNormal()
            for w in self.widgets_to_hide:
                # ВАЖНО: Кнопку анализа показываем только если путь к логу существует (сессия закончилась)
                if w == self.analyze_btn:
                    if self.log_path and not self.session_active and os.path.exists(self.log_path):
                        w.setVisible(True)
                    else:
                        w.setVisible(False)
                else:
                    w.setVisible(True)
            
            self.main_layout.setContentsMargins(5, 5, 5, 5)
            self.video.exit_fullscreen_mode()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.toggle_video_fullscreen(False)
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def load_video_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.mkv *.avi *.mov)"
        )
        if file_path:
            self.current_video_path = file_path # Сохраняем путь
            self.video.load(file_path)
            self.title_label.setText(os.path.basename(file_path))
            self.session_btn.setEnabled(True)
            # При загрузке нового видео скрываем кнопку анализа предыдущего
            self.analyze_btn.setVisible(False)
            self.log_path = None

    def on_engagement_update(self, value, calib_percent=None, relaxation=None, focus=None, artifact=None):
        self.graph.add_external_value(value)
        self.video.update_heatmap_from_value(value)

        if calib_percent is not None:
            if calib_percent >= 100:
                self.calib_label.setText(f"Calibration: {calib_percent:.1f}% (Готов к работе)")
                if not self.is_calibrated:
                    self.is_calibrated = True
                    self.video.set_play_allowed(True)
            else:
                self.calib_label.setText(f"Calibration: {calib_percent:.1f}%")
                if self.is_calibrated:
                    self.is_calibrated = False
                    self.video.set_play_allowed(False)

        if not self.session_active or not self.video.is_playing():
            return

        current_ms = self.video.get_timestamp_ms()

        if current_ms < self.next_log_timestamp:
            self.next_log_timestamp = (current_ms // self.log_interval_ms) * self.log_interval_ms
            self.buffer_relaxation.clear()
            self.buffer_focus.clear()
            self.buffer_artifact.clear()

        rel_val = relaxation if relaxation is not None else value * 100
        foc_val = focus if focus is not None else value * 100
        art_val = int(artifact) if artifact is not None else 0

        self.buffer_relaxation.append(rel_val)
        self.buffer_focus.append(foc_val)
        self.buffer_artifact.append(art_val)

        if current_ms >= (self.next_log_timestamp + self.log_interval_ms):
            self.write_averaged_log()
            self.next_log_timestamp += self.log_interval_ms

    def write_averaged_log(self):
        if not self.buffer_relaxation:
            return

        avg_rel = sum(self.buffer_relaxation) / len(self.buffer_relaxation)
        avg_foc = sum(self.buffer_focus) / len(self.buffer_focus)
        avg_art = sum(self.buffer_artifact) / len(self.buffer_artifact)

        log_line = f"{self.next_log_timestamp},{int(round(avg_rel))},{int(round(avg_foc))},{int(round(avg_art))}\n"
        self.log_event(log_line)

        self.buffer_relaxation.clear()
        self.buffer_focus.clear()
        self.buffer_artifact.clear()

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
            # --- START ---
            try:
                self.connector.start_signal_from_ui()
                self.session_btn.setText("Stop EEG Session")
                self.session_active = True
                self.analyze_btn.setVisible(False) # Скрываем кнопку анализа во время записи
                
                self.status_label.setText("EEG: ONLINE")
                self.status_label.setStyleSheet("color: green; font-size: 12px;")

                # Генерация пути CSV (папка session_logs, рандомное имя на основе времени)
                ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                logs_dir = "session_logs"
                os.makedirs(logs_dir, exist_ok=True)
                self.log_path = os.path.join(logs_dir, f"session_{ts}.csv")

                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.write("Timestamp_ms,Relaxation,Focus,Artifact\n")

                vid_ms = self.video.get_timestamp_ms()
                self.next_log_timestamp = (vid_ms // self.log_interval_ms) * self.log_interval_ms
                
                self.buffer_relaxation = []
                self.buffer_focus = []
                self.buffer_artifact = []

            except Exception as e:
                print(f"Cannot start EEG session: {e}")
        else:
            # --- STOP ---
            try:
                self.connector.stop_signal_from_ui()
                self.session_btn.setText("Start EEG Session")
                self.session_active = False
                self.status_label.setText("EEG: OFFLINE")
                self.status_label.setStyleSheet("color: red; font-size: 12px;")
                
                if self.log_path:
                    
                    # Сессия завершена, файл записан. Показываем кнопку перехода к анализу.
                    self.analyze_btn.setVisible(True)

            except Exception as e:
                print(f"Cannot stop EEG session: {e}")

    def launch_analyzer(self):
        """Запускает программу анализатора и закрывает текущую."""
        run_with_params(
            video=self.current_video_path,
            csv=self.log_path
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
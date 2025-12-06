import os
import sys

dll_dir = os.path.abspath(os.path.dirname(__file__))

os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

# вручную укажем путь, чтобы ctypes видел DLL
os.add_dll_directory(dll_dir)


import mpv
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QWindow
from heatmap import HeatmapWidget
import random

class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph = None 
        self.player = None
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors)

        # Основной layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(5)

        # Видео-плейсхолдер
        self.video_placeholder = QWidget()
        self.video_placeholder.setMinimumSize(640, 360)
        self.layout.addWidget(self.video_placeholder)

        # Таймкод
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.time_label)

        # Progress bar
        self.progress = QSlider(Qt.Orientation.Horizontal)
        self.progress.setRange(0, 1000)
        self.layout.addWidget(self.progress)

        # Heatmap placeholder
        self.heatmap = HeatmapWidget()
        self.layout.addWidget(self.heatmap)
        self.heatmap.setFixedHeight(30)

        # Кнопки управления
        self.controls = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.pause_btn = QPushButton("⏸")
        self.stop_btn = QPushButton("■")
        for btn in [self.play_btn, self.pause_btn, self.stop_btn]:
            btn.setFixedHeight(40)
            btn.setFixedWidth(60)
            self.controls.addWidget(btn)
        self.layout.addLayout(self.controls)

        # Сигналы кнопок
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)
        self.progress.sliderReleased.connect(self.on_slider_released)

        # Таймер для обновления прогресса и таймкода
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start()


    def init_player(self):
        if self.player is not None:
            return
        wid = int(self.video_placeholder.winId())
        self.player = mpv.MPV(
            wid=str(wid),
            osc=False,
            hwdec='auto',
            vo='gpu',
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.init_player()

    def load(self, filepath: str):
        if self.player is None:
            self.init_player()
        self.player.play(filepath)

        duration = int(self.player.duration) if self.player.duration else 300

        # → Инициализируем модель данных заранее
        self.heatmap_data = [None] * duration

        # → Пересоздаём heatmap один раз
        self.heatmap.set_data(self.heatmap_data)


    def play(self):
        if self.player:
            self.player.pause = False

    def pause(self):
        if self.player:
            self.player.pause = True

    def stop(self):
        if self.player:
            self.player.stop()
            self.progress.setValue(0)
            self.time_label.setText("00:00 / 00:00")
            self.heatmap.set_data([])

    def update_progress(self):
        if (
            self.player is None 
            or self.player.time_pos is None 
            or self.player.duration is None
        ):
            return

        # --- Прогресс и таймкод ---
        cur = int(self.player.time_pos)
        dur = int(self.player.duration)

        if dur == 0:
            return

        # progress bar (0–1000)
        value = int((cur / dur) * 1000)

        self.progress.blockSignals(True)
        self.progress.setValue(value)
        self.progress.blockSignals(False)

        # таймкод
        cur_t = f"{cur//60:02d}:{cur%60:02d}"
        dur_t = f"{dur//60:02d}:{dur%60:02d}"
        self.time_label.setText(f"{cur_t} / {dur_t}")

        # --- Heatmap + график ---
        if self.graph and self.heatmap_data:

            if 0 <= cur < dur:
                # если пришёл новый момент времени
                if self.heatmap_data[cur] is None:
                    engagement = self.graph.get_latest_value()
                    self.heatmap_data[cur] = engagement

                    # обновляем полосу heatmap только при изменении данных
                    self.heatmap.set_data(self.heatmap_data)

            # двигаем позицию индикатора
            self.heatmap.set_position(cur)
    
    def on_slider_released(self):
        if self.player and self.player.duration:
            value = self.progress.value()
            new_time = (value / 1000) * self.player.duration
            self.player.seek(new_time, reference="absolute")
import os

dll_dir = os.path.abspath(os.path.dirname(__file__))
os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(dll_dir)

import mpv
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QFrame
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from ui.heatmap import HeatmapWidget

class VideoWidget(QWidget):
    # Сигнал: True = вошли в фулскрин, False = вышли
    fullscreen_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.graph = None 
        self.player = None
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors)

        self.play_allowed = False
        self.is_fullscreen = False 

        # --- Основной layout ---
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0) # Убираем отступы между видео и контролами

        # 1. Область видео (всегда видна)
        self.video_placeholder = QWidget()
        self.video_placeholder.setMinimumSize(640, 360)
        self.video_placeholder.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_placeholder, stretch=1)

        # 2. Контейнер для всего интерфейса (скроем его в фулскрине)
        self.ui_container = QWidget()
        self.ui_layout = QVBoxLayout(self.ui_container)
        self.ui_layout.setContentsMargins(0, 5, 0, 0)
        self.ui_layout.setSpacing(5)
        
        # --- Наполняем контейнер интерфейса ---
        
        # Таймкод
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui_layout.addWidget(self.time_label)

        # Progress bar
        self.progress = QSlider(Qt.Orientation.Horizontal)
        self.progress.setRange(0, 1000)
        self.ui_layout.addWidget(self.progress)

        # Heatmap
        self.heatmap = HeatmapWidget()
        self.heatmap.setFixedHeight(30)
        self.ui_layout.addWidget(self.heatmap)

        # Кнопки
        self.controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.pause_btn = QPushButton("⏸")
        self.stop_btn = QPushButton("■")
        self.fullscreen_btn = QPushButton("⛶")

        for btn in [self.play_btn, self.pause_btn, self.stop_btn, self.fullscreen_btn]:
            btn.setFixedHeight(40)
            btn.setFixedWidth(60)
            self.controls_layout.addWidget(btn)
        
        self.ui_layout.addLayout(self.controls_layout)
        
        # Добавляем контейнер в основной layout
        self.layout.addWidget(self.ui_container, stretch=0)


        # Сигналы
        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen_signal)
        self.progress.sliderReleased.connect(self.on_slider_released)

        # Таймер
        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start()

        self.set_play_allowed(False)

    def init_player(self):
        if self.player is not None:
            return
        wid = int(self.video_placeholder.winId())
        self.player = mpv.MPV(
            wid=str(wid),
            osc=False,
            hwdec='auto',
            vo='gpu',
            input_default_bindings=True, 
            input_vo_keyboard=True,
        )
        # Отключаем встроенную обработку ESC в MPV, чтобы Qt могло его поймать
        # (Или просто надеемся, что фокус останется у окна)

    def showEvent(self, event):
        super().showEvent(event)
        self.init_player()

    def load(self, filepath: str):
        if self.player is None:
            self.init_player()
        
        self.player.play(filepath)
        self.player.pause = True

        duration = int(self.player.duration) if self.player.duration else 300
        self.heatmap_data = [None] * duration
        self.heatmap.set_data(self.heatmap_data)

    # --- Управление фулскрином ---
    def toggle_fullscreen_signal(self):
        self.is_fullscreen = not self.is_fullscreen
        self.fullscreen_changed.emit(self.is_fullscreen)

    def enter_fullscreen_mode(self):
        """Скрывает интерфейс плеера, оставляя только видео."""
        self.is_fullscreen = True
        self.ui_container.setVisible(False)
        self.layout.setContentsMargins(0, 0, 0, 0)
    
    def exit_fullscreen_mode(self):
        """Возвращает интерфейс плеера."""
        self.is_fullscreen = False
        self.ui_container.setVisible(True)
        # Отступы можно вернуть при желании, но у нас 0 по умолчанию
        self.layout.setContentsMargins(0, 0, 0, 0)

    # --- Стандартные методы ---
    def set_play_allowed(self, allowed: bool):
        self.play_allowed = allowed
        self.play_btn.setEnabled(allowed)
        self.progress.setEnabled(allowed)

    def play(self):
        if not self.play_allowed:
            return
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

    def is_playing(self):
        if self.player and not self.player.pause:
            return True
        return False

    def get_timestamp_ms(self):
        if self.player and self.player.time_pos is not None:
            return int(self.player.time_pos * 1000)
        return 0

    def update_progress(self):
        if (
            self.player is None 
            or self.player.time_pos is None 
            or self.player.duration is None
        ):
            return

        cur = int(self.player.time_pos)
        dur = int(self.player.duration)

        if dur == 0:
            return

        value = int((cur / dur) * 1000)

        if not self.progress.isSliderDown():
            self.progress.blockSignals(True)
            self.progress.setValue(value)
            self.progress.blockSignals(False)

        cur_t = f"{cur//60:02d}:{cur%60:02d}"
        dur_t = f"{dur//60:02d}:{dur%60:02d}"
        self.time_label.setText(f"{cur_t} / {dur_t}")

        if self.graph and self.heatmap_data:
            if 0 <= cur < dur:
                if self.heatmap_data[cur] is None:
                    engagement = self.graph.get_latest_value()
                    self.heatmap_data[cur] = engagement
                    self.heatmap.set_data(self.heatmap_data)
            self.heatmap.set_position(cur)
    
    def on_slider_released(self):
        if not self.play_allowed:
            return
        if self.player and self.player.duration:
            value = self.progress.value()
            new_time = (value / 1000) * self.player.duration
            self.player.seek(new_time, reference="absolute")

    def update_heatmap_from_value(self, value):
        if self.heatmap:
            self.heatmap.set_value(value)
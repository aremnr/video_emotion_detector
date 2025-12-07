from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import time


class EngagementGraph(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # --- Настройки графика ---
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMouseEnabled(x=True, y=True)  # разрешаем zoom/scroll мышью
        self.plot_widget.setMenuEnabled(True)
        self.plot_widget.setLabel('left', 'Engagement (%)')
        self.plot_widget.setLabel('bottom', 'Time (frames)')

        self.plot = self.plot_widget.plot(
            [],
            pen=pg.mkPen(color=(0, 200, 255), width=2),
            antialias=True
        )

        self.engagement_label = QLabel("Current engagement: 0%")
        self.engagement_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.engagement_label.setStyleSheet(
            "font-weight: bold; color: #0040c0; font-size: 13px;"
        )
        layout.addWidget(self.engagement_label)

        self.data = []
        self.max_points = 10000
        self.alpha = 0.2

        # Таймер для плавного обновления графика
        self._timer = pg.QtCore.QTimer()
        self._timer.setInterval(50)  # 20 FPS
        self._timer.timeout.connect(self._update_plot)
        self._pending_update = False
        self._timer.start()

        self._last_smooth = 50

    def add_external_value(self, value):
        raw = value * 100
        last = self.data[-1] if self.data else 50
        smooth = self.alpha * raw + (1 - self.alpha) * last
        self._last_smooth = smooth
        self.data.append(smooth)
        if len(self.data) > self.max_points:
            self.data.pop(0)
        self._pending_update = True

    def _update_plot(self):
        if not self._pending_update:
            return
        self._pending_update = False
        self.plot.setData(self.data, clear=False)
        # Автоматическое масштабирование по Y
        if self.data:
            min_y = min(self.data)
            max_y = max(self.data)
            if min_y == max_y:
                min_y -= 1
                max_y += 1
            self.plot_widget.setYRange(min_y - 5, max_y + 5, padding=0)
            self.plot_widget.setXRange(max(0, len(self.data) - 200), len(self.data), padding=0)
        self.engagement_label.setText(f"Current engagement: {int(self._last_smooth)}%")

    def get_latest_value(self):
        if not self.data:
            return 0.5
        val = self.data[-1]
        return min(1.0, max(0.0, val / 100.0))

    def get_data(self):
        return self.data


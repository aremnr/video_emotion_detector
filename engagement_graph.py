from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QTimer, Qt
import pyqtgraph as pg
import random

class EngagementGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # --- Настройка графика ---
        #self.plot_widget.setBackground('w')  # белое поле
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)  # лёгкая сетка
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.setXRange(0, 200)

        # Линия графика
        self.plot = self.plot_widget.plot([], pen=pg.mkPen(color=(0, 200, 255), width=2), antialias=True)

        # --- Метка текущей вовлеченности ---
        self.engagement_label = QLabel("Current engagement: 0%")
        self.engagement_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.engagement_label.setStyleSheet("font-weight: bold; color: #0040c0; font-size: 13px;")
        layout.addWidget(self.engagement_label)

        # Данные
        self.data = []
        self.max_points = 200

        # Таймер обновления
        self.timer = QTimer()
        self.timer.setInterval(300)  # каждые 100 мс
        self.timer.timeout.connect(self.update_graph)
        self.timer.start()

        # Параметр сглаживания
        self.alpha = 0.2  # чем меньше, тем плавнее

    # --- Обновление графика ---
    def update_graph(self):
        new_raw = random.randint(0, 100)

        # сглаживание с предыдущим значением
        last_value = self.data[-1] if self.data else 50
        new_value = self.alpha * new_raw + (1 - self.alpha) * last_value

        self.data.append(new_value)
        if len(self.data) > self.max_points:
            self.data.pop(0)

        # обновляем линию
        self.plot.setData(self.data)

        # обновляем метку под графиком
        self.engagement_label.setText(f"Current engagement: {int(new_value)}%")

    # --- Получить все данные ---
    def get_data(self):
        return self.data

    # --- Последнее значение для heatmap ---
    def get_latest_value(self):
        if not self.data:
            return 0.5  # если нет данных, середина
        val = self.data[-1]
        return min(1.0, max(0.0, val / 100.0))

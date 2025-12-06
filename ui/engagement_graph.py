from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
import pyqtgraph as pg


class EngagementGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # --- Настройки графика ---
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.setXRange(0, 200)

        # Линия графика
        self.plot = self.plot_widget.plot(
            [],
            pen=pg.mkPen(color=(0, 200, 255), width=2),
            antialias=True
        )

        # Подпись текущего значения
        self.engagement_label = QLabel("Current engagement: 0%")
        self.engagement_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.engagement_label.setStyleSheet(
            "font-weight: bold; color: #0040c0; font-size: 13px;"
        )
        layout.addWidget(self.engagement_label)

        # Данные
        self.data = []
        self.max_points = 10000

        # Параметр сглаживания
        self.alpha = 0.2

    # ----------------------------------------
    # 🔥 Метод для получения реальных значений
    # вызывается из MainWindow.on_engagement_update()
    # ----------------------------------------
    def add_external_value(self, value):
        raw = value * 100

        # сглаживание
        last = self.data[-1] if self.data else 50
        smooth = self.alpha * raw + (1 - self.alpha) * last

        # обновление массива
        self.data.append(smooth)
        if len(self.data) > self.max_points:
            self.data.pop(0)

        # обновление графика
        self.plot.setData(self.data)

        # 🎯 --- АВТОПРОКРУТКА X ---
        visible_width = 200  # сколько секунд видно в окне
        n = len(self.data)

        if n > visible_width:
            self.plot_widget.setXRange(n - visible_width, n)
        else:
            self.plot_widget.setXRange(0, visible_width)

        # подпись
        self.engagement_label.setText(f"Current engagement: {int(smooth)}%")

    def get_latest_value(self):
        if not self.data:
            return 0.5
        val = self.data[-1]
        return min(1.0, max(0.0, val / 100.0))

    def get_data(self):
        return self.data

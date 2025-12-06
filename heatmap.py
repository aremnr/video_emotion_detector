from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen
import mpv
import random

class HeatmapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = []          # список значений (0..1)
        self.current_pos = 0    # текущая секунда

    def set_data(self, data):
        self.data = data
        self.update()

    def set_position(self, pos):
        self.current_pos = pos
        self.update()

    def value_to_color(self, v):
        """Преобразует значение 0..1 в цвет heatmap."""
        if v is None:
            return QColor(60, 60, 60)  # серый, если нет данных

        # Градиент: красный → жёлтый → зелёный
        r = int(255 * (1 - v))
        g = int(255 * v)
        b = 0
        return QColor(r, g, b)

    def paintEvent(self, event):
        if not self.data:
            return

        painter = QPainter(self)
        try:
            w = self.width()
            h = self.height()

            n = len(self.data)
            if n <= 0:
                return

            bar_w = w / n

            for i, v in enumerate(self.data):
                painter.setBrush(self.value_to_color(v))
                painter.setPen(Qt.PenStyle.NoPen)

                painter.drawRect(int(i * bar_w), 0, int(bar_w) + 1, h)

            # линия текущей позиции
            x = int(self.current_pos * bar_w)
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawLine(x, 0, x, h)

        finally:
            painter.end()
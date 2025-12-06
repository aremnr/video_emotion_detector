from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QApplication

def apply_light_theme(app: QApplication):
    # --- Общая палитра ---
    palette = QPalette()
    
    # Фон окон и виджетов
    palette.setColor(QPalette.ColorRole.Window, QColor(245, 245, 245))
    palette.setColor(QPalette.ColorRole.Base, QColor(245, 245, 245))  # текстовые поля, таблицы
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(255, 255, 255))
    
    # Цвет текста
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    
    # Кнопки
    palette.setColor(QPalette.ColorRole.Button, QColor(230, 240, 255))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 60, 180))
    
    # Выделение
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    
    app.setPalette(palette)
    
    # --- QSS для тонкой настройки ---
    app.setStyleSheet("""
        QWidget {
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
        }
        QPushButton {
            background-color: #e6f0ff;
            border: 1px solid #a0c0ff;
            border-radius: 4px;
            color: #0040c0;
            padding: 4px 10px;
        }
        QPushButton:hover {
            background-color: #cce0ff;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #c0d8ff;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #0078d7;
            width: 14px;
            border-radius: 7px;
        }
        QLabel {
            color: #000000;
        }
    """)

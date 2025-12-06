from ui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication
import sys
import threading

def backend_thread(connector):
    connector.scan_and_connect()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()

    # запускаем бэкенд в отдельном потоке
    t = threading.Thread(target=backend_thread, args=(w.connector,), daemon=True)
    t.start()

    sys.exit(app.exec())

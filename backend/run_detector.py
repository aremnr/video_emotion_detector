from src.sensor_event_handler import SensorEventHandler
from src.emotion_math_manager import EmotionMathManager
from src.sensor_connector import SensorConnector

if __name__ == "__main__":
    # Создаем менеджер математики и обработчик событий
    math_manager = EmotionMathManager()
    event_handler = SensorEventHandler(math_manager=math_manager)
    # Создаем и запускаем коннектор сенсоров
    connector = SensorConnector(event_handler=event_handler, math_manager=math_manager)
    try:
        connector.scan_and_connect()
    except Exception as err:
        print(f"Error: {err}")

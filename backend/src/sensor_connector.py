from neurosdk.scanner import Scanner
from neurosdk.cmn_types import SensorFamily, SensorCommand
from time import sleep
import keyboard

class SensorConnector:
    def __init__(self, event_handler, math_manager, known_busy_address="C8:E6:24:1C:EA:D8"):
        self.event_handler = event_handler
        self.math_manager = math_manager
        self.known_busy_address = known_busy_address
        self.scanner = Scanner([SensorFamily.LEBrainBit, SensorFamily.LEBrainBitBlack])
        #self.active_sensor = None # <-----------

    def scan_and_connect(self):
        self.scanner.sensorsChanged = self.event_handler.sensor_found
        self.scanner.start()
        print("Starting search for 25 sec...")
        sleep(10)
        self.scanner.stop()
        sensorsInfo = self.scanner.sensors()
        
        print("Stop resistance")
        for i in range(len(sensorsInfo)):
            if sensorsInfo[i].Address != self.known_busy_address:
                print(f"Skipping known busy device {sensorsInfo[i]}")
                continue
            current_sensor_info = sensorsInfo[i]
            sensor = self.scanner.create_sensor(current_sensor_info)
            print(f"Current connected device {current_sensor_info}")
            sensor.sensorStateChanged = self.event_handler.on_sensor_state_changed
            sensor.batteryChanged = self.event_handler.on_battery_changed
            sensor.signalDataReceived = self.event_handler.on_signal_received
            sensor.resistDataReceived = self.event_handler.on_resist_received
            sensor.exec_command(SensorCommand.StartResist)
            print("Start resistance")
            sleep(10)
            sensor.exec_command(SensorCommand.StopResist)
            if sensor.is_supported_command(SensorCommand.StartSignal):
                print("Нажмите Alt для СТАРТА сигнала...")
                keyboard.wait('alt')
                sensor.exec_command(SensorCommand.StartSignal)
                print("Start signal")
                self.math_manager.math.start_calibration()
                print("Нажмите Alt для ОСТАНОВКИ сигнала...")
                keyboard.wait('alt')
                sensor.exec_command(SensorCommand.StopSignal)
                print("Stop signal")
            sensor.disconnect()
            print("Disconnect from sensor")
            del sensor
        del self.scanner
        print('Remove scanner')

    # def start_signal_from_ui(self):
    #     """
    #     Запуск сигнала EEG с сенсора.
    #     Предполагается, что скан и подключение уже выполнены.
    #     """
    #     self._active_sensors = []
    #     sensors_info = self.scanner.sensors()
    #     for info in sensors_info:
    #         if info.Address != self.known_busy_address:
    #             continue
    #         sensor = self.scanner.create_sensor(info)
    #         sensor.sensorStateChanged = self.event_handler.on_sensor_state_changed
    #         sensor.batteryChanged = self.event_handler.on_battery_changed
    #         sensor.signalDataReceived = self.event_handler.on_signal_received
    #         sensor.resistDataReceived = self.event_handler.on_resist_received

    #         if sensor.is_supported_command(SensorCommand.StartSignal):
    #             sensor.exec_command(SensorCommand.StartSignal)
    #             print(f"EEG signal started on {info.Address}")
    #             self.math_manager.math.start_calibration()

    #         self._active_sensors.append(sensor)


    # def stop_signal_from_ui(self):
    #     """Остановить все активные сенсоры и отключить их"""
    #     if hasattr(self, "_active_sensors"):
    #         for sensor in self._active_sensors:
    #             try:
    #                 if sensor.is_supported_command(SensorCommand.StopSignal):
    #                     sensor.exec_command(SensorCommand.StopSignal)
    #                     print(f"EEG signal stopped on {sensor.name}")
    #                 sensor.disconnect()
    #             except Exception as e:
    #                 print(f"Error stopping sensor {sensor}: {e}")
    #         self._active_sensors = []


    # def scan_and_connect(self):
    #     self.scanner.sensorsChanged = self.event_handler.sensor_found
    #     self.scanner.start()
    #     print("Starting search for 25 sec...")
    #     sleep(5)
    #     self.scanner.stop()
    #     sensorsInfo = self.scanner.sensors()
    #     print("Stop resistance")

    #     for info in sensorsInfo:
    #         if info.Address != self.known_busy_address:
    #             continue
    #         self.active_sensor = self.scanner.create_sensor(info)
    #         print(f"Connected to {info.Address}")

    #         self.active_sensor.sensorStateChanged = self.event_handler.on_sensor_state_changed
    #         self.active_sensor.batteryChanged = self.event_handler.on_battery_changed
    #         self.active_sensor.signalDataReceived = self.event_handler.on_signal_received
    #         self.active_sensor.resistDataReceived = self.event_handler.on_resist_received

    #         # опционально: старт/стоп сопротивления, если нужно
    #         self.active_sensor.exec_command(SensorCommand.StartResist)
    #         sleep(10)
    #         self.active_sensor.exec_command(SensorCommand.StopResist)
    #     del self.scanner
    #     print("Scanner removed")

    # # методы UI для кнопки
    # def start_signal_from_ui(self):
    #     if self.active_sensor and self.active_sensor.is_supported_command(SensorCommand.StartSignal):
    #         self.active_sensor.exec_command(SensorCommand.StartSignal)
    #         print(f"EEG signal started on {self.active_sensor.name}")
    #         self.math_manager.math.start_calibration()

    # def stop_signal_from_ui(self):
    #     if self.active_sensor and self.active_sensor.is_supported_command(SensorCommand.StopSignal):
    #         self.active_sensor.exec_command(SensorCommand.StopSignal)
    #         print(f"EEG signal stopped on {self.active_sensor.name}")
    #         self.active_sensor.disconnect()
    #         self.active_sensor = None
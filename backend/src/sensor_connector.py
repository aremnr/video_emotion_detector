from neurosdk.scanner import Scanner
from neurosdk.cmn_types import SensorFamily, SensorCommand
from time import sleep
import keyboard

class SensorConnector:
    def __init__(self, event_handler, math_manager, known_busy_address="C8:E6:24:1C:EA:D8"):
        self.event_handler = event_handler
        self.math_manager = math_manager
        self.known_busy_address = known_busy_address
        #self.scanner = Scanner([SensorFamily.LEBrainBit, SensorFamily.LEBrainBitBlack])
        self.scanner = None
        self.active_sensor = None

    def scan_and_connect(self):
        if self.scanner is None:
            self.scanner = Scanner([SensorFamily.LEBrainBit, SensorFamily.LEBrainBitBlack])
            self.scanner.sensorsChanged = self.event_handler.sensor_found
        self.scanner.start()
        print("Starting search for sensors...")
        sleep(10)
        self.scanner.stop()
        sensorsInfo = self.scanner.sensors()
        print("Search finished")

        for info in sensorsInfo:
            if info.Address != self.known_busy_address:
                continue
            if self.active_sensor is None:
                self.active_sensor = self.scanner.create_sensor(info)
                print(f"Connected to {info.Address}")

                self.active_sensor.sensorStateChanged = self.event_handler.on_sensor_state_changed
                self.active_sensor.batteryChanged = self.event_handler.on_battery_changed
                self.active_sensor.signalDataReceived = self.event_handler.on_signal_received
                self.active_sensor.resistDataReceived = self.event_handler.on_resist_received

    def start_signal_from_ui(self):
        if self.active_sensor is None:
            print("No active sensor, scanning...")
            self.scan_and_connect()
            if self.active_sensor is None:
                print("Cannot find sensor")
                return

        if not self.active_sensor.is_supported_command(SensorCommand.StartSignal):
            print("StartSignal not supported")
            return

        try:
            self.active_sensor.exec_command(SensorCommand.StartSignal)
            print(f"EEG signal started on {self.active_sensor.name}")
            self.math_manager.math.start_calibration()
        except Exception as e:
            print(f"Failed to start signal: {e}")

    def stop_signal_from_ui(self):
        if self.active_sensor is None:
            print("No active sensor to stop session")
            return
        try:
            if self.active_sensor.is_supported_command(SensorCommand.StopSignal):
                self.active_sensor.exec_command(SensorCommand.StopSignal)
                print(f"EEG signal stopped on {self.active_sensor.name}")
            # Не удаляем сразу сенсор, можно оставить подключенным
        except Exception as e:
            print(f"Error during stop: {e}")

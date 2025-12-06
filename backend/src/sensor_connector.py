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

    def scan_and_connect(self):
        self.scanner.sensorsChanged = self.event_handler.sensor_found
        self.scanner.start()
        print("Starting search for 25 sec...")
        sleep(5)
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

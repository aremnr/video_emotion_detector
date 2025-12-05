
"""
Session wrapper for BrainBit / BrainBitBlack that allows start/pause/resume/stop
control and records averaged mental values every N seconds. The sensor keeps
streaming at its own sampling rate; we only average and store values at the
requested interval.
"""

import threading
import time
from typing import List, Optional, Tuple

from neurosdk.scanner import Scanner
from neurosdk.cmn_types import SensorFamily, SensorCommand
from em_st_artifacts.utils import lib_settings
from em_st_artifacts.utils import support_classes
from em_st_artifacts import emotional_math


class NeuroSession:
    """
    Управляет жизненным циклом сессии:
    - initialize(): поиск и подключение сенсора, подготовка math.
    - start(): запуск сигнала и калибровки, очистка накопителей.
    - pause()/resume(): приостановка/возобновление записи.
    - stop(): остановка сигнала, отключение сенсора, возврат накопленных данных.

    Значения сохраняются как средние за sample_interval секунд, при этом сам
    датчик продолжает работать на своей частоте дискретизации.
    """

    def __init__(self, sample_interval: float = 10.0, search_timeout: float = 5.0):
        self.sample_interval = sample_interval
        self.search_timeout = search_timeout

        self._scanner: Optional[Scanner] = None
        self._sensor = None
        self._math: Optional[emotional_math.EmotionalMath] = None

        self._stop_event = threading.Event()
        self._paused_event = threading.Event()
        self._lock = threading.Lock()

        self._samples: List[Tuple[float, float]] = []
        self._acc_sum = 0.0
        self._acc_count = 0
        self._start_time: Optional[float] = None
        self._last_save_time: Optional[float] = None
        self._initialized = False

        # math settings (can be parameterized later)
        self.calibration_length = 6
        self.nwins_skip_after_artifact = 10

    # ------------------------------------------------------------------ public
    def initialize(self):
        """Инициализация: поиск сенсора, подключение, настройка math."""
        if self._initialized:
            return

        self._scanner = Scanner([SensorFamily.LEBrainBit, SensorFamily.LEBrainBitBlack])
        self._scanner.sensorsChanged = self._on_sensor_found
        self._scanner.start()
        print(f"Starting search for {self.search_timeout} sec...")
        time.sleep(self.search_timeout)
        self._scanner.stop()

        sensors_info = self._scanner.sensors()
        if not sensors_info:
            raise RuntimeError("No sensors found")

        current_sensor_info = sensors_info[0]
        self._sensor = self._scanner.create_sensor(current_sensor_info)
        print(f"Connected to device {current_sensor_info}")

        self._sensor.sensorStateChanged = self._on_sensor_state_changed
        self._sensor.batteryChanged = self._on_battery_changed
        self._sensor.signalDataReceived = self._on_signal_received
        self._sensor.resistDataReceived = self._on_resist_received

        mls = lib_settings.MathLibSetting(
            sampling_rate=250,
            process_win_freq=25,
            n_first_sec_skipped=4,
            fft_window=1000,
            bipolar_mode=True,
            channels_number=4,
            channel_for_analysis=0,
        )
        ads = lib_settings.ArtifactDetectSetting(
            art_bord=110,
            allowed_percent_artpoints=70,
            raw_betap_limit=800_000,
            global_artwin_sec=4,
            num_wins_for_quality_avg=125,
            hamming_win_spectrum=True,
            hanning_win_spectrum=False,
            total_pow_border=100,
            spect_art_by_totalp=True,
        )
        mss = lib_settings.MentalAndSpectralSetting(
            n_sec_for_averaging=2,
            n_sec_for_instant_estimation=4,
        )

        self._math = emotional_math.EmotionalMath(mls, ads, mss)
        self._math.set_calibration_length(self.calibration_length)
        self._math.set_mental_estimation_mode(False)
        self._math.set_skip_wins_after_artifact(self.nwins_skip_after_artifact)
        self._math.set_zero_spect_waves(True, 0, 1, 1, 1, 0)
        self._math.set_spect_normalization_by_bands_width(True)

        self._initialized = True

    def start(self):
        """Запуск сигнала и калибровки, очистка накопителей."""
        if not self._initialized:
            raise RuntimeError("Call initialize() first")

        self._stop_event.clear()
        self._paused_event.clear()
        self._samples.clear()
        self._acc_sum = 0.0
        self._acc_count = 0
        now = time.monotonic()
        self._start_time = now
        self._last_save_time = now

        if self._sensor.is_supported_command(SensorCommand.StartSignal):
            self._sensor.exec_command(SensorCommand.StartSignal)
            print("Start signal")
            self._math.start_calibration()
        else:
            raise RuntimeError("StartSignal is not supported by sensor")

    def pause(self):
        """Поставить на паузу: данные не пишутся, таймер не тикает."""
        self._paused_event.set()

    def resume(self):
        """Снять паузу и сбросить точку отсчета интервала."""
        if self._paused_event.is_set():
            self._paused_event.clear()
            with self._lock:
                self._acc_sum = 0.0
                self._acc_count = 0
                self._last_save_time = time.monotonic()

    def stop(self):
        """
        Полная остановка: остановить сигнал, отключить сенсор, вернуть накопленные
        данные (включая незаписанный хвост).
        """
        self._stop_event.set()

        if self._sensor and self._sensor.is_supported_command(SensorCommand.StopSignal):
            self._sensor.exec_command(SensorCommand.StopSignal)
            print("Stop signal")

        if self._sensor:
            self._sensor.disconnect()
            print("Disconnect from sensor")

        samples = self._flush_final()
        self._cleanup()
        return samples

    # ----------------------------------------------------------------- internal
    def _on_sensor_found(self, scanner, sensors):
        for sensor_info in sensors:
            print(f"Sensor found: {sensor_info}")

    def _on_sensor_state_changed(self, sensor, state):
        print(f"Sensor {sensor.name} is {state}")

    def _on_battery_changed(self, sensor, battery):
        print(f"Battery: {battery}")

    def _on_resist_received(self, sensor, data):
        # Placeholder: resistance handling can be added if needed
        return

    def _on_signal_received(self, sensor, data):
        """Главный колбек: получает сырые данные, считает эмоции, усредняет."""
        if self._stop_event.is_set() or self._math is None:
            return
        if self._paused_event.is_set():
            return

        raw_channels = []
        for sample in data:
            left_bipolar = sample.T3 - sample.O1
            right_bipolar = sample.T4 - sample.O2
            raw_channels.append(support_classes.RawChannels(left_bipolar, right_bipolar))

        self._math.push_bipolars(raw_channels)
        self._math.process_data_arr()

        if not self._math.calibration_finished():
            return

        mental_data = self._math.read_mental_data_arr()
        if not mental_data:
            return

        value = mental_data[0].rel_relaxation
        self._record_value(value)

    def _record_value(self, value: float):
        """Накапливает значения и записывает среднее каждые sample_interval сек."""
        now = time.monotonic()
        if self._last_save_time is None or self._start_time is None:
            return

        with self._lock:
            self._acc_sum += value
            self._acc_count += 1

            if now - self._last_save_time >= self.sample_interval and self._acc_count > 0:
                avg = self._acc_sum / self._acc_count
                t_rel = now - self._start_time
                self._samples.append((t_rel, avg))
                self._acc_sum = 0.0
                self._acc_count = 0
                self._last_save_time = now

    def _flush_final(self) -> List[Tuple[float, float]]:
        """Сохраняет хвост накопленных значений при остановке."""
        now = time.monotonic()
        if self._start_time is None:
            return []

        with self._lock:
            if self._acc_count > 0:
                avg = self._acc_sum / self._acc_count
                t_rel = now - self._start_time
                self._samples.append((t_rel, avg))
                self._acc_sum = 0.0
                self._acc_count = 0
                self._last_save_time = now

            return list(self._samples)

    def _cleanup(self):
        """Освобождает ресурсы сенсора и сканера."""
        if self._sensor:
            del self._sensor
            self._sensor = None
        if self._math:
            del self._math
            self._math = None
        if self._scanner:
            del self._scanner
            self._scanner = None
        self._initialized = False


def create_session(sample_interval: float = 10.0, search_timeout: float = 5.0) -> NeuroSession:
    """Удобный фабричный метод для внешнего кода."""
    return NeuroSession(sample_interval=sample_interval, search_timeout=search_timeout)


if __name__ == "__main__":
    """
    Простейшая заглушка для ручного запуска файла:
    - инициализация,
    - старт,
    - пауза на 2 секунды,
    - возобновление,
    - остановка и выгрузка всех накопленных точек в XML.
    Формат XML: <session><row time="X.XX"><rel_relaxation>Y</rel_relaxation></row>...</session>
    """
    import xml.etree.ElementTree as ET

    session = create_session(sample_interval=1.0, search_timeout=5.0)
    try:
        session.initialize()
        session.start()
        time.sleep(10)          # немного поработать
        session.pause()
        print("Paused for 2 seconds")
        time.sleep(2)          # пауза
        session.resume()
        print("Resumed")
        time.sleep(10)          # ещё немного поработать
        data = session.stop()
    except Exception as exc:
        print(f"Demo run failed: {exc}")
        data = []

    root = ET.Element("session")
    for t_rel, val in data:
        row = ET.SubElement(root, "row", time=f"{t_rel:.2f}")
        ET.SubElement(row, "rel_relaxation").text = f"{val:.6f}"

    xml_bytes = ET.tostring(root, encoding="utf-8")
    with open("demo_output.xml", "wb") as f:
        f.write(xml_bytes)

    print(f"Saved {len(data)} rows to demo_output.xml")

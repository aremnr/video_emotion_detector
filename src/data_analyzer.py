import time
from datetime import datetime, timedelta

class DataAnalyzer:
    def __init__(self, tick_interval=5, reaction_delay=2):
        self.start_time = None
        self.data_buffer = []
        self.averaged_data = []
        self.peaks = []
        self.tick_interval = tick_interval
        self.reaction_delay = reaction_delay
        self.last_tick_time = None

    def start(self):
        self.start_time = time.time()
        self.last_tick_time = self.start_time

    def add_data(self, value):
        now = time.time()
        if self.start_time is None:
            self.start()
        self.data_buffer.append((now, value))
        # Check if tick interval passed
        if len(self.data_buffer) >= self.tick_interval:
            self._process_tick()

    def _process_tick(self):
        # Apply reaction delay
        delayed_time = self.data_buffer[-1][0] + self.reaction_delay
        avg_value = sum([v for t, v in self.data_buffer]) / len(self.data_buffer)
        rel_time = delayed_time - self.start_time
        msk_time = datetime.fromtimestamp(delayed_time) + timedelta(hours=3)  # MSK = UTC+3
        self.averaged_data.append((rel_time, msk_time, avg_value))
        print(f"[MSK: {msk_time.strftime('%H:%M:%S')}] [Rel: {rel_time:.2f}s] Avg: {avg_value:.3f}")
        self.data_buffer.clear()
        self.last_tick_time = delayed_time
        self._find_peaks()

    def _find_peaks(self):
        # Simple peak detection: value > previous and next
        if len(self.averaged_data) < 3:
            return
        prev = self.averaged_data[-3][2]
        curr = self.averaged_data[-2][2]
        next_ = self.averaged_data[-1][2]
        if curr > prev and curr > next_:
            peak_time = self.averaged_data[-2][1]
            print(f"Peak detected at {peak_time.strftime('%H:%M:%S')}, value: {curr:.3f}")
            self.peaks.append((self.averaged_data[-2][0], peak_time, curr))

    def print_peaks(self):
        print("\nDetected peaks:")
        for rel_time, msk_time, value in self.peaks:
            print(f"[MSK: {msk_time.strftime('%H:%M:%S')}] [Rel: {rel_time:.2f}s] Value: {value:.3f}")

class SensorEventHandler:
    def __init__(self, math_manager=None):
        self.math_manager = math_manager

    def sensor_found(self, scanner, sensors):
        for index in range(len(sensors)):
            print('Sensor found: %s' % sensors[index])

    def on_sensor_state_changed(self, sensor, state):
        print('Sensor {0} is {1}'.format(sensor.name, state))

    def on_battery_changed(self, sensor, battery):
        print('Battery: {0}'.format(battery))

    def on_signal_received(self, sensor, data):
        if self.math_manager is None:
            return
        raw_channels = []
        for sample in data:
            left_bipolar = sample.T3 - sample.O1
            right_bipolar = sample.T4 - sample.O2
            raw_channels.append(self.math_manager.create_raw_channels(left_bipolar, right_bipolar))
        self.math_manager.push_bipolars(raw_channels)
        self.math_manager.process_data_arr()
        if not self.math_manager.calibration_finished():
            calib_percent = self.math_manager.get_calibration_percents()
            bar_length = int(calib_percent / 5)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"Calibration: |{bar}| {calib_percent:.1f}%")
        else:
            mental_data = self.math_manager.read_mental_data_arr()
            if len(mental_data) > 0:
                r_relaxation = mental_data[0].rel_relaxation
                r_attention = mental_data[0].rel_attention
                relaxation = mental_data[0].inst_relaxation
                attention = mental_data[0].inst_attention
                bar_length = int(r_relaxation / 5)
                bar_length_2 = int(r_attention / 5)
                bar_length_3 = int(attention / 5)
                bar_length_4 = int(relaxation / 5)
                bar = '█' * bar_length + '░' * (20 - bar_length)
                bar2 = '█' * bar_length_2 + '░' * (20 - bar_length_2)
                bar3 = '█' * bar_length_3 + '░' * (20 - bar_length_3)
                bar4 = '█' * bar_length_4 + '░' * (20 - bar_length_4)
                print(f"R_Attention:   |{bar2}| {r_attention:.1f}% R_Relaxation:  |{bar}| {r_relaxation:.1f}")
            spectral_data = self.math_manager.read_spectral_data_percents_arr()# % Relaxation:   |{bar4}| {relaxation:.1f}% Attention:    |{bar3}| {attention:.1f}%
            if len(spectral_data) > 0:
                pass

    def on_resist_received(self, sensor, data):
        pass

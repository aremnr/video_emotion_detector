class SensorEventHandler:
    def __init__(self, math_manager=None):
        self.math_manager = math_manager
        self.on_engagement_update = None

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
            spectral_data = self.math_manager.read_spectral_data_percents_arr()
            if len(spectral_data) > 0 and len(mental_data) > 0:
                # 1. Средние значения по mental_data
                r_relaxation = mental_data[0].rel_relaxation
                r_attention = mental_data[0].rel_attention
                # 2. Средние значения по spectral_data
                alpha = getattr(spectral_data[0], 'alpha', None)
                beta = getattr(spectral_data[0], 'beta', None)
                if alpha is not None and beta is not None:
                    total = alpha + beta if (alpha + beta) != 0 else 1
                    relaxations = (alpha / total * 100)
                    concentrations = (beta / total * 100)
                if relaxations and concentrations:
                    relaxation_avg = (relaxations + r_relaxation) / 2
                    concentration_avg = (concentrations + r_attention) / 2
                    engagement = concentration_avg / 100  # 0..1
                    # если UI привязан → посылаем ему
                    if self.on_engagement_update:
                        self.on_engagement_update(engagement)
                    bar_relax = '█' * int(relaxation_avg / 5) + '░' * (20 - int(relaxation_avg / 5))
                    bar_conc = '█' * int(concentration_avg / 5) + '░' * (20 - int(concentration_avg / 5))
                    print(f"Mean Relaxation:    |{bar_relax}| {relaxation_avg:.1f}%\tMean Concentration: |{bar_conc}| {concentration_avg:.1f}%")

    def on_resist_received(self, sensor, data):
        print(f"Resistance data received from {sensor}: {data}")
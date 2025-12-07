from collections import deque
import statistics

class SensorEventHandler:
    def __init__(self, math_manager=None, on_engagement_update=None):
        self.math_manager = math_manager
        self.on_engagement_update = None
        self.on_engagement_update = on_engagement_update
        self.window_size = 20 
        
        # Буферы для хранения истории спектров
        self.history = {
            'alpha': deque(maxlen=self.window_size),
            'beta':  deque(maxlen=self.window_size),
            'theta': deque(maxlen=self.window_size)
        }

        # Для финального сглаживания (0.05 = очень плавно, 0.2 = отзывчиво)
        self.smooth_factor = 0.05 
        
        self.last_relax = 0.0
        self.last_focus = 0.0

    def sensor_found(self, scanner, sensors):
        for index in range(len(sensors)):
            print('Sensor found: %s' % sensors[index])

    def on_sensor_state_changed(self, sensor, state):
        print('Sensor {0} is {1}'.format(sensor.name, state))

    def on_battery_changed(self, sensor, battery):
        print('Battery: {0}'.format(battery))

    def _update_buffers(self, spectral):
        """Добавляем новые данные в историю"""
        # Используем max(..., 0.001) чтобы избежать деления на ноль в будущем
        self.history['alpha'].append(max(getattr(spectral, 'alpha', 0), 0.001))
        self.history['beta'].append(max(getattr(spectral, 'beta', 0), 0.001))
        self.history['theta'].append(max(getattr(spectral, 'theta', 0), 0.001))

    def _get_averaged_spectrum(self):
        """Получаем средние значения спектров за последние N пакетов"""
        if len(self.history['alpha']) == 0:
            return 0, 0, 0
            
        avg_alpha = statistics.mean(self.history['alpha'])
        avg_beta = statistics.mean(self.history['beta'])
        avg_theta = statistics.mean(self.history['theta'])
        return avg_alpha, avg_beta, avg_theta

    def on_signal_received(self, sensor, data):
        if self.math_manager is None:
            return

        # --- [Обработка сырых данных как раньше] ---
        raw_channels = []
        for sample in data:
            left_bipolar = sample.T3 - sample.O1
            right_bipolar = sample.T4 - sample.O2
            raw_channels.append(self.math_manager.create_raw_channels(left_bipolar, right_bipolar))
        self.math_manager.push_bipolars(raw_channels)
        self.math_manager.process_data_arr()

        if not self.math_manager.calibration_finished():
            # ... (код калибровки без изменений)
            pass
        else:
            spectral_data_arr = self.math_manager.read_spectral_data_percents_arr()
            
            # Проверка на пустые данные
            if not spectral_data_arr:
                return

            # Берем САМЫЙ СВЕЖИЙ пакет
            latest_spectral = spectral_data_arr[-1]

            # 1. ЗАКИДЫВАЕМ В БУФЕР (накапливаем историю)
            self._update_buffers(latest_spectral)

            # 2. СЧИТАЕМ ПО СРЕДНЕМУ (стабильные входные данные)
            avg_alpha, avg_beta, avg_theta = self._get_averaged_spectrum()
            
            # Если буфер еще не заполнился, можно ждать или считать как есть
            if avg_alpha == 0: return 

            # --- СТАБИЛЬНЫЕ ФОРМУЛЫ ---
            # Расслабление: Alpha доминирует
            target_relax = (avg_alpha / (avg_alpha + avg_beta + avg_theta)) * 100
            
            # Концентрация: Beta доминирует над Theta (сонливость) и Alpha (расслабление)
            # Умножаем Beta на коэффициент (например 1.5), чтобы легче достичь 100%
            target_focus = (avg_beta / (avg_theta + avg_alpha + avg_beta)) * 100
            
            # Ограничиваем диапазоном 0-100
            target_relax = min(100, max(0, target_relax))
            target_focus = min(100, max(0, target_focus))

            # 3. ФИНАЛЬНОЕ СГЛАЖИВАНИЕ (Low-pass filter)
            # Плавный переход от старого значения к новому
            self.last_relax = self.last_relax + self.smooth_factor * (target_relax - self.last_relax)
            self.last_focus = self.last_focus + self.smooth_factor * (target_focus - self.last_focus)

            # 4. ВЫВОД
            engagement = self.last_focus / 100.0
            
            if self.on_engagement_update:
                self.on_engagement_update(
                    engagement, 
                    calib_percent=100, 
                    relaxation=self.last_relax, 
                    focus=self.last_focus, 
                    artifact=0
                )

            # Визуализация в консоль (чтобы не спамило, можно добавить условие времени)
            self._print_bar_chart(self.last_relax, self.last_focus)

    def _print_bar_chart(self, r, f):
        # Очистка строки через \r
        r_bar = '█' * int(r / 5) + '░' * (20 - int(r / 5))
        f_bar = '█' * int(f / 5) + '░' * (20 - int(f / 5))
        print(f"\rREL: {r_bar} {r:5.1f}% | FOC: {f_bar} {f:5.1f}%", end='')

    def on_resist_received(self, sensor, data):
        print(f"Resistance data received from {sensor}: {data}")
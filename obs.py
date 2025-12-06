import obspython as obs
import threading
import time
import subprocess
import os
import datetime
import shutil
import sys
import traceback  # Важно для отладки ошибок в потоке

# --- Импорт neurosdk ---
try:
    from neurosdk.scanner import Scanner
    from neurosdk.cmn_types import *
    from neurosdk.brainbit_sensor import BrainBitSensor
    from em_st_artifacts.emotional_math import EmotionalMath
    from em_st_artifacts.utils.lib_settings import (
        MathLibSetting, ArtifactDetectSetting,
        MentalAndSpectralSetting, ShortArtifactDetectSetting
    )
    from em_st_artifacts.utils.support_classes import RawChannels
    SDK_LOADED = True
except ImportError as e:
    SDK_LOADED = False
    print(f"[BrainBit ERROR] Failed to import NeuroSDK: {e}")

# --- Глобальные переменные ---
scanner = None
sensor = None
math_lib = None
worker_thread = None
is_running = False

session_folder = ""
log_file_path = ""
analysis_script_path = ""

file_lock = threading.Lock()
file_handle = None
packet_counter = 0

# -------------------------------------------------
# Логирование в консоль OBS
# -------------------------------------------------
def log_obs(message):
    print(f"[BrainBit] {message}")

# -------------------------------------------------
# Создание папки сессии (Надежный метод)
# -------------------------------------------------
def create_session_folder():
    global session_folder
    # Используем стандартную папку Видео пользователя, чтобы не зависеть от OBS API в момент старта
    base_dir = os.path.join(os.path.expanduser("~"), "Videos", "BrainBit_Sessions")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"Session_{timestamp}"
    
    session_folder = os.path.join(base_dir, folder_name)
    try:
        os.makedirs(session_folder, exist_ok=True)
        log_obs(f"Session folder created: {session_folder}")
        return True
    except Exception as e:
        log_obs(f"ERROR creating folder: {e}")
        return False

# -------------------------------------------------
# CSV лог
# -------------------------------------------------
def init_log_file():
    global log_file_path, file_handle, session_folder
    if not session_folder:
        log_obs("Error: Session folder is not set, cannot create CSV.")
        return False
        
    log_file_path = os.path.join(session_folder, "brainbit_log.csv")
    try:
        file_handle = open(log_file_path, "w", encoding="utf-8")
        # Пишем заголовок
        file_handle.write("Timestamp_ms,Relaxation,Focus,Artifact,O1,O2,T3,T4\n")
        file_handle.flush()
        log_obs(f"Log file initialized: {log_file_path}")
        return True
    except Exception as e:
        log_obs(f"ERROR creating CSV file: {e}")
        return False

def close_log_file():
    global file_handle
    with file_lock:
        if file_handle:
            try:
                file_handle.close()
            except:
                pass
            file_handle = None

def log_data(relax, focus, is_artifact, raw_values=None):
    if not file_handle:
        return
    
    timestamp_ms = int(time.time() * 1000)
    o1, o2, t3, t4 = (raw_values or (0, 0, 0, 0))
    # 1 - артефакт есть, 0 - нет
    art_val = 1 if is_artifact else 0
    
    line = f"{timestamp_ms},{relax},{focus},{art_val},{o1:.6f},{o2:.6f},{t3:.6f},{t4:.6f}\n"
    
    try:
        with file_lock:
            if file_handle:
                file_handle.write(line)
                file_handle.flush() # Принудительная запись на диск
    except Exception as e:
        print(f"Write error: {e}")

# -------------------------------------------------
# Emotional Math Setup
# -------------------------------------------------
def create_math_instance():
    # Настройки из вашего примера
    mls = MathLibSetting(sampling_rate=250, process_win_freq=25, n_first_sec_skipped=4,
                         fft_window=1000, bipolar_mode=True, squared_spectrum=True,
                         channels_number=4, channel_for_analysis=0)
                         
    ads = ArtifactDetectSetting(art_bord=110, allowed_percent_artpoints=70,
                                raw_betap_limit=800000, global_artwin_sec=4,
                                num_wins_for_quality_avg=125, hamming_win_spectrum=True,
                                hanning_win_spectrum=False, total_pow_border=400000000,
                                spect_art_by_totalp=True)
                                
    sads = ShortArtifactDetectSetting(ampl_art_detect_win_size=200,
                                      ampl_art_zerod_area=200,
                                      ampl_art_extremum_border=25)
                                      
    mss = MentalAndSpectralSetting(n_sec_for_averaging=2,
                                   n_sec_for_instant_estimation=4)
                                   
    math = EmotionalMath(mls, ads, sads, mss)
    math.set_calibration_length(6)
    math.set_mental_estimation_mode(False)
    math.set_skip_wins_after_artifact(10)
    math.set_zero_spect_waves(True, 0, 1, 1, 1, 0)
    math.set_spect_normalization_by_bands_width(True)
    return math

# -------------------------------------------------
# Обработка данных с BrainBit
# -------------------------------------------------
def on_signal_received(sensor, data):
    global math_lib, packet_counter
    
    # Защита от вызова до инициализации
    if math_lib is None: 
        return
    if not data:
        return

    try:
        raw_channels = []
        last_o1 = last_o2 = last_t3 = last_t4 = 0

        # Конвертация в биполярный режим согласно документации
        for sample in data:
            # T3-O1 и T4-O2 как в примере документации
            left_bipolar = sample.T3 - sample.O1
            right_bipolar = sample.T4 - sample.O2
            raw_channels.append(RawChannels(left_bipolar, right_bipolar))
            
            # Сохраняем последние сырые значения для лога
            last_o1, last_o2, last_t3, last_t4 = sample.O1, sample.O2, sample.T3, sample.T4

        # Отправка в библиотеку
        math_lib.push_data(raw_channels)
        math_lib.process_data_arr()
        
        is_artifacted = math_lib.is_artifacted_sequence()

        relax = 0
        focus = 0
        
        # Чтение ментальных уровней, если калибровка прошла
        if math_lib.calibration_finished():
            mental_data = math_lib.read_mental_data_arr()
            if mental_data:
                last_mental = mental_data[-1]
                relax = int(last_mental.Rel_Relaxation)
                focus = int(last_mental.Rel_Attention)

        log_data(relax, focus, is_artifacted, (last_o1, last_o2, last_t3, last_t4))
        
        packet_counter += 1
        if packet_counter % 250 == 0: # Лог в консоль раз в секунду (примерно)
            print(f"Stats: Relax={relax}, Focus={focus}, Artifact={is_artifacted}")
            
    except Exception as e:
        print(f"Error in signal callback: {e}")

def on_sensor_state_changed(sensor, state):
    log_obs(f"Sensor State Changed: {state}")

def on_battery_changed(sensor, battery):
    pass

# -------------------------------------------------
# Рабочий поток (Main Logic)
# -------------------------------------------------
def brainbit_worker():
    global scanner, sensor, math_lib, is_running, packet_counter

    log_obs("Worker thread started...")
    
    # 1. Создаем CSV
    if not init_log_file():
        log_obs("Failed to init log file. Aborting worker.")
        return

    packet_counter = 0

    try:
        # 2. Поиск устройства
        log_obs("Scanning for BrainBit...")
        scanner = Scanner([SensorFamily.LEBrainBit])
        scanner.start()
        time.sleep(5) # Ждем 5 секунд
        scanner.stop()

        sensors = scanner.sensors()
        if not sensors:
            log_obs("No sensors found! Check if device is ON.")
            close_log_file()
            return

        sensor_info = sensors[0]
        log_obs(f"Found sensor: {sensor_info.Name} ({sensor_info.Address})")

        # 3. Подключение
        sensor = scanner.create_sensor(sensor_info)
        sensor.sensorStateChanged = on_sensor_state_changed
        sensor.batteryChanged = on_battery_changed
        
        # Подключение происходит автоматически при create_sensor, но проверим
        if sensor.state == SensorState.StateOutOfRange:
             log_obs("Sensor created but OutOfRange. Trying connect...")
             sensor.connect()

        # 4. Инициализация математики
        log_obs("Initializing Math Lib...")
        math_lib = create_math_instance()
        math_lib.start_calibration()

        # 5. Старт сигнала
        # Важно: сначала подписываемся, потом запускаем команду
        sensor.signalDataReceived = on_signal_received
        sensor.exec_command(SensorCommand.CommandStartSignal)
        log_obs("Signal command sent. Waiting for data...")

        # --- Цикл ожидания данных ---
        # Ждем первых пакетов
        start_wait = time.time()
        while packet_counter == 0 and is_running:
            time.sleep(0.1)
            if time.time() - start_wait > 10:
                log_obs("WARNING: No data received in 10 seconds.")
                break
        
        if packet_counter > 0:
            log_obs("Data flow established!")

        # Основной цикл жизни потока
        while is_running:
            time.sleep(0.5)

    except Exception as e:
        log_obs(f"CRITICAL WORKER ERROR: {traceback.format_exc()}")
    finally:
        log_obs("Cleaning up BrainBit resources...")
        cleanup_brainbit()

def cleanup_brainbit():
    global sensor, scanner, math_lib
    if sensor:
        try:
            sensor.exec_command(SensorCommand.CommandStopSignal)
            # Удаляем колбэк перед отключением
            sensor.signalDataReceived = None
            sensor.disconnect()
            log_obs("Sensor disconnected.")
        except Exception as e:
            log_obs(f"Error disconnecting sensor: {e}")
        sensor = None
        
    if scanner:
        del scanner
        scanner = None
        
    if math_lib:
        del math_lib
        math_lib = None
        
    close_log_file()
    log_obs("Cleanup finished.")

# -------------------------------------------------
# Старт / стоп
# -------------------------------------------------
def start_processing():
    global is_running, worker_thread
    
    if not SDK_LOADED:
        log_obs("Cannot start: SDK not loaded.")
        return

    if is_running:
        return
        
    if create_session_folder():
        is_running = True
        worker_thread = threading.Thread(target=brainbit_worker, daemon=True)
        worker_thread.start()
    else:
        log_obs("Failed to start: Could not create session folder.")

def stop_processing():
    global is_running, worker_thread
    is_running = False
    
    if worker_thread and worker_thread.is_alive():
        log_obs("Waiting for thread to finish...")
        worker_thread.join(timeout=3.0)
    
    log_obs("Processing stopped.")
    
    # Попытка перенести видео
    move_last_video_to_session()
    run_analysis_script()

# -------------------------------------------------
# Перемещение видео
# -------------------------------------------------
def move_last_video_to_session():
    if not session_folder or not os.path.exists(session_folder):
        return
        
    # OBS может еще писать файл, даем небольшую задержку
    time.sleep(2.0)
    
    last_rec = obs.obs_frontend_get_last_recording()
    if not last_rec or not os.path.exists(last_rec):
        log_obs("Last recording not found via OBS API.")
        return

    filename = os.path.basename(last_rec)
    dest = os.path.join(session_folder, filename)
    
    log_obs(f"Moving video from {last_rec} to {dest}")
    
    # Пробуем переместить несколько раз (файл может быть занят)
    for i in range(10):
        try:
            shutil.move(last_rec, dest)
            log_obs("Video moved successfully.")
            return
        except Exception as e:
            log_obs(f"Move attempt {i+1} failed: {e}")
            time.sleep(1.0)

def run_analysis_script():
    if not analysis_script_path or not os.path.exists(analysis_script_path):
        return
    if not log_file_path or not os.path.exists(log_file_path):
        return
    try:
        py_exe = sys.executable # Используем текущий интерпретатор OBS
        # Если это embeddable python, иногда sys.executable урезан, можно попробовать найти системный
        if "python" not in os.path.basename(py_exe).lower():
             py_exe = "python"
             
        subprocess.Popen([py_exe, analysis_script_path, log_file_path])
        log_obs("Analysis script started.")
    except Exception as e:
        log_obs(f"Failed to run analysis script: {e}")

# -------------------------------------------------
# OBS callbacks
# -------------------------------------------------
def on_event(event):
    if event == obs.OBS_FRONTEND_EVENT_RECORDING_STARTED:
        log_obs("Recording Started Event received.")
        start_processing()
    elif event == obs.OBS_FRONTEND_EVENT_RECORDING_STOPPED:
        log_obs("Recording Stopped Event received.")
        stop_processing()
    elif event == obs.OBS_FRONTEND_EVENT_EXIT:
        stop_processing()

def script_description():
    return "BrainBit Logger (Fixed)\nLogs BrainBit data to CSV in ~/Videos/BrainBit_Sessions/"

def script_properties():
    props = obs.obs_properties_create()
    obs.obs_properties_add_path(props, "analysis_script", "Analysis Script (.py)", obs.OBS_PATH_FILE, "*.py", None)
    return props

def script_update(settings):
    global analysis_script_path
    analysis_script_path = obs.obs_data_get_string(settings, "analysis_script")

def script_load(settings):
    obs.obs_frontend_add_event_callback(on_event)
    log_obs("Script loaded.")

def script_unload():
    stop_processing()
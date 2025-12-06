import obspython as obs
import threading
import time
import subprocess 
import os 

# --- ИМПОРТЫ NEUROSDK / БИБЛИОТЕКИ ЭМОЦИЙ ---
from neurosdk.scanner import Scanner
from neurosdk.cmn_types import *
from em_st_artifacts.emotional_math import EmotionalMath
from em_st_artifacts.utils.lib_settings import (MathLibSetting, ArtifactDetectSetting, MentalAndSpectralSetting)
from em_st_artifacts.utils.support_classes import RawChannels


# --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
scanner = None
sensor = None
math_lib = None
worker_thread = None
is_running = False
source_name = "" # Имя текстового источника в OBS

# Переменные для записи в файл
file_lock = threading.Lock()        # Блокировка для потокобезопасной записи
log_file_path = ""                  # Путь к файлу для записи данных (из настроек)
file_handle = None                  # Дескриптор открытого файла

# Переменная для пути к скрипту анализа
analysis_script_path = "" 

# --- Настройки для Emotional Math (без изменений) ---
def create_math_instance():
    # Настройки
    mls = MathLibSetting(sampling_rate=250,
                         process_win_freq=25,
                         n_first_sec_skipped=4,
                         fft_window=1000,
                         bipolar_mode=True,
                         squared_spectrum=True,
                         channels_number=4,
                         channel_for_analysis=0)

    ads = ArtifactDetectSetting(art_bord=110, allowed_percent_artpoints=70, raw_betap_limit=800_000,
                                global_artwin_sec=4, num_wins_for_quality_avg=125,
                                hamming_win_spectrum=True, hanning_win_spectrum=False,
                                total_pow_border=400_000_000, spect_art_by_totalp=True)

    mss = MentalAndSpectralSetting(n_sec_for_averaging=2, n_sec_for_instant_estimation=4)

    math = EmotionalMath(mls, ads, mss) 
    math.set_calibration_length(6)
    math.set_mental_estimation_mode(False)
    math.set_skip_wins_after_artifact(10)
    math.set_zero_spect_waves(True, 0, 1, 1, 1, 0)
    math.set_spect_normalization_by_bands_width(True)
    return math

# --- Функции для записи в файл ---
def init_log_file():
    global log_file_path, file_handle
    
    if not log_file_path:
        return
        
    try:
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(log_file_path) or '.', exist_ok=True)
        
        # Открытие файла в режиме добавления ('a')
        file_handle = open(log_file_path, 'a', encoding='utf-8')
        # Запись заголовка, если файл только что создан/пуст
        if file_handle.tell() == 0:
             file_handle.write("Timestamp_ms,Relaxation,Focus,Artifact\n")
        print(f"Log file opened: {log_file_path}")
    except Exception as e:
        print(f"Error opening log file: {str(e)}")
        file_handle = None

def close_log_file():
    global file_handle
    if file_handle:
        try:
            file_handle.close()
            print("Log file closed.")
        except Exception as e:
            print(f"Error closing log file: {str(e)}")
        file_handle = None

def log_data(relaxation, focus, is_artifacted):
    global file_handle, file_lock
    
    if file_handle is None:
        return

    timestamp_ms = int(time.time() * 1000)
    artifact_flag = 1 if is_artifacted else 0
    log_line = f"{timestamp_ms},{relaxation},{focus},{artifact_flag}\n"
    
    with file_lock:
        try:
            file_handle.write(log_line)
            file_handle.flush() # Сразу записываем на диск
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")

# --- Обработка сигнала (с добавлением записи в файл) ---
def on_signal_received(sensor, data):
    global math_lib, source_name
    
    if math_lib is None:
        return

    # BrainBit работает в биполярном режиме для библиотеки эмоций:
    # Left: T3 - O1, Right: T4 - O2
    raw_channels = []
    for sample in data:
        left_bipolar = sample.T3 - sample.O1
        right_bipolar = sample.T4 - sample.O2
        raw_channels.append(RawChannels(left_bipolar, right_bipolar))
    
    math_lib.push_bipolars(raw_channels)
    math_lib.process_data_arr()

    is_artifacted = math_lib.is_artifacted_sequence()

    if not math_lib.calibration_finished():
        progress = math_lib.get_calibration_percents()
        update_obs_text(f"Calibration: {progress}%")
        log_data(0, 0, is_artifacted)
    else:
        mental_data = math_lib.read_mental_data_arr()
        
        if is_artifacted:
            update_obs_text("Artifact detected (Adjust Headband)")
            log_data(0, 0, True)
        elif mental_data:
            last_sample = mental_data[-1]
            relax = int(last_sample.Rel_Relaxation)
            focus = int(last_sample.Rel_Attention)
            update_obs_text(f"Relax: {relax}% | Focus: {focus}%")
            log_data(relax, focus, False)

# --- Обновление текстового источника OBS ---
def update_obs_text(text):
    global source_name
    if not source_name:
        return

    source = obs.obs_get_source_by_name(source_name)
    if source is not None:
        settings = obs.obs_data_create()
        obs.obs_data_set_string(settings, "text", text)
        obs.obs_source_update(source, settings)
        obs.obs_data_release(settings)
        obs.obs_source_release(source)

# --- Основной рабочий процесс (в отдельном потоке) ---
def brainbit_worker():
    global scanner, sensor, math_lib, is_running

    # Инициализация файла ДО начала процесса
    init_log_file() 
    
    update_obs_text("Scanning for BrainBit...")
    
    try:
        # 1. Сканирование
        scanner = Scanner([SensorFamily.LEBrainBit])
        scanner.start()
        time.sleep(3)
        scanner.stop()
        
        sensors = scanner.sensors()
        if len(sensors) == 0:
            update_obs_text("BrainBit not found!")
            is_running = False
            return
        
        # 2. Подключение 
        sensor_info = sensors[0]
        update_obs_text(f"Connecting to {sensor_info.Name}...")
        sensor = scanner.create_sensor(sensor_info)
        
        # 3. Инициализация математики
        math_lib = create_math_instance()
        math_lib.start_calibration()
        
        # 4. Подписка на сигнал
        sensor.signalDataReceived = on_signal_received
        sensor.exec_command(SensorCommand.CommandStartSignal)
        
        update_obs_text("Signal Started. Calibrating...")

        while is_running:
            time.sleep(1)

    except Exception as e:
        print(f"Error in BrainBit worker: {str(e)}")
        update_obs_text(f"Error: {str(e)}")
    finally:
        cleanup_brainbit()

def cleanup_brainbit():
    global sensor, scanner, math_lib
    
    update_obs_text("Stopping BrainBit...")
    
    if sensor:
        try:
            sensor.exec_command(SensorCommand.CommandStopSignal)
            sensor.signalDataReceived = None
            sensor.disconnect()
            del sensor
        except:
            pass
        sensor = None

    if scanner:
        try:
            del scanner
        except:
            pass
        scanner = None
        
    if math_lib:
        math_lib.__del__()
        math_lib = None
        
    # Закрытие лог-файла
    close_log_file()
    
    update_obs_text("")

# --- Управление потоком ---
def start_processing():
    global is_running, worker_thread
    if is_running:
        return
    
    is_running = True
    worker_thread = threading.Thread(target=brainbit_worker)
    worker_thread.start()

def stop_processing():
    global is_running, worker_thread
    is_running = False
    if worker_thread:
        worker_thread.join(timeout=5) 
        worker_thread = None

# --- Запуск внешнего скрипта анализа ---
def run_analysis_script():
    global analysis_script_path, log_file_path
    
    if not analysis_script_path or not os.path.exists(analysis_script_path):
        print(f"Analysis script not found or path empty: {analysis_script_path}")
        update_obs_text("Analysis script path not set.")
        return
    
    if not log_file_path or not os.path.exists(log_file_path):
         print(f"Log file not found at {log_file_path}. Cannot run analysis.")
         update_obs_text("Log file not found for analysis.")
         return

    # Передаем актуальный путь к CSV в качестве аргумента
    log_file = log_file_path
    
    # Рекомендуется запускать скрипт с Python
    # Если python находится в PATH:
    command = ["python", analysis_script_path, log_file]
    
    try:
        print(f"Starting analysis script: {' '.join(command)}")
        # Запуск скрипта в неблокирующем режиме
        subprocess.Popen(command) 
        print("Analysis script started successfully.")
        update_obs_text("Analysis script launched.")
    except Exception as e:
        print(f"Failed to start analysis script: {e}")
        update_obs_text(f"Analysis Error: {str(e)}")


# --- События OBS ---
def on_event(event):
    if event == obs.OBS_FRONTEND_EVENT_RECORDING_STARTED or event == obs.OBS_FRONTEND_EVENT_STREAMING_STARTED:
        print("Starting BrainBit tracking...")
        start_processing()
        
    elif event == obs.OBS_FRONTEND_EVENT_RECORDING_STOPPED or event == obs.OBS_FRONTEND_EVENT_STREAMING_STOPPED:
        # Проверяем, не идет ли еще какой-то процесс
        streaming = obs.obs_frontend_streaming_active()
        recording = obs.obs_frontend_recording_active()
        
        if not streaming and not recording:
            print("Stopping BrainBit tracking...")
            # 1. Сначала останавливаем и очищаем ресурсы BrainBit (это закроет лог-файл)
            stop_processing() 
            # 2. Затем запускаем скрипт анализа
            run_analysis_script()

# --- Настройки скрипта в UI OBS ---
def script_description():
    return "BrainBit Neurofeedback Script\nStarts tracking when Recording or Streaming begins.\n**Requires 'pyneurosdk2', 'pyem-st-artifacts'**.\nRecords data to a CSV file and launches an external Python script upon recording/streaming stop."

def script_properties():
    props = obs.obs_properties_create()
    
    # 1. Поле для пути к лог-файлу (CSV)
    obs.obs_properties_add_path(props, "log_file", "Log File Path (CSV)", obs.OBS_PATH_FILE, "*.csv", None)
    
    # 2. Поле для пути к скрипту анализа
    obs.obs_properties_add_path(props, "analysis_script", "Analysis Script Path (Python)", obs.OBS_PATH_FILE, "*.py", None)
    
    # 3. Выпадающий список всех текстовых источников
    p = obs.obs_properties_add_list(props, "source", "Text Source", obs.OBS_COMBO_TYPE_EDITABLE, obs.OBS_COMBO_FORMAT_STRING)
    sources = obs.obs_enum_sources()
    if sources is not None:
        for source in sources:
            source_id = obs.obs_source_get_unversioned_id(source)
            if source_id == "text_gdiplus" or source_id == "text_ft2_source":
                name = obs.obs_source_get_name(source)
                obs.obs_property_list_add_string(p, name, name)
        obs.source_list_release(sources)
        
    return props

def script_update(settings):
    global source_name, log_file_path, analysis_script_path
    source_name = obs.obs_data_get_string(settings, "source")
    # Обновление путей
    log_file_path = obs.obs_data_get_string(settings, "log_file")
    analysis_script_path = obs.obs_data_get_string(settings, "analysis_script")

def script_load(settings):
    obs.obs_frontend_add_event_callback(on_event)

def script_unload():
    stop_processing()
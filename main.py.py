import sys
import os
import pandas as pd
import shutil
import time
import glob
import gc
import threading
import wave
import numpy as np
import concurrent.futures
from datetime import datetime
from dataclasses import dataclass

# --- LOCAL DATACLASS DEFINITIONS ---
@dataclass
class MathLibSetting:
    sampling_rate: int = 250
    process_win_freq: int = 25
    fft_window: int = 1000
    n_first_sec_skipped: int = 4
    bipolar_mode: bool = True
    channels_number: int = 4
    channel_for_analysis: int = 0

@dataclass
class ArtifactDetectSetting:
    art_bord: int = 110
    allowed_percent_artpoints: int = 70
    raw_betap_limit: int = 800000
    total_pow_border: int = 100
    global_artwin_sec: int = 4
    spect_art_by_totalp: bool = False
    hanning_win_spectrum: bool = False
    hamming_win_spectrum: bool = False
    num_wins_for_quality_avg: int = 100

@dataclass
class MentalAndSpectralSetting:
    n_sec_for_instant_estimation: int = 2
    n_sec_for_averaging: int = 2


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLineEdit, QLabel, QFileDialog, QTabWidget, 
    QMessageBox, QGroupBox, QSlider, QStyle, QFrame, QSizePolicy, 
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QScrollArea, QComboBox, QSplitter, QSpacerItem, QTextEdit
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize, QStandardPaths
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QCursor, QImage, QPixmap

# --- MOVIEPY IMPORTS ---
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips

try:
    from moviepy.video.fx.all import speedx
except ImportError:
    try:
        from moviepy.video.fx import speedx
    except ImportError:
        speedx = None

# --- OPTIONAL LIBS ---
try:
    import vlc
except ImportError:
    vlc = None
    print("VLC module not found. Install: pip install python-vlc")

try:
    import cv2
except ImportError:
    cv2 = None
    print("OpenCV not found. Install: pip install opencv-python")

try:
    import pyaudio
except ImportError:
    pyaudio = None
    print("PyAudio not found. Install: pip install pyaudio")

try:
    import mss
except ImportError:
    mss = None
    print("MSS not found. Install: pip install mss")

try:
    import keyboard
except ImportError:
    keyboard = None
    print("Keyboard not found. Install: pip install keyboard")

# --- BRAINBIT / NEUROSDK IMPORTS ---
SDK_LOADED = False
try:
    from neurosdk.scanner import Scanner
    from neurosdk.cmn_types import *
    from neurosdk.brainbit_sensor import BrainBitSensor
    from em_st_artifacts.emotional_math import EmotionalMath
    try:
        from em_st_artifacts.utils.lib_settings import (
            MathLibSetting, ArtifactDetectSetting,
            MentalAndSpectralSetting
        )
    except ImportError:
        pass 
        
    from em_st_artifacts.utils.support_classes import RawChannels
    SDK_LOADED = True
except ImportError as e:
    print(f"[BrainBit] SDK Libraries not found or incompatible: {e}")

# ==============================================================================
# 0. СТИЛИ ИНТЕРФЕЙСА
# ==============================================================================
STYLESHEET = """
QMainWindow, QWidget {
    background-color: #181818; 
    color: #ffffff;
    font-family: 'Roboto', 'Segoe UI', sans-serif;
    font-size: 14px;
}
QGroupBox {
    border: 1px solid #333;
    border-radius: 6px;
    margin-top: 20px;
    font-weight: bold;
    color: #aaaaaa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QLineEdit, QComboBox, QTextEdit {
    background-color: #121212;
    border: 1px solid #333;
    border-radius: 2px;
    padding: 8px;
    color: #fff;
}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus { border: 1px solid #3ea6ff; }
QComboBox::drop-down { border: none; }

QPushButton {
    background-color: #333;
    border: none;
    border-radius: 2px;
    padding: 8px 16px;
    color: #fff;
    font-weight: 500;
}
QPushButton:hover { background-color: #444; }
QPushButton:pressed { background-color: #3ea6ff; color: #000; }
QPushButton:disabled { background-color: #222; color: #555; }

QPushButton#ActionBtn {
    background-color: #3ea6ff;
    color: #000;
    font-weight: bold;
    padding: 10px;
}
QPushButton#ActionBtn:hover { background-color: #65b8ff; }

QPushButton#RecordBtn {
    background-color: #cc0000;
    color: #fff;
    font-weight: bold;
    padding: 12px;
    border-radius: 25px; 
}
QPushButton#RecordBtn:checked { background-color: #ff4444; border: 2px solid #fff; }

QTabWidget::pane { border: 1px solid #333; }
QTabBar::tab {
    background: #202020;
    color: #aaa;
    padding: 10px 20px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected {
    background: #333;
    color: #3ea6ff;
    border-bottom: 2px solid #3ea6ff;
}

QFrame#VideoCard { background-color: #202020; border: 1px solid #333; border-radius: 8px; }
QFrame#VideoCard:hover { border: 1px solid #555; background-color: #2a2a2a; }
QPushButton#CardPlayBtn { background-color: transparent; border: none; }
QPushButton#CardPlayBtn:hover { background-color: rgba(255, 255, 255, 0.1); border-radius: 30px; }
QTableWidget { background-color: #121212; gridline-color: #333; border: none; font-size: 13px; }
QHeaderView::section { background-color: #202020; color: #aaa; padding: 6px; border: none; border-bottom: 1px solid #333; }
QTableWidget::item:selected { background-color: #333; color: #3ea6ff; }
"""

# ==============================================================================
# 1. ЛОГИКА АНАЛИЗА (MoviePy)
# ==============================================================================

CATEGORY_CONFIG = {
    'Focus_Highlights': {'col': 'Focus', 'threshold': 70, 'condition': 'ge'},
    'Relaxation_Moments': {'col': 'Relaxation', 'threshold': 60, 'condition': 'ge'} 
}

MIN_SEGMENT_DURATION_SEC = 2.0 
MAX_GAP_MS = 2000

# === MOVIEPY V2 COMPATIBILITY HELPERS ===
def safe_subclip(clip, start, end):
    if hasattr(clip, 'subclipped'): return clip.subclipped(start, end)
    if hasattr(clip, 'subclip'): return clip.subclip(start, end)
    return clip

def safe_with_duration(clip, duration):
    if hasattr(clip, 'with_duration'): return clip.with_duration(duration)
    if hasattr(clip, 'set_duration'): return clip.set_duration(duration)
    return clip

def safe_set_audio(video, audio):
    if hasattr(video, 'with_audio'): return video.with_audio(audio)
    if hasattr(video, 'set_audio'): return video.set_audio(audio)
    return video

def safe_speed_correction(video, factor):
    if speedx: return video.fx(speedx, factor=factor)
    return video

def calculate_segments(df_filtered, metric_col):
    if df_filtered.empty: return pd.DataFrame()
    df_filtered = df_filtered.copy()
    df_filtered['diff'] = df_filtered['Timestamp_ms'].diff() > MAX_GAP_MS
    df_filtered['segment_id'] = df_filtered['diff'].cumsum()
    segments = df_filtered.groupby('segment_id').agg(
        start_ms=('Timestamp_ms', 'min'),
        end_ms=('Timestamp_ms', 'max'),
        avg_score=(metric_col, 'mean') 
    ).reset_index()
    segments['duration_sec'] = (segments['end_ms'] - segments['start_ms']) / 1000
    segments = segments[segments['duration_sec'] >= MIN_SEGMENT_DURATION_SEC]
    return segments

def analyze_and_concatenate_video(csv_path, video_file):
    if not os.path.exists(csv_path): raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(video_file): raise FileNotFoundError(f"Video not found: {video_file}")
    
    df = pd.read_csv(csv_path)
    req = ['Timestamp_ms', 'Focus', 'Relaxation']
    if not all(c in df.columns for c in req): raise ValueError(f"Missing cols: {req}")
    if 'Artifact' in df.columns:
        df = df[df['Artifact'] == 0].copy()

    start_time_ms = df['Timestamp_ms'].iloc[0]
    
    video_dir = os.path.dirname(os.path.abspath(video_file))
    output_root_folder = os.path.join(video_dir, "Processed_Output")
    if not os.path.exists(output_root_folder): os.makedirs(output_root_folder)

    results_data = {} 
    full_clip = None

    try:
        full_clip = VideoFileClip(video_file)
        all_segments_df = {}
        
        for cat_name, config in CATEGORY_CONFIG.items():
            col, thresh = config['col'], config['threshold']
            df_cat = df[df[col] >= thresh].copy() if config['condition'] == 'ge' else df[df[col] <= thresh].copy()
            all_segments_df[cat_name] = calculate_segments(df_cat, metric_col=col)

        for category, segments in all_segments_df.items():
            if segments.empty: continue
            
            clips_for_montage = []
            segment_metadata = []
            current_montage_time = 0.0 
            
            folder_name = "Focus clips" if "Focus" in category else "Relaxation clips"
            montage_filename = "focus montage.mp4" if "Focus" in category else "relaxation montage.mp4"

            segments_folder = os.path.join(output_root_folder, folder_name)
            if not os.path.exists(segments_folder): os.makedirs(segments_folder)
            
            for idx, row in segments.iterrows():
                s = (row['start_ms'] - start_time_ms) / 1000
                e = (row['end_ms'] - start_time_ms) / 1000
                
                if s >= full_clip.duration: continue
                if e > full_clip.duration: e = full_clip.duration
                
                sub = safe_subclip(full_clip, s, e)
                clips_for_montage.append(sub)
                
                seg_filename = f"segment_{idx+1:03d}_{row['avg_score']:.0f}.mp4"
                seg_path = os.path.join(segments_folder, seg_filename)
                
                sub.write_videofile(seg_path, codec='libx264', audio_codec='aac', 
                                     remove_temp=True, logger=None, preset='ultrafast') 
                
                segment_metadata.append({
                    'id': idx + 1, 'orig_start': s, 'orig_end': e, 
                    'montage_start': current_montage_time, 'duration': e-s, 
                    'score': row['avg_score'], 'file_path': seg_path
                })
                current_montage_time += (e-s)
            
            if clips_for_montage:
                final = concatenate_videoclips(clips_for_montage)
                out_name = os.path.join(output_root_folder, montage_filename)
                final.write_videofile(out_name, codec='libx264', audio_codec='aac', 
                                     temp_audiofile='temp-audio.m4a', remove_temp=True, logger=None)
                results_data[category] = {'path': out_name, 'segments': segment_metadata, 'total_duration': final.duration}
                final.close()
                for c in clips_for_montage: 
                    try: c.close()
                    except: pass
                
    except Exception as e: raise Exception(f"Processing Error: {e}")
    finally:
        if full_clip: full_clip.close()
            
    return results_data

class VideoAnalyzerWorker(QThread):
    finished = pyqtSignal(dict) 
    error = pyqtSignal(str)
    def __init__(self, csv, video):
        super().__init__()
        self.csv, self.video = csv, video
    def run(self):
        try:
            res = analyze_and_concatenate_video(self.csv, self.video)
            self.finished.emit(res)
        except Exception as e: self.error.emit(str(e))

# ==============================================================================
# 2. WORKERS (BrainBit & Debug & Screen)
# ==============================================================================

class DebugWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def run(self):
        if not SDK_LOADED:
            self.log_signal.emit("SDK not loaded!")
            self.finished_signal.emit()
            return

        def log(msg):
            self.log_signal.emit(str(msg))

        scanner = None
        sensor = None

        try:
            log("Initializing Scanner for LEBrainBit / Black...")
            scanner = Scanner([SensorFamily.LEBrainBit, SensorFamily.LEBrainBitBlack])
            
            # Callbacks for scanner
            def sensor_found(scanner, sensors):
                for index in range(len(sensors)):
                    log('Sensor found: %s' % sensors[index])
            scanner.sensorsChanged = sensor_found

            log("Starting search for 5 sec...")
            scanner.start()
            time.sleep(5)
            scanner.stop()

            sensorsInfo = scanner.sensors()
            if not sensorsInfo:
                log("No sensors found.")
                return

            current_sensor_info = sensorsInfo[0]
            log(f"Connecting to: {current_sensor_info.Name}...")

            # Blocking connection
            sensor = scanner.create_sensor(current_sensor_info)
            sensor.connect()
            log("Device connected object created.")

            # Status callbacks
            def on_sensor_state_changed(sensor, state):
                log(f'State Changed: {state}')

            def on_battery_changed(sensor, battery):
                log(f'Battery: {battery}%')

            sensor.sensorStateChanged = on_sensor_state_changed
            sensor.batteryChanged = on_battery_changed

            if sensor.state == SensorState.StateInRange:
                log("Status: Connected (InRange)")
            else:
                log(f"Status: {sensor.state}")

            log(f"Family: {sensor.sens_family}")
            log(f"Name: {sensor.name}")
            log(f"Serial: {sensor.serial_number}")
            log(f"Batt: {sensor.batt_power}")
            log(f"Version: {sensor.version}")

            # Data Callbacks
            def on_signal_received(sensor, data):
                pass 

            def on_resist_received(sensor, data):
                 # ИСПРАВЛЕНИЕ ЗДЕСЬ: data - это объект, а не массив
                 log(f"Resist: O1={data.O1}, O2={data.O2}, T3={data.T3}, T4={data.T4}")

            def on_mems_received(sensor, data):
                 log(f"MEMS: {data}")

            if sensor.is_supported_feature(SensorFeature.Signal):
                sensor.signalDataReceived = on_signal_received

            if sensor.is_supported_feature(SensorFeature.Resist):
                sensor.resistDataReceived = on_resist_received

            if sensor.is_supported_feature(SensorFeature.MEMS):
                sensor.memsDataReceived = on_mems_received

            # --- TEST SEQUENCE ---
            
            # 1. Signal
            if sensor.is_supported_command(SensorCommand.StartSignal):
                log(">>> STARTING SIGNAL (5s)...")
                sensor.exec_command(SensorCommand.StartSignal)
                time.sleep(5)
                sensor.exec_command(SensorCommand.StopSignal)
                log(">>> STOPPED SIGNAL")
            else:
                log("Signal command not supported")
            
            # 2. Resist
            if sensor.is_supported_command(SensorCommand.StartResist):
                log(">>> STARTING RESIST (5s)...")
                sensor.exec_command(SensorCommand.StartResist)
                time.sleep(5)
                sensor.exec_command(SensorCommand.StopResist)
                log(">>> STOPPED RESIST")

            # 3. MEMS
            if sensor.is_supported_command(SensorCommand.StartMEMS):
                log(">>> STARTING MEMS (3s)...")
                sensor.exec_command(SensorCommand.StartMEMS)
                time.sleep(3)
                sensor.exec_command(SensorCommand.StopMEMS)
                log(">>> STOPPED MEMS")

            sensor.disconnect()
            log("Disconnected from sensor")

        except Exception as err:
            log(f"CRITICAL ERROR: {err}")
            import traceback
            log(traceback.format_exc())
        
        finally:
            if sensor: 
                del sensor
            if scanner:
                del scanner
            log('Scanner removed. Test Finished.')
            self.finished_signal.emit()


class BrainBitWorker(QThread):
    status_update = pyqtSignal(str)
    
    def __init__(self, output_folder):
        super().__init__()
        self.output_folder = output_folder
        self.is_running = True
        self.sensor = None
        self.scanner = None
        self.math_lib = None
        self.csv_file = None
        
    def init_math(self):
        # 1. Main Settings
        mls = MathLibSetting(
            sampling_rate=250,
            process_win_freq=25,
            n_first_sec_skipped=4,
            fft_window=1000,
            bipolar_mode=True,
            channels_number=4,
            channel_for_analysis=0 
        )

        # 2. Artifact Settings
        ads = ArtifactDetectSetting(
            art_bord=110,
            allowed_percent_artpoints=70,
            raw_betap_limit=800_000,
            global_artwin_sec=4,
            num_wins_for_quality_avg=125,
            hamming_win_spectrum=True,
            hanning_win_spectrum=False,
            total_pow_border=400_000_000,
            spect_art_by_totalp=True
        )

        # 4. Mental Settings
        mss = MentalAndSpectralSetting(
            n_sec_for_averaging=2,
            n_sec_for_instant_estimation=4
        )
        
        # Init without sads
        math = EmotionalMath(mls, ads, mss)
        
        math.set_calibration_length(6)
        math.set_mental_estimation_mode(False)
        math.set_skip_wins_after_artifact(10)
        math.set_zero_spect_waves(True, 0, 1, 1, 1, 0)
        math.set_spect_normalization_by_bands_width(True)
        math.start_calibration()
        
        return math

    def run(self):
        if not SDK_LOADED:
            self.status_update.emit("BrainBit SDK not loaded")
            return

        try:
            csv_path = os.path.join(self.output_folder, "brainbit_log.csv")
            self.csv_file = open(csv_path, "w", encoding="utf-8", buffering=1)
            self.csv_file.write("Timestamp_ms,Relaxation,Focus,Artifact,O1,O2,T3,T4\n")
            
            self.status_update.emit("Scanning BrainBit...")
            self.scanner = Scanner([SensorFamily.LEBrainBit])
            self.scanner.start()
            time.sleep(3)
            self.scanner.stop()
            
            sensors = self.scanner.sensors()
            if not sensors:
                self.status_update.emit("BrainBit not found!")
                return

            self.sensor = self.scanner.create_sensor(sensors[0])
            self.sensor.connect()
            self.status_update.emit("BrainBit Connected")
            
            self.math_lib = self.init_math()
            
            def on_data_received(sensor, data):
                if not self.math_lib: return
                try:
                    raw_channels = []
                    last_vals = (0,0,0,0)
                    for sample in data:
                        left = sample.T3 - sample.O1
                        right = sample.T4 - sample.O2
                        raw_channels.append(RawChannels(left, right))
                        last_vals = (sample.O1, sample.O2, sample.T3, sample.T4)
                    
                    self.math_lib.push_data(raw_channels)
                    self.math_lib.process_data_arr()
                    
                    is_art = self.math_lib.is_artifacted_sequence()
                    relax, focus = 0, 0
                    
                    if self.math_lib.calibration_finished():
                        mental = self.math_lib.read_mental_data_arr()
                        if mental:
                            relax = int(mental[-1].Rel_Relaxation)
                            focus = int(mental[-1].Rel_Attention)
                    
                    ts = int(time.time() * 1000)
                    art_val = 1 if is_art else 0
                    line = f"{ts},{relax},{focus},{art_val},{last_vals[0]:.2f},{last_vals[1]:.2f},{last_vals[2]:.2f},{last_vals[3]:.2f}\n"
                    
                    if self.csv_file and not self.csv_file.closed:
                        self.csv_file.write(line)
                        
                except Exception as e:
                    print(f"Data error: {e}")

            self.sensor.signalDataReceived = on_data_received
            
            self.sensor.exec_command(SensorCommand.StartSignal)
            
            while self.is_running:
                time.sleep(1)

        except Exception as e:
            self.status_update.emit(f"BB Error: {e}")
            print(f"BB Critical Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.sensor:
            try:
                self.sensor.exec_command(SensorCommand.StopSignal)
                self.sensor.disconnect()
            except: pass
            self.sensor = None
        
        if self.csv_file:
            try: self.csv_file.close()
            except: pass
            self.csv_file = None
            
        if self.scanner:
            del self.scanner
            self.scanner = None
            
        if self.math_lib:
            del self.math_lib
            self.math_lib = None

    def stop(self):
        self.is_running = False
        self.wait()

class ScreenRecorderWorker(QThread):
    update_frame = pyqtSignal(QImage)
    recording_finished = pyqtSignal(str) 
    error = pyqtSignal(str)

    def __init__(self, output_folder=".", fps=30, monitor_idx=1): 
        super().__init__()
        self.output_folder = output_folder
        self.target_fps = fps
        self.monitor_idx = monitor_idx
        self.is_running = True     
        self.is_recording = False  
        self.stop_requested = False 
        
        self.start_time = 0
        self.real_duration = 0
        
        self.audio_format = pyaudio.paInt16 if pyaudio else None
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        
    def start_recording(self):
        self.is_recording = True 
        
    def stop_recording(self):
        if self.is_recording:
            self.stop_requested = True
        
    def run(self):
        if not mss or not cv2: 
            self.error.emit("MSS or OpenCV not installed")
            return
        
        writer = None
        audio_frames = []
        p = None
        stream = None
        video_filename = ""
        audio_filename = ""
        final_filename = ""
        
        with mss.mss() as sct:
            try: 
                monitor = sct.monitors[self.monitor_idx]
            except IndexError: 
                monitor = sct.monitors[1]
            
            width = monitor["width"]
            height = monitor["height"]
            if width % 2 != 0: width -= 1
            if height % 2 != 0: height -= 1
            monitor_rect = {"top": monitor["top"], "left": monitor["left"], "width": width, "height": height}

            while self.is_running:
                loop_start = time.time()
                
                try:
                    raw_img = sct.grab(monitor_rect)
                    frame_np = np.array(raw_img) # BGRA
                except Exception as e:
                    print(f"Screen grab failed: {e}")
                    break
                
                if self.is_recording and writer is None and not self.stop_requested:
                    video_filename = os.path.join(self.output_folder, "temp_video.mp4")
                    audio_filename = os.path.join(self.output_folder, "temp_audio.wav")
                    final_filename = os.path.join(self.output_folder, "recording.mp4")
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    writer = cv2.VideoWriter(video_filename, fourcc, self.target_fps, (width, height))
                    self.start_time = time.time()
                    
                    if pyaudio:
                        try:
                            p = pyaudio.PyAudio()
                            stream = p.open(format=self.audio_format, channels=self.channels,
                                            rate=self.rate, input=True, 
                                            frames_per_buffer=self.chunk)
                        except Exception as e:
                            print(f"Audio init failed: {e}")
                            stream = None
                    
                    audio_frames = []
                    
                if self.is_recording and writer is not None:
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2BGR)
                    writer.write(frame_bgr)
                    
                    if stream and stream.is_active():
                        try:
                            while stream.get_read_available() >= self.chunk:
                                data = stream.read(self.chunk, exception_on_overflow=False)
                                audio_frames.append(data)
                        except: pass
                
                if self.stop_requested and writer is not None:
                    self.real_duration = time.time() - self.start_time
                    writer.release()
                    writer = None
                    
                    if stream: 
                        stream.stop_stream()
                        stream.close()
                    if p: 
                        p.terminate()
                        
                    if audio_frames and pyaudio:
                        try:
                            wf = wave.open(audio_filename, 'wb')
                            wf.setnchannels(self.channels)
                            wf.setsampwidth(p.get_sample_size(self.audio_format))
                            wf.setframerate(self.rate)
                            wf.writeframes(b''.join(audio_frames))
                            wf.close()
                        except Exception as e:
                            print(f"Audio save error: {e}")
                    
                    self.merge_files(video_filename, audio_filename, final_filename, self.real_duration)
                    
                    self.is_recording = False
                    self.stop_requested = False
                    p = None; stream = None; audio_frames = []

                if not self.is_recording and self.is_running:
                    rgb_small = cv2.resize(frame_np, (640, 360))
                    rgb_small = cv2.cvtColor(rgb_small, cv2.COLOR_BGRA2RGB)
                    h, w, ch = rgb_small.shape
                    qimg = QImage(rgb_small.data, w, h, ch * w, QImage.Format_RGB888)
                    self.update_frame.emit(qimg)

                elapsed = time.time() - loop_start
                wait = (1.0 / self.target_fps) - elapsed
                if wait > 0: time.sleep(wait)

        if writer: writer.release()
        if p: p.terminate()
            
    def merge_files(self, vid_file, aud_file, out_file, real_duration):
        try:
            has_video = os.path.exists(vid_file)
            has_audio = os.path.exists(aud_file)
            
            if has_video:
                video_clip = VideoFileClip(vid_file)
                
                if abs(video_clip.duration - real_duration) > (real_duration * 0.05):
                   factor = video_clip.duration / real_duration
                   video_clip = safe_speed_correction(video_clip, factor)
                
                final_audio = None
                if has_audio:
                    try:
                        audio_clip = AudioFileClip(aud_file)
                        if hasattr(audio_clip, 'with_duration'): 
                             final_audio = audio_clip.with_duration(video_clip.duration)
                        else: 
                             final_audio = audio_clip.set_duration(video_clip.duration)
                    except Exception as e:
                        print(f"Audio clip load error: {e}")

                if final_audio:
                      final_clip = safe_set_audio(video_clip, final_audio)
                else:
                      final_clip = video_clip

                final_clip.write_videofile(out_file, codec='libx264', audio_codec='aac', logger=None)
                
                video_clip.close()
                if final_audio: final_audio.close()
                
                try: 
                    os.remove(vid_file)
                    if has_audio: os.remove(aud_file)
                except: pass
                
                self.recording_finished.emit(out_file)
                
        except Exception as e:
            self.error.emit(f"Merge Error: {e}")
            if os.path.exists(vid_file):
                 self.recording_finished.emit(vid_file)

    def stop(self):
        self.is_running = False
        self.wait()

# ==============================================================================
# 4. GUI TABS
# ==============================================================================

class DebugTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        info = QLabel("Runs the official NeuroSDK debug sequence: Scan -> Connect -> Signal(5s) -> Resist(5s) -> MEMS(3s).")
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("START DEBUG SEQUENCE")
        self.btn_start.setObjectName("ActionBtn")
        self.btn_start.clicked.connect(self.start_debug)
        
        self.btn_clear = QPushButton("Clear Log")
        self.btn_clear.clicked.connect(self.clear_log)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_clear)
        layout.addLayout(btn_layout)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("font-family: Consolas, monospace; font-size: 12px; background-color: #0d0d0d; color: #00ff00;")
        layout.addWidget(self.console)

    def start_debug(self):
        self.btn_start.setEnabled(False)
        self.console.append("--- STARTING DEBUG ---")
        self.worker = DebugWorker()
        self.worker.log_signal.connect(self.log_msg)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def log_msg(self, text):
        self.console.append(text)
        # Auto scroll
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.console.append("--- FINISHED ---")

    def clear_log(self):
        self.console.clear()

class RecorderTab(QWidget):
    trigger_record = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.screen_worker = None
        self.brainbit_worker = None
        self.init_ui()
        self.trigger_record.connect(self.toggle_recording)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Source
        src_layout = QHBoxLayout()
        lbl_src = QLabel("Capture Source:")
        lbl_src.setStyleSheet("font-weight: bold; color: #aaa;")
        self.combo_screens = QComboBox()
        self.populate_screens() 
        self.combo_screens.currentIndexChanged.connect(self.change_screen_source)
        src_layout.addWidget(lbl_src); src_layout.addWidget(self.combo_screens); src_layout.addStretch()
        layout.addLayout(src_layout)
        
        # Preview
        self.preview_container = QWidget()
        self.preview_container.setStyleSheet("background-color: #000; border: 2px solid #333; border-radius: 4px;")
        self.preview_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pc_layout = QVBoxLayout(self.preview_container)
        pc_layout.setContentsMargins(0,0,0,0)
        self.lbl_camera = QLabel("Press 'Activate' to preview")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("color: #555;")
        self.lbl_camera.setScaledContents(True) 
        pc_layout.addWidget(self.lbl_camera)
        layout.addWidget(self.preview_container)
        
        # Controls
        control_panel = QFrame()
        control_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed) 
        control_panel.setMinimumHeight(120) 
        control_panel.setStyleSheet("background-color: #202020; border-top: 1px solid #333; padding: 10px;")
        cp_layout = QVBoxLayout(control_panel)
        
        btns = QHBoxLayout()
        self.btn_toggle_cam = QPushButton("Activate Preview")
        self.btn_toggle_cam.clicked.connect(self.toggle_camera)
        btns.addWidget(self.btn_toggle_cam)

        self.btn_record = QPushButton("● REC (F9)")
        self.btn_record.setObjectName("RecordBtn")
        self.btn_record.setCheckable(True)
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self.toggle_recording)
        btns.addWidget(self.btn_record)
        
        self.lbl_status = QLabel("Ready")
        btns.addWidget(self.lbl_status)
        cp_layout.addLayout(btns)
        
        # Paths
        path_layout = QHBoxLayout()
        # Default to Videos folder
        default_path = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation) or os.path.abspath(".")
        self.input_base_path = QLineEdit(default_path)
        btn_browse = QPushButton("Base Folder")
        btn_browse.clicked.connect(self.browse_folder)
        path_layout.addWidget(self.input_base_path)
        path_layout.addWidget(btn_browse)
        cp_layout.addLayout(path_layout)
        
        layout.addWidget(control_panel)

    def populate_screens(self):
        if not mss: return
        with mss.mss() as sct:
            for i, m in enumerate(sct.monitors[1:], 1):
                self.combo_screens.addItem(f"Screen {i}: {m['width']}x{m['height']}", i)

    def change_screen_source(self):
        if self.screen_worker:
            self.toggle_camera() 
            QTimer.singleShot(200, self.toggle_camera)

    def browse_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select Base Folder")
        if f: self.input_base_path.setText(f)

    def toggle_camera(self):
        if self.screen_worker is None:
            if not mss or not cv2:
                QMessageBox.critical(self, "Error", "Libraries missing.")
                return

            monitor_idx = self.combo_screens.currentData() or 1
            
            self.screen_worker = ScreenRecorderWorker(
                output_folder=self.input_base_path.text(), 
                fps=30, 
                monitor_idx=monitor_idx
            )
            self.screen_worker.update_frame.connect(self.update_image)
            self.screen_worker.recording_finished.connect(self.on_save_finished)
            self.screen_worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
            self.screen_worker.start()
            
            self.btn_toggle_cam.setText("Deactivate")
            self.btn_record.setEnabled(True)
            self.lbl_status.setText("Preview Active")
        else:
            self.screen_worker.stop()
            self.screen_worker = None
            if self.brainbit_worker: self.brainbit_worker.stop()
            
            self.lbl_camera.clear()
            self.lbl_camera.setText("Preview Stopped")
            self.btn_toggle_cam.setText("Activate Preview")
            self.btn_record.setEnabled(False)
            self.btn_record.setChecked(False)
            self.lbl_status.setText("Ready")

    def toggle_recording(self):
        if not self.screen_worker: return 
        if self.sender() != self.btn_record:
            self.btn_record.setChecked(not self.btn_record.isChecked())

        if self.btn_record.isChecked():
            # 1. CREATE SESSION FOLDER
            base_dir = self.input_base_path.text()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_name = f"Session_{timestamp}"
            session_path = os.path.join(base_dir, session_name)
            
            try:
                os.makedirs(session_path, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot create folder: {e}")
                self.btn_record.setChecked(False)
                return

            # 2. START BRAINBIT WORKER (WITH SAFETY CHECK) 
            try:
                self.brainbit_worker = BrainBitWorker(session_path)
                if self.brainbit_worker:
                    self.brainbit_worker.status_update.connect(lambda s: self.lbl_status.setText(s))
                    self.brainbit_worker.start()
            except Exception as e:
                print(f"Failed to start BrainBit worker: {e}")
                self.lbl_status.setText("BrainBit Error - Only Video Recording")

            # 3. START SCREEN RECORDING
            self.screen_worker.output_folder = session_path
            self.screen_worker.start_recording()
            
            self.lbl_status.setText(f"RECORDING to: {session_name}")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
        else:
            # STOP
            self.lbl_status.setText("Finishing... Processing...")
            self.lbl_status.setStyleSheet("color: orange; font-weight: bold;")
            self.btn_record.setEnabled(False) 
            
            self.screen_worker.stop_recording()
            if self.brainbit_worker:
                self.brainbit_worker.stop()
                self.brainbit_worker = None
            
    def on_save_finished(self, path):
        self.lbl_status.setText(f"Saved Session!")
        self.lbl_status.setStyleSheet("color: #4CAF50;")
        self.btn_record.setEnabled(True)
        
    def update_image(self, image):
        self.lbl_camera.setPixmap(QPixmap.fromImage(image))
    
    def closeEvent(self, event):
        if self.screen_worker: self.screen_worker.stop()
        if self.brainbit_worker: self.brainbit_worker.stop()

class VideoClickableFrame(QFrame):
    doubleClicked = pyqtSignal()
    def mouseDoubleClickEvent(self, event): self.doubleClicked.emit()

class VideoCardWidget(QFrame):
    playClicked = pyqtSignal(str) 
    def __init__(self, title, video_path="", is_small=False):
        super().__init__()
        self.setObjectName("VideoCard")
        self.video_path = video_path
        layout = QHBoxLayout(self)
        self.btn_play = QPushButton()
        self.btn_play.setObjectName("CardPlayBtn")
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setFixedSize(40, 40)
        self.btn_play.clicked.connect(lambda: self.playClicked.emit(self.video_path))
        layout.addWidget(self.btn_play)
        
        info_lay = QVBoxLayout()
        lbl = QLabel(title); lbl.setStyleSheet("font-weight:bold; color:white;")
        self.sub = QLabel("Duration: " + video_path) 
        info_lay.addWidget(lbl); info_lay.addWidget(self.sub)
        layout.addLayout(info_lay)
        layout.addStretch()
    
    def set_data(self, path, dur, sub):
        self.video_path = path
        self.sub.setText(f"{dur} {sub}")

class VideoAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Video Suite + BrainBit")
        self.setGeometry(50, 50, 1000, 700) 
        self.setStyleSheet(STYLESHEET)
        
        self.instance = None
        self.media_player = None
        self.results_data = {} 
        self.current_video_path = None
        
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_ui_timer)

        self.init_vlc()
        self.init_main_ui()
        
        if keyboard:
            keyboard.add_hotkey('f9', self.handle_hotkey_f9)

    def handle_hotkey_f9(self):
        self.tab_recorder.trigger_record.emit()

    def init_vlc(self):
        if vlc:
            try:
                self.instance = vlc.Instance("--no-xlib", "--quiet", "--no-video-title-show")
                self.media_player = vlc.MediaPlayer(self.instance)
            except: self.media_player = None

    def init_main_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # --- NEW DEBUG TAB (INDEX 0) ---
        self.tab_debug = DebugTab()
        self.tabs.addTab(self.tab_debug, "DEBUG SDK")

        self.tab_recorder = RecorderTab()
        self.tabs.addTab(self.tab_recorder, "RECORDER & BRAINBIT")

        self.tab_analyzer = QWidget()
        self.build_analyzer_ui(self.tab_analyzer)
        self.tabs.addTab(self.tab_analyzer, "ANALYZER")
        
    def build_analyzer_ui(self, parent_widget):
        main_layout = QVBoxLayout(parent_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        self.file_group = QGroupBox("Project Configuration")
        flayout = QVBoxLayout(self.file_group)
        
        self.btn_load_project = QPushButton("📂 OPEN SESSION FOLDER")
        self.btn_load_project.setObjectName("LoadFolderBtn")
        self.btn_load_project.setCursor(Qt.PointingHandCursor)
        self.btn_load_project.clicked.connect(self.open_project_folder)
        flayout.addWidget(self.btn_load_project)

        r1 = QHBoxLayout()
        self.csv_path_input = QLineEdit(); self.csv_path_input.setPlaceholderText("Select CSV...")
        b1 = QPushButton("Browse CSV"); b1.clicked.connect(lambda: self.browse_file('csv'))
        r1.addWidget(self.csv_path_input); r1.addWidget(b1)
        flayout.addLayout(r1)
        
        r2 = QHBoxLayout()
        self.video_path_input = QLineEdit(); self.video_path_input.setPlaceholderText("Select Raw Video File...")
        b2 = QPushButton("Browse Video"); b2.clicked.connect(lambda: self.browse_file('video'))
        r2.addWidget(self.video_path_input); r2.addWidget(b2)
        flayout.addLayout(r2)
        
        self.analyze_button = QPushButton("START PROCESSING")
        self.analyze_button.setObjectName("ActionBtn")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)
        flayout.addWidget(self.analyze_button)
        main_layout.addWidget(self.file_group)

        self.video_container = QWidget()
        v_layout = QVBoxLayout(self.video_container)
        v_layout.setContentsMargins(0, 0, 0, 0); v_layout.setSpacing(0)
        
        self.video_frame = VideoClickableFrame()
        self.video_frame.setStyleSheet("background-color: #000;")
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_frame.setMinimumHeight(250)
        self.video_frame.doubleClicked.connect(self.toggle_fullscreen)
        if not vlc: self.video_frame.setLayout(QVBoxLayout()); self.video_frame.layout().addWidget(QLabel("VLC Required"))
        v_layout.addWidget(self.video_frame)
        
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background-color: #202020;")
        ctrl = QHBoxLayout(self.controls_widget)
        self.play_button = QPushButton()
        self.play_button.setFixedSize(40, 40)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_video)
        ctrl.addWidget(self.play_button)
        self.time_label = QLabel("00:00 / 00:00")
        ctrl.addWidget(self.time_label)
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.sliderMoved.connect(self.set_position)
        ctrl.addWidget(self.position_slider)
        self.volume_slider = QSlider(Qt.Horizontal); self.volume_slider.setFixedWidth(80)
        self.volume_slider.setRange(0, 100); self.volume_slider.setValue(70)
        self.volume_slider.valueChanged.connect(self.set_volume)
        ctrl.addWidget(self.volume_slider)
        self.fs_button = QPushButton(); self.fs_button.setObjectName("FsBtn")
        self.fs_button.setFixedSize(40, 40); self.fs_button.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.fs_button.clicked.connect(self.toggle_fullscreen)
        ctrl.addWidget(self.fs_button)
        v_layout.addWidget(self.controls_widget)
        main_layout.addWidget(self.video_container)

        self.inner_tabs = QTabWidget()
        self.inner_tabs.setMinimumHeight(200)
        self.inner_tabs.currentChanged.connect(self.tab_changed)
        self.tables = {}; self.gallery_layouts = {}
        
        w1, t1 = self.create_list_tab(); self.tables['Focus_Highlights'] = t1; self.inner_tabs.addTab(w1, "Focus (Full + List)")
        w2, l2 = self.create_gallery_tab(); self.gallery_layouts['Focus_Highlights'] = l2; self.inner_tabs.addTab(w2, "Focus (Clips Gallery)")
        w3, t3 = self.create_list_tab(); self.tables['Relaxation_Moments'] = t3; self.inner_tabs.addTab(w3, "Relaxation (Full + List)")
        w4, l4 = self.create_gallery_tab(); self.gallery_layouts['Relaxation_Moments'] = l4; self.inner_tabs.addTab(w4, "Relaxation (Clips Gallery)")
        
        main_layout.addWidget(self.inner_tabs)
        self.check_enable_button()

    def create_list_tab(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(10,10,10,10)
        t = QTableWidget(); t.setColumnCount(4)
        t.setHorizontalHeaderLabels(["#", "Time (Source)", "Duration", "Score"])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.setSelectionBehavior(QAbstractItemView.SelectRows); t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.cellClicked.connect(self.on_segment_clicked)
        l.addWidget(t)
        return w, t

    def create_gallery_tab(self):
        s = QScrollArea(); s.setWidgetResizable(True)
        c = QWidget(); c.setStyleSheet("background: transparent;")
        l = QVBoxLayout(c); l.setSpacing(10); l.setAlignment(Qt.AlignTop)
        s.setWidget(c)
        return s, l

    def reset_state(self):
        if self.timer.isActive(): self.timer.stop()
        if self.media_player: 
            self.media_player.stop(); self.media_player.set_media(None)
        gc.collect(); QApplication.processEvents()
        
        self.current_video_path = None
        self.play_button.setEnabled(False)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.position_slider.setValue(0); self.time_label.setText("00:00 / 00:00")
        
        self.results_data = {}
        for t in self.tables.values(): t.setRowCount(0)
        for l in self.gallery_layouts.values():
            while l.count(): 
                child = l.takeAt(0)
                if child.widget(): child.widget().deleteLater()
        QApplication.processEvents()

    def browse_file(self, mode):
        if mode == 'video': self.reset_state()
        fltr = "CSV (*.csv)" if mode == 'csv' else "Video (*.mp4 *.avi *.mov *.mkv)"
        target = self.csv_path_input if mode == 'csv' else self.video_path_input
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", fltr)
        if path:
            target.setText(path)
            self.check_enable_button()

    def check_enable_button(self):
        c, v = os.path.exists(self.csv_path_input.text()), os.path.exists(self.video_path_input.text())
        self.analyze_button.setEnabled(c and v)
        if v:
            if self.load_existing_results(self.video_path_input.text(), self.csv_path_input.text()):
                self.populate_data(); self.analyze_button.setText("RE-PROCESS (Data Loaded)")
            else: self.analyze_button.setText("START PROCESSING")

    def open_project_folder(self):
        self.reset_state()
        folder = QFileDialog.getExistingDirectory(self, "Select Video Project Folder")
        if not folder: return
        
        csvs = glob.glob(os.path.join(folder, "*.csv"))
        if not csvs: return QMessageBox.warning(self, "Error", "No CSV found.")
        
        csv_path = csvs[0]
        v_exts = ['.mp4', '.MP4', '.avi', '.mov', '.mkv']
        vid_path = None
        
        for ext in v_exts:
            p = os.path.join(folder, "recording" + ext)
            if os.path.exists(p): vid_path = p; break
            candidates = glob.glob(os.path.join(folder, "*" + ext))
            if candidates: vid_path = candidates[0]; break
            
        self.csv_path_input.setText(csv_path)
        if vid_path: self.video_path_input.setText(vid_path)
        self.check_enable_button()

    def start_analysis(self):
        self.reset_state()
        self.analyze_button.setText("PROCESSING..."); self.analyze_button.setEnabled(False)
        self.worker = VideoAnalyzerWorker(self.csv_path_input.text(), self.video_path_input.text())
        self.worker.finished.connect(self.on_success); self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_success(self, res):
        self.analyze_button.setText("RE-PROCESS"); self.analyze_button.setEnabled(True)
        if not res: return QMessageBox.info(self, "Info", "No highlights.")
        self.results_data = res; self.populate_data()
        QMessageBox.information(self, "Done", "Analysis complete.")

    def on_error(self, err):
        self.analyze_button.setText("ERROR"); self.analyze_button.setEnabled(True)
        QMessageBox.critical(self, "Error", err)

    def populate_data(self):
        for cat, data in self.results_data.items():
            segs = data.get('segments', [])
            if cat in self.tables:
                t = self.tables[cat]; t.setRowCount(len(segs))
                for i, s in enumerate(segs):
                    fmt = lambda x: f"{int(x)//60:02}:{int(x)%60:02}"
                    t.setItem(i,0, QTableWidgetItem(str(s['id'])))
                    t.setItem(i,1, QTableWidgetItem(f"{fmt(s['orig_start'])} - {fmt(s['orig_end'])}"))
                    t.setItem(i,2, QTableWidgetItem(f"{s['duration']:.1f}s"))
                    t.setItem(i,3, QTableWidgetItem(f"{s['score']:.1f}"))
                    t.item(i,0).setData(Qt.UserRole, s['montage_start'])
            if cat in self.gallery_layouts:
                l = self.gallery_layouts[cat]
                for s in segs:
                    card = VideoCardWidget(f"Seg #{s['id']} ({s['score']:.0f})", s['file_path'])
                    card.set_data(s['file_path'], f"{s['duration']:.1f}s", "")
                    card.playClicked.connect(self.force_play_video)
                    l.addWidget(card)
        self.inner_tabs.setCurrentIndex(0); self.tab_changed(0)

    def load_existing_results(self, v_path, c_path):
        v_dir = os.path.dirname(os.path.abspath(v_path))
        out_dir = os.path.join(v_dir, "Processed_Output")
        if not os.path.exists(out_dir): return False
        
        try:
            df = pd.read_csv(c_path)
            if 'Artifact' in df.columns: df = df[df['Artifact']==0]
            start_ms = df['Timestamp_ms'].iloc[0]
            
            all_segments_df = {}
            for cat_name, config in CATEGORY_CONFIG.items():
                col, thresh = config['col'], config['threshold']
                df_cat = df[df[col] >= thresh].copy() if config['condition'] == 'ge' else df[df[col] <= thresh].copy()
                all_segments_df[cat_name] = calculate_segments(df_cat, metric_col=col)
        except: 
            start_ms = 0; all_segments_df = {}
            
        res = {}; loaded = False
        for cat in CATEGORY_CONFIG:
            fname = "Focus clips" if "Focus" in cat else "Relaxation clips"
            mname = "focus montage.mp4" if "Focus" in cat else "relaxation montage.mp4"
            seg_dir = os.path.join(out_dir, fname); m_path = os.path.join(out_dir, mname)
            
            if os.path.exists(m_path) and os.path.isdir(seg_dir):
                segs = []
                files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.mp4')])
                curr_mnt = 0.0
                
                meta_df = all_segments_df.get(cat)
                
                for i, f in enumerate(files):
                    try:
                        p = f[:-4].split('_')
                        sid = int(p[1]); sc = float(p[2])
                        fpath = os.path.join(seg_dir, f)
                        with VideoFileClip(fpath) as c: dur = c.duration
                        
                        orig_s, orig_e = 0, 0
                        if meta_df is not None and not meta_df.empty:
                            row_idx = sid - 1
                            if 0 <= row_idx < len(meta_df):
                                row = meta_df.iloc[row_idx]
                                orig_s = (row['start_ms'] - start_ms)/1000
                                orig_e = (row['end_ms'] - start_ms)/1000

                        segs.append({'id': sid, 'score': sc, 'file_path': fpath, 'duration': dur, 
                                     'montage_start': curr_mnt, 'orig_start': orig_s, 'orig_end': orig_e}) 
                        curr_mnt += dur
                    except: pass
                if segs:
                    res[cat] = {'path': m_path, 'segments': segs, 'total_duration': curr_mnt}
                    loaded = True
        if loaded: self.results_data = res
        return loaded

    def load_video(self, path, autoplay=True, seek_time_ms=0):
        if not vlc or not self.media_player: return
        curr = self.media_player.get_media()
        if self.current_video_path == path and curr:
            if seek_time_ms > 0: self.media_player.set_time(seek_time_ms)
            if autoplay and not self.media_player.is_playing(): self.play_video()
            elif not autoplay and self.media_player.is_playing(): self.media_player.pause()
            return

        self.timer.stop(); self.media_player.stop()
        self.current_video_path = path
        self.media_player.set_media(self.instance.media_new(path))
        
        if sys.platform == 'win32': self.media_player.set_hwnd(int(self.video_frame.winId()))
        elif sys.platform == 'darwin': 
            try: self.media_player.set_nsobject(int(self.video_frame.winId()))
            except: pass
        else: self.media_player.set_xwindow(int(self.video_frame.winId()))
        
        self.media_player.play()
        QTimer.singleShot(100, lambda: self._finalize_load(autoplay, seek_time_ms))

    def _finalize_load(self, auto, seek):
        if not self.media_player: return
        if seek > 0: self.media_player.set_time(seek)
        if auto: 
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.timer.start()
        else: QTimer.singleShot(150, lambda: self._pause_start())
        self.play_button.setEnabled(True)

    def _pause_start(self):
        if self.media_player: 
            self.media_player.set_pause(1)
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.start()

    def play_video(self):
        if not self.media_player: return
        if self.media_player.is_playing():
            self.media_player.pause()
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            if self.current_video_path:
                self.media_player.play()
                self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                self.timer.start()

    def set_position(self, pos): 
        if self.media_player: self.media_player.set_position(pos / 1000.0)
    def set_volume(self, vol): 
        if self.media_player: self.media_player.audio_set_volume(vol)
    def update_ui_timer(self):
        if not self.media_player: return
        try:
            if self.media_player.get_state() == vlc.State.Ended:
                self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                self.position_slider.setValue(0); return
            if self.media_player.is_playing() or self.media_player.get_state() == vlc.State.Paused:
                self.position_slider.blockSignals(True)
                self.position_slider.setValue(int(self.media_player.get_position() * 1000))
                self.position_slider.blockSignals(False)
                c, t = self.media_player.get_time(), self.media_player.get_length()
                fmt = lambda x: f"{int(x/1000)//60:02}:{int(x/1000)%60:02}"
                self.time_label.setText(f"{fmt(c)} / {fmt(t)}")
        except: pass

    def tab_changed(self, idx):
        cat = 'Focus_Highlights' if idx in [0,1] else 'Relaxation_Moments'
        data = self.results_data.get(cat)
        if data and os.path.exists(data['path']): self.load_video(data['path'], False)
        else: 
            self.timer.stop(); 
            if self.media_player: self.media_player.stop()
            self.current_video_path = None
    
    def on_segment_clicked(self, r, c):
        sender = self.sender()
        cat = next((k for k,v in self.tables.items() if v == sender), None)
        if cat:
            path = self.results_data.get(cat, {}).get('path')
            seek = sender.item(r,0).data(Qt.UserRole)
            if path: self.load_video(path, True, int(seek*1000))

    def force_play_video(self, path): 
        if path and os.path.exists(path): self.load_video(path, True)
    
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.file_group.show(); self.inner_tabs.show(); self.controls_widget.show()
            self.tabs.show()
        else:
            self.file_group.hide(); self.inner_tabs.hide(); self.controls_widget.hide()
            self.tabs.hide()
            self.showFullScreen()

    def closeEvent(self, event):
        self.timer.stop()
        if self.media_player: self.media_player.stop(); self.media_player.release()
        if hasattr(self.tab_recorder, 'screen_worker') and self.tab_recorder.screen_worker:
            self.tab_recorder.screen_worker.stop()
        if hasattr(self.tab_recorder, 'brainbit_worker') and self.tab_recorder.brainbit_worker:
            self.tab_recorder.brainbit_worker.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VideoAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
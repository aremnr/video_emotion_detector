import sys
import os
import pandas as pd
import shutil
import time
import glob
import gc
import threading
import argparse
import concurrent.futures
import subprocess
from datetime import datetime
import cv2
import numpy as np
from PIL import ImageGrab
import mss
import mss.tools

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLineEdit, QLabel, QFileDialog, QTabWidget, 
    QMessageBox, QGroupBox, QSlider, QStyle, QFrame, QSizePolicy, 
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QScrollArea, QSplitter, QSpacerItem, QTextEdit, QProgressBar,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize, QRect, QThreadPool, QRunnable
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QCursor, QImage, QPixmap, QPainter

# --- MOVIEPY IMPORTS ---
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips

try:
    from moviepy.video.fx.all import speedx
except ImportError:
    try:
        from moviepy.video.fx import speedx
    except ImportError:
        speedx = None

# --- OPTIONAL LIBS (VLC) ---
try:
    import vlc
except ImportError:
    vlc = None
    pass

# --- NEUROSDK IMPORTS (BrainBit) ---
NEUROSDK_AVAILABLE = False
try:
    from neurosdk.scanner import Scanner
    from neurosdk.cmn_types import *
    NEUROSDK_AVAILABLE = True
except ImportError:
    print("Warning: 'pyneurosdk2' not found. BrainBit features disabled.")

# --- EMOTION LIBRARY (BrainBit Algorithms) ---
EMOTION_LIB_AVAILABLE = False
try:
    from em_st_artifacts.emotional_math import EmotionalMath
    from em_st_artifacts import lib_settings
    from em_st_artifacts import support_classes
    EMOTION_LIB_AVAILABLE = True
except ImportError:
    print("Warning: 'pyem-st-artifacts' not found. Emotional analysis disabled.")

# ==============================================================================
# 0. UI STYLES
# ==============================================================================
STYLESHEET = """
QMainWindow, QWidget {
    background-color: #181818; 
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
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
QLineEdit:focus { border: 1px solid #3ea6ff; }

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

/* Exit Fullscreen Button */
QPushButton#OverlayExitBtn {
    background-color: rgba(0, 0, 0, 0.8);
    border: 1px solid #fff;
    border-radius: 4px;
    color: white;
    font-weight: bold;
    font-size: 14px;
    padding: 8px 16px;
    min-width: 120px;
}
QPushButton#OverlayExitBtn:hover {
    background-color: rgba(255, 255, 255, 0.2);
}

/* BrainBit Status */
QLabel#BrainBitStatus {
    font-weight: bold;
    padding: 0 10px;
}

QProgressBar {
    border: 1px solid #444;
    border-radius: 2px;
    background-color: #121212;
    text-align: center;
    color: white;
}
QProgressBar::chunk {
    background-color: #3ea6ff;
}

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

/* Recorder specific */
QPushButton#RecordBtn {
    background-color: #ff3b30;
    color: white;
    font-weight: bold;
    padding: 12px;
}
QPushButton#RecordBtn:hover { background-color: #ff5e54; }
QPushButton#RecordBtn:disabled { background-color: #5a2c2c; color: #777; }

QPushButton#StopBtn {
    background-color: #3ea6ff;
    color: white;
    font-weight: bold;
    padding: 12px;
}
QPushButton#StopBtn:hover { background-color: #65b8ff; }
QPushButton#StopBtn:disabled { background-color: #2a4a66; color: #777; }
"""

# ==============================================================================
# 1. ANALYSIS LOGIC
# ==============================================================================

CATEGORY_CONFIG = {
    'Focus_Highlights': {'col': 'Focus', 'threshold': 70, 'condition': 'ge'},
    'Relaxation_Moments': {'col': 'Relaxation', 'threshold': 60, 'condition': 'ge'} 
}

MIN_SEGMENT_DURATION_SEC = 2.0 
MAX_GAP_MS = 2000

def safe_subclip(clip, start, end):
    if hasattr(clip, 'subclipped'): return clip.subclipped(start, end)
    if hasattr(clip, 'subclip'): return clip.subclip(start, end)
    return clip

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

def analyze_and_concatenate_video(csv_path, video_file, progress_callback=None):
    def update_progress(val, msg):
        if progress_callback:
            progress_callback(val, msg)

    update_progress(0, "Validating files...")
    if not os.path.exists(csv_path): raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(video_file): raise FileNotFoundError(f"Video not found: {video_file}")
    
    update_progress(5, "Reading CSV data...")
    df = pd.read_csv(csv_path)
    req = ['Timestamp_ms', 'Focus', 'Relaxation']
    if not all(c in df.columns for c in req): raise ValueError(f"Missing cols: {req}")
    if 'Artifact' in df.columns:
        df = df[df['Artifact'] == 0].copy()

    start_time_ms = df['Timestamp_ms'].iloc[0]
    
    base_dir = os.path.join(os.path.expanduser('~'), 'video', 'NeuroProjects')
    os.makedirs(base_dir, exist_ok=True)
    
    project_num = 1
    while os.path.exists(os.path.join(base_dir, f'Project_{project_num}')):
        project_num += 1
        
    output_root_folder = os.path.join(base_dir, f'Project_{project_num}')
    os.makedirs(output_root_folder, exist_ok=True)
    
    csv_filename = os.path.basename(csv_path)
    dest_csv_path = os.path.join(output_root_folder, csv_filename)
    shutil.copy2(csv_path, dest_csv_path)
    
    update_progress(10, "Loading video file (this may take a moment)...")
    results_data = {} 
    full_clip = None

    try:
        full_clip = VideoFileClip(video_file)
        all_segments_df = {}
        
        for cat_name, config in CATEGORY_CONFIG.items():
            col, thresh = config['col'], config['threshold']
            df_cat = df[df[col] >= thresh].copy() if config['condition'] == 'ge' else df[df[col] <= thresh].copy()
            all_segments_df[cat_name] = calculate_segments(df_cat, metric_col=col)

        total_clips = sum(len(segs) for segs in all_segments_df.values())
        processed_clips_count = 0
        
        if total_clips == 0:
            update_progress(100, "No highlights found.")
            return {}, output_root_folder

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
                
                processed_clips_count += 1
                prog_val = 10 + int((processed_clips_count / total_clips) * 70)
                update_progress(prog_val, f"Extracting {category}: Clip {idx+1}/{len(segments)}")

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
                update_progress(prog_val + 5, f"Rendering montage for {category}...")
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
    
    update_progress(100, "Processing Complete!")
    return results_data, output_root_folder

# ==============================================================================
# 2. RECORDING LOGIC
# ==============================================================================

class ScreenRecorder:
    def __init__(self, output_path, fps=30, monitor=1):
        self.output_path = output_path
        self.fps = fps
        self.monitor = monitor
        self.is_recording = False
        self.video_writer = None
        self.frame_count = 0
        self.start_time = None
        
    def start(self):
        self.is_recording = True
        self.frame_count = 0
        self.start_time = time.time()
        
        # Get screen size using mss
        with mss.mss() as sct:
            monitor_info = sct.monitors[self.monitor]
            width = monitor_info['width']
            height = monitor_info['height']
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            (width, height)
        )
        
    def capture_frame(self):
        if not self.is_recording or not self.video_writer:
            return
            
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[self.monitor])
            
            # Convert to numpy array
            img = np.array(screenshot)
            # Convert BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            self.video_writer.write(img)
            self.frame_count += 1
    
    def stop(self):
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
    def get_recording_time(self):
        if self.start_time:
            return time.time() - self.start_time
        return 0

class BrainBitDataCollector:
    def __init__(self, sensor, emotion_lib=None):
        self.sensor = sensor
        self.emotion_lib = emotion_lib
        self.is_collecting = False
        self.data = []
        self.start_time = None
        self.sample_count = 0
        
    def start(self):
        self.is_collecting = True
        self.data = []
        self.start_time = time.time() * 1000  # Convert to milliseconds
        self.sample_count = 0
        
        # Start signal if sensor supports it
        if self.sensor and hasattr(self.sensor, 'exec_command'):
            try:
                self.sensor.exec_command(SensorCommand.CommandStartSignal)
            except:
                pass
        
    def collect_sample(self, relaxation=0, focus=0, artifact=0):
        if not self.is_collecting:
            return
            
        timestamp = int(time.time() * 1000 - self.start_time)
        self.data.append({
            'Timestamp_ms': timestamp,
            'Relaxation': relaxation,
            'Focus': focus,
            'Artifact': artifact
        })
        self.sample_count += 1
        
    def stop(self):
        self.is_collecting = False
        
        # Stop signal if sensor supports it
        if self.sensor and hasattr(self.sensor, 'exec_command'):
            try:
                self.sensor.exec_command(SensorCommand.CommandStopSignal)
            except:
                pass
        
    def save_to_csv(self, filepath):
        if not self.data:
            return False
            
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)
        return True

# ==============================================================================
# 3. WORKER THREADS
# ==============================================================================

class VideoAnalyzerWorker(QThread):
    finished = pyqtSignal(dict, str) 
    progress = pyqtSignal(int, str)  
    error = pyqtSignal(str)

    def __init__(self, csv, video):
        super().__init__()
        self.csv, self.video = csv, video

    def run(self):
        try:
            res, proj_path = analyze_and_concatenate_video(
                self.csv, 
                self.video, 
                progress_callback=lambda p, m: self.progress.emit(p, m)
            )
            self.finished.emit(res, proj_path)
        except Exception as e: 
            self.error.emit(str(e))

class BrainBitWorker(QThread):
    finished = pyqtSignal(object, str) # (sensor_object, message)
    log = pyqtSignal(str)
    battery_signal = pyqtSignal(int)
    state_signal = pyqtSignal(str)
    signal_data = pyqtSignal(list)  # New signal for raw signal data

    def __init__(self, target_serial=None):
        super().__init__()
        self.target_serial = target_serial
        self.sensor_info_list = []
        self.collect_signal = False

    def on_sensor_found(self, scanner, sensors):
        for index in range(len(sensors)):
            msg = f"Sensor found: {sensors[index].Name} [{sensors[index].SerialNumber}]"
            self.log.emit(msg)

    def on_sensor_state_changed(self, sensor, state):
        msg = f"Sensor {sensor.Name} state: {state}"
        self.log.emit(msg)
        self.state_signal.emit(str(state))

    def on_battery_changed(self, sensor, battery):
        msg = f"Battery: {battery}%"
        self.log.emit(msg)
        self.battery_signal.emit(battery)
        
    def on_signal_received(self, sensor, data):
        # Emit signal data for processing
        if self.collect_signal:
            self.signal_data.emit(data)

    def run(self):
        if not NEUROSDK_AVAILABLE:
            self.finished.emit(None, "SDK not installed")
            return

        try:
            # 1. Setup Scanner
            scanner = Scanner([SensorFamily.LEBrainBit, SensorFamily.LEBrainBitBlack])
            scanner.sensorsChanged = self.on_sensor_found
            
            # 2. Start Search
            self.log.emit("Starting search for 5 sec...")
            scanner.start()
            time.sleep(5)
            scanner.stop()

            # 3. Get results
            sensors_info = scanner.sensors()
            if not sensors_info:
                self.log.emit("No sensors found.")
                self.finished.emit(None, "No devices found")
                del scanner
                return

            # 4. Determine target sensor
            target_sensor_info = None
            if self.target_serial:
                self.log.emit(f"Looking for serial: {self.target_serial}")
                for s in sensors_info:
                    if s.SerialNumber == self.target_serial:
                        target_sensor_info = s
                        break
                if not target_sensor_info:
                    self.finished.emit(None, f"Device {self.target_serial} not found in scan results.")
                    del scanner
                    return
            else:
                target_sensor_info = sensors_info[0]

            self.log.emit(f"Connecting to {target_sensor_info.Name} ({target_sensor_info.SerialNumber})...")

            # 5. Connect using ThreadPoolExecutor
            def device_connection(s_info):
                return scanner.create_sensor(s_info)

            sensor = None
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(device_connection, target_sensor_info)
                sensor = future.result()
                self.log.emit("Device connected (created).")

            # 6. Configure Sensor Callbacks
            sensor.sensorStateChanged = self.on_sensor_state_changed
            sensor.batteryChanged = self.on_battery_changed
            
            # Setup signal callback
            if sensor.is_supported_feature(SensorFeature.Signal):
                sensor.signalDataReceived = self.on_signal_received
            
            # Print details
            self.log.emit(f"Family: {sensor.sens_family}")
            self.log.emit(f"Name: {sensor.name}")
            self.log.emit(f"State: {sensor.state}")
            self.log.emit(f"Serial: {sensor.serial_number}")
            self.log.emit(f"Battery: {sensor.batt_power}%")

            # Check connection state
            if sensor.state == SensorState.StateInRange:
                self.log.emit("Connection verified: StateInRange")
                self.finished.emit(sensor, "Connected")
            else:
                self.finished.emit(sensor, "Created but not InRange")

            # 7. Cleanup Scanner
            del scanner
            self.log.emit("Scanner removed.")

        except Exception as e:
            self.log.emit(f"Error: {e}")
            self.finished.emit(None, str(e))

class RecordingWorker(QThread):
    progress = pyqtSignal(int, str, float, float)  # progress, status, focus, relaxation
    finished = pyqtSignal(str, str)  # video_path, csv_path
    error = pyqtSignal(str)
    
    def __init__(self, output_dir, fps=30, monitor=1, sensor=None):
        super().__init__()
        self.output_dir = output_dir
        self.fps = fps
        self.monitor = monitor
        self.sensor = sensor
        self.is_recording = False
        self.recorder = None
        self.data_collector = None
        self.emotion_lib = None
        
    def init_emotion_lib(self):
        """Initialize emotion analysis library if available"""
        if not EMOTION_LIB_AVAILABLE:
            return None
            
        try:
            # Default settings for BrainBit
            mls = lib_settings.MathLibSetting(
                sampling_rate=250,
                process_win_freq=25,
                n_first_sec_skipped=4,
                fft_window=1000,
                bipolar_mode=True,
                squared_spectrum=True,
                channels_number=4,
                channel_for_analysis=0
            )
            
            ads = lib_settings.ArtifactDetectSetting(
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
            
            sads = lib_settings.ShortArtifactDetectSetting(
                ampl_art_detect_win_size=200,
                ampl_art_zerod_area=200,
                ampl_art_extremum_border=25
            )
            
            mss = lib_settings.MentalAndSpectralSetting(
                n_sec_for_averaging=2,
                n_sec_for_instant_estimation=4
            )
            
            math = EmotionalMath(mls, ads, sads, mss)
            math.set_calibration_length(6)  # 6 seconds calibration
            math.set_mental_estimation_mode(False)
            math.set_skip_wins_after_artifact(10)
            math.set_zero_spect_waves(True, 0, 1, 1, 1, 0)
            math.set_spect_normalization_by_bands_width(True)
            
            return math
            
        except Exception as e:
            print(f"Failed to init emotion lib: {e}")
            return None
    
    def on_signal_data(self, data):
        """Process incoming signal data from BrainBit"""
        if not self.is_recording or not self.emotion_lib:
            return
            
        try:
            raw_channels = []
            for sample in data:
                # Calculate bipolar channels (T3-O1 and T4-O2)
                left_bipolar = sample.T3 - sample.O1
                right_bipolar = sample.T4 - sample.O2
                raw_channels.append(support_classes.RawChannels(left_bipolar, right_bipolar))
            
            # Push data to emotion library
            self.emotion_lib.push_data(raw_channels)
            self.emotion_lib.process_data_arr()
            
            # Read mental data if calibration is finished
            if self.emotion_lib.calibration_finished():
                mental_data = self.emotion_lib.read_mental_data_arr()
                if mental_data:
                    latest = mental_data[-1]
                    # Emit focus and relaxation values
                    self.progress.emit(-1, "", latest.Inst_Attention, latest.Inst_Relaxation)
            
        except Exception as e:
            print(f"Error processing signal: {e}")
    
    def run(self):
        try:
            self.is_recording = True
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Generate filenames with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
            csv_path = os.path.join(self.output_dir, f"data_{timestamp}.csv")
            
            # Initialize screen recorder
            self.recorder = ScreenRecorder(video_path, self.fps, self.monitor)
            
            # Initialize data collector
            self.data_collector = BrainBitDataCollector(self.sensor)
            
            # Initialize emotion library if sensor is available
            if self.sensor:
                self.emotion_lib = self.init_emotion_lib()
                if self.emotion_lib:
                    # Start calibration
                    self.emotion_lib.start_calibration()
            
            # Start recording
            self.recorder.start()
            self.data_collector.start()
            
            # Start signal collection if sensor is available
            if self.sensor and hasattr(self.sensor, 'exec_command'):
                try:
                    self.sensor.exec_command(SensorCommand.CommandStartSignal)
                except:
                    pass
            
            start_time = time.time()
            calibration_done = False
            
            # Main recording loop
            while self.is_recording:
                elapsed = time.time() - start_time
                
                # Capture screen frame
                self.recorder.capture_frame()
                
                # Update progress
                progress = int((elapsed % 60) / 60 * 100)  # Progress within each minute
                status = f"Recording... {int(elapsed)}s"
                
                # Handle emotion library calibration
                if self.emotion_lib and not calibration_done:
                    if elapsed >= 6:  # After 6 seconds
                        calibration_done = True
                        status += " (Calibration complete)"
                    
                    # Get calibration progress
                    calibration_progress = self.emotion_lib.get_calibration_percents()
                    status += f" [Calibrating: {calibration_progress:.0f}%]"
                
                # Collect sample with default or calculated values
                focus_val = 50  # Default
                relax_val = 50  # Default
                artifact_val = 0
                
                # Check for artifacts if emotion lib is available
                if self.emotion_lib:
                    if self.emotion_lib.is_both_sides_artifacted():
                        artifact_val = 1
                
                self.data_collector.collect_sample(
                    relaxation=relax_val,
                    focus=focus_val,
                    artifact=artifact_val
                )
                
                # Emit progress
                self.progress.emit(progress, status, focus_val, relax_val)
                
                # Sleep to maintain FPS
                time.sleep(1.0 / self.fps)
                
        except Exception as e:
            self.error.emit(str(e))
        finally:
            # Stop everything
            self.is_recording = False
            
            if self.recorder:
                self.recorder.stop()
            
            if self.data_collector:
                self.data_collector.stop()
                # Save CSV data
                self.data_collector.save_to_csv(csv_path)
            
            # Stop signal if sensor is available
            if self.sensor and hasattr(self.sensor, 'exec_command'):
                try:
                    self.sensor.exec_command(SensorCommand.CommandStopSignal)
                except:
                    pass
            
            # Cleanup emotion library
            if self.emotion_lib:
                del self.emotion_lib
            
            # Emit finished signal
            self.finished.emit(video_path, csv_path)

# ==============================================================================
# 4. GUI WIDGETS
# ==============================================================================

class VideoClickableFrame(QFrame):
    doubleClicked = pyqtSignal()
    exitFullscreen = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.exit_btn = QPushButton("✕ Exit Fullscreen (Esc)", self)
        self.exit_btn.setObjectName("OverlayExitBtn")
        self.exit_btn.setFixedSize(150, 40)
        self.exit_btn.hide()
        self.exit_btn.clicked.connect(self.exitFullscreen.emit)
        
        self.controls_overlay = QWidget(self)
        self.controls_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 0.5);")
        self.controls_overlay.hide()
        
        self.play_btn = QPushButton(self)
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.setIconSize(QSize(64, 64))
        self.play_btn.setFixedSize(80, 80)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 0.7);
                border: 2px solid white;
                border-radius: 40px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 0.9);
            }
        """)
        self.play_btn.hide()
        
        self.setMouseTracking(True)
        self._is_fs = False
        self._controls_visible = False

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
    
    def mouseMoveEvent(self, event):
        if self._is_fs:
            pos = event.pos()
            at_bottom = pos.y() > self.height() - 100
            at_center = (self.width()/2 - 100 < pos.x() < self.width()/2 + 100 and 
                        self.height()/2 - 100 < pos.y() < self.height()/2 + 100)
            
            self.exit_btn.move(self.width() - self.exit_btn.width() - 20, 20)
            self.exit_btn.setVisible(True)
            
            self.play_btn.move(self.width()//2 - 40, self.height()//2 - 40)
            self.play_btn.setVisible(at_center)
            
            self.controls_overlay.setVisible(at_bottom)
            if at_bottom:
                self.controls_overlay.setGeometry(0, self.height() - 60, self.width(), 60)
    
    def set_fullscreen_mode(self, enabled):
        self._is_fs = enabled
        if enabled:
            self.controls_overlay.show()
            self.exit_btn.show()
            self.play_btn.show()
            self.setCursor(Qt.ArrowCursor)
            QTimer.singleShot(3000, lambda: self.setCursor(Qt.BlankCursor) if not self.underMouse() else None)
        else:
            self.controls_overlay.hide()
            self.exit_btn.hide()
            self.play_btn.hide()
            self.unsetCursor()

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
    def __init__(self, cli_video=None, cli_csv=None):
        super().__init__()
        self.setWindowTitle("Neuro Video Analyzer & Recorder")
        self.setGeometry(50, 50, 1000, 850) 
        self.setStyleSheet(STYLESHEET)
        
        self.instance = None
        self.media_player = None
        self.results_data = {} 
        self.current_video_path = None
        
        # BrainBit Data
        self.brainbit_sensor = None
        self.brainbit_worker = None
        
        # Recording
        self.recording_worker = None
        self.is_recording = False
        
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_ui_timer)

        self.init_vlc()
        self.init_ui()

        if cli_video and cli_csv:
            self.video_path_input.setText(os.path.abspath(cli_video))
            self.csv_path_input.setText(os.path.abspath(cli_csv))
            self.check_enable_button()
            QTimer.singleShot(100, self.start_analysis)
        
    def init_vlc(self):
        if vlc:
            try:
                self.instance = vlc.Instance("--no-xlib", "--quiet", "--no-video-title-show")
                self.media_player = vlc.MediaPlayer(self.instance)
            except: self.media_player = None

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main tab widget
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.North)
        
        # Create analyzer and recorder tabs
        self.analyzer_widget = QWidget()
        self.recorder_widget = QWidget()
        
        self.build_analyzer_ui(self.analyzer_widget)
        self.build_recorder_ui(self.recorder_widget)
        
        # Add tabs to main widget
        self.main_tabs.addTab(self.analyzer_widget, "🎬 Analyzer")
        self.main_tabs.addTab(self.recorder_widget, "📹 Recorder")
        
        # Set main layout
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.addWidget(self.main_tabs)
        
    def build_analyzer_ui(self, parent_widget):
        main_layout = QVBoxLayout(parent_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 10, 20, 20)

        # 0. HEADER (BrainBit Connection)
        header_layout = QHBoxLayout()
        header_layout.addStretch()
        
        lbl_sn = QLabel("BrainBit SN:")
        lbl_sn.setStyleSheet("color: #888;")
        header_layout.addWidget(lbl_sn)
        
        self.bb_serial_input = QLineEdit()
        self.bb_serial_input.setPlaceholderText("Serial (Opt)")
        self.bb_serial_input.setFixedWidth(120)
        header_layout.addWidget(self.bb_serial_input)
        
        self.btn_connect_bb = QPushButton("Connect")
        self.btn_connect_bb.setFixedWidth(80)
        self.btn_connect_bb.clicked.connect(self.toggle_brainbit_connection)
        header_layout.addWidget(self.btn_connect_bb)
        
        self.lbl_bb_status = QLabel("●")
        self.lbl_bb_status.setObjectName("BrainBitStatus")
        self.lbl_bb_status.setStyleSheet("color: #444; font-size: 18px;")
        header_layout.addWidget(self.lbl_bb_status)
        
        main_layout.addLayout(header_layout)

        # 1. Project Configuration
        self.file_group = QGroupBox("Analysis Source Files")
        flayout = QVBoxLayout(self.file_group)
        
        self.btn_load_project = QPushButton("📂 OPEN PROJECT FOLDER")
        self.btn_load_project.setObjectName("LoadFolderBtn")
        self.btn_load_project.setCursor(Qt.PointingHandCursor)
        self.btn_load_project.setStyleSheet("padding: 12px; font-weight: bold;")
        self.btn_load_project.clicked.connect(self.open_project_folder)
        flayout.addWidget(self.btn_load_project)

        r1 = QHBoxLayout()
        self.csv_path_input = QLineEdit(); self.csv_path_input.setPlaceholderText("Select CSV File...")
        b1 = QPushButton("Browse CSV"); b1.clicked.connect(lambda: self.browse_file('csv'))
        r1.addWidget(self.csv_path_input); r1.addWidget(b1)
        flayout.addLayout(r1)
        
        r2 = QHBoxLayout()
        self.video_path_input = QLineEdit(); self.video_path_input.setPlaceholderText("Select Video File...")
        b2 = QPushButton("Browse Video"); b2.clicked.connect(lambda: self.browse_file('video'))
        r2.addWidget(self.video_path_input); r2.addWidget(b2)
        flayout.addLayout(r2)
        
        action_layout = QHBoxLayout()
        self.analyze_button = QPushButton("START PROCESSING")
        self.analyze_button.setObjectName("ActionBtn")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)
        action_layout.addWidget(self.analyze_button)
        flayout.addLayout(action_layout)
        
        self.progress_container = QWidget()
        prog_layout = QVBoxLayout(self.progress_container)
        prog_layout.setContentsMargins(0, 10, 0, 0)
        self.lbl_progress_status = QLabel("Ready")
        self.lbl_progress_status.setStyleSheet("color: #888; font-style: italic;")
        prog_layout.addWidget(self.lbl_progress_status)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        prog_layout.addWidget(self.progress_bar)
        flayout.addWidget(self.progress_container)
        main_layout.addWidget(self.file_group)

        # 2. Video Player Area
        self.video_container = QWidget()
        v_layout = QVBoxLayout(self.video_container)
        v_layout.setContentsMargins(0, 0, 0, 0); v_layout.setSpacing(0)
        
        self.video_frame = VideoClickableFrame()
        self.video_frame.setStyleSheet("background-color: #000; border: 1px solid #444;")
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_frame.setMinimumHeight(350)
        self.video_frame.doubleClicked.connect(self.toggle_fullscreen)
        self.video_frame.exitFullscreen.connect(self.exit_fullscreen)
        if not vlc: 
            self.video_frame.setLayout(QVBoxLayout())
            lbl = QLabel("VLC Player not found.\nInstall python-vlc and VLC Media Player to view video.")
            lbl.setAlignment(Qt.AlignCenter)
            self.video_frame.layout().addWidget(lbl)
        v_layout.addWidget(self.video_frame)
        
        # 3. Player Controls
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background-color: #202020; border-bottom-left-radius: 6px; border-bottom-right-radius: 6px;")
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

        # 4. Results Tabs
        self.inner_tabs = QTabWidget()
        self.inner_tabs.setMinimumHeight(250)
        self.inner_tabs.currentChanged.connect(self.tab_changed)
        self.tables = {}; self.gallery_layouts = {}
        w1, t1 = self.create_list_tab(); self.tables['Focus_Highlights'] = t1; self.inner_tabs.addTab(w1, "Focus (List)")
        w2, l2 = self.create_gallery_tab(); self.gallery_layouts['Focus_Highlights'] = l2; self.inner_tabs.addTab(w2, "Focus (Gallery)")
        w3, t3 = self.create_list_tab(); self.tables['Relaxation_Moments'] = t3; self.inner_tabs.addTab(w3, "Relaxation (List)")
        w4, l4 = self.create_gallery_tab(); self.gallery_layouts['Relaxation_Moments'] = l4; self.inner_tabs.addTab(w4, "Relaxation (Gallery)")
        main_layout.addWidget(self.inner_tabs)
        self.check_enable_button()

    def build_recorder_ui(self, parent_widget):
        layout = QVBoxLayout(parent_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 1. BrainBit Connection
        bb_group = QGroupBox("BrainBit Connection")
        bb_layout = QVBoxLayout(bb_group)
        
        conn_layout = QHBoxLayout()
        lbl_sn = QLabel("Serial Number:")
        self.recorder_bb_serial = QLineEdit()
        self.recorder_bb_serial.setPlaceholderText("Leave empty to auto-detect")
        self.btn_recorder_connect = QPushButton("Connect")
        self.btn_recorder_connect.clicked.connect(self.toggle_recorder_brainbit)
        
        conn_layout.addWidget(lbl_sn)
        conn_layout.addWidget(self.recorder_bb_serial)
        conn_layout.addWidget(self.btn_recorder_connect)
        conn_layout.addWidget(QLabel("Status:"))
        
        self.lbl_recorder_bb_status = QLabel("●")
        self.lbl_recorder_bb_status.setObjectName("BrainBitStatus")
        self.lbl_recorder_bb_status.setStyleSheet("color: #444; font-size: 18px;")
        conn_layout.addWidget(self.lbl_recorder_bb_status)
        conn_layout.addStretch()
        
        bb_layout.addLayout(conn_layout)
        
        # BrainBit Info
        info_layout = QHBoxLayout()
        self.lbl_bb_info = QLabel("Not connected")
        self.lbl_bb_info.setStyleSheet("color: #888; font-style: italic;")
        info_layout.addWidget(self.lbl_bb_info)
        info_layout.addStretch()
        
        self.lbl_battery = QLabel("Battery: --%")
        info_layout.addWidget(self.lbl_battery)
        
        bb_layout.addLayout(info_layout)
        layout.addWidget(bb_group)
        
        # 2. Recording Settings
        settings_group = QGroupBox("Recording Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # FPS Setting
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frame Rate (FPS):"))
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(30)
        fps_layout.addWidget(self.spin_fps)
        fps_layout.addStretch()
        settings_layout.addLayout(fps_layout)
        
        # Monitor Selection
        monitor_layout = QHBoxLayout()
        monitor_layout.addWidget(QLabel("Monitor:"))
        self.combo_monitor = QComboBox()
        self.combo_monitor.addItems(["Primary Monitor", "Monitor 2", "Monitor 3", "Monitor 4"])
        monitor_layout.addWidget(self.combo_monitor)
        monitor_layout.addStretch()
        settings_layout.addLayout(monitor_layout)
        
        # Output Directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.recorder_output_dir = QLineEdit()
        self.recorder_output_dir.setText(os.path.join(os.path.expanduser('~'), 'video', 'Recordings'))
        self.btn_browse_output = QPushButton("Browse")
        self.btn_browse_output.clicked.connect(self.browse_recording_output)
        output_layout.addWidget(self.recorder_output_dir)
        output_layout.addWidget(self.btn_browse_output)
        settings_layout.addLayout(output_layout)
        
        # Enable emotion analysis
        self.check_enable_emotion = QCheckBox("Enable Emotional Analysis (requires pyem-st-artifacts)")
        self.check_enable_emotion.setChecked(True)
        self.check_enable_emotion.setEnabled(EMOTION_LIB_AVAILABLE)
        if not EMOTION_LIB_AVAILABLE:
            self.check_enable_emotion.setToolTip("Install pyem-st-artifacts for emotional analysis")
        settings_layout.addWidget(self.check_enable_emotion)
        
        layout.addWidget(settings_group)
        
        # 3. Recording Controls
        controls_group = QGroupBox("Recording Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Status and Progress
        self.lbl_recording_status = QLabel("Ready to record")
        self.lbl_recording_status.setStyleSheet("font-size: 14px; font-weight: bold;")
        controls_layout.addWidget(self.lbl_recording_status)
        
        self.recording_progress = QProgressBar()
        self.recording_progress.setRange(0, 100)
        self.recording_progress.setValue(0)
        self.recording_progress.setVisible(False)
        controls_layout.addWidget(self.recording_progress)
        
        # Metrics Display
        metrics_layout = QHBoxLayout()
        self.lbl_focus = QLabel("Focus: --")
        self.lbl_relaxation = QLabel("Relaxation: --")
        self.lbl_artifact = QLabel("Artifact: No")
        
        metrics_layout.addWidget(self.lbl_focus)
        metrics_layout.addWidget(self.lbl_relaxation)
        metrics_layout.addWidget(self.lbl_artifact)
        metrics_layout.addStretch()
        controls_layout.addLayout(metrics_layout)
        
        # Control Buttons
        button_layout = QHBoxLayout()
        self.btn_start_recording = QPushButton("⏺ START RECORDING")
        self.btn_start_recording.setObjectName("RecordBtn")
        self.btn_start_recording.clicked.connect(self.start_recording)
        self.btn_start_recording.setEnabled(False)
        
        self.btn_stop_recording = QPushButton("⏹ STOP")
        self.btn_stop_recording.setObjectName("StopBtn")
        self.btn_stop_recording.clicked.connect(self.stop_recording)
        self.btn_stop_recording.setEnabled(False)
        
        button_layout.addWidget(self.btn_start_recording)
        button_layout.addWidget(self.btn_stop_recording)
        button_layout.addStretch()
        
        controls_layout.addLayout(button_layout)
        layout.addWidget(controls_group)
        
        # 4. Recent Recordings
        recent_group = QGroupBox("Recent Recordings")
        recent_layout = QVBoxLayout(recent_group)
        
        self.list_recent = QTextEdit()
        self.list_recent.setReadOnly(True)
        self.list_recent.setMaximumHeight(150)
        self.list_recent.setStyleSheet("background-color: #121212; border: 1px solid #333;")
        recent_layout.addWidget(self.list_recent)
        
        layout.addWidget(recent_group)
        layout.addStretch()

    # --- BRAINBIT METHODS (Analyzer Tab) ---
    def toggle_brainbit_connection(self):
        if self.brainbit_sensor is not None:
            # Disconnect
            try:
                self.brainbit_sensor.disconnect()
                del self.brainbit_sensor
            except Exception as e:
                print(f"Disconnect error: {e}")
                
            self.brainbit_sensor = None
            self.lbl_bb_status.setStyleSheet("color: #444;") # Gray
            self.btn_connect_bb.setText("Connect")
            self.bb_serial_input.setEnabled(True)
            self.lbl_progress_status.setText("Disconnected from BrainBit.")
        else:
            # Connect
            serial = self.bb_serial_input.text().strip()
            
            self.btn_connect_bb.setEnabled(False)
            self.btn_connect_bb.setText("Scanning...")
            self.lbl_bb_status.setStyleSheet("color: #FFD700;") # Yellow
            self.bb_serial_input.setEnabled(False)
            
            self.brainbit_worker = BrainBitWorker(serial if serial else None)
            self.brainbit_worker.log.connect(lambda s: self.lbl_progress_status.setText(s))
            self.brainbit_worker.finished.connect(self.on_brainbit_connected)
            self.brainbit_worker.start()

    def on_brainbit_connected(self, sensor, msg):
        self.btn_connect_bb.setEnabled(True)
        if sensor:
            self.brainbit_sensor = sensor
            self.lbl_bb_status.setStyleSheet("color: #00FF00;") # Green
            self.btn_connect_bb.setText("Disconnect")
            self.lbl_progress_status.setText(f"Connected: {sensor.serial_number}")
            QMessageBox.information(self, "BrainBit", f"Successfully connected to {sensor.name}!")
        else:
            self.brainbit_sensor = None
            self.lbl_bb_status.setStyleSheet("color: #FF0000;") # Red
            self.btn_connect_bb.setText("Connect")
            self.bb_serial_input.setEnabled(True)
            self.lbl_progress_status.setText(msg)
            QMessageBox.warning(self, "Connection Failed", msg)

    # --- BRAINBIT METHODS (Recorder Tab) ---
    def toggle_recorder_brainbit(self):
        if hasattr(self, 'recorder_brainbit_sensor') and self.recorder_brainbit_sensor is not None:
            # Disconnect
            try:
                self.recorder_brainbit_sensor.disconnect()
                del self.recorder_brainbit_sensor
            except Exception as e:
                print(f"Disconnect error: {e}")
                
            self.recorder_brainbit_sensor = None
            self.lbl_recorder_bb_status.setStyleSheet("color: #444;")
            self.btn_recorder_connect.setText("Connect")
            self.recorder_bb_serial.setEnabled(True)
            self.lbl_bb_info.setText("Not connected")
            self.btn_start_recording.setEnabled(False)
        else:
            # Connect
            serial = self.recorder_bb_serial.text().strip()
            
            self.btn_recorder_connect.setEnabled(False)
            self.btn_recorder_connect.setText("Scanning...")
            self.lbl_recorder_bb_status.setStyleSheet("color: #FFD700;")
            self.recorder_bb_serial.setEnabled(False)
            
            self.recorder_brainbit_worker = BrainBitWorker(serial if serial else None)
            self.recorder_brainbit_worker.log.connect(lambda s: self.lbl_bb_info.setText(s))
            self.recorder_brainbit_worker.battery_signal.connect(self.update_battery_display)
            self.recorder_brainbit_worker.finished.connect(self.on_recorder_brainbit_connected)
            self.recorder_brainbit_worker.start()

    def on_recorder_brainbit_connected(self, sensor, msg):
        self.btn_recorder_connect.setEnabled(True)
        if sensor:
            self.recorder_brainbit_sensor = sensor
            self.lbl_recorder_bb_status.setStyleSheet("color: #00FF00;")
            self.btn_recorder_connect.setText("Disconnect")
            self.lbl_bb_info.setText(f"Connected: {sensor.name} ({sensor.serial_number})")
            self.btn_start_recording.setEnabled(True)
            
            # Update battery display
            if hasattr(sensor, 'batt_power'):
                self.update_battery_display(sensor.batt_power)
        else:
            self.recorder_brainbit_sensor = None
            self.lbl_recorder_bb_status.setStyleSheet("color: #FF0000;")
            self.btn_recorder_connect.setText("Connect")
            self.recorder_bb_serial.setEnabled(True)
            self.lbl_bb_info.setText(msg)
            self.btn_start_recording.setEnabled(False)

    def update_battery_display(self, battery):
        self.lbl_battery.setText(f"Battery: {battery}%")

    # --- RECORDING METHODS ---
    def browse_recording_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.recorder_output_dir.setText(folder)

    def start_recording(self):
        if self.is_recording:
            return
            
        # Check output directory
        output_dir = self.recorder_output_dir.text()
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please specify an output folder.")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get recording settings
        fps = self.spin_fps.value()
        monitor = self.combo_monitor.currentIndex() + 1  # Convert to 1-based index
        
        # Start recording worker
        self.is_recording = True
        self.recording_worker = RecordingWorker(
            output_dir=output_dir,
            fps=fps,
            monitor=monitor,
            sensor=self.recorder_brainbit_sensor
        )
        
        # Connect signals
        self.recording_worker.progress.connect(self.update_recording_progress)
        self.recording_worker.finished.connect(self.on_recording_finished)
        self.recording_worker.error.connect(self.on_recording_error)
        
        # Update UI
        self.btn_start_recording.setEnabled(False)
        self.btn_stop_recording.setEnabled(True)
        self.recording_progress.setVisible(True)
        self.lbl_recording_status.setText("Recording started...")
        
        # Start worker
        self.recording_worker.start()

    def update_recording_progress(self, progress, status, focus, relaxation):
        self.recording_progress.setValue(progress)
        self.lbl_recording_status.setText(status)
        
        # Update metrics
        self.lbl_focus.setText(f"Focus: {focus:.1f}")
        self.lbl_relaxation.setText(f"Relaxation: {relaxation:.1f}")

    def stop_recording(self):
        if not self.is_recording or not self.recording_worker:
            return
            
        self.is_recording = False
        self.recording_worker.is_recording = False
        
        # Update UI immediately
        self.btn_stop_recording.setEnabled(False)
        self.lbl_recording_status.setText("Stopping recording...")

    def on_recording_finished(self, video_path, csv_path):
        self.is_recording = False
        
        # Update UI
        self.btn_start_recording.setEnabled(True)
        self.btn_stop_recording.setEnabled(False)
        self.recording_progress.setVisible(False)
        self.lbl_recording_status.setText("Recording complete!")
        
        # Update recent recordings list
        self.update_recent_recordings(video_path, csv_path)
        
        # Show success message
        QMessageBox.information(self, "Recording Complete", 
                              f"Recording saved successfully!\n\n"
                              f"Video: {os.path.basename(video_path)}\n"
                              f"CSV Data: {os.path.basename(csv_path)}")
        
        # Switch to analyzer tab with the new files
        self.main_tabs.setCurrentIndex(0)  # Switch to analyzer tab
        self.csv_path_input.setText(csv_path)
        self.video_path_input.setText(video_path)
        
        # Load and analyze the new recording
        self.check_enable_button()

    def on_recording_error(self, error_msg):
        self.is_recording = False
        
        # Update UI
        self.btn_start_recording.setEnabled(True)
        self.btn_stop_recording.setEnabled(False)
        self.recording_progress.setVisible(False)
        self.lbl_recording_status.setText("Recording failed!")
        
        QMessageBox.critical(self, "Recording Error", f"Error during recording:\n{error_msg}")

    def update_recent_recordings(self, video_path, csv_path):
        if not hasattr(self, 'recent_recordings'):
            self.recent_recordings = []
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.recent_recordings.insert(0, {
            'time': timestamp,
            'video': video_path,
            'csv': csv_path
        })
        
        # Keep only last 5 recordings
        if len(self.recent_recordings) > 5:
            self.recent_recordings = self.recent_recordings[:5]
        
        # Update display
        text = ""
        for rec in self.recent_recordings:
            text += f"{rec['time']}\n"
            text += f"  Video: {os.path.basename(rec['video'])}\n"
            text += f"  Data: {os.path.basename(rec['csv'])}\n\n"
        
        self.list_recent.setText(text)

    # --- UI HELPERS (Analyzer Tab) ---
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
            if self.load_existing_results(None, self.csv_path_input.text()):
                self.populate_data()
                self.analyze_button.setText("RE-PROCESS (Data Loaded)")
            else:
                self.analyze_button.setText("START PROCESSING")

    def open_project_folder(self):
        self.reset_state()
        folder = QFileDialog.getExistingDirectory(self, "Select Video Project Folder")
        if not folder: return
        
        csvs = glob.glob(os.path.join(folder, "*.csv"))
        if not csvs: return QMessageBox.warning(self, "Error", "No CSV file found in this folder.")
        csv_path = csvs[0]
        
        self.csv_path_input.setText(csv_path)
        
        if self.load_existing_results(None, csv_path):
            self.populate_data()
            self.analyze_button.setText("RE-PROCESS (Data Loaded)")
            
            v_exts = ['.mp4', '.MP4', '.avi', '.mov', '.mkv']
            vid_path = None
            for ext in v_exts:
                p = os.path.join(folder, "recording" + ext)
                if os.path.exists(p): vid_path = p; break
            
            if vid_path:
                self.video_path_input.setText(vid_path)
                self.analyze_button.setEnabled(True)
            else:
                self.video_path_input.setText("")
                self.analyze_button.setEnabled(False)
                self.analyze_button.setText("DATA LOADED (No Source Video)")
                
            QMessageBox.information(self, "Success", "Project loaded from existing results.")
            
        else:
            v_exts = ['.mp4', '.MP4', '.avi', '.mov', '.mkv']
            vid_path = None
            for ext in v_exts:
                p = os.path.join(folder, "recording" + ext)
                if os.path.exists(p): vid_path = p; break
            
            if not vid_path:
                 QMessageBox.information(self, "Info", "CSV found, but no video file or results found.\nPlease select the source Video file manually.")
                 self.video_path_input.setText("")
            else:
                self.video_path_input.setText(vid_path)
                
            self.check_enable_button()

    def start_analysis(self):
        self.reset_state()
        self.analyze_button.setText("PROCESSING..."); self.analyze_button.setEnabled(False)
        self.file_group.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress_status.setText("Initializing...")
        
        self.worker = VideoAnalyzerWorker(self.csv_path_input.text(), self.video_path_input.text())
        self.worker.progress.connect(self.update_worker_progress)
        self.worker.finished.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def update_worker_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.lbl_progress_status.setText(msg)

    def on_success(self, res, project_path):
        self.analyze_button.setText("RE-PROCESS"); self.analyze_button.setEnabled(True)
        self.file_group.setEnabled(True)
        self.lbl_progress_status.setText("Done!")
        QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
        
        filename = os.path.basename(self.csv_path_input.text())
        new_csv = os.path.join(project_path, filename)
        self.csv_path_input.setText(new_csv)
        
        if not res: QMessageBox.info(self, "Info", "No highlights found based on thresholds.")
        self.results_data = res; self.populate_data()
        QMessageBox.information(self, "Done", f"Analysis complete!\nProject saved to:\n{project_path}")

    def on_error(self, err):
        self.analyze_button.setText("ERROR"); self.analyze_button.setEnabled(True)
        self.file_group.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress_status.setText("Error occurred.")
        QMessageBox.critical(self, "Processing Error", err)

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
        out_dir = os.path.dirname(os.path.abspath(c_path))
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
            seg_dir = os.path.join(out_dir, fname)
            m_path = os.path.join(out_dir, mname)
            
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
                if hasattr(self, 'video_frame') and hasattr(self.video_frame, 'play_btn'):
                    self.video_frame.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                self.position_slider.setValue(0)
                return
                
            if self.media_player.is_playing() or self.media_player.get_state() == vlc.State.Paused:
                position = int(self.media_player.get_position() * 1000)
                self.position_slider.blockSignals(True)
                self.position_slider.setValue(position)
                self.position_slider.blockSignals(False)
                
                current_time = self.media_player.get_time()
                total_time = self.media_player.get_length()
                fmt = lambda x: f"{int(x/1000)//60:02}:{int(x/1000)%60:02}"
                self.time_label.setText(f"{fmt(current_time)} / {fmt(total_time)}")
                
                is_playing = self.media_player.is_playing()
                self.play_button.setIcon(
                    self.style().standardIcon(
                        QStyle.SP_MediaPause if is_playing else QStyle.SP_MediaPlay
                    )
                )
                if hasattr(self, 'video_frame') and hasattr(self.video_frame, 'play_btn'):
                    self.video_frame.play_btn.setIcon(
                        self.style().standardIcon(
                            QStyle.SP_MediaPause if is_playing else QStyle.SP_MediaPlay
                        )
                    )
        except Exception as e:
            print(f"Error in update_ui_timer: {e}")

    def tab_changed(self, idx):
        cat = 'Focus_Highlights' if idx in [0,1] else 'Relaxation_Moments'
        data = self.results_data.get(cat)
        if data and os.path.exists(data['path']): self.load_video(data['path'], False)
    
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
            self.exit_fullscreen()
        else:
            self.file_group.hide()
            self.inner_tabs.hide()
            self.controls_widget.hide()
            self.showFullScreen()
            self.video_frame.set_fullscreen_mode(True)
            self.video_frame.play_btn.clicked.connect(self.play_video)

    def exit_fullscreen(self):
        self.showNormal()
        self.file_group.show()
        self.inner_tabs.show()
        self.controls_widget.show()
        self.video_frame.set_fullscreen_mode(False)
        try:
            self.video_frame.play_btn.clicked.disconnect()
        except:
            pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.exit_fullscreen()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        if self.media_player: self.media_player.stop(); self.media_player.release()
        
        # Stop recording if active
        if self.is_recording and self.recording_worker:
            self.recording_worker.is_recording = False
        
        # Disconnect BrainBit sensors
        if self.brainbit_sensor:
            try:
                self.brainbit_sensor.disconnect()
                del self.brainbit_sensor
            except: pass
            
        if hasattr(self, 'recorder_brainbit_sensor') and self.recorder_brainbit_sensor:
            try:
                self.recorder_brainbit_sensor.disconnect()
                del self.recorder_brainbit_sensor
            except: pass
            
        super().closeEvent(event)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuro Video Analyzer & Recorder")
    parser.add_argument("--video", "-v", type=str, help="Path to the video file")
    parser.add_argument("--csv", "-c", type=str, help="Path to the CSV file")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VideoAnalyzerApp(cli_video=args.video, cli_csv=args.csv)
    window.show()
    sys.exit(app.exec_())
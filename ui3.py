import sys
import os
import pandas as pd
import shutil
import time
import glob
import gc
import argparse
import subprocess
from datetime import datetime
import cv2
import numpy as np
import mss
import mss.tools

# --- PYQT6 IMPORTS ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLineEdit, QLabel, QFileDialog, QTabWidget, 
    QMessageBox, QGroupBox, QSlider, QStyle, QFrame, QSizePolicy, 
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QScrollArea, QSplitter, QSpacerItem, QTextEdit, QProgressBar,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize, QRect
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor, QCursor, QImage, QPixmap, QPainter, QAction

# --- CUSTOM UI IMPORTS (From main_window dependencies) ---
try:
    from ui.video_widget import VideoWidget
    from ui.engagement_graph import EngagementGraph
    UI_MODULES_AVAILABLE = True
except ImportError:
    print("Warning: ui.video_widget or ui.engagement_graph not found. Session tab will be disabled.")
    UI_MODULES_AVAILABLE = False

# --- BACKEND IMPORTS ---
try:
    from backend.src.sensor_event_handler import SensorEventHandler
    from backend.src.sensor_connector import SensorConnector
    from backend.src.emotion_math_manager import EmotionMathManager
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Backend import error: {e}")
    BACKEND_AVAILABLE = False

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
"""

# ==============================================================================
# 1. ANALYSIS LOGIC (UNCHANGED)
# ==============================================================================
CATEGORY_CONFIG = {
    'Focus_Highlights': {'col': 'Focus', 'threshold': 40, 'condition': 'ge'},
    'Relaxation_Moments': {'col': 'Relaxation', 'threshold': 40, 'condition': 'ge'} 
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
        if progress_callback: progress_callback(val, msg)

    update_progress(0, "Validating files...")
    if not os.path.exists(csv_path): raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(video_file): raise FileNotFoundError(f"Video not found: {video_file}")
    
    df = pd.read_csv(csv_path)
    if 'Artifact' in df.columns: df = df[df['Artifact'] == 0].copy()
    start_time_ms = df['Timestamp_ms'].iloc[0]
    
    base_dir = os.path.join(os.path.expanduser('~'), 'video', 'NeuroProjects')
    os.makedirs(base_dir, exist_ok=True)
    project_num = 1
    while os.path.exists(os.path.join(base_dir, f'Project_{project_num}')): project_num += 1
    output_root_folder = os.path.join(base_dir, f'Project_{project_num}')
    os.makedirs(output_root_folder, exist_ok=True)
    shutil.copy2(csv_path, os.path.join(output_root_folder, os.path.basename(csv_path)))
    
    update_progress(10, "Loading video...")
    results_data = {} 
    full_clip = None
    try:
        full_clip = VideoFileClip(video_file)
        all_segments_df = {}
        for cat, config in CATEGORY_CONFIG.items():
            col, thresh = config['col'], config['threshold']
            df_cat = df[df[col] >= thresh].copy() if config['condition'] == 'ge' else df[df[col] <= thresh].copy()
            all_segments_df[cat] = calculate_segments(df_cat, metric_col=col)

        total_clips = sum(len(segs) for segs in all_segments_df.values())
        processed_clips = 0
        
        if total_clips == 0:
            update_progress(100, "No highlights found.")
            return {}, output_root_folder

        for category, segments in all_segments_df.items():
            if segments.empty: continue
            clips_for_montage = []
            segment_metadata = []
            curr_montage_time = 0.0 
            folder_name = "Focus clips" if "Focus" in category else "Relaxation clips"
            montage_filename = "focus montage.mp4" if "Focus" in category else "relaxation montage.mp4"
            segments_folder = os.path.join(output_root_folder, folder_name)
            os.makedirs(segments_folder, exist_ok=True)
            
            for idx, row in segments.iterrows():
                s = (row['start_ms'] - start_time_ms) / 1000
                e = (row['end_ms'] - start_time_ms) / 1000
                if s >= full_clip.duration: continue
                if e > full_clip.duration: e = full_clip.duration
                processed_clips += 1
                prog_val = 10 + int((processed_clips / total_clips) * 70)
                update_progress(prog_val, f"Extracting {category}: Clip {idx+1}/{len(segments)}")

                sub = safe_subclip(full_clip, s, e)
                clips_for_montage.append(sub)
                seg_name = f"segment_{idx+1:03d}_{row['avg_score']:.0f}.mp4"
                seg_path = os.path.join(segments_folder, seg_name)
                sub.write_videofile(seg_path, codec='libx264', audio_codec='aac', remove_temp=True, logger=None, preset='ultrafast') 
                segment_metadata.append({'id': idx + 1, 'orig_start': s, 'orig_end': e, 'montage_start': curr_montage_time, 'duration': e-s, 'score': row['avg_score'], 'file_path': seg_path})
                curr_montage_time += (e-s)
            
            if clips_for_montage:
                update_progress(prog_val + 5, f"Rendering montage for {category}...")
                final = concatenate_videoclips(clips_for_montage)
                out_name = os.path.join(output_root_folder, montage_filename)
                final.write_videofile(out_name, codec='libx264', audio_codec='aac', temp_audiofile='temp-audio.m4a', remove_temp=True, logger=None)
                results_data[category] = {'path': out_name, 'segments': segment_metadata, 'total_duration': final.duration}
                final.close()
                for c in clips_for_montage: c.close()
    except Exception as e: raise Exception(f"Processing Error: {e}")
    finally:
        if full_clip: full_clip.close()
    update_progress(100, "Processing Complete!")
    return results_data, output_root_folder

# ==============================================================================
# 2. RECORDING / THREADS
# ==============================================================================
class ScreenRecorder:
    def __init__(self, output_path, fps=30, monitor=1):
        self.output_path = output_path
        self.fps = fps
        self.monitor = monitor
        self.video_writer = None
        
    def start(self):
        with mss.mss() as sct:
            monitor_idx = self.monitor if self.monitor < len(sct.monitors) else 1
            monitor_info = sct.monitors[monitor_idx]
            width, height = monitor_info['width'], monitor_info['height']
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        
    def capture_frame(self):
        if not self.video_writer: return
        with mss.mss() as sct:
            monitor_idx = self.monitor if self.monitor < len(sct.monitors) else 1
            img = np.array(sct.grab(sct.monitors[monitor_idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self.video_writer.write(img)
    
    def stop(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

class VideoAnalyzerWorker(QThread):
    finished = pyqtSignal(dict, str) 
    progress = pyqtSignal(int, str)  
    error = pyqtSignal(str)
    def __init__(self, csv, video):
        super().__init__()
        self.csv, self.video = csv, video
    def run(self):
        try:
            res, proj_path = analyze_and_concatenate_video(self.csv, self.video, lambda p, m: self.progress.emit(p, m))
            self.finished.emit(res, proj_path)
        except Exception as e: self.error.emit(str(e))

class RecordingWorker(QThread):
    progress = pyqtSignal(int, str, float, float)
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    
    def __init__(self, output_dir, fps=30, monitor=1, data_source=None):
        super().__init__()
        self.output_dir = output_dir
        self.fps = fps
        self.monitor = monitor
        self.data_source = data_source
        self.is_recording = False
        self.log_interval_ms = 2000
        self.next_log_timestamp = 0
        self.buffer_relaxation = []
        self.buffer_focus = []
        self.buffer_artifact = []
        
    def run(self):
        try:
            self.is_recording = True
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.output_dir, f"recording_{timestamp_str}.mp4")
            csv_path = os.path.join(self.output_dir, f"data_{timestamp_str}.csv")
            
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("Timestamp_ms,Relaxation,Focus,Artifact\n")
            
            self.recorder = ScreenRecorder(video_path, self.fps, self.monitor)
            self.recorder.start()
            
            start_time = time.time()
            self.next_log_timestamp = 0
            
            while self.is_recording:
                self.recorder.capture_frame()
                elapsed_sec = time.time() - start_time
                current_ms = int(elapsed_sec * 1000)
                
                focus_val = 0; relax_val = 0; artifact_val = 0; calib_val = 0
                if self.data_source:
                    focus_val = self.data_source.get('focus', 0)
                    relax_val = self.data_source.get('relaxation', 0)
                    artifact_val = self.data_source.get('artifact', 0)
                    calib_val = self.data_source.get('calibration', 0)
                
                self.buffer_relaxation.append(relax_val)
                self.buffer_focus.append(focus_val)
                self.buffer_artifact.append(artifact_val)
                
                if current_ms >= self.next_log_timestamp + self.log_interval_ms:
                    self.write_averaged_log(csv_path)
                    self.next_log_timestamp += self.log_interval_ms
                
                status = f"Recording... {int(elapsed_sec)}s"
                if calib_val < 100: status += f" [Calib: {calib_val:.0f}%]"
                self.progress.emit(100, status, focus_val, relax_val)
                time.sleep(1.0 / self.fps)
                
        except Exception as e: self.error.emit(str(e))
        finally:
            self.is_recording = False
            if hasattr(self, 'recorder') and self.recorder: self.recorder.stop()
            if self.buffer_focus: self.write_averaged_log(csv_path)
            self.finished.emit(video_path, csv_path)

    def write_averaged_log(self, csv_path):
        if not self.buffer_relaxation: return
        
        # 1. Считаем среднее (получится float)
        avg_rel = sum(self.buffer_relaxation) / len(self.buffer_relaxation)
        avg_foc = sum(self.buffer_focus) / len(self.buffer_focus)
        avg_art = sum(self.buffer_artifact) / len(self.buffer_artifact)
        
        # 2. Округляем и приводим к INT (целому числу)
        ts_int = int(self.next_log_timestamp)
        rel_int = int(round(avg_rel))
        foc_int = int(round(avg_foc))
        art_int = int(round(avg_art))

        try:
            with open(csv_path, "a", encoding="utf-8") as f:
                # 3. Записываем строго целые числа
                f.write(f"{ts_int},{rel_int},{foc_int},{art_int}\n")
        except: pass
        
        self.buffer_relaxation.clear()
        self.buffer_focus.clear()
        self.buffer_artifact.clear()
# ==============================================================================
# 3. GUI COMPONENTS (WIDGETS)
# ==============================================================================

# --- Session Tab (Adapted from main_window.py) ---
class SessionTab(QWidget):
    request_analysis = pyqtSignal(str, str) # Signals parent to switch tab (video_path, csv_path)

    def __init__(self, connector):
        super().__init__()
        self.connector = connector
        self.log_path = None
        self.current_video_path = None
        self.session_active = False
        
        self.log_interval_ms = 2000
        self.next_log_timestamp = 0
        self.buffer_relaxation = []
        self.buffer_focus = []
        self.buffer_artifact = []
        self.is_calibrated = False

        self.init_ui()

    def init_ui(self):
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(10)

        # Left Container (Controls + Video)
        self.left_container = QWidget()
        self.left_layout = QVBoxLayout(self.left_container)
        self.left_layout.setSpacing(8)
        self.left_layout.setContentsMargins(0, 0, 0, 0)

        self.load_btn = QPushButton("Load Video for Session")
        self.load_btn.setFixedHeight(32)
        self.load_btn.clicked.connect(self.load_video_dialog)
        self.left_layout.addWidget(self.load_btn)

        self.title_label = QLabel("No video loaded")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.calib_label = QLabel("Calibration: 0%")
        self.calib_label.setStyleSheet("color: #0078d7; font-size: 12px;")
        self.left_layout.addWidget(self.title_label)
        self.left_layout.addWidget(self.calib_label)

        if UI_MODULES_AVAILABLE:
            self.video = VideoWidget()
            self.video.fullscreen_changed.connect(self.toggle_video_fullscreen)
            self.left_layout.addWidget(self.video, stretch=1)
        else:
            self.left_layout.addWidget(QLabel("Video Widget Not Found"), stretch=1)

        self.session_btn = QPushButton("Start EEG Session")
        self.session_btn.setFixedHeight(32)
        self.session_btn.clicked.connect(self.toggle_session)
        self.session_btn.setEnabled(False) 
        self.left_layout.addWidget(self.session_btn)

        self.analyze_btn = QPushButton("ANALYZE RESULTS")
        self.analyze_btn.setFixedHeight(40)
        self.analyze_btn.setStyleSheet("background-color: #3ea6ff; color: white; font-weight: bold; border-radius: 4px;")
        self.analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_btn.clicked.connect(self.trigger_analysis)
        self.analyze_btn.setVisible(False)
        self.left_layout.addWidget(self.analyze_btn)

        # Right Container (Graph)
        self.right_container = QWidget()
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        
        if UI_MODULES_AVAILABLE:
            self.graph = EngagementGraph()
            self.right_layout.addWidget(self.graph, stretch=1)
            self.video.graph = self.graph
            self.video.set_play_allowed(False)
        else:
            self.right_layout.addWidget(QLabel("Graph Widget Not Found"))

        self.main_layout.addWidget(self.left_container, stretch=2)
        self.main_layout.addWidget(self.right_container, stretch=1)

        # Widgets to hide in fullscreen
        self.widgets_to_hide = [self.load_btn, self.title_label, self.calib_label, self.session_btn, self.analyze_btn, self.right_container]

    def update_data(self, value, calib_percent=None, relaxation=None, focus=None, artifact=None):
        """Called by main window when data arrives"""
        if UI_MODULES_AVAILABLE:
            self.graph.add_external_value(value)
            self.video.update_heatmap_from_value(value)

        if calib_percent is not None:
            if calib_percent >= 100:
                self.calib_label.setText(f"Calibration: {calib_percent:.1f}% (Ready)")
                if not self.is_calibrated:
                    self.is_calibrated = True
                    if UI_MODULES_AVAILABLE: self.video.set_play_allowed(True)
            else:
                self.calib_label.setText(f"Calibration: {calib_percent:.1f}%")

        # Logging Logic
        if self.session_active and UI_MODULES_AVAILABLE and self.video.is_playing():
            current_ms = self.video.get_timestamp_ms()

            if current_ms < self.next_log_timestamp:
                self.next_log_timestamp = (current_ms // self.log_interval_ms) * self.log_interval_ms
                self.buffer_relaxation.clear(); self.buffer_focus.clear(); self.buffer_artifact.clear()

            rel_val = relaxation if relaxation is not None else value * 100
            foc_val = focus if focus is not None else value * 100
            art_val = int(artifact) if artifact is not None else 0

            self.buffer_relaxation.append(rel_val)
            self.buffer_focus.append(foc_val)
            self.buffer_artifact.append(art_val)

            if current_ms >= (self.next_log_timestamp + self.log_interval_ms):
                self.write_averaged_log()
                self.next_log_timestamp += self.log_interval_ms

    def write_averaged_log(self):
        if not self.buffer_relaxation or not self.log_path: return
        
        # 1. Считаем среднее
        avg_rel = sum(self.buffer_relaxation) / len(self.buffer_relaxation)
        avg_foc = sum(self.buffer_focus) / len(self.buffer_focus)
        avg_art = sum(self.buffer_artifact) / len(self.buffer_artifact)
        
        # 2. Приводим к INT
        ts_int = int(self.next_log_timestamp)
        rel_int = int(round(avg_rel))
        foc_int = int(round(avg_foc))
        art_int = int(round(avg_art))
        
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{ts_int},{rel_int},{foc_int},{art_int}\n")
        except: pass
        
        self.buffer_relaxation.clear() 
        self.buffer_focus.clear() 
        self.buffer_artifact.clear()
    def toggle_session(self):
        if not self.session_active:
            # START
            try:
                self.connector.start_signal_from_ui()
                self.session_btn.setText("Stop EEG Session")
                self.session_active = True
                self.analyze_btn.setVisible(False)
                
                # Create Log File
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                logs_dir = os.path.join(os.path.expanduser('~'), 'video', 'SessionLogs')
                os.makedirs(logs_dir, exist_ok=True)
                self.log_path = os.path.join(logs_dir, f"session_{ts}.csv")
                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.write("Timestamp_ms,Relaxation,Focus,Artifact\n")
                
                if UI_MODULES_AVAILABLE:
                    vid_ms = self.video.get_timestamp_ms()
                    self.next_log_timestamp = (vid_ms // self.log_interval_ms) * self.log_interval_ms
                
                self.buffer_relaxation = []; self.buffer_focus = []; self.buffer_artifact = []
            except Exception as e: QMessageBox.critical(self, "Error", f"Cannot start session: {e}")
        else:
            # STOP
            try:
                self.connector.stop_signal_from_ui()
                self.session_btn.setText("Start EEG Session")
                self.session_active = False
                if self.log_path: self.analyze_btn.setVisible(True)
            except Exception as e: print(f"Stop error: {e}")

    def load_video_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
        if path:
            self.current_video_path = path
            if UI_MODULES_AVAILABLE: self.video.load(path)
            self.title_label.setText(os.path.basename(path))
            self.session_btn.setEnabled(True)
            self.analyze_btn.setVisible(False)

    def trigger_analysis(self):
        if self.current_video_path and self.log_path:
            self.request_analysis.emit(self.current_video_path, self.log_path)

    def toggle_video_fullscreen(self, enable: bool):
        if not UI_MODULES_AVAILABLE: return
        if enable:
            for w in self.widgets_to_hide: w.setVisible(False)
            self.layout().setContentsMargins(0, 0, 0, 0)
            self.video.enter_fullscreen_mode()
        else:
            for w in self.widgets_to_hide: 
                if w == self.analyze_btn: w.setVisible(bool(self.log_path and not self.session_active))
                else: w.setVisible(True)
            self.layout().setContentsMargins(5, 5, 5, 5)
            self.video.exit_fullscreen_mode()

# --- Helpers for Analyzer Tab ---
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
        self.play_btn = QPushButton(self)
        self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_btn.setIconSize(QSize(64, 64))
        self.play_btn.setFixedSize(80, 80)
        self.play_btn.setStyleSheet("QPushButton { background-color: rgba(0, 0, 0, 0.7); border: 2px solid white; border-radius: 40px; } QPushButton:hover { background-color: rgba(0, 0, 0, 0.9); }")
        self.play_btn.hide()
        self.setMouseTracking(True)
        self._is_fs = False
    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.doubleClicked.emit()
    def mouseMoveEvent(self, e):
        if self._is_fs:
            self.exit_btn.move(self.width() - 170, 20)
            self.exit_btn.setVisible(True)
            at_center = (self.width()/2 - 100 < e.pos().x() < self.width()/2 + 100 and self.height()/2 - 100 < e.pos().y() < self.height()/2 + 100)
            self.play_btn.move(int(self.width()//2 - 40), int(self.height()//2 - 40))
            self.play_btn.setVisible(at_center)
    def set_fullscreen_mode(self, enabled):
        self._is_fs = enabled
        if enabled:
            self.exit_btn.show(); self.play_btn.show(); self.setCursor(Qt.CursorShape.ArrowCursor)
            QTimer.singleShot(3000, lambda: self.setCursor(Qt.CursorShape.BlankCursor) if not self.underMouse() else None)
        else:
            self.exit_btn.hide(); self.play_btn.hide(); self.unsetCursor()

class VideoCardWidget(QFrame):
    playClicked = pyqtSignal(str) 
    def __init__(self, title, video_path=""):
        super().__init__()
        self.setObjectName("VideoCard")
        self.video_path = video_path
        l = QHBoxLayout(self)
        b = QPushButton(); b.setObjectName("CardPlayBtn")
        b.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        b.setFixedSize(40, 40); b.clicked.connect(lambda: self.playClicked.emit(self.video_path))
        l.addWidget(b)
        il = QVBoxLayout(); lbl = QLabel(title); lbl.setStyleSheet("font-weight:bold; color:white;")
        self.sub = QLabel("Duration: " + video_path); il.addWidget(lbl); il.addWidget(self.sub)
        l.addLayout(il); l.addStretch()
    def set_data(self, path, dur, sub):
        self.video_path = path; self.sub.setText(f"{dur} {sub}")

# ==============================================================================
# 4. MAIN APPLICATION WINDOW
# ==============================================================================
class VideoAnalyzerApp(QMainWindow):
    def __init__(self, cli_video=None, cli_csv=None):
        super().__init__()
        self.setWindowTitle("Neuro Video Analyzer & Recorder (Integrated)")
        self.setGeometry(50, 50, 1300, 850) 
        self.setStyleSheet(STYLESHEET)
        
        # VLC init
        self.instance = None
        self.media_player = None
        self.init_vlc()

        # Backend Init
        self.connector = None
        self.math = None
        if BACKEND_AVAILABLE:
            self.math = EmotionMathManager()
            self.handler = SensorEventHandler(math_manager=self.math)
            self.handler.on_engagement_update = self.on_engagement_update
            self.connector = SensorConnector(event_handler=self.handler, math_manager=self.math)
        
        self.current_metrics = {'focus': 0, 'relaxation': 0, 'artifact': 0, 'calibration': 0}
        self.results_data = {} 
        self.current_video_path = None
        self.recording_worker = None
        self.is_recording = False
        
        self.timer = QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self.update_ui_timer)

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

    def on_engagement_update(self, value, calib_percent=None, relaxation=None, focus=None, artifact=None):
        # 1. Update internal state
        if focus is not None: self.current_metrics['focus'] = focus
        if relaxation is not None: self.current_metrics['relaxation'] = relaxation
        if artifact is not None: self.current_metrics['artifact'] = 1 if artifact else 0
        if calib_percent is not None: self.current_metrics['calibration'] = calib_percent

        # 2. Update Recorder Tab (if visible/recording)
        if calib_percent is not None:
            if calib_percent < 100:
                self.lbl_recorder_bb_status.setStyleSheet("color: #FFD700;")
                self.lbl_bb_info.setText(f"Calibrating: {calib_percent:.0f}%")
            else:
                self.lbl_recorder_bb_status.setStyleSheet("color: #00FF00;")
                self.lbl_bb_info.setText(f"Active | F: {focus:.0f} R: {relaxation:.0f}")

        # 3. Update Session Tab
        if hasattr(self, 'session_tab'):
            self.session_tab.update_data(value, calib_percent, relaxation, focus, artifact)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # --- TAB 1: ANALYZER ---
        self.analyzer_widget = QWidget()
        self.build_analyzer_ui(self.analyzer_widget)
        self.main_tabs.addTab(self.analyzer_widget, "🎬 Analyzer")
        
        # --- TAB 2: SESSION (WATCH VIDEO) ---
        self.session_tab = SessionTab(self.connector)
        self.session_tab.request_analysis.connect(self.load_from_session_and_switch)
        self.main_tabs.addTab(self.session_tab, "👁 Watch & Record")

        # --- TAB 3: SCREEN RECORDER ---
        self.recorder_widget = QWidget()
        self.build_recorder_ui(self.recorder_widget)
        self.main_tabs.addTab(self.recorder_widget, "📹 Screen Recorder")
        
        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.main_tabs)

    def load_from_session_and_switch(self, vid_path, csv_path):
        """Called when user clicks 'Analyze Results' in Session tab"""
        self.video_path_input.setText(vid_path)
        self.csv_path_input.setText(csv_path)
        self.check_enable_button()
        self.main_tabs.setCurrentIndex(0) # Switch to Analyzer
        # Optional: Auto-start analysis
        # self.start_analysis()

    # --- ANALYZER UI BUILDER ---
    def build_analyzer_ui(self, parent):
        l = QVBoxLayout(parent); l.setSpacing(15); l.setContentsMargins(20, 10, 20, 20)
        
        # File inputs
        gb = QGroupBox("Analysis Source Files")
        gl = QVBoxLayout(gb)
        btn_proj = QPushButton("📂 OPEN PROJECT FOLDER")
        btn_proj.clicked.connect(self.open_project_folder); gl.addWidget(btn_proj)
        
        r1 = QHBoxLayout(); self.csv_path_input = QLineEdit(); self.csv_path_input.setPlaceholderText("CSV Path...")
        b1 = QPushButton("Browse CSV"); b1.clicked.connect(lambda: self.browse_file('csv'))
        r1.addWidget(self.csv_path_input); r1.addWidget(b1); gl.addLayout(r1)
        
        r2 = QHBoxLayout(); self.video_path_input = QLineEdit(); self.video_path_input.setPlaceholderText("Video Path...")
        b2 = QPushButton("Browse Video"); b2.clicked.connect(lambda: self.browse_file('video'))
        r2.addWidget(self.video_path_input); r2.addWidget(b2); gl.addLayout(r2)
        
        self.analyze_button = QPushButton("START PROCESSING"); self.analyze_button.setObjectName("ActionBtn")
        self.analyze_button.clicked.connect(self.start_analysis); self.analyze_button.setEnabled(False)
        gl.addWidget(self.analyze_button)
        
        self.progress_bar = QProgressBar(); self.progress_bar.setVisible(False)
        self.lbl_progress_status = QLabel("Ready"); gl.addWidget(self.lbl_progress_status); gl.addWidget(self.progress_bar)
        l.addWidget(gb)
        
        # Video Player
        self.video_container = QWidget(); vl = QVBoxLayout(self.video_container); vl.setContentsMargins(0,0,0,0)
        self.video_frame = VideoClickableFrame(); self.video_frame.setMinimumHeight(350)
        self.video_frame.setStyleSheet("background-color: #000;"); self.video_frame.doubleClicked.connect(self.toggle_fullscreen)
        self.video_frame.exitFullscreen.connect(self.exit_fullscreen)
        if not vlc: self.video_frame.setLayout(QVBoxLayout()); self.video_frame.layout().addWidget(QLabel("VLC not found"))
        vl.addWidget(self.video_frame)
        
        # Controls
        self.controls_widget = QWidget(); self.controls_widget.setStyleSheet("background: #202020;")
        cl = QHBoxLayout(self.controls_widget)
        self.play_button = QPushButton(); self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_video); cl.addWidget(self.play_button)
        self.time_label = QLabel("00:00 / 00:00"); cl.addWidget(self.time_label)
        self.position_slider = QSlider(Qt.Orientation.Horizontal); self.position_slider.setRange(0, 1000)
        self.position_slider.sliderMoved.connect(self.set_position); cl.addWidget(self.position_slider)
        fs = QPushButton(); fs.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMaxButton))
        fs.clicked.connect(self.toggle_fullscreen); cl.addWidget(fs)
        vl.addWidget(self.controls_widget); l.addWidget(self.video_container)
        
        # Results Tabs
        self.inner_tabs = QTabWidget(); self.inner_tabs.setMinimumHeight(250)
        self.inner_tabs.currentChanged.connect(self.tab_changed)
        self.tables = {}; self.gallery_layouts = {}
        for cat in ['Focus_Highlights', 'Relaxation_Moments']:
            n = cat.split('_')[0]
            w1, t1 = self.create_list_tab(); self.tables[cat] = t1; self.inner_tabs.addTab(w1, f"{n} (List)")
            w2, l2 = self.create_gallery_tab(); self.gallery_layouts[cat] = l2; self.inner_tabs.addTab(w2, f"{n} (Gallery)")
        l.addWidget(self.inner_tabs); self.check_enable_button()

    # --- RECORDER UI BUILDER ---
    def build_recorder_ui(self, parent):
        l = QVBoxLayout(parent); l.setSpacing(15); l.setContentsMargins(20, 20, 20, 20)
        
        bb = QGroupBox("BrainBit Connection"); bl = QVBoxLayout(bb)
        cl = QHBoxLayout(); self.btn_recorder_connect = QPushButton("Connect")
        self.btn_recorder_connect.clicked.connect(self.toggle_recorder_brainbit)
        if not BACKEND_AVAILABLE: self.btn_recorder_connect.setEnabled(False)
        self.lbl_recorder_bb_status = QLabel("●"); self.lbl_recorder_bb_status.setObjectName("BrainBitStatus")
        cl.addWidget(self.btn_recorder_connect); cl.addWidget(QLabel("Status:")); cl.addWidget(self.lbl_recorder_bb_status); cl.addStretch()
        bl.addLayout(cl)
        il = QHBoxLayout(); self.lbl_bb_info = QLabel("Not connected"); il.addWidget(self.lbl_bb_info); il.addStretch()
        bl.addLayout(il); l.addWidget(bb)
        
        sett = QGroupBox("Recording Settings"); sl = QVBoxLayout(sett)
        self.spin_fps = QSpinBox(); self.spin_fps.setValue(30)
        self.combo_monitor = QComboBox(); self.combo_monitor.addItems([f"Monitor {i+1}" for i in range(4)])
        self.recorder_output_dir = QLineEdit(); self.recorder_output_dir.setText(os.path.join(os.path.expanduser('~'), 'video', 'Recordings'))
        btn_br = QPushButton("Browse"); btn_br.clicked.connect(self.browse_recording_output)
        sl.addWidget(QLabel("FPS:")); sl.addWidget(self.spin_fps)
        sl.addWidget(QLabel("Monitor:")); sl.addWidget(self.combo_monitor)
        sl.addWidget(QLabel("Output:")); 
        ol = QHBoxLayout(); ol.addWidget(self.recorder_output_dir); ol.addWidget(btn_br); sl.addLayout(ol)
        l.addWidget(sett)
        
        ctrl = QGroupBox("Controls"); ctl = QVBoxLayout(ctrl)
        self.lbl_recording_status = QLabel("Ready"); ctl.addWidget(self.lbl_recording_status)
        self.recording_progress = QProgressBar(); self.recording_progress.setVisible(False); ctl.addWidget(self.recording_progress)
        ml = QHBoxLayout(); self.lbl_focus = QLabel("Focus: --"); self.lbl_relaxation = QLabel("Relaxation: --")
        ml.addWidget(self.lbl_focus); ml.addWidget(self.lbl_relaxation); ml.addStretch(); ctl.addLayout(ml)
        bl = QHBoxLayout()
        self.btn_start_recording = QPushButton("⏺ START"); self.btn_start_recording.setObjectName("RecordBtn")
        self.btn_start_recording.clicked.connect(self.start_recording); self.btn_start_recording.setEnabled(False)
        self.btn_stop_recording = QPushButton("⏹ STOP"); self.btn_stop_recording.setObjectName("StopBtn")
        self.btn_stop_recording.clicked.connect(self.stop_recording); self.btn_stop_recording.setEnabled(False)
        bl.addWidget(self.btn_start_recording); bl.addWidget(self.btn_stop_recording); ctl.addLayout(bl)
        l.addWidget(ctrl)
        
        rec = QGroupBox("Recent"); rl = QVBoxLayout(rec)
        self.list_recent = QTextEdit(); self.list_recent.setReadOnly(True); self.list_recent.setMaximumHeight(100)
        rl.addWidget(self.list_recent); l.addWidget(rec); l.addStretch()

    # --- BRAINBIT SHARED LOGIC ---
    def toggle_recorder_brainbit(self):
        if not BACKEND_AVAILABLE: return
        if self.btn_recorder_connect.text() == "Connect":
            try:
                self.connector.start_signal_from_ui()
                self.btn_recorder_connect.setText("Disconnect")
                self.lbl_recorder_bb_status.setStyleSheet("color: #FFD700;")
                self.lbl_bb_info.setText("Searching...")
                self.btn_start_recording.setEnabled(True)
            except Exception as e: QMessageBox.critical(self, "Error", str(e))
        else:
            try:
                self.connector.stop_signal_from_ui()
                self.btn_recorder_connect.setText("Connect")
                self.lbl_recorder_bb_status.setStyleSheet("color: #444;")
                self.lbl_bb_info.setText("Not connected")
                self.btn_start_recording.setEnabled(False)
                self.current_metrics = {'focus': 0, 'relaxation': 0, 'artifact': 0, 'calibration': 0}
            except: pass

    # --- SCREEN RECORDER LOGIC ---
    def start_recording(self):
        if self.is_recording: return
        self.is_recording = True
        self.recording_worker = RecordingWorker(
            self.recorder_output_dir.text(), 
            self.spin_fps.value(), 
            self.combo_monitor.currentIndex() + 1, 
            self.current_metrics
        )
        self.recording_worker.progress.connect(self.update_recording_progress)
        self.recording_worker.finished.connect(self.on_recording_finished)
        self.recording_worker.start()
        self.btn_start_recording.setEnabled(False); self.btn_stop_recording.setEnabled(True)
        self.recording_progress.setVisible(True)

    def stop_recording(self):
        if self.recording_worker: self.recording_worker.is_recording = False
        self.btn_stop_recording.setEnabled(False); self.lbl_recording_status.setText("Stopping...")

    def update_recording_progress(self, p, s, f, r):
        self.recording_progress.setValue(p); self.lbl_recording_status.setText(s)
        self.lbl_focus.setText(f"Focus: {f:.1f}"); self.lbl_relaxation.setText(f"Relaxation: {r:.1f}")

    def on_recording_finished(self, vid, csv):
        self.is_recording = False
        self.btn_start_recording.setEnabled(True); self.recording_progress.setVisible(False)
        self.lbl_recording_status.setText("Done")
        self.list_recent.append(f"{os.path.basename(vid)}"); self.list_recent.append(f"{os.path.basename(csv)}\n")
        self.csv_path_input.setText(csv); self.video_path_input.setText(vid)
        self.main_tabs.setCurrentIndex(0); self.check_enable_button()

    def browse_recording_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output Folder")
        if d: self.recorder_output_dir.setText(d)

    # --- ANALYZER LOGIC ---
    def create_list_tab(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(10,10,10,10)
        t = QTableWidget(); t.setColumnCount(4); t.setHorizontalHeaderLabels(["#", "Time", "Dur", "Score"])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows); t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.cellClicked.connect(self.on_segment_clicked); l.addWidget(t)
        return w, t
    def create_gallery_tab(self):
        s = QScrollArea(); s.setWidgetResizable(True); c = QWidget(); c.setStyleSheet("background: transparent;")
        l = QVBoxLayout(c); l.setSpacing(10); l.setAlignment(Qt.AlignmentFlag.AlignTop); s.setWidget(c)
        return s, l
    def browse_file(self, mode):
        if mode == 'video': self.reset_state()
        f = "CSV (*.csv)" if mode == 'csv' else "Video (*.mp4 *.avi *.mov)"
        t = self.csv_path_input if mode == 'csv' else self.video_path_input
        p, _ = QFileDialog.getOpenFileName(self, "Select", "", f)
        if p: t.setText(p); self.check_enable_button()
    def check_enable_button(self,loaded_proj = False):
        csv_ok = os.path.exists(self.csv_path_input.text())
        video_ok = os.path.exists(self.video_path_input.text())

        # Пытаемся загрузить проект по одному CSV
        project_loaded = False
        if csv_ok:
            project_loaded = self.load_existing_results(None, self.csv_path_input.text())

        # Доступно два режима:
        # 1) CSV + VIDEO → можно заново анализировать
        # 2) CSV + найден проект → можно открыть без видео
        if (csv_ok and video_ok) or project_loaded:
            self.analyze_button.setEnabled(csv_ok and video_ok)
            self.analyze_button.setText("RE-PROCESS" if video_ok else "ANALYSIS DISABLED (no video)")
            if project_loaded:
                self.populate_data()
        else:
            self.analyze_button.setEnabled(False)
            self.analyze_button.setText("START PROCESSING")
    def start_analysis(self):
        self.reset_state(); self.analyze_button.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.worker = VideoAnalyzerWorker(self.csv_path_input.text(), self.video_path_input.text())
        self.worker.progress.connect(lambda v, m: (self.progress_bar.setValue(v), self.lbl_progress_status.setText(m)))
        self.worker.finished.connect(self.on_success); self.worker.error.connect(self.on_error); self.worker.start()
    def on_success(self, res, path):
        self.analyze_button.setEnabled(True); self.progress_bar.setVisible(False)
        self.csv_path_input.setText(os.path.join(path, os.path.basename(self.csv_path_input.text())))
        self.results_data = res; self.populate_data(); QMessageBox.information(self, "Done", f"Saved to:\n{path}")
    def on_error(self, e):
        self.analyze_button.setEnabled(True); self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", e)
    def tab_changed(self, idx):
        """Обработка переключения вкладок результатов анализа"""
        # Индексы 0 и 1 - это вкладки Фокуса (List и Gallery)
        # Индексы 2 и 3 - это вкладки Релаксации
        cat = 'Focus_Highlights' if idx in [0, 1] else 'Relaxation_Moments'
        
        # Загружаем соответствующее видео-монтаж, если оно есть
        if self.results_data:
            data = self.results_data.get(cat)
            if data and 'path' in data and os.path.exists(data['path']):
                # False = не автозапуск, просто загрузить файл
                self.load_video(data['path'], auto=False)

    def load_existing_results(self, v_path, c_path):
        """Полная логика загрузки существующего проекта"""
        out_dir = os.path.dirname(os.path.abspath(c_path))

        # Если исходного видео нет — это НЕ ошибка
        video_available = os.path.exists(v_path) if v_path else False
        try:
            # Пытаемся восстановить метаданные сегментов из исходного CSV
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
                # Сортируем файлы, чтобы порядок соответствовал хронологии
                files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.mp4')])
                curr_mnt = 0.0
                meta_df = all_segments_df.get(cat)
                
                for i, f in enumerate(files):
                    try:
                        # Парсим имя файла: segment_001_85.mp4 -> id=1, score=85
                        p = f[:-4].split('_')
                        sid = int(p[1]); sc = float(p[2])
                        fpath = os.path.join(seg_dir, f)
                        
                        # Получаем длительность клипа
                        with VideoFileClip(fpath) as c: dur = c.duration
                        
                        # Пытаемся найти исходное время в оригинальном видео
                        orig_s, orig_e = 0, 0
                        if meta_df is not None and not meta_df.empty:
                            row_idx = sid - 1
                            if 0 <= row_idx < len(meta_df):
                                row = meta_df.iloc[row_idx]
                                orig_s = (row['start_ms'] - start_ms)/1000
                                orig_e = (row['end_ms'] - start_ms)/1000
                                
                        segs.append({
                            'id': sid, 'score': sc, 'file_path': fpath, 
                            'duration': dur, 'montage_start': curr_mnt, 
                            'orig_start': orig_s, 'orig_end': orig_e
                        }) 
                        curr_mnt += dur
                    except: pass
                
                if segs:
                    res[cat] = {'path': m_path, 'segments': segs, 'total_duration': curr_mnt}
                    loaded = True
        
        if loaded: 
            self.results_data = res
        return loaded

    def populate_data(self):
        for cat, data in self.results_data.items():
            if cat in self.tables:
                t = self.tables[cat]; t.setRowCount(len(data.get('segments', [])))
                for i, s in enumerate(data['segments']):
                    fmt = lambda x: f"{int(x)//60:02}:{int(x)%60:02}"
                    t.setItem(i,0, QTableWidgetItem(str(s['id'])))
                    t.setItem(i,1, QTableWidgetItem(f"{fmt(s['orig_start'])}-{fmt(s['orig_end'])}"))
                    t.setItem(i,2, QTableWidgetItem(f"{s['duration']:.1f}s"))
                    t.setItem(i,3, QTableWidgetItem(f"{s['score']:.1f}"))
                    t.item(i,0).setData(Qt.ItemDataRole.UserRole, s['montage_start'])
            if cat in self.gallery_layouts:
                l = self.gallery_layouts[cat]
                while l.count(): l.takeAt(0).widget().deleteLater()
                for s in data['segments']:
                    card = VideoCardWidget(f"Seg #{s['id']} ({s['score']:.0f})", s['file_path'])
                    card.set_data(s['file_path'], f"{s['duration']:.1f}s", "")
                    card.playClicked.connect(self.force_play_video)
                    l.addWidget(card)
        
        # Обновляем текущую вкладку
        self.tab_changed(self.inner_tabs.currentIndex())
    def reset_state(self):
        if self.media_player: self.media_player.stop()
        self.results_data = {}; self.current_video_path = None
        for t in self.tables.values(): t.setRowCount(0)
        for l in self.gallery_layouts.values(): 
            while l.count(): l.takeAt(0).widget().deleteLater()
    
    # --- VIDEO PLAYER HELPERS ---
    def load_video(self, path, auto=True, seek=0):
        if not self.media_player: return
        self.current_video_path = path
        self.media_player.set_media(self.instance.media_new(path))
        if sys.platform == 'win32': self.media_player.set_hwnd(int(self.video_frame.winId()))
        else: self.media_player.set_xwindow(int(self.video_frame.winId()))
        self.media_player.play()
        QTimer.singleShot(100, lambda: (self.media_player.set_time(seek) if seek else None, self.media_player.set_pause(0) if auto else self.media_player.set_pause(1), self.timer.start()))
    def play_video(self):
        if not self.media_player: return
        if self.media_player.is_playing(): self.media_player.pause()
        else: self.media_player.play(); self.timer.start()
    def set_position(self, p): 
        if self.media_player: self.media_player.set_position(p/1000.0)
    def force_play_video(self, p): 
        if os.path.exists(p): self.load_video(p, True)
    def update_ui_timer(self):
        if not self.media_player or not self.media_player.is_playing(): return
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(int(self.media_player.get_position()*1000))
        self.position_slider.blockSignals(False)
        self.time_label.setText(f"{int(self.media_player.get_time()/1000)//60:02}:{int(self.media_player.get_time()/1000)%60:02}")
    def on_segment_clicked(self, r, c):
        cat = next((k for k,v in self.tables.items() if v == self.sender()), None)
        if cat: self.load_video(self.results_data[cat]['path'], True, int(self.sender().item(r,0).data(Qt.ItemDataRole.UserRole)*1000))
    def toggle_fullscreen(self):
        if self.isFullScreen(): self.exit_fullscreen()
        else: 
            self.file_group.hide(); self.inner_tabs.hide(); self.controls_widget.hide(); self.main_tabs.tabBar().hide()
            self.showFullScreen(); self.video_frame.set_fullscreen_mode(True)
            self.video_frame.play_btn.clicked.connect(self.play_video)
    def exit_fullscreen(self):
        self.showNormal(); self.file_group.show(); self.inner_tabs.show(); self.controls_widget.show(); self.main_tabs.tabBar().show()
        self.video_frame.set_fullscreen_mode(False)
        try: self.video_frame.play_btn.clicked.disconnect()
        except: pass
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape and self.isFullScreen(): self.exit_fullscreen()
        else: super().keyPressEvent(e)
    def closeEvent(self, e):
        if self.connector: self.connector.stop_signal_from_ui()
        super().closeEvent(e)
    def open_project_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Project Folder")
        if d:
            csvs = glob.glob(os.path.join(d, "*.csv"))
            if csvs: self.csv_path_input.setText(csvs[0]); self.check_enable_button(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VideoAnalyzerApp()
    window.show()
    sys.exit(app.exec())
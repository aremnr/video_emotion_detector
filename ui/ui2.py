import sys
import os
import pandas as pd
import shutil
import glob
import gc
import argparse

# --- PYQT6 IMPORTS ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLineEdit, QLabel, QFileDialog, QTabWidget, 
    QMessageBox, QGroupBox, QSlider, QStyle, QFrame, QSizePolicy, 
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QScrollArea, QSplitter, QSpacerItem, QTextEdit, QProgressBar
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize, QRect
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor, QCursor, QImage, QPixmap, QPainter, QAction

# --- MOVIEPY IMPORTS ---
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips

# --- OPTIONAL LIBS ---
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
"""

# ==============================================================================
# 1. ANALYSIS LOGIC
# ==============================================================================

CATEGORY_CONFIG = {
    'Focus_Highlights': {'col': 'Focus', 'threshold': 35, 'condition': 'ge'},
    'Relaxation_Moments': {'col': 'Relaxation', 'threshold': 35, 'condition': 'ge'} 
}

MIN_SEGMENT_DURATION_SEC = 1.0 
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
    csv_path = os.path.abspath(csv_path)
    video_file = os.path.abspath(video_file)

    if not os.path.exists(csv_path): raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(video_file): raise FileNotFoundError(f"Video not found: {video_file}")
    
    update_progress(5, "Reading CSV data...")
    df = pd.read_csv(csv_path)
    req = ['Timestamp_ms', 'Focus', 'Relaxation']
    if not all(c in df.columns for c in req): raise ValueError(f"Missing cols: {req}")
    if 'Artifact' in df.columns:
        df = df[df['Artifact'] == 0].copy()

    start_time_ms = df['Timestamp_ms'].iloc[0]
    
    base_dir = os.path.dirname(csv_path)
    csv_name_no_ext = os.path.splitext(os.path.basename(csv_path))[0]
    output_root_folder = os.path.join(base_dir, f"{csv_name_no_ext}_Results")
    os.makedirs(output_root_folder, exist_ok=True)
    
    dest_csv_path = os.path.join(output_root_folder, os.path.basename(csv_path))
    if os.path.abspath(csv_path) != os.path.abspath(dest_csv_path):
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
                                    temp_audiofile='temp-audio.m4a', remove_temp=True, logger=None, preset='ultrafast')
                
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

# ==============================================================================
# 2. GUI WIDGETS
# ==============================================================================

class FullscreenControls(QFrame):
    """Плавающая панель управления для полноэкранного режима"""
    playClicked = pyqtSignal()
    seekMoved = pyqtSignal(int)
    volumeChanged = pyqtSignal(int)
    exitClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame { background-color: rgba(20, 20, 20, 0.9); border-radius: 10px; border: 1px solid #555; }
            QLabel { color: white; background: transparent; border: none; font-weight: bold; }
            QSlider::groove:horizontal { height: 6px; background: #444; border-radius: 3px; }
            QSlider::handle:horizontal { background: #3ea6ff; width: 14px; margin: -5px 0; border-radius: 7px; }
            QPushButton { background: transparent; border: none; border-radius: 5px; }
            QPushButton:hover { background-color: rgba(255, 255, 255, 0.2); }
        """)
        self.setFixedHeight(70)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 5, 15, 5)
        layout.setSpacing(15)

        # Play Button
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_play.setIconSize(QSize(24, 24))
        self.btn_play.setFixedSize(40, 40)
        self.btn_play.clicked.connect(self.playClicked.emit)
        layout.addWidget(self.btn_play)

        # Time Label
        self.lbl_time = QLabel("00:00 / 00:00")
        layout.addWidget(self.lbl_time)

        # Seek Slider
        self.slider_seek = QSlider(Qt.Orientation.Horizontal)
        self.slider_seek.setRange(0, 1000)
        self.slider_seek.setStyleSheet("background: transparent;")
        self.slider_seek.sliderPressed.connect(self.slider_pressed)
        self.slider_seek.sliderReleased.connect(self.slider_released)
        self.slider_seek.sliderMoved.connect(self.seekMoved.emit)
        layout.addWidget(self.slider_seek)

        # Volume Icon & Slider
        vol_icon = QLabel(); vol_icon.setPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume).pixmap(20, 20))
        vol_icon.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(vol_icon)
        
        self.slider_vol = QSlider(Qt.Orientation.Horizontal)
        self.slider_vol.setFixedWidth(100)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(70)
        self.slider_vol.setStyleSheet("background: transparent;")
        self.slider_vol.valueChanged.connect(self.volumeChanged.emit)
        layout.addWidget(self.slider_vol)

        # Exit Button
        self.btn_exit = QPushButton("✕")
        self.btn_exit.setFixedSize(40, 40)
        self.btn_exit.setStyleSheet("color: white; font-weight: bold; font-size: 16px; background-color: rgba(200,0,0,0.5); border-radius: 20px;")
        self.btn_exit.clicked.connect(self.exitClicked.emit)
        layout.addWidget(self.btn_exit)
        
        self._is_seeking = False

    def slider_pressed(self): self._is_seeking = True
    def slider_released(self): self._is_seeking = False

    def update_state(self, is_playing, position, duration_str):
        icon = QStyle.StandardPixmap.SP_MediaPause if is_playing else QStyle.StandardPixmap.SP_MediaPlay
        self.btn_play.setIcon(self.style().standardIcon(icon))
        
        if not self._is_seeking:
            self.slider_seek.blockSignals(True)
            self.slider_seek.setValue(position)
            self.slider_seek.blockSignals(False)
        
        self.lbl_time.setText(duration_str)

class VideoClickableFrame(QFrame):
    doubleClicked = pyqtSignal()
    exitFullscreen = pyqtSignal()
    
    # Signals from overlay to main app
    overlayPlay = pyqtSignal()
    overlaySeek = pyqtSignal(int)
    overlayVol = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._is_fs = False
        
        # LAYOUT STRUCTURE to ensure Overlay is ON TOP of video
        # We need a layout to hold the video surface
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0)
        
        # 1. Video Surface (Where VLC draws)
        self.video_surface = QFrame(self)
        self.video_surface.setStyleSheet("background-color: black;")
        self.video_surface.setMouseTracking(True) # Pass mouse events up
        self.layout.addWidget(self.video_surface)
        
        # 2. Overlay (The Controls) - Manually positioned, not in layout
        self.overlay = FullscreenControls(self)
        self.overlay.hide()
        
        # Connect overlay signals
        self.overlay.exitClicked.connect(self.exitFullscreen.emit)
        self.overlay.playClicked.connect(self.overlayPlay.emit)
        self.overlay.seekMoved.connect(self.overlaySeek.emit)
        self.overlay.volumeChanged.connect(self.overlayVol.emit)

        # Timer to hide controls
        self.hide_timer = QTimer(self)
        self.hide_timer.setInterval(3000)
        self.hide_timer.timeout.connect(self.hide_controls)
        
        # Install event filter to catch mouse events on the video surface too
        self.video_surface.installEventFilter(self)

    def eventFilter(self, source, event):
        # Если мышь двигается над поверхностью видео, показываем меню
        if source == self.video_surface and event.type() == Qt.EventType.MouseMove:
            self.show_controls()
        if source == self.video_surface and event.type() == Qt.EventType.MouseButtonDblClick:
            self.doubleClicked.emit()
        return super().eventFilter(source, event)

    def mouseMoveEvent(self, event):
        self.show_controls()
        super().mouseMoveEvent(event)

    def show_controls(self):
        if self._is_fs:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.overlay.show()
            self.overlay.raise_() # CRITICAL: Bring to front
            self.hide_timer.start()

    def hide_controls(self):
        if self._is_fs:
            # Check if mouse is currently over the overlay (don't hide if interacting)
            if not self.overlay.underMouse():
                self.overlay.hide()
                self.setCursor(Qt.CursorShape.BlankCursor)

    def resizeEvent(self, event):
        # Manually position overlay at the bottom center
        if self.overlay:
            w = self.width()
            h = self.height()
            ov_w = min(800, w - 40)
            ov_h = 70
            self.overlay.setGeometry((w - ov_w) // 2, h - ov_h - 20, ov_w, ov_h)
            self.overlay.raise_()
        super().resizeEvent(event)

    def set_fullscreen_mode(self, enabled):
        self._is_fs = enabled
        if enabled:
            self.show_controls()
        else:
            self.overlay.hide()
            self.hide_timer.stop()
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def update_overlay(self, is_playing, pos, dur_text):
        if self._is_fs and self.overlay.isVisible():
            self.overlay.update_state(is_playing, pos, dur_text)
            self.overlay.raise_() # Keep raising it just in case

    def get_video_handle(self):
        return int(self.video_surface.winId())

class VideoCardWidget(QFrame):
    playClicked = pyqtSignal(str) 
    def __init__(self, title, video_path="", is_small=False):
        super().__init__()
        self.setObjectName("VideoCard")
        self.video_path = video_path
        layout = QHBoxLayout(self)
        self.btn_play = QPushButton()
        self.btn_play.setObjectName("CardPlayBtn")
        self.btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
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
        self.setWindowTitle("Pro Video Analyzer")
        self.setGeometry(50, 50, 1000, 850)
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setStyleSheet(STYLESHEET)
        
        self.instance = None
        self.media_player = None
        self.results_data = {} 
        self.current_video_path = None
        
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_ui_timer)

        self.init_vlc()
        self.init_ui()

        if cli_video and cli_csv:
            self.video_path_input.setText(os.path.abspath(cli_video))
            self.csv_path_input.setText(os.path.abspath(cli_csv))
            self.analyze_button.setEnabled(True)
            QTimer.singleShot(500, self.start_analysis)
        
    def init_vlc(self):
        if vlc:
            try:
                self.instance = vlc.Instance("--no-xlib", "--quiet", "--no-video-title-show")
                self.media_player = vlc.MediaPlayer(self.instance)
            except: self.media_player = None

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.build_analyzer_ui(self.central_widget)
        
    def build_analyzer_ui(self, parent_widget):
        main_layout = QVBoxLayout(parent_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 1. Project Configuration
        self.file_group = QGroupBox("Analysis Source Files")
        flayout = QVBoxLayout(self.file_group)
        
        self.btn_load_project = QPushButton("📂 OPEN PROJECT FOLDER")
        self.btn_load_project.setObjectName("LoadFolderBtn")
        self.btn_load_project.setCursor(Qt.CursorShape.PointingHandCursor)
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
        self.video_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_frame.setMinimumHeight(350)
        
        # --- Connections ---
        self.video_frame.doubleClicked.connect(self.toggle_fullscreen)
        self.video_frame.exitFullscreen.connect(self.exit_fullscreen)
        self.video_frame.overlayPlay.connect(self.play_video)
        self.video_frame.overlaySeek.connect(self.set_position)
        self.video_frame.overlayVol.connect(self.set_volume)

        if not vlc: 
            self.video_frame.setLayout(QVBoxLayout())
            lbl = QLabel("VLC Player not found.\nInstall python-vlc and VLC Media Player to view video.")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_frame.layout().addWidget(lbl)
        v_layout.addWidget(self.video_frame)
        
        # 3. Player Controls (Standard)
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background-color: #202020; border-bottom-left-radius: 6px; border-bottom-right-radius: 6px;")
        ctrl = QHBoxLayout(self.controls_widget)
        self.play_button = QPushButton()
        self.play_button.setFixedSize(40, 40)
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_video)
        ctrl.addWidget(self.play_button)
        self.time_label = QLabel("00:00 / 00:00")
        ctrl.addWidget(self.time_label)
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.sliderMoved.connect(self.set_position)
        ctrl.addWidget(self.position_slider)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal); self.volume_slider.setFixedWidth(80)
        self.volume_slider.setRange(0, 100); self.volume_slider.setValue(70)
        self.volume_slider.valueChanged.connect(self.set_volume)
        ctrl.addWidget(self.volume_slider)
        self.fs_button = QPushButton(); self.fs_button.setObjectName("FsBtn")
        self.fs_button.setFixedSize(40, 40); self.fs_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMaxButton))
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

    def create_list_tab(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(10,10,10,10)
        t = QTableWidget(); t.setColumnCount(4)
        t.setHorizontalHeaderLabels(["#", "Time (Source)", "Duration", "Score"])
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows); t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.cellClicked.connect(self.on_segment_clicked)
        l.addWidget(t)
        return w, t

    def create_gallery_tab(self):
        s = QScrollArea(); s.setWidgetResizable(True)
        c = QWidget(); c.setStyleSheet("background: transparent;")
        l = QVBoxLayout(c); l.setSpacing(10); l.setAlignment(Qt.AlignmentFlag.AlignTop)
        s.setWidget(c)
        return s, l

    def reset_state(self):
        if self.timer.isActive(): self.timer.stop()
        if self.media_player: 
            self.media_player.stop(); self.media_player.set_media(None)
        gc.collect(); QApplication.processEvents()
        
        self.current_video_path = None
        self.play_button.setEnabled(False)
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
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
        c_path = self.csv_path_input.text()
        v_path = self.video_path_input.text()
        if not os.path.exists(c_path) or not os.path.exists(v_path):
            QMessageBox.critical(self, "Error", "Input files not found!")
            self.analyze_button.setEnabled(True)
            return

        self.analyze_button.setText("PROCESSING..."); self.analyze_button.setEnabled(False)
        self.file_group.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress_status.setText("Initializing...")
        
        self.worker = VideoAnalyzerWorker(c_path, v_path)
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
        
        if not res: QMessageBox.information(self, "Info", "No highlights found based on thresholds.")
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
                    t.item(i,0).setData(Qt.ItemDataRole.UserRole, s['montage_start'])
            if cat in self.gallery_layouts:
                l = self.gallery_layouts[cat]
                for s in segs:
                    card = VideoCardWidget(f"Seg #{s['id']} ({s['score']:.0f})", s['file_path'])
                    card.set_data(s['file_path'], f"{s['duration']:.1f}s", "")
                    card.playClicked.connect(self.force_play_video)
                    l.addWidget(card)
        self.inner_tabs.setCurrentIndex(0); self.tab_changed(0)

    def load_existing_results(self, v_path, c_path):
        if not c_path or not os.path.exists(c_path): return False
        
        csv_dir = os.path.dirname(os.path.abspath(c_path))
        csv_name_no_ext = os.path.splitext(os.path.basename(c_path))[0]
        out_dir = os.path.join(csv_dir, f"{csv_name_no_ext}_Results")
        
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
                        sid = int(p[1])
                        sc = float(p[2]) if len(p) > 2 else 0.0
                        fpath = os.path.join(seg_dir, f)
                        
                        dur = 2.0
                        orig_s, orig_e = 0, 0
                        
                        if meta_df is not None and not meta_df.empty:
                            row_idx = sid - 1
                            if 0 <= row_idx < len(meta_df):
                                row = meta_df.iloc[row_idx]
                                orig_s = (row['start_ms'] - start_ms)/1000
                                orig_e = (row['end_ms'] - start_ms)/1000
                                dur = row['duration_sec']
                        
                        segs.append({
                            'id': sid, 'score': sc, 'file_path': fpath, 
                            'duration': dur, 'montage_start': curr_mnt, 
                            'orig_start': orig_s, 'orig_end': orig_e
                        }) 
                        curr_mnt += dur
                    except: continue
                
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
        
        # --- VLC WINDOW HANDLE FIX ---
        # Instead of giving the whole frame, we give ONLY the video_surface ID
        win_id = self.video_frame.get_video_handle()
        
        if sys.platform == 'win32': self.media_player.set_hwnd(win_id)
        elif sys.platform == 'darwin': 
            try: self.media_player.set_nsobject(win_id)
            except: pass
        else: self.media_player.set_xwindow(win_id)
        # -----------------------------
        
        self.media_player.play()
        QTimer.singleShot(100, lambda: self._finalize_load(autoplay, seek_time_ms))

    def _finalize_load(self, auto, seek):
        if not self.media_player: return
        if seek > 0: self.media_player.set_time(seek)
        if auto: 
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
            self.timer.start()
        else: QTimer.singleShot(150, lambda: self._pause_start())
        self.play_button.setEnabled(True)

    def _pause_start(self):
        if self.media_player: 
            self.media_player.set_pause(1)
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.timer.start()

    def play_video(self):
        if not self.media_player: return
        if self.media_player.is_playing():
            self.media_player.pause()
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            if self.current_video_path:
                self.media_player.play()
                self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
                self.timer.start()

    def set_position(self, pos): 
        if self.media_player: self.media_player.set_position(pos / 1000.0)
    def set_volume(self, vol): 
        if self.media_player: self.media_player.audio_set_volume(vol)
        
    def update_ui_timer(self):
        if not self.media_player: return
        try:
            if self.media_player.get_state() == vlc.State.Ended:
                self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
                self.position_slider.setValue(0)
                self.video_frame.update_overlay(False, 0, "00:00 / 00:00")
                return

            if self.media_player.is_playing() or self.media_player.get_state() == vlc.State.Paused:
                pos_float = self.media_player.get_position()
                pos_int = int(pos_float * 1000)
                cur_ms = self.media_player.get_time()
                tot_ms = self.media_player.get_length()
                
                fmt = lambda x: f"{int(x/1000)//60:02}:{int(x/1000)%60:02}"
                time_str = f"{fmt(cur_ms)} / {fmt(tot_ms)}"

                self.position_slider.blockSignals(True)
                self.position_slider.setValue(pos_int)
                self.position_slider.blockSignals(False)
                self.time_label.setText(time_str)
                
                is_playing = (self.media_player.get_state() == vlc.State.Playing)
                self.video_frame.update_overlay(is_playing, pos_int, time_str)
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
            seek = sender.item(r,0).data(Qt.ItemDataRole.UserRole)
            if path: self.load_video(path, True, int(seek*1000))

    def force_play_video(self, path): 
        if path and os.path.exists(path): self.load_video(path, True)
    
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.exit_fullscreen()
        else:
            self.file_group.hide(); self.inner_tabs.hide(); self.controls_widget.hide()
            self.showFullScreen()
            self.video_frame.set_fullscreen_mode(True)

    def exit_fullscreen(self):
        self.showNormal()
        self.file_group.show(); self.inner_tabs.show(); self.controls_widget.show()
        self.video_frame.set_fullscreen_mode(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.exit_fullscreen()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        if self.media_player: self.media_player.stop(); self.media_player.release()
        super().closeEvent(event)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuro Video Analyzer")
    parser.add_argument("--video", "-v", type=str, help="Path to the video file")
    parser.add_argument("--csv", "-c", type=str, help="Path to the CSV file")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VideoAnalyzerApp(cli_video=args.video, cli_csv=args.csv)
    window.show()
    sys.exit(app.exec())
def run_with_params(video, csv):
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    w = VideoAnalyzerApp(cli_video=video, cli_csv=csv)
    w.show()
    app.exec()
import sys
import os
import pandas as pd
import shutil  # Добавлено для копирования CSV
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLineEdit, QLabel, QFileDialog, QTabWidget, 
    QMessageBox, QGroupBox, QSlider, QStyle, QFrame, QSizePolicy, 
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QScrollArea
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QSize
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QCursor

# ==============================================================================
# --- ИМПОРТЫ БИБЛИОТЕК ОБРАБОТКИ ---
# ==============================================================================
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except ImportError:
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
    except ImportError:
        print("MoviePy not found. Install: pip install moviepy")

try:
    import vlc
except ImportError:
    vlc = None
    print("VLC module not found. Install: pip install python-vlc")

# ==============================================================================
# 0. СТИЛИ ИНТЕРФЕЙСА (YOUTUBE DARK)
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

QLineEdit {
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

QFrame#VideoCard {
    background-color: #202020;
    border: 1px solid #333;
    border-radius: 8px;
}
QFrame#VideoCard:hover {
    border: 1px solid #555;
    background-color: #2a2a2a;
}

QPushButton#CardPlayBtn {
    background-color: transparent;
    border: none;
}
QPushButton#CardPlayBtn:hover {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 30px;
}

QPushButton#FsBtn {
    background-color: #e0e0e0;
    border: 1px solid #ccc;
    color: #000;
    border-radius: 4px;
}
QPushButton#FsBtn:hover { background-color: #fff; }

QTableWidget {
    background-color: #121212;
    gridline-color: #333;
    border: none;
    font-size: 13px;
}
QHeaderView::section {
    background-color: #202020;
    color: #aaa;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #333;
}
QTableWidget::item:selected {
    background-color: #333;
    color: #3ea6ff;
}

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    border: none;
    background: #181818;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #444;
    min-height: 20px;
    border-radius: 5px;
}

QSlider::groove:horizontal {
    border: 1px solid #333;
    height: 4px;
    background: #333;
    margin: 2px 0;
}
QSlider::sub-page:horizontal {
    background: #f00; 
}
QSlider::handle:horizontal {
    background: #f00;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}
"""

# ==============================================================================
# 1. ЛОГИКА АНАЛИЗА И МОНТАЖА
# ==============================================================================

CATEGORY_CONFIG = {
    'Focus_Highlights': {'col': 'Focus', 'threshold': 70, 'condition': 'ge'},       
    'Relaxation_Moments': {'col': 'Relaxation', 'threshold': 60, 'condition': 'ge'} 
}

MIN_SEGMENT_DURATION_SEC = 2.0 
MAX_GAP_MS = 2000               

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
    required = ['Timestamp_ms', 'Focus', 'Relaxation', 'Artifact']
    if not all(c in df.columns for c in required): raise ValueError(f"Missing cols: {required}")

    df = df[df['Artifact'] == 0].copy()
    if df.empty: raise ValueError("No valid data after artifact removal.")

    start_time_ms = df['Timestamp_ms'].iloc[0]
    
    # --- НАСТРОЙКА ПУТЕЙ (ИЗМЕНЕНО) ---
    # 1. Получаем директорию видео и имя файла без расширения
    video_dir = os.path.dirname(os.path.abspath(video_file))
    video_filename = os.path.basename(video_file)
    video_name_no_ext = os.path.splitext(video_filename)[0]

    # 2. Создаем главную папку для результатов (например, 'test')
    output_root_folder = os.path.join(video_dir, video_name_no_ext)
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)

    # 3. Копируем CSV внутрь этой папки (как на скриншоте)
    try:
        csv_filename = os.path.basename(csv_path)
        dest_csv_path = os.path.join(output_root_folder, csv_filename)
        shutil.copy2(csv_path, dest_csv_path)
    except Exception as e:
        print(f"Warning: Could not copy CSV: {e}")

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
            
            # --- ОПРЕДЕЛЕНИЕ ИМЕН ПАПОК И ФАЙЛОВ ---
            if "Focus" in category:
                folder_name = "Focus clips"
                montage_filename = "focus montage.mp4"
            else:
                folder_name = "Relaxation clips"
                montage_filename = "relaxation montage.mp4"

            # Создаем папку для клипов внутри главной папки
            segments_folder = os.path.join(output_root_folder, folder_name)
            if not os.path.exists(segments_folder): os.makedirs(segments_folder)
            
            for idx, row in segments.iterrows():
                s = (row['start_ms'] - start_time_ms) / 1000
                e = (row['end_ms'] - start_time_ms) / 1000
                
                if s >= full_clip.duration: continue
                if e > full_clip.duration: e = full_clip.duration
                
                # Создаем подклип
                sub = full_clip.subclipped(s, e) if hasattr(full_clip, 'subclipped') else full_clip.subclip(s, e)
                clips_for_montage.append(sub)
                
                # Сохраняем отдельный файл (для Галереи)
                seg_filename = f"segment_{idx+1:03d}_{row['avg_score']:.0f}.mp4"
                seg_path = os.path.join(segments_folder, seg_filename)
                
                # Preset ultrafast для скорости
                sub.write_videofile(seg_path, codec='libx264', audio_codec='aac', 
                                    remove_temp=True, logger=None, preset='ultrafast') 
                
                clip_duration = e - s
                segment_metadata.append({
                    'id': idx + 1,
                    'orig_start': s, 'orig_end': e, 
                    'montage_start': current_montage_time,
                    'duration': clip_duration, 
                    'score': row['avg_score'], 
                    'file_path': seg_path
                })
                current_montage_time += clip_duration
            
            # Собираем полный монтаж (для центрального плеера)
            if clips_for_montage:
                final = concatenate_videoclips(clips_for_montage)
                # Сохраняем монтаж в корень папки видео
                out_name = os.path.join(output_root_folder, montage_filename)
                
                final.write_videofile(out_name, codec='libx264', audio_codec='aac', 
                                      temp_audiofile='temp-audio.m4a', remove_temp=True, logger=None)
                
                results_data[category] = {
                    'path': out_name, # Путь к полному монтажу
                    'segments': segment_metadata,
                    'total_duration': final.duration
                }
                final.close()
                
    except Exception as e: raise Exception(f"Processing Error: {e}")
    finally:
        if full_clip: full_clip.close()
            
    return results_data

# ==============================================================================
# 2. WORKER (Фоновый поток)
# ==============================================================================

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

class VideoClickableFrame(QFrame):
    doubleClicked = pyqtSignal()
    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()

# ==============================================================================
# 3. ВИДЖЕТ-КАРТОЧКА ВИДЕО (Используется только в ГАЛЕРЕЕ)
# ==============================================================================
class VideoCardWidget(QFrame):
    """Карточка видео для галереи сегментов"""
    playClicked = pyqtSignal(str) 

    def __init__(self, title, video_path="", is_small=False):
        super().__init__()
        self.setObjectName("VideoCard")
        self.video_path = video_path
        self.is_small = is_small
        self.init_ui(title)

    def init_ui(self, title):
        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 10, 15, 10)

        # 1. Кнопка Play
        self.btn_play = QPushButton()
        self.btn_play.setObjectName("CardPlayBtn")
        icon_size = 40 
        self.btn_play.setFixedSize(icon_size, icon_size)
        self.btn_play.setIconSize(QSize(icon_size, icon_size))
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setCursor(Qt.PointingHandCursor)
        self.btn_play.clicked.connect(lambda: self.playClicked.emit(self.video_path))
        layout.addWidget(self.btn_play)

        # Контейнер текста
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0,0,0,0)
        text_layout.setSpacing(4)
        text_layout.setAlignment(Qt.AlignVCenter)

        # 2. Заголовок
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: #fff;")
        text_layout.addWidget(lbl_title)

        # 3. Инфо
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #aaa; font-size: 13px;")
        text_layout.addWidget(self.lbl_info)

        layout.addWidget(text_container)
        layout.addStretch()

    def set_data(self, path, duration_str, sub_info=""):
        self.video_path = path
        self.lbl_info.setText(f"{duration_str} {sub_info}")

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

class VideoAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Highlight Analyzer (Central Player Edition)")
        self.setGeometry(100, 100, 1100, 850)
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

    def init_vlc(self):
        if vlc:
            try:
                self.instance = vlc.Instance("--no-xlib")
                self.media_player = vlc.MediaPlayer(self.instance)
            except Exception: self.media_player = None

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Блок выбора файлов ---
        self.file_group = QGroupBox("Configuration")
        flayout = QVBoxLayout(self.file_group)
        
        # CSV
        r1 = QHBoxLayout()
        self.csv_path_input = QLineEdit()
        self.csv_path_input.setPlaceholderText("Select CSV...")
        b1 = QPushButton("Browse CSV")
        b1.clicked.connect(lambda: self.browse_file('csv'))
        r1.addWidget(self.csv_path_input)
        r1.addWidget(b1)
        flayout.addLayout(r1)
        
        # Video
        r2 = QHBoxLayout()
        self.video_path_input = QLineEdit()
        self.video_path_input.setPlaceholderText("Select Video...")
        b2 = QPushButton("Browse Video")
        b2.clicked.connect(lambda: self.browse_file('video'))
        r2.addWidget(self.video_path_input)
        r2.addWidget(b2)
        flayout.addLayout(r2)
        
        # Button
        self.analyze_button = QPushButton("START PROCESSING")
        self.analyze_button.setObjectName("ActionBtn")
        self.analyze_button.setCursor(Qt.PointingHandCursor)
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)
        flayout.addWidget(self.analyze_button)
        self.main_layout.addWidget(self.file_group)

        # --- ЦЕНТРАЛЬНЫЙ ПЛЕЕР ---
        self.video_container = QWidget()
        v_layout = QVBoxLayout(self.video_container)
        v_layout.setContentsMargins(0, 0, 0, 0)
        v_layout.setSpacing(0)
        
        self.video_frame = VideoClickableFrame()
        self.video_frame.setStyleSheet("background-color: #000;")
        self.video_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_frame.setMinimumHeight(420)
        self.video_frame.doubleClicked.connect(self.toggle_fullscreen)
        if not vlc:
            l = QVBoxLayout(); l.addWidget(QLabel("VLC Required", alignment=Qt.AlignCenter))
            self.video_frame.setLayout(l)
        v_layout.addWidget(self.video_frame)
        
        # Controls
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background-color: #202020;")
        ctrl = QHBoxLayout(self.controls_widget)
        
        self.play_button = QPushButton()
        self.play_button.setFixedSize(40, 40)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_video)
        ctrl.addWidget(self.play_button)
        
        self.time_label = QLabel("00:00 / 00:00")
        ctrl.addWidget(self.time_label)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.sliderMoved.connect(self.set_position)
        ctrl.addWidget(self.position_slider)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.valueChanged.connect(self.set_volume)
        ctrl.addWidget(self.volume_slider)

        self.fs_button = QPushButton()
        self.fs_button.setObjectName("FsBtn")
        self.fs_button.setFixedSize(40, 40)
        self.fs_button.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self.fs_button.clicked.connect(self.toggle_fullscreen)
        ctrl.addWidget(self.fs_button)

        v_layout.addWidget(self.controls_widget)
        self.main_layout.addWidget(self.video_container)

        # --- ВКЛАДКИ ---
        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tab_widget.setMinimumHeight(300)
        self.tab_widget.currentChanged.connect(self.tab_changed)
        
        self.tables = {}
        self.gallery_layouts = {}

        # Tab 1: Focus List (ТОЛЬКО ТАБЛИЦА)
        w1, t1 = self.create_list_tab()
        self.tables['Focus_Highlights'] = t1
        self.tab_widget.addTab(w1, "Focus (Full + List)")

        # Tab 2: Focus Video (Gallery)
        w2, l2 = self.create_gallery_tab()
        self.gallery_layouts['Focus_Highlights'] = l2
        self.tab_widget.addTab(w2, "Focus (Clips Gallery)")

        # Tab 3: Relax List (ТОЛЬКО ТАБЛИЦА)
        w3, t3 = self.create_list_tab()
        self.tables['Relaxation_Moments'] = t3
        self.tab_widget.addTab(w3, "Relaxation (Full + List)")

        # Tab 4: Relax Video (Gallery)
        w4, l4 = self.create_gallery_tab()
        self.gallery_layouts['Relaxation_Moments'] = l4
        self.tab_widget.addTab(w4, "Relaxation (Clips Gallery)")
            
        self.main_layout.addWidget(self.tab_widget)
        self.check_enable_button()

    def create_list_tab(self):
        """Создает вкладку с таблицей. Видео-карточка удалена, видео будет в центре."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Убрана карточка видео (card)

        # Table
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["#", "Time (Source)", "Duration", "Score"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.cellClicked.connect(self.on_segment_clicked)
        layout.addWidget(table)

        return widget, table

    def create_gallery_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(container)
        return scroll, layout

    # --- ЛОГИКА ---
    def check_enable_button(self):
        c = os.path.exists(self.csv_path_input.text())
        v = os.path.exists(self.video_path_input.text())
        self.analyze_button.setEnabled(c and v)

    def browse_file(self, mode):
        fltr = "CSV (*.csv)" if mode == 'csv' else "Video (*.mp4 *.avi *.mov *.mkv)"
        target = self.csv_path_input if mode == 'csv' else self.video_path_input
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", fltr)
        if path:
            target.setText(path)
            self.check_enable_button()

    def start_analysis(self):
        self.analyze_button.setText("PROCESSING...")
        self.analyze_button.setEnabled(False)
        self.worker = VideoAnalyzerWorker(self.csv_path_input.text(), self.video_path_input.text())
        self.worker.finished.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_success(self, res_data):
        self.analyze_button.setText("START PROCESSING")
        self.analyze_button.setEnabled(True)
        if not res_data:
            QMessageBox.information(self, "Info", "No highlights found.")
            return
        self.results_data = res_data
        QMessageBox.information(self, "Success", "Analysis Complete.")
        self.populate_data()

    def populate_data(self):
        for category, data in self.results_data.items():
            segments = data.get('segments', [])
            # full_path и total_dur используем при загрузке видео, но не отображаем в карточке внутри таба

            # 1. Заполнение LIST (Только таблица)
            if category in self.tables:
                table = self.tables[category]
                table.setRowCount(len(segments))
                for i, seg in enumerate(segments):
                    fmt = lambda s: f"{int(s)//60:02}:{int(s)%60:02}"
                    orig_time_str = f"{fmt(seg['orig_start'])} - {fmt(seg['orig_end'])}"
                    table.setItem(i, 0, QTableWidgetItem(str(seg['id'])))
                    table.setItem(i, 1, QTableWidgetItem(orig_time_str))
                    table.setItem(i, 2, QTableWidgetItem(f"{seg['duration']:.1f}s"))
                    table.setItem(i, 3, QTableWidgetItem(f"{seg['score']:.1f}"))
                    table.item(i, 0).setData(Qt.UserRole, seg['montage_start'])

            # 2. Заполнение GALLERY
            if category in self.gallery_layouts:
                layout = self.gallery_layouts[category]
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget(): child.widget().deleteLater()

                for seg in segments:
                    title = f"Segment #{seg['id']} (Score: {seg['score']:.0f})"
                    dur_str = f"{seg['duration']:.1f}s"
                    path = seg['file_path'] 
                    
                    seg_card = VideoCardWidget(title, path, is_small=True)
                    seg_card.playClicked.connect(self.force_play_video)
                    layout.addWidget(seg_card)
        
        self.tab_widget.setCurrentIndex(0)
        self.tab_changed(0) # Принудительный вызов для загрузки первого видео

    # --- ПЛЕЕР И ПЕРЕКЛЮЧЕНИЕ ---
    
    def load_video(self, path, autoplay=True):
        """
        Загружает видео в ЦЕНТРАЛЬНЫЙ плеер.
        autoplay=True: Сразу играет.
        autoplay=False: Показывает кадр и пауза.
        """
        if not vlc: return
        
        # Если путь тот же, управляем только состоянием Play/Pause
        if self.current_video_path == path and self.media_player.get_media():
            if autoplay and not self.media_player.is_playing():
                self.play_video()
            elif not autoplay and self.media_player.is_playing():
                self.media_player.pause()
                self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            return

        self.current_video_path = path
        self.media_player.stop()
        self.media_player.set_media(self.instance.media_new(path))
        
        if sys.platform == 'win32': self.media_player.set_hwnd(int(self.video_frame.winId()))
        elif sys.platform == 'darwin': 
            try: self.media_player.set_nsobject(int(self.video_frame.winId()))
            except: pass
        else: self.media_player.set_xwindow(int(self.video_frame.winId()))
        
        self.media_player.play()
        
        if autoplay:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.timer.start()
        else:
            # Трюк: Ставим на паузу почти сразу, чтобы отрисовался кадр
            self.media_player.set_pause(1) 
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.start()

        self.play_button.setEnabled(True)

    def tab_changed(self, idx):
        """
        Переключение вкладки -> Загрузка Полного Монтажа на Паузе в ЦЕНТРАЛЬНЫЙ плеер.
        """
        category = None
        if idx in [0, 1]: 
            category = 'Focus_Highlights'
        elif idx in [2, 3]: 
            category = 'Relaxation_Moments'
            
        if category:
            data = self.results_data.get(category)
            if data:
                full_montage_path = data.get('path')
                if full_montage_path and os.path.exists(full_montage_path):
                    # Загружаем полное видео, но не запускаем (autoplay=False)
                    self.load_video(full_montage_path, autoplay=False)

    def on_segment_clicked(self, row, col):
        """Клик по таблице -> Автоплей в центральном плеере"""
        sender = self.sender()
        if not sender: return
        item = sender.item(row, 0)
        if item:
            cat = None
            for c, t in self.tables.items():
                if t == sender: cat = c; break
            
            if cat:
                full_path = self.results_data.get(cat, {}).get('path')
                if full_path:
                    self.load_video(full_path, autoplay=True)
                    seek_time = item.data(Qt.UserRole)
                    self.media_player.set_time(int(seek_time * 1000))

    def force_play_video(self, path):
        """Клик по кнопке Play (в Галерее) -> Автоплей в центральном плеере"""
        if path and os.path.exists(path):
            self.load_video(path, autoplay=True)

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
    def on_error(self, msg):
        self.analyze_button.setText("ERROR")
        self.analyze_button.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.file_group.show()
            self.tab_widget.show()
            self.main_layout.setContentsMargins(15, 15, 15, 15)
            self.controls_widget.show()
        else:
            self.file_group.hide()
            self.tab_widget.hide()
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            self.showFullScreen()

    def update_ui_timer(self):
        if not self.media_player: return
        if self.media_player.get_state() == vlc.State.Ended:
             self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
             self.position_slider.setValue(0)
             return
        if self.media_player.is_playing():
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(int(self.media_player.get_position() * 1000))
            self.position_slider.blockSignals(False)
            cur = self.media_player.get_time()
            tot = self.media_player.get_length()
            fmt = lambda ms: f"{int(ms/1000)//60:02}:{int(ms/1000)%60:02}"
            self.time_label.setText(f"{fmt(cur)} / {fmt(tot)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VideoAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
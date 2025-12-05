import dearpygui.dearpygui as dpg
import numpy as np
import time
from threading import Thread, Event
from tkinter import Tk, filedialog
import cv2
import threading 

WIDTH = 1200
HEIGHT = 700

# Глобальные переменные
engagement_data = [50.0] * 100  # буфер графика (100 точек)
is_session_running = False
current_engagement = 50.0
time_counter = 0  # для плавной синусоиды
loaded_video_path = None

# Переменные для видео
video_player = None
video_thread = None
video_stop_event = Event()
current_video_frame = None
should_update_frame = False
frame_lock = threading.Lock()

# Имитация EEG-функции (в реальном проекте будет твоя функция)
def get_engagement():
    global time_counter
    # Плавная синусоида с шумом для красоты
    wave = 50 + 30 * np.sin(time_counter * 0.2)  # основа
    noise = np.random.uniform(-5, 5)  # шум
    time_counter += 0.1
    return max(0, min(100, wave + noise))

def update_ui():
    global is_session_running, current_engagement
    while True:
        if is_session_running:
            # Получаем уровень вовлечённости
            current_engagement = get_engagement()

            # Обновляем буфер графика
            engagement_data.pop(0)
            engagement_data.append(current_engagement)

            # Обновляем график и текст в UI-потоке (безопасно)
            try:
                dpg.set_value('engagement_series', [list(range(len(engagement_data))), engagement_data])
                dpg.set_value('engagement_text', f'Current engagement: {current_engagement:.1f}')
            except:
                pass  # игнорируем ошибки, если элементы ещё не созданы
        time.sleep(0.2)

def toggle_session():
    global is_session_running
    is_session_running = not is_session_running
    dpg.set_item_label('session_button', 'STOP SESSION' if is_session_running else 'START SESSION')
    dpg.set_value('status_text', 'EEG Reading: ONLINE' if is_session_running else 'EEG Reading: WAITING')

    # Управление видео при старте/остановке сессии
    if video_player and loaded_video_path:
        if is_session_running:
            start_video_playback()
        else:
            stop_video_playback()

def load_video():
    global loaded_video_path, video_player, current_video_frame, should_update_frame
    root = Tk()
    root.withdraw()  # скрываем главное окно
    path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("All files", "*.*")]
    )
    root.destroy()

    if path:
        loaded_video_path = path
        filename = path.split('/')[-1].split('\\')[-1]  # имя файла

        # Инициализируем видеоплеер, если нужно
        if video_player is None:
            video_player = VideoPlayer()

        # Загружаем видео
        if video_player.load_video(path):
            dpg.set_value('video_label', f"Loaded video: {filename}")

            # Обновляем комбобокс
            current_items = dpg.get_item_configuration('video_combo')['items']
            if filename not in dpg.get_item_configuration('video_combo')['items']:
                dpg.configure_item('video_combo', items=dpg.get_item_configuration('video_combo')['items'] + [filename])
                dpg.set_value('video_combo', filename)

            # Останавливаем предыдущее воспроизведение
            stop_video_playback()

            # Показываем первый кадр
            first_frame = video_player.read_next_frame()
            if first_frame is not None:
                with frame_lock:
                    current_video_frame = first_frame
                    should_update_frame = True
        else:
            dpg.set_value('video_label', f"Failed to load: {filename}")
            loaded_video_path = None

def video_playback_thread():
    """Поток для воспроизведения видео - только читает кадры"""
    global current_video_frame, should_update_frame
    while not video_stop_event.is_set():
        if video_player and video_player.is_playing and loaded_video_path:
            frame_data = video_player.read_next_frame()
            if frame_data is not None:
                with frame_lock:
                    current_video_frame = frame_data
                    should_update_frame = True
                # Задержка для соответствия FPS видео
                time.sleep(1.0 / video_player.fps if video_player.fps > 0 else 0.033)
            else:
                time.sleep(0.01)
        else:
            time.sleep(0.01)

def start_video_playback():
    """Начать воспроизведение видео"""
    global video_thread
    if video_player and loaded_video_path:
        video_player.play()

        # Запускаем поток воспроизведения, если его нет
        if video_thread is None or not video_thread.is_alive():
            video_stop_event.clear()
            video_thread = Thread(target=video_playback_thread, daemon=True)
            video_thread.start()

def stop_video_playback():
    """Остановить воспроизведение видео"""
    if video_player:
        video_player.pause()
        video_stop_event.set()

def update_video_frame():
    """Обновление кадра видео (вызывается в основном цикле)"""
    global should_update_frame, current_video_frame

    if should_update_frame and current_video_frame is not None:
        with frame_lock:
            frame = current_video_frame
            should_update_frame = False

        # Обновляем texture через queue, чтобы быть в UI-потоке
        try:
            dpg.set_value("video_texture_data", frame)
        except:
            pass  # игнорируем ошибки обновления

# Класс видеоплеера
class VideoPlayer:
    def __init__(self):
        self.cap = None
        self.is_playing = False
        self.video_width = 600
        self.video_height = 350
        self.fps = 30

    def load_video(self, file_path):
        """Загрузить видеофайл"""
        try:
            if self.cap:
                self.cap.release()

            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                print(f"Не удалось открыть видео {file_path}")
                return False

            # Получаем свойства видео
            original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30

            # Масштабируем для отображения
            max_width = 600
            max_height = 350
            scale = min(max_width / original_width, max_height / original_height)
            self.video_width = int(original_width * scale)
            self.video_height = int(original_height * scale)

            print(f"Видео загружено: {original_width}x{original_height} -> {self.video_width}x{self.video_height}, FPS: {self.fps:.1f}")

            return True

        except Exception as e:
            print(f"Ошибка загрузки видео: {e}")
            return False

    def read_next_frame(self):
        """Прочитать следующий кадр"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Изменяем размер кадра
                frame_resized = cv2.resize(frame, (self.video_width, self.video_height))
                # Конвертируем BGR в RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                # Нормализуем
                frame_normalized = frame_rgb.astype(np.float32) / 255.0
                return frame_normalized.flatten()
            else:
                # Если видео закончилось, перематываем в начало
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return self.read_next_frame()
        return None

    def play(self):
        """Начать воспроизведение"""
        self.is_playing = True

    def pause(self):
        """Приостановить воспроизведение"""
        self.is_playing = False

    def release(self):
        """Освободить ресурсы"""
        if self.cap:
            self.cap.release()
        self.is_playing = False


# Инициализация Dear PyGui
dpg.create_context()

# РЕГИСТР ТЕКСТУР (важно!)
with dpg.texture_registry():
    # Чёрный кадр по умолчанию
    black_frame = np.zeros((350 * 600 * 3), dtype=np.float32)
    dpg.add_raw_texture(
        width=600,
        height=350,
        default_value=black_frame,
        format=dpg.mvFormat_Float_rgb,
        tag="video_texture_data"
    )

dpg.create_viewport(title='Engagement Tracker', width=WIDTH, height=HEIGHT)

# Основное окно
with dpg.window(label="Engagement Tracker", width=WIDTH, height=HEIGHT, no_collapse=True, no_resize=True):

    # Основной layout: видео слева, график справа
    with dpg.group(horizontal=True):
        # Левая колонка: видео + таймлайн
        with dpg.group(horizontal=False, width=600):
            dpg.add_text("Selected video:", color=(200, 200, 200))
            dpg.add_combo(items=["video1.mp4", "video2.mp4"], default_value="video1.mp4", tag='video_combo')

            dpg.add_text("Demo video", color=(255, 255, 0))
            dpg.add_text("EEG Reading: WAITING", tag='status_text', color=(150, 150, 150))

            # Кнопка загрузки видео
            dpg.add_button(label="Load Video", callback=load_video, width=80)

            # Кнопки управления видео
            with dpg.group(horizontal=True, width=200):
                dpg.add_button(label="Play Video", callback=start_video_playback, width=80)
                dpg.add_button(label="Pause Video", callback=stop_video_playback, width=80)

            # Отображение имени загруженного видео
            dpg.add_text("No video loaded", tag='video_label', color=(180, 180, 180))

            # Отображаем видео
            dpg.add_image(
                "video_texture_data",
                width=600,
                height=350,
                tag="video_image"
            )

            # Heatmap timeline под видео
            dpg.add_text("Timeline heatmap", color=(200, 200, 200))
            with dpg.drawlist(width=600, height=30, tag="heatmap_drawlist"):
                pass

        # Правая колонка: график и текущее значение
        with dpg.group(horizontal=False, width=500):
            dpg.add_text("Engagement Level", color=(255, 255, 255), indent=20)

            # Текущее значение
            dpg.add_text("Current engagement: —", tag='engagement_text', color=(100, 200, 255), indent=20)

            # График
            with dpg.plot(label="Engagement Graph", height=350, width=500):
                dpg.add_plot_axis(dpg.mvXAxis, tag='x_axis')
                dpg.add_plot_axis(dpg.mvYAxis, tag='y_axis')
                dpg.add_line_series(list(range(100)), engagement_data, tag='engagement_series', parent='y_axis')

    # Кнопка Start/Stop внизу по центру
    dpg.add_button(label='START SESSION', callback=toggle_session, tag='session_button', width=200, height=40)
    # Центрируем кнопку вручную
    dpg.set_item_pos('session_button', (WIDTH // 2 - 100, HEIGHT - 60))

# Обновление heatmap в основном цикле
def render_heatmap():
    if not is_session_running:
        return
    dpg.delete_item("heatmap_drawlist", children_only=True)
    for i in range(100):
        val = engagement_data[i] / 100
        r, g = int(255 * (1 - val)), int(255 * val)
        dpg.draw_rectangle((i * 6, 0), ((i + 1) * 6, 30), fill=(r, g, 0, 200), thickness=0, parent="heatmap_drawlist")

# Запуск обновления данных в отдельном потоке (запускаем ДО show_viewport)
ui_thread = Thread(target=update_ui, daemon=True)
ui_thread.start()

# Инициализация видеоплеера
video_player = VideoPlayer()

# Основной цикл
dpg.setup_dearpygui()
dpg.show_viewport()

last_time = time.time()
frame_count = 0

while dpg.is_dearpygui_running():
    render_heatmap()
    update_video_frame()
    dpg.render_dearpygui_frame()

    # Простой FPS счетчик для отладки
    frame_count += 1
    current_time = time.time()
    if current_time - last_time >= 1.0:
        # print(f"FPS: {frame_count}")
        frame_count = 0
        last_time = current_time

# Очистка при выходе
if video_player:
    video_player.release()
video_stop_event.set()
dpg.destroy_context()
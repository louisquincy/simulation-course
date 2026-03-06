import sys
import numpy as np
from numba import njit
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QFrame, 
                             QStyleFactory, QGridLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPainter, QColor

# --- НАСТРОЙКИ (можно крутить тут) ---
GRID_SIZE = 200       # размер карты, если лагает - ставим 100 или 150
CELL_SCALE = 6        # масштаб отрисовки (не влияет на логику)

# состояния клеток (просто цифры для массива)
S_DIRT  = 0
S_TREE  = 1
S_FIRE  = 2
S_ASH   = 3
S_WATER = 4
S_ROCK  = 5

COLORS = np.array([
    [30,  40,  60,  255],   # земля
    [34,  139, 34,  255],   # дерево
    [10,  80,  255, 255],   # огонь 
    [20,  20,  20,  255],   # пепел
    [220, 120, 30,  255],   # вода
    [180, 180, 180, 255],   # скалы
], dtype=np.uint8)


# ==========================================
# ЛОГИЧЕСКОЕ ЯДРО (КЛЕТОЧНЫЙ АВТОМАТ)
# ==========================================

# @njit ускоряет питон в 100 раз, компилируя эту функцию
@njit(fastmath=True)
def logic(grid, next_grid, height_map, timers, 
                 p_grow, p_ignite, wind_x, wind_y, wind_str):
    
    rows, cols = grid.shape
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1), 
               (1, 0), (1, -1), (0, -1), (-1, -1)]

    for r in range(rows):
        for c in range(cols):
            state = grid[r, c]
            
            if state == S_ROCK or state == S_WATER:
                next_grid[r, c] = state
                continue

            if state == S_ASH:
                if timers[r, c] > 0:
                    timers[r, c] -= 1 
                    next_grid[r, c] = S_ASH
                else:
                    next_grid[r, c] = S_DIRT # остыло, можно растить лес
            
            elif state == S_FIRE:
                next_grid[r, c] = S_ASH
                timers[r, c] = 10  # ставим таймер на 10 ходов
            
            elif state == S_DIRT:
                if np.random.random() < p_grow:
                    next_grid[r, c] = S_TREE
                else:
                    next_grid[r, c] = S_DIRT
            
            elif state == S_TREE:
                if p_ignite <= 0.0001:
                    next_grid[r, c] = S_TREE
                    continue

                catch_fire = False
                
                if np.random.random() < 0.000002:
                    catch_fire = True
                else:
                    for dr, dc in offsets:
                        nr, nc = r + dr, c + dc
                        
                        if 0 <= nr < rows and 0 <= nc < cols:
                            # если сосед горит
                            if grid[nr, nc] == S_FIRE:
                                prob = p_ignite
                            
                                vec_x, vec_y = -dc, -dr
                                dot = (vec_x * wind_x + vec_y * wind_y)
                                prob += (dot * wind_str * 0.4) # 0.4 - это коэффициент влияния ветра

                                slope = height_map[r, c] - height_map[nr, nc]
                                if slope > 0:
                                    prob += slope * 6.0 # сильный бонус если мы выше огня
                                else:
                                    prob -= 0.05        # если мы ниже - горит хуже
                                
                                if np.random.random() < prob:
                                    catch_fire = True
                                    break
                
                if catch_fire:
                    next_grid[r, c] = S_FIRE
                else:
                    next_grid[r, c] = S_TREE


# генерация карты (шум + сглаживание)
@njit
def generate_map_numba(size):
    h_map = np.random.rand(size, size)
    
    # сглаживаем шум, чтобы получились холмы
    for _ in range(4):
        temp = np.empty_like(h_map)
        for r in range(size):
            for c in range(size):
                # берем среднее арифметическое соседей
                val = h_map[r, c] * 4
                cnt = 4
                if r > 0: val += h_map[r-1, c]; cnt+=1
                if r < size-1: val += h_map[r+1, c]; cnt+=1
                if c > 0: val += h_map[r, c-1]; cnt+=1
                if c < size-1: val += h_map[r, c+1]; cnt+=1
                temp[r, c] = val / cnt
        h_map = temp

    min_h, max_h = h_map.min(), h_map.max()
    if max_h > min_h:
        h_map = (h_map - min_h) / (max_h - min_h)

    grid = np.zeros((size, size), dtype=np.int8)
    for r in range(size):
        for c in range(size):
            h = h_map[r, c]
            if h < 0.20: grid[r, c] = S_WATER  # низины - вода
            elif h > 0.65: grid[r, c] = S_ROCK # верхушки - скалы
            else:
                if np.random.random() < 0.8: grid[r, c] = S_TREE
                else: grid[r, c] = S_DIRT
                    
    return grid, h_map


# ==========================================
# GUI И ОТРИСОВКА (PYQT5)
# ==========================================

class SimDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage(GRID_SIZE, GRID_SIZE, QImage.Format_ARGB32)
        self.image.fill(Qt.black)

    # обновление картинки из массива numpy
    def update_view(self, grid_data):
        buf = COLORS[grid_data].tobytes()
        self.image = QImage(buf, GRID_SIZE, GRID_SIZE, GRID_SIZE * 4, QImage.Format_ARGB32)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image)

class FireApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Симуляция лесных пожаров | Финал")
        self.resize(1100, 900)
        
        # стили интерфейса (цвета, отступы)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #eeeeee; font-family: 'Segoe UI', sans-serif; font-size: 14px; }
            QFrame#Panel { background-color: #2b2b2b; border-top: 1px solid #3d3d3d; }
            QPushButton { 
                background-color: #3a3a3a; color: white; border: 1px solid #555; 
                padding: 8px; border-radius: 4px; font-weight: bold; min-width: 120px;
            }
            QPushButton:hover { background-color: #4a4a4a; border-color: #777; }
            QPushButton:checked { background-color: #e67e22; border-color: #e67e22; color: #fff; }
            QLabel#H1 { font-size: 16px; font-weight: bold; color: #aaa; margin-bottom: 5px; }
            QSlider::groove:horizontal { height: 6px; background: #444; border-radius: 3px; }
            QSlider::handle:horizontal { background: #3498db; width: 16px; margin: -5px 0; border-radius: 8px; }
        """)

        # инициализация массивов
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.next_grid = np.zeros_like(self.grid)
        self.timers = np.zeros_like(self.grid)
        self.height_map = np.zeros((GRID_SIZE, GRID_SIZE))
        self.is_running = False

        self.init_ui()
        self.start_new_map()
        
        # таймер основного цикла (30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.loop)
        self.timer.start(30)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # холст симуляции
        self.display = SimDisplay()
        layout.addWidget(self.display, stretch=1)
        
        # панель управления снизу
        panel = QFrame()
        panel.setObjectName("Panel")
        p_layout = QHBoxLayout(panel)
        p_layout.setContentsMargins(20, 20, 20, 20)
        p_layout.setSpacing(30)
        
        # колонка 1: кнопки
        col1 = QVBoxLayout()
        col1.addWidget(QLabel("Управление", objectName="H1"))
        self.btn_run = QPushButton("СТАРТ / ПАУЗА")
        self.btn_run.setCheckable(True)
        self.btn_run.clicked.connect(self.toggle)
        col1.addWidget(self.btn_run)
        
        self.btn_new = QPushButton("НОВАЯ КАРТА")
        self.btn_new.clicked.connect(self.start_new_map)
        col1.addWidget(self.btn_new)
        col1.addStretch()
        p_layout.addLayout(col1, stretch=0)

        # колонка 2: параметры симуляции
        col2 = QVBoxLayout()
        col2.addWidget(QLabel("Параметры", objectName="H1"))
        self.sl_grow = self.mk_slider(col2, "Скорость роста", 0, 100, 5)
        self.sl_fire = self.mk_slider(col2, "Шанс возгорания", 0, 100, 25)
        col2.addStretch()
        p_layout.addLayout(col2, stretch=1)

        # колонка 3: настройки ветра
        col3 = QVBoxLayout()
        col3.addWidget(QLabel("Ветер", objectName="H1"))
        self.sl_wdir = self.mk_slider(col3, "Направление", 0, 360, 90)
        self.sl_wspd = self.mk_slider(col3, "Сила ветра", 0, 50, 10)
        col3.addStretch()
        p_layout.addLayout(col3, stretch=1)

        # колонка 4: легенда (сеткой)
        col4 = QVBoxLayout()
        col4.addWidget(QLabel("Легенда", objectName="H1"))
        
        grid_leg = QGridLayout()
        grid_leg.setSpacing(10)
        
        items = [
            ("Лес", "#228b22"), ("Огонь", "#ff5500"),
            ("Вода", "#1e90ff"), ("Скалы", "#b4b4b4")
        ]
        
        for i, (txt, hex_c) in enumerate(items):
            frame = QFrame()
            frame.setFixedSize(16, 16)
            frame.setStyleSheet(f"background-color: {hex_c}; border-radius: 3px;")
            lbl = QLabel(txt)
            lbl.setStyleSheet("font-weight: bold; color: #ccc;")
            grid_leg.addWidget(frame, i, 0)
            grid_leg.addWidget(lbl, i, 1)
            
        col4.addLayout(grid_leg)
        col4.addStretch()
        p_layout.addLayout(col4, stretch=0)

        layout.addWidget(panel)

    # вспомогательный метод для создания слайдеров
    def mk_slider(self, layout, title, vmin, vmax, vdef):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 5, 0, 5)
        l.setSpacing(2)
        
        top = QHBoxLayout()
        lbl_t = QLabel(title)
        lbl_v = QLabel(str(vdef))
        lbl_v.setStyleSheet("color: #3498db; font-weight: bold;")
        top.addWidget(lbl_t)
        top.addStretch()
        top.addWidget(lbl_v)
        
        sl = QSlider(Qt.Horizontal)
        sl.setRange(vmin, vmax)
        sl.setValue(vdef)
        sl.valueChanged.connect(lambda v: lbl_v.setText(str(v)))
        
        l.addLayout(top)
        l.addWidget(sl)
        layout.addWidget(w)
        return sl

    def start_new_map(self):
        self.is_running = False
        self.btn_run.setChecked(False)
        self.grid, self.height_map = generate_map_numba(GRID_SIZE)
        self.timers.fill(0)
        self.display.update_view(self.grid)

    def toggle(self):
        self.is_running = not self.is_running

    # игровой цикл
    def loop(self):
        if not self.is_running:
            return

        # собираем значения со слайдеров
        p_grow = self.sl_grow.value() / 1000.0
        p_ignite = self.sl_fire.value() / 100.0
        
        angle = np.radians(self.sl_wdir.value())
        wx, wy = np.cos(angle), np.sin(angle)
        wstr = self.sl_wspd.value() / 10.0

        # вызываем расчет следующего кадра
        logic(self.grid, self.next_grid, self.height_map, self.timers,
                     p_grow, p_ignite, wx, wy, wstr)
        
        # обновляем массив и картинку
        self.grid[:] = self.next_grid[:]
        self.display.update_view(self.grid)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    w = FireApp()
    w.show()
    sys.exit(app.exec_())
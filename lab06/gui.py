import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QTabWidget,
    QHeaderView, QGroupBox, QGridLayout, QProgressBar, QTextEdit, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QDoubleValidator

from rng import MultiKongGen
from logic import (
    generate_discr_samples, compute_discr_stats, chi2_critical_discr,
    generate_norm_samples, compute_norm_stats, NORM_BIN_LABELS
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

class AnimatedButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(80)
        self.animation.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.original_geometry = None
        self.animation.finished.connect(self._restore_geometry)

    def mousePressEvent(self, event):
        if self.original_geometry is None:
            self.original_geometry = self.geometry()
        self.animation.stop()
        self.animation.setStartValue(self.original_geometry)
        self.animation.setEndValue(self.original_geometry.adjusted(2, 2, -2, -2))
        self.animation.start()
        super().mousePressEvent(event)

    def _restore_geometry(self):
        if self.original_geometry:
            self.setGeometry(self.original_geometry)

class DiscrTab(QWidget):
    def __init__(self, rng, log_callback):
        super().__init__()
        self.rng = rng
        self.log = log_callback
        self.values = [1, 2, 3, 4, 5]
        self.prob_inputs = []
        self.theor_probs = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Группа задания вероятностей
        prob_group = QGroupBox("Задайте вероятности (неотрицательные числа, сумма может быть любой)")
        prob_layout = QGridLayout()
        default_probs = [10, 20, 30, 20, 20]
        for i, val in enumerate(self.values):
            lbl = QLabel(f"P(X={val}):")
            le = QLineEdit()
            le.setValidator(QDoubleValidator(0.0, 1e9, 5))
            le.setText(str(default_probs[i]))
            self.prob_inputs.append(le)
            prob_layout.addWidget(lbl, 0, 2*i)
            prob_layout.addWidget(le, 0, 2*i+1)
        self.btn_update = AnimatedButton("Обновить распределение")
        prob_layout.addWidget(self.btn_update, 1, 0, 1, 10)
        prob_group.setLayout(prob_layout)
        layout.addWidget(prob_group)

        # Группа выбора N
        ctrl_group = QGroupBox("Выбор объёма выборки")
        ctrl_layout = QHBoxLayout()
        self.btn_n10 = AnimatedButton("N = 10")
        self.btn_n100 = AnimatedButton("N = 100")
        self.btn_n1000 = AnimatedButton("N = 1 000")
        self.btn_n10000 = AnimatedButton("N = 10 000")
        self.btn_clear = AnimatedButton("Сброс таблицы")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ctrl_layout.addWidget(self.btn_n10)
        ctrl_layout.addWidget(self.btn_n100)
        ctrl_layout.addWidget(self.btn_n1000)
        ctrl_layout.addWidget(self.btn_n10000)
        ctrl_layout.addWidget(self.btn_clear)
        ctrl_layout.addWidget(self.progress)
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        # Таблица результатов
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "N", "Теор. среднее", "Эмп. среднее", "Погр. средн.(%)",
            "Теор. дисп.", "Эмп. дисп.", "Погр. дисп.(%)", "χ² (гипотеза)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        # Эмпирические вероятности
        emp_group = QGroupBox("Эмпирические вероятности (последний эксперимент)")
        emp_layout = QGridLayout()
        self.emp_prob_labels = []
        for i, val in enumerate(self.values):
            lbl = QLabel(f"P(X={val}): --")
            emp_layout.addWidget(lbl, 0, i)
            self.emp_prob_labels.append(lbl)
        emp_group.setLayout(emp_layout)
        layout.addWidget(emp_group)

        self.setLayout(layout)

        # Сигналы
        self.btn_update.clicked.connect(self.update_distribution)
        self.btn_n10.clicked.connect(lambda: self.run_experiment(10))
        self.btn_n100.clicked.connect(lambda: self.run_experiment(100))
        self.btn_n1000.clicked.connect(lambda: self.run_experiment(1000))
        self.btn_n10000.clicked.connect(lambda: self.run_experiment(10000))
        self.btn_clear.clicked.connect(self.clear_table)

        # Инициализация распределения по умолчанию
        self.update_distribution()

    def update_distribution(self):
        raw = []
        for le in self.prob_inputs:
            try:
                v = float(le.text().replace(',', '.'))
                if v < 0:
                    raise ValueError
                raw.append(v)
            except:
                raw.append(0.0)
        total = sum(raw)
        if total == 0:
            self.log("Ошибка: сумма вероятностей равна нулю. Использую равномерное распределение.")
            self.theor_probs = [1.0/len(self.values)] * len(self.values)
        else:
            self.theor_probs = [w / total for w in raw]
        norm_str = ", ".join([f"{p:.4f}" for p in self.theor_probs])
        self.log(f"Распределение обновлено: нормированные вероятности = [{norm_str}]")
        self.clear_table()

    def run_experiment(self, n):
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        QTimer.singleShot(10, lambda: self._do_run(n))

    def _do_run(self, n):
        samples = generate_discr_samples(self.rng, self.values, self.theor_probs, n)
        stats = compute_discr_stats(samples, self.values, self.theor_probs)
        
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(stats["n"])))
        self.table.setItem(row, 1, QTableWidgetItem(f"{stats['theor_mean']:.4f}"))
        self.table.setItem(row, 2, QTableWidgetItem(f"{stats['emp_mean']:.4f}"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{stats['mean_error']:.2f}"))
        self.table.setItem(row, 4, QTableWidgetItem(f"{stats['theor_var']:.4f}"))
        self.table.setItem(row, 5, QTableWidgetItem(f"{stats['emp_var']:.4f}"))
        self.table.setItem(row, 6, QTableWidgetItem(f"{stats['var_error']:.2f}"))
        self.table.setItem(row, 7, QTableWidgetItem(stats["chi2_comparison"]))
        
        for i, p in enumerate(stats["emp_probs"]):
            self.emp_prob_labels[i].setText(f"P(X={self.values[i]}): {p:.4f}")
        
        self.progress.setVisible(False)
        self.log(f"[ДСВ] N={n}: среднее={stats['emp_mean']:.4f} (погр.{stats['mean_error']:.2f}%), χ²={stats['chi2']:.2f} → {stats['hypothesis']}")
    
    def clear_table(self):
        self.table.setRowCount(0)
        for i, val in enumerate(self.values):
            self.emp_prob_labels[i].setText(f"P(X={val}): --")
        self.log("[ДСВ] Таблица результатов очищена")


class NormTab(QWidget):
    def __init__(self, rng, log_callback):
        super().__init__()
        self.rng = rng
        self.log = log_callback
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Поля для задания мат. ожидания и дисперсии
        params_group = QGroupBox("Параметры распределения N(μ, σ²)")
        params_layout = QHBoxLayout()
        
        self.mean_input = QLineEdit("0.0")
        self.var_input = QLineEdit("1.0")
        
        params_layout.addWidget(QLabel("Мат. ожидание (μ):"))
        params_layout.addWidget(self.mean_input)
        params_layout.addWidget(QLabel("Дисперсия (σ² > 0):"))
        params_layout.addWidget(self.var_input)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        ctrl_group = QGroupBox("Выбор объёма выборки")
        ctrl_layout = QHBoxLayout()
        self.btn_n10 = AnimatedButton("N = 10")
        self.btn_n100 = AnimatedButton("N = 100")
        self.btn_n1000 = AnimatedButton("N = 1 000")
        self.btn_n10000 = AnimatedButton("N = 10 000")
        self.btn_clear = AnimatedButton("Сброс")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ctrl_layout.addWidget(self.btn_n10)
        ctrl_layout.addWidget(self.btn_n100)
        ctrl_layout.addWidget(self.btn_n1000)
        ctrl_layout.addWidget(self.btn_n10000)
        ctrl_layout.addWidget(self.btn_clear)
        ctrl_layout.addWidget(self.progress)
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["N", "Среднее", "Дисперсия", "Погр. средн.(абс)", "Погр. дисп.(%)", "χ² (результат)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        if MATPLOTLIB:
            self.figure = Figure(figsize=(5, 3), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
        else:
            self.hist_label = QLabel("Matplotlib не установлен, гистограмма недоступна")
            self.hist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.hist_label)

        self.setLayout(layout)

        self.btn_n10.clicked.connect(lambda: self.run(10))
        self.btn_n100.clicked.connect(lambda: self.run(100))
        self.btn_n1000.clicked.connect(lambda: self.run(1000))
        self.btn_n10000.clicked.connect(lambda: self.run(10000))
        self.btn_clear.clicked.connect(self.clear)

    def run(self, n):
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        QTimer.singleShot(10, lambda: self._do_run(n))

    def _do_run(self, n):
        # Парсинг параметров
        try:
            mean = float(self.mean_input.text().replace(',', '.'))
            variance = float(self.var_input.text().replace(',', '.'))
            if variance <= 0:
                raise ValueError
        except ValueError:
            self.log("Ошибка: Некорректные μ или σ². Используются значения по умолчанию (μ=0, σ²=1).")
            mean, variance = 0.0, 1.0
            self.mean_input.setText("0.0")
            self.var_input.setText("1.0")

        samples = generate_norm_samples(self.rng, n, mean, variance)
        stats = compute_norm_stats(samples, mean, variance)
        
        chi2_text = f"{stats['chi2']:.2f} {'<' if stats['chi2'] <= stats['chi2_crit'] else '>'} {stats['chi2_crit']}"
        hypo = "принимается" if stats['chi2'] <= stats['chi2_crit'] else "отвергается"
        
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(stats["n"])))
        self.table.setItem(row, 1, QTableWidgetItem(f"{stats['emp_mean']:.4f}"))
        self.table.setItem(row, 2, QTableWidgetItem(f"{stats['emp_var']:.4f}"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{stats['mean_abs_err']:.4f}"))
        self.table.setItem(row, 4, QTableWidgetItem(f"{stats['var_rel_err']:.2f}"))
        self.table.setItem(row, 5, QTableWidgetItem(f"{chi2_text} ({hypo})"))

        if MATPLOTLIB:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Динамическая гистограмма, подстраивается под сгенерированные данные
            ax.hist(samples, bins=15, density=True, alpha=0.7, label='Эмпирическая плотность', color='#89b4fa')
            
            # Построение теоретической кривой
            std_dev = variance ** 0.5
            x_min, x_max = mean - 4 * std_dev, mean + 4 * std_dev
            x = [x_min + i*(x_max - x_min)/100 for i in range(101)]
            y = [1.0/(std_dev * (2*3.14159)**0.5) * 2.71828**(-0.5 * ((xi - mean)/std_dev)**2) for xi in x]
            
            ax.plot(x, y, 'r-', label=f'N({mean}, {variance})')
            ax.set_title(f"Гистограмма для n={stats['n']}")
            ax.set_xlabel("Значение")
            ax.set_ylabel("Плотность")
            ax.legend()
            self.canvas.draw()

        self.progress.setVisible(False)
        self.log(f"[Нормальная N({mean}, {variance})] N={n}: среднее={stats['emp_mean']:.4f} (абс.погр.{stats['mean_abs_err']:.4f}), χ²={stats['chi2']:.2f} → гипотеза {hypo}")

    def clear(self):
        self.table.setRowCount(0)
        if MATPLOTLIB:
            self.figure.clear()
            self.canvas.draw()
        self.log("[Нормальная] Таблица и гистограмма очищены")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа №6: Моделирование ДСВ и Нормальной СВ (с настраиваемыми параметрами)")
        self.setMinimumSize(1100, 800)
        self.rng = MultiKongGen()

        self.setStyleSheet("""
            QMainWindow { background: #1e1e2f; }
            QGroupBox { color: #cdd6f4; font: bold 12px; border: 1px solid #313244; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            QLabel, QTextEdit { color: #cdd6f4; }
            QLineEdit { background: #181825; color: #cdd6f4; border: 1px solid #313244; border-radius: 4px; padding: 4px; }
            QTableWidget { background: #181825; color: #cdd6f4; alternate-background-color: #1e1e2e; }
            QHeaderView::section { background: #313244; color: #cdd6f4; }
            QPushButton { background: #89b4fa; color: #1e1e2e; border-radius: 8px; padding: 6px 12px; font-weight: bold; }
            QPushButton:hover { background: #b4befe; }
            QProgressBar { border: 1px solid #313244; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background: #a6e3a1; border-radius: 5px; }
            QTextEdit { background: #181825; border: 1px solid #313244; font-family: monospace; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Сначала создаём текстовое поле для логов (чтобы вкладки могли его использовать)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)

        # Теперь создаём вкладки
        self.tabs = QTabWidget()
        self.discr_tab = DiscrTab(self.rng, self.log_message)
        self.norm_tab = NormTab(self.rng, self.log_message)
        self.tabs.addTab(self.discr_tab, "Дискретная СВ (настраиваемая)")
        self.tabs.addTab(self.norm_tab, "Нормальная СВ (N(μ, σ²))")

        layout.addWidget(self.tabs)
        layout.addWidget(QLabel("Выводы и отладочная информация:"))
        layout.addWidget(self.log_text)

        self.log_message("Приложение запущено. Задайте вероятности и нажмите 'Обновить распределение'.")
        self.log_message("ГПСЧ собственный (MultiKongGen).")

    def log_message(self, msg):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import time
from numba import njit
import threading

L = 0.1                 # толщина пластины
T_left = 0.0            # температура на левой границе
T_right = 200.0         # температура на правой границе
rho = 5326.0            # плотность
c = 310.0               # удельная теплоёмкость
lambda_ = 60.2          # теплопроводность

def solve_step(T_curr, h, tau, rho, c, lam, T_left, T_right):
    n = len(T_curr)
    coeff = lam / h ** 2
    time_coeff = rho * c / tau

    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    F = np.zeros(n)

    for i in range(1, n - 1):
        A[i] = coeff
        B[i] = 2.0 * coeff + time_coeff
        C[i] = coeff
        F[i] = -time_coeff * T_curr[i]

    return progonka(A, B, C, F, T_left, T_right)

@njit
def progonka(A, B, C, F, T_left, T_right):

    n = len(B)
    alpha = np.zeros(n - 1)
    beta = np.zeros(n - 1)

    alpha[0] = 0.0
    beta[0] = T_left

    for i in range(1, n - 1):
        denom = B[i] - C[i] * alpha[i - 1]
        alpha[i] = A[i] / denom
        beta[i] = (C[i] * beta[i - 1] - F[i]) / denom

    T = np.zeros(n)
    T[n - 1] = T_right
    for i in range(n - 2, -1, -1):
        T[i] = alpha[i] * T[i + 1] + beta[i]

    return T

# ----------------------------------------------------------------------
# Класс приложения с возможностью выбора шагов
# ----------------------------------------------------------------------
class HeatConductionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Моделирование теплопроводности в пластине")
        self.root.geometry("1300x800")

        # Все возможные шаги
        self.all_dt = [0.1, 0.01, 0.001, 0.0001]
        self.all_h = [0.1, 0.01, 0.001, 0.0001]

        # Выбранные шаги (по умолчанию только крупные, чтобы не зависало)
        self.selected_dt = [0.1, 0.01]
        self.selected_h = [0.1, 0.01]

        self.total_time = 2.0  # время моделирования, с

        # Результаты
        self.T_center_dict = {}  # ключ (dt, h) -> значение
        self.cpu_time_dict = {}
        self.last_profile = None
        self.last_x = None

        self.create_widgets()
        self.update_estimate()

    def create_widgets(self):
        # Левая панель (управление)
        left_frame = ttk.Frame(self.root, padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        # Физические параметры
        params_frame = ttk.LabelFrame(left_frame, text="Параметры задачи", padding=5)
        params_frame.pack(fill=tk.X, pady=5)

        ttk.Label(params_frame, text=f"L = {L} м").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(params_frame, text=f"ρ = {rho} кг/м³").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(params_frame, text=f"c = {c} Дж/(кг·°С)").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(params_frame, text=f"λ = {lambda_} Вт/(м·°С)").grid(row=3, column=0, sticky=tk.W)
        ttk.Label(params_frame, text=f"T_left = {T_left} °C").grid(row=0, column=1, sticky=tk.W)
        ttk.Label(params_frame, text=f"T_right = {T_right} °C").grid(row=1, column=1, sticky=tk.W)

        # Выбор шагов
        select_frame = ttk.LabelFrame(left_frame, text="Выбор шагов", padding=5)
        select_frame.pack(fill=tk.X, pady=5)

        # dt
        ttk.Label(select_frame, text="Шаг по времени dt (с):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.dt_vars = []
        dt_frame = ttk.Frame(select_frame)
        dt_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W)
        for i, dt in enumerate(self.all_dt):
            var = tk.BooleanVar(value=(dt in self.selected_dt))
            self.dt_vars.append(var)
            cb = ttk.Checkbutton(dt_frame, text=str(dt), variable=var,
                                 command=self.update_selection)
            cb.grid(row=0, column=i, padx=5)

        # h
        ttk.Label(select_frame, text="Шаг по пространству h (м):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.h_vars = []
        h_frame = ttk.Frame(select_frame)
        h_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W)
        for i, h in enumerate(self.all_h):
            var = tk.BooleanVar(value=(h in self.selected_h))
            self.h_vars.append(var)
            cb = ttk.Checkbutton(h_frame, text=str(h), variable=var,
                                 command=self.update_selection)
            cb.grid(row=0, column=i, padx=5)

        # Кнопки выбора
        btn_frame = ttk.Frame(select_frame)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(btn_frame, text="Выбрать все", command=self.select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Сбросить (крупные)", command=self.select_coarse).pack(side=tk.LEFT, padx=2)

        # Оценка времени
        self.estimate_label = ttk.Label(select_frame, text="Оценка времени: —", foreground="blue")
        self.estimate_label.grid(row=5, column=0, columnspan=2, pady=5)

        # Кнопка запуска
        self.calc_button = ttk.Button(left_frame, text="Запустить расчёт", command=self.start_calculation)
        self.calc_button.pack(pady=10)

        # Прогресс
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)

        # Правая панель (таблицы и график)
        right_frame = ttk.Frame(self.root, padding=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Блокнот для таблиц
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Вкладка с температурой
        self.temp_frame = ttk.Frame(notebook)
        notebook.add(self.temp_frame, text="Температура в центре, °C")
        self.create_temp_table()

        # Вкладка с временем
        self.time_frame = ttk.Frame(notebook)
        notebook.add(self.time_frame, text="Время расчёта, с")
        self.create_time_table()

        # График
        graph_frame = ttk.LabelFrame(right_frame, text="Профиль температуры (для выбранной комбинации)", padding=5)
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax.set_xlim(0, L)
        self.ax.set_ylim(-10, T_right + 20)
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("Температура, °C")
        self.ax.grid(True)
        self.ax.set_title("Профиль не построен")
        self.canvas.draw()

    def create_temp_table(self):
        """Создаёт таблицу для температуры, но пока пустую (заполнится позже)."""
        self.temp_tree = ttk.Treeview(self.temp_frame, show="headings")
        self.temp_tree.pack(fill=tk.BOTH, expand=True)
        self.update_temp_table()

    def update_temp_table(self):
        """Обновляет заголовки и строки таблицы температуры в соответствии с выбранными шагами."""
        # Очищаем
        self.temp_tree.delete(*self.temp_tree.get_children())
        self.temp_tree["columns"] = ["dt"] + [str(h) for h in self.selected_h]

        # Заголовки
        self.temp_tree.heading("dt", text="dt / h")
        self.temp_tree.column("dt", width=80)
        for h in self.selected_h:
            col = str(h)
            self.temp_tree.heading(col, text=f"{h}")
            self.temp_tree.column(col, width=80, anchor=tk.CENTER)

        # Строки
        for dt in self.selected_dt:
            values = [str(dt)] + ["" for _ in self.selected_h]
            self.temp_tree.insert("", tk.END, values=values, iid=f"temp_{dt}")

    def create_time_table(self):
        self.time_tree = ttk.Treeview(self.time_frame, show="headings")
        self.time_tree.pack(fill=tk.BOTH, expand=True)
        self.update_time_table()

    def update_time_table(self):
        self.time_tree.delete(*self.time_tree.get_children())
        self.time_tree["columns"] = ["dt"] + [str(h) for h in self.selected_h]

        self.time_tree.heading("dt", text="dt / h")
        self.time_tree.column("dt", width=80)
        for h in self.selected_h:
            col = str(h)
            self.time_tree.heading(col, text=f"{h}")
            self.time_tree.column(col, width=80, anchor=tk.CENTER)

        for dt in self.selected_dt:
            values = [str(dt)] + ["" for _ in self.selected_h]
            self.time_tree.insert("", tk.END, values=values, iid=f"time_{dt}")

    def update_selection(self):
        """Обновляет списки выбранных шагов на основе чекбоксов."""
        self.selected_dt = [dt for dt, var in zip(self.all_dt, self.dt_vars) if var.get()]
        self.selected_h = [h for h, var in zip(self.all_h, self.h_vars) if var.get()]
        self.update_temp_table()
        self.update_time_table()
        self.update_estimate()

    def select_all(self):
        for var in self.dt_vars:
            var.set(True)
        for var in self.h_vars:
            var.set(True)
        self.update_selection()

    def select_coarse(self):
        # Крупные шаги: dt 0.1, 0.01; h 0.1, 0.01
        for dt, var in zip(self.all_dt, self.dt_vars):
            var.set(dt in [0.1, 0.01])
        for h, var in zip(self.all_h, self.h_vars):
            var.set(h in [0.1, 0.01])
        self.update_selection()

    def update_estimate(self):
        """Оценивает примерное время выполнения."""
        if not self.selected_dt or not self.selected_h:
            self.estimate_label.config(text="Оценка времени: —")
            return

        # Приблизительная формула: время ~ сумма по комбинациям (Nx * Nt)
        total_ops = 0
        for dt in self.selected_dt:
            Nt = int(self.total_time / dt)
            for h in self.selected_h:
                Nx = int(L / h)
                total_ops += Nx * Nt

        # Примерно 1e6 операций в секунду (грубо)
        est_sec = total_ops / 1e6
        if est_sec < 60:
            est_str = f"~{est_sec:.1f} с"
        elif est_sec < 3600:
            est_str = f"~{est_sec/60:.1f} мин"
        else:
            est_str = f"~{est_sec/3600:.1f} ч"

        self.estimate_label.config(text=f"Оценка времени: {est_str}")

    def start_calculation(self):
        if not self.selected_dt or not self.selected_h:
            return

        self.calc_button.config(state=tk.DISABLED)
        self.progress.start()

        # Запуск в потоке
        thread = threading.Thread(target=self.run_calculation)
        thread.daemon = True
        thread.start()

    def run_calculation(self):
        try:
            for dt in self.selected_dt:
                for h in self.selected_h:
                    T_center, cpu_time, profile = self.compute_pair(dt, h)
                    self.T_center_dict[(dt, h)] = T_center
                    self.cpu_time_dict[(dt, h)] = cpu_time
                    self.last_profile = profile
                    self.last_x = np.linspace(0, L, len(profile))

            # Обновление таблиц и графика в главном потоке
            self.root.after(0, self.update_tables)
            self.root.after(0, self.update_graph)

        except Exception as e:
            print(f"Ошибка: {e}")  # можно заменить на лог
        finally:
            self.root.after(0, self.finish_calculation)

    def compute_pair(self, dt, h):
        Nx = int(L / h)
        n_nodes = Nx + 1
        center_index = Nx // 2

        T = np.zeros(n_nodes)
        T[0] = T_left
        T[-1] = T_right

        Nt = int(self.total_time / dt)

        start = time.time()
        for _ in range(Nt):
            T = solve_step(T, h, dt, rho, c, lambda_, T_left, T_right)
        elapsed = time.time() - start

        return T[center_index], elapsed, T

    def update_tables(self):
        # Обновление таблицы температур
        for dt in self.selected_dt:
            row_values = [str(dt)]
            for h in self.selected_h:
                val = self.T_center_dict.get((dt, h), "")
                if val != "":
                    row_values.append(f"{val:.2f}")
                else:
                    row_values.append("")
            self.temp_tree.item(f"temp_{dt}", values=row_values)

        # Обновление таблицы времени
        for dt in self.selected_dt:
            row_values = [str(dt)]
            for h in self.selected_h:
                val = self.cpu_time_dict.get((dt, h), "")
                if val != "":
                    row_values.append(f"{val:.4f}")
                else:
                    row_values.append("")
            self.time_tree.item(f"time_{dt}", values=row_values)

    def update_graph(self):
        if self.last_profile is not None:
            self.ax.clear()
            self.ax.plot(self.last_x, self.last_profile, 'r-', lw=2)
            self.ax.set_xlim(0, L)
            self.ax.set_ylim(-10, T_right + 20)
            self.ax.set_xlabel("x, м")
            self.ax.set_ylabel("Температура, °C")
            self.ax.grid(True)
            self.ax.set_title(f"Профиль (последняя комбинация)")
            self.canvas.draw()

    def finish_calculation(self):
        self.progress.stop()
        self.calc_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = HeatConductionApp(root)
    root.mainloop()
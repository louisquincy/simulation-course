import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Simulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Моделирование полёта тела в атмосфере")

        self.rho = 1.225
        self.g = 9.81
        self.default_v0 = 331.46
        self.default_angle = 25.0
        self.default_m = 1850.0
        self.default_S = 2.4
        self.default_C = 0.52
        self.default_dt_new = 0.05
        self.dt_auto = [1.0, 0.1, 0.01, 0.001, 0.0001]

        self.trajectories = []
        self.results = []

        self.n_frames = 100
        self.pause_ms = 20
        self.anim_queue = []

        self.setup_ui()
        self._setup_axes()
        self.canvas.draw()

    def simulate(self, v0, angle_deg, dt, m, S, C):
        angle_rad = np.radians(angle_deg)
        vx0 = v0 * np.cos(angle_rad)
        vy0 = v0 * np.sin(angle_rad)

        x, y = [0.0], [0.0]
        vx, vy = [vx0], [vy0]

        while y[-1] >= 0:
            v = np.sqrt(vx[-1]**2 + vy[-1]**2)
            F = 0.5 * self.rho * C * S * v
            if v != 0:
                Fx = -F * (vx[-1] / v)
                Fy = -F * (vy[-1] / v)
            else:
                Fx = Fy = 0.0
            ax_ = Fx / m
            ay_ = -self.g + Fy / m

            x_new = x[-1] + vx[-1] * dt
            y_new = y[-1] + vy[-1] * dt

            if y_new < 0:
                alpha = y[-1] / (y[-1] - y_new)
                x.append(x[-1] + vx[-1] * alpha * dt)
                y.append(0.0)
                vx.append(vx[-1] + ax_ * alpha * dt)
                vy.append(vy[-1] + ay_ * alpha * dt)
                break
            else:
                x.append(x_new)
                y.append(y_new)
                vx.append(vx[-1] + ax_ * dt)
                vy.append(vy[-1] + ay_ * dt)
        return x, y, vx, vy

    def setup_ui(self):
        left = ttk.Frame(self.root, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        param_frame = ttk.LabelFrame(left, text="Параметры модели", padding=10)
        param_frame.pack(fill=tk.X, pady=(0,10))

        self.entries = {}
        params = [
            ("Начальная скорость v0 (м/с):", "v0", self.default_v0),
            ("Угол бросания (град):", "angle", self.default_angle),
            ("Масса m (кг):", "m", self.default_m),
            ("Площадь S (м²):", "S", self.default_S),
            ("Коэфф. сопротивления C:", "C", self.default_C),
            ("Новый шаг dt (с):", "dt_new", self.default_dt_new),
        ]
        for i, (text, key, default) in enumerate(params):
            ttk.Label(param_frame, text=text).grid(row=i, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(param_frame, width=12)
            entry.grid(row=i, column=1, padx=5, pady=2)
            entry.insert(0, str(default))
            self.entries[key] = entry

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text=" Добавить траекторию (ввести dt)",
                   command=self.add_trajectory).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text=" Автоматический прогон (все dt)",
                   command=self.run_auto).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text=" Очистить график и таблицу",
                   command=self.clear_all).pack(fill=tk.X, pady=2)

        table_frame = ttk.LabelFrame(left, text="Таблица результатов", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(10,0))
        columns = ("dt", "range", "hmax", "vfinal")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=12)
        for col in columns:
            self.tree.heading(col, text=col.capitalize())
            self.tree.column(col, width=100 if col!="dt" else 80, anchor=tk.CENTER)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.figure, self.ax = plt.subplots(figsize=(9, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_axes(self):
        self.ax.set_xlabel("Расстояние, м")
        self.ax.set_ylabel("Высота, м")
        self.ax.set_title("Анимация траекторий с разным шагом dt")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(0, 9000)
        self.ax.set_ylim(0, 1100)

    def get_base_params(self):
        try:
            return {key: float(self.entries[key].get()) for key in ('v0','angle','m','S','C')}
        except ValueError:
            self.root.title("Ошибка: проверьте числовые параметры!")
            return None

    def get_dt_new(self):
        try:
            return float(self.entries['dt_new'].get())
        except ValueError:
            self.root.title("Ошибка: неверное значение dt!")
            return None

    def create_annotation(self, x_pos, y_pos, text):
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        dx = 15 if x_pos < xlim[0] + 0.7*(xlim[1]-xlim[0]) else -80
        dy = -25 if y_pos > ylim[0] + 0.8*(ylim[1]-ylim[0]) else 15
        return self.ax.annotate(text, xy=(x_pos,y_pos), xytext=(dx,dy),
                                textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                                arrowprops=dict(arrowstyle='->'), zorder=5)

    def _add_trajectory(self, dt):
        params = self.get_base_params()
        if params is None: return
        x, y, vx, vy = self.simulate(params['v0'], params['angle'], dt,
                                     params['m'], params['S'], params['C'])
        idx_max = np.argmax(y)
        hmax = y[idx_max]
        range_val = x[-1]
        vfinal = np.sqrt(vx[-1]**2 + vy[-1]**2)

        self.results.append({'dt': dt, 'range': range_val, 'hmax': hmax, 'vfinal': vfinal})
        self.update_table()

        color = plt.cm.viridis(len(self.trajectories) * 0.2 % 1.0)
        line, = self.ax.plot([], [], color=color, lw=2, label=f'dt = {dt}')
        point, = self.ax.plot([], [], 'ro', markersize=8)

        traj = {
            'dt': dt,
            'x': x, 'y': y,
            'idx_max': idx_max, 'hmax': hmax,
            'color': color,
            'line': line, 'point': point,
            'annot': None,
            'annotated': False,
            'frames': np.linspace(0, len(x)-1, self.n_frames, dtype=int),
            'current_frame': 0
        }
        self.trajectories.append(traj)
        self.anim_queue.append(len(self.trajectories)-1)
        if len(self.anim_queue) == 1:
            self.root.after(10, self.animate_next)

    def add_trajectory(self):
        dt = self.get_dt_new()
        if dt is not None:
            self._add_trajectory(dt)

    def run_auto(self):
        for dt in self.dt_auto:
            self._add_trajectory(dt)

    def animate_next(self):
        if not self.anim_queue:
            return
        traj_idx = self.anim_queue[0]
        traj = self.trajectories[traj_idx]
        i = traj['frames'][traj['current_frame']] if traj['current_frame'] < len(traj['frames']) else -1

        if i != -1:
            traj['line'].set_data(traj['x'][:i+1], traj['y'][:i+1])
            if i >= traj['idx_max'] and not traj['annotated']:
                traj['point'].set_data([traj['x'][traj['idx_max']]], [traj['y'][traj['idx_max']]])
                traj['annot'] = self.create_annotation(
                    traj['x'][traj['idx_max']], traj['y'][traj['idx_max']],
                    f'Hmax = {traj["hmax"]:.2f} м'
                )
                traj['annotated'] = True
            self.canvas.draw()
            self.canvas.flush_events()
            traj['current_frame'] += 1
            self.root.after(self.pause_ms, self.animate_next)
        else:
            self.anim_queue.pop(0)
            traj['current_frame'] = 0
            if self.anim_queue:
                self.root.after(500, self.animate_next)

    def update_table(self):
        self.tree.delete(*self.tree.get_children())
        for r in self.results:
            self.tree.insert('', tk.END, values=(
                f"{r['dt']:.4f}", f"{r['range']:.2f}",
                f"{r['hmax']:.2f}", f"{r['vfinal']:.2f}"
            ))

    def clear_all(self):
        self.anim_queue.clear()
        self.trajectories.clear()
        self.results.clear()
        self.ax.clear()
        self._setup_axes()
        self.canvas.draw()
        self.update_table()

if __name__ == "__main__":
    root = tk.Tk()
    app = Simulator(root)
    root.mainloop()
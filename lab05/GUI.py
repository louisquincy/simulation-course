import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)
from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve,
    QTimer, QPoint, QRect
)
from PyQt6.QtGui import (
    QPainter, QRadialGradient, QColor,
    QFont, QPen, QBrush
)

from rng import MultiKongGen
from logic import yes_no_once, gacha_pull_once, RARITY

# Цвета для разных редкостей (только визуальное оформление)
RARITY_COLORS = {
    "legendary": QColor(255, 185, 15),
    "epic":      QColor(168, 85, 247),
    "rare":      QColor(59, 130, 246),
    "common":    QColor(200, 200, 200),
    "default":   QColor(180, 180, 180),
}

# ----- Виджет "Магический шар" -----
class MagicBall(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 300)
        self._text = "Press\nthe button"
        self._text_color = RARITY_COLORS["default"]
        self.is_shaking = False
        self._base_pos = None

    def set_text(self, text: str, rarity: str = "default"):
        self._text = text
        self._text_color = RARITY_COLORS.get(rarity, RARITY_COLORS["default"])
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy, r = 150, 150, 140

        outer = QRadialGradient(cx - 30, cy - 40, r * 1.6)
        outer.setColorAt(0.0, QColor(80, 80, 90))
        outer.setColorAt(0.5, QColor(30, 30, 35))
        outer.setColorAt(1.0, QColor(10, 10, 12))
        p.setBrush(QBrush(outer))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        iw, ih = 130, 110
        inner = QRadialGradient(cx, cy - 10, 80)
        inner.setColorAt(0.0, QColor(50, 50, 60))
        inner.setColorAt(1.0, QColor(20, 20, 28))
        p.setBrush(QBrush(inner))
        p.drawEllipse(cx - iw // 2, cy - ih // 2, iw, ih)

        p.setPen(QPen(QColor(255, 255, 255, 35), 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(cx - r + 25, cy - r + 15,
                  r * 2 - 50, r * 2 - 30, 30 * 16, 120 * 16)

        p.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        p.setPen(self._text_color)
        rect = QRect(cx - iw // 2 + 10, cy - ih // 2 + 10,
                     iw - 20, ih - 20)
        p.drawText(rect,
                   Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                   self._text)
        p.end()

    def shake(self, callback):
        if self.is_shaking:
            return
        self.is_shaking = True

        if self._base_pos is None:
            self._base_pos = self.pos()

        anim = QPropertyAnimation(self, b"pos", self)
        anim.setDuration(300)
        anim.setLoopCount(3)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        b = self._base_pos
        anim.setKeyValueAt(0.00, b)
        anim.setKeyValueAt(0.25, b + QPoint(10, -5))
        anim.setKeyValueAt(0.50, b)
        anim.setKeyValueAt(0.75, b + QPoint(-10, 5))
        anim.setKeyValueAt(1.00, b)

        def on_finished():
            self.move(self._base_pos)
            QTimer.singleShot(200, lambda: self._reveal(callback))

        anim.finished.connect(on_finished)
        self._anim = anim
        anim.start()

    def _reveal(self, callback):
        self.is_shaking = False
        callback()

# ----- Главное окно приложения -----
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lab 5 — Gacha Simulator")
        self.setMinimumSize(420, 520)
        self.pull_count = 0

        # Единый экземпляр ГПСЧ для всего приложения
        self.rng = MultiKongGen()

        self.setStyleSheet("""
            QMainWindow { background: #1a1a2e; }
            QLabel { color: #e0e0e0; font-family: 'Segoe UI'; }
            QPushButton {
                background: #16213e; color: #e0e0e0;
                border: 1px solid #0f3460; border-radius: 8px;
                padding: 10px 18px;
                font: bold 13px 'Segoe UI';
            }
            QPushButton:hover   { background: #0f3460; }
            QPushButton:pressed { background: #533483; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ball = MagicBall()
        layout.addWidget(self.ball, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(20)

        btn_row = QHBoxLayout()
        btn_yn    = QPushButton("Yes / No")
        btn_gacha = QPushButton("Gacha Pull")
        btn_clear = QPushButton("Clear")

        btn_yn.clicked.connect(self.on_yes_no)
        btn_gacha.clicked.connect(self.on_gacha)
        btn_clear.clicked.connect(self.on_clear)

        for btn in (btn_yn, btn_gacha, btn_clear):
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        self.counter_lbl = QLabel("Pulls: 0")
        self.counter_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.counter_lbl.setStyleSheet("font-size: 14px; color: #888;")
        layout.addWidget(self.counter_lbl)

        hint = QLabel("Ask a question and pull!")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("font-size: 12px; color: #555; margin-top: 6px;")
        layout.addWidget(hint)

    def on_yes_no(self):
        def reveal():
            result = yes_no_once(self.rng)
            self.ball.set_text(result, "default")
        self.ball.shake(reveal)

    def on_gacha(self):
        def reveal():
            name = gacha_pull_once(self.rng)
            rarity = RARITY[name]
            self.pull_count += 1
            self.counter_lbl.setText(f"Pulls: {self.pull_count}")
            self.ball.set_text(name, rarity)
        self.ball.shake(reveal)

    def on_clear(self):
        self.ball.set_text("Press\nthe button", "default")
        self.pull_count = 0
        self.counter_lbl.setText("Pulls: 0")

# ----- Точка входа -----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
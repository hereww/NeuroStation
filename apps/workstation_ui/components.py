"""Reusable Qt presentation components. All stimuli here are deliberately static."""
from math import sin, exp

from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QToolButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QStyle, QSizePolicy,
)

from .gateway import FREQUENCIES


def label(text: str, role: str = "", wrap: bool = True) -> QLabel:
    widget = QLabel(text)
    widget.setWordWrap(wrap)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    if role:
        widget.setObjectName(role)
    widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return widget


def action(text: str, callback, primary: bool = False) -> QPushButton:
    widget = QPushButton(text)
    widget.setMinimumHeight(36)
    if primary:
        widget.setObjectName("primaryButton")
    widget.clicked.connect(callback)
    return widget


class Page(QWidget):
    def __init__(self, tr, title: str, subtitle: str = ""):
        super().__init__()
        self.tr = tr
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(24, 22, 24, 24)
        self.layout.setSpacing(16)
        self.title_label = label(title, "pageTitle")
        self.layout.addWidget(self.title_label)
        if subtitle:
            self.layout.addWidget(label(subtitle, "muted"))


class Section(QFrame):
    def __init__(self, title: str = ""):
        super().__init__()
        self.setObjectName("section")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(12)
        if title:
            self.layout.addWidget(label(title, "sectionTitle"))


class KeyValues(QWidget):
    def __init__(self, rows: list[tuple[str, str]]):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(24)
        layout.setVerticalSpacing(11)
        self.values: list[QLabel] = []
        for i, (key, value) in enumerate(rows):
            layout.addWidget(label(key, "muted"), i, 0)
            node = label(str(value))
            layout.addWidget(node, i, 1)
            self.values.append(node)
        layout.setColumnStretch(1, 1)


class AppTile(QToolButton):
    def __init__(self, title: str, subtitle: str, icon_type, callback=None):
        super().__init__()
        self.setObjectName("appTile")
        self.setMinimumSize(145, 144)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setIcon(self.style().standardIcon(icon_type))
        self.setIconSize(QSize(36, 36))
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.setText(f"{title}\n{subtitle}")
        self.setAccessibleName(f"{title}: {subtitle}")
        if callback:
            self.clicked.connect(callback)
        else:
            self.setEnabled(False)


class StaticTargets(Section):
    def __init__(self, tr):
        super().__init__(tr("ssvep.static"))
        self.setAccessibleName(tr("ssvep.static"))
        grid = QGridLayout()
        grid.setSpacing(20)
        for i, frequency in enumerate(FREQUENCIES):
            target = label(f"{frequency} Hz\n{tr('ssvep.target', number=i+1)}", "staticTarget")
            target.setAlignment(Qt.AlignmentFlag.AlignCenter)
            target.setMinimumSize(96, 90)
            grid.addWidget(target, i // 2, i % 2)
        self.layout.addLayout(grid)
        self.layout.addWidget(label(tr("ssvep.production"), "muted"))


class WaveformWidget(QWidget):
    """Animated illustrative traces for the in-memory acquisition test."""
    def __init__(self, accessible_name: str, badge_text: str = ""):
        super().__init__()
        self.setMinimumSize(280, 345)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName(accessible_name)
        self._phase = 0.0
        self._active = False
        self._badge_text = badge_text
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._advance)

    def _advance(self):
        self._phase = (self._phase + 0.035) % 1.0
        self.update()

    def set_active(self, active: bool):
        self._active = active
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark = self.palette().window().color().lightness() < 128
        grid = QColor("#34434e" if dark else "#e0e7ea")
        ink = QColor("#e0e9ef" if dark else "#263c48")
        accent = QColor("#69c8d3" if dark else "#087c88")
        if not self._active:
            accent.setAlpha(185)
        left, right = 58, self.width()-14
        top, bottom = 12, self.height()-32
        step = (bottom-top)/8
        painter.setPen(QPen(grid, 1))
        for tick in range(6):
            x = left+(right-left)*tick/5
            painter.drawLine(int(x), top, int(x), bottom)
            painter.setPen(ink)
            painter.drawText(int(x)-14, self.height()-10, f"{tick-5} s")
            painter.setPen(QPen(grid, 1))
        for channel, name in enumerate(("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2")):
            center = top+step*(channel+.5)
            painter.setPen(QPen(grid, 1))
            painter.drawLine(left, int(center), right, int(center))
            painter.setPen(ink)
            painter.drawText(9, int(center), name)
            path = QPainterPath()
            samples = max(240, right-left)
            for j in range(samples+1):
                u = j/samples
                shift = self._phase * 2.0 * 3.14159
                amplitude = (sin(u*147+channel*.81-shift)*.39+sin(u*323+channel*1.21-shift*1.4)*.19
                    +sin(u*67+channel*2.18-shift*.6)*.22+sin(u*829+channel*3.1-shift*2.2)*.11)*11
                amplitude += exp(-((u-(.22+channel*.055))/.024)**2)*(9 if channel < 2 else 3)
                x, y = left+(right-left)*u, center+amplitude
                path.lineTo(x, y) if j else path.moveTo(x, y)
            painter.setPen(QPen(accent, 1.2))
            painter.drawPath(path)
        scan_x = left + (right-left) * self._phase
        painter.setPen(QPen(QColor("#d9a441" if not dark else "#f2c66d"), 1.5))
        painter.drawLine(int(scan_x), top, int(scan_x), bottom)
        if self._badge_text:
            painter.setPen(QPen(ink, 1))
            painter.drawText(right - 150, top + 14, self._badge_text)

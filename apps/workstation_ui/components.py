"""Reusable Qt presentation components."""
from collections import deque
from math import cos, exp, isfinite, pi, sin
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QTimer, QRect, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
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


CHANNEL_NAMES = ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2")
CHANNEL_COLORS = (
    (8, 124, 136),
    (47, 118, 183),
    (107, 92, 165),
    (155, 91, 142),
    (189, 109, 74),
    (181, 144, 49),
    (92, 143, 84),
    (92, 125, 143),
)


class HeadElectrodeMap(QWidget):
    """Clickable schematic for assigning and reviewing scalp electrode sites."""

    position_clicked = Signal(str)
    channel_clicked = Signal(int)

    SITE_COORDS = {
        # The eight default sites follow OpenBCI GUI's
        # electrode_positions_default.txt coordinate ratios.
        "Fp1": (0.375, 0.084), "Fp2": (0.625, 0.084),
        "C3": (0.300, 0.500), "C4": (0.700, 0.500),
        "P7": (0.1575, 0.770), "P8": (0.8425, 0.770),
        "O1": (0.375, 0.916), "O2": (0.625, 0.916),
        "AF3": (0.335, 0.220), "AF4": (0.665, 0.220),
        "Fz": (0.500, 0.220), "Cz": (0.500, 0.500),
        "P3": (0.350, 0.720), "P4": (0.650, 0.720),
        "T7": (0.084, 0.500), "T8": (0.916, 0.500),
        "Oz": (0.500, 0.950),
    }

    def __init__(self, channel_names: tuple[str, ...] = CHANNEL_NAMES):
        super().__init__()
        self.setMinimumSize(330, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Head electrode map")
        self._channel_names = tuple(channel_names)
        self._assignments = ["" for _ in self._channel_names]
        self._selected_channel = 0
        self._hint_text = "Click a site to assign the selected channel"
        asset_path = Path(__file__).resolve().parents[2] / "assets" / "calibration_head_plot_v1.png"
        self._head_plot = QPixmap(str(asset_path)) if asset_path.is_file() else QPixmap()

    def set_hint_text(self, text: str) -> None:
        self._hint_text = str(text or "")
        self.update()

    def set_assignments(self, assignments: tuple[str, ...] | list[str]) -> None:
        values = [str(value or "").strip() for value in assignments]
        self._assignments = (values + [""] * len(self._channel_names))[:len(self._channel_names)]
        self.update()

    def set_selected_channel(self, channel_index: int) -> None:
        if 0 <= int(channel_index) < len(self._channel_names):
            self._selected_channel = int(channel_index)
            self.update()

    def _site_points(self) -> dict[str, tuple[float, float]]:
        plot = self._plot_rect()
        return {
            name: (plot.left() + x * plot.width(), plot.top() + y * plot.height())
            for name, (x, y) in self.SITE_COORDS.items()
        }

    def _plot_rect(self) -> QRect:
        bounds = self.rect().adjusted(12, 8, -12, -24)
        side = min(bounds.width(), bounds.height())
        return QRect(
            bounds.left() + (bounds.width() - side) // 2,
            bounds.top() + (bounds.height() - side) // 2,
            side,
            side,
        )

    def _channel_site(self, channel_index: int) -> str:
        assignment = self._assignments[channel_index].strip()
        if assignment.casefold() in {name.casefold() for name in self.SITE_COORDS}:
            return next(
                name for name in self.SITE_COORDS
                if name.casefold() == assignment.casefold()
            )
        if channel_index < len(self._channel_names):
            default = self._channel_names[channel_index]
            if default in self.SITE_COORDS:
                return default
        return "Cz"

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        position = event.position()
        nearest = min(
            self._site_points().items(),
            key=lambda item: (item[1][0] - position.x()) ** 2 + (item[1][1] - position.y()) ** 2,
        )
        distance = (nearest[1][0] - position.x()) ** 2 + (nearest[1][1] - position.y()) ** 2
        if distance <= 28 ** 2:
            selected = next(
                (index for index, value in enumerate(self._assignments) if value.casefold() == nearest[0].casefold()),
                None,
            )
            if selected is not None:
                self.channel_clicked.emit(selected)
            self.position_clicked.emit(nearest[0])
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark = self.palette().window().color().lightness() < 128
        muted = QColor("#a6b6c1" if dark else "#677781")
        ink = QColor("#e0e9ef" if dark else "#263c48")
        surface = QColor("#1b222a" if dark else "#f5f7f8")
        accent = QColor("#69c8d3" if dark else "#087c88")
        plot = self._plot_rect()
        painter.setBrush(surface)
        if not self._head_plot.isNull():
            scaled = self._head_plot.scaled(
                plot.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            target = QRect(
                plot.left() + (plot.width() - scaled.width()) // 2,
                plot.top() + (plot.height() - scaled.height()) // 2,
                scaled.width(),
                scaled.height(),
            )
            painter.drawPixmap(target, scaled)
        else:
            painter.setPen(QPen(muted, 2))
            painter.drawEllipse(plot)
        painter.setPen(QPen(muted, 1))
        painter.drawText(8, self.height() - 7, self._hint_text)

        points = self._site_points()
        active_sites = {self._channel_site(index) for index in range(len(self._channel_names))}
        # Keep extra anatomical sites available as click targets without
        # changing the OpenBCI-like eight-channel visual at rest.
        for name, (x, y) in points.items():
            if name in active_sites:
                continue
            painter.setPen(QPen(muted, 1))
            painter.setBrush(surface)
            painter.drawEllipse(int(x - 4), int(y - 4), 8, 8)

        for channel_index in range(len(self._channel_names)):
            name = self._channel_site(channel_index)
            x, y = points[name]
            selected = channel_index == self._selected_channel
            color = QColor(*CHANNEL_COLORS[channel_index % len(CHANNEL_COLORS)])
            radius = max(11, int(plot.width() * 0.028))
            painter.setPen(QPen(accent if selected else color, 3 if selected else 1))
            painter.setBrush(color)
            painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawText(
                QRect(int(x - radius), int(y - radius), radius * 2, radius * 2),
                Qt.AlignmentFlag.AlignCenter,
                str(channel_index + 1),
            )
            painter.setPen(QPen(ink, 1))
            painter.drawText(int(x - 14), int(y + radius + 16), name)


class CalibrationSignalWidget(QWidget):
    """Single-channel rolling trace used while confirming a physical electrode."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(360, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._values = deque(maxlen=1250)
        self._channel_label = "CH1"
        self._status = ""
        self._active = False
        self._empty_text = "Waiting for calibration samples"

    def set_empty_text(self, text: str) -> None:
        self._empty_text = str(text or "")
        self.update()

    def set_channel(self, channel_label: str) -> None:
        self._channel_label = str(channel_label or "CH1")
        self.update()

    def set_status(self, status: str) -> None:
        self._status = str(status or "")
        self.update()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if active:
            self._values.clear()
        self.update()

    def clear(self) -> None:
        self._values.clear()
        self.update()

    def append_samples(self, values: list[float] | tuple[float, ...]) -> None:
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if isfinite(number):
                self._values.append(number)
        self.update()

    @property
    def sample_count(self) -> int:
        return len(self._values)

    @property
    def variation(self) -> float:
        if not self._values:
            return 0.0
        return max(self._values) - min(self._values)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark = self.palette().window().color().lightness() < 128
        grid = QColor("#34434e" if dark else "#dfe6e9")
        ink = QColor("#e0e9ef" if dark else "#263c48")
        muted = QColor("#a6b6c1" if dark else "#677781")
        accent = QColor("#69c8d3" if dark else "#087c88")
        left, right = 48, max(55, self.width() - 14)
        top, bottom = 34, max(48, self.height() - 24)
        center = (top + bottom) / 2
        painter.setPen(ink)
        painter.drawText(10, 19, self._channel_label)
        painter.setPen(muted)
        painter.drawText(56, 19, self._status)
        painter.setPen(QPen(grid, 1))
        painter.drawRect(int(left), int(top), int(right - left), int(bottom - top))
        painter.drawLine(int(left), int(center), int(right), int(center))
        for tick in range(1, 5):
            x = left + (right - left) * tick / 5
            painter.drawLine(int(x), int(top), int(x), int(bottom))
        values = list(self._values)
        if values:
            scale = max(25.0, max(abs(value) for value in values) * 1.15)
            path = QPainterPath()
            for index, value in enumerate(values):
                x = left + (right - left) * index / max(1, len(values) - 1)
                y = center - max(-scale, min(scale, value)) / scale * (bottom - top) * 0.46
                if index == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(accent, 1.5 if self._active else 1.0))
            painter.drawPath(path)
        else:
            painter.setPen(muted)
            painter.drawText(int(left + 12), int(center + 4), self._empty_text)
        painter.setPen(muted)
        painter.drawText(int(left), self.height() - 7, "-5 s")
        painter.drawText(int(right - 12), self.height() - 7, "0 s")


class _Biquad:
    """Small stateful second-order section for the display-only filter."""

    def __init__(self, coefficients: tuple[float, float, float, float, float]):
        self.b0, self.b1, self.b2, self.a1, self.a2 = coefficients
        self.x1 = self.x2 = 0.0
        self.y1 = self.y2 = 0.0

    def reset(self) -> None:
        self.x1 = self.x2 = 0.0
        self.y1 = self.y2 = 0.0

    def process(self, value: float) -> float:
        output = (
            self.b0 * value
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2
        )
        self.x2, self.x1 = self.x1, value
        self.y2, self.y1 = self.y1, output
        return output


def _biquad_coefficients(
    kind: str,
    frequency_hz: float,
    sample_rate_hz: float,
    quality: float,
) -> tuple[float, float, float, float, float]:
    """Return normalized RBJ coefficients for one second-order section."""

    omega = 2.0 * pi * frequency_hz / sample_rate_hz
    sine = sin(omega)
    cosine = cos(omega)
    alpha = sine / (2.0 * quality)
    if kind == "lowpass":
        b0, b1, b2 = (1.0 - cosine) / 2.0, 1.0 - cosine, (1.0 - cosine) / 2.0
    elif kind == "highpass":
        b0, b1, b2 = (1.0 + cosine) / 2.0, -(1.0 + cosine), (1.0 + cosine) / 2.0
    elif kind == "notch":
        b0, b1, b2 = 1.0, -2.0 * cosine, 1.0
    else:
        raise ValueError(f"unsupported biquad kind: {kind}")
    a0, a1, a2 = 1.0 + alpha, -2.0 * cosine, 1.0 - alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


class _OpenBCIDisplayFilter:
    """Causal 4th-order 5-50 Hz Butterworth path plus 50/60 Hz notches."""

    _BUTTERWORTH_Q = (0.5411961001461971, 1.3065629648763766)

    def __init__(self, sample_rate_hz: int):
        self.sample_rate_hz = sample_rate_hz
        self.sections = [
            _Biquad(_biquad_coefficients("highpass", 5.0, sample_rate_hz, quality))
            for quality in self._BUTTERWORTH_Q
        ]
        self.sections.extend(
            _Biquad(_biquad_coefficients("lowpass", 50.0, sample_rate_hz, quality))
            for quality in self._BUTTERWORTH_Q
        )
        self.sections.append(
            _Biquad(_biquad_coefficients("notch", 50.0, sample_rate_hz, 25.0))
        )
        self.sections.append(
            _Biquad(_biquad_coefficients("notch", 60.0, sample_rate_hz, 30.0))
        )

    def reset(self) -> None:
        for section in self.sections:
            section.reset()

    def process(self, values: list[float]) -> list[float]:
        output: list[float] = []
        for value in values:
            current = value if isfinite(value) else 0.0
            for section in self.sections:
                current = section.process(current)
            output.append(current)
        return output


class WaveformDisplayModel:
    """OpenBCI-style rolling raw/display buffers for the in-memory test signal."""

    CHANNEL_NAMES = CHANNEL_NAMES
    CHANNEL_COUNT = len(CHANNEL_NAMES)
    SAMPLE_RATE_HZ = 250
    UPDATE_MILLIS = 40
    SAMPLES_PER_UPDATE = SAMPLE_RATE_HZ * UPDATE_MILLIS // 1000
    BUFFER_SECONDS = 22
    DISPLAY_SECONDS = 5
    RAW_CAPACITY = SAMPLE_RATE_HZ * BUFFER_SECONDS
    DISPLAY_SAMPLES = SAMPLE_RATE_HZ * DISPLAY_SECONDS
    Y_LIMIT_UV = 200

    def __init__(self, channel_names: tuple[str, ...] | list[str] | None = None, sample_rate_hz: int = SAMPLE_RATE_HZ):
        names = tuple(channel_names or CHANNEL_NAMES)
        if not names:
            raise ValueError("waveform channel names")
        if sample_rate_hz <= 0:
            raise ValueError("waveform sample rate")
        self.CHANNEL_NAMES = names
        self.CHANNEL_COUNT = len(names)
        self.SAMPLE_RATE_HZ = int(sample_rate_hz)
        self.SAMPLES_PER_UPDATE = max(1, round(self.SAMPLE_RATE_HZ * self.UPDATE_MILLIS / 1000))
        self.RAW_CAPACITY = self.SAMPLE_RATE_HZ * self.BUFFER_SECONDS
        self.DISPLAY_SAMPLES = self.SAMPLE_RATE_HZ * self.DISPLAY_SECONDS
        self.raw_buffers: list[deque[float]] = []
        self.filtered_buffers: list[deque[float]] = []
        self._filters: list[_OpenBCIDisplayFilter] = []
        self.sample_index = 0
        self.invalid_sample_count = 0
        self.reset()

    def reset(self):
        self.raw_buffers = [deque(maxlen=self.RAW_CAPACITY) for _ in range(self.CHANNEL_COUNT)]
        self.filtered_buffers = [deque(maxlen=self.DISPLAY_SAMPLES) for _ in range(self.CHANNEL_COUNT)]
        self._filters = [
            _OpenBCIDisplayFilter(self.SAMPLE_RATE_HZ)
            for _ in range(self.CHANNEL_COUNT)
        ]
        self.sample_index = 0
        self.invalid_sample_count = 0

    def configure(self, channel_names: tuple[str, ...] | list[str], sample_rate_hz: int) -> None:
        names = tuple(channel_names)
        if not names:
            raise ValueError("waveform channel names")
        if sample_rate_hz <= 0:
            raise ValueError("waveform sample rate")
        if names == self.CHANNEL_NAMES and int(sample_rate_hz) == self.SAMPLE_RATE_HZ:
            return
        self.__init__(names, int(sample_rate_hz))

    def append(self, samples: list[list[float]]):
        if len(samples) != self.CHANNEL_COUNT:
            raise ValueError("waveform channel count")
        sample_count = len(samples[0]) if samples else 0
        if any(len(channel) != sample_count for channel in samples):
            raise ValueError("waveform sample lengths")
        if sample_count == 0:
            return

        for channel_index, values in enumerate(samples):
            converted = []
            for value in values:
                number = float(value)
                if not isfinite(number):
                    self.invalid_sample_count += 1
                converted.append(number)
            self.raw_buffers[channel_index].extend(converted)
            self.filtered_buffers[channel_index].extend(
                self._filters[channel_index].process(converted)
            )
        self.sample_index += sample_count

    def visible_samples(self, channel_index: int) -> tuple[float, ...]:
        if not 0 <= channel_index < self.CHANNEL_COUNT:
            raise IndexError("waveform channel index")
        values = list(self.filtered_buffers[channel_index])
        if len(values) < self.DISPLAY_SAMPLES:
            values = [0.0] * (self.DISPLAY_SAMPLES - len(values)) + values
        return tuple(values)

    @property
    def visible_sample_count(self) -> int:
        return len(self.filtered_buffers[0]) if self.filtered_buffers else 0

    @property
    def visible_duration_s(self) -> float:
        return self.visible_sample_count / self.SAMPLE_RATE_HZ


class WaveformWidget(QWidget):
    """Stacked EEG traces fed by a sample provider or an explicit test source."""

    def __init__(
        self,
        accessible_name: str,
        badge_text: str = "",
        *,
        empty_text: str = "No valid samples",
        channel_names: tuple[str, ...] | list[str] | None = None,
        sample_rate_hz: int = WaveformDisplayModel.SAMPLE_RATE_HZ,
        test_signal: bool = False,
    ):
        super().__init__()
        self.setMinimumSize(280, 390)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName(accessible_name)
        self._active = False
        self._badge_text = badge_text
        self._empty_text = empty_text
        self._test_signal_enabled = bool(test_signal)
        self._sample_provider = None
        self._model = WaveformDisplayModel(channel_names, sample_rate_hz)
        self._timer = QTimer(self)
        self._timer.setInterval(self._model.UPDATE_MILLIS)
        self._timer.timeout.connect(self._advance)

    @property
    def model(self) -> WaveformDisplayModel:
        return self._model

    def configure(self, channel_names: tuple[str, ...] | list[str], sample_rate_hz: int) -> None:
        self._model.configure(channel_names, sample_rate_hz)
        self.update()

    def set_sample_provider(self, provider) -> None:
        self._sample_provider = provider

    def set_test_signal(self, enabled: bool) -> None:
        self._test_signal_enabled = bool(enabled)

    def _generate_test_samples(self, count: int) -> list[list[float]]:
        start = self._model.sample_index
        samples = [[] for _ in self._model.CHANNEL_NAMES]
        for offset in range(count):
            time_s = (start + offset) / self._model.SAMPLE_RATE_HZ
            for channel in range(self._model.CHANNEL_COUNT):
                signal = (
                    16.0 * sin(2.0 * pi * (9.0 + channel * 0.15) * time_s + channel * 0.47)
                    + 7.0 * sin(2.0 * pi * (21.0 + channel * 0.2) * time_s + channel * 0.91)
                    + 3.0 * sin(2.0 * pi * 42.0 * time_s + channel * 0.23)
                )
                burst = 6.0 * exp(-((time_s % 4.0 - (0.6 + channel * 0.08)) / 0.035) ** 2)
                samples[channel].append(signal + burst)
        return samples

    def _advance(self):
        if self._active:
            samples = None
            if self._sample_provider is not None:
                samples = self._sample_provider(self._model.SAMPLES_PER_UPDATE)
            elif self._test_signal_enabled:
                samples = self._generate_test_samples(self._model.SAMPLES_PER_UPDATE)
            if samples:
                self._model.append(samples)
        self.update()

    def push_samples(self, samples: list[list[float]]) -> None:
        if self._active:
            self._model.append(samples)
        self.update()

    def set_active(self, active: bool):
        active = bool(active)
        if active != self._active:
            self._active = active
            self._model.reset()
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _colors(self, dark: bool) -> tuple[QColor, ...]:
        colors = []
        for red, green, blue in CHANNEL_COLORS:
            if dark:
                colors.append(QColor(min(255, red + 50), min(255, green + 50), min(255, blue + 50)))
            else:
                colors.append(QColor(red, green, blue))
        return tuple(colors)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        dark = self.palette().window().color().lightness() < 128
        grid = QColor("#34434e" if dark else "#dfe6e9")
        ink = QColor("#e0e9ef" if dark else "#263c48")
        muted = QColor("#a6b6c1" if dark else "#677781")
        colors = self._colors(dark)
        left, right = 58, max(60, self.width() - 14)
        top, bottom = 8, max(10, self.height() - 30)
        gap = 2
        names = self._model.CHANNEL_NAMES
        bar_height = max(20, (bottom - top - gap * (len(names) - 1)) / len(names))

        for channel, name in enumerate(names):
            bar_top = top + channel * (bar_height + gap)
            bar_bottom = bar_top + bar_height
            center = (bar_top + bar_bottom) / 2.0
            painter.setPen(QPen(grid, 1))
            painter.drawRect(int(left), int(bar_top), int(right - left), int(bar_height))
            painter.drawLine(int(left), int(center), int(right), int(center))
            for tick in range(1, 5):
                x = left + (right - left) * tick / 5.0
                painter.drawLine(int(x), int(bar_top), int(x), int(bar_bottom))

            painter.setPen(ink)
            painter.drawText(7, int(center + 4), name)
            painter.setPen(QPen(muted, 1))
            painter.drawText(7, int(bar_top + 11), f"+{self._model.Y_LIMIT_UV} uV")
            painter.drawText(7, int(bar_bottom - 3), f"-{self._model.Y_LIMIT_UV} uV")

            values = self._model.visible_samples(channel)
            path = QPainterPath()
            scale = (bar_height * 0.46) / self._model.Y_LIMIT_UV
            for index, value in enumerate(values):
                normalized = max(-self._model.Y_LIMIT_UV, min(self._model.Y_LIMIT_UV, value))
                x = left + (right - left) * index / max(1, len(values) - 1)
                y = center - normalized * scale
                if index == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            pen = QPen(colors[channel], 1.1)
            if not self._active:
                pen.setColor(QColor(colors[channel].red(), colors[channel].green(), colors[channel].blue(), 100))
            painter.setPen(pen)
            painter.drawPath(path)

        painter.setPen(QPen(muted, 1))
        for tick in range(6):
            x = left + (right - left) * tick / 5.0
            painter.drawText(int(x - 12), self.height() - 8, f"-{5 - tick} s" if tick < 5 else "0 s")
        if self._badge_text:
            painter.setPen(ink)
            painter.drawText(int(right - 130), 14, self._badge_text)

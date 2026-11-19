"""Read-only frequency view of the selected channel's raw Welch spectrum."""
from bisect import bisect_left, bisect_right
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QProgressBar, QSizePolicy, QToolTip, QVBoxLayout, QWidget)


# Display bands only. The upper gamma limit keeps mains frequencies outside the denominator.
BANDS = (
    ('δ Delta', 0.5, 4, '#6db8ff'), ('θ Theta', 4, 8, '#43cdce'),
    ('α Alpha', 8, 13, '#7fd695'), ('β Beta', 13, 30, '#eeaf68'),
    ('γ Gamma', 30, 45, '#be9df5'),
)


def band_shares(frequencies, power):
    """Integrate linear PSD on the measured Hz axis, interpolating at band edges."""
    if not frequencies or frequencies[0] > BANDS[0][1] or frequencies[-1] < BANDS[-1][2]:
        return None
    areas = []
    for _, low, high, _ in BANDS:
        area = 0.
        for i in range(max(1, bisect_left(frequencies, low)), bisect_left(frequencies, high) + 1):
            x0, x1 = frequencies[i - 1:i + 1]
            left, right = max(x0, low), min(x1, high)
            if right <= left:
                continue
            y0, y1 = power[i - 1:i + 1]
            if y0 is None or y1 is None:
                return None
            midpoint = ((left - x0) + (right - left) / 2) / (x1 - x0)
            area += (right - left) * (y0 * (1 - midpoint) + y1 * midpoint)
        areas.append(area)
    total = sum(areas)
    if total <= 0 or not math.isfinite(total):
        return None
    return [value / total for value in areas]


def visible_peaks(frequencies, power, limit_hz):
    """Strongest distinct local maxima in the viewport, within 30 dB of its peak."""
    peaks = [i for i in range(1, len(power) - 1) if 0.5 <= frequencies[i] <= limit_hz
             and None not in power[i - 1:i + 2] and power[i] > power[i - 1]
             and power[i] > power[i + 1]]
    peaks.sort(key=power.__getitem__, reverse=True)
    return [i for i in peaks[:6] if power[i] >= power[peaks[0]] * .001]


class SpectrumChart(QWidget):
    def __init__(self, tr):
        super().__init__()
        self.tr = tr
        self.frequency, self.power = [], []
        self.full_range = False
        self.peaks = []
        self.setMinimumHeight(155)
        self.setMouseTracking(True)
        self.setAccessibleName(tr('quality.spectrum.raw'))

    def hz_limit(self):
        end = self.frequency[-1] if self.frequency else 0
        return end if self.full_range else min(65, end)

    def axes(self):
        fm = self.fontMetrics()
        left = fm.horizontalAdvance('−200') + 12
        return QRectF(left, fm.height() + 7, max(1, self.width() - left - 25),
                      max(1, self.height() - 2 * fm.height() - 16))

    def mouseMoveEvent(self, event):
        axes = self.axes()
        if not self.frequency or not axes.contains(event.position()):
            QToolTip.hideText()
            return
        hz = (event.position().x() - axes.left()) / axes.width() * self.hz_limit()
        i = min(bisect_left(self.frequency, hz), len(self.frequency) - 1)
        if i and hz - self.frequency[i - 1] < self.frequency[i] - hz:
            i -= 1
        value = self.power[i]
        density = '—' if value is None else f'{value:.3g} µV²/Hz'
        QToolTip.showText(event.globalPosition().toPoint(), f'{self.frequency[i]:g} Hz · {density}', self)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor('#111b29'))
        p.setPen(QColor('#d6e3ef'))
        end = self.hz_limit()
        if end <= 0 or not any(v is not None for v in self.power):
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.tr('quality.plot.wait_psd'))
            return
        axes = self.axes()
        x = lambda hz: axes.left() + hz / end * axes.width()
        fm = p.fontMetrics()
        stop = bisect_right(self.frequency, end)
        maximum = max((v for v in self.power[:stop] if v is not None), default=0.)
        if maximum <= 0:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.tr('quality.spectrum.zero'))
            return
        top_db = 10 * math.ceil(math.log10(maximum))
        bottom_db = top_db - 60
        y = lambda db: axes.bottom() - max(0, db - bottom_db) / 60 * axes.height()

        for _, lo, hi, color in BANDS:
            if lo < end:
                tint = QColor(color)
                tint.setAlpha(35)
                p.fillRect(QRectF(x(lo), axes.top(), x(min(end, hi)) - x(lo), axes.height()), tint)
        for db in range(bottom_db, top_db + 1, 20):
            p.setPen(QColor('#344457'))
            p.drawLine(QPointF(axes.left(), y(db)), QPointF(axes.right(), y(db)))
            p.setPen(QColor('#c7d5e2'))
            p.drawText(QRectF(0, y(db) - fm.height()/2, axes.left()-6, fm.height()),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(db))
        step_guess = end / max(2, int(axes.width() / 56))
        magnitude = 10 ** math.floor(math.log10(step_guess)) if step_guess else 1
        step = next(s for s in (1, 2, 5, 10) if s * magnitude >= step_guess) * magnitude
        ticks = [i * step for i in range(int(end / step) + 1) if (end - i * step) / end * axes.width() > 42]
        for hz in [*ticks, end]:
            p.setPen(QColor('#334155'))
            p.drawLine(QPointF(x(hz), axes.top()), QPointF(x(hz), axes.bottom()))
            p.setPen(QColor('#c7d5e2'))
            text = f'{hz:g} Hz' if hz == end else f'{hz:g}'
            p.drawText(QPointF(x(hz) - fm.horizontalAdvance(text) / 2, self.height()-5), text)

        for hz in (50, 60):
            if hz <= end:
                p.setPen(QPen(QColor('#eeb679'), 1, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(x(hz), axes.top()), QPointF(x(hz), axes.bottom()))
                text = f'{hz}'
                p.drawText(QPointF(x(hz) - fm.horizontalAdvance(text) / 2, fm.ascent() + 2), text)

        path = QPainterPath()
        active = False
        for hz, value in zip(self.frequency[:stop], self.power[:stop]):
            if value is None:
                active = False
                continue
            point = QPointF(x(hz), y(10 * math.log10(value) if value else bottom_db))
            if active:
                path.lineTo(point)
            else:
                path.moveTo(point)
                active = True
        p.setPen(QPen(QColor('#e4f3ff'), 1.6))
        p.drawPath(path)
        used = []
        for i in self.peaks:
            hz = self.frequency[i]
            point = QPointF(x(hz), y(10 * math.log10(self.power[i])))
            p.setBrush(QColor('#e4f3ff'))
            p.drawEllipse(point, 2.5, 2.5)
            text = f'{hz:g} Hz'
            width = fm.horizontalAdvance(text) + 10
            left = min(max(axes.left(), point.x()-width/2), axes.right()-width)
            top = max(axes.top(), point.y()-fm.height()-6)
            rect = QRectF(left, top, width, fm.height()+2)
            if any(rect.intersects(other) for other in used):
                continue
            used.append(rect.adjusted(-2, -2, 2, 2))
            p.fillRect(rect, QColor('#22354a'))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class SpectrumPanel(QFrame):
    def __init__(self, tr):
        super().__init__()
        self.tr = tr
        self.setObjectName('spectrumPanel')
        self.setStyleSheet('QFrame#spectrumPanel {background:#111b29; border-radius:6px;} '
                           'QFrame#spectrumPanel QLabel {background:transparent; color:#e4edf6;}')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(4)
        header = QHBoxLayout()
        self.title = QLabel(tr('quality.spectrum.raw'))
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.title.setWordWrap(True)
        header.addWidget(self.title, 1)
        self.range = QComboBox()
        self.range.addItems(['0–65 Hz', tr('quality.spectrum.full')])
        self.range.setAccessibleName(tr('quality.spectrum.range'))
        header.addWidget(self.range)
        layout.addLayout(header)
        self.chart = SpectrumChart(tr)
        layout.addWidget(self.chart, 1)
        self.peak_readout = QLabel()
        self.peak_readout.setWordWrap(True)
        layout.addWidget(self.peak_readout)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        band_row = QWidget()
        band_row.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.cards = QGridLayout(band_row)
        self.cards.setContentsMargins(0, 0, 0, 0)
        self.cards.setHorizontalSpacing(8)
        layout.addWidget(band_row)
        self.band_widgets = []
        for i, (name, low, high, color) in enumerate(BANDS):
            card = QWidget()
            column = QVBoxLayout(card)
            column.setContentsMargins(2, 4, 2, 2)
            column.setSpacing(2)
            heading = QLabel(name)
            heading.setStyleSheet(f'color:{color}; font-weight:600;')
            column.addWidget(heading)
            reading = QLabel()
            reading.setTextFormat(Qt.TextFormat.PlainText)
            column.addWidget(reading)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            bar.setFixedHeight(4)
            bar.setStyleSheet(f'QProgressBar {{background:#304153; border:0;}} '
                              f'QProgressBar::chunk {{background:{color};}}')
            bar.setAccessibleName(f'{name} · {low:g}–{high:g} Hz')
            column.addWidget(bar)
            self.band_widgets.append((card, reading, bar))
            self.cards.addWidget(card, 0, i)
        self.columns = len(BANDS)
        self.range.currentIndexChanged.connect(self._range_changed)
        self.set_spectrum('', {}, None)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        fm = self.fontMetrics()
        card_width = fm.horizontalAdvance('0.5–45 Hz · 100.0%') + 15
        columns = min(len(BANDS), max(1, (event.size().width() - 24) // card_width))
        if columns != self.columns:
            self.columns = columns
            for i, (card, _, _) in enumerate(self.band_widgets):
                self.cards.addWidget(card, i // columns, i % columns)

    def _range_changed(self, index):
        self.chart.full_range = index == 1
        self._update_peaks()

    def _update_peaks(self):
        chart = self.chart
        chart.peaks = visible_peaks(chart.frequency, chart.power, chart.hz_limit())
        peaks = '  ·  '.join(f'{chart.frequency[i]:g} Hz' for i in chart.peaks)
        self.peak_readout.setText(self.tr('quality.spectrum.peaks', values=peaks or '—'))
        chart.update()

    def set_spectrum(self, channel, psd, power):
        freq = psd.get('frequency_hz', []) if power else []
        self.chart.frequency, self.chart.power = freq, power or []
        title = ' · '.join(filter(None, (channel, self.tr('quality.spectrum.raw'), 'Welch · dB µV²/Hz')))
        self.title.setText(title)
        shares = band_shares(freq, power) if freq else None
        note = self.tr('quality.spectrum.note')
        if type(psd.get('sample_stop')) is int and freq:
            note += '  ·  ' + self.tr('quality.spectrum.window_end', stop=psd['sample_stop'])
        self.note.setText(note)
        descriptions = []
        for i, (name, low, high, _) in enumerate(BANDS):
            _, reading, bar = self.band_widgets[i]
            share = shares[i] if shares else None
            percent = '—' if share is None else ('<0.1%' if 0 < share < .001 else f'{100*share:.1f}%')
            text = f'{low:g}–{high:g} Hz · {percent}'
            reading.setText(text)
            bar.setValue(round(1000 * share) if share is not None else 0)
            descriptions.append(f'{name} {text}')
        self.chart.setAccessibleDescription(' | '.join(descriptions))
        self._update_peaks()

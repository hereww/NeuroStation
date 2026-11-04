"""Read-only operator quality evidence, with explicit live/demo/replay context."""
from collections import deque
import math
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)

from .i18n import Translator


class QualityPlot(QWidget):
    def __init__(self, parent=None, tr=None):
        super().__init__(parent)
        self.tr = tr or Translator()
        self.setMinimumHeight(150)
        self.raw, self.display, self.psd, self.frequency = [], [], None, None

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#111b29'))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        wave_right = width * .65
        p.setPen(QColor('#d3deef'))
        p.drawText(8, 18, self.tr('quality.plot.raw'))
        p.drawText(int(wave_right + 12), 18, 'PSD · Welch 8s / 4s / 50%')
        finite = [abs(v) for seq in (self.raw, self.display) for v in seq
                  if v is not None and math.isfinite(v)]
        scale = max(max(finite, default=1), 1)

        def draw(values, color, left, right, top, bottom, lo, hi):
            path, active = QPainterPath(), False
            for i, value in enumerate(values):
                if value is None or not math.isfinite(value):
                    active = False
                    continue
                x = left + i * (right - left) / max(1, len(values) - 1)
                y = bottom - (value - lo) / (hi - lo) * (bottom - top)
                if active:
                    path.lineTo(x, y)
                else:
                    path.moveTo(x, y)
                    active = True
            p.setPen(QPen(QColor(color), 1))
            p.drawPath(path)

        draw(self.raw, '#97b7e8', 8, wave_right - 8, 40, height - 26, -scale, scale)
        draw(self.display, '#3ce5bb', 8, wave_right - 8, 40, height - 26, -scale, scale)
        p.setPen(QColor('#d3deef'))
        p.drawText(8, height - 5, self.tr('quality.plot.legend', scale=f'{scale:.1f}'))
        if not self.raw:
            p.drawText(12, 80, self.tr('quality.plot.empty'))
        if self.psd:
            logs = [10 * math.log10(max(v, 1e-20)) for v in self.psd]
            draw(logs, '#f0b663', wave_right + 12, width - 12, 40, height - 26,
                 min(logs) - 1, max(logs) + 1)
            p.setPen(QColor('#d3deef'))
            end = self.frequency[-1] if self.frequency else 0
            p.drawText(int(wave_right + 12), height - 5, f'0 Hz                 {end:g} Hz · dB µV²/Hz')
        else:
            p.drawText(int(wave_right + 12), 80, self.tr('quality.plot.wait_psd'))
        p.end()


class QualityPanel(QWidget):
    def __init__(self, parent=None, tr=None):
        super().__init__(parent)
        self.tr = tr or Translator()
        self.context, self.running = 'live', False
        self.latest = self.identity = None
        self.last_stop = -1
        self.raw, self.display = [], []
        layout, bar = QVBoxLayout(self), QHBoxLayout()
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        self.channel = QComboBox()
        self.channel.setAccessibleName(self.tr('quality.channel_select'))
        bar.addWidget(self.status, 1)
        bar.addWidget(self.channel)
        layout.addLayout(bar)
        self.dimensions = QLabel()
        self.dimensions.setTextFormat(Qt.TextFormat.PlainText)
        self.dimensions.setWordWrap(True)
        layout.addWidget(self.dimensions)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([self.tr('quality.column.' + k) for k in
            ('channel', 'mean', 'rms', 'p2p', 'drift', 'status')])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(150)
        self.table.setMaximumHeight(300)
        layout.addWidget(self.table)
        self.plot = QualityPlot(tr=self.tr)
        layout.addWidget(self.plot)
        self.details = QLabel()
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)
        self.channel.currentIndexChanged.connect(self._plot)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._age)
        self.timer.start()
        self.clear()

    def clear(self):
        self.latest = self.identity = None
        self.last_stop = -1
        self.raw, self.display = [], []
        self.channel.clear()
        self.table.setRowCount(8)
        for i in range(8):
            for j in range(6):
                self.table.setItem(i, j, QTableWidgetItem(f'CH{i+1}' if j == 0 else '—'))
        self.status.setText(self.tr('quality.panel_empty'))
        self.dimensions.setText(self.tr('quality.dimensions_empty'))
        self.details.setText(self.tr('quality.panel_note'))
        self.plot.raw, self.plot.display, self.plot.psd, self.plot.frequency = [], [], None, None
        self.plot.update()

    def set_context(self, context='live', running=False):
        self.context, self.running = context, running
        self._age()

    def _name(self, prefix, value):
        key = prefix + str(value)
        translated = self.tr(key)
        return str(value) if translated == key else translated

    def update_quality(self, result):
        if not isinstance(result, dict) or result.get('schema_version') != 2:
            return
        self.latest = result
        channels = result.get('channels', [])
        key = (result.get('stream_id'), result.get('processing_id'))
        if self.identity != key:
            self.identity, self.last_stop = key, -1
            self.raw = [deque(maxlen=1250) for _ in channels]
            self.display = [deque(maxlen=1250) for _ in channels]
            self.channel.blockSignals(True)
            self.channel.clear()
            self.channel.addItems([r['channel'] for r in channels])
            self.channel.blockSignals(False)
        if result.get('stop', -1) > self.last_stop:
            if self.last_stop >= 0 and result.get('start', self.last_stop) > self.last_stop:
                missing = min(1250, result['start'] - self.last_stop)
                for buffer in (*self.raw, *self.display):
                    buffer.extend([None] * missing)
            for buffers, name in ((self.raw, 'raw_samples'), (self.display, 'display_samples')):
                for buffer, values in zip(buffers, result.get(name, [])):
                    buffer.extend(values)
            self.last_stop = result.get('stop', -1)
        self.table.setRowCount(len(channels))
        for i, row in enumerate(channels):
            status = row.get('quality_status', 'unknown')
            values = [row['channel'], *[row.get(k) for k in
                      ('mean_uv', 'ac_rms_uv', 'peak_to_peak_uv', 'drift_range_uv')],
                      self._name('quality.state.', status)]
            for j, value in enumerate(values):
                item = QTableWidgetItem(self.tr('quality.pending') if value is None else
                                       f'{value:.2f}' if isinstance(value, float) else str(value))
                if j == 5:
                    item.setToolTip('；'.join(self._name('quality.reason.', r)
                                    for r in row.get('quality_reasons', [])))
                    item.setForeground(QColor('#b44242' if status == 'fail' else '#856100'))
                self.table.setItem(i, j, item)
        dimensions = result.get('dimensions', {})
        self.dimensions.setText('  |  '.join(self.tr('quality.dimension.' + k) + ': ' +
            self._name('quality.state.', dimensions.get(k, 'unknown')) for k in
            ('acquisition', 'integrity', 'signal_quality', 'stimulus_timing', 'coverage')))
        intervals = result.get('artifacts', {}).get('intervals', [])
        reasons = '；'.join(self._name('quality.reason.', r) for r in result.get('reasons', []))
        detail = '；'.join(f"{r['channel']} {self._name('quality.reason.', r['rule'])} [{r['start']},{r['stop']})"
                          for r in intervals[-6:]) or self.tr('quality.no_confirmed_artifacts')
        watermarks = result.get('watermarks', {})
        water_text = ' / '.join(str(watermarks.get(k, '—')) for k in
                               ('received', 'written', 'durable', 'assessed'))
        self.details.setText(self.tr('quality.details', reasons=reasons or '—', intervals=detail,
            watermarks=water_text, identity=str(result.get('processing_id', ''))[:12]))
        self._age()
        self._plot()

    def _age(self):
        if not self.latest:
            return
        state = self._name('quality.state.', self.latest.get('dimensions', {}).get('signal_quality', 'unknown'))
        if self.context != 'live':
            self.status.setText(self.tr('quality.recorded_state', state=state,
                activity=self.tr('quality.playing' if self.running else 'quality.static')))
            return
        stamp = self.latest.get('updated_monotonic_s')
        age = max(0, time.monotonic() - stamp) if isinstance(stamp, (float, int)) else float('inf')
        if age > self.latest.get('stale_after_s', 2):
            state = self.tr('quality.expired')
        self.status.setText(self.tr('quality.live_state', state=state, age=f'{age:.1f}'))

    def _plot(self):
        index = self.channel.currentIndex()
        if index < 0 or index >= len(self.raw):
            return
        self.plot.raw, self.plot.display = list(self.raw[index]), list(self.display[index])
        psd = (self.latest or {}).get('psd') or {}
        power = psd.get('power_uv2_hz', [])
        self.plot.psd = power[index] if index < len(power) else None
        self.plot.frequency = psd.get('frequency_hz')
        self.plot.update()

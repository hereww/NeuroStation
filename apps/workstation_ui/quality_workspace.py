"""Operator-facing, hardware-free preview/replay/report workflow.

The controller owns one cancellable worker. A bounded mailbox protects the Qt
queue. Live data comes only from MainWindow's existing consuming reader.
"""
from __future__ import annotations

from collections import deque
import importlib
import os
from pathlib import Path
import threading

from PySide6.QtCore import QObject, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QProgressBar, QFileDialog,
                               QTabWidget, QTextBrowser, QGridLayout, QRadioButton,
                               QVBoxLayout, QStackedWidget)

from .components import Page, Section, action, label
from .quality_panel import QualityPanel
from neurostation_quality_report import render_report_html


class _QualityWorker(QObject):
    finished = Signal(object)

    def __init__(self, job, root, session, options, cancel, mailbox, lock):
        super().__init__()
        self.job, self.root, self.session, self.options = job, root, session, options
        self.cancel, self.mailbox, self.lock = cancel, mailbox, lock
        self.skipped = 0

    def publish(self, result, done, total):
        if self.cancel.is_set():
            return
        with self.lock:
            if len(self.mailbox) == self.mailbox.maxlen:
                self.skipped += 1
            self.mailbox.append((result, done, total, self.skipped))

    @Slot()
    def run(self):
        try:
            backend = importlib.import_module('eeg_tools.quality.preview')
            if self.job == 'demo':
                result = backend.run_demo(self.root, self.publish, self.cancel, **self.options)
            elif self.job == 'replay':
                result = backend.run_replay(self.session, self.root, self.publish,
                                            self.cancel, **self.options)
            elif self.job == 'export':
                result = backend.export_report(self.session, self.options['report'],
                                                self.options['destination'], self.cancel)
            else:
                result = backend.run_report(self.session, self.root, self.cancel)
            self.finished.emit({'result': result, 'cancelled': self.cancel.is_set()})
        except Exception as error:
            self.finished.emit({'error': str(error), 'cancelled': self.cancel.is_set()})


class QualityWorkspaceController(QObject):
    changed = Signal()
    quality_received = Signal(object)
    busy_changed = Signal(bool)

    def __init__(self, parent=None, output_root=None):
        super().__init__(parent)
        root = output_root or os.environ.get('NEUROSTATION_QUALITY_ROOT')
        if not root:
            root = Path(os.environ.get('NEUROSTATION_DATASETS',
                        str(Path.home() / 'AppData/Local/NeuroStationQualityUpgrade/recordings'))).parent
        self.output_root = Path(root) / 'quality-workspace'
        self.mode = 'live'
        self.phase = 'idle'
        self.job = None
        self.session = None
        self.has_blocks = False
        self.latest = None
        self.report = None
        self.verification = None
        self.export_dir = None
        self.message = 'quality.waiting_live'
        self.progress = (0, 0)
        self.skipped = 0
        self.capture_busy = False
        self.thread = None
        self.worker = None
        self.cancel_event = threading.Event()
        self.mailbox = deque(maxlen=8)
        self.lock = threading.Lock()
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._flush)
        # Test hooks control only software jobs, never a capture configuration.
        self.demo_options = {}
        self.replay_options = {}

    @property
    def busy(self):
        return self.thread is not None

    def set_capture_busy(self, busy):
        busy = bool(busy)
        if self.capture_busy != busy:
            self.capture_busy = busy
            if busy and self.mode == 'live':
                self.latest = None
            self.changed.emit()

    def use_live(self):
        return self._reset_source('live', 'quality.waiting_live')

    def select_mode(self, mode):
        if mode == self.mode:
            return True
        if mode not in ('live', 'replay', 'demo') or self.capture_busy:
            return False
        message = {'live': 'quality.waiting_live', 'replay': 'quality.choose_hint',
                   'demo': 'quality.demo_hint'}[mode]
        return self._reset_source(mode, message)

    def _reset_source(self, mode, message, session=None):
        if self.busy:
            return False
        self.mode, self.phase, self.job = mode, 'idle', None
        self.session = session
        self.has_blocks = session is not None and (session / 'raw_blocks' / 'blocks.jsonl').is_file()
        self.latest = self.report = self.verification = self.export_dir = None
        self.skipped, self.progress = 0, (0, 0)
        self.message = 'quality.error.legacy_no_blocks' if session and not self.has_blocks else message
        self.changed.emit()
        return True

    def accept_live(self, result):
        if self.mode == 'live' and isinstance(result, dict) and result.get('schema_version') == 2:
            first_block = self.latest is None
            self.latest = result
            self.quality_received.emit(result)
            if first_block:
                self.changed.emit()

    def select_session(self, selected):
        if self.busy or self.capture_busy:
            return False
        path = Path(selected).expanduser().resolve()
        if path.name == 'raw_blocks' and (path / 'blocks.jsonl').is_file():
            path = path.parent
        if not path.is_dir() or not any((path / name).is_file() for name in
                                       ('session.json', 'events.tsv', 'raw_blocks/blocks.jsonl')):
            self._reset_source('replay', 'quality.error.not_session')
            return False
        return self._reset_source('replay', 'quality.session_selected', path)

    def start_demo(self):
        if self.busy or self.capture_busy:
            return False
        self.mode, self.session, self.has_blocks = 'demo', None, False
        self.latest = self.report = self.verification = self.export_dir = None
        return self._start('demo', self.demo_options)

    def start_replay(self):
        if self.busy or self.capture_busy or self.session is None:
            return False
        if not self.has_blocks:
            self.phase, self.message = 'error', 'quality.error.legacy_no_blocks'
            self.changed.emit()
            return False
        self.mode = 'replay'
        self.latest = self.verification = None
        return self._start('replay', self.replay_options)

    def start_report(self):
        if self.session is None or self.busy or self.capture_busy:
            return False
        # A failed refresh must not leave an old report presented as the new result.
        # Previous files remain available on disk; only this view's result is reset.
        self.report = self.export_dir = None
        return self._start('report')

    def export_report(self, destination):
        if not self.report or self.busy or self.capture_busy:
            return False
        self.export_dir = None
        return self._start('export', {'report': self.report, 'destination': str(destination)})

    def _start(self, job, options=None):
        if self.busy or self.capture_busy:
            return False
        self.cancel_event = threading.Event()
        self.mailbox.clear()
        self.job, self.phase = job, 'running'
        self.message = 'quality.running.' + job
        self.progress, self.skipped = (0, 0), 0
        self.thread = QThread(self)
        self.worker = _QualityWorker(job, self.output_root, self.session, options or {},
                                     self.cancel_event, self.mailbox, self.lock)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.timer.start()
        self.busy_changed.emit(True)
        self.changed.emit()
        self.thread.start()
        return True

    @Slot()
    def _flush(self):
        with self.lock:
            batch = list(self.mailbox)
            self.mailbox.clear()
        if self.cancel_event.is_set():
            return
        for result, done, total, skipped in batch:
            self.latest, self.progress, self.skipped = result, (done, total), skipped
            self.quality_received.emit(result)
        if batch:
            self.changed.emit()

    @Slot(object)
    def _finished(self, reply):
        self._flush()
        self.timer.stop()
        result = reply.get('result')
        if result and result.get('kind') == 'demo':
            self.session = Path(result['session_dir'])
            self.has_blocks = (self.session / 'raw_blocks' / 'blocks.jsonl').is_file()
        if reply.get('cancelled'):
            self.phase, self.message = 'cancelled', 'quality.cancelled'
        elif 'error' in reply:
            self.phase, self.message = 'error', reply['error']
        else:
            self.phase, self.message = 'completed', 'quality.done.' + self.job
            if self.job == 'report':
                self.report = result
            elif self.job == 'export':
                self.export_dir = result['export_dir']
            elif self.job == 'replay':
                self.verification = result['verification']
                if not self.verification['online_replay_match']:
                    self.message = 'quality.replay_mismatch'
        self.changed.emit()

    @Slot()
    def _thread_finished(self):
        self.thread = self.worker = None
        self.busy_changed.emit(False)
        self.changed.emit()

    def cancel(self):
        if self.busy:
            self.cancel_event.set()
            self.phase, self.message = 'stopping', 'quality.stopping'
            self.changed.emit()

    def close(self):
        if self.busy:
            self.cancel()
            return False
        self.timer.stop()
        return True


class ChannelCheckPage(Page):
    """Show the shared Cyton preflight result without starting a live reader."""

    def __init__(self, tr, run_check, open_details):
        super().__init__(tr, tr('quality.entry_title'), tr('channel_check.subtitle'))
        section = Section(tr('channel_check.title'))
        self.summary = label('', 'sectionTitle')
        section.layout.addWidget(self.summary)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        self.channel_labels = []
        for index in range(8):
            badge = label('')
            badge.setObjectName('channelCheckBadge')
            self.channel_labels.append(badge)
            grid.addWidget(badge, index // 2, index % 2)
        section.layout.addLayout(grid)
        self.layout.addWidget(section)
        self.check_button = action(tr('channel_check.run'), run_check, True)
        self.details_button = action(tr('channel_check.details'), open_details)
        actions = QHBoxLayout()
        actions.addWidget(self.check_button)
        actions.addWidget(self.details_button)
        actions.addStretch()
        self.layout.addLayout(actions)
        self.layout.addStretch()
        self.render(None)

    def render(self, report, *, running=False):
        states = {}
        status = None if report is None else report.get('status')
        if not running and status in ('passed', 'warning', 'degraded'):
            for check in report.get('checks') or ():
                if check.get('name') == 'channels':
                    metrics = check.get('metrics') or {}
                    states = {
                        row['channel']: row['status']
                        for row in metrics.get('channels', ())
                        if row.get('channel') and row.get('status') in ('passed', 'warning')
                    }
                    break
        for channel, badge in enumerate(self.channel_labels, 1):
            state = states.get(channel, 'unknown')
            badge.setText(f'CH{channel}  ·  {self.tr("channel_check." + state)}')
            if badge.property('state') != state:
                badge.setProperty('state', state)
                badge.style().unpolish(badge)
                badge.style().polish(badge)
        flagged = ', '.join(f'CH{i}' for i in range(1, 9) if states.get(i) == 'warning')
        if running:
            key = 'running'
        elif report is None:
            key = 'idle'
        elif status == 'failed':
            key = 'failed'
        elif flagged:
            key = 'flagged'
        elif status == 'degraded':
            key = 'review'
        elif all(states.get(i) == 'passed' for i in range(1, 9)):
            key = 'all_passed'
        else:
            key = 'incomplete'
        self.summary.setText(self.tr('channel_check.' + key, channels=flagged))
        return self.summary.text()


class QualityWorkspacePage(Page):
    def __init__(self, tr, controller, back_to_check):
        super().__init__(tr, tr('quality.title'), tr('quality.subtitle'))
        self.controller = controller
        self.context_key = None
        self.displayed_report = None
        source_box = Section(tr('quality.data_source'))
        sources = QHBoxLayout()
        self.mode_buttons = {}
        for mode in ('live', 'replay', 'demo'):
            button = QRadioButton(tr('quality.option.' + mode))
            button.clicked.connect(lambda _checked, selected=mode: controller.select_mode(selected))
            self.mode_buttons[mode] = button
            sources.addWidget(button)
        sources.addStretch()
        source_box.layout.addLayout(sources)
        self.source = label('', 'muted')
        source_box.layout.addWidget(self.source)
        tools = QHBoxLayout()
        self.choose_button = action(tr('quality.choose'), self.choose_session, True)
        self.demo_button = action(tr('quality.demo'), controller.start_demo, True)
        self.replay_button = action(tr('quality.replay'), controller.start_replay, True)
        self.stop_button = action(tr('quality.stop'), controller.cancel)
        for button in (self.choose_button, self.demo_button, self.replay_button, self.stop_button):
            tools.addWidget(button)
        tools.addStretch()
        source_box.layout.addLayout(tools)
        self.path = label('', 'path')
        source_box.layout.addWidget(self.path)
        self.layout.addWidget(source_box)
        self.status = label('')
        self.status.setObjectName('qualityJobStatus')
        self.layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)
        self.tabs = QTabWidget()
        self.signal_view = QStackedWidget()
        self.signal_empty = label(tr('quality.signal_empty'), 'muted')
        self.signal_view.addWidget(self.signal_empty)
        self.panel = QualityPanel(tr=tr, spectrum_first=True)
        self.panel.table.setMinimumHeight(275)
        self.panel.plot.setMinimumHeight(120)
        self.signal_view.addWidget(self.panel)
        self.tabs.addTab(self.signal_view, tr('quality.signal_tab'))
        report_widget = QWidget()
        report_layout = QVBoxLayout(report_widget)
        report_layout.addWidget(label(tr('quality.report_note'), 'muted'))
        row = QHBoxLayout()
        self.report_button = action(tr('quality.report'), controller.start_report, True)
        self.export_button = action(tr('quality.export'), self.export_report)
        self.folder_button = action(tr('quality.folder'), self.open_report_folder)
        for button in (self.report_button, self.export_button, self.folder_button):
            row.addWidget(button)
        report_layout.addLayout(row)
        self.report_view = QTextBrowser()
        self.report_view.setMinimumHeight(240)
        self.report_view.setPlainText(tr('quality.report_empty'))
        self.report_view.setOpenLinks(False)
        self.report_view.setOpenExternalLinks(False)
        report_layout.addWidget(self.report_view)
        self.tabs.addTab(report_widget, tr('quality.report_tab'))
        self.layout.addWidget(self.tabs)
        self.verification_label = label('', 'muted')
        self.layout.addWidget(self.verification_label)
        self.layout.addWidget(label(tr('quality.safety'), 'muted'))
        footer = QHBoxLayout()
        footer.addWidget(action(tr('channel_check.back'), back_to_check))
        footer.addStretch()
        self.layout.addLayout(footer)
        self.layout.addStretch()
        controller.changed.connect(self.refresh)
        controller.quality_received.connect(self.receive_quality)
        self.refresh()

    def choose_session(self):
        path = QFileDialog.getExistingDirectory(self, self.tr('quality.choose'))
        if path:
            self.controller.select_session(path)

    def export_report(self):
        path = QFileDialog.getExistingDirectory(self, self.tr('quality.export'))
        if path:
            self.controller.export_report(path)

    def open_report_folder(self):
        report = self.controller.report
        if report:
            QDesktopServices.openUrl(QUrl.fromLocalFile(
                self.controller.export_dir or report['output_dir']))

    def receive_quality(self, result):
        self.panel.set_context(self.controller.mode,
                               self.controller.busy and self.controller.phase == 'running' and self.controller.job in ('demo', 'replay'))
        self.panel.update_quality(result)
        self.signal_view.setCurrentWidget(self.panel)

    def refresh(self):
        c = self.controller
        key = c.mode, c.session
        new_source = key != self.context_key
        if new_source or (c.latest is None and self.panel.latest is not None):
            self.panel.clear()
            self.context_key = key
            if c.latest is not None:
                self.panel.update_quality(c.latest)
        self.mode_buttons[c.mode].setChecked(True)
        self.panel.set_context(c.mode, c.busy and c.phase == 'running' and c.job in ('demo', 'replay'))
        self.source.setText(self.tr('quality.source.' + c.mode))
        self.path.setVisible(c.session is not None)
        if c.session:
            self.path.setText(self.tr('quality.selected_path', path=str(c.session)))
        self.signal_view.setCurrentWidget(self.panel if c.latest is not None else self.signal_empty)
        self.tabs.setVisible(c.latest is not None or c.session is not None)
        self.tabs.setTabEnabled(1, c.session is not None)
        self.tabs.setTabEnabled(0, c.session is None or c.has_blocks or c.latest is not None)
        if new_source or (c.job in ('demo', 'replay') and c.busy and c.latest is None):
            self.tabs.setCurrentIndex(0 if self.tabs.isTabEnabled(0) else 1)
        message = self.tr(c.message)
        if c.mode == 'live' and c.latest is not None and c.message == 'quality.waiting_live':
            message = ''  # Freshness and quality are shown alongside the signal.
        if c.capture_busy:
            message = self.tr('quality.capture_busy') + '\n' + message
        if c.skipped:
            message += '\n' + self.tr('quality.skipped', count=c.skipped)
        if c.export_dir:
            message += '\n' + self.tr('quality.exported', path=c.export_dir)
        self.status.setText(message)
        enabled = not c.busy and not c.capture_busy
        self.status.setVisible(bool(message.strip()))
        for button in self.mode_buttons.values():
            button.setEnabled(enabled)
        self.choose_button.setVisible(c.mode == 'replay')
        self.demo_button.setVisible(c.mode == 'demo' and not c.busy)
        self.replay_button.setVisible(c.session is not None and c.has_blocks and not c.busy)
        self.stop_button.setVisible(c.busy)
        self.export_button.setVisible(c.report is not None)
        self.folder_button.setVisible(c.report is not None)
        self.demo_button.setEnabled(enabled)
        self.choose_button.setEnabled(enabled)
        self.replay_button.setEnabled(enabled and c.has_blocks)
        self.report_button.setEnabled(enabled and c.session is not None)
        self.export_button.setEnabled(enabled and c.report is not None)
        self.folder_button.setEnabled(c.report is not None)
        self.stop_button.setEnabled(c.busy and c.phase == 'running')
        done, total = c.progress
        self.progress.setRange(0, total or (0 if c.busy else 1))
        self.progress.setValue(done)
        self.progress.setVisible(c.busy)
        self.verification_label.setVisible(c.verification is not None or c.latest is not None)
        if c.verification:
            v = c.verification
            self.verification_label.setText(self.tr('quality.verification',
                blocks=v['verified_blocks'], samples=v['verified_samples'],
                match=self.tr('quality.yes' if v['online_replay_match'] else 'quality.no'),
                evidence=self.tr('quality.yes' if v.get('quality_evidence_verified') is True else 'quality.no'),
                hashes=self.tr('quality.yes' if v['raw_hashes_unchanged'] else 'quality.no'),
                complete=self.tr('quality.yes' if v['recording_complete'] else 'quality.no')))
        else:
            self.verification_label.setText(self.tr('quality.windows'))
        if c.report is not self.displayed_report:
            self.displayed_report = c.report
            if c.report:
                self.report_view.setHtml(render_report_html(c.report['report'], self.tr.locale))
                self.tabs.setCurrentIndex(1)
            else:
                self.report_view.setPlainText(self.tr('quality.report_empty'))

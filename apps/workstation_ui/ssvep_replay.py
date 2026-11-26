"""Run four-frequency replay from the selected recording's result page."""
from pathlib import Path
import json
import sys
from uuid import uuid4

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QComboBox, QDialog, QFileDialog, QTextEdit, QVBoxLayout, QWidget

from .components import Page, Section, action, label
from .gateway import CaptureMode
from .v9_offline import V9OfflineController


class ReplayController(V9OfflineController):
    def compare(self, test, calibration=None):
        if self.busy or self._closed or self.capture_busy or self.owner._offline_capture_busy():
            return False
        key = self.session_key(test)
        root = Path(__file__).resolve().parents[2]
        output = Path(key) / 'derived' / ('continuous_' + uuid4().hex)
        worker = [sys.executable]
        if not (getattr(sys, 'frozen', False) or hasattr(sys.modules['__main__'], '__compiled__')):
            worker.append(str(root / 'workstation.py'))
        command = [*worker, '--ssvep-replay', key, '--process-only', '--output-dir', str(output)]
        if calibration is not None:
            command.extend(['--calibration-session', str(calibration)])
        return self._launch(key, output, command, Path(key), 'cpu')

    def _read_verified(self, output, key):
        session_id = json.loads((Path(key) / 'session.json').read_text(encoding='utf-8-sig'))['session_id']
        manifest = json.loads((output / 'processing_manifest.json').read_text(encoding='utf-8'))
        predictions = manifest['predictions']
        if (manifest['kind'] != 'ssvep_signal_export' or manifest['schema_version'] != 1
                or manifest['session_id'] != session_id
                or not isinstance(predictions, dict) or 'baseline' not in predictions
                or any(mode not in ('personal', 'baseline') or filename != f'{mode}_predictions.tsv'
                       or not (output / filename).is_file() for mode, filename in predictions.items())
                or not all((output / file).is_file() for file in ('protocol.json', 'frozen_models.json'))
                or any(name not in ('events.tsv', 'frame_timing.tsv') or not (output / name).is_file()
                       for name in manifest['annotations'])):
            raise ValueError('Incomplete or mismatched SSVEP export')
        return manifest

    def restore(self, session_dir):
        key = self.session_key(session_dir)
        if key in self.states:
            return
        derived = Path(key) / 'derived'
        try:
            candidates = sorted((*derived.glob('continuous_*/processing_manifest.json'),
                                 *derived.glob('continuous_*/report.md')),
                                key=lambda p: p.stat().st_mtime_ns, reverse=True)
        except OSError:
            return
        for path in candidates:
            try:
                if path.name == 'report.md':
                    summary = json.loads((path.parent / 'metrics.json').read_text(encoding='utf-8'))
                    session = json.loads((Path(key) / 'session.json').read_text(encoding='utf-8-sig'))
                    if summary['test'] != session['session_id']:
                        continue
                    result = summary['metrics']
                else:
                    result = self._read_verified(path.parent, key)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            self.states[key] = dict(phase='complete', output_dir=str(path.parent), result=result,
                                    legacy_report=path.name == 'report.md')
            break

    def _progress_line(self, data):
        try:
            value = json.loads(data)
            if isinstance(value, dict) and value.get('status') == 'error':
                self._worker_error = str(value['error'])
        except (ValueError, KeyError):
            pass


class ReplayPanel(QWidget):
    def __init__(self, tr, test, controller):
        super().__init__()
        self.tr, self.test, self.controller = tr, None, controller
        layout = QVBoxLayout(self)
        self.baseline_button = action(tr('ssvep.baseline_run'), lambda: self.controller.compare(self.test), True)
        self.run_button = action(tr('ssvep.compare'), self.choose)
        self.open_button = action(tr('ssvep.open_report'), self.open_report)
        self.folder_button = action(tr('ssvep.open_outputs'), self.open_outputs)
        self.status = label('')
        for w in (self.baseline_button, self.run_button, self.status, self.folder_button, self.open_button):
            layout.addWidget(w)
        controller.updated.connect(self.refresh)
        controller.busy_changed.connect(self.refresh)
        self.set_session(test)

    def set_session(self, test):
        self.test = Path(test) if test else None
        if self.test is not None:
            self.controller.restore(self.test)
        self.refresh()

    def choose(self):
        earlier = QFileDialog.getExistingDirectory(self, self.tr('ssvep.choose_earlier'), str(self.test.parent))
        if earlier:
            self.controller.compare(self.test, earlier)

    def refresh(self, *_):
        state = self.controller.state_for(self.test) if self.test is not None else {'phase': 'idle'}
        ready = self.test is not None and not self.controller.busy and not self.controller.capture_busy
        self.run_button.setEnabled(ready)
        self.baseline_button.setEnabled(ready)
        self.folder_button.setVisible(state['phase'] == 'complete')
        self.open_button.setVisible(state['phase'] == 'complete' and bool(state.get('legacy_report')))
        text = state.get('error', '')
        if self.test is None:
            text = self.tr('ssvep.no_recording')
        elif state['phase'] == 'idle' and not ready:
            text = self.tr('ssvep.replay_busy')
        if state['phase'] == 'running':
            text = self.tr('ssvep.replaying')
        elif state['phase'] == 'complete':
            text = state['output_dir']
            if state.get('legacy_report'):
                text = self.tr('ssvep.legacy_report') + '\n' + text
        self.status.setText(text)

    def open_outputs(self):
        state = self.controller.state_for(self.test)
        if state['phase'] == 'complete':
            QDesktopServices.openUrl(QUrl.fromLocalFile(state['output_dir']))

    def open_report(self):
        state = self.controller.state_for(self.test)
        if state['phase'] != 'complete':
            return
        directory = Path(state['output_dir'])
        try:
            report = (directory / 'report.md').read_text(encoding='utf-8')
        except (OSError, UnicodeError) as error:
            self.status.setText(str(error))
            return
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle(self.tr('ssvep.open_report'))
        dialog.resize(800, 580)
        layout = QVBoxLayout(dialog)
        reader = QTextEdit(dialog)
        reader.setReadOnly(True)
        reader.setMarkdown(report)
        layout.addWidget(reader)
        layout.addWidget(action(self.tr('ssvep.open_outputs'),
                                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))))
        dialog.open()


class EEGProcessingPage(Page):
    def __init__(self, tr, datasets, controller, configure_capture):
        super().__init__(tr, tr('nav.processing'), tr('ssvep.processing_note'))
        offline = Section(tr('ssvep.offline'))
        offline.layout.addWidget(label(tr('ssvep.select_recording')))
        self.recordings = QComboBox()
        self.recordings.setAccessibleName(tr('ssvep.select_recording'))
        for item in datasets:
            if (item.source == CaptureMode.CYTON and not item.imported and not item.simulated
                    and not item.deleted_at and item.protocol.lower() == 'ssvep'
                    and (item.path / 'raw_brainflow.tsv').is_file()):
                self.recordings.addItem(f'{item.name} · {item.participant} · {item.created_at}', str(item.path))
        offline.layout.addWidget(self.recordings)
        self.panel = ReplayPanel(tr, self.recordings.currentData(), controller)
        self.recordings.currentIndexChanged.connect(lambda _: self.panel.set_session(self.recordings.currentData()))
        self.recordings.setDisabled(controller.busy)
        controller.busy_changed.connect(self.recordings.setDisabled)
        offline.layout.addWidget(self.panel)
        self.layout.addWidget(offline)
        live = Section(tr('ssvep.live'))
        live.layout.addWidget(label(tr('ssvep.live_note'), 'muted'))
        self.configure_button = action(tr('ssvep.configure_live'), configure_capture)
        live.layout.addWidget(self.configure_button)
        self.layout.addWidget(live)
        self.layout.addStretch()

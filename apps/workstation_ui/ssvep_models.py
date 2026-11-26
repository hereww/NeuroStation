"""Model library UI. Validation/import run in a separate local process."""
import json
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Signal
from PySide6.QtWidgets import QComboBox, QFileDialog, QHBoxLayout

from .components import Page, Section, action, label
from neurostation_models import BUILTIN, list_models


def populate_models(combo, selected):
    combo.blockSignals(True)
    combo.clear()
    for model in list_models():
        combo.addItem(model.label, model.id)
    # Keep a missing selection explicit rather than silently running another model.
    index = combo.findData(selected)
    if index < 0:
        combo.addItem(f'[{selected}]', selected)
        index = combo.count() - 1
    combo.setCurrentIndex(index)
    combo.blockSignals(False)


class ModelsPage(Page):
    apply_requested = Signal(str)
    imported = Signal(str)
    busy_changed = Signal()

    def __init__(self, tr):
        super().__init__(tr, tr('nav.models'), tr('models.subtitle'))
        section = Section(tr('models.library'))
        self.combo = QComboBox()
        section.layout.addWidget(self.combo)
        self.details = label('', 'muted')
        section.layout.addWidget(self.details)
        self.combo.currentIndexChanged.connect(self.show_details)
        self.use = action(tr('models.apply'), lambda: self.apply_requested.emit(self.combo.currentData()), True)
        self.probe = action(tr('models.check'), lambda: self.run('check', self.combo.currentData()))
        self.add = action(tr('models.import'), self.choose_file)
        buttons = QHBoxLayout()
        for button in (self.use, self.probe, self.add):
            buttons.addWidget(button)
        section.layout.addLayout(buttons)
        self.status = label('')
        section.layout.addWidget(self.status)
        self.layout.addWidget(section)
        self.layout.addStretch(1)
        self.process = None
        self.capture_busy = False
        self.refresh(BUILTIN)

    def refresh(self, selected):
        populate_models(self.combo, selected)
        self.show_details()

    def show_details(self):
        selected = self.combo.currentData()
        model = next((m for m in list_models() if m.id == selected), None)
        self.details.setText(self.tr('models.details', digest=model.sha256) if model else self.tr('models.missing'))
        self.status.clear()

    def set_capture_busy(self, busy):
        self.capture_busy = busy
        for widget in (self.combo, self.use, self.probe, self.add):
            widget.setEnabled(not busy and self.process is None)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr('models.import'), '', 'ONNX (*.onnx)')
        if path:
            self.run('import', path)

    def run(self, mode, value):
        if self.capture_busy or self.process is not None:
            return
        proc = self.process = QProcess(self)
        self.set_capture_busy(False)
        self.busy_changed.emit()
        self.status.setText(self.tr('models.checking'))
        args = [] if getattr(sys, 'frozen', False) or '__compiled__' in globals() else [str(Path(__file__).resolve().parents[2] / 'workstation.py')]
        proc.finished.connect(lambda code, _status: self._done(proc, code, mode))
        proc.errorOccurred.connect(lambda err: self._done(proc, 1, mode) if err == QProcess.ProcessError.FailedToStart else None)
        proc.start(sys.executable, [*args, '--ssvep-model', mode, value])

    def _done(self, proc, code, mode):
        if proc is not self.process:
            return
        try:
            result = json.loads(bytes(proc.readAllStandardOutput()).decode('utf-8'))
            if code:
                raise ValueError(result.get('error') or proc.errorString())
            if mode == 'import':
                self.refresh(result['id'])
                self.imported.emit(result['id'])
            self.status.setText(self.tr('models.ready', name=result['name'], device=result['device']))
        except (ValueError, TypeError, KeyError) as error:
            self.status.setText(self.tr('models.failed', error=str(error)))
        finally:
            proc.deleteLater()
            self.process = None
            self.set_capture_busy(self.capture_busy)
            self.busy_changed.emit()

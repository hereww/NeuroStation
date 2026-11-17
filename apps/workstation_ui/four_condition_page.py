"""Four-condition setup and a manually scrubbed, non-flickering geometry preview.

This page intentionally has no acquisition/start signal. Exported parameter locks
are not device preflight, optical validation, or participant risk acknowledgement.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QWidget, QFormLayout, QLineEdit, QComboBox, QCheckBox,
                              QSlider, QHBoxLayout, QFileDialog, QTabWidget)

from neurostation_four_condition import CONDITIONS, PARAMETERS, FourConditionProtocol, parameter_digest
from neurostation_protocol_errors import SSVEPProtocolError
from .components import Page, Section, action, label

INTEGER_FIELDS = {'refresh_rate_hz', 'trials_per_block', 'blocks_per_condition'}
CHOICES = {
    'shape': ('rectangle', 'circle'),
    'motion_mode': ('smooth_horizontal',),
    'gaze_instruction': ('central_fixation_attention_follow',),
    'makeup_policy': ('none_preserve_all_trials',),
}
GROUPS = (
    ('display', PARAMETERS[:14]),
    ('timing', PARAMETERS[14:]),
)


class ConditionGeometryPreview(QWidget):
    """Never toggles luminance or advances automatically; no timer is installed."""
    def __init__(self, tr):
        super().__init__()
        self.tr = tr
        self.protocol = None
        self.frame = None
        self.setMinimumHeight(210)
        self.setObjectName('fourConditionStaticPreview')

    def set_frame(self, protocol, frame):
        self.protocol, self.frame = protocol, frame
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#172331'))
        painter.setPen(QColor('#b7c6d5'))
        viewport = QRectF(24, 38, max(1, self.width() - 48), max(1, self.height() - 62))
        painter.drawRect(viewport)
        painter.drawText(12, 23, self.tr('four.preview_note'))
        painter.drawText(viewport, Qt.AlignmentFlag.AlignCenter, '+')
        if self.protocol is not None and self.frame is not None:
            p, f = self.protocol.parameters, self.frame
            painter.setPen(QPen(QColor('#54738a'), 1, Qt.PenStyle.DashLine))
            yy = viewport.top() + p['y'] * viewport.height()
            painter.drawLine(int(viewport.left() + p['left_x'] * viewport.width()), int(yy),
                             int(viewport.left() + p['right_x'] * viewport.width()), int(yy))
            if f.target_visible:
                w, h = p['target_width'] * viewport.width(), p['target_height'] * viewport.height()
                if p['shape'] == 'circle':
                    w = h = min(w, h)
                rect = QRectF(viewport.left() + f.x * viewport.width() - w / 2, yy - h / 2, w, h)
                # Deliberately constant muted color, ignoring on/off waveform.
                painter.setPen(QColor('#76b6bd'))
                painter.setBrush(QColor('#467e88'))
                if p['shape'] == 'circle':
                    painter.drawEllipse(rect)
                else:
                    painter.drawRect(rect)
        painter.end()


class FourConditionPage(Page):
    def __init__(self, tr, navigate):
        super().__init__(tr, tr('four.title'), tr('four.subtitle'))
        self.layout.addWidget(action(tr('four.back'), lambda: navigate('ssvep')))
        self.layout.addWidget(label(tr('four.boundary'), 'notice'))
        self.layout.addWidget(label(tr('four.order'), 'estimate'))
        self._base = {
            'schema_version': 3, 'kind': 'ssvep_four_condition',
            'protocol_id': 'ssvep_four_condition_v1', 'status': 'formal',
            'condition_order': list(CONDITIONS),
            'parameters': dict.fromkeys(PARAMETERS), 'parameter_confirmation': None,
        }
        self.locked_protocol = None
        self.fields = {}
        tabs = QTabWidget()
        for group, keys in GROUPS:
            panel = QWidget()
            form = QFormLayout(panel)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for key in keys:
                if key in CHOICES:
                    widget = QComboBox()
                    widget.addItem(tr('four.pending'), None)
                    for value in CHOICES[key]:
                        widget.addItem(tr('four.choice.' + value), value)
                    widget.currentIndexChanged.connect(self._invalidate)
                else:
                    widget = QLineEdit()
                    widget.setPlaceholderText(tr('four.pending'))
                    widget.textChanged.connect(self._invalidate)
                widget.setObjectName('four_' + key)
                self.fields[key] = widget
                form.addRow(tr('four.param.' + key), widget)
            tabs.addTab(panel, tr('four.group.' + group))
        self.layout.addWidget(tabs)
        self.layout.addWidget(label(tr('four.suggestions'), 'muted'))
        confirmation = Section(tr('four.confirm_title'))
        form = QFormLayout()
        self.operator = QLineEdit()
        self.operator.setObjectName('fourConditionOperator')
        self.operator.textChanged.connect(self._invalidate)
        form.addRow(tr('four.operator'), self.operator)
        confirmation.layout.addLayout(form)
        self.reviewed = QCheckBox(tr('four.reviewed'))
        self.reviewed.toggled.connect(self._invalidate)
        confirmation.layout.addWidget(self.reviewed)
        controls = QHBoxLayout()
        self.load_button = action(tr('four.load'), self._load)
        self.confirm_button = action(tr('four.confirm'), self.confirm_parameters, True)
        self.export_button = action(tr('four.export'), self._export)
        self.export_button.setEnabled(False)
        for button in (self.load_button, self.confirm_button, self.export_button):
            controls.addWidget(button)
        confirmation.layout.addLayout(controls)
        self.status_label = label(tr('four.unlocked'), 'notice')
        confirmation.layout.addWidget(self.status_label)
        self.layout.addWidget(confirmation)
        self.condition = QComboBox()
        for value in CONDITIONS:
            self.condition.addItem(tr('four.condition.' + value), value)
        self.condition.currentIndexChanged.connect(self._update_preview)
        self.layout.addWidget(self.condition)
        self.preview = ConditionGeometryPreview(tr)
        self.layout.addWidget(self.preview)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._update_preview)
        self.layout.addWidget(self.frame_slider)
        self.frame_label = label(tr('four.unlocked'), 'muted')
        self.layout.addWidget(self.frame_label)
        self.start_button = action(tr('four.start_unavailable'), lambda: None)
        self.start_button.setEnabled(False)
        self.layout.addWidget(self.start_button)

    def _invalidate(self, *_):
        self.locked_protocol = None
        if hasattr(self, 'export_button'):
            self.export_button.setEnabled(False)
            self.frame_slider.setEnabled(False)
            self.preview.set_frame(None, None)
            self.frame_label.setText(self.tr('four.unlocked'))
            self.status_label.setText(self.tr('four.unlocked'))

    def confirm_parameters(self):
        try:
            if not self.reviewed.isChecked() or not self.operator.text().strip():
                raise ValueError(self.tr('four.confirm_required'))
            parameters = {}
            for key, widget in self.fields.items():
                if key in CHOICES:
                    parameters[key] = widget.currentData()
                else:
                    text = widget.text().strip()
                    try:
                        parameters[key] = None if not text else (int(text) if key in INTEGER_FIELDS else float(text))
                    except ValueError as exc:
                        raise ValueError(self.tr('four.param.' + key) + ': ' + self.tr('four.invalid_number')) from exc
            value = {**self._base, 'parameters': parameters, 'parameter_confirmation': None}
            protocol = FourConditionProtocol.from_dict(value)
            missing = [key for key in protocol.pending if key != 'parameter_confirmation']
            if missing:
                raise ValueError(self.tr('four.missing') + ': ' + ', '.join(self.tr('four.param.' + k) for k in missing))
            value['parameter_confirmation'] = {
                'confirmed_by': self.operator.text().strip(),
                'confirmed_at_utc': datetime.now(timezone.utc).isoformat(),
                'parameter_sha256': parameter_digest(parameters),
            }
            self.locked_protocol = FourConditionProtocol.from_dict(value)
            trials = self.locked_protocol.build_trials()
            self.status_label.setText(self.tr('four.locked', trials=len(trials),
                duration=f'{trials[-1].rest_end_s:.1f}', identity=self.locked_protocol.processing_identity[:16]))
            self.export_button.setEnabled(True)
            self.frame_slider.setRange(0, round(parameters['stimulus_s'] * parameters['refresh_rate_hz']))
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            self._update_preview()
        except (ValueError, OverflowError) as exc:
            self._invalidate()
            self.status_label.setText(str(exc))

    def _update_preview(self, *_):
        if self.locked_protocol is None:
            return
        frame = self.locked_protocol.frame(self.condition.currentData(), self.frame_slider.value())
        self.preview.set_frame(self.locked_protocol, frame)
        self.frame_label.setText(self.tr('four.frame', index=frame.frame_index,
            time=f'{frame.scheduled_s:.3f}', x=f'{frame.x:.3f}', y=f'{frame.y:.3f}',
            phase=frame.phase, lit=int(frame.lit)))

    def draft_state(self):
        return {
            'base': self._base,
            'values': {k: w.currentData() if k in CHOICES else w.text() for k,w in self.fields.items()},
            'operator': self.operator.text(),
        }

    def restore_draft(self, state):
        self._base = state['base']
        for key, value in state['values'].items():
            widget = self.fields[key]
            if key in CHOICES:
                widget.setCurrentIndex(max(0, widget.findData(value)))
            else:
                widget.setText(value)
        self.operator.setText(state['operator'])
        # Preserve edits across language rebuilds, but never reconfirm implicitly.
        self.reviewed.setChecked(False)
        self._invalidate()

    def load_path(self, path: Path):
        # Validate fully before changing any current form. Imported locks are never
        # accepted as a new operator confirmation in this UI.
        protocol = FourConditionProtocol.load(path)
        self._base = json.loads(protocol.canonical_json)
        self._base['parameter_confirmation'] = None
        for key, value in protocol.parameters.items():
            widget = self.fields[key]
            if key in CHOICES:
                widget.setCurrentIndex(max(0, widget.findData(value)))
            else:
                widget.setText('' if value is None else str(value))
        self.operator.clear()
        self.reviewed.setChecked(False)
        self._invalidate()

    def export_path(self, path: Path):
        if self.locked_protocol is None:
            raise SSVEPProtocolError(self.tr('four.unlocked'))
        text = json.dumps(json.loads(self.locked_protocol.canonical_json), ensure_ascii=False, indent=2) + '\n'
        # No overwriting existing settings, protocol snapshots, or session files.
        with path.open('x', encoding='utf-8') as file:
            file.write(text)

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr('four.load'), '', 'JSON (*.json)')
        if path:
            try:
                self.load_path(Path(path))
            except (OSError, ValueError) as exc:
                self.status_label.setText(str(exc))

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, self.tr('four.export'), 'ssvep_four_condition_parameters.json', 'JSON (*.json)')
        if path:
            try:
                self.export_path(Path(path))
                self.status_label.setText(self.tr('four.saved', path=path))
            except (OSError, ValueError) as exc:
                self.status_label.setText(str(exc))

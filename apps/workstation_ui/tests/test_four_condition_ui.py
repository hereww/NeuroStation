"""Offscreen-only setup/render tests. No device, animation timer or flicker window."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HAS_QT = importlib.util.find_spec('PySide6') is not None


@unittest.skipUnless(HAS_QT, 'PySide6 not installed')
class FourConditionPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from apps.workstation_ui.four_condition_page import FourConditionPage
        from apps.workstation_ui.i18n import Translator
        self.temp = tempfile.TemporaryDirectory()
        self.navigation = []
        self.page = FourConditionPage(Translator(), self.navigation.append)

    def tearDown(self):
        self.page.close()
        self.page.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def fill_fixture(self):
        from apps.workstation_ui.four_condition_page import CHOICES
        from tests.test_ssvep_four_conditions import synthetic_value
        for key, value in synthetic_value()['parameters'].items():
            widget = self.page.fields[key]
            if key in CHOICES:
                widget.setCurrentIndex(widget.findData(value))
            else:
                widget.setText(str(value))
        self.page.operator.setText('unit-test-only')
        self.page.reviewed.setChecked(True)

    def test_defaults_are_empty_and_do_not_start_capture(self):
        from apps.workstation_ui.four_condition_page import CHOICES
        from PySide6.QtCore import QTimer
        self.assertTrue(all(w.currentData() is None if k in CHOICES else not w.text()
                            for k,w in self.page.fields.items()))
        self.assertFalse(self.page.reviewed.isChecked())
        self.assertFalse(self.page.start_button.isEnabled())
        self.assertFalse(self.page.export_button.isEnabled())
        self.assertFalse(self.page.frame_slider.isEnabled())
        self.assertEqual([], self.page.findChildren(QTimer))
        self.assertFalse(hasattr(self.page, 'start_requested'))

    def test_confirmation_needs_operator_and_review(self):
        self.fill_fixture()
        self.page.reviewed.setChecked(False)
        self.page.confirm_parameters()
        self.assertIsNone(self.page.locked_protocol)
        self.page.reviewed.setChecked(True)
        self.page.operator.clear()
        self.page.confirm_parameters()
        self.assertIsNone(self.page.locked_protocol)

    def test_confirmation_missing_numbers_not_replaced_by_suggestions(self):
        self.page.operator.setText('test')
        self.page.reviewed.setChecked(True)
        self.page.confirm_parameters()
        self.assertIsNone(self.page.locked_protocol)
        self.assertIn('仍需确认', self.page.status_label.text())

    def test_confirmation_enables_static_preview_not_capture(self):
        self.fill_fixture()
        self.page.confirm_parameters()
        self.assertIsNotNone(self.page.locked_protocol)
        self.assertTrue(self.page.export_button.isEnabled())
        self.assertTrue(self.page.frame_slider.isEnabled())
        self.assertFalse(self.page.start_button.isEnabled())
        self.assertIn('120', self.page.status_label.text())
        self.assertIn('1260', self.page.status_label.text())
        self.page.condition.setCurrentIndex(2)
        self.page.frame_slider.setValue(239)
        self.assertAlmostEqual(.8, self.page.preview.frame.x)
        self.page.frame_slider.setValue(240)
        self.assertFalse(self.page.preview.frame.target_visible)

    def test_any_parameter_edit_invalidates_lock_and_preview(self):
        self.fill_fixture()
        self.page.confirm_parameters()
        self.page.fields['frequency_hz'].setText('12')
        self.assertIsNone(self.page.locked_protocol)
        self.assertIsNone(self.page.preview.frame)
        self.assertFalse(self.page.export_button.isEnabled())
        self.assertFalse(self.page.frame_slider.isEnabled())

    def test_bad_numeric_input_never_enables_export(self):
        for bad in ('nan', 'inf', 'text', '-10'):
            self.fill_fixture()
            self.page.fields['frequency_hz'].setText(bad)
            self.page.confirm_parameters()
            self.assertIsNone(self.page.locked_protocol)
            self.assertFalse(self.page.export_button.isEnabled())

    def test_operator_or_review_changes_invalidate_lock(self):
        self.fill_fixture()
        self.page.confirm_parameters()
        self.page.operator.setText('another')
        self.assertIsNone(self.page.locked_protocol)
        self.page.confirm_parameters()
        self.page.reviewed.setChecked(False)
        self.assertIsNone(self.page.locked_protocol)

    def test_export_roundtrip_and_no_overwrite(self):
        from neurostation_four_condition import FourConditionProtocol
        self.fill_fixture()
        self.page.confirm_parameters()
        path = Path(self.temp.name) / 'parameters.json'
        self.page.export_path(path)
        saved = path.read_bytes()
        loaded = FourConditionProtocol.load(path)
        self.assertEqual(self.page.locked_protocol.processing_identity, loaded.processing_identity)
        self.assertIn('confirmed_at_utc', json.loads(saved)['parameter_confirmation'])
        with self.assertRaises(FileExistsError):
            self.page.export_path(path)
        self.assertEqual(saved, path.read_bytes())
        self.page.load_path(path)
        self.assertIsNone(self.page.locked_protocol)
        self.assertFalse(self.page.reviewed.isChecked())
        self.assertEqual(10, float(self.page.fields['frequency_hz'].text()))

    def test_rejected_import_preserves_current_form(self):
        self.fill_fixture()
        self.page.confirm_parameters()
        original = self.page.locked_protocol.processing_identity
        path = Path(self.temp.name) / 'bad.json'
        path.write_text('{}', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.page.load_path(path)
        self.assertEqual(original, self.page.locked_protocol.processing_identity)

    def test_static_preview_ignores_flicker_phase(self):
        self.fill_fixture()
        self.page.confirm_parameters()
        self.page.condition.setCurrentIndex(0)
        self.page.preview.resize(600, 240)
        self.page.frame_slider.setValue(0)
        on_image = self.page.preview.grab().toImage()
        self.page.frame_slider.setValue(3)
        off_image = self.page.preview.grab().toImage()
        self.assertEqual(on_image, off_image)

    def test_renderer_only_paints_target_not_half_screen(self):
        from PySide6.QtGui import QImage, QPainter
        from neurostation_condition_render import paint_condition_state
        from eeg_tools.workstation.four_condition_presenter import DisplayState
        from neurostation_four_condition import FourConditionProtocol
        from tests.test_ssvep_four_conditions import synthetic_value
        p = FourConditionProtocol.from_dict(synthetic_value())
        image = QImage(600, 400, QImage.Format.Format_ARGB32)
        painter = QPainter(image)
        paint_condition_state(painter, 600, 400, p.parameters,
                              DisplayState('stimulus', 'L_FIXED', frame=p.frame('L_FIXED', 0)))
        painter.end()
        self.assertEqual(255, image.pixelColor(120, 200).red())
        self.assertEqual(51, image.pixelColor(30, 100).red())
        self.assertEqual(51, image.pixelColor(480, 200).red())
        painter = QPainter(image)
        paint_condition_state(painter, 600, 400, p.parameters,
                              DisplayState('stimulus_offset', 'L_FIXED', frame=p.frame('L_FIXED', 240)))
        painter.end()
        self.assertEqual(51, image.pixelColor(120, 200).red())

    def test_circle_has_equal_pixel_dimensions(self):
        from neurostation_condition_render import target_pixel_rect
        from neurostation_four_condition import FourConditionProtocol
        from tests.test_ssvep_four_conditions import synthetic_value
        p = FourConditionProtocol.from_dict(synthetic_value())
        parameters = p.parameters
        parameters['shape'] = 'circle'
        _,_,w,h = target_pixel_rect(1920,1080,parameters,p.frame('L_FIXED',0))
        self.assertEqual(w,h)

    def test_page_and_labels_render_in_both_languages(self):
        from apps.workstation_ui.i18n import Translator
        from apps.workstation_ui.four_condition_page import FourConditionPage
        for locale in ('zh-CN','en-US'):
            page = FourConditionPage(Translator(locale), lambda key: None)
            page.resize(1000, 1300)
            self.assertFalse(page.grab().isNull())
            self.assertNotIn('four.title', page.title_label.text())
            page.close()
            page.deleteLater()


if __name__ == '__main__':
    unittest.main()

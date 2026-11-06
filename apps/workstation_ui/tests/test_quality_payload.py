"""Malformed live evidence must clear stale UI state without a Qt paint exception."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from copy import deepcopy
import time
import unittest
from PySide6.QtWidgets import QApplication
from apps.workstation_ui.quality_panel import QualityPanel


class QualityPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = QualityPanel()
        self.addCleanup(self.panel.close)
        self.addCleanup(self.panel.deleteLater)
        self.valid = {'schema_version': 2, 'stream_id': 's', 'processing_id': 'p',
                      'start': 0, 'stop': 2, 'updated_monotonic_s': time.monotonic(),
                      'dimensions': {'signal_quality': 'pass'},
                      'channels': [{'channel': 'CH1', 'quality_status': 'pass'}],
                      'raw_samples': [[1., 2.]], 'display_samples': [[0., 1.]]}

    def test_malformed_payload_clears_previously_good_view(self):
        for patch in ({'dimensions': None}, {'stop': None}, {'channels': [None]},
                      {'raw_samples': [['bad', 1]]}, {'artifacts': None},
                      {'psd': {'power_uv2_hz': [['bad']], 'frequency_hz': [1]}},
                      {'updated_monotonic_s': float('nan')}, {'watermarks': []}):
            with self.subTest(patch=patch):
                self.panel.update_quality(deepcopy(self.valid))
                self.panel.update_quality({**self.valid, **patch})
                self.assertIsNone(self.panel.latest)
                self.assertIn('UNKNOWN', self.panel.status.text())
                self.assertEqual(self.panel.plot.raw, [])

    def test_absent_payload_does_not_retain_previous_pass(self):
        self.panel.update_quality(deepcopy(self.valid))
        self.panel.update_quality(None)
        self.assertIsNone(self.panel.latest)
        self.assertIn('UNKNOWN', self.panel.status.text())

    def test_future_monotonic_timestamp_is_not_fresh(self):
        self.panel.update_quality({**self.valid, 'updated_monotonic_s': time.monotonic() + 100})
        self.assertIn('UNKNOWN', self.panel.status.text())

    def test_rendered_payload_is_a_copy_not_a_mutable_live_reference(self):
        result = deepcopy(self.valid)
        self.panel.update_quality(result)
        result['dimensions'] = None
        self.panel.set_context('live')
        self.assertIsNotNone(self.panel.latest['dimensions'])

"""UI-only fixtures carry no hardware-quality authority."""
import os
import time
import unittest
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from apps.workstation_ui.quality_panel import QualityPanel

class QualityPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel=QualityPanel();self.panel.timer.stop()

    def tearDown(self):
        self.panel.close();self.panel.deleteLater();self.app.processEvents()

    def record(self,start,stop,**overrides):
        return {'schema_version':2,'processing_id':'fixture','stream_id':'fixture-only',
                'start':start,'stop':stop,'updated_monotonic_s':time.monotonic(),
                'channels':[{'channel':'O1','quality_status':'unknown'}],
                'raw_samples':[[1.]*(stop-start)],'display_samples':[[2.]*(stop-start)],
                'dimensions':{'signal_quality':'unknown'}, **overrides}

    def test_skipped_poll_preserves_unknown_waveform_gap(self):
        self.panel.update_quality(self.record(0,5))
        self.panel.update_quality(self.record(8,10))
        self.assertEqual(list(self.panel.raw[0]),[1.]*5+[None]*3+[1.]*2)
        self.assertEqual(list(self.panel.display[0]),[2.]*5+[None]*3+[2.]*2)

    def test_new_identity_clears_tracks_and_stale_record_is_unknown(self):
        self.panel.update_quality(self.record(0,5))
        self.panel.update_quality(self.record(5,8,processing_id='new',updated_monotonic_s=time.monotonic()-3))
        self.assertEqual(len(self.panel.raw[0]),3)
        self.assertIn('UNKNOWN',self.panel.status.text())
        self.assertIn('unknown',self.panel.table.item(0,5).text())

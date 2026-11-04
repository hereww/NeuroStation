"""Operator workflows exercise production sinks; they never open a board."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from apps.workstation_ui.app import MainWindow
from apps.workstation_ui.i18n import Translator
from apps.workstation_ui.quality_panel import QualityPanel
from apps.workstation_ui.quality_smoke import run_quality_workflow_smoke
from eeg_tools.workstation.desktop_gateway import DesktopGateway
from neurostation_diagnostics import DiagnosticStore

ROOT = Path(__file__).resolve().parents[3]


class QualityWorkspaceUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.gateway = DesktopGateway(
            protocol_path=ROOT / 'configs/protocols/ssvep_four_target_v2.json',
            channel_config_path=ROOT / 'configs/channel_config_v1_auto.json',
            dataset_root=self.root / 'datasets')
        self.gateway.auto_import_default = False
        self.gateway.start_ssvep = Mock(side_effect=AssertionError('must not acquire'))
        self.gateway.preflight_cyton = Mock(side_effect=AssertionError('must not open a device'))
        self.window = MainWindow(self.gateway, timer_enabled=False,
                                 diagnostic_store=DiagnosticStore(persist=False))
        self.c = self.window.quality_controller
        self.c.output_root = self.root / 'quality'
        self.c.demo_options = {'seconds': 2, 'realtime': False}
        self.c.replay_options = {'realtime_preview': False}
        self.window.show()
        self.app.processEvents()

    def wait_idle(self, timeout=15):
        end = time.monotonic() + timeout
        while self.c.busy and time.monotonic() < end:
            self.app.processEvents(); time.sleep(.005)
        self.app.processEvents()
        self.assertFalse(self.c.busy)

    def tearDown(self):
        self.c.cancel()
        self.wait_idle()
        self.window.close(); self.window.deleteLater(); self.app.processEvents()
        self.temp.cleanup()

    def test_visible_entry_without_capture_or_extra_reader(self):
        self.window.pages['apps'].quality_button.click()
        self.assertEqual(self.window.current_page, 'quality')
        page = self.window.pages['quality']
        self.assertEqual(page.panel.table.rowCount(), 8)
        self.assertIsNone(page.panel.latest)
        self.assertTrue(page.demo_button.isEnabled())
        self.assertFalse(page.report_button.isEnabled())
        page.live_button.click()
        self.gateway.start_ssvep.assert_not_called()
        self.gateway.preflight_cyton.assert_not_called()

    def test_full_delivery_workflow_with_both_languages_and_export(self):
        result = run_quality_workflow_smoke(self.window, self.root / 'workflow')
        self.assertTrue(result['all_passed'], result)
        self.assertEqual(result['hardware_calls'], [])

    def test_stop_retains_aborted_prefix_and_late_results_do_not_become_live(self):
        self.c.demo_options = {'seconds': 30, 'realtime': True}
        self.c.start_demo()
        end = time.monotonic() + 3
        while self.c.latest is None and time.monotonic() < end:
            self.app.processEvents(); time.sleep(.01)
        self.assertIsNotNone(self.c.latest)
        self.assertFalse(self.c.use_live())
        self.c.cancel(); self.wait_idle()
        self.assertEqual(self.c.phase, 'cancelled')
        meta = json.loads((self.c.session / 'session.json').read_text())
        self.assertEqual(meta['recording_status'], 'aborted')
        self.c.use_live()
        self.app.processEvents()
        self.assertIsNone(self.c.latest)
        self.assertIsNone(self.window.pages['quality'].panel.latest)

    def test_language_switch_during_worker_keeps_controller_and_safe_shutdown(self):
        self.c.demo_options = {'seconds': 30, 'realtime': True}
        self.c.start_demo()
        self.window.change_language('en-US')
        self.app.processEvents()
        self.assertIs(self.window.quality_controller, self.c)
        self.assertIn('SYNTHETIC', self.window.pages['quality'].source.text())
        self.assertFalse(self.c.close())
        self.wait_idle()
        self.assertTrue(self.c.close())

    def test_capture_and_preview_mutually_exclusive(self):
        self.c.set_capture_busy(True)
        self.assertFalse(self.c.start_demo())
        self.c.set_capture_busy(False)
        self.c.demo_options = {'seconds': 30, 'realtime': True}
        self.c.start_demo()
        self.window.start_ssvep(self.gateway.config, 1)
        self.gateway.start_ssvep.assert_not_called()
        self.gateway.preflight_cyton.assert_not_called()
        with self.assertRaises(ValueError):
            self.window._offline_guarded_mutation(lambda: None)

    def test_legacy_missing_blocks_explained_but_report_still_works(self):
        source = self.root / 'old'
        source.mkdir()
        (source / 'session.json').write_text('{"status":"aborted","trial_count":2}')
        self.window.open_dataset_quality(source)
        self.assertEqual(self.window.current_page, 'quality')
        self.c.start_replay(); self.wait_idle()
        self.assertEqual(self.c.phase, 'error')
        self.assertIn('没有连续数据块', self.window.pages['quality'].status.text())
        self.c.start_report(); self.wait_idle()
        self.assertEqual(self.c.phase, 'completed')
        self.assertIn('旧会话', self.window.pages['quality'].report_view.toPlainText())

    def test_live_payload_uses_existing_reader_only_once(self):
        fixture = {'schema_version':2, 'stream_id':'x', 'processing_id':'p',
                   'start':0, 'stop':2, 'channels':[{'channel':'CH1','quality_status':'unknown'}],
                   'raw_samples':[[1,2]], 'display_samples':[[0,1]],
                   'updated_monotonic_s':time.monotonic(), 'dimensions':{'signal_quality':'unknown'}}
        self.gateway.read_live_waveform = Mock(return_value={'quality_v2':fixture,
            'samples':[[1,2]], 'channel_names':['CH1'], 'sample_rate_hz':250})
        self.window._read_live_waveform()
        self.gateway.read_live_waveform.assert_called_once()
        self.assertEqual(self.window.pages['quality'].panel.latest, fixture)

    def test_new_source_clears_old_report_and_quality(self):
        self.c.start_demo(); self.wait_idle()
        first = self.c.session
        self.c.start_report(); self.wait_idle()
        self.assertIsNotNone(self.c.report)
        self.c.start_demo(); self.wait_idle()
        self.assertNotEqual(first, self.c.session)
        self.assertIsNone(self.c.report)
        self.assertFalse(self.window.pages['quality'].export_button.isEnabled())

    def test_recorded_monotonic_timestamp_not_treated_as_live_freshness(self):
        panel = QualityPanel(tr=Translator())
        panel.set_context('replay', False)
        panel.update_quality({'schema_version':2,'stream_id':'x','processing_id':'p',
            'start':0,'stop':1,'updated_monotonic_s':0,
            'channels':[{'channel':'CH1','quality_status':'unknown'}],
            'raw_samples':[[1]],'display_samples':[[1]],'dimensions':{'signal_quality':'unknown'}})
        self.assertIn('静态记录', panel.status.text())
        self.assertNotIn('过期', panel.status.text())
        panel.set_context('live')
        self.assertIn('过期 UNKNOWN', panel.status.text())
        panel.close(); panel.deleteLater()

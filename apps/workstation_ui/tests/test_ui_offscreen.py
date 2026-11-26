"""Optional real-Qt smoke tests; automatically skipped when PySide6 is absent."""

import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "configs" / "protocols" / "ssvep_four_target_v2.json"
CHANNELS = ROOT / "configs" / "channel_config_v1_auto.json"


@unittest.skipUnless(HAS_QT, "PySide6 not installed; pure logic tests still run")
class QtOffscreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from apps.workstation_ui.app import MainWindow
        from eeg_tools.workstation.desktop_gateway import DesktopGateway
        from neurostation_diagnostics import DiagnosticStore

        self.temp = tempfile.TemporaryDirectory()
        self.gateway = DesktopGateway(
            protocol_path=PROTOCOL,
            channel_config_path=CHANNELS,
            dataset_root=Path(self.temp.name) / "Datasets",
        )
        self.start_calls = []
        self.diagnostic_store = DiagnosticStore(persist=False)
        self.window = MainWindow(
            self.gateway,
            timer_enabled=False,
            diagnostic_store=self.diagnostic_store,
        )

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()
        self.temp.cleanup()

    def _stub_start(self, config, speed=1):
        """Record routing without creating samples or a dataset."""

        from apps.workstation_ui.gateway import Phase, TaskSnapshot

        self.start_calls.append((config, speed))
        acquisition = self.gateway._acquisition
        acquisition.config = config
        acquisition._snapshot = TaskSnapshot(
            phase=Phase.COUNTDOWN,
            countdown=5,
            remaining=config.recording_seconds,
            trial_count=config.trials,
            speed=1,
        )
        self.gateway._active = acquisition
        return acquisition.snapshot

    def test_models_module_applies_selection_and_displays_actual_gpu(self):
        from dataclasses import replace
        from unittest.mock import patch
        from neurostation_models import BUILTIN, Model, model_by_id
        from apps.workstation_ui.gateway import CaptureMode

        form = self.window.pages['ssvep']
        imported = Model('a' * 64, 'Fixture model', 'v2', 'a' * 64, Path(self.temp.name))
        with patch('apps.workstation_ui.ssvep_models.list_models', return_value=[model_by_id(BUILTIN), imported]):
            self.window.navigate('models')
            models = self.window.pages['models']
            for key in (imported.id, BUILTIN, imported.id):
                models.combo.setCurrentIndex(models.combo.findData(key))
                models.use.click()
                self.assertEqual(self.window.current_page, 'ssvep')
                self.assertEqual(form.config().realtime_model_id, key)
                self.assertTrue(form.continuous.isChecked())
                self.assertTrue(form.save_raw.isChecked())
                self.assertEqual(form.gate_picker.isEnabled(), key == BUILTIN)
                self.window.navigate('models')
            config = replace(form.config(), mode=CaptureMode.CYTON)
            self._stub_start(config)
            actual = dict(state='no_target', end_sample=499, model=dict(name='Actual GPU model', version='v2', device='Fixture GPU'))
            with patch.object(self.gateway, 'read_live_ssvep', return_value=actual):
                self.window._pump_live_samples()
            self.assertIn('Actual GPU model · v2 · Fixture GPU', self.window.pages['task'].ssvep_prediction.text())

    def test_processing_opens_live_configuration_without_starting_capture(self):
        self.gateway.start_ssvep = self._stub_start
        self.window.navigate('processing')
        module = self.window.pages['processing']
        self.assertFalse(module.panel.baseline_button.isEnabled())
        self.assertFalse(module.panel.run_button.isEnabled())
        module.configure_button.click()
        self.assertEqual(self.window.current_page, 'ssvep')
        form = self.window.pages['ssvep']
        self.assertTrue(form.config().continuous_ssvep)
        self.assertFalse(form.acknowledge.isChecked())
        self.assertFalse(self.gateway.snapshot.active)
        self.assertEqual(self.start_calls, [])
        form.continuous_model.setText('old_model.json')
        form.continuous.setChecked(False)
        self.assertIsNone(form.config().continuous_ssvep_model)

    def test_capture_outputs_route_to_worker_and_filter_paired_sessions(self):
        from dataclasses import replace
        from neurostation_contract import Dataset, CaptureMode
        form = self.window.pages['ssvep']
        self.assertTrue(form.save_raw.isChecked())
        self.assertFalse(form.continuous.isChecked())
        acq = self.gateway._acquisition
        acq._status_file = Path(self.temp.name)/'status.json'
        acq._cancel_file = Path(self.temp.name)/'cancel'
        for raw, live in ((True, False), (False, True), (True, True), (False, False)):
            form.save_raw.setChecked(raw)
            form.continuous.setChecked(live)
            config = replace(form.config(), participant='fixture', eye_side='left', acknowledge_flicker_risk=True)
            if not (raw or live):
                with self.assertRaisesRegex(ValueError, 'validation.signal_outputs'):
                    config.validate()
                continue
            command = acq._command(config)
            self.assertEqual('--no-raw-eeg' in command, not raw)
            self.assertEqual('--continuous-ssvep' in command, live)
        self.window.navigate('datasets')
        page = self.window.pages['datasets']
        item = Dataset(id='fixture', name='fixture', participant='p', protocol='ssvep',
                       recording_seconds=3, preparation_seconds=0, demo_seconds=0, trials=1,
                       samples_per_channel=750, event_count=0, path=Path(self.temp.name), created_at='',
                       source=CaptureMode.CYTON)
        for selected, expected in [('raw_eeg', (True,False,True)), ('realtime_ssvep',(False,True,True)), ('both',(False,False,True))]:
            page.signal_filter.setCurrentIndex(page.signal_filter.findData(selected))
            self.assertEqual(self.window._dataset_signal, selected)
            for tags, matches in zip([('raw_eeg',), ('realtime_ssvep',), ('raw_eeg','realtime_ssvep')], expected):
                self.assertEqual(page._matches(replace(item,signal_types=tags), '', '', ''), matches)

    def test_live_decisions_reach_task_via_desktop_gateway(self):
        import json
        import time
        self.assertIsNone(self.gateway.read_live_ssvep())
        form = self.window.pages['ssvep']
        form.continuous.setChecked(True)
        self._stub_start(form.config())
        runtime = Path(self.temp.name)
        self.gateway._acquisition._runtime_dir = runtime
        try:
            for state, hz, expected in [('target', 12, '12 Hz'),
                                         ('no_target', '', self.window.tr('ssvep.no_target')),
                                         ('unavailable', '', self.window.tr('ssvep.unavailable'))]:
                row = dict(received_monotonic_s=time.monotonic(), state=state, frequency_hz=hz, end_sample=999)
                (runtime / 'ssvep-live.json').write_text(json.dumps(row))
                self.window._pump_live_samples()
                self.assertIn(expected, self.window.pages['task'].ssvep_prediction.text())
            from eeg_tools.live_ssvep import LiveSSVEP, Gate
            import numpy as np
            t = np.arange(1000) / 250
            eeg = np.random.default_rng(42).normal(0, .1, (8, len(t)))
            eeg[0, 500:] += np.sin(2 * np.pi * 12 * t[500:])
            sink = LiveSSVEP(runtime, range(2, 10), 0, 1, Gate(), runtime / 'ssvep-live.json')
            with sink.file:
                sink.consume(np.vstack((np.arange(len(t)) % 256, t, eeg)),
                             dict(start=0, stop=len(t), received_monotonic_s=time.monotonic() - 9))
            actual = json.loads((runtime / 'ssvep-live.json').read_text())
            self.assertEqual(actual['frequency_hz'], 12)
            self.window._pump_live_samples()
            self.assertIn(self.window.tr('ssvep.stale_input'), self.window.pages['task'].ssvep_prediction.text())
            self.assertNotIn('12 Hz', self.window.pages['task'].ssvep_prediction.text())
        finally:
            self.gateway._acquisition._runtime_dir = None
            self.gateway._active = self.gateway._metadata

    def test_processing_restores_report_for_selected_recording(self):
        from PySide6.QtWidgets import QDialog, QTextEdit
        from apps.workstation_ui.ssvep_replay import ReplayPanel
        from unittest.mock import patch

        session = Path(self.temp.name) / 'recording'
        output = session / 'derived' / 'continuous_complete'
        output.mkdir(parents=True)
        (session / 'session.json').write_text('{"session_id":"fixture"}')
        (output / 'metrics.json').write_text('{"test":"fixture","metrics":{"baseline":{}}}')
        (output / 'report.md').write_text('# Fixture report', encoding='utf-8')
        partial = output.with_name('continuous_partial')
        partial.mkdir()
        (partial / 'report.md').write_text('Incomplete')
        panel = ReplayPanel(self.window.tr, session, self.window.replay_controller)
        self.assertTrue(panel.baseline_button.isEnabled())
        self.assertEqual(self.window.replay_controller.state_for(session)['output_dir'], str(output))
        with patch.object(self.window.replay_controller, 'compare') as compare:
            panel.baseline_button.click()
            compare.assert_called_once_with(session)
            with patch('apps.workstation_ui.ssvep_replay.QFileDialog.getExistingDirectory', return_value='earlier'):
                panel.run_button.click()
            compare.assert_called_with(session, 'earlier')
        self.assertIn(self.window.tr('ssvep.legacy_report'), panel.status.text())
        panel.open_button.click()
        dialog = panel.findChild(QDialog)
        self.assertIn('Fixture report', dialog.findChild(QTextEdit).toPlainText())
        dialog.close()
        panel.deleteLater()

    def test_processing_exports_without_a_report_and_restores_output_folder(self):
        from apps.workstation_ui.ssvep_replay import ReplayPanel
        from unittest.mock import patch

        session = Path(self.temp.name) / 'recording-export'
        output = session / 'derived' / 'continuous_export'
        output.mkdir(parents=True)
        (session / 'session.json').write_text('{"session_id":"fixture"}')
        manifest = dict(kind='ssvep_signal_export', schema_version=1, session_id='fixture',
                        predictions={'baseline': 'baseline_predictions.tsv'}, annotations=[])
        (output / 'processing_manifest.json').write_text(json.dumps(manifest))
        for filename in ('baseline_predictions.tsv', 'frozen_models.json', 'protocol.json'):
            (output / filename).touch()
        controller = self.window.replay_controller
        panel = ReplayPanel(self.window.tr, session, controller)
        self.assertFalse(panel.folder_button.isHidden())
        self.assertTrue(panel.open_button.isHidden())
        with patch.object(controller, '_launch') as launch:
            panel.baseline_button.click()
            self.assertIn('--process-only', launch.call_args.args[2])
        with patch('apps.workstation_ui.ssvep_replay.QDesktopServices.openUrl') as open_folder:
            panel.folder_button.click()
            self.assertEqual(Path(open_folder.call_args.args[0].toLocalFile()), output)
        panel.deleteLater()

    def test_ssvep_default_protocol_and_estimate_in_both_languages(self):
        from PySide6.QtWidgets import QLabel

        for locale in ("zh-CN", "en-US"):
            with self.subTest(locale=locale):
                self.window.change_language(locale)
                page = self.window.pages["ssvep"]
                self.assertEqual(5, page.stimulus.value())
                self.assertEqual(5, page.rest.value())
                self.assertEqual(10, page.repetitions.value())
                self.assertIn("00:06:40", page.estimate.text())
                self.assertIn("00:06:35", page.estimate.text())
                self.assertIn("40", page.estimate.text())
                expected = self.window.tr(
                    "ssvep.defaults", stimulus=5, rest=5, repetitions=10, refresh=60,
                )
                self.assertIn(expected, [label.text() for label in page.findChildren(QLabel)])
                self.assertFalse(page.acknowledge.isChecked())
                self.assertFalse(page.allow_draft.isChecked())

    def test_ssvep_parameter_edits_update_estimate_and_survive_language_change(self):
        page = self.window.pages["ssvep"]
        page.rest.setValue(3)
        page.repetitions.setValue(3)
        self.assertIn("00:01:38", page.estimate.text())
        self.assertIn("00:01:33", page.estimate.text())
        self.window.change_language("en-US")
        page = self.window.pages["ssvep"]
        self.assertEqual(3, page.rest.value())
        self.assertEqual(3, page.repetitions.value())
        self.assertIn("00:01:38", page.estimate.text())

    def test_formal_window_title_has_no_branch_badge(self):
        for locale in ("zh-CN", "en-US"):
            self.window.change_language(locale)
            self.assertEqual(self.window.tr("app.title"), self.window.windowTitle())
            self.assertNotIn("Quality Upgrade", self.window.windowTitle())
            self.assertNotIn("未合并", self.window.windowTitle())

    def test_four_condition_navigation_and_language_preserves_edits(self):
        self.window.pages["ssvep"].four_condition_button.click()
        self.assertEqual("ssvep_four_condition", self.window.current_page)
        page = self.window.pages["ssvep_four_condition"]
        page.fields["frequency_hz"].setText("12")
        page.operator.setText("synthetic operator")
        self.window.change_language("en-US")
        page = self.window.pages["ssvep_four_condition"]
        self.assertEqual("ssvep_four_condition", self.window.current_page)
        self.assertEqual("12", page.fields["frequency_hz"].text())
        self.assertEqual("synthetic operator", page.operator.text())
        self.assertFalse(page.reviewed.isChecked())
        self.assertFalse(page.start_button.isEnabled())
        self.assertEqual([], self.start_calls)

    def _add_user(self, user_id="U0001", name="Test user"):
        from neurostation_contract import UserProfile

        self.gateway.add_user(UserProfile(
            user_id=user_id,
            name=name,
            age=30,
            medical_conditions=("none",),
        ))
        self.window.refresh_users()

    def test_navigation_language_and_static_render(self):
        from apps.workstation_ui.app import NAVIGATION

        self.assertNotIn("live", NAVIGATION)
        self.assertNotIn("live", self.window.pages)
        self.window.show()
        for page in NAVIGATION:
            self.window.navigate(page)
            self.application.processEvents()
            self.assertEqual(self.window.current_page, page)
        self.window.change_language("en-US")
        self.assertEqual(self.window.tr("nav.apps"), "Acquisition apps")
        self.assertFalse(self.window.grab().isNull())

    def test_ssvep_form_is_cyton_only_and_runs_at_real_time(self):
        from apps.workstation_ui.gateway import CaptureMode

        page = self.window.pages["ssvep"]
        self.assertEqual(1, page.mode.count())
        self.assertEqual(CaptureMode.CYTON, page.mode.currentData())
        self.assertFalse(hasattr(page, "speed"))
        self.assertEqual(CaptureMode.CYTON, page.config().mode)
        self.assertTrue(page.port.isEnabled())
        self.assertFalse(page.channel_auto_label.isHidden())

        page.channel_manual.setChecked(True)
        channel_path = ROOT / "configs" / "channel_config_v1_auto.json"
        page.channel.setText(str(channel_path))
        self.assertEqual(channel_path, page.config().channel_config)

    def test_channel_guide_follows_head_map_selection_and_electrode_edits(self):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        self.window.navigate("devices")
        self.window.show()
        self.application.processEvents()
        page = self.window.pages["devices"]
        guide = page.channel_guide
        self.assertEqual(
            [f"CH{index} · {site}" for index, site in enumerate(page.DEFAULT_CHANNEL_POSITIONS, 1)],
            [channel.text() for channel in guide.channel_labels],
        )
        self.assertTrue(guide.channel_labels[0].property("selected"))
        x, y = page.head_map._site_points()["Fp2"]
        QTest.mouseClick(page.head_map, Qt.MouseButton.LeftButton, pos=QPoint(round(x), round(y)))
        self.assertEqual(1, page.calibration_channel_selector.currentIndex())
        self.assertTrue(guide.channel_labels[1].property("selected"))
        self.assertFalse(guide.channel_labels[0].property("selected"))
        page.position_boxes[1].setCurrentText(" oz ")
        self.assertEqual("CH2 · Oz", guide.channel_labels[1].text())
        self.assertEqual(self.window.tr("device.channel_guide.Oz"), guide.descriptions[1].text())
        page.position_boxes[1].setCurrentText("custom site")
        self.assertIn("custom site", guide.descriptions[1].text())
        page.position_boxes[1].setCurrentText("")
        self.assertEqual(self.window.tr("device.channel_guide.unassigned_detail"), guide.descriptions[1].text())

    def test_channel_guide_reflows_with_viewport_in_both_languages(self):
        from time import monotonic

        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QLabel

        self.window.navigate("devices")
        self.window.show()
        for locale, title in (("zh-CN", "通道解读"), ("en-US", "Channel guide")):
            with self.subTest(locale=locale):
                self.window.change_language(locale)
                page = self.window.pages["devices"]
                guide = page.channel_guide
                signal_panel = page.calibration_signal.parentWidget()
                map_panel = guide.parentWidget()
                self.assertEqual(title, guide.findChild(QLabel, "sectionTitle").text())
                for site in page.ELECTRODE_POSITIONS:
                    key = f"device.channel_guide.{site}"
                    self.assertIn(key, self.window.tr.catalogs[locale])
                def is_before(first, second, horizontal):
                    a, b = first.geometry(), second.geometry()
                    if not first.isVisible() or not second.isVisible() or a.isEmpty() or b.isEmpty():
                        return False
                    return a.right() < b.left() if horizontal else a.bottom() < b.top()

                for width, height, map_horizontal, overview_horizontal in (
                    (1800, 1000, True, True),
                    (1180, 850, True, False),
                    (750, 850, False, False),
                    (1800, 1000, True, True),
                ):
                    with self.subTest(width=width):
                        self.window.resize(width, height)
                        # Resizing the viewport queues reflow and Qt's geometry update.
                        # Wait for the geometry itself on slower offscreen runners.
                        deadline = monotonic() + 2
                        while monotonic() < deadline and not (
                            is_before(page.head_map, guide, map_horizontal)
                            and is_before(map_panel, signal_panel, overview_horizontal)
                        ):
                            QTest.qWait(10)
                        viewport_width = self.window.screens["devices"].viewport().width()
                        self.assertTrue(
                            is_before(page.head_map, guide, map_horizontal),
                            f"Viewport {viewport_width}: {page.head_map.geometry()} and {guide.geometry()}",
                        )
                        self.assertTrue(
                            is_before(map_panel, signal_panel, overview_horizontal),
                            f"Viewport {viewport_width}: {map_panel.geometry()} and {signal_panel.geometry()}",
                        )

    def test_device_page_has_one_test_action_per_channel(self):
        self.window.navigate("devices")
        page = self.window.pages["devices"]
        self.assertEqual(8, page.calibration_table.rowCount())
        self.assertEqual(8, len(page.test_buttons))
        self.assertEqual(8, page.calibration_channel_selector.count())
        self.assertIsNotNone(page.head_map)
        self.assertIsNotNone(page.calibration_signal)
        self.assertEqual("开始脑电测试", page.calibration_test_toggle.text())
        self.assertEqual(
            ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"),
            tuple(box.currentText() for box in page.position_boxes),
        )
        self.assertEqual("ear_clip", page.reference_position.currentText())
        self.assertEqual("ear_clip", page.bias_position.currentText())
        self.assertEqual("board_agnd", page.ground_position.currentText())
        self.assertTrue(all(button.isEnabled() for button in page.test_buttons))

        page._select_calibration_channel(2)
        page._assign_map_position("C3")
        self.assertEqual("C3", page.position_boxes[2].currentText())
        page.append_channel_calibration_samples({
            "channel": 3,
            "channels": [[1.0], [2.0], [3.0, 4.0], [4.0], [5.0], [6.0], [7.0], [8.0]],
        })
        self.assertEqual(2, page.calibration_signal.sample_count)
        self.assertEqual(1.0, page.calibration_signal.variation)

        started = []
        stopped = []
        page.calibration_preview_requested.disconnect()
        page.calibration_test_stop_requested.disconnect()
        page.calibration_preview_requested.connect(started.append)
        page.calibration_test_stop_requested.connect(lambda: stopped.append(True))
        page._select_calibration_channel(2)
        page.calibration_test_toggle.click()
        self.assertEqual([2], started)

        page.set_channel_test_busy(2, True)
        self.assertEqual("停止脑电测试", page.calibration_test_toggle.text())
        page.calibration_test_toggle.click()
        self.assertEqual([True], stopped)
        self.assertTrue(all(not button.isEnabled() for button in page.test_buttons))
        page.set_channel_test_busy(-1, False)
        self.assertEqual("开始脑电测试", page.calibration_test_toggle.text())
        self.assertTrue(all(button.isEnabled() for button in page.test_buttons))

        page.set_channel_test_result(2, {
            "status": "passed",
            "detail": "ok",
            "metrics": {
                "finite_fraction": 1.0,
                "flat_fraction": 0.0,
                "saturation_fraction": 0.0,
            },
        })
        self.assertIn("通过", page.test_statuses[2].text())

    def test_row_assessment_then_close_during_continuous_preview(self):
        import threading
        import time

        entered_preview = threading.Event()
        calls = []
        def test_channel(channel, seconds, port, sample_callback, cancel_event):
            calls.append((channel, seconds))
            if seconds is None:
                entered_preview.set()
                cancel_event.wait(2)
                return {"channel": channel, "status": "cancelled"}
            return {"channel": channel, "status": "passed", "selected_port": "COM5", "metrics": {
                "finite_fraction": 1.0, "flat_fraction": 0.0, "saturation_fraction": 0.0}}

        def wait_for_idle():
            deadline = time.monotonic() + 3
            while self.window.channel_calibration_thread is not None and time.monotonic() < deadline:
                self.application.processEvents()
                time.sleep(.005)
            self.assertIsNone(self.window.channel_calibration_thread)

        self.gateway.test_cyton_channel = test_channel
        self.window.show()
        self.window.navigate("devices")
        page = self.window.pages["devices"]
        page.test_buttons[2].click()
        wait_for_idle()
        self.assertEqual(page._test_results[2]["status"], "passed")
        page.calibration_test_toggle.click()
        self.assertTrue(entered_preview.wait(1))
        self.assertFalse(self.window.language_selector.isEnabled())
        self.assertFalse(self.window.close())
        wait_for_idle()
        self.assertFalse(self.window.isVisible())
        self.assertEqual(calls, [(3, 3.0), (3, None)])
        self.assertEqual(page._test_results[2]["status"], "passed")

    def test_editing_calibration_invalidates_affected_results(self):
        page = self.window.pages["devices"]
        result = {"status": "passed", "selected_port": "COM5"}
        page.set_channel_test_result(0, dict(result, channel=1))
        page.set_channel_test_result(1, dict(result, channel=2))
        page.position_boxes[1].setCurrentText("Cz")
        self.assertIsNotNone(page._test_results[0])
        self.assertIsNone(page._test_results[1])
        page.calibration_port.setText("COM6")
        self.assertEqual(page._test_results, [None] * 8)

    def test_device_connection_status_updates_after_cyton_handshake(self):
        from dataclasses import replace

        self.window.navigate("devices")
        page = self.window.pages["devices"]
        self.assertIn("未连接", page.device_status.text())

        acquisition = self.gateway._acquisition
        acquisition.device = replace(acquisition.device, connected=True, port="COM7")
        self.window._update_controls()

        self.assertIn("已连接", page.device_status.text())
        self.assertIn("COM7", page.device_status.text())

    def test_capture_form_requires_a_real_user_and_photosensitivity_ack(self):
        from apps.workstation_ui.gateway import CaptureMode
        from PySide6.QtWidgets import QApplication
        from neurostation_display import resolve_eye_regions

        page = self.window.pages["ssvep"]
        self.assertEqual("", page.eye_side.currentData())
        self.assertEqual(self.window.tr("eye.left"), page.eye_side.itemText(page.eye_side.findData("left")))
        self.assertEqual(self.window.tr("eye.right"), page.eye_side.itemText(page.eye_side.findData("right")))
        display = resolve_eye_regions(QApplication.instance())["left"]
        screen_label = f"{display['screen_index']} · {display['screen'].name()}"
        self.assertEqual(
            self.window.tr("ssvep.screen_mapping", screen=screen_label),
            page.screen_mapping.text(),
        )
        self.assertIn("选择眼别", page.dataset_preview.text())
        page._start()
        self.assertTrue(page.error.text())
        self._add_user()
        page.acknowledge.setChecked(True)
        page._start()
        self.assertEqual(self.window.tr("validation.eye_side"), page.error.text())
        page.eye_side.setCurrentIndex(page.eye_side.findData("left"))
        self.assertIn("_左眼", page.dataset_preview.text())
        self.assertEqual("left", page.stimulus_preview.eye_side)
        config = page.config()
        self.assertEqual(CaptureMode.CYTON, config.mode)
        config.validate()

        page.start_requested.disconnect(self.window.start_ssvep)
        requested = []
        page.start_requested.connect(lambda candidate, speed: requested.append(candidate.eye_side))
        page._start()
        page.eye_side.setCurrentIndex(page.eye_side.findData("right"))
        self.assertEqual("right", page.stimulus_preview.eye_side)
        page._start()
        self.assertEqual(["left", "right"], requested)

    def test_static_stimulus_preview_tracks_eye_side_and_center_divider(self):
        page = self.window.pages["ssvep"]
        display = page.stimulus_preview.display
        display.resize(400, 140)
        display.show()
        self.application.processEvents()

        page.stimulus_preview.set_eye_side("left")
        image = display.grab().toImage()
        midpoint = image.width() // 2
        self.assertEqual((245, 245, 245), image.pixelColor(image.width() // 4, 20).getRgb()[:3])
        self.assertEqual((0, 0, 0), image.pixelColor(image.width() * 3 // 4, 20).getRgb()[:3])
        self.assertEqual((255, 255, 255), image.pixelColor(midpoint, 20).getRgb()[:3])

        page.stimulus_preview.set_eye_side("right")
        image = display.grab().toImage()
        self.assertEqual((0, 0, 0), image.pixelColor(image.width() // 4, 20).getRgb()[:3])
        self.assertEqual((245, 245, 245), image.pixelColor(image.width() * 3 // 4, 20).getRgb()[:3])
        self.assertEqual((255, 255, 255), image.pixelColor(image.width() // 2, 20).getRgb()[:3])

    def test_openbci_waveform_model_uses_rolling_buffers_and_display_copy(self):
        from apps.workstation_ui.components import WaveformDisplayModel

        model = WaveformDisplayModel()
        self.assertEqual(
            ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"),
            model.CHANNEL_NAMES,
        )
        self.assertEqual(8, model.CHANNEL_COUNT)
        self.assertEqual(10, model.SAMPLES_PER_UPDATE)
        self.assertEqual(1250, model.DISPLAY_SAMPLES)
        self.assertEqual(5500, model.RAW_CAPACITY)
        self.assertEqual(200, model.Y_LIMIT_UV)

        source = [[float(sample + channel) for sample in range(6000)] for channel in range(8)]
        original_first_channel = list(source[0])
        model.append(source)

        self.assertEqual(6000, model.sample_index)
        self.assertTrue(all(len(channel) == model.RAW_CAPACITY for channel in model.raw_buffers))
        self.assertTrue(all(len(channel) == model.DISPLAY_SAMPLES for channel in model.filtered_buffers))
        self.assertEqual(original_first_channel, source[0])
        self.assertEqual(tuple(source[0][-model.RAW_CAPACITY:]), tuple(model.raw_buffers[0]))
        self.assertEqual(model.DISPLAY_SAMPLES, len(model.visible_samples(0)))
        self.assertNotEqual(tuple(source[0][-model.DISPLAY_SAMPLES:]), model.visible_samples(0))

    def test_openbci_display_filter_has_requested_bandpass_and_notches(self):
        from apps.workstation_ui.components import _OpenBCIDisplayFilter

        sample_rate = 250
        sample_count = 5_000

        def gain(frequency_hz: float) -> float:
            display_filter = _OpenBCIDisplayFilter(sample_rate)
            output = []
            for index in range(sample_count):
                value = math.sin(2.0 * math.pi * frequency_hz * index / sample_rate)
                output.extend(display_filter.process([value]))
            tail = output[-1_000:]
            return math.sqrt(2.0 * sum(value * value for value in tail) / len(tail))

        self.assertAlmostEqual(0.707, gain(5.0), delta=0.03)
        self.assertGreater(gain(10.0), 0.95)
        self.assertLess(gain(50.0), 0.02)
        self.assertLess(gain(60.0), 0.02)
        self.assertLess(gain(100.0), 0.02)

    def test_cyton_start_runs_preflight_before_routing_to_worker(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase

        self._add_user()
        self.gateway.preflight_cyton = lambda port="AUTO": {
            "status": "passed",
            "selected_port": port,
            "checks": [{"name": "channels", "status": "passed", "metrics": {
                "channels": [{"channel": i, "status": "passed"} for i in range(1, 9)],
            }}],
        }
        self.gateway.start_ssvep = self._stub_start
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM5",
            user_id="U0001",
            participant="U0001",
            name="real hardware routing",
            eye_side="left",
            acknowledge_flicker_risk=True,
        )
        self.window.start_ssvep(config, 1)
        for _ in range(50):
            self.application.processEvents()
            if self.window.preflight_thread is None:
                break
        self.assertEqual(1, len(self.start_calls))
        self.assertEqual("COM5", self.start_calls[0][0].port)
        self.assertEqual(1, self.start_calls[0][1])
        self.assertEqual(Phase.COUNTDOWN, self.gateway.snapshot.phase)
        self.assertIn("8/8", self.window.pages["task"].channel_check_status.text())
        self.assertFalse(self.window.pages["task"].channel_check_status.isHidden())
        self.assertFalse(self.window.pages["quality"].check_button.isEnabled())

    def test_cyton_failed_preflight_blocks_worker_routing(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode

        self._add_user()
        errors = []
        self.window._error = errors.append
        self.gateway.start_ssvep = self._stub_start
        self.gateway.preflight_cyton = lambda port="AUTO": {
            "status": "failed",
            "error": "no device",
            "checks": [],
        }
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM9",
            user_id="U0001",
            participant="U0001",
            name="blocked hardware",
            eye_side="left",
            acknowledge_flicker_risk=True,
        )
        self.window.start_ssvep(config, 1)
        for _ in range(50):
            self.application.processEvents()
            if self.window.preflight_thread is None:
                break
        self.assertEqual([], self.start_calls)
        self.assertEqual(1, len(errors))

    def test_cyton_preflight_timestamp_jitter_does_not_show_repeated_warning(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase

        self._add_user()
        self.gateway.start_ssvep = self._stub_start
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM5",
            user_id="U0001",
            participant="U0001",
            name="warning hardware",
            eye_side="left",
            acknowledge_flicker_risk=True,
            allow_draft_hardware_config=False,
        )
        self.window._pending_ssvep = (config, 1)
        self.window._preflight_finished({
            "status": "warning",
            "error": "",
            "selected_port": "COM5",
            "checks": [{"name": "timestamps", "status": "warning", "metrics": {
                "timestamp_gap_count": 22,
                "timestamp_gap_ratio": 0.031,
                "timestamp_diff_max_s": 0.3546,
                "packet_sequence_available": True,
                "packet_loss_count": 0,
                "packet_sequence_mismatch_count": 0,
                "packet_duplicate_count": 0,
            }}],
        })
        self.assertEqual(Phase.COUNTDOWN, self.gateway.snapshot.phase)
        self.assertEqual("", self.window.pages["ssvep"].error.text())
        self.assertTrue(self.window.pages["task"].quality_notice.isHidden())

    def test_cyton_preflight_packet_warning_is_visible_when_worker_continues(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase

        self._add_user()
        self.gateway.start_ssvep = self._stub_start
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM5",
            user_id="U0001",
            participant="U0001",
            name="packet warning hardware",
            eye_side="left",
            acknowledge_flicker_risk=True,
            allow_draft_hardware_config=False,
        )
        self.window._pending_ssvep = (config, 1)
        self.window._preflight_finished({
            "status": "warning",
            "error": "",
            "selected_port": "COM5",
            "checks": [
                {"name": "timestamps", "status": "warning", "metrics": {
                    "timestamp_gap_count": 22,
                    "timestamp_gap_ratio": 0.031,
                    "timestamp_diff_max_s": 0.3546,
                    "packet_sequence_available": True,
                    "packet_loss_count": 1,
                    "packet_sequence_mismatch_count": 1,
                    "packet_duplicate_count": 0,
                }},
                {"name": "packet_sequence", "status": "warning", "metrics": {
                    "packet_sequence_available": True,
                    "packet_loss_count": 1,
                    "packet_sequence_mismatch_count": 1,
                    "packet_duplicate_count": 0,
                }},
            ],
        })
        self.assertEqual(Phase.COUNTDOWN, self.gateway.snapshot.phase)
        self.assertIn("继续", self.window.pages["ssvep"].error.text())
        self.assertFalse(self.window.pages["task"].quality_notice.isHidden())

    def test_cyton_preflight_degraded_requires_review(self):
        from apps.workstation_ui.gateway import CaptureConfig, CaptureMode, Phase

        self._add_user()
        config = CaptureConfig(
            mode=CaptureMode.CYTON,
            port="COM5",
            user_id="U0001",
            participant="U0001",
            name="degraded hardware",
            eye_side="left",
            acknowledge_flicker_risk=True,
        )
        self.window._pending_ssvep = (config, 1)
        self.window._preflight_finished({
            "status": "degraded",
            "error": "",
            "selected_port": "COM5",
            "checks": [{"name": "timestamps", "status": "degraded", "detail": "large gap"}],
        })
        self.assertFalse(self.gateway.snapshot.active)
        self.assertNotEqual(Phase.COUNTDOWN, self.gateway.snapshot.phase)
        self.assertIn("复核", self.window.pages["ssvep"].error.text())

    def test_task_page_exposes_current_trial_banner(self):
        from apps.workstation_ui.gateway import CaptureConfig, Phase, TaskSnapshot

        config = CaptureConfig(
            participant="U0001",
            user_id="U0001",
            name="task view",
            eye_side="left",
            acknowledge_flicker_risk=True,
        )
        self.window.pages["task"].update_snapshot(
            TaskSnapshot(
                phase=Phase.RUNNING,
                trial=2,
                trial_count=12,
                frequency=12,
                elapsed=8,
                remaining=85,
                progress=8.6,
            ),
            config,
        )
        self.assertIn("/ 12", self.window.pages["task"].trial_banner.text())

    def test_diagnostics_page_navigates_refreshes_and_shows_events(self):
        self.diagnostic_store.warning(
            "test",
            "diagnostic event visible",
            context={"attempt": 1},
        )
        self.window.navigate("diagnostics")
        page = self.window.pages["diagnostics"]

        self.assertGreater(page.checks.rowCount(), 0)
        self.assertIn("diagnostic event visible", page.events.toPlainText())
        self.assertIn("test", page.events.toPlainText())

        self.diagnostic_store.error("test", "second diagnostic event")
        self.window._refresh_diagnostics()
        self.assertIn("second diagnostic event", page.events.toPlainText())

        self.window.change_language("en-US")
        self.window.navigate("diagnostics")
        self.assertEqual("Diagnostics", self.window.pages["diagnostics"].title_label.text())

    def test_language_storage_and_window_geometry_persist(self):
        from PySide6.QtCore import QSettings
        from apps.workstation_ui.app import MainWindow
        from eeg_tools.workstation.desktop_gateway import DesktopGateway

        settings_path = Path(self.temp.name) / "settings.ini"
        dataset_path = Path(self.temp.name) / "persistent-datasets"
        settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        first_gateway = DesktopGateway(
            protocol_path=PROTOCOL,
            channel_config_path=CHANNELS,
            dataset_root=Path(self.temp.name) / "first-datasets",
        )
        first = MainWindow(
            first_gateway,
            locale="zh-CN",
            timer_enabled=False,
            persist_settings=True,
            settings=settings,
        )
        first.change_language("en-US")
        first.pages["ssvep"].save.setText(str(dataset_path))
        first.resize(780, 650)
        first.close()
        settings.sync()

        second_gateway = DesktopGateway(
            protocol_path=PROTOCOL,
            channel_config_path=CHANNELS,
            dataset_root=Path(self.temp.name) / "second-datasets",
        )
        second = MainWindow(
            second_gateway,
            locale=None,
            timer_enabled=False,
            persist_settings=True,
            settings=QSettings(str(settings_path), QSettings.Format.IniFormat),
        )
        self.assertEqual("en-US", second.tr.locale)
        self.assertEqual(dataset_path, second.draft_config.save_directory)
        self.assertEqual((780, 650), (second.width(), second.height()))
        second.close()

    def test_dataset_filters_persist_when_page_rebuilt(self):
        self.window.navigate("datasets")
        page = self.window.pages["datasets"]
        page.search.setText("U0001")
        page.source_filter.setCurrentIndex(page.source_filter.findData("cyton"))
        page.status_filter.setCurrentIndex(page.status_filter.findData("completed"))
        self.window.navigate("apps")
        self.window.navigate("datasets")
        page = self.window.pages["datasets"]
        self.assertEqual("U0001", page.search.text())
        self.assertEqual("cyton", page.source_filter.currentData())
        self.assertEqual("completed", page.status_filter.currentData())

    def test_narrow_window_keeps_sidebar_compact_and_content_scrollable(self):
        self.window.show()
        self.application.processEvents()
        self.window.resize(320, 560)
        self.application.processEvents()
        self.assertEqual(68, self.window.sidebar.width())
        self.assertFalse(self.window.sidebar_summary.isVisible())
        self.assertFalse(self.window.platform_label.isVisible())
        self.assertGreaterEqual(self.window.minimumWidth(), 320)
        self.window.resize(1024, 640)
        self.application.processEvents()
        self.assertEqual(186, self.window.sidebar.width())
        self.assertTrue(self.window.sidebar_summary.isVisible())


if __name__ == "__main__":
    unittest.main()

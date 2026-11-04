"""Offline UI contract tests with synthetic results and fake QProcess only."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QObject, Signal, QProcess
from PySide6.QtWidgets import QApplication, QWidget, QLabel
from apps.workstation_ui.i18n import Translator
from apps.workstation_ui.v9_offline import V9OfflineController, V9OfflinePanel, V9ComparisonWaveform

ROOT = Path(__file__).resolve().parents[3]


class FakeProcess(QObject):
    readyReadStandardOutput = Signal()
    readyReadStandardError = Signal()
    errorOccurred = Signal(object)
    finished = Signal(int, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.stdout = self.stderr = b''
        self.running = False
        self.terminate_calls = self.kill_calls = 0
        self.waits = []
        self.fail_start = False
        self.stop_on_kill = False

    def setProcessEnvironment(self, environment): self.environment = environment
    def setProcessChannelMode(self, mode): self.mode = mode
    def setWorkingDirectory(self, cwd): self.cwd = cwd
    def setProgram(self, program): self.program = program
    def setArguments(self, args): self.args = args
    def errorString(self): return 'fake Python launch failure'
    def state(self): return QProcess.ProcessState.Running if self.running else QProcess.ProcessState.NotRunning
    def start(self):
        self.running = not self.fail_start
        if self.fail_start:
            self.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
    def terminate(self): self.terminate_calls += 1
    def kill(self):
        self.kill_calls += 1
        if self.stop_on_kill:
            self.exit(-1, QProcess.ExitStatus.CrashExit)
    def waitForFinished(self, ms):
        self.waits.append(ms)
        return not self.running
    def readAllStandardOutput(self):
        data, self.stdout = self.stdout, b''
        return data
    def readAllStandardError(self):
        data, self.stderr = self.stderr, b''
        return data
    def push_stdout(self, data):
        self.stdout += data
        self.readyReadStandardOutput.emit()
    def push_stderr(self, data):
        self.stderr += data
        self.readyReadStandardError.emit()
    def exit(self, code=0, status=QProcess.ExitStatus.NormalExit):
        self.running = False
        self.finished.emit(code, status)


class FakeBackend:
    def __init__(self, root):
        self.root = root
        self.python = root / 'python.exe'
        self.python.write_bytes(b'fake interpreter; never executed')
        self.results = {}
        self.reads = []
        self.commands = []
        self.eligible = True
        self.read_error = None
        self.serial = 0
    def eligible_native_session(self, source): return self.eligible, 'invalid synthetic session' if not self.eligible else ''
    def default_ml_python(self): return self.python
    def new_output_dir(self, source):
        self.serial += 1
        return self.root / f'output_{self.serial}'
    def build_command(self, source, output, *, device, python):
        self.commands.append((source, output, device, python))
        return [str(python), '-u', 'offline.py', '--device', device, '--acknowledge-out-of-domain'], str(self.root)
    def find_latest_result(self, source):
        return next((output for output, result in reversed(list(self.results.items()))
                     if result['source_session'] == str(source)), None)
    def read_result(self, output, session_dir=None):
        self.reads.append((output, session_dir))
        if self.read_error:
            raise ValueError(self.read_error)
        return self.results[output]


def synthetic_result(source):
    return {'status': 'formal', 'completion_status': 'complete', 'source_session': str(source.resolve()),
        'source_domain': 'personal_Cyton', 'source_domain_validated': False, 'production_validated': False,
        'gpu': {'device': 'cuda:0', 'device_name': 'fake device'},
        'processing': {'processed_samples': 512, 'preserved_samples': 1, 'windows': {'processed': 2}},
        'comparison': {'tracks': {name: {'trial_count': 1, 'presented_match_count': 1,
            'gaze_truth_count': 0, 'available': True} for name in V9ComparisonWaveform.TRACKS}},
        'preview': {'sample_rate_hz': 250, 'start_sample': 0, 'time_s': [0., .004, .008],
            'channel_labels': [f'CH{i}' for i in range(1,9)],
            'raw': [[100001., 100003., 100002.]] * 8,
            'traditional': [[-.5, .5, 0.]] * 8, 'v9': [[100001.5, 100002.5, 100002.]] * 8},
        'trials': [{'trial_index': 1, 'presented_target_id': 'T1', 'gaze_target_id': '',
            'predictions': {name: {'target_id': 'T1', 'confidence': .7} for name in V9ComparisonWaveform.TRACKS}}]}


def synthetic_personal_decoder():
    return {
        'schema_version': 1, 'status': 'formal', 'method': 'personal_ssvep_cca_v1',
        'available': True, 'unavailable_reason': None, 'source_domain': 'personal_Cyton',
        'source_domain_validated': False, 'production_validated': False,
        'independent_gaze_truth_available': False, 'confidence_is_probability': False,
        'detection_validated': False, 'development_data': True,
        'tracks': {name: {'available': True, 'trial_count': 2, 'recorded_trial_count': 3,
            'presented_match_count': 1, 'presented_comparable_count': 1, 'reason': None}
            for name in V9ComparisonWaveform.TRACKS},
        'trials': [{'trial_index': 7, 'presented_target_id': 'T2', 'gaze_target_id': '',
            'window_samples': 375,
            'predictions': {name: {'target_id': 'T2', 'confidence': .12345,
                'available': True, 'scores': {'T1': .1, 'T2': .9}}
                for name in V9ComparisonWaveform.TRACKS},
            'prediction_status': {name: {'available': True}
                for name in V9ComparisonWaveform.TRACKS}}],
    }


class V9OfflineUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'session_a'
        self.other = self.root / 'session_b'
        self.source.mkdir()
        self.other.mkdir()
        self.owner = QWidget()
        self.backend = FakeBackend(self.root)
        self.processes = []
        def factory(parent):
            process = FakeProcess(parent)
            self.processes.append(process)
            return process
        self.controller = V9OfflineController(self.owner, backend=self.backend, process_factory=factory)
        self.tr = Translator()
        self.panels = []
    def tearDown(self):
        for process in self.processes:
            process.stop_on_kill = True
        self.controller.close()
        self.owner.deleteLater()
        for panel in self.panels:
            panel.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()
    def panel(self, source=None, **kwargs):
        panel = V9OfflinePanel(self.tr, source or self.source, self.controller, **kwargs)
        self.panels.append(panel)
        return panel
    def start(self):
        self.assertTrue(self.controller.start(self.source, 'cuda:0'))
        return self.processes[-1]
    def complete(self, process):
        self.backend.results[self.controller.output_dir] = synthetic_result(self.source)
        process.exit()

    def test_launch_is_separate_process_with_explicit_gpu_and_ack(self):
        process = self.start()
        self.assertEqual(str(self.backend.python), process.program)
        self.assertIn('--acknowledge-out-of-domain', process.args)
        self.assertEqual('cuda:0', process.args[-2])
        self.assertEqual(QProcess.ProcessChannelMode.SeparateChannels, process.mode)
        self.assertEqual([], process.waits)
    def test_worker_utf8_environment_preserves_system_without_global_changes(self):
        before = dict(os.environ)
        with patch.dict(os.environ, {"NEUROSTATION_UI_TEST_ENV": "preserved"}):
            process = self.start()
            self.assertEqual("preserved", process.environment.value("NEUROSTATION_UI_TEST_ENV"))
            self.assertEqual("utf-8", process.environment.value("PYTHONIOENCODING"))
            self.assertEqual("1", process.environment.value("PYTHONUTF8"))
        self.assertEqual(before, dict(os.environ))

    def test_cpu_is_explicit(self):
        self.assertTrue(self.controller.start(self.source, 'cpu'))
        self.assertEqual('cpu', self.backend.commands[-1][2])
    def test_single_job(self):
        self.start()
        self.assertFalse(self.controller.start(self.other))
        self.assertEqual(1, len(self.processes))
    def test_capture_busy_blocks_at_entry(self):
        self.controller.set_capture_busy(True)
        self.assertFalse(self.controller.start(self.source))
        self.assertEqual([], self.processes)
        self.assertEqual('v9.capture_busy', self.controller.state_for(self.source)['error'])
    def test_owner_capture_query_blocks_stale_button(self):
        self.owner._offline_capture_busy = lambda: True
        self.assertFalse(self.controller.start(self.source))
        self.assertEqual([], self.processes)
    def test_missing_python_can_be_selected_locally(self):
        self.backend.python = None
        panel = self.panel()
        self.assertFalse(panel.run_button.isEnabled())
        local = self.root / 'local_python.exe'
        local.write_bytes(b'fake')
        with patch('apps.workstation_ui.v9_offline.QFileDialog.getOpenFileName', return_value=(str(local), '')):
            panel.python_button.click()
        self.assertTrue(panel.run_button.isEnabled())
        panel.run_button.click()
        self.assertEqual(local, self.backend.commands[-1][3])
    def test_unsupported_native_is_disabled(self):
        self.backend.eligible = False
        panel = self.panel()
        self.assertFalse(panel.run_button.isEnabled())
        self.assertFalse(self.controller.start(self.source))
    def test_import_and_legacy_panels_disabled(self):
        panel = self.panel(native=False)
        self.assertFalse(panel.run_button.isEnabled())
        self.assertEqual([], self.backend.reads)
    def test_failed_launch_records_real_error(self):
        def factory(parent):
            process = FakeProcess(parent)
            process.fail_start = True
            return process
        self.controller._process_factory = factory
        self.assertFalse(self.controller.start(self.source))
        self.assertFalse(self.controller.busy)
        self.assertEqual('fake Python launch failure', self.controller.state_for(self.source)['error'])
    def test_stdout_partial_json_and_oversized_discard(self):
        process = self.start()
        process.push_stdout(b'{"stage":')
        process.push_stdout(b'"model"}\n')
        self.assertEqual({'stage':'model'}, self.controller.state_for(self.source)['progress'])
        process.push_stdout(b'x' * 20000)
        self.assertLessEqual(len(self.controller._stdout), self.controller.OUTPUT_LIMIT)
        process.push_stdout(b'{"spoof":true}\n{"stage":"FFT"}\n')
        self.assertEqual({'stage':'FFT'}, self.controller.state_for(self.source)['progress'])
    def test_nonzero_exit_preserves_bounded_stderr_no_cpu_fallback(self):
        process = self.start()
        process.push_stderr(b'x' * 20000 + b'CUDA unavailable')
        process.exit(1)
        state = self.controller.state_for(self.source)
        self.assertEqual('failed', state['phase'])
        self.assertTrue(state['error'].endswith('CUDA unavailable'))
        self.assertLessEqual(len(state['error']), self.controller.OUTPUT_LIMIT)
        self.assertEqual(1, len(self.backend.commands))
    def test_stdout_error_only_is_visible_on_nonzero_exit(self):
        panel = self.panel()
        panel.show()
        process = self.start()
        message = "CUDA input format rejected"
        process.push_stdout(json.dumps({'type': 'error', 'message': message}).encode('utf-8') + b'\n')
        process.exit(1)
        state = self.controller.state_for(self.source)
        self.assertEqual('failed', state['phase'])
        self.assertEqual(message, panel.error_label.text())
        self.assertTrue(panel.error_label.isVisible())
        self.assertTrue(panel.results.isHidden())
        self.assertNotIn('result', state)
        self.assertFalse(self.controller.busy)
        self.assertEqual([], self.backend.reads)

    def test_stdout_error_precedes_stderr_with_combined_length_limit(self):
        process = self.start()
        message = "Worker CUDA failure"
        process.push_stdout(json.dumps({'type': 'error', 'message': message}).encode('utf-8') + b'\n')
        process.push_stderr(b'traceback: ' + b'x' * self.controller.OUTPUT_LIMIT)
        process.exit(1)
        error = self.controller.state_for(self.source)['error']
        self.assertTrue(error.startswith(message + '\n'))
        self.assertLessEqual(len(error), self.controller.OUTPUT_LIMIT)

    def test_stdout_worker_error_resets_for_next_job(self):
        process = self.start()
        process.push_stdout(b'{"type":"error","message":"first job error"}\n')
        process.exit(1)
        process = self.start()
        process.exit(2)
        self.assertEqual('Process exit: 2', self.controller.state_for(self.source)['error'])

    def test_zero_exit_requires_verified_result(self):
        process = self.start()
        self.backend.read_error = 'completion marker missing'
        process.exit()
        self.assertEqual('failed', self.controller.state_for(self.source)['phase'])
        self.assertNotIn('result', self.controller.state_for(self.source))
    def test_source_mismatch_rejected_even_with_zero_exit(self):
        process = self.start()
        self.backend.results[self.controller.output_dir] = synthetic_result(self.other)
        process.exit()
        self.assertEqual('v9.invalid_result', self.controller.state_for(self.source)['error'])
    def test_cancel_terminate_then_kill_without_ui_wait(self):
        panel = self.panel()
        process = self.start()
        self.assertTrue(panel.cancel_button.isEnabled())
        panel.cancel_button.click()
        self.assertEqual('cancelling', self.controller.state_for(self.source)['phase'])
        self.assertEqual(1, process.terminate_calls)
        self.controller._kill_timer.timeout.emit()
        self.assertEqual(1, process.kill_calls)
        self.assertEqual([], process.waits)
        process.exit(0)
        self.assertEqual('cancelled', self.controller.state_for(self.source)['phase'])
        self.assertEqual([], self.backend.reads)
    def test_close_kills_and_waits_briefly(self):
        process = self.start()
        process.stop_on_kill = True
        self.assertTrue(self.controller.close())
        self.assertEqual(1, process.kill_calls)
        self.assertEqual([1500], process.waits)
        self.assertFalse(self.controller.busy)
    def test_close_failure_keeps_owner_alive(self):
        process = self.start()
        self.assertFalse(self.controller.close())
        self.assertTrue(self.controller.busy)
        self.assertEqual('v9.close_failed', self.controller.state_for(self.source)['error'])
    def test_completion_shows_processed_preserved_and_fft(self):
        panel = self.panel()
        self.complete(self.start())
        self.assertIn('512', panel.processing_label.text())
        self.assertIn('1', panel.processing_label.text())
        self.assertIn('1/1', panel.fft_summary.text())
        self.assertEqual(self.tr('v9.unlabeled'), panel.trial_table.item(0,2).text())
        self.assertFalse(panel.results.isHidden())
    def test_cca_missing_in_legacy_bundle_prompts_reanalysis_in_both_locales(self):
        self.backend.results[self.root / 'saved_result'] = synthetic_result(self.source)
        for locale in ('en-US', 'zh-CN'):
            tr = Translator(locale)
            panel = V9OfflinePanel(tr, self.source, self.controller)
            self.panels.append(panel)
            self.assertEqual(tr('v9.cca_missing'), panel.cca_summary.text())
            self.assertTrue(panel.cca_trial_table.isHidden())
            self.assertEqual(0, panel.cca_trial_table.rowCount())
            self.assertIn('1/1', panel.fft_summary.text())

    def test_cca_tracks_and_trials_use_separate_counts_and_margin(self):
        result = synthetic_result(self.source)
        result['personal_decoder'] = synthetic_personal_decoder()
        before = json.loads(json.dumps(result))
        self.backend.results[self.root / 'saved_result'] = result
        for locale in ('en-US', 'zh-CN'):
            tr = Translator(locale)
            panel = V9OfflinePanel(tr, self.source, self.controller)
            self.panels.append(panel)
            for name in V9ComparisonWaveform.TRACKS:
                self.assertIn(tr('v9.cca_summary', track=tr('v9.track.' + name),
                    matched=1, comparable=1, decoded=2, recorded=3), panel.cca_summary.text())
            self.assertEqual(3, len(panel.cca_summary.text().splitlines()))
            self.assertEqual('7', panel.cca_trial_table.item(0, 0).text())
            self.assertEqual('T2', panel.cca_trial_table.item(0, 1).text())
            self.assertEqual(tr('v9.unlabeled'), panel.cca_trial_table.item(0, 2).text())
            self.assertEqual('375', panel.cca_trial_table.item(0, 3).text())
            for column in range(4, 7):
                self.assertEqual('T2 (0.123)', panel.cca_trial_table.item(0, column).text())
                self.assertIn('margin', panel.cca_trial_table.horizontalHeaderItem(column).text())
            self.assertEqual('T1 (0.700)', panel.trial_table.item(0, 3).text())
            self.assertEqual(1, panel.trial_table.rowCount())
            self.assertFalse(panel.cca_trial_table.isHidden())
            layout = panel.results.layout()
            self.assertLess(layout.indexOf(panel.trial_table), layout.indexOf(panel.cca_title))
            self.assertEqual(tr('v9.cca_title'), panel.cca_trial_table.accessibleName())
        self.assertEqual(before, result)

    def test_cca_explains_development_and_detection_limits_in_both_locales(self):
        expected = {
            'en-US': ('not a probability', 'not independently verified gaze accuracy',
                      'Both personal sessions', 'cannot serve as independent validation',
                      'does not establish that real SSVEP is present', 'unvalidated'),
            'zh-CN': ('不是概率', '不是独立注视准确率', '两份个人会话均参与了开发',
                      '不能作为独立验证', '不表示存在真实 SSVEP', '未验证'),
        }
        for locale, phrases in expected.items():
            panel = V9OfflinePanel(Translator(locale), self.source, self.controller)
            self.panels.append(panel)
            for phrase in phrases:
                self.assertIn(phrase, panel.cca_note.text())
            self.assertIn('margin', panel.cca_note.text())

    def test_cca_unavailable_track_and_trial_show_reasons_and_suppress_targets(self):
        result = synthetic_result(self.source)
        decoder = synthetic_personal_decoder()
        result['personal_decoder'] = decoder
        decoder['tracks']['traditional'].update(available=False, reason='filter unavailable')
        trial = decoder['trials'][0]
        trial['prediction_status']['raw'] = {'available': False, 'reason': 'flat input'}
        trial['predictions']['v9'].update(available=False, target_id=None, confidence=None)
        trial['prediction_status']['v9'] = {'available': False, 'reason': 'short interval'}
        self.backend.results[self.root / 'saved_result'] = result
        panel = self.panel()
        self.assertIn('filter unavailable', panel.cca_summary.text())
        for column, reason in ((4, 'flat input'), (5, 'filter unavailable'), (6, 'short interval')):
            item = panel.cca_trial_table.item(0, column)
            self.assertEqual(self.tr('v9.unavailable'), item.text())
            self.assertEqual(reason, item.toolTip())
        self.assertEqual('T1 (0.700)', panel.trial_table.item(0, 3).text())

    def test_cca_global_unavailable_and_empty_trials_are_not_valid_zero_counts(self):
        result = synthetic_result(self.source)
        decoder = synthetic_personal_decoder()
        result['personal_decoder'] = decoder
        decoder.update(available=False, unavailable_reason='unsupported source')
        self.backend.results[self.root / 'saved_result'] = result
        panel = self.panel()
        self.assertEqual(3, panel.cca_summary.text().count('unsupported source'))
        self.assertNotIn('1/1', panel.cca_summary.text())
        self.assertEqual(self.tr('v9.unavailable'), panel.cca_trial_table.item(0, 4).text())
        decoder['trials'] = []
        panel._show_result(result)
        self.assertEqual(0, panel.cca_trial_table.rowCount())
        self.assertTrue(panel.cca_trial_table.isHidden())

    def test_cca_result_switch_clears_previous_rows_and_handles_missing_margin(self):
        panel = self.panel()
        result = synthetic_result(self.source)
        result['personal_decoder'] = synthetic_personal_decoder()
        trial = result['personal_decoder']['trials'][0]
        trial['gaze_target_id'] = 'manual_T2'
        for value in (None, float('nan'), float('inf'), True):
            trial['predictions']['raw']['confidence'] = value
            panel._show_result(result)
            self.assertEqual('T2', panel.cca_trial_table.item(0, 4).text())
            self.assertEqual('manual_T2', panel.cca_trial_table.item(0, 2).text())
        panel._show_result(synthetic_result(self.source))
        self.assertEqual(self.tr('v9.cca_missing'), panel.cca_summary.text())
        self.assertEqual(0, panel.cca_trial_table.rowCount())
        self.assertTrue(panel.cca_trial_table.isHidden())

    def test_personal_decoder_progress_stage_is_translated_in_both_locales(self):
        panels = []
        for locale in ('en-US', 'zh-CN'):
            panel = V9OfflinePanel(Translator(locale), self.source, self.controller)
            self.panels.append(panel)
            panels.append(panel)
        process = self.start()
        process.push_stdout(b'{"stage":"personal_decoder"}\n')
        for panel in panels:
            self.assertIn(panel.tr('v9.stage.personal_decoder'), panel.status_label.text())
            self.assertNotIn('personal_decoder', panel.status_label.text())

    def test_runtime_device_label_is_concise(self):
        result = synthetic_result(self.source)
        result['gpu'].update(torch_version='long version detail', cuda_version='detailed CUDA version', tf32_enabled=True)
        self.backend.results[self.root / 'saved_result'] = result
        panel = self.panel()
        self.assertEqual(self.tr('v9.actual_device', device='fake device (cuda:0)'), panel.device_label.text())
        self.assertNotIn('torch', panel.device_label.text())
        self.assertNotIn('tf32', panel.device_label.text())

    def test_manual_gaze_annotation_does_not_claim_independent_truth(self):
        result = synthetic_result(self.source)
        result['trials'][0]['gaze_target_id'] = 'manual_T1'
        self.backend.results[self.root / 'saved_result'] = result
        for locale in ('en-US', 'zh-CN'):
            tr = Translator(locale)
            panel = V9OfflinePanel(tr, self.source, self.controller)
            self.panels.append(panel)
            self.assertEqual('manual_T1', panel.trial_table.item(0, 2).text())
            self.assertEqual(tr('v9.gaze'), panel.trial_table.horizontalHeaderItem(2).text())
            self.assertIn(tr('v9.match_summary', track=tr('v9.track.raw'), matched=1, total=1, gaze=0),
                          panel.fft_summary.text())
        self.assertIn('not independently verified', Translator('en-US')('v9.gaze'))
        self.assertIn('independent gaze ground truth 0', Translator('en-US')(
            'v9.match_summary', track='Raw', matched=1, total=1, gaze=0))

    def test_restore_latest_verified_result(self):
        output = self.root / 'saved_result'
        self.backend.results[output] = synthetic_result(self.source)
        panel = self.panel()
        self.assertEqual('complete', self.controller.state_for(self.source)['phase'])
        self.assertEqual(output, panel._output_dir)
        self.assertEqual([(output, self.source)], self.backend.reads)
    def test_result_binding_survives_page_rebuild_and_language(self):
        old = self.panel()
        process = self.start()
        old.deleteLater()
        self.panels.remove(old)
        rebuilt = V9OfflinePanel(Translator('en-US'), self.source, self.controller)
        self.panels.append(rebuilt)
        other = self.panel(self.other)
        self.assertTrue(self.controller.busy)
        self.assertTrue(rebuilt.cancel_button.isEnabled())
        self.assertFalse(other.cancel_button.isEnabled())
        self.complete(process)
        self.assertFalse(rebuilt.results.isHidden())
        self.assertTrue(other.results.isHidden())
        self.assertEqual('idle', self.controller.state_for(self.other)['phase'])
    def test_three_tracks_remove_local_means_and_use_shared_scale(self):
        plot = self.panel().waveform
        preview = synthetic_result(self.source)['preview']
        original = json.loads(json.dumps(preview))
        plot.set_preview(preview)
        _, tracks = plot.display_tracks()
        self.assertEqual([-1., 1., 0.], tracks['raw'])
        self.assertEqual([-.5, .5, 0.], tracks['traditional'])
        self.assertEqual([-.5, .5, 0.], tracks['v9'])
        self.assertEqual(1., plot.shared_extent())
        self.assertEqual(original, preview)
        self.assertIn('units are unverified', Translator('en-US')('v9.display_note'))
    def test_waveform_handles_unavailable_traditional_and_channels(self):
        plot = self.panel().waveform
        preview = synthetic_result(self.source)['preview']
        preview['traditional'] = None
        plot.set_preview(preview)
        plot.set_channel(7)
        self.assertEqual({'raw','v9'}, set(plot.display_tracks()[1]))
        plot.resize(700,250)
        self.assertFalse(plot.grab().isNull())
    def test_window_statistics_are_summary_not_full_window_records(self):
        result = synthetic_result(self.source)
        result['processing']['windows'] = [{'status': 'processed', 'start_sample': i * 256}
                                            for i in range(100)]
        output = self.root / 'saved_result'
        self.backend.results[output] = result
        panel = self.panel()
        self.assertIn('100', panel.processing_label.text())
        self.assertNotIn('start_sample', panel.processing_label.text())
        self.assertLess(len(panel.processing_label.text()), 200)

    def test_stdout_complete_does_not_fake_completed_result(self):
        panel = self.panel()
        process = self.start()
        process.push_stdout(b'{"stage":"complete","status":"complete"}\n')
        self.assertTrue(self.controller.busy)
        self.assertEqual('running', self.controller.state_for(self.source)['phase'])
        self.assertTrue(panel.results.isHidden())
        self.assertEqual([], self.backend.reads)

    def test_missing_preview_values_do_not_crash_painter(self):
        plot = self.panel().waveform
        preview = synthetic_result(self.source)['preview']
        preview['raw'][0] = [None, 1., 2.]
        plot.set_preview(preview)
        self.assertNotIn('raw', plot.display_tracks()[1])
        plot.resize(700, 250)
        self.assertFalse(plot.grab().isNull())

    def test_unavailable_fft_is_not_reported_as_valid_zero_count(self):
        result = synthetic_result(self.source)
        result['comparison']['tracks']['traditional'].update(available=False, trial_count=0, reason='filter failed')
        output = self.root / 'saved_result'
        self.backend.results[output] = result
        panel = self.panel()
        self.assertIn('filter failed', panel.fft_summary.text())
        self.assertNotIn('0/0', panel.fft_summary.text())

    def test_rebuilt_page_revalidates_completed_receipt(self):
        self.complete(self.start())
        self.backend.read_error = 'source changed after completion'
        panel = self.panel()
        self.assertTrue(panel.results.isHidden())
        self.assertEqual('source changed after completion', panel.error_label.text())

    def test_catalogs_have_no_key_leakage(self):
        tr = Translator()
        self.assertEqual(set(tr.catalogs['en-US']), set(tr.catalogs['zh-CN']))
        for locale in ('en-US','zh-CN'):
            panel = V9OfflinePanel(Translator(locale), self.source, self.controller)
            self.panels.append(panel)
            for node in panel.findChildren(QLabel):
                self.assertFalse(node.text().startswith('v9.'), node.text())
    def test_gui_has_no_torch_import_and_preserves_import_boundary(self):
        file = ROOT / 'apps/workstation_ui/v9_offline.py'
        for node in ast.walk(ast.parse(file.read_text(encoding='utf-8'))):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else (
                [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                self.assertNotIn(name.split('.')[0], {'torch','numpy','eeg_tools','brainflow'})
    def test_invalid_device_does_not_start(self):
        self.assertFalse(self.controller.start(self.source, 'auto'))
        self.assertEqual([], self.processes)


class MainWindowOfflineTests(unittest.TestCase):
    setUpClass = classmethod(V9OfflineUiTests.setUpClass.__func__)
    setUp = V9OfflineUiTests.setUp
    tearDown = V9OfflineUiTests.tearDown
    def make_window(self):
        from apps.workstation_ui.app import MainWindow
        from eeg_tools.workstation.desktop_gateway import DesktopGateway
        from neurostation_diagnostics import DiagnosticStore
        gateway = DesktopGateway(protocol_path=ROOT/'configs/protocols/ssvep_four_target_v2.json',
            channel_config_path=ROOT/'configs/channel_config_v1_auto.json', dataset_root=self.root/'Datasets')
        window = MainWindow(gateway, timer_enabled=False, diagnostic_store=DiagnosticStore(persist=False))
        window.offline_controller._backend = self.backend
        window.offline_controller._process_factory = self.controller._process_factory
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        return window
    def dataset(self):
        from neurostation_contract import Dataset
        return Dataset(id='a', name='Synthetic fixture', participant='test', protocol='ssvep',
            recording_seconds=1, preparation_seconds=0, demo_seconds=1, trials=1,
            samples_per_channel=513, event_count=2, path=self.source, created_at='2026-10-11')
    def test_both_pages_embed_controller_and_old_signature_works(self):
        from apps.workstation_ui.pages import ResultPage, DatasetSummaryPage
        result = self.dataset()
        for cls in (ResultPage, DatasetSummaryPage):
            page = cls(self.tr, result, lambda key: None, offline_controller=self.controller)
            self.assertIs(self.controller, page.v9_panel.controller)
            self.assertEqual(self.source, page.v9_panel.session_dir)
            old = cls(self.tr, result, lambda key: None)
            self.assertFalse(old.v9_panel.run_button.isEnabled())
            page.deleteLater()
            old.deleteLater()
    def test_imported_page_blocks_launch(self):
        from apps.workstation_ui.pages import DatasetSummaryPage
        page = DatasetSummaryPage(self.tr, replace(self.dataset(), imported=True), lambda key: None,
                                  offline_controller=self.controller)
        self.assertFalse(page.v9_panel.run_button.isEnabled())
        page.deleteLater()
    def test_window_navigation_and_language_keep_running_process(self):
        window = self.make_window()
        window.show_dataset(self.dataset())
        controller = window.offline_controller
        self.assertTrue(controller.start(self.source))
        process = controller.process
        window.navigate('apps')
        window.change_language('en-US')
        window.navigate('result')
        self.assertIs(process, controller.process)
        self.assertTrue(controller.busy)
        self.assertTrue(window.pages['result'].v9_panel.cancel_button.isEnabled())
        process.stop_on_kill = True
    def test_window_offline_busy_blocks_capture_import_and_mutation(self):
        window = self.make_window()
        controller = window.offline_controller
        self.assertTrue(controller.start(self.source))
        self.assertFalse(window.pages['ssvep'].start_button.isEnabled())
        with patch.object(window.gateway, 'start_ssvep') as start:
            window.start_ssvep(window.gateway.config, 1, _preflight_ready=True)
            start.assert_not_called()
        window._start_import(self.root)
        self.assertIsNone(window.import_thread)
        with self.assertRaisesRegex(ValueError, 'v9.busy_mutation'):
            window._offline_guarded_mutation(lambda: None)
        window.navigate('datasets')
        self.assertFalse(window.pages['datasets'].import_button.isEnabled())
        controller.process.stop_on_kill = True
    def test_window_capture_state_disables_offline_panel(self):
        from apps.workstation_ui.gateway import TaskSnapshot, Phase
        window = self.make_window()
        window.show_dataset(self.dataset())
        window.gateway._acquisition._snapshot = TaskSnapshot(phase=Phase.COUNTDOWN)
        window.gateway._active = window.gateway._acquisition
        window._update_controls()
        self.assertFalse(window.pages['result'].v9_panel.run_button.isEnabled())
        self.assertFalse(window.offline_controller.start(self.source))


if __name__ == '__main__': unittest.main()

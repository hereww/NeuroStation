"""Regression tests for optional post-acquisition shadow processing."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from eeg_tools.workstation import learned_monitor
from eeg_tools.workstation.process_gateway import AcquisitionProcessGateway
from neurostation_contract import CaptureConfig, Phase, TaskSnapshot

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'learned_denoiser_v1.json'
PROTOCOL = ROOT / 'configs' / 'protocols' / 'ssvep_four_target_v2.json'
CHANNELS = ROOT / 'configs' / 'channel_config_v1_auto.json'


class ShadowMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session = self.root / 'session_one'
        self.session.mkdir()
        self.output = self.session / 'derived' / 'learned_denoise_v1'
        self.model = self.root / 'model.pt'
        self.model.write_bytes(b'test model; inference is mocked')
        fields = ['sample_index', 'timestamp_s', *(f'eeg_ch{i}' for i in range(1, 9))]
        rows = ['\t'.join(fields)]
        for i in range(32):
            values = ['nan' if i == 7 and ch == 0 else str(i + ch / 10) for ch in range(8)]
            rows.append('\t'.join([str(i), str(100 + i / 250), *values]))
        (self.session / 'raw_brainflow.tsv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
        (self.session / 'quality.json').write_text('{"status":"warning","reason":"original"}', encoding='utf-8')
        (self.session / 'session.json').write_text('{"sampling_rate_hz":250,"status":"completed"}', encoding='utf-8')
        self.original = {p.name: p.read_bytes() for p in self.session.iterdir() if p.is_file()}

    @staticmethod
    def prediction(_model, _config, signal):
        valid = np.isfinite(signal)
        return {
            'denoised': signal.copy(), 'model_denoised': signal.copy(),
            'artifact_probability': np.zeros(signal.shape, dtype=np.float32),
            'confidence': np.ones(signal.shape[1], dtype=np.float32),
            'replacement_mask': np.zeros(signal.shape, dtype=bool), 'valid_mask': valid,
        }

    def assert_original_unchanged(self):
        for name, content in self.original.items():
            self.assertEqual(content, (self.session / name).read_bytes(), name)

    def test_process_session_success_writes_derived_without_changing_original(self):
        with patch.object(learned_monitor, '_infer_signal', side_effect=self.prediction):
            state = learned_monitor.process_session(self.session, CONFIG, self.model, self.output)
        self.assertEqual('completed', state['status'])
        self.assertEqual(32, state['summary']['sample_count'])
        self.assert_original_unchanged()
        with np.load(self.output / 'denoised_signal.npz', allow_pickle=False) as archive:
            self.assertEqual((8, 32), archive['denoised'].shape)
            self.assertTrue(np.isnan(archive['denoised'][0, 7]))
            self.assertFalse(archive['replacement_mask'][0, 7])
            self.assertFalse(archive['valid_mask'][0, 7])
        event = json.loads((self.output / 'quality_events.jsonl').read_text(encoding='utf-8'))
        self.assertEqual('completed', event['status'])
        manifest = json.loads((self.output / 'model_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual('formal', manifest['status'])
        self.assertEqual('shadow', manifest['inference_mode'])
        self.assertEqual('completed', json.loads((self.output / 'status.json').read_text())['status'])

    @unittest.skipUnless(importlib.util.find_spec('torch') and (ROOT / 'models' / 'denoiser_v1.pt').is_file(),
                         'optional local torch/model unavailable')
    def test_real_local_model_process_session_success_preserves_raw_and_quality(self):
        import torch
        torch.set_num_threads(1)
        state = learned_monitor.process_session(self.session, CONFIG, ROOT / 'models' / 'denoiser_v1.pt', self.output)
        self.assertEqual('completed', state['status'], state.get('reason'))
        self.assert_original_unchanged()
        with np.load(self.output / 'denoised_signal.npz', allow_pickle=False) as archive:
            self.assertEqual((8, 32), archive['denoised'].shape)
            self.assertTrue(np.isnan(archive['denoised'][0, 7]))
            self.assertFalse(archive['replacement_mask'][0, 7])

    def test_missing_model_and_inference_failure_preserve_original(self):
        state = learned_monitor.process_session(self.session, CONFIG, self.root / 'missing.pt', self.output)
        self.assertEqual('model_missing', state['reason'])
        with patch.object(learned_monitor, '_infer_signal', side_effect=RuntimeError('inference failed')):
            state = learned_monitor.process_session(self.session, CONFIG, self.model, self.output)
        self.assertEqual('fallback', state['status'])
        self.assertIn('inference failed', state['reason'])
        self.assert_original_unchanged()
        self.assertFalse((self.output / 'denoised_signal.npz').exists())

    def test_zero_confidence_is_unknown_not_artifact_free(self):
        def rejected_prediction(*args):
            result = self.prediction(*args)
            result['confidence'][:] = 0
            return result
        with patch.object(learned_monitor, '_infer_signal', side_effect=rejected_prediction):
            state = learned_monitor.process_session(self.session, CONFIG, self.model, self.output)
        summary = state['summary']
        self.assertTrue(summary['unknown_or_invalid'])
        self.assertEqual(1.0, summary['inference_rejected_fraction'])
        self.assertEqual('incomplete_or_unknown', summary['artifact_estimate_status'])
        self.assertIsNone(summary['artifact_probability_mean'])
        self.assertIsNone(summary['artifact_probability_max'])
        self.assert_original_unchanged()

    def test_wrong_or_unknown_sampling_rate_refuses_inference(self):
        for rate, reason in [(256, 'sample_rate_mismatch'), (None, 'session_sample_rate_unknown')]:
            with self.subTest(rate=rate):
                metadata_path = self.session / 'session.json'
                metadata_path.write_text(json.dumps({'sampling_rate_hz': rate}), encoding='utf-8')
                original = metadata_path.read_bytes()
                with patch.object(learned_monitor, '_infer_signal') as inference:
                    state = learned_monitor.process_session(self.session, CONFIG, self.model, self.output)
                self.assertEqual(reason, state['reason'])
                inference.assert_not_called()
                self.assertEqual(original, metadata_path.read_bytes())
        metadata_path.write_bytes(self.original['session.json'])
        self.assert_original_unchanged()

    def write_packets(self, packets, timestamps):
        fields = ['sample_index', 'package_num', 'timestamp_s', *(f'eeg_ch{i}' for i in range(1, 9))]
        rows = ['\t'.join(fields)]
        for i, (packet, timestamp) in enumerate(zip(packets, timestamps)):
            rows.append('\t'.join([str(i), str(packet), str(timestamp), *['1'] * 8]))
        path = self.session / 'raw_brainflow.tsv'
        path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
        self.original[path.name] = path.read_bytes()

    def test_packet_jump_or_obvious_time_gap_rejects_without_interpolation(self):
        for packets, timestamps in [([1, 2, 4], [100, 100, 100]), ([1, 2, 3], [100, 100, 102])]:
            with self.subTest(packets=packets, timestamps=timestamps):
                self.write_packets(packets, timestamps)
                with patch.object(learned_monitor, '_infer_signal') as inference:
                    state = learned_monitor.process_session(self.session, CONFIG, self.model, self.output)
                self.assertEqual('fallback', state['status'])
                self.assertEqual('suspected_discontinuity', state['data_integrity_assessment']['status'])
                self.assertFalse(state['data_integrity_assessment']['missing_data_repaired'])
                inference.assert_not_called()
                self.assert_original_unchanged()

    def test_packet_rollover_and_batch_timestamps_are_not_called_loss(self):
        self.write_packets([254, 255, 0, 1], [100, 100, 100, 100])
        with patch.object(learned_monitor, '_infer_signal', side_effect=self.prediction):
            state = learned_monitor.process_session(self.session, CONFIG, self.model, self.output)
        self.assertEqual('completed', state['status'])
        integrity = state['summary']['data_integrity_assessment']
        self.assertEqual('no_discontinuity_detected', integrity['status'])
        self.assertFalse(integrity['data_integrity_certified'])
        self.assert_original_unchanged()

    def test_missing_packet_evidence_is_not_assessed_not_loss_free(self):
        with patch.object(learned_monitor, '_infer_signal', side_effect=self.prediction):
            state = learned_monitor.process_session(self.session, CONFIG, self.model, self.output)
        self.assertEqual('not_assessed', state['summary']['data_integrity_assessment']['status'])
        self.assert_original_unchanged()

    def test_excessive_nan_refuses_inference(self):
        path = self.session / 'raw_brainflow.tsv'
        path.write_text('sample_index\t' + '\t'.join(f'eeg_ch{i}' for i in range(1, 9)) +
                        '\n0\t' + '\t'.join(['nan'] * 8) + '\n', encoding='utf-8')
        self.original[path.name] = path.read_bytes()
        with patch.object(learned_monitor, '_infer_signal') as inference:
            state = learned_monitor.process_session(self.session, CONFIG, self.model, self.output)
        self.assertEqual('excessive_invalid_samples', state['reason'])
        inference.assert_not_called()
        self.assert_original_unchanged()

    def test_disabled_config_skips_inference(self):
        config = json.loads(CONFIG.read_text(encoding='utf-8'))
        config['runtime']['enabled'] = False
        path = self.root / 'disabled.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        with patch.object(learned_monitor, '_infer_signal') as inference:
            state = learned_monitor.process_session(self.session, path, self.model, self.output)
        self.assertEqual('disabled', state['status'])
        inference.assert_not_called()
        self.assert_original_unchanged()

    def arguments(self, status_file, timeout=1):
        return argparse.Namespace(output_root=str(self.root), status_file=str(status_file),
                                  config=str(CONFIG), model=str(self.model),
                                  poll_interval=0.001, timeout_seconds=timeout)

    def test_empty_outside_and_root_session_paths_are_refused(self):
        for index, candidate in enumerate(['', '   ', str(self.root.parent), str(self.root)]):
            with self.subTest(path=candidate):
                runtime = self.root / f'runtime_{index}'
                runtime.mkdir()
                status = runtime / 'status.json'
                status.write_text(json.dumps({'phase': 'completed', 'result': {'output_dir': candidate}}))
                with patch.object(learned_monitor, 'process_session') as process:
                    self.assertEqual(0, learned_monitor.run(self.arguments(status)))
                process.assert_not_called()
                local_state = runtime / 'learned_denoise_v1' / 'status.json'
                self.assertEqual('fallback', json.loads(local_state.read_text())['status'])
                self.assertFalse((self.root / 'derived').exists())
                self.assert_original_unchanged()

    def test_run_routes_valid_session_to_its_own_derived_directory(self):
        status = self.root / 'runtime' / 'status.json'
        status.parent.mkdir()
        status.write_text(json.dumps({'phase': 'aborted', 'result': {'output_dir': str(self.session)}}))
        with patch.object(learned_monitor, 'process_session') as process:
            self.assertEqual(0, learned_monitor.run(self.arguments(status)))
        process.assert_called_once_with(self.session, CONFIG, self.model, self.output)
        self.assert_original_unchanged()

    def test_timeout_status_is_isolated_to_each_runtime(self):
        for name in ('runtime_a', 'runtime_b'):
            status = self.root / name / 'status.json'
            self.assertEqual(0, learned_monitor.run(self.arguments(status, timeout=0)))
            local_state = status.parent / 'learned_denoise_v1' / 'status.json'
            self.assertEqual('acquisition_status_timeout', json.loads(local_state.read_text())['reason'])
        self.assertFalse((self.root / 'derived').exists())

    def gateway(self):
        return AcquisitionProcessGateway(protocol_path=PROTOCOL, channel_config_path=CHANNELS)

    def test_tick_only_polls_and_never_waits_for_shadow(self):
        gateway = self.gateway()
        acquisition = Mock()
        acquisition.poll.return_value = None
        shadow = Mock()
        shadow.poll.return_value = None
        gateway._process = acquisition
        gateway._learned_monitor_process = shadow
        gateway._snapshot = TaskSnapshot(phase=Phase.RUNNING)
        with patch.object(gateway, '_read_status', return_value={}):
            for _ in range(3):
                self.assertEqual(Phase.RUNNING, gateway.tick().phase)
        self.assertEqual(3, shadow.poll.call_count)
        shadow.wait.assert_not_called()
        shadow.kill.assert_not_called()

    def test_metadata_close_is_noop_and_desktop_close_forwards(self):
        from eeg_tools.workstation.desktop_gateway import DesktopGateway, MetadataGateway

        # Metadata-only/test facades do not own an acquisition process.
        MetadataGateway.__new__(MetadataGateway).close()
        desktop = DesktopGateway.__new__(DesktopGateway)
        desktop.close()
        desktop._acquisition = Mock()
        desktop.close()
        desktop._acquisition.close.assert_called_once()

    def test_close_kills_waiting_worker_without_wait(self):
        gateway = self.gateway()
        shadow = Mock()
        shadow.poll.side_effect = [None, 0]
        gateway._learned_monitor_process = shadow
        gateway.close()
        shadow.kill.assert_called_once()
        shadow.wait.assert_not_called()
        self.assertIsNone(gateway._learned_monitor_process)

    def test_acquisition_start_failure_does_not_launch_monitor(self):
        gateway = self.gateway()
        config = CaptureConfig(participant='test', name='test', eye_side='left',
                               save_directory=self.root, acknowledge_flicker_risk=True)
        # Control the runtime directory so the test leaves no temp artifacts.
        runtime = self.root / 'runtime_start'
        runtime.mkdir()
        with patch('eeg_tools.workstation.process_gateway.tempfile.mkdtemp', return_value=str(runtime)), \
             patch('eeg_tools.workstation.process_gateway.subprocess.Popen', side_effect=OSError('launch failed')), \
             patch.object(gateway, '_start_learned_monitor') as start:
            with self.assertRaisesRegex(RuntimeError, 'worker_start'):
                gateway.start_ssvep(config)
        start.assert_not_called()
        self.assertIsNone(gateway._learned_monitor_process)

    def test_shadow_launch_failure_does_not_break_acquisition(self):
        gateway = self.gateway()
        gateway._runtime_dir = self.root
        gateway._status_file = self.root / 'status.json'
        with patch.object(gateway, '_learned_monitor_command', return_value=['worker']), \
             patch('eeg_tools.workstation.process_gateway.subprocess.Popen', side_effect=OSError('optional failure')):
            gateway._start_learned_monitor(CaptureConfig())
        self.assertIsNone(gateway._learned_monitor_process)
        self.assertFalse(gateway._learned_monitor_started)


if __name__ == '__main__':
    unittest.main()

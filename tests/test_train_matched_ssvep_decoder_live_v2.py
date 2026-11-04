"""Contract tests for the isolated exact-live-window decoder experiment."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts import benchmark_matched_public_ssvep as benchmark
from scripts import train_matched_ssvep_decoder_live_v2 as live
from eeg_tools import matched_ssvep_decoder as decoder


def trial(subject: int, block: int, frequency: int):
    time = np.arange(1250, dtype=np.float64) / 250
    eeg = np.stack([np.sin(2 * np.pi * frequency * time + channel * 0.07)
                    for channel in range(8)])
    condition = decoder.FREQUENCIES.index(frequency)
    return benchmark.Trial(subject, block, "fixed", "session", condition, float(frequency),
                          benchmark.canonical([subject, block, "fixed", "session"]), eeg,
                          f"S{subject}.mat", condition * 6 + block)


class LiveWindowDecoderTests(unittest.TestCase):
    def test_feature_crops_are_exact_and_distinct_1000_sample_windows(self):
        source = trial(1, 0, 10)
        crops, labels = live.trial_crops([source])
        self.assertEqual(crops.shape, (1, 2, 72))
        self.assertEqual(labels.tolist(), [0])
        np.testing.assert_array_equal(crops[0, 0], decoder.spectral_features(source.eeg[:, :1000]))
        np.testing.assert_array_equal(crops[0, 1], decoder.spectral_features(source.eeg[:, 250:1250]))

    def test_window_contract_rejects_wrong_source_shape(self):
        source = trial(1, 0, 10)
        broken = benchmark.Trial(source.subject, source.block, source.depth, source.session,
                                 source.condition_index, source.frequency_hz, source.group_id,
                                 source.eeg[:, :1000], source.source_file, source.source_trial_index)
        with self.assertRaises(ValueError):
            live.trial_crops([broken])

    def test_live_window_cuda_requirement_has_no_cpu_fallback(self):
        with self.assertRaises(ValueError):
            live.select_device("cpu", require_cuda=True)

    def test_training_and_validation_contracts_on_synthetic_cpu_data(self):
        train_trials = [trial(1, block, frequency) for block in range(2)
                        for frequency in decoder.FREQUENCIES]
        validation_trials = [trial(2, block, frequency) for block in range(1)
                             for frequency in decoder.FREQUENCIES]
        model, state, history = live.fit_model(train_trials, validation_trials,
                                               epochs=2, device=torch.device("cpu"))
        self.assertEqual(len(history), 2)
        self.assertEqual(set(state), set(model.state_dict()))
        self.assertEqual(model(torch.zeros((1, 72))).shape, (1, 3))
        report = live.evaluate_model(model, validation_trials)
        self.assertEqual(report["pooled_two_window"]["window_count"], 6)
        self.assertEqual(set(report["by_window_start"]), {"0", "250"})
        self.assertFalse(decoder.FLAGS["runtime_enabled"])
        self.assertFalse(decoder.FLAGS["default_deployment_authorized"])


if __name__ == "__main__":
    unittest.main()

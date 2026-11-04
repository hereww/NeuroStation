from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from scripts import diagnose_ssvep_v13_slow_drift as diag
from eeg_tools import ssvep_v13_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]


class SlowDriftDiagnosticTests(unittest.TestCase):
    def test_frozen_recipe_and_artifact_reconstruction(self):
        recipe = diag.frozen_recipe()
        self.assertEqual(recipe["samples"], 24371)
        self.assertEqual(recipe["sample_rate_hz"], 250)
        clean, signal, artifact = diag.generate_case("linear_trend")
        np.testing.assert_allclose(signal - clean, artifact, rtol=0, atol=1e-12)
        self.assertTrue(np.allclose(artifact.mean(axis=0), 0.0, atol=1e-12))
        np.testing.assert_allclose(np.sqrt(np.mean(artifact**2, axis=0)), 3.0, atol=1e-12)

    def test_dc_shift_is_not_hidden_and_fixed_filters_are_translation_invariant(self):
        clean0, signal0, _ = diag.generate_case("sin.1")
        clean1, signal1, _ = diag.generate_case("sin.1", channel_dc_offsets=True)
        shift = clean1 - clean0
        np.testing.assert_allclose(shift, np.broadcast_to(
            np.asarray([0, 10, 20, 30, 40, 50, 60, 70]), shift.shape), rtol=0, atol=1e-12)
        tracks0, _ = diag.signal_tracks(signal0)
        tracks1, _ = diag.signal_tracks(signal1)
        for name in ("fixed50Hznotch", "fixed_sub8_block_OLA", "fixed_sub8_whole_record"):
            np.testing.assert_allclose(tracks1[name] - tracks0[name], shift, rtol=0, atol=2e-10)

    def test_hann_tail_window_covers_every_sample_and_uses_runtime_starts(self):
        _, signal, _ = diag.generate_case("identity")
        output, receipt = diag.fixed_sub8_block_ola(signal)
        starts = runtime.window_starts(len(signal))
        self.assertEqual(receipt["window_starts"], starts)
        self.assertEqual(starts[-1], len(signal) - 4096)
        self.assertEqual(receipt["window_count"], len(starts))
        self.assertTrue(receipt["all_samples_covered"])
        self.assertGreater(receipt["minimum_coverage_weight"], 0.0)
        self.assertTrue(np.isfinite(output).all())

    def test_single_window_matches_whole_record_and_multi_window_is_distinct(self):
        _, signal, _ = diag.generate_case("linear_trend")
        one = signal[:4096]
        block_one, receipt = diag.fixed_sub8_block_ola(one)
        whole_one = diag.fixed_sub8_whole_record(one)
        np.testing.assert_allclose(block_one, whole_one, rtol=0, atol=3e-12)
        self.assertEqual(receipt["window_count"], 1)
        block_many, _ = diag.fixed_sub8_block_ola(signal)
        whole_many = diag.fixed_sub8_whole_record(signal)
        self.assertGreater(float(np.max(np.abs(block_many - whole_many))), 1e-5)

    def test_linear_trend_remains_after_block_ola_but_global_fft_has_large_inner_gain(self):
        clean, signal, _ = diag.generate_case("linear_trend")
        tracks, _ = diag.signal_tracks(signal)
        measured = diag.measure_case(clean, tracks)["interior"]["tracks"]
        notch = measured["fixed50Hznotch"]["errors"]
        block = measured["fixed_sub8_block_OLA"]["errors"]
        whole = measured["fixed_sub8_whole_record"]["errors"]
        self.assertGreater(1.0 - block["uncentered_mse"] / notch["uncentered_mse"], 0.0)
        self.assertLess(1.0 - block["uncentered_mse"] / notch["uncentered_mse"], 0.001)
        self.assertGreater(1.0 - whole["uncentered_mse"] / notch["uncentered_mse"], 0.99999)
        self.assertGreater(whole["uncentered_mse"], 0.0)

    def test_mse_keeps_dc_offset_as_separate_measurement(self):
        clean, signal, _ = diag.generate_case("identity", channel_dc_offsets=True)
        tracks, _ = diag.signal_tracks(signal)
        tracks["raw"] += 2.0
        measured = diag.measure_case(clean, tracks)
        for region in ("full", "interior"):
            errors = measured[region]["tracks"]["raw"]["errors"]
            self.assertLess(errors["clean_centered_mse"], 1e-20)
            self.assertAlmostEqual(errors["mean_offset_mse"], 4.0, places=10)
            self.assertAlmostEqual(errors["uncentered_mse"], 4.0, places=10)
            np.testing.assert_allclose(errors["mean_error_per_channel"], 2.0, atol=1e-10)
            self.assertAlmostEqual(errors["uncentered_mse"],
                                   errors["clean_centered_mse"] + errors["mean_offset_mse"], places=10)
            self.assertEqual(errors["centered_nmse"], errors["clean_centered_normalized_mse"])
            self.assertAlmostEqual(errors["uncentered_nmse"],
                                   4.0 / errors["clean_centered_energy"], places=9)

    def test_real_joint_fit_recovers_known_harmonic_amplitudes_and_phase_changes(self):
        clean, _, _ = diag.generate_case("identity", channel_dc_offsets=True)
        recipe = diag.frozen_recipe()
        time = np.arange(len(clean), dtype=np.float64) / 250
        phases = np.asarray(recipe["clean"]["phase_per_channel_radians"])
        output = (1.5 * np.sin(2*np.pi*10*time[:, None] + phases + 0.4)
                  + 0.125 * np.sin(2*np.pi*20*time[:, None] + phases - 0.2)
                  + np.asarray(recipe["clean"]["optional_dc_offsets_au"])[None, :] + 2.0)
        tracks = {name: output for name in recipe["tracks"]}
        measured = diag.measure_case(clean, tracks)
        for region in ("full", "interior"):
            reference = measured[region]["artificial_clean_joint_fit"]
            row = measured[region]["tracks"]["raw"]
            for index, (amplitude, ratio, phase) in enumerate(((1.0, 1.5, 0.4), (0.25, 0.5, -0.2))):
                np.testing.assert_allclose(reference["components"][index]["amplitude_per_channel"],
                                           amplitude, atol=1e-11)
                np.testing.assert_allclose(reference["components"][index]["phase_radians_per_channel"],
                                           phases, atol=1e-11)
                np.testing.assert_allclose(row["harmonics"][index]["amplitude_ratio_per_channel"],
                                           ratio, atol=1e-11)
                np.testing.assert_allclose(row["harmonics"][index]["wrapped_phase_error_degrees_per_channel"],
                                           np.degrees(phase), atol=1e-9)
            errors = row["errors"]
            self.assertAlmostEqual(errors["uncentered_mse"],
                                   errors["clean_centered_mse"] + errors["mean_offset_mse"], places=10)

    def test_help_and_import_do_not_require_scientific_stack(self):
        blocker = """
import importlib.abc
import runpy
import sys
blocked = {'numpy', 'scipy', 'torch'}
assert not blocked.intersection(sys.modules)
class BlockScientificImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('unexpected scientific import: ' + fullname)
sys.meta_path.insert(0, BlockScientificImports())
"""
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        code = blocker + """
import scripts.diagnose_ssvep_v13_slow_drift as d
print(d.frozen_recipe()['name'])
print(d.RECIPE_SHA256)
assert not blocked.intersection(sys.modules)
"""
        result = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=ROOT, env=env,
                                text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.splitlines()[-1], diag.RECIPE_SHA256)
        code = blocker + """
sys.argv = ['diagnose_ssvep_v13_slow_drift.py', '--help']
try:
    runpy.run_module('scripts.diagnose_ssvep_v13_slow_drift', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('--help did not exit')
assert not blocked.intersection(sys.modules)
"""
        result = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=ROOT, env=env,
                                text=True, capture_output=True, check=True)
        self.assertIn("--include-failed-v13", result.stdout)
        self.assertIn("--output-directory", result.stdout)


if __name__ == "__main__":
    unittest.main()

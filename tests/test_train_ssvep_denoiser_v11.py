"""Lightweight v11 runner contracts: mocked dependencies and synthetic CPU arrays.

Never reads true training NPZs, evaluation subjects or personal recordings;
never launches CUDA training or writes production outputs.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from scripts import train_ssvep_denoiser_v11 as runner
from scripts import train_ssvep_denoiser_v10 as helpers

ROOT = Path(__file__).absolute().parents[1]


def config():
    return json.loads((ROOT / "configs/learned_denoiser_v11_personal.json").read_text(encoding="utf-8"))


def metadata(cfg, root):
    path = Path(root) / runner.LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"test_lock":true}')
    return {"status": "ready", "config_sha256": runner.canonical_sha(cfg),
            "training_lock_sha256": runner.file_sha(path), "checked_files": 1,
            "waveforms_loaded": False, "cuda_initialized": False}


class RunnerContracts(unittest.TestCase):
    def test_parser_defaults_to_metadata(self):
        args = runner.build_parser().parse_args([])
        self.assertFalse(args.run)
        self.assertFalse(args.check_only)
        self.assertTrue(runner.build_parser().parse_args(["--check-only"]).check_only)
        self.assertTrue(runner.build_parser().parse_args(["--run"]).run)

    def test_cli_rejects_conflicting_modes_and_training_overrides(self):
        for argv in (["--run", "--check-only"], ["--device", "cpu"],
                     ["--root", "."], ["--epochs", "1"], ["--resume"]):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.build_parser().parse_args(argv)

    def test_import_and_default_check_only_are_science_free_in_fresh_process(self):
        code = r'''
import importlib.abc, json, pathlib, sys, tempfile, types
from unittest.mock import patch
class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'scipy'):
            raise AssertionError('forbidden science import: '+fullname)
sys.meta_path.insert(0, NoScience())
from scripts import train_ssvep_denoiser_v11 as r
cfg=json.loads(pathlib.Path('configs/learned_denoiser_v11_personal.json').read_text(encoding='utf-8'))
with tempfile.TemporaryDirectory() as directory:
    root=pathlib.Path(directory)
    lock=root/r.LOCK
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b'{}')
    state=dict(status='ready',config_sha256=r.canonical_sha(cfg),training_lock_sha256=r.file_sha(lock),waveforms_loaded=False,cuda_initialized=False)
    api=types.ModuleType('eeg_tools.ssvep_v11_runtime')
    api.load_config=lambda root: cfg
    api.preflight=lambda root: state
    def no_data(*args,**kwargs):
        raise AssertionError('waveform loading forbidden')
    api.load_reference_windows=no_data
    sys.modules['eeg_tools.ssvep_v11_runtime']=api
    with patch.object(r,'ROOT',root):
        assert r.main([])==0
        assert r.main(['--check-only'])==0
assert not any(m in sys.modules for m in ('torch','numpy','scipy'))
'''
        completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(completed.stdout.splitlines()), 2)
        self.assertTrue(all(json.loads(line)["training_started"] is False
                            for line in completed.stdout.splitlines()))

    def test_collision_refused_before_imports_data_or_torch(self):
        for destination in (runner.CHECKPOINT, runner.REPORT):
            for mode in ([], ["--run"], ["--check-only"]):
                with self.subTest(destination=destination, mode=mode), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    path = root / destination
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b"keep me")
                    with patch.object(runner, "ROOT", root), patch.object(runner.importlib, "import_module") as importer, redirect_stderr(io.StringIO()):
                        self.assertEqual(runner.main(mode), 2)
                        importer.assert_not_called()
                    self.assertEqual(path.read_bytes(), b"keep me")

    def test_config_accepts_fixed_contract_only(self):
        cfg = config()
        runner.validate_training_config(cfg)
        mutations = [("training", "device", "cpu"), ("training", "require_cuda", False),
                     ("training", "amp", True), ("training", "allow_tf32", True),
                     ("training", "identity_loss_weight", 1.0), ("training", "epochs", 1),
                     ("training", "batch_size", 1), ("training", "learning_rate", 0.1),
                     ("data", "fit_subjects", [1, 7]), ("data", "selection_subjects", [8]),
                     ("data", "personal_sessions_used_for_training", True),
                     ("data", "task_labels_enter_forward", True),
                     ("data", "raw_reference_is_clean_ground_truth", True),
                     ("data", "evaluation_is_pristine_project_holdout", True),
                     ("data", "all_public_subjects_previously_available_in_project", False),
                     ("training", "seed", 20261013),
                     ("training", "identity_loss_weight", 20.0),
                     ("training", "learning_rate", 0.001),
                     ("runtime", "enabled", True), ("runtime", "production_validated", True),
                     ("augmentation", "synthetic_training_examples", 0)]
        for section, key, value in mutations:
            bad = deepcopy(cfg)
            bad[section][key] = value
            with self.subTest(key=key), self.assertRaises((runner.TrainingV11Error, helpers.TrainingV10Error)):
                runner.validate_training_config(bad)

    def test_checked_metadata_hash_and_ready_state_required(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = config()
            state = metadata(cfg, folder)
            api = Mock()
            api.load_config.return_value = cfg
            api.preflight.return_value = state
            with patch.object(runner.importlib, "import_module", return_value=api):
                self.assertEqual(runner.checked_metadata(Path(folder))[2], state)
                for key, value in (("status", "failed"), ("config_sha256", "bad"),
                                   ("training_lock_sha256", "bad"), ("waveforms_loaded", True),
                                   ("cuda_initialized", True)):
                    api.preflight.return_value = {**state, key: value}
                    with self.subTest(key=key), self.assertRaises((runner.TrainingV11Error, helpers.TrainingV10Error)):
                        runner.checked_metadata(Path(folder))
            api.load_reference_windows.assert_not_called()

    def test_missing_lock_fails_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            api = Mock()
            api.load_config.return_value = config()
            api.preflight.side_effect = FileNotFoundError("missing independent lock")
            with patch.object(runner, "ROOT", Path(folder)), patch.object(runner.importlib, "import_module", return_value=api), patch.object(runner, "run_training") as train, redirect_stderr(io.StringIO()):
                self.assertEqual(runner.main(["--run"]), 2)
                train.assert_not_called()

    def test_cuda_unavailable_never_falls_back(self):
        fake = Mock()
        fake.cuda.is_available.return_value = False
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), patch.object(runner.importlib, "import_module", return_value=fake) as importer:
            with self.assertRaisesRegex((runner.TrainingV11Error, helpers.TrainingV10Error), "CUDA required"):
                runner.initialize_cuda(config())
            importer.assert_called_once_with("torch")
            fake.device.assert_not_called()
            fake.set_num_threads.assert_not_called()

    def test_cuda_initialization_enforces_determinism_and_tf32_off(self):
        fake = Mock()
        fake.cuda.is_available.return_value = True
        fake.cuda.device_count.return_value = 1
        fake.device.return_value = types.SimpleNamespace(type="cuda", index=0)
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), patch.object(runner.importlib, "import_module", return_value=fake):
            runner.initialize_cuda(config())
        fake.use_deterministic_algorithms.assert_called_once_with(True, warn_only=False)
        fake.set_float32_matmul_precision.assert_called_once_with("highest")
        fake.cuda.manual_seed_all.assert_called_once_with(20261014)
        self.assertIs(fake.backends.cudnn.benchmark, False)
        self.assertIs(fake.backends.cudnn.deterministic, True)
        self.assertIs(fake.backends.cudnn.allow_tf32, False)
        self.assertIs(fake.backends.cuda.matmul.allow_tf32, False)

    def test_invalid_cublas_setting_rejected_before_torch(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "wrong"}), patch.object(runner.importlib, "import_module") as importer, self.assertRaises((runner.TrainingV11Error, helpers.TrainingV10Error)):
            runner.initialize_cuda(config())
        importer.assert_not_called()

    def test_fit_and_selection_seeds_are_fixed_disjoint_and_complete(self):
        fit = {helpers.augmentation_seed(20261014, i, epoch=e) for e in range(1, 25) for i in range(5)}
        selection = {helpers.augmentation_seed(20261014, i) for i in range(5)}
        self.assertEqual(len(fit), 120)
        self.assertTrue(fit.isdisjoint(selection))
        self.assertEqual(selection, set(range(20361014, 20361019)))
        for i, e in ((-1, 1), (5, 1), (0, 0), (0, 25)):
            with self.assertRaises((runner.TrainingV11Error, helpers.TrainingV10Error)):
                helpers.augmentation_seed(1, i, epoch=e)

    def test_selection_metric_weight_and_inclusive_gate(self):
        metrics = dict(zip(runner.CONDITIONS, [0.002, 0.1, 0.2, 0.3, 0.4]))
        summary = helpers.selection_summary(metrics, config())
        self.assertAlmostEqual(summary["corrupted_mean_nmse"], 0.25)
        self.assertAlmostEqual(summary["selection_loss"], (100 * 0.002 + 1.0) / 5)
        self.assertTrue(summary["identity_gate_passed"])
        metrics["identity"] = 0.0020001
        self.assertFalse(helpers.selection_summary(metrics, config())["identity_gate_passed"])

    def test_selection_rejects_nan_negative_missing_conditions(self):
        for value in (float("nan"), float("inf"), -0.1):
            values = {c: 0.0 for c in runner.CONDITIONS}
            values["gaussian"] = value
            with self.assertRaises((runner.TrainingV11Error, helpers.TrainingV10Error)):
                helpers.selection_summary(values, config())
        with self.assertRaises((runner.TrainingV11Error, helpers.TrainingV10Error)):
            helpers.selection_summary({"identity": 0.0}, config())

    def test_eligible_ranking_and_failed_fallback_use_separate_objectives(self):
        old = {"identity_gate_passed": True, "corrupted_mean_nmse": 0.2, "selection_loss": 0.3}
        candidate = {"identity_gate_passed": False, "corrupted_mean_nmse": 0.1, "selection_loss": 0.25}
        self.assertFalse(runner.preferred_candidate(candidate, old, eligible=True))
        self.assertTrue(runner.preferred_candidate(candidate, old, eligible=False))
        candidate["identity_gate_passed"] = True
        candidate["selection_loss"] = 0.4
        self.assertTrue(runner.preferred_candidate(candidate, old, eligible=True))
        self.assertFalse(runner.preferred_candidate(candidate, old, eligible=False))
        self.assertFalse(runner.preferred_candidate(old, old, eligible=True))
        self.assertFalse(runner.preferred_candidate(old, old, eligible=False))

    def test_exclusive_publication_complete_no_replace_or_temp_leaks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new" / "checkpoint.bin"
            runner.publish_exclusive(path, lambda stream: stream.write(b"complete"))
            self.assertEqual(path.read_bytes(), b"complete")
            self.assertEqual(list(path.parent.iterdir()), [path])
            with self.assertRaises((runner.TrainingV11Error, helpers.TrainingV10Error)):
                runner.publish_exclusive(path, lambda stream: stream.write(b"overwrite"))
            self.assertEqual(path.read_bytes(), b"complete")

    def test_writer_failure_never_publishes_partial_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.bin"
            def fail(stream):
                stream.write(b"partial")
                raise OSError("write failed")
            with self.assertRaises(OSError):
                runner.publish_exclusive(path, fail)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_exclusive_publication_race_preserves_competing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.bin"
            def racing_writer(stream):
                stream.write(b"ours")
                path.write_bytes(b"competitor")
            with self.assertRaises(FileExistsError):
                runner.publish_exclusive(path, racing_writer)
            self.assertEqual(path.read_bytes(), b"competitor")
            self.assertEqual(list(path.parent.iterdir()), [path])


class SyntheticArrayFixture:
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        cls.np = np

    def refs(self, n):
        t = self.np.arange(512, dtype=self.np.float64)
        channel = self.np.sin(2 * self.np.pi * t / 32)
        channel -= channel.mean()
        channel /= self.np.sqrt(self.np.mean(channel**2))
        return self.np.tile(channel.astype(self.np.float32), (n, 8, 1))


class SyntheticArrayContracts(SyntheticArrayFixture, unittest.TestCase):
    def test_only_fit_and_selection_load_and_provenance_retained(self):
        api = Mock()
        cfg = config()
        def load(subjects, root):
            return self.refs(len(subjects)), [{"dataset": "Wearable", "subject_id": s,
                "trial_index": 0, "electrode_id": 1} for s in subjects]
        api.load_reference_windows.side_effect = load
        api.synthetic_references.return_value = self.refs(480)
        fit, fit_ids, synthetic_ids, selection, selection_ids = runner.load_training_references(api, self.np, cfg, root=Path("unused"))
        self.assertEqual([call.args[0] for call in api.load_reference_windows.call_args_list], [[1, 2, 3, 4], [5, 6]])
        self.assertEqual(fit.shape, (484, 8, 512))
        self.assertEqual(selection.shape, (2, 8, 512))
        self.assertEqual(len(fit_ids), 4)
        self.assertEqual(len(selection_ids), 2)
        self.assertEqual(len(synthetic_ids), 480)
        self.assertEqual(fit_ids[0]["electrode_id"], 1)

    def test_reference_evaluation_personal_nonunit_and_duplicate_rejected(self):
        ids = [{"dataset": "Wearable", "subject_id": 1, "trial_index": 0}]
        for ref, identities in ((self.refs(1), [{**ids[0], "subject_id": 7}]),
                                (self.refs(1), [{**ids[0], "dataset": "personal"}]),
                                (self.refs(1) * 2, ids),
                                (self.refs(1).astype(self.np.float64), ids),
                                (self.refs(2), ids * 2)):
            with self.assertRaises((runner.TrainingV11Error, helpers.TrainingV10Error)):
                helpers._validate_references(self.np, ref, identities, (1,))

    def test_corruption_is_not_renormalized_and_labels_never_passed(self):
        cfg, api = config(), Mock()
        reference = self.refs(2)
        api.corrupt.side_effect = lambda ref, condition, seed, cfg: ref.copy() if condition == "identity" else ref * 3
        arrays = runner.condition_inputs(api, self.np, reference, cfg, epoch=1)
        self.assertEqual(len(arrays), 5)
        self.assertTrue(self.np.array_equal(arrays[1], reference * 3))
        self.assertTrue(all(call.args[0] is reference and len(call.args) == 4
                            for call in api.corrupt.call_args_list))
        self.assertEqual([call.args[2] for call in api.corrupt.call_args_list], list(range(20262014, 20262019)))
        api.corrupt.side_effect = lambda ref, condition, seed, cfg: ref * 2
        with self.assertRaisesRegex((runner.TrainingV11Error, helpers.TrainingV10Error), "identity corruption"):
            runner.condition_inputs(api, self.np, reference, cfg)


class SyntheticTorchContracts(SyntheticArrayFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        cls.torch = torch

    def test_weighted_fit_loss_true_backward_and_condition_balanced(self):
        torch, np = self.torch, self.np
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shift = torch.nn.Parameter(torch.tensor(1.0))
                self.seen = 0
            def forward(self, signal):
                self.seen += len(signal)
                return signal + self.shift
        model = Tiny()
        cfg = config()
        cfg["training"]["batch_size"] = 3  # Exercise partial last batch in helper.
        refs = self.refs(2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        loss = runner.fit_epoch(torch, np, model, optimizer, refs,
            [refs.copy() for _ in runner.CONDITIONS], cfg, torch.device("cpu"), 1)
        self.assertAlmostEqual(loss, (100 + 4) / 5, places=5)
        self.assertEqual(model.seen, 10)
        self.assertIsNotNone(model.shift.grad)
        self.assertLessEqual(float(model.shift.grad.abs()), 1.00001)

    def test_selection_uses_fixed_raw_scale_mean_and_eval_mode(self):
        torch = self.torch
        model = torch.nn.Identity()
        cfg = config()
        cfg["training"]["batch_size"] = 2
        refs = self.refs(3)
        values = [refs.copy()] + [refs * 3 for _ in range(4)]
        summary = runner.evaluate_selection(torch, model, refs, values, cfg, torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertTrue(summary["identity_gate_passed"])
        self.assertEqual(summary["identity_mean_normalized_change"], 0)
        self.assertAlmostEqual(summary["corrupted_mean_nmse"], 4.0, places=5)
        self.assertEqual(summary["condition_nmse"], summary["input_condition_nmse"])

    def test_prediction_refuses_wrong_shape_nonfinite_and_dtype(self):
        torch = self.torch
        x = torch.zeros(2, 8, 512)
        for result in (torch.zeros(1, 8, 512), torch.ones_like(x) * float("nan"), x.double(), {}):
            with self.assertRaises((runner.TrainingV11Error, helpers.TrainingV10Error)):
                helpers._prediction(torch, lambda signal: result, x)

    def test_snapshot_is_detached_cpu_and_independent(self):
        model = self.torch.nn.Linear(2, 1)
        state = runner._snapshot(model)
        with self.torch.no_grad():
            model.weight.add_(10)
        self.assertEqual(str(state["weight"].device), "cpu")
        self.assertFalse(state["weight"].requires_grad)
        self.assertFalse(self.torch.equal(state["weight"], model.weight))

    def mock_training(self, root, *, eligible=True, mutate_lock=False, fail_report=False):
        """Exercise publication/selection with all actual epoch computation mocked."""
        torch = self.torch
        cfg = config()
        state = metadata(cfg, root)
        api = Mock()
        api.preflight.return_value = state
        model = torch.nn.Linear(1, 1, bias=False)
        epoch = [0]
        def fake_fit(*args):
            epoch[0] += 1
            with torch.no_grad():
                model.weight.fill_(epoch[0])
            if mutate_lock and epoch[0] == 1:
                (Path(root) / runner.LOCK).write_bytes(b'changed')
            return 0.5
        def fake_eval(*args):
            if eligible and epoch[0] in (1, 3):
                identity, corrupted = (0.001, 0.3) if epoch[0] == 1 else (0.002, 0.2)
            else:
                identity, corrupted = 0.003, (0.001 if epoch[0] == 2 else 0.4)
            return helpers.selection_summary(dict(zip(runner.CONDITIONS, [identity] + [corrupted] * 4)), cfg)
        sources = [{"dataset": "Wearable", "subject_id": 1, "trial_index": 0}]
        refs = self.refs(1)
        data = (refs, sources, [], refs, [{**sources[0], "subject_id": 5}])
        api.build_model.return_value = model
        data_api = Mock()
        real_import = importlib.import_module
        def fake_import(name, package=None):
            return data_api if name == "eeg_tools.ssvep_v10_data" else real_import(name, package)
        publish = runner.publish_exclusive
        publication_order = []
        def record_publish(path, writer):
            publication_order.append(path.name)
            if fail_report and path.name.endswith('.json'):
                raise OSError('report write failed')
            return publish(path, writer)
        with patch.object(runner, "initialize_cuda", return_value=(torch, torch.device("cpu"))), \
             patch.object(runner, "load_training_references", return_value=data), \
             patch.object(runner, "condition_inputs", return_value=[refs] * 5), \
             patch.object(runner, "fit_epoch", side_effect=fake_fit), \
             patch.object(runner, "evaluate_selection", side_effect=fake_eval), \
             patch.object(runner.importlib, "import_module", side_effect=fake_import), \
             patch.object(torch.cuda, "get_device_name", return_value="mocked CPU test"), \
             patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
             patch.object(runner, "publish_exclusive", side_effect=record_publish), redirect_stdout(io.StringIO()):
            report = runner.run_training(api, cfg, state, root)
        self.assertEqual(publication_order, [Path(runner.CHECKPOINT).name, Path(runner.REPORT).name])
        return report

    def test_mocked_run_saves_identity_eligible_checkpoint_even_if_fallback_lower(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = self.mock_training(root)
            checkpoint = self.torch.load(root / runner.CHECKPOINT, weights_only=True, map_location="cpu")
            self.assertEqual(report["selected_epoch"], 3)
            self.assertEqual(checkpoint["selected_epoch"], 3)
            self.assertEqual(checkpoint["state_dict"]["weight"].item(), 3)
            self.assertTrue(checkpoint["selection_gate_passed"])
            self.assertEqual(len(report["epochs"]), 24)
            self.assertEqual(report["checkpoint_sha256"], runner.file_sha(root / runner.CHECKPOINT))
            self.assertEqual(json.loads((root / runner.REPORT).read_text(encoding="utf-8")), report)
            self.assertEqual(report["evaluation_subjects_loaded"], [])
            self.assertFalse(report["personal_data_loaded"])
            self.assertTrue(report["prior_v10_failed"])
            self.assertTrue(report["public_evaluation_not_pristine"])
            self.assertFalse(report["evaluation_is_pristine_project_holdout"])
            self.assertIn("identity weight 100", report["loss_definition"])
            self.assertEqual(checkpoint["config"], config())
            self.assertEqual(checkpoint["model_version"], "learned_denoise_v11_personal")
            self.assertEqual(checkpoint["training_lock_sha256"], runner.file_sha(root / runner.LOCK))
            self.assertTrue(all(t.device.type == "cpu" for t in checkpoint["state_dict"].values()))
            self.assertFalse(report["runtime_enabled"])
            self.assertFalse(report["production_validated"])

    def test_mocked_no_eligible_run_persists_failed_gate_and_lowest_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = self.mock_training(root, eligible=False)
            checkpoint = self.torch.load(root / runner.CHECKPOINT, weights_only=True, map_location="cpu")
            self.assertEqual(report["selected_epoch"], 2)
            self.assertFalse(report["selection_gate_passed"])
            self.assertFalse(checkpoint["selection_gate_passed"])
            self.assertEqual(checkpoint["state_dict"]["weight"].item(), 2)
            self.assertIn("gate_failed", report["selection_reason"])

    def test_mocked_changed_lock_refuses_all_success_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex((runner.TrainingV11Error, helpers.TrainingV10Error), "bindings changed"):
                self.mock_training(root, mutate_lock=True)
            self.assertFalse((root / runner.CHECKPOINT).exists())
            self.assertFalse((root / runner.REPORT).exists())

    def test_mocked_report_failure_never_leaves_success_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(OSError, "report write failed"):
                self.mock_training(root, fail_report=True)
            self.assertTrue((root / runner.CHECKPOINT).exists())
            self.assertFalse((root / runner.REPORT).exists())
            with self.assertRaisesRegex((runner.TrainingV11Error, helpers.TrainingV10Error), "already exists"):
                runner.assert_new_outputs(root)

    def test_checkpoint_primitives_are_weights_only_loadable(self):
        torch = self.torch
        cfg = config()
        checkpoint = {"schema_version": 1, "model_version": cfg["model_version"], "config": cfg,
            "config_sha256": runner.canonical_sha(cfg), "training_lock_sha256": "a" * 64,
            "state_dict": runner._snapshot(torch.nn.Linear(2, 1)), "selected_epoch": 1,
            "selection_gate_passed": False, "runtime_enabled": False, "production_validated": False}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.pt"
            runner.publish_exclusive(path, lambda stream: torch.save(checkpoint, stream))
            loaded = torch.load(path, weights_only=True, map_location="cpu")
        self.assertEqual(loaded["config_sha256"], runner.canonical_sha(cfg))
        self.assertFalse(loaded["selection_gate_passed"])
        self.assertEqual(loaded["state_dict"].keys(), checkpoint["state_dict"].keys())


if __name__ == "__main__":
    unittest.main()

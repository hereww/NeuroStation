"""Synthetic/in-memory tests only. Never opens NPZ/PT or runs actual training.

Run with -B and unittest (no pytest cache/log artifacts are created).
"""
from __future__ import annotations

import contextlib
import hashlib
import tempfile
import copy
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Required before torch import in this test process as well.
if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
    raise RuntimeError("conflicting test CUBLAS_WORKSPACE_CONFIG")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from eeg_tools import ssvep_denoise_v6 as core
from eeg_tools import ssvep_v6_contract as api
from scripts import train_ssvep_denoiser_v6 as trainer

torch.set_num_threads(4)
CFG = json.loads((ROOT / "configs/learned_denoiser_v6_ssvep.json").read_text(encoding="utf-8"))
LOCK_SHA = "a" * 64


def synthetic_records(domain, subjects, trials=1):
    """Shape-minimized synthetic records, never real data objects/files."""
    rng = np.random.default_rng(19)
    start = CFG["start_samples"][domain]
    n = start + 256
    records = []
    for sid in subjects:
        signal = rng.normal(size=(trials, n, 8)).astype(np.float64)
        records.append(SimpleNamespace(dataset=domain, subject_id=sid,
                                      sample_rate_hz=CFG["sample_rates_hz"][domain], signal=signal,
                                      npz_sha256="b" * 64, sidecar_sha256="c" * 64))
    return records


def synthetic_summary(domain, ratio=0.8, identity=0.0):
    count = 360 if domain == core.DOMAINS[0] else 480
    result = {}
    for condition in core.CONDITIONS:
        is_identity = condition == "identity"
        result[condition] = {
            "case_count": count, "mean_input_normalized_mse": 0.0 if is_identity else 1.0,
            "mean_output_normalized_mse": identity if is_identity else ratio,
            "ratio_of_mean_reduction": None if is_identity else 1.0 - ratio,
            "mean_identity_normalized_change": identity if is_identity else None,
            "mean_protected_complex_energy_ratio": 0.0 if is_identity else None,
            "mean_protected_weighted_abs_phase_radians": 0.0 if is_identity else None,
            "undefined_phase_case_count": 0,
        }
    return result


def synthetic_losses(cases):
    return {"total": 0.8, "corrupted": 0.8, "identity": 0.0, "change": 0.0,
            "case_count": cases, "corrupted_count": cases * 4 // 5, "identity_count": cases // 5}


def synthetic_report():
    # Read only the class-local, hash-bound synthetic metadata, never history.
    prereg = core._frozen_prereg()
    records = []
    domain_reports = {}
    for domain in core.DOMAINS:
        for sid in range(1, 7):
            path = f"{api.REFERENCE_DIR}/{domain}/subject_{sid:03d}.npz"
            records.append(SimpleNamespace(dataset=domain, subject_id=sid, sample_rate_hz=core.RATES[domain],
                                           npz_sha256=prereg["bound_files"][path],
                                           sidecar_sha256=prereg["bound_files"][path + ".manifest.json"]))
        history = []
        for epoch in range(1, 19):
            summary = synthetic_summary(domain)
            gates = api.gate_summary(summary, CFG)
            key = core.selection_key(summary, gates, 0.8, epoch)
            history.append({"epoch": epoch,
                            "train_loss": synthetic_losses(3600 if domain == core.DOMAINS[0] else 4800),
                            "internal_loss": synthetic_losses(1800 if domain == core.DOMAINS[0] else 2400),
                            "conditions": summary, "gates": gates, "selection_key": list(key), "best_epoch": 1})
        domain_reports[domain] = {"dataset": domain, "sample_rate_hz": core.RATES[domain],
                                  "fit_subjects": [1, 2, 3, 4], "internal_validation_subjects": [5, 6],
                                  "fit_case_count": 3600 if domain == core.DOMAINS[0] else 4800,
                                  "internal_case_count": 1800 if domain == core.DOMAINS[0] else 2400,
                                  "best_epoch": 1, "best_selection_key": history[0]["selection_key"],
                                  "internal_gates_passed": True, "selection_status": "internal_gates_passed",
                                  "history": history}
    state = SimpleNamespace(config=CFG, lock_sha256=LOCK_SHA)
    return trainer.make_training_report(CFG, state, records, domain_reports)


def synthetic_checkpoint():
    ck = core.checkpoint_metadata(CFG, synthetic_report(), LOCK_SHA)
    ck["models"] = {domain: core.build_model(CFG, core.RATES[domain]).state_dict() for domain in core.DOMAINS}
    return ck


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)

    def test_zero_init_exact_identity_both_native_rates(self):
        x = torch.randn(3, 8, 256)
        for fs in (250, 256):
            with self.subTest(fs=fs):
                model = core.build_model(CFG, fs)
                self.assertTrue(torch.equal(model(x), x))
                self.assertEqual(torch.count_nonzero(model.output.weight).item(), 0)
                self.assertEqual(torch.count_nonzero(model.output.bias).item(), 0)
                self.assertNotIn("keep_mask", model.state_dict())
                self.assertEqual(len(model.blocks), 6)
                self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))

    def test_native_masks_differ_and_inclusive_boundaries(self):
        a, b = core.build_model(CFG, 256), core.build_model(CFG, 250)
        self.assertFalse(torch.equal(a.keep_mask, b.keep_mask))
        self.assertFalse(a.keep_mask[8].item())
        self.assertFalse(a.keep_mask[40].item())
        self.assertTrue(a.keep_mask[7].item())
        self.assertTrue(a.keep_mask[41].item())
        self.assertTrue(b.keep_mask[8].item())  # 7.8125 Hz, not 8 Hz.
        self.assertFalse(b.keep_mask[9].item())
        self.assertFalse(b.keep_mask[40].item())
        self.assertTrue(b.keep_mask[41].item())

    def test_fft_protection_nonzero_output_no_postclip(self):
        x = torch.randn(2, 8, 256)
        for fs in (250, 256):
            model = core.build_model(CFG, fs)
            with torch.no_grad():
                model.output.weight.normal_(std=0.7)
                model.output.bias.normal_(std=0.3)
                pred = model(x)
            delta = pred - x
            spectrum = torch.fft.rfft(delta)
            protected = spectrum[..., ~model.keep_mask].abs().square().sum()
            total = spectrum.abs().square().sum()
            self.assertGreater(total.item(), 1.0)
            # Numerical float32 addition/FFT leakage; well below frozen 1e-6 gate.
            self.assertLess((protected / total).item(), 1e-10)
            ref = x[0].numpy().astype(np.float64)
            out = api.restore_window(ref, x[0].numpy(), pred[0].numpy(), np.ones((8, 1)))
            metrics = api.case_metrics(ref, ref, out, identity=True, sample_rate_hz=fs)
            self.assertLessEqual(metrics["protected_complex_energy_ratio"], 1e-6)
            self.assertLessEqual(metrics["protected_weighted_abs_phase_radians"], 0.02)

    def test_gradient_reaches_output_and_second_step_backbone(self):
        model = core.build_model(CFG, 256)
        x, target = torch.randn(4, 8, 256), torch.randn(4, 8, 256)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001,
                                       foreach=False, fused=False)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            core.compute_loss(pred, target, x, torch.tensor([True, False, False, False]), CFG)["total"].backward()
            self.assertGreater(model.output.weight.grad.abs().sum().item(), 0.0)
            if step == 1:
                self.assertGreater(model.input.weight.grad.abs().sum().item(), 0.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        self.assertGreater(torch.count_nonzero(model.output.weight).item(), 0)
        self.assertFalse(torch.equal(model(x), x))

    def test_shape_dtype_finite_rejection(self):
        model = core.build_model(CFG, 256)
        for x in (torch.zeros(8, 256), torch.zeros(1, 7, 256), torch.zeros(1, 8, 255),
                  torch.zeros(0, 8, 256), torch.zeros(1, 8, 256, dtype=torch.float64),
                  torch.full((1, 8, 256), float("nan")), torch.full((1, 8, 256), float("inf"))):
            with self.subTest(shape=x.shape, dtype=x.dtype):
                with self.assertRaises(ValueError):
                    model(x)

    def test_frozen_architecture_and_rate(self):
        for fs in (True, 256.0, 128):
            with self.assertRaises(ValueError):
                core.build_model(CFG, fs)
        bad = copy.deepcopy(CFG)
        bad["model"]["dropout"] = 0.1
        with self.assertRaises(api.V6Error):
            core.build_model(bad, 256)
        bad = copy.deepcopy(CFG)
        bad["runtime"]["enabled"] = True
        with self.assertRaises(api.V6Error):
            core.build_model(bad, 256)


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.target = torch.randn(7, 8, 256)
        self.noisy = self.target + torch.randn(7, 8, 256) * 0.4
        self.pred = self.target + torch.randn(7, 8, 256) * 0.2
        self.identity = torch.tensor([True, True, False, False, False, False, False])

    def test_formula_and_counts(self):
        loss = core.compute_loss(self.pred, self.target, self.noisy, self.identity, CFG)
        var = (self.target - self.target.mean(-1, keepdim=True)).square().mean(-1, keepdim=True) + 1e-8
        out = ((self.pred - self.target).square() / var).mean((1, 2))
        baseline = ((self.noisy - self.target).square() / var).mean((1, 2))
        expected = (out[~self.identity] / baseline[~self.identity].clamp(min=0.01)).mean()
        expected += 20 * out[self.identity].mean() + 0.001 * (self.pred - self.noisy).square().mean()
        self.assertTrue(torch.allclose(loss["total"], expected))
        self.assertEqual((loss["case_count"], loss["identity_count"], loss["corrupted_count"]), (7, 2, 5))

    def test_missing_groups_are_zero_and_floor_is_point01(self):
        for value, empty in ((False, "identity"), (True, "corrupted")):
            loss = core.compute_loss(self.pred, self.target, self.target,
                                     torch.full((7,), value, dtype=torch.bool), CFG)
            self.assertEqual(loss[empty].item(), 0.0)
            if not value:
                var = self.target.var(-1, correction=0, keepdim=True) + 1e-8
                expected = ((self.pred - self.target).square() / var).mean() / 0.01
                self.assertTrue(torch.allclose(loss["corrupted"], expected))

    def test_aggregate_by_group_case_counts_not_batch_means(self):
        complete = core.compute_loss(self.pred, self.target, self.noisy, self.identity, CFG)
        acc = core.LossAccumulator()
        for part in (slice(0, 1), slice(1, 3), slice(3, 7)):
            acc.update(core.compute_loss(self.pred[part], self.target[part], self.noisy[part], self.identity[part], CFG))
        result = acc.result()
        for key in ("total", "identity", "corrupted", "change"):
            self.assertAlmostEqual(result[key], complete[key].item(), places=6)
        self.assertEqual(result["identity_count"], 2)
        self.assertEqual(result["corrupted_count"], 5)
        with self.assertRaises(ValueError):
            core.LossAccumulator().result()

    def test_identity_boolean_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            core.compute_loss(self.pred, self.target, self.noisy, self.identity.to(torch.int64), CFG)
        bad = self.pred.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            core.compute_loss(bad, self.target, self.noisy, self.identity, CFG)


class SelectorTests(unittest.TestCase):
    def key(self, *, eligible=True, ratio=0.8, identity=0.0, loss=1.0, epoch=1):
        return core.selection_key(synthetic_summary(core.DOMAINS[0], ratio, identity),
                                  {"all_passed": eligible}, loss, epoch)

    def test_full_lexicographic_order(self):
        self.assertLess(self.key(ratio=100.0), self.key(eligible=False, ratio=0.01))
        self.assertLess(self.key(ratio=0.7, identity=1.0), self.key(ratio=0.8))
        self.assertLess(self.key(identity=0.0, loss=100.0), self.key(identity=0.1, loss=0.0))
        self.assertLess(self.key(loss=0.1, epoch=18), self.key(loss=0.2, epoch=1))
        self.assertLess(self.key(epoch=1), self.key(epoch=2))
        failed = [self.key(eligible=False, ratio=ratio, epoch=e) for e, ratio in ((1, 0.9), (2, 0.8), (3, 0.8))]
        self.assertEqual(min(failed)[-1], 2)

    def test_worst_family_ratio_of_means(self):
        summary = synthetic_summary(core.DOMAINS[0])
        summary["transient"]["mean_input_normalized_mse"] = 2.0
        summary["transient"]["mean_output_normalized_mse"] = 3.0
        summary["transient"]["ratio_of_mean_reduction"] = -0.5
        key = core.selection_key(summary, {"all_passed": False}, 1.0, 5)
        self.assertEqual(key[1], 1.5)

    def test_epoch_zero_nonfinite_and_bool_refused(self):
        for epoch in (0, 19, True):
            with self.assertRaises(ValueError):
                self.key(epoch=epoch)
        with self.assertRaises(ValueError):
            self.key(loss=float("nan"))
        with self.assertRaises(ValueError):
            core.selection_key(synthetic_summary(core.DOMAINS[0]), {"all_passed": 1}, 1.0, 1)


class TrainerSyntheticTests(unittest.TestCase):
    def test_all_conditions_fixed_window_and_epoch_seeds(self):
        records = synthetic_records(core.DOMAINS[0], (1, 2, 3, 4))
        dataset = trainer.ReferenceWindowDataset(records, core.DOMAINS[0], CFG, purpose="training", epoch=1)
        self.assertEqual(len(dataset), 20)
        self.assertEqual([dataset[i]["condition"] for i in range(5)], list(core.CONDITIONS))
        first = dataset[1]
        self.assertEqual(first["ref64"].dtype, np.float64)
        self.assertEqual(first["noisy"].dtype, np.float32)
        self.assertTrue(np.array_equal(first["ref64"], records[0].signal[0, 38:294].T))
        dataset.set_epoch(2)
        second = dataset[1]
        self.assertNotEqual(first["seed"], second["seed"])
        self.assertFalse(np.array_equal(first["noisy64"], second["noisy64"]))
        with self.assertRaises(api.V6Error):
            dataset.set_epoch(0)

    def test_internal_fixed_seeds_float64_restore_and_no_grad_both_rates(self):
        for domain in core.DOMAINS:
            records = synthetic_records(domain, (5, 6))
            dataset = trainer.ReferenceWindowDataset(records, domain, CFG, purpose="internal_selection", epoch=0)
            before = dataset[1]
            dataset.set_epoch(0)
            self.assertTrue(np.array_equal(before["noisy64"], dataset[1]["noisy64"]))
            with self.assertRaises(api.V6Error):
                dataset.set_epoch(1)
            loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)
            model = core.build_model(CFG, core.RATES[domain])
            grad_modes = []
            hook = model.register_forward_pre_hook(lambda m, args: grad_modes.append(torch.is_grad_enabled()))
            try:
                with patch.object(api, "case_metrics", wraps=api.case_metrics) as metrics:
                    loss, rows = trainer.run_epoch(model, loader, CFG, "cpu")
            finally:
                hook.remove()
            self.assertEqual(loss["case_count"], 10)
            self.assertTrue(grad_modes and not any(grad_modes))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertFalse(model.training)
            self.assertTrue(all(call.kwargs["sample_rate_hz"] == core.RATES[domain] for call in metrics.call_args_list))
            for row in rows:
                if row["condition"] == "identity":
                    self.assertEqual(row["output_normalized_mse"], 0.0)
                else:
                    self.assertEqual(row["output_normalized_mse"], row["input_normalized_mse"])
            summary = api.summarize_cases(rows)
            self.assertEqual(set(summary), set(core.CONDITIONS))
            self.assertTrue(all(v["case_count"] == 2 for v in summary.values()))
            self.assertFalse(api.gate_summary(summary, CFG)["all_passed"])

    def test_mocked_18_epoch_orchestration_no_gpu_data_or_actual_training(self):
        """Test schedule/selection with dummy calls, not a real training run."""
        domain = core.DOMAINS[0]
        # MagicMock supplies __len__ without ever reading/materializing records.
        from unittest.mock import MagicMock
        fit, internal = MagicMock(), MagicMock()
        fit.__len__.return_value, internal.__len__.return_value = 3600, 1800
        dummy_model = core.build_model(CFG, 256)
        summaries = []
        for epoch in range(1, 19):
            ratio = 0.99 if epoch == 1 else 0.9 if epoch == 2 else 0.8
            summary = synthetic_summary(domain, ratio=ratio)
            if epoch == 18:
                # Better noisy error, but identity gate failed: cannot displace eligible3.
                summary = synthetic_summary(domain, ratio=0.1, identity=0.01)
            summaries.append(summary)
        def epoch_result(model, loader, cfg, device, optimizer=None):
            if optimizer is not None:
                return synthetic_losses(3600), []
            return synthetic_losses(1800), [None] * 1800
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(api, "verify_frozen") as verify, \
             patch.object(trainer, "set_seed"), \
             patch.object(trainer, "ReferenceWindowDataset", side_effect=[fit, internal]) as dataset, \
             patch.object(torch.utils.data, "DataLoader", side_effect=[object(), object()]) as loaders, \
             patch.object(core, "build_model", return_value=dummy_model), \
             patch.object(dummy_model, "to", return_value=dummy_model) as move, \
             patch.object(trainer, "run_epoch", side_effect=epoch_result) as runner, \
             patch.object(api, "summarize_cases", side_effect=summaries), \
             patch.object(torch.optim, "AdamW", wraps=torch.optim.AdamW) as optimizer, \
             contextlib.redirect_stdout(io.StringIO()):
            best_state, report = trainer.train_domain([], domain, CFG, "cuda:0", frozen_state=SimpleNamespace(check_only=False, stage="training"))
        self.assertEqual(report["best_epoch"], 3)
        self.assertIs(report["internal_gates_passed"], True)
        self.assertEqual(len(report["history"]), 18)
        self.assertEqual(fit.set_epoch.call_args_list, [unittest.mock.call(e) for e in range(1, 19)])
        self.assertEqual(dataset.call_args_list[0].kwargs, {"purpose": "training", "epoch": 1})
        self.assertEqual(dataset.call_args_list[1].kwargs, {"purpose": "internal_selection", "epoch": 0})
        self.assertEqual(runner.call_count, 36)
        self.assertEqual(verify.call_count, 19)
        self.assertEqual(move.call_args.kwargs, {"device": "cuda:0", "dtype": torch.float32})
        self.assertEqual(optimizer.call_args.kwargs,
                         {"lr": 0.0005, "weight_decay": 0.0001, "foreach": False, "fused": False})
        for call in loaders.call_args_list:
            self.assertEqual(call.kwargs["batch_size"], 64)
            self.assertEqual(call.kwargs["num_workers"], 0)
            self.assertFalse(call.kwargs["drop_last"])
        name = next(iter(best_state))
        self.assertEqual(best_state[name].device.type, "cpu")
        self.assertFalse(best_state[name].requires_grad)
        self.assertNotEqual(best_state[name].data_ptr(), dummy_model.state_dict()[name].data_ptr())

    def test_check_only_preflight_cannot_train_or_publish_even_when_cuda_mocked(self):
        state = SimpleNamespace(check_only=True, stage="training", config=CFG)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(api, "verify_frozen", side_effect=AssertionError("check-only cannot proceed")), \
             patch.object(trainer, "_core", wraps=trainer._core) as model_module:
            with self.assertRaisesRegex(api.V6Error, "check-only cannot train/publish"):
                trainer.train_domain([], core.DOMAINS[0], CFG, "cuda:0", frozen_state=state)
            with self.assertRaisesRegex(api.V6Error, "check-only cannot train/publish"):
                trainer.publish_training_bundle(state, {}, {})
        self.assertEqual(model_module.call_count, 0)  # Both guards precede torch/model import.

    def test_real_training_entry_rejects_cpu_before_model_or_records(self):
        with patch.object(trainer, "set_seed", side_effect=AssertionError("must not seed")), \
             patch.object(core, "build_model", side_effect=AssertionError("must not build")), \
             patch.object(api, "verify_frozen", side_effect=AssertionError("must not verify")):
            with self.assertRaises(api.V6Error):
                trainer.train_domain([], core.DOMAINS[0], CFG, "cpu", frozen_state=object())
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.cuda, "set_device", side_effect=AssertionError("must not initialize CUDA")):
            with self.assertRaises(api.V6Error):
                trainer.configure_training(CFG)


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A fixture-only pin and temporary root keep real byte/hash checks active.
        # Production constants, original preregistration and training locks stay
        # untouched. These source identities describe inert generated strings.
        cls.temp = tempfile.TemporaryDirectory(prefix="synthetic-v6-contract-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        bound = {}
        for domain in core.DOMAINS:
            for sid in range(1, 7):
                for suffix in ("", ".manifest.json"):
                    name = f"{api.REFERENCE_DIR}/{domain}/subject_{sid:03d}.npz{suffix}"
                    bound[name] = hashlib.sha256(("synthetic: " + name).encode()).hexdigest()
        cls.prereg = cls.root / api.PREREG_RELATIVE
        cls.prereg.parent.mkdir(parents=True)
        cls.prereg.write_text(json.dumps({"fixture_scope":"synthetic_contract_only_not_historical_training_evidence",
                                         "bound_files":bound}),encoding="utf-8")
        for field, value in (("ROOT", cls.root), ("PREREG_SHA256", hashlib.sha256(cls.prereg.read_bytes()).hexdigest())):
            binding = patch.object(core, field, value); binding.start(); cls.addClassCleanup(binding.stop)
        cls.base = synthetic_checkpoint()

    def test_synthetic_preregistration_tamper_is_still_rejected(self):
        before = self.prereg.read_bytes()
        try:
            self.prereg.write_bytes(before + b" ")
            with self.assertRaisesRegex(ValueError, "preregistration SHA mismatch"):
                core.validate_checkpoint(self.base, CFG, LOCK_SHA)
        finally:
            self.prereg.write_bytes(before)

    def test_strict_cpu_validator_and_inmemory_weights_only_roundtrip(self):
        stream = io.BytesIO()
        torch.save(self.base, stream)
        stream.seek(0)
        ck = torch.load(stream, weights_only=True, map_location="cpu")
        before = torch.random.get_rng_state().clone()
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("no CUDA")):
            models = core.validate_checkpoint(ck, CFG, LOCK_SHA)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(set(models), set(core.DOMAINS))
        for domain, model in models.items():
            self.assertEqual(model.sample_rate_hz, core.RATES[domain])
            self.assertFalse(model.training)
            self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()))

    def assert_invalid(self, change):
        ck = copy.deepcopy(self.base)
        change(ck)
        with self.assertRaises(ValueError):
            core.validate_checkpoint(ck, CFG, LOCK_SHA)

    def test_exact_top_keys_types_and_provenance(self):
        self.assert_invalid(lambda ck: ck.update(extra=True))
        self.assert_invalid(lambda ck: ck.pop("clean_ground_truth_used"))
        self.assert_invalid(lambda ck: ck.update(schema_version=True))
        for flag in core.PROVENANCE:
            for bad in (True, 0):
                with self.subTest(flag=flag, value=bad):
                    self.assert_invalid(lambda ck, f=flag, v=bad: ck.update({f: v}))
                    self.assert_invalid(lambda ck, f=flag, v=bad: ck["training_report"].update({f: v}))
        self.assert_invalid(lambda ck: ck["config"]["runtime"].update(enabled=True))
        self.assert_invalid(lambda ck: ck["config"]["model"].update(post_projection_clip=0))

    def test_binding_and_domain_schema(self):
        for key in ("config_sha256", "training_lock_sha256", "reference_manifest_sha256", "partition_lock_sha256"):
            self.assert_invalid(lambda ck, k=key: ck.update({k: "f" * 64}))
        self.assert_invalid(lambda ck: ck["models"].pop(core.DOMAINS[0]))
        self.assert_invalid(lambda ck: ck["models"].update(extra={}))
        self.assert_invalid(lambda ck: ck["training_report"]["sample_rates_hz"].update(Wearable=256))
        self.assert_invalid(lambda ck: ck["training_report"]["source_binding"][0].update(subject_id=7))
        self.assert_invalid(lambda ck: ck["training_report"]["source_binding"][0].update(npz_sha256="f" * 64))
        self.assert_invalid(lambda ck: ck["training_report"]["source_binding"].pop())
        self.assert_invalid(lambda ck: ck["training_report"].update(created_at_utc="not a timestamp"))
        with self.assertRaises(ValueError):
            core.validate_checkpoint(self.base, CFG, "invalid")

    def test_state_finite_dtype_shape_keys(self):
        domain = core.DOMAINS[0]
        name = next(iter(self.base["models"][domain]))
        def replace(value):
            return lambda ck: ck["models"][domain].update({name: value})
        original = self.base["models"][domain][name]
        for value in (original.double(), original.to(torch.bool), original.to(torch.int32), original[:1],
                      torch.full_like(original, float("nan")), torch.full_like(original, float("inf")),
                      original.clone().requires_grad_(True)):
            with self.subTest(dtype=value.dtype, shape=value.shape):
                self.assert_invalid(replace(value))
        self.assert_invalid(lambda ck: ck["models"][domain].pop(name))
        self.assert_invalid(lambda ck: ck["models"][domain].update(keep_mask=torch.ones(129)))

    def test_epoch_gate_loss_and_condition_provenance(self):
        domain = core.DOMAINS[0]
        def row(ck):
            return ck["training_report"]["domain_reports"][domain]["history"][0]
        self.assert_invalid(lambda ck: row(ck).update(epoch=0))
        self.assert_invalid(lambda ck: row(ck)["internal_loss"].update(case_count=True))
        self.assert_invalid(lambda ck: row(ck)["internal_loss"].update(total=float("nan")))
        self.assert_invalid(lambda ck: row(ck)["internal_loss"].update(total=1.0))
        self.assert_invalid(lambda ck: row(ck)["conditions"]["identity"].update(case_count=True))
        self.assert_invalid(lambda ck: row(ck)["conditions"]["identity"].update(mean_protected_complex_energy_ratio=float("nan")))
        self.assert_invalid(lambda ck: row(ck)["conditions"].pop("transient"))
        self.assert_invalid(lambda ck: row(ck)["gates"].update(all_passed=False))
        self.assert_invalid(lambda ck: row(ck).update(selection_key=[0, 0.8, 0.0, 0.8, 18]))
        self.assert_invalid(lambda ck: ck["training_report"]["domain_reports"][domain].update(best_epoch=2))

    def test_no_eligible_still_valid_failed_best_no_runtime_acceptance(self):
        report = synthetic_report()
        for domain in core.DOMAINS:
            dr = report["domain_reports"][domain]
            for entry in dr["history"]:
                entry["conditions"] = synthetic_summary(domain, ratio=1.1)
                entry["gates"] = api.gate_summary(entry["conditions"], CFG)
                entry["selection_key"] = list(core.selection_key(entry["conditions"], entry["gates"], 0.8, entry["epoch"]))
            dr.update(best_selection_key=dr["history"][0]["selection_key"], internal_gates_passed=False,
                      selection_status="gates_failed")
        ck = core.checkpoint_metadata(CFG, report, LOCK_SHA)
        ck["models"] = self.base["models"]
        self.assertEqual(set(core.validate_checkpoint(ck, CFG, LOCK_SHA)), set(core.DOMAINS))
        self.assertIs(ck["runtime_enabled"], False)
        self.assertIs(ck["production_validated"], False)

    def test_publication_selfload_uses_one_bundle_and_rechecks_freeze(self):
        state = SimpleNamespace(root=ROOT, config=CFG, lock_sha256=LOCK_SHA, check_only=False, stage="training")
        with patch.object(api, "verify_frozen") as verify, \
             patch.object(api, "assert_new_outputs") as outputs, \
             patch.object(api, "publish_bundle") as publish:
            trainer.publish_training_bundle(state, self.base["models"], self.base["training_report"])
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(outputs.call_count, 1)
        self.assertEqual(publish.call_count, 1)
        payload = publish.call_args.args[1]
        self.assertEqual(set(payload), {api.CHECKPOINT_RELATIVE, api.TRAIN_REPORT_RELATIVE})
        self.assertTrue(all(type(v) is bytes for v in payload.values()))
        self.assertEqual(json.loads(payload[api.TRAIN_REPORT_RELATIVE]), self.base["training_report"])


    def test_changed_freeze_or_collision_cannot_publish(self):
        state = SimpleNamespace(root=ROOT, config=CFG, lock_sha256=LOCK_SHA, check_only=False, stage="training")
        with patch.object(api, "verify_frozen", side_effect=[None, api.V6Error("freeze changed")]), \
             patch.object(api, "assert_new_outputs"), patch.object(api, "publish_bundle") as publish:
            with self.assertRaisesRegex(api.V6Error, "freeze changed"):
                trainer.publish_training_bundle(state, self.base["models"], self.base["training_report"])
            publish.assert_not_called()
        with patch.object(api, "verify_frozen"), \
             patch.object(api, "assert_new_outputs", side_effect=api.V6Error("existing output")), \
             patch.object(torch, "save", side_effect=AssertionError("no serialization before collision")), \
             patch.object(api, "publish_bundle") as publish:
            with self.assertRaisesRegex(api.V6Error, "existing output"):
                trainer.publish_training_bundle(state, self.base["models"], self.base["training_report"])
            publish.assert_not_called()


class CLISafetyTests(unittest.TestCase):
    def test_collision_before_preflight_core_cuda_or_data(self):
        with patch.object(api, "assert_new_outputs", side_effect=api.V6Error("existing output")), \
             patch.object(api, "preflight", side_effect=AssertionError("must not preflight")), \
             patch.object(trainer, "_core", side_effect=AssertionError("must not import torch/model")), \
             patch.object(api, "load_training_records", side_effect=AssertionError("must not read waveforms")):
            for args in (["--run"], ["--check-only"], []):
                with self.assertRaisesRegex(api.V6Error, "existing output"):
                    trainer.main(args)

    def test_check_only_uses_preflight_report_no_core_binary_or_cuda(self):
        state = SimpleNamespace(report=lambda: {"check_only": True, "cuda_initialized": False,
                                               "waveforms_loaded": False, "checkpoint_loaded": False})
        with patch.object(api, "assert_new_outputs"), \
             patch.object(api, "preflight", return_value=state) as preflight, \
             patch.object(trainer, "_core", side_effect=AssertionError("must not import torch/model")), \
             patch.object(api, "load_training_records", side_effect=AssertionError("must not read waveforms")), \
             patch.object(api, "verify_frozen", side_effect=AssertionError("must not full verify")):
            for args in (["--check-only"], []):
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(trainer.main(args), 0)
                self.assertTrue(json.loads(stream.getvalue())["check_only"])
        self.assertTrue(all(c.kwargs["check_only"] is True for c in preflight.call_args_list))

    def test_cuda_configuration_fixed_and_no_real_cuda_initialization(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch, "set_num_threads") as threads, \
             patch.object(torch, "use_deterministic_algorithms") as deterministic, \
             patch.object(torch.backends, "cuda", SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))), \
             patch.object(torch.backends, "cudnn", SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False)), \
             patch.object(torch.cuda, "set_device") as device, \
             patch.object(trainer, "set_seed") as seed:
            self.assertEqual(trainer.configure_training(CFG), "cuda:0")
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
        threads.assert_called_once_with(4)
        deterministic.assert_called_once_with(True)
        device.assert_called_once_with(0)
        seed.assert_called_once_with(2026100706)

    def test_main_cuda_refusal_precedes_waveform_loader(self):
        state = SimpleNamespace(config=CFG, check_only=False, stage="training")
        with patch.object(api, "assert_new_outputs"), patch.object(api, "preflight", return_value=state), \
             patch.object(api, "verify_frozen"), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(api, "load_training_records", side_effect=AssertionError("no waveforms")):
            with self.assertRaisesRegex(api.V6Error, "no CPU fallback"):
                trainer.main(["--run"])

    def test_unfrozen_or_changed_freeze_refused_before_training(self):
        with patch.object(api, "assert_new_outputs"), \
             patch.object(api, "preflight", side_effect=api.V6Error("missing freeze")), \
             patch.object(trainer, "configure_training", side_effect=AssertionError("no CUDA")), \
             patch.object(api, "load_training_records", side_effect=AssertionError("no data")):
            with self.assertRaisesRegex(api.V6Error, "missing freeze"):
                trainer.main(["--run"])
        state = SimpleNamespace(config=CFG, check_only=False, stage="training")
        with patch.object(api, "assert_new_outputs"), patch.object(api, "preflight", return_value=state), \
             patch.object(api, "verify_frozen", side_effect=api.V6Error("freeze changed")), \
             patch.object(trainer, "configure_training", side_effect=AssertionError("no CUDA")):
            with self.assertRaisesRegex(api.V6Error, "freeze changed"):
                trainer.main(["--run"])

    def test_override_arguments_rejected_before_any_reads(self):
        with patch.object(trainer, "_contract", side_effect=AssertionError("no contract read")):
            for option in ("--config", "--reference-dir", "--cohort", "--device", "--output-checkpoint", "--output-report"):
                with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as err:
                        trainer.main([option, "override"])
                    self.assertEqual(err.exception.code, 2)

    def test_import_is_torch_free_in_fresh_process(self):
        code = "import sys; from scripts import train_ssvep_denoiser_v6; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        process = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_cublas_conflict_precedes_torch_import(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}), \
             patch.object(trainer, "_core", side_effect=AssertionError("must not import torch")):
            with self.assertRaisesRegex(RuntimeError, "conflicting"):
                trainer.configure_training(CFG)


if __name__ == "__main__":
    unittest.main()

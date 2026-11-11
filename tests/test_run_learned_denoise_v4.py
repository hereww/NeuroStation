"""Synthetic-only offline runner/launcher tests; never train or read holdout."""
from __future__ import annotations
from tests.synthetic_research_fixtures import research_contract
from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from eeg_tools import learned_denoise_v4 as core
from scripts import run_learned_denoise_v4 as run
from scripts import evaluate_denoiser_v4 as ev


def fake_apply(bundle, signal, valid_mask=None):
    valid = np.isfinite(signal) if valid_mask is None else valid_mask & np.isfinite(signal)
    rejected = np.broadcast_to(~np.all(valid, axis=0), signal.shape).copy()
    # Change only safe channels with one low-band residual (single window).
    residual = .01 * np.sin(2*np.pi*np.arange(len(signal))/500)[:, None]
    candidate = signal.copy()
    candidate[:, ~rejected[0]] += residual
    replacement = valid & ~rejected
    return SimpleNamespace(denoised=candidate.copy(), model_denoised=candidate,
        artifact_probability=np.where(rejected, np.nan, .8), confidence=np.where(rejected, 0., .2),
        replacement_mask=replacement, valid_mask=valid, rejected_mask=rejected, diagnostics={"synthetic": True})


def fixture_root(root):
    for rel in ("scripts/run_learned_denoise_v4.py", "scripts/evaluate_denoiser_v4.py",
                "scripts/verify_denoiser_v4_artifacts.py", "eeg_tools/learned_denoise_v4.py"):
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"synthetic")
    config = ev.read_json(ev.ROOT / "configs/learned_denoiser_v4.json")
    interface = research_contract('v4-interface.json')
    for rel, value in (("configs/learned_denoiser_v4.json", config),
                       ("analysis_outputs/learned_denoiser_v4_interface_contract.json", interface)):
        p=root/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value), encoding="utf-8")
    return config


class InputAndReconstructionTests(unittest.TestCase):
    def test_numpy_csv_npz_explicit_rate_and_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = np.arange(1200, dtype=float).reshape(600,2)
            mask = np.ones(signal.shape, dtype=bool); mask[21,1]=False
            np.save(root / "raw.npy", signal)
            np.save(root / "mask.npy", mask)
            np.savetxt(root / "raw.csv", signal, delimiter=",")
            np.savez(root / "raw.npz", signal=signal, valid_mask=mask, sample_rate_hz=np.array(250))
            raw, valid, metadata = run.load_input(root / "raw.npy", 250, mask_path=root / "mask.npy")
            np.testing.assert_array_equal(raw, signal); np.testing.assert_array_equal(valid, mask)
            raw, valid, _ = run.load_input(root / "raw.npz", 250)
            np.testing.assert_array_equal(valid, mask)
            raw, valid, _ = run.load_input(root / "raw.csv", 250, all_finite_valid=True)
            np.testing.assert_array_equal(raw, signal)
            self.assertEqual(metadata["layout"], "T,C_no_transpose")
            with self.assertRaises(ValueError): run.load_input(root / "raw.npy", 250)
            with self.assertRaises(ValueError): run.load_input(root / "raw.npy", 200, all_finite_valid=True)
            with self.assertRaises(ValueError): run.load_input(root / "raw.npz", 250, all_finite_valid=True)
            np.savez(root / "bad.npz", signal=signal, valid_mask=mask, sample_rate_hz=np.array([250]))
            with self.assertRaises(ValueError): run.load_input(root / "bad.npz", 250)

    def test_mask_no_broadcast_binary_nonfinite_and_complex(self):
        with self.assertRaises(ValueError): run.binary_mask(np.ones(500), (500,2))
        with self.assertRaises(ValueError): run.binary_mask(np.full((500,2), .5), (500,2))
        with self.assertRaises(ValueError): run.signal_array(np.ones((8,1250,40,4)))
        with self.assertRaises(ValueError): run.signal_array(np.ones(500, dtype=complex))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "raw.npy"
            np.save(p, np.array([1., np.nan]))
            with self.assertRaises(ValueError): run.load_input(p, 250, all_finite_valid=True)
        self.assertEqual(run.signal_array(np.arange(501)).shape, (501,1))

    def test_chunked_order_independent_channel_tail_and_invalid_unchanged(self):
        t = np.arange(1007)/250
        raw = np.stack([3+np.sin(2*np.pi*12*t), -2+2*np.cos(2*np.pi*20*t)], axis=1)
        raw[521,1] = np.nan
        mask = np.isfinite(raw)
        batch = run.process_array(None, raw, mask, chunk_size=3, apply=fake_apply)
        single = run.process_array(None, raw, mask, chunk_size=1, apply=fake_apply)
        for name in ev.ARRAY_FIELDS:
            np.testing.assert_array_equal(batch[name], single[name])
        np.testing.assert_array_equal(batch["denoised"][1000:], raw[1000:])
        np.testing.assert_array_equal(batch["denoised"][500:1000,1], raw[500:1000,1])
        self.assertTrue(batch["rejected_mask"][1000:].all())
        self.assertFalse(batch["replacement_mask"][1000:].any())
        self.assertTrue(np.isnan(batch["artifact_probability"][1000:]).all())
        self.assertEqual(batch["diagnostics"]["full_channel_window_count"], 4)
        self.assertEqual(batch["diagnostics"]["tail_samples_per_channel_unchanged"], 7)
        tiny = run.process_array(None, raw[:10], mask[:10], apply=Mock(side_effect=AssertionError("tail should not infer")))
        np.testing.assert_array_equal(tiny["denoised"], raw[:10])

    @unittest.skipIf(core.torch is None, "optional CPU torch unavailable")
    def test_actual_core_chunked_vs_direct_no_mix_without_training(self):
        torch = core.torch
        torch.set_num_threads(2)
        class FixedResidual(torch.nn.Module):
            def forward(self, x):
                t = torch.arange(500, dtype=x.dtype, device=x.device)/250
                residual = .04*torch.sin(2*torch.pi*2*t)[None,None]
                return x-residual, torch.full((len(x),), 3., dtype=x.dtype, device=x.device)
        bundle = core.ModelBundle(FixedResidual().eval(), deepcopy(core.DEFAULT_CONFIG), .5, "cpu")
        t=np.arange(1007)/250
        raw=np.stack([2+np.sin(2*np.pi*12*t), -3+2*np.cos(2*np.pi*20*t)],axis=1)
        mask=np.ones(raw.shape,dtype=bool); mask[501,1]=False
        expected=core.apply_model_to_array(bundle,raw,valid_mask=mask)
        actual=run.process_array(bundle,raw,mask,chunk_size=3)
        for name in ev.ARRAY_FIELDS:
            np.testing.assert_array_equal(actual[name],getattr(expected,name))


class OfflineCLITests(unittest.TestCase):
    def invoke(self, args):
        with contextlib.redirect_stdout(io.StringIO()): return run.main(args)

    def test_missing_checkpoint_check_only_honest_and_no_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fixture_root(root)
            with patch.object(run,"load_bundle",side_effect=AssertionError("must not load missing checkpoint")), \
                 patch.object(run,"load_input",side_effect=AssertionError("must not parse source")):
                self.assertEqual(self.invoke(["--root",str(root),"--check-only","--check-report","check.json"]),1)
            report=ev.read_json(root/"check.json")
            self.assertFalse(report["diagnostic_ready"])
            self.assertFalse(report["training_executed"])
            self.assertFalse(report["inference_executed"])
            self.assertEqual(report["model_load_status"],"missing_checkpoint_training_not_confirmed")
            before=(root/"check.json").read_bytes()
            self.assertEqual(self.invoke(["--root",str(root),"--check-only","--check-report","check.json"]),1)
            self.assertEqual((root/"check.json").read_bytes(),before)

    def test_success_check_does_not_parse_source_and_threads_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fixture_root(root)
            model=root/"models/denoiser_v4.pt";model.parent.mkdir();model.write_bytes(b"mock checkpoint")
            source=root/"raw.npy"; source.write_bytes(b"invalid numpy, check-only must NOT parse values")
            bundle=SimpleNamespace(device="cpu",artifact_threshold=.5)
            stub=SimpleNamespace(torch=SimpleNamespace(__version__="mock"))
            with patch.object(run,"load_bundle",return_value=(stub,bundle)) as load, \
                 patch.object(run,"load_input",side_effect=AssertionError("must not parse source")):
                self.assertEqual(self.invoke(["--root",str(root),"--check-only","--input",str(source),"--sample-rate","250","--device","cpu"]),0)
            self.assertEqual(load.call_args.args[2:],("cpu",2))

    def test_cpu_machine_cuda_error_has_no_fallback_or_package_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture_root(root)
            model=root/"models/denoiser_v4.pt";model.parent.mkdir();model.write_bytes(b"mock")
            with patch.object(run,"load_bundle",side_effect=ValueError("requested CUDA unavailable; no fallback")) as load:
                self.assertEqual(self.invoke(["--root",str(root),"--check-only","--device","cuda"]),1)
            self.assertEqual(load.call_count,1)
            self.assertEqual(load.call_args.args[2],"cuda")

    def test_offline_output_immutable_no_existing_directory_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture_root(root)
            model=root/"models/denoiser_v4.pt";model.parent.mkdir();model.write_bytes(b"mock")
            raw=(2+np.sin(2*np.pi*12*np.arange(507)/250))[:,None]
            source=root/"raw.npy";np.save(source,raw)
            before=ev.sha256(source)
            bundle=SimpleNamespace(device="cpu",artifact_threshold=.5)
            stub=SimpleNamespace(torch=SimpleNamespace(__version__="mock"),apply_model_to_array=fake_apply)
            args=["--root",str(root),"--input",str(source),"--sample-rate","250","--all-finite-valid","--output","new-output","--device","cpu"]
            with patch.object(run,"load_bundle",return_value=(stub,bundle)):
                self.assertEqual(self.invoke(args),0)
            self.assertEqual(ev.sha256(source),before)
            report=ev.read_json(root/"new-output/report.json")
            self.assertTrue(report["inference_executed"])
            self.assertFalse(report["test_evaluation_executed"])
            with np.load(root/"new-output/diagnostics.npz",allow_pickle=False) as data:
                np.testing.assert_array_equal(data["denoised"][500:],raw[500:])
            existing={p.name:p.read_bytes() for p in (root/"new-output").iterdir()}
            with patch.object(run,"load_bundle",side_effect=AssertionError("stop before existing output")):
                self.assertEqual(self.invoke(args),1)
            self.assertEqual({p.name:p.read_bytes() for p in (root/"new-output").iterdir()},existing)
            # Existing failed directory with started.json cannot be adopted.
            old=root/"old-output";old.mkdir();(old/"started.json").write_text("old")
            with patch.object(run,"load_bundle",side_effect=AssertionError("stop")):
                self.assertEqual(self.invoke(args[:-3]+["old-output","--device","cpu"]),1)
            self.assertEqual(sorted(p.name for p in old.iterdir()),["started.json"])

    def test_source_tamper_fails_before_publishing_output_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture_root(root)
            model=root/"models/denoiser_v4.pt";model.parent.mkdir();model.write_bytes(b"mock")
            source=root/"raw.npy";np.save(source,np.sin(np.arange(500))[:,None])
            def tamper(bundle, signal, valid_mask=None):
                result=fake_apply(bundle,signal,valid_mask)
                source.write_bytes(b"changed by synthetic external actor")
                return result
            stub=SimpleNamespace(torch=SimpleNamespace(__version__="mock"),apply_model_to_array=tamper)
            bundle=SimpleNamespace(device="cpu",artifact_threshold=.5)
            with patch.object(run,"load_bundle",return_value=(stub,bundle)):
                self.assertEqual(self.invoke(["--root",str(root),"--input",str(source),"--sample-rate","250", "--all-finite-valid","--output","new","--device","cpu"]),1)
            self.assertFalse((root/"new/diagnostics.npz").exists())
            self.assertEqual(ev.read_json(root/"new/report.json")["result"],"failed")

    def test_threads_positive_and_protected_output(self):
        for value in ("0","-1","1.5"):
            with self.subTest(value=value),self.assertRaises(SystemExit): self.invoke(["--check-only","--threads",value])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/"raw.npy";np.save(p,np.ones(500))
            with self.assertRaises(ValueError): run.validate_output(root,root,[p])
            with self.assertRaises(ValueError): run.validate_output(root/"datasets/new",root,[p])


class PowerShellLauncherTests(unittest.TestCase):
    def test_release_first_gpu_isolated_no_window_or_fallback(self):
        text=(ev.ROOT/"scripts/start_denoiser_v4_diagnostics.ps1").read_text(encoding="utf-8")
        self.assertLess(text.index("scripts\\validate_release_configs.py"),text.index("scripts\\run_learned_denoise_v4.py"))
        self.assertIn(".venv-ml-gpu\\Scripts\\python.exe",text)
        self.assertNotIn("Start-Process",text)
        self.assertNotIn("pip install",text)
        self.assertNotIn(".venv\\Scripts\\python.exe",text)
        self.assertIn("--check-only",text)

    @unittest.skipUnless(shutil.which("pwsh"),"PowerShell unavailable")
    def test_actual_launcher_release_gate_first_and_numeric_rate(self):
        with tempfile.TemporaryDirectory(prefix="v4 中文 fixture ") as directory:
            root=Path(directory);scripts=root/"scripts";scripts.mkdir()
            validator=scripts/"validate_release_configs.py"
            validator.write_text("from pathlib import Path\nPath('release-was-first').write_text('checked')\n",encoding="utf-8")
            runner=scripts/"run_learned_denoise_v4.py"
            runner.write_text("import sys\nfrom pathlib import Path\nassert Path('release-was-first').is_file()\nprint(sys.argv)\n",encoding="utf-8")
            command=[shutil.which("pwsh"),"-NoProfile","-NonInteractive","-File",str(ev.ROOT/"scripts/start_denoiser_v4_diagnostics.ps1"),
                     "-Root",str(root),"-PythonPath",sys.executable,"-CheckOnly","-InputPath",str(root/"synthetic.npy"),"-SampleRate","250"]
            env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"}
            result=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=30,env=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("250",result.stdout)
            (root/"release-was-first").unlink()
            validator.write_text("raise SystemExit(7)\n",encoding="utf-8")
            runner.write_text("raise AssertionError('must not run after failed release')\n",encoding="utf-8")
            result=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=30,env=env)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("Formal release configuration check failed",result.stdout)


if __name__ == "__main__":
    unittest.main()

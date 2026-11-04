"""Offline CLI contracts and synthetic pipeline only; no real model or CUDA."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_ssvep_v12_offline_analysis as cli

ROOT = Path(__file__).absolute().parents[1]
EXPECTED_FLAGS = (
    "runtime_enabled", "production_validated", "source_domain_validated",
    "denoising_efficacy_validated", "presented_matches_are_gaze_accuracy",
)


class OfflineCLIContractTests(unittest.TestCase):
    def test_import_and_help_do_not_import_scientific_packages_or_cuda(self):
        code = '''
import contextlib, io, sys
class RejectScientific:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'torch'}:
            raise AssertionError('unexpected scientific/CUDA import: ' + fullname)
sys.meta_path.insert(0, RejectScientific())
from scripts import run_ssvep_v12_offline_analysis as cli
with contextlib.redirect_stdout(io.StringIO()) as output:
    try:
        cli.main(['--help'])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError('help must exit')
assert '--acknowledge-out-of-domain' in output.getvalue()
assert 'offline research' in output.getvalue().lower()
assert not {'numpy', 'scipy', 'torch'} & sys.modules.keys()
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_required_paths_device_choices_and_default_cpu(self):
        parser = cli.build_parser()
        base = ["--session", "source", "--output", "result", "--acknowledge-out-of-domain"]
        args = parser.parse_args(base)
        self.assertEqual(args.session, Path("source"))
        self.assertEqual(args.output, Path("result"))
        self.assertEqual(args.device, "cpu")
        self.assertIs(args.acknowledge_out_of_domain, True)
        for device in ("cpu", "cuda:0"):
            with self.subTest(device=device):
                self.assertEqual(parser.parse_args(base + ["--device", device]).device, device)
        for device in ("cuda", "cuda:1", "auto"):
            with self.subTest(device=device), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parser.parse_args(base + ["--device", device])
                self.assertEqual(error.exception.code, 2)
        for argv in (["--output", "x", "--acknowledge-out-of-domain"],
                     ["--session", "x", "--acknowledge-out-of-domain"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parser.parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_missing_acknowledgement_rejected_before_pipeline(self):
        with patch.object(cli, "run_analysis") as run, contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as exit_info:
                cli.main(["--session", "missing", "--output", "missing-output"])
        self.assertEqual(exit_info.exception.code, 2)
        self.assertIn("--acknowledge-out-of-domain", error.getvalue())
        run.assert_not_called()

    def test_programmatic_entry_also_requires_acknowledgement(self):
        with self.assertRaisesRegex(ValueError, "--acknowledge-out-of-domain"):
            cli.run_analysis("missing", "missing-output")

    def test_cli_forwards_explicit_device_and_acknowledgement(self):
        report = {"usage_scope": "explicit_offline_research_only", "output_directory": "result",
                  "source_hashes_match": True}
        with patch.object(cli, "run_analysis", return_value=report) as run:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                status = cli.main(["--session", "source", "--output", "result",
                                   "--device", "cuda:0", "--acknowledge-out-of-domain"])
        self.assertEqual(status, 0)
        run.assert_called_once_with(Path("source"), Path("result"), device="cuda:0",
                                    acknowledge_out_of_domain=True)
        summary = json.loads(output.getvalue())
        for name in EXPECTED_FLAGS:
            self.assertIs(summary[name], False)

    def test_existing_output_refused_without_importing_cuda(self):
        code = '''
import contextlib, io, json, sys, tempfile
from pathlib import Path
class RejectScientific:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'torch'}:
            raise AssertionError('unexpected scientific/CUDA import: ' + fullname)
sys.meta_path.insert(0, RejectScientific())
from scripts import run_ssvep_v12_offline_analysis as cli
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / 'existing'
    output.mkdir()
    sentinel = output / 'report.json'
    sentinel.write_bytes(b'existing report, must not be changed')
    with contextlib.redirect_stderr(io.StringIO()) as error:
        status = cli.main(['--session', str(Path(directory) / 'missing'), '--output', str(output),
                           '--device', 'cuda:0', '--acknowledge-out-of-domain'])
    assert status == 1
    message = json.loads(error.getvalue())
    assert 'already exists' in message['error'], message
    assert message['complete'] is False
    assert sentinel.read_bytes() == b'existing report, must not be changed'
    assert not {'numpy', 'scipy', 'torch'} & sys.modules.keys()
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


class OfflineMockPipelineTests(unittest.TestCase):
    """Exercise real native parsing, fixed filtering, file output and reporting."""

    def setUp(self):
        import numpy as np
        from eeg_tools import ssvep_v12_runtime as runtime
        from eeg_tools.workstation import v9_offline as backend
        from scripts import diagnose_neurostation_v9_output as adapter

        self.np = np
        self.backend = backend
        self.adapter = adapter
        self.tmp = tempfile.TemporaryDirectory(prefix="v12_cli_mock_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "synthetic_native_session"
        self.source.mkdir()
        self.output = self.root / "offline_output"
        time_axis = np.arange(640) / 250
        self.signal = np.column_stack([
            (1 + channel / 10) * np.sin(2 * np.pi * 10 * time_axis)
            + 0.4 * np.sin(2 * np.pi * 50 * time_axis) + channel
            for channel in range(8)
        ])
        session = {"board": "cyton", "channel_count": 8, "sampling_rate_hz": 250,
                   "simulated": False, "recorded_samples_per_channel": len(self.signal)}
        channels = {"board": "cyton", "sampling_rate_hz": 250,
                    "channel_order": list(adapter.CHANNEL_LABELS),
                    "channels": [{"name": name} for name in adapter.CHANNEL_LABELS]}
        for name, value in (("session.json", session), ("channel_config.json", channels)):
            (self.source / name).write_text(json.dumps(value), encoding="utf-8")
        # Scrambled columns detect accidental source-channel reordering.
        fields = ["sample_index", "marker", "eeg_ch8", "eeg_ch2", "eeg_ch6", "eeg_ch4",
                  "eeg_ch1", "eeg_ch7", "eeg_ch3", "eeg_ch5"]
        with (self.source / "raw_brainflow.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fields, delimiter="\t")
            writer.writeheader()
            for index, row in enumerate(self.signal):
                record = {"sample_index": index, "marker": 0}
                record.update({f"eeg_ch{ch + 1}": value for ch, value in enumerate(row)})
                writer.writerow(record)
        extras = self.source / "additional_source_files"
        extras.mkdir()
        (extras / "notes.txt").write_bytes(b"synthetic fixture; must remain unchanged")
        self.before = self.source_bytes()
        self.model = object()
        self.metadata = {"device": "cpu", "model_version": "learned_denoise_v12_personal",
                         "device_name": "MOCK CPU", "runtime_enabled": False,
                         "production_validated": False, "source_domain_validated": False}
        self.processing = {"normalization_contract": "MOCK", "processed_samples": len(self.signal)}
        self.notch = self.signal + 0.125
        self.v12 = self.notch + 0.0625
        loader = patch.object(runtime, "load_model", return_value=(self.model, self.metadata))
        processor = patch.object(runtime, "process_recording",
                                 return_value=(self.v12, self.notch, self.processing))
        self.loader = loader.start()
        self.processor = processor.start()
        self.addCleanup(loader.stop)
        self.addCleanup(processor.stop)

    def source_bytes(self):
        return {path.relative_to(self.source).as_posix(): path.read_bytes()
                for path in self.source.rglob("*") if path.is_file()}

    def run_cli(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(["--session", str(self.source), "--output", str(self.output),
                               "--device", "cpu", "--acknowledge-out-of-domain"])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_mock_pipeline_writes_all_tracks_and_formal_research_report(self):
        with patch.object(self.adapter, "parse_raw", wraps=self.adapter.parse_raw) as parse:
            with patch.object(self.adapter, "validate_metadata", wraps=self.adapter.validate_metadata) as validate:
                status, stdout, stderr = self.run_cli()
        self.assertEqual(status, 0, stderr)
        self.assertEqual(json.loads(stdout)["status"], "complete")
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {"raw.npy", "notch.npy", "traditional.npy", "v12.npy", "report.json"})
        expected = {"raw.npy": self.signal, "notch.npy": self.notch,
                    "traditional.npy": self.backend._traditional(self.signal), "v12.npy": self.v12}
        report = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        for name, value in expected.items():
            actual = self.np.load(self.output / name, allow_pickle=False)
            self.np.testing.assert_array_equal(actual, value)
            self.assertEqual(report["artifacts"][name]["sha256"],
                             hashlib.sha256((self.output / name).read_bytes()).hexdigest())
        self.assertEqual(report["status"], "formal")
        self.assertEqual(report["completion_status"], "complete")
        self.assertEqual(report["usage_scope"], "explicit_offline_research_only")
        for name in EXPECTED_FLAGS:
            self.assertIs(report[name], False)
        self.assertIs(report["training_performed"], False)
        self.assertIs(report["classifier_executed"], False)
        self.assertIs(report["fixed_filter_benefits_attributed_to_network"], False)
        self.assertIs(report["tracks"]["traditional"]["learned"], False)
        self.assertEqual(report["tracks"]["v12"]["neural_comparison_reference"], "notch.npy")
        self.assertEqual(report["processing"], self.processing)
        self.assertEqual(report["model"], self.metadata)
        self.assertEqual(report["source_sha256_before"], report["source_sha256_after"])
        self.assertEqual(report["source_sha256_before"],
                         {name: hashlib.sha256(data).hexdigest() for name, data in self.before.items()})
        self.assertIs(report["source_hashes_match"], True)
        self.assertEqual(self.source_bytes(), self.before)
        parse.assert_called_once()
        validate.assert_called_once()
        self.loader.assert_called_once_with(cli.ROOT, device="cpu")
        self.processor.assert_called_once()
        self.np.testing.assert_array_equal(self.processor.call_args.args[0], self.signal)
        self.assertIs(self.processor.call_args.args[1], self.model)

    def test_existing_output_never_calls_model_or_changes_files(self):
        self.output.mkdir()
        sentinel = self.output / "report.json"
        sentinel.write_bytes(b"previous result")
        status, _, stderr = self.run_cli()
        self.assertEqual(status, 1)
        self.assertIn("already exists", json.loads(stderr)["error"])
        self.assertEqual(sentinel.read_bytes(), b"previous result")
        self.loader.assert_not_called()
        self.processor.assert_not_called()
        self.assertEqual(self.source_bytes(), self.before)

    def test_output_inside_source_refused_before_loading(self):
        self.output = self.source / "unsafe-output"
        status, _, stderr = self.run_cli()
        self.assertEqual(status, 1)
        self.assertIn("overlap", json.loads(stderr)["error"])
        self.assertFalse(self.output.exists())
        self.loader.assert_not_called()
        self.assertEqual(self.source_bytes(), self.before)

    def test_model_failure_leaves_no_completed_marker(self):
        self.loader.side_effect = RuntimeError("mock model unavailable")
        status, _, stderr = self.run_cli()
        self.assertEqual(status, 1)
        self.assertFalse(json.loads(stderr)["complete"])
        self.assertFalse((self.output / "report.json").exists())
        self.assertEqual(self.source_bytes(), self.before)

    def test_array_write_failure_leaves_no_completed_marker(self):
        save = self.np.save
        calls = 0

        def fail_second_save(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("mock disk write failure")
            return save(*args, **kwargs)

        with patch.object(self.np, "save", side_effect=fail_second_save):
            status, _, stderr = self.run_cli()
        self.assertEqual(status, 1)
        self.assertFalse(json.loads(stderr)["complete"])
        self.assertTrue((self.output / "raw.npy").is_file())
        self.assertFalse((self.output / "report.json").exists())
        self.assertFalse((self.output / ".report.json.tmp").exists())
        self.assertEqual(self.source_bytes(), self.before)

    def test_report_publication_failure_leaves_no_completed_marker(self):
        with patch.object(Path, "rename", side_effect=OSError("mock report publication failure")):
            status, _, stderr = self.run_cli()
        self.assertEqual(status, 1)
        self.assertFalse(json.loads(stderr)["complete"])
        self.assertTrue((self.output / "v12.npy").exists())
        self.assertFalse((self.output / "report.json").exists())
        self.assertFalse((self.output / ".report.json.tmp").exists())
        self.assertEqual(self.source_bytes(), self.before)

    def test_changed_source_after_writing_refuses_completed_marker(self):
        save = self.np.save

        def change_fixture_after_output(*args, **kwargs):
            result = save(*args, **kwargs)
            # Only this temporary fixture is changed, simulating another writer.
            (self.source / "additional_source_files" / "notes.txt").write_bytes(b"external change")
            return result

        with patch.object(self.np, "save", side_effect=change_fixture_after_output):
            status, _, stderr = self.run_cli()
        self.assertEqual(status, 1)
        self.assertIn("source SHA256 changed", json.loads(stderr)["error"])
        self.assertFalse((self.output / "report.json").exists())


if __name__ == "__main__":
    unittest.main()

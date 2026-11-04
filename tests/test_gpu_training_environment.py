from __future__ import annotations

import ast
from contextlib import nullcontext
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import check_gpu_training_environment as check


class FakeDevice:
    def __init__(self, name="cuda:0"):
        self.name = name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and self.name == other.name


class FakeTensor:
    """Logical tensor fixture; never imports torch or touches an actual GPU."""
    def __init__(self, value=.2, *, device=None, shape=(), finite=True, backward=None):
        self.value = value
        self.device = device or FakeDevice()
        self.shape = shape
        self.finite = finite
        self.backward_action = backward
        self.requires_grad = True
        self.grad = None

    @property
    def is_cuda(self):
        return str(self.device).startswith("cuda:")

    def item(self):
        return self.value

    def all(self):
        return self

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.value, device=self.device, shape=self.shape, finite=self.finite)

    def fill_(self, value):
        self.value = value
        return self

    def view(self, *shape):
        self.shape = tuple(shape)
        return self

    def repeat(self, *repetitions):
        self.shape = tuple(x * y for x, y in zip(self.shape, repetitions))
        return self

    def __sub__(self, other):
        return FakeTensor(self.value - other.value, device=self.device, shape=self.shape,
                          finite=self.finite and other.finite, backward=self.backward_action)

    def square(self):
        return FakeTensor(self.value ** 2, device=self.device, shape=self.shape,
                          finite=self.finite, backward=self.backward_action)

    def mean(self):
        return FakeTensor(self.value, device=self.device, finite=self.finite, backward=self.backward_action)

    def backward(self):
        if self.backward_action is None:
            raise RuntimeError("fixture has no backward action")
        self.backward_action()


class FakeTorch:
    def __init__(self, *, available=True, fault=None):
        self.__version__ = "fixture-2.14.1+cu130"
        self.version = SimpleNamespace(cuda="13.0")
        self.fault = fault
        self.device = Mock(side_effect=FakeDevice)
        self.cuda = SimpleNamespace(
            is_available=Mock(return_value=available), device_count=Mock(return_value=1 if available else 0),
            current_device=Mock(return_value=0), get_device_name=Mock(return_value="fixture CUDA GPU"),
            get_device_capability=Mock(return_value=(8, 9)),
            get_device_properties=Mock(return_value=SimpleNamespace(total_memory=24 * 1024 ** 3)),
            synchronize=Mock(), reset_peak_memory_stats=Mock(),
            max_memory_allocated=Mock(return_value=2048), max_memory_reserved=Mock(return_value=4096))
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(version=Mock(return_value=99999)))
        self.set_num_threads = Mock()
        self.get_num_threads = Mock(return_value=2)
        self.float32 = "fixture float32"
        self.parameters = [FakeTensor(.3), FakeTensor(.4)]
        self.model = Mock()
        self.model.parameters = Mock(return_value=iter(self.parameters))
        self.model.side_effect = self.forward
        self.nn = SimpleNamespace(Conv1d=Mock(return_value="conv"), GELU=Mock(return_value="gelu"),
                                  Sequential=Mock(return_value=self.model))
        self.optimizer = SimpleNamespace(zero_grad=Mock(), step=Mock(side_effect=self.step))
        self.optim = SimpleNamespace(SGD=Mock(return_value=self.optimizer))
        self.linspace = Mock(side_effect=self.linspace_tensor)
        self.zeros_like = Mock(side_effect=lambda t: FakeTensor(0, device=t.device, shape=t.shape))
        self.isfinite = Mock(side_effect=lambda t: FakeTensor(t.finite and math.isfinite(t.value)))
        self.equal = Mock(side_effect=lambda left, right: left.value == right.value)
        self.no_grad = Mock(side_effect=nullcontext)
        self.forward_count = 0

    def linspace_tensor(self, _a, _b, samples, *, device, dtype):
        return FakeTensor(.3, device=FakeDevice("cpu") if self.fault == "input_cpu" else device, shape=(samples,))

    def backward(self):
        if self.fault == "backward_error":
            raise RuntimeError("CUDA backward fixture error")
        for p in self.parameters:
            p.grad = FakeTensor(.1, finite=self.fault != "gradient_nan")
        if self.fault == "missing_gradient":
            self.parameters[-1].grad = None
        if self.fault == "gradient_cpu":
            self.parameters[-1].grad.device = FakeDevice("cpu")

    def forward(self, inputs):
        self.forward_count += 1
        if self.fault == "forward_error":
            raise RuntimeError("CUDA forward fixture error")
        return FakeTensor(.2, device=FakeDevice("cpu") if self.fault == "prediction_cpu" else inputs.device,
                          shape=(4, 1, 499) if self.fault == "output_shape" else inputs.shape,
                          finite=self.fault != "prediction_nan" and not (self.fault == "post_step_nan" and self.forward_count > 1),
                          backward=self.backward)

    def step(self):
        if self.fault == "optimizer_error":
            raise RuntimeError("CUDA optimizer fixture error")
        if self.fault != "optimizer_noop":
            for p in self.parameters:
                p.value -= .001
        if self.fault == "parameter_nan":
            self.parameters[-1].finite = False


def probe(torch=None, require_cuda=True):
    with patch.object(check, "package_version", side_effect=lambda name: f"fixture_{name}"):
        return check.check_environment(require_cuda=require_cuda, torch_loader=lambda: torch or FakeTorch())


class EnvironmentTests(unittest.TestCase):
    def test_mock_cuda_forward_backward_optimizer_and_metadata(self):
        torch = FakeTorch()
        result = probe(torch)
        self.assertEqual(result["check_status"], "passed")
        self.assertEqual(result["status"], "formal")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["torch"]["version"], torch.__version__)
        self.assertEqual(result["torch"]["cuda_build_version"], "13.0")
        self.assertEqual(result["cuda"]["selected_device"], "cuda:0")
        self.assertEqual(result["cuda"]["capability"], [8, 9])
        self.assertEqual(result["cuda"]["total_vram_bytes"], 24 * 1024 ** 3)
        self.assertEqual(result["cuda"]["name"], "fixture CUDA GPU")
        self.assertEqual(result["smoke"]["peak_allocated_bytes"], 2048)
        self.assertEqual(result["smoke"]["peak_reserved_bytes"], 4096)
        self.assertEqual(result["smoke"]["input_shape"], [4, 1, 500])
        self.assertTrue(result["smoke"]["parameters_updated"])
        for field in ("forward_finite", "gradients_finite", "parameters_finite"):
            self.assertTrue(result["smoke"][field])
        self.assertFalse(result["production_validated"])
        self.assertFalse(result["automatic_processing_enabled"])
        self.assertFalse(result["dataset_training_performed"])
        self.assertFalse(result["holdout_accessed"])
        self.assertEqual(result["v3_training_status"], "not_assessed")
        self.assertEqual(result["v3_model_quality_status"], "not_assessed")
        self.assertIsNone(result["cuda"]["driver_version"])
        self.assertEqual(torch.nn.Conv1d.call_count, 2)
        self.assertTrue(all(str(c.kwargs["device"]) == "cuda:0" for c in torch.nn.Conv1d.call_args_list))
        torch.optimizer.zero_grad.assert_called_once_with(set_to_none=True)
        torch.optimizer.step.assert_called_once()
        torch.set_num_threads.assert_called_once_with(2)
        self.assertEqual(torch.cuda.synchronize.call_count, 3)
        torch.cuda.reset_peak_memory_stats.assert_called_once_with(FakeDevice("cuda:0"))
        json.dumps(result, allow_nan=False)

    def test_cpu_only_fails_even_without_require_cuda_no_cpu_model(self):
        for required in (False, True):
            torch = FakeTorch(available=False)
            torch.version.cuda = None
            result = probe(torch, require_cuda=required)
            self.assertEqual(result["require_cuda"], required)
            self.assertEqual(result["exit_code"], 1)
            self.assertFalse(result["cuda"]["available"])
            self.assertEqual(result["smoke"]["status"], "not_assessed")
            self.assertEqual(result["error"]["stage"], "cuda_availability")
            self.assertFalse(result["cpu_fallback_allowed"])
            torch.nn.Sequential.assert_not_called()
            torch.cuda.synchronize.assert_not_called()
            torch.device.assert_not_called()

    def test_torch_import_error_produces_failure_report(self):
        for error in (ModuleNotFoundError("torch"), OSError("CUDA DLL load failure"), RuntimeError("broken installation")):
            with self.subTest(error=error), patch.object(check, "package_version", return_value=None):
                result = check.check_environment(torch_loader=Mock(side_effect=error), require_cuda=True)
            self.assertEqual(result["exit_code"], 1)
            self.assertEqual(result["error"]["stage"], "import_torch")
            self.assertIsNone(result["cuda"]["available"])
            self.assertEqual(result["smoke"]["status"], "not_assessed")
            self.assertIn("executable", result["python"])
            json.dumps(result, allow_nan=False)

    def test_inconsistent_cuda_available_rejected_without_smoke(self):
        for field in ("build", "count"):
            torch = FakeTorch()
            if field == "build":
                torch.version.cuda = None
            else:
                torch.cuda.device_count.return_value = 0
            result = probe(torch)
            self.assertEqual(result["exit_code"], 1)
            torch.nn.Sequential.assert_not_called()

    def test_cuda_enumeration_and_device_errors_are_captured(self):
        for api in ("is_available", "device_count", "current_device", "get_device_properties", "get_device_name", "get_device_capability"):
            torch = FakeTorch()
            getattr(torch.cuda, api).side_effect = RuntimeError(f"fixture {api}")
            result = probe(torch)
            self.assertEqual(result["check_status"], "failed")
            self.assertEqual(result["exit_code"], 1)
            self.assertIn(api, result["error"]["message"])
            torch.nn.Sequential.assert_not_called()

    def test_backward_gradient_failure_and_optimizer_error(self):
        for fault, stage in (("backward_error", "backward"), ("missing_gradient", "backward"),
                             ("gradient_nan", "backward"), ("gradient_cpu", "backward"),
                             ("optimizer_error", "optimizer_step"), ("optimizer_noop", "synchronize_after_step"),
                             ("parameter_nan", "synchronize_after_step")):
            torch = FakeTorch(fault=fault)
            result = probe(torch)
            self.assertEqual(result["exit_code"], 1, fault)
            self.assertEqual(result["smoke"]["status"], "failed", fault)
            self.assertEqual(result["error"]["stage"], stage, fault)
            self.assertTrue(result["smoke"]["forward_finite"])
            if stage == "backward":
                torch.optimizer.step.assert_not_called()

    def test_forward_nonfinite_cpu_and_shape_fail_closed(self):
        for fault, stage in (("input_cpu", "construct_synthetic_cuda_input"), ("forward_error", "forward"),
                             ("prediction_cpu", "forward"), ("prediction_nan", "forward"), ("output_shape", "forward"),
                             ("post_step_nan", "forward_after_step")):
            torch = FakeTorch(fault=fault)
            result = probe(torch)
            self.assertEqual(result["exit_code"], 1, fault)
            self.assertEqual(result["error"]["stage"], stage, fault)
            if fault != "post_step_nan":
                torch.optimizer.step.assert_not_called()

    def test_nonfinite_loss_fails_before_backward_or_optimizer(self):
        torch = FakeTorch()
        with patch.object(FakeTensor, "mean", return_value=FakeTensor(float("nan"))):
            result = probe(torch)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("loss contains nonfinite", result["error"]["message"])
        self.assertIsNone(result["smoke"]["loss_before_step"])
        self.assertIsNone(result["smoke"]["gradients_finite"])
        torch.optimizer.step.assert_not_called()
        json.dumps(result, allow_nan=False)

    def test_parameter_device_and_trainability_are_not_assumed(self):
        for fault in ("cpu", "not_trainable", "empty"):
            torch = FakeTorch()
            if fault == "cpu":
                torch.parameters[0].device = FakeDevice("cpu")
            elif fault == "not_trainable":
                torch.parameters[0].requires_grad = False
            else:
                torch.model.parameters.return_value = iter([])
            result = probe(torch)
            self.assertEqual(result["exit_code"], 1, fault)
            self.assertEqual(result["error"]["stage"], "construct_cuda_conv1d", fault)
            torch.model.assert_not_called()
            torch.optimizer.step.assert_not_called()

    def test_build_or_oom_failure_is_reported_without_fallback(self):
        torch = FakeTorch()
        torch.nn.Conv1d.side_effect = RuntimeError("CUDA out of memory fixture")
        result = probe(torch)
        self.assertEqual(result["error"]["stage"], "construct_cuda_conv1d")
        self.assertEqual(result["exit_code"], 1)
        torch.nn.Conv1d.assert_called_once()
        self.assertEqual(str(torch.nn.Conv1d.call_args.kwargs["device"]), "cuda:0")
        self.assertFalse(result["cpu_fallback_allowed"])

    def test_synchronization_memory_and_peak_evidence_failures(self):
        for api in ("synchronize", "reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
            torch = FakeTorch()
            getattr(torch.cuda, api).side_effect = RuntimeError(api)
            result = probe(torch)
            self.assertEqual(result["exit_code"], 1)
            self.assertEqual(result["smoke"]["status"], "failed")
        for allocated, reserved in ((0, 4096), (-1, 4096), (4096, 2048)):
            torch = FakeTorch()
            torch.cuda.max_memory_allocated.return_value = allocated
            torch.cuda.max_memory_reserved.return_value = reserved
            result = probe(torch)
            self.assertEqual(result["error"]["stage"], "read_peak_memory")
            self.assertEqual(result["exit_code"], 1)

    def test_synchronized_timing_is_reported_not_device_enumeration_only(self):
        clock = Mock(side_effect=[10., 12., 15., 16.])
        with patch.object(check, "package_version", return_value=None):
            result = check.check_environment(torch_loader=lambda: FakeTorch(), clock=clock)
        self.assertEqual(result["smoke"]["elapsed_seconds"], 3.)
        self.assertEqual(result["elapsed_seconds"], 6.)
        self.assertEqual(clock.call_count, 4)
        clock = Mock(side_effect=[10., 12., float("nan"), 16.])
        with patch.object(check, "package_version", return_value=None):
            result = check.check_environment(torch_loader=lambda: FakeTorch(), clock=clock)
        self.assertEqual(result["exit_code"], 1)
        self.assertIsNone(result["smoke"]["elapsed_seconds"])

    def test_package_discovery_missing_package_does_not_crash(self):
        with patch.object(check.metadata, "version", return_value="pinned") as version:
            self.assertEqual(check.package_version("torch"), "pinned")
            version.assert_called_once_with("torch")
        with patch.object(check.metadata, "version", side_effect=check.metadata.PackageNotFoundError):
            self.assertIsNone(check.package_version("torch"))


class CLITests(unittest.TestCase):
    def test_success_and_cpu_failure_report_written_new_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            for available in (False, True):
                output = Path(tmp) / "new" / f"result_{available}.json"
                torch = FakeTorch(available=available)
                with patch.object(check, "import_torch", return_value=torch), patch.object(check, "package_version", return_value=None), patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                    code = check.main(["--output", str(output), "--require-cuda"])
                self.assertEqual(code, 0 if available else 1)
                report = json.loads(output.read_text(encoding="utf-8"))
                self.assertTrue(report["require_cuda"])
                self.assertEqual(code, report["exit_code"])

    def test_missing_or_broken_torch_still_creates_failure_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "missing_torch.json"
            with patch.object(check, "import_torch", side_effect=ModuleNotFoundError("no torch")), patch.object(check, "package_version", return_value=None), patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                code = check.main(["--output", str(output), "--require-cuda"])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(report["error"]["stage"], "import_torch")
            self.assertEqual(report["smoke"]["status"], "not_assessed")

    def test_existing_report_not_overwritten_no_torch_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "old.json"
            output.write_bytes(b"old report or protected raw bytes")
            before = output.read_bytes()
            with patch.object(check, "check_environment") as probe_mock, patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(check.main(["--output", str(output), "--require-cuda"]), 2)
            probe_mock.assert_not_called()
            self.assertEqual(output.read_bytes(), before)

    def test_exclusive_create_race_failure_before_probe(self):
        with patch.object(Path, "open", side_effect=FileExistsError("race")), patch.object(check, "check_environment") as probe_mock, patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(check.main(["--output", "race.json"]), 2)
        probe_mock.assert_not_called()

    def test_output_missing_or_permission_error_does_not_probe(self):
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            check.main([])
        with patch.object(Path, "mkdir", side_effect=PermissionError("not writable")), patch.object(check, "check_environment") as probe_mock, patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(check.main(["--output", "denied.json"]), 2)
        probe_mock.assert_not_called()

    def test_no_nvidia_smi_install_training_or_holdout_dependencies(self):
        source = Path(check.__file__).read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(n and n.split(".")[0] in {"subprocess", "requests", "numpy", "scipy", "eeg_tools"} for n in imports))
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        for forbidden in ("load", "save", "fit", "train", "system", "Popen", "check_output"):
            self.assertNotIn(forbidden, calls)
        self.assertNotIn("nvidia-smi", source)
        options = check.build_parser()._option_string_actions
        self.assertEqual(set(options), {"-h", "--help", "--output", "--require-cuda"})


class DeliveryContractTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(check.__file__).resolve().parents[1]

    def test_requirements_are_isolated_requested_pins_not_acquisition_stack(self):
        requirements = (self.root / "requirements-ml-gpu.txt").read_text(encoding="utf-8-sig")
        active = [line.strip() for line in requirements.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        self.assertEqual(active, ["--extra-index-url https://download.pytorch.org/whl/cu130",
                                  "torch==2.14.1+cu130", "numpy==2.5.3", "scipy==1.18.1"])
        ignore = (self.root / ".gitignore").read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual(ignore.count(".venv-ml-gpu/"), 1)

    def test_document_keeps_smoke_and_data_training_evidence_separate(self):
        document = next((self.root / "docs").glob("*v3.md"))
        text = document.read_text(encoding="utf-8-sig")
        for marker in ("formal", "not_assessed", "NOT 数据训练", "not new holdout", "independent=false",
                       "CC BY-NC 4.0", "数据许可尚未确认", "target_axis_mapping_confirmed=false",
                       "不自动加到", "保留 raw", "不是有效去噪证据", "不接入采集工作站自动处理"):
            self.assertIn(marker, text)
        source_hash = hashlib.sha256(Path(check.__file__).read_bytes()).hexdigest()
        self.assertIn(source_hash, text)
        self.assertIn("135,680", text)
        self.assertIn("3.6544200999996974", text)
        # Commands must use absolute interpreter/script/report/test paths.
        for command in text.split("```powershell")[1:]:
            command = command.split("```")[0]
            self.assertIn("C:\\Users\\Administrator\\Documents\\ChatGPT", command)
            self.assertNotIn("& .", command)


if __name__ == "__main__":
    unittest.main()

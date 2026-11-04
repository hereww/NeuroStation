"""Explicit, isolated CUDA readiness diagnostic; synthetic Conv1d smoke only.

No package installs, driver changes, data reads, checkpoints, holdout access,
CPU fallback, workstation integration, or overwrite of existing reports.
CUDA unavailable is nonzero even without --require-cuda; the flag records the
caller's explicit prerequisite, not permission to silently fall back to CPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

# This affects only this diagnostic process. Respect pre-existing caller limits.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

SMOKE_SHAPE = (4, 1, 500)
SMOKE_LEARNING_RATE = 0.001


class GPUEnvironmentError(RuntimeError):
    """Environment operations did not satisfy the explicit CUDA smoke contract."""


def import_torch() -> Any:
    # Lazy import permits a JSON failure report on CPU-only/missing/DLL-broken installs.
    return importlib.import_module("torch")


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def finite(tensor: Any, torch: Any, label: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise GPUEnvironmentError(f"{label} contains nonfinite values")


def on_cuda(tensor: Any, device: Any, label: str) -> None:
    if not tensor.is_cuda or tensor.device != device:
        raise GPUEnvironmentError(f"{label} is not on required {device}; CPU fallback forbidden")


def run_cuda_smoke(torch: Any, device: Any, state: dict,
                   clock: Callable[[], float] = time.perf_counter) -> None:
    """One disposable SGD step on synthetic [4,1,500]; NEVER the v2/v3 model.

    state is mutated so a failure report retains the exact failed stage.
    CUDA synchronization brackets the timing; peak is PyTorch process allocated
    and reserved memory after reset, NOT system-wide GPU usage or capacity.
    """
    state["stage"] = "synchronize_before_smoke"
    torch.cuda.synchronize(device)
    state["stage"] = "reset_peak_memory_stats"
    torch.cuda.reset_peak_memory_stats(device)
    started = clock()
    state["stage"] = "construct_cuda_conv1d"
    model = torch.nn.Sequential(
        torch.nn.Conv1d(1, 4, kernel_size=5, padding=2, device=device, dtype=torch.float32),
        torch.nn.GELU(),
        torch.nn.Conv1d(4, 1, kernel_size=5, padding=2, device=device, dtype=torch.float32),
    )
    parameters = list(model.parameters())
    if not parameters or not all(p.requires_grad for p in parameters):
        raise GPUEnvironmentError("smoke requires trainable parameters")
    # Constants avoid seed-dependent conclusions; no scientific labels/data are read.
    with torch.no_grad():
        for parameter in parameters:
            on_cuda(parameter, device, "parameter")
            parameter.fill_(0.01)
    before = [p.detach().clone() for p in parameters]
    state["stage"] = "construct_synthetic_cuda_input"
    inputs = torch.linspace(-1, 1, SMOKE_SHAPE[-1], device=device, dtype=torch.float32)
    inputs = inputs.view(1, 1, SMOKE_SHAPE[-1]).repeat(SMOKE_SHAPE[0], 1, 1)
    target = torch.zeros_like(inputs)
    on_cuda(inputs, device, "synthetic input")
    on_cuda(target, device, "synthetic target")
    finite(inputs, torch, "input")
    optimizer = torch.optim.SGD(parameters, lr=SMOKE_LEARNING_RATE)
    optimizer.zero_grad(set_to_none=True)
    state["stage"] = "forward"
    prediction = model(inputs)
    on_cuda(prediction, device, "forward prediction")
    if tuple(prediction.shape) != SMOKE_SHAPE:
        raise GPUEnvironmentError("Conv1d output shape differs from synthetic EEG500 input")
    finite(prediction, torch, "forward prediction")
    state["forward_finite"] = True
    loss = (prediction - target).square().mean()
    on_cuda(loss, device, "loss")
    finite(loss, torch, "loss")
    state["loss_before_step"] = float(loss.detach().item())
    state["stage"] = "backward"
    loss.backward()
    for parameter in parameters:
        if parameter.grad is None:
            raise GPUEnvironmentError("backward did not produce a gradient for every parameter")
        on_cuda(parameter.grad, device, "gradient")
        finite(parameter.grad, torch, "gradient")
    state["gradients_finite"] = True
    state["stage"] = "optimizer_step"
    optimizer.step()
    state["stage"] = "synchronize_after_step"
    torch.cuda.synchronize(device)
    for parameter in parameters:
        on_cuda(parameter, device, "updated parameter")
        finite(parameter, torch, "updated parameter")
    state["parameters_finite"] = True
    state["parameters_updated"] = any(not torch.equal(p.detach(), old) for p, old in zip(parameters, before))
    if not state["parameters_updated"]:
        raise GPUEnvironmentError("optimizer step did not change any parameter")
    state["stage"] = "forward_after_step"
    with torch.no_grad():
        after = model(inputs)
        on_cuda(after, device, "post-step prediction")
        finite(after, torch, "post-step prediction")
        after_loss = (after - target).square().mean()
        on_cuda(after_loss, device, "post-step loss")
        finite(after_loss, torch, "post-step loss")
        state["loss_after_step"] = float(after_loss.item())
    state["stage"] = "synchronize_after_smoke"
    torch.cuda.synchronize(device)
    elapsed = float(clock() - started)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise GPUEnvironmentError("invalid synchronized smoke elapsed time")
    state["elapsed_seconds"] = elapsed
    state["stage"] = "read_peak_memory"
    state["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    state["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    if state["peak_allocated_bytes"] <= 0 or state["peak_reserved_bytes"] < state["peak_allocated_bytes"]:
        raise GPUEnvironmentError("invalid CUDA peak memory evidence")
    state["status"] = "passed"
    state["stage"] = "completed"


def check_environment(*, require_cuda: bool = False,
                      torch_loader: Callable[[], Any] | None = None,
                      clock: Callable[[], float] = time.perf_counter) -> dict:
    """Return report on success OR failure; no CUDA use at module import time."""
    started = clock()
    report = {
        "schema_version": 1, "status": "formal", "check_status": "not_assessed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "explicit_offline_synthetic_gpu_environment_check_only",
        "require_cuda": require_cuda, "cpu_fallback_allowed": False,
        "python": {"version": platform.python_version(), "executable": str(Path(sys.executable).resolve()),
                   "platform": platform.platform(), "prefix": str(Path(sys.prefix).resolve())},
        "packages": {name: package_version(name) for name in ("torch", "numpy", "scipy")},
        "torch": {"version": None, "cuda_build_version": None, "cudnn_version": None, "cpu_threads": None},
        "cuda": {"available": None, "device_count": None, "selected_device": None,
                 "name": None, "capability": None, "total_vram_bytes": None,
                 "driver_version": None, "driver_version_status": "not_assessed_no_driver_probe"},
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "smoke": {"status": "not_assessed", "stage": "not_started", "device": None,
                  "input_shape": list(SMOKE_SHAPE), "dtype": "float32",
                  "model": "disposable Conv1d(1,4,5)-GELU-Conv1d(4,1,5)",
                  "optimizer": "SGD", "learning_rate": SMOKE_LEARNING_RATE, "optimizer_steps": 1,
                  "data": "synthetic numeric input; NOT real EEG or clean truth",
                  "forward_finite": None, "gradients_finite": None, "parameters_finite": None,
                  "parameters_updated": None, "loss_before_step": None, "loss_after_step": None,
                  "elapsed_seconds": None, "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                  "memory_scope": "selected device, this process PyTorch allocator, reset immediately before smoke"},
        "error": None, "elapsed_seconds": None, "exit_code": 1,
        "dataset_training_performed": False, "holdout_accessed": False,
        "v3_training_status": "not_assessed", "v3_model_quality_status": "not_assessed",
        "production_validated": False, "automatic_processing_enabled": False,
    }
    stage = "import_torch"
    try:
        torch = (torch_loader or import_torch)()
        report["torch"]["version"] = str(torch.__version__)
        report["torch"]["cuda_build_version"] = torch.version.cuda
        stage = "cuda_availability"
        available = bool(torch.cuda.is_available())
        report["cuda"]["available"] = available
        report["cuda"]["device_count"] = int(torch.cuda.device_count())
        if not available:
            raise GPUEnvironmentError("CUDA is unavailable; no CPU fallback or training-readiness claim")
        if torch.version.cuda is None or report["cuda"]["device_count"] <= 0:
            raise GPUEnvironmentError("CUDA build/device enumeration is inconsistent with availability")
        stage = "cuda_device_metadata"
        index = int(torch.cuda.current_device())
        device = torch.device(f"cuda:{index}")
        report["cuda"]["selected_device"] = str(device)
        report["smoke"]["device"] = str(device)
        props = torch.cuda.get_device_properties(index)
        report["cuda"]["name"] = str(torch.cuda.get_device_name(index))
        report["cuda"]["capability"] = [int(x) for x in torch.cuda.get_device_capability(index)]
        report["cuda"]["total_vram_bytes"] = int(props.total_memory)
        stage = "torch_runtime_configuration"
        torch.set_num_threads(2)
        report["torch"]["cpu_threads"] = int(torch.get_num_threads())
        report["torch"]["cudnn_version"] = torch.backends.cudnn.version()
        stage = "cuda_smoke"
        run_cuda_smoke(torch, device, report["smoke"], clock)
        report["check_status"] = "passed"
        report["exit_code"] = 0
    except Exception as error:
        report["check_status"] = "failed"
        if stage == "cuda_smoke":
            report["smoke"]["status"] = "failed"
            stage = report["smoke"]["stage"]
        report["error"] = {"stage": stage, "type": type(error).__name__, "message": str(error)}
    elapsed = float(clock() - started)
    report["elapsed_seconds"] = elapsed if math.isfinite(elapsed) and elapsed >= 0 else None
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new JSON report path; existing files never overwritten")
    parser.add_argument("--require-cuda", action="store_true", help="record CUDA as an explicit prerequisite; CUDA failure always exits nonzero")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Reserve exclusively BEFORE import/device operations: no duplicate smoke
        # against an existing report, including races or dangling symlinks.
        with args.output.open("x", encoding="utf-8") as handle:
            report = check_environment(require_cuda=args.require_cuda)
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError) as error:
        print(f"GPU environment report refused: {error}", file=sys.stderr)
        return 2
    print(f"GPU environment check: {report['check_status']}; report: {args.output.resolve()}")
    if report["error"] is not None:
        print(f"{report['error']['stage']}: {report['error']['message']}", file=sys.stderr)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())

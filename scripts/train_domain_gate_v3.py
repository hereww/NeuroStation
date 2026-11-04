"""Preregistered v3 applicability-gate training, with frozen v2 weights.

CUDA required by default; explicit CPU is for synthetic tests, never GPU claims.
No test selector, holdout loader, test inference, raw writes or denoiser training.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from eeg_tools import learned_denoise_v3 as core
from eeg_tools.learned_domain_dataset_v3 import DomainDatasetV3Error, load_domain_split, sha256, PREREGISTRATION

torch = core.torch
EPOCHS, SEED, BATCH_SIZE, LEARNING_RATE = 20, 20261005, 128, 0.001


class DomainGateTrainingError(ValueError):
    """Invalid preregistration, training data/device, model output or publication."""


def calibrate_support_threshold(unsupported_scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    scores = np.asarray(unsupported_scores, dtype=np.float64)
    if (scores.ndim != 1 or scores.size < 1 or not np.all(np.isfinite(scores))
            or np.any((scores < 0) | (scores > 1))):
        raise DomainGateTrainingError("unsupported validation scores must be finite nonempty [0,1]")
    count = int(scores.size)
    allowance = int(np.floor(count * 0.01))
    order_index = count - allowance - 1
    threshold = max(0.5, float(np.sort(scores)[order_index]))
    accepted = int(np.count_nonzero(scores > threshold))
    observed = float(accepted / count)
    if accepted > allowance or observed > 0.01:
        raise DomainGateTrainingError("strict calibration false acceptance exceeded 1%")
    return threshold, {
        "scope": "validation_only", "method": "unsupported_score_order_statistic",
        "comparison": ">", "empirical_only": True, "production_calibrated": False,
        "support_threshold_floor": 0.5, "unsupported_window_count": count,
        "max_unsupported_false_acceptance_fraction": 0.01,
        "allowed_false_acceptance_windows": allowance,
        "order_statistic_index_zero_based": order_index,
        "observed_unsupported_false_acceptance_windows": accepted,
        "observed_unsupported_false_acceptance_fraction": observed,
        "warning": "Validation empirical domain-score control only; NOT clean, efficacy or production probability calibration.",
    }


def _device(requested: str):
    if torch is None:
        raise DomainGateTrainingError("optional PyTorch is required")
    try:
        device = torch.device(requested)
    except (ValueError, RuntimeError, TypeError) as error:
        raise DomainGateTrainingError("device must be explicit cuda[:index], or cpu for tests") from error
    if device.type == "cpu" and device.index is None:
        return device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise DomainGateTrainingError("actual CUDA required; no silent CPU fallback (tests must explicitly use --device cpu)")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if not 0 <= index < torch.cuda.device_count():
        raise DomainGateTrainingError("requested CUDA index is unavailable")
    return torch.device(f"cuda:{index}")


def _validate_preregistration(root: Path) -> dict:
    contract = json.loads((root / PREREGISTRATION).read_text(encoding="utf-8"))
    if (type(contract) is not dict or contract.get("seed") != SEED
            or contract.get("epochs") != EPOCHS or contract.get("batch_size") != BATCH_SIZE
            or contract.get("learning_rate") != LEARNING_RATE or contract.get("threads") != 2
            or contract.get("training", {}).get("comparison") != ">"
            or contract.get("training", {}).get("max_unsupported_false_acceptance") != 0.01
            or contract.get("training", {}).get("test_access_before_freeze") is not False
            or contract.get("training", {}).get("denoiser_v2_frozen") is not True):
        raise DomainGateTrainingError("training settings differ from frozen v3 preregistration")
    return contract


def class_balanced_weights(labels: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype != np.dtype("bool") or len(labels) == 0:
        raise DomainGateTrainingError("supported labels must be nonempty bool [N]")
    supported_count = int(np.count_nonzero(labels))
    unsupported_count = int(len(labels) - supported_count)
    if min(supported_count, unsupported_count) == 0:
        raise DomainGateTrainingError("each full split must contain both domain classes")
    return float(len(labels) / (2 * unsupported_count)), float(len(labels) / (2 * supported_count))


def _validate_data(data: dict, name: str) -> None:
    if type(data) is not dict:
        raise DomainGateTrainingError("domain loader must return a mapping")
    features, labels = data.get("normalized"), data.get("supported")
    if (not isinstance(features, np.ndarray) or features.dtype != np.dtype("float32")
            or features.ndim != 3 or features.shape[1:] != (1, 500)
            or len(features) == 0 or not np.all(np.isfinite(features))):
        raise DomainGateTrainingError("domain features must be finite float32 [N,1,500]")
    if (not isinstance(labels, np.ndarray) or labels.shape != (len(features),)
            or labels.dtype != np.dtype("bool")):
        raise DomainGateTrainingError("supported labels must be bool [N]")
    class_balanced_weights(labels)
    provenance = data.get("provenance")
    if (type(provenance) is not list or len(provenance) != len(features)
            or not core.primitive_metadata(provenance)
            or data.get("split") != name or data.get("test_values_accessed") is not False
            or data.get("new_unsupported_holdout_accessed") is not False):
        raise DomainGateTrainingError("domain provenance/scope contract failed")
    if not core.primitive_metadata(data.get("sources")):
        raise DomainGateTrainingError("source metadata must remain finite plain values")
    for label, item in zip(labels, provenance):
        if type(item) is not dict or item.get("split") != name:
            raise DomainGateTrainingError("window provenance split mismatch")
        if (not label and item.get("subject") not in (("S6.mat", "S7.mat") if name == "train" else ("S8.mat", "S9.mat"))):
            raise DomainGateTrainingError("training attempted to use an unauthorized/holdout subject")


def _run_epoch(model, loader, optimizer, device, weights: tuple[float, float]) -> tuple[float, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss, count, scores = 0.0, 0, []
    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            if (not isinstance(logits, torch.Tensor) or logits.shape != (len(features),)
                    or not bool(torch.isfinite(logits).all())):
                raise DomainGateTrainingError("gate forward must return finite logits [B]")
            per_window = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            weighted = per_window * torch.where(labels > 0.5, weights[1], weights[0])
            loss = weighted.mean()
            if not bool(torch.isfinite(loss)):
                raise DomainGateTrainingError("nonfinite balanced BCE")
            if training:
                loss.backward()
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                    raise DomainGateTrainingError("nonfinite gate gradient")
                optimizer.step()
            total_loss += float(weighted.detach().double().sum().cpu())
            count += int(len(features))
            if not training:
                scores.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64))
    if count < 1:
        raise DomainGateTrainingError("empty epoch")
    value = float(total_loss / count)
    return value, np.concatenate(scores) if scores else np.empty(0, dtype=np.float64)


def _validation_report(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict, dict]:
    if scores.shape != labels.shape or not np.all(np.isfinite(scores)):
        raise DomainGateTrainingError("validation scores/labels mismatch")
    threshold, calibration = calibrate_support_threshold(scores[~labels])
    accepted = scores > threshold
    positive_count, negative_count = int(labels.sum()), int((~labels).sum())
    positive_accepted = int(np.count_nonzero(accepted & labels))
    negative_accepted = int(np.count_nonzero(accepted & ~labels))
    return threshold, calibration, {
        "scope": "validation_only", "comparison": ">", "support_threshold": threshold,
        "supported_window_count": positive_count, "unsupported_window_count": negative_count,
        "supported_accepted_windows": positive_accepted,
        "supported_acceptance_fraction": float(positive_accepted / positive_count),
        "unsupported_accepted_windows": negative_accepted,
        "unsupported_false_acceptance_fraction": float(negative_accepted / negative_count),
        "all_supported_rejected": positive_accepted == 0,
        "threshold_lowered_to_raise_positive_acceptance": False,
    }


def _cpu_state(model) -> dict:
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise DomainGateTrainingError("nonfinite checkpoint gate_state")
    return state


def _publish(output: Path, report_path: Path, checkpoint: dict, report: dict) -> None:
    """Stage both artifacts, publish exclusively; never replace an existing file."""
    if output.resolve() == report_path.resolve() or output.exists() or report_path.exists():
        raise DomainGateTrainingError("checkpoint/report must be distinct and must not already exist")
    staged, published = [], []
    try:
        for target, content in ((output, checkpoint), (report_path, report)):
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".pending", dir=target.parent)
            os.close(descriptor)
            temporary = Path(name)
            staged.append((temporary, target))
            if target == output:
                torch.save(content, temporary)
            else:
                temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for temporary, target in staged:
            os.link(temporary, target)  # atomic exclusive publication, same volume
            published.append((temporary, target))
    except Exception:
        for temporary, target in published:
            if target.exists() and temporary.samefile(target):
                target.unlink()
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--denoiser-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    started = time.perf_counter()
    try:
        root = args.root.resolve()
        output = args.output or root / "models/domain_gate_v3.pt"
        report_path = args.report or output.with_suffix(".training.json")
        if output.resolve() == report_path.resolve() or output.exists() or report_path.exists():
            raise DomainGateTrainingError("output/report already exists or paths collide; refusing overwrite")
        if args.threads != 2:
            raise DomainGateTrainingError("frozen v3 contract requires --threads 2")
        device = _device(args.device)  # fail before reading any training arrays
        contract = _validate_preregistration(root)
        preregistration_digest = sha256(root / PREREGISTRATION)
        config_path = args.config or root / "configs/learned_denoiser_v3.json"
        config = core.load_config(config_path)
        config_digest = sha256(config_path)
        expected_model = {k:v for k,v in contract["model"].items() if k not in ("type", "normalization")}
        if config["model"] != expected_model:
            raise DomainGateTrainingError("gate architecture differs from preregistration")
        denoiser_path = args.denoiser_checkpoint or root / "models/denoiser_v2.pt"
        denoiser_digest = sha256(denoiser_path)
        if denoiser_digest != contract["protected_sha256"].get("models/denoiser_v2.pt"):
            raise DomainGateTrainingError("denoiser checkpoint differs from frozen v2 SHA256")
        # Identity only. The frozen denoiser is never loaded, optimized or run.
        train, validation = load_domain_split(root, "train"), load_domain_split(root, "validation")
        _validate_data(train, "train"); _validate_data(validation, "validation")
        if train["dataset_sha256"] != validation["dataset_sha256"]:
            raise DomainGateTrainingError("train/validation supported dataset identity differs")
        torch.set_num_threads(args.threads)
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        is_cuda = device.type == "cuda"
        if is_cuda:
            torch.cuda.manual_seed_all(SEED)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        from torch.utils.data import DataLoader, TensorDataset
        loaders, weights = {}, {}
        for name, data in (("train", train), ("validation", validation)):
            weights[name] = class_balanced_weights(data["supported"])
            tensor_data = TensorDataset(torch.from_numpy(data["normalized"]),
                                        torch.from_numpy(data["supported"].astype(np.float32)))
            generator = torch.Generator().manual_seed(SEED)
            loaders[name] = DataLoader(tensor_data, batch_size=BATCH_SIZE, shuffle=name == "train",
                                       generator=generator, num_workers=0, drop_last=False)
        model = core.build_gate(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        best_loss, best_epoch, best_state, best_validation = float("inf"), 0, None, None
        history = []
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            train_loss, _ = _run_epoch(model, loaders["train"], optimizer, device, weights["train"])
            validation_loss, scores = _run_epoch(model, loaders["validation"], None, device, weights["validation"])
            threshold, calibration, validation_report = _validation_report(scores, validation["supported"])
            item = {"epoch": epoch, "train_balanced_BCE": train_loss,
                    "validation_balanced_BCE": validation_loss, "validation": validation_report,
                    "elapsed_seconds": float(time.perf_counter() - epoch_start)}
            history.append(item)
            if validation_loss < best_loss:  # first epoch wins an exact tie
                best_loss, best_epoch, best_state = validation_loss, epoch, _cpu_state(model)
                best_validation = (threshold, calibration, deepcopy(validation_report))
            print(json.dumps(item, ensure_ascii=True, allow_nan=False), flush=True)
        if is_cuda:
            torch.cuda.synchronize(device)
        threshold, calibration, validation_report = best_validation
        environment = {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": str(np.__version__), "torch": str(torch.__version__),
            "torch_cuda_build": str(torch.version.cuda) if torch.version.cuda is not None else None,
            "training_device": str(device), "gpu_training_performed": is_cuda,
            "gpu_name": str(torch.cuda.get_device_name(device)) if is_cuda else None,
            "gpu_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if is_cuda else 0,
            "gpu_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if is_cuda else 0,
            "torch_threads": int(torch.get_num_threads()),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "cpu_execution_warning": None if is_cuda else "Explicit CPU execution (test-only mode); NOT GPU training.",
        }
        sources = {"train": train["sources"], "validation": validation["sources"]}
        domain_digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        metadata = {
            "scope": "train_validation_only", "label_semantics": "supported_domain_not_clean_or_denoising_efficacy",
            "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "threads": args.threads,
            "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE},
            "best_epoch": best_epoch, "best_validation_balanced_BCE": float(best_loss),
            "checkpoint_selection": "minimum_validation_balanced_BCE_first_exact_tie",
            "validation_report": validation_report, "history": history,
            "sources": sources, "domain_sources_sha256": domain_digest,
            "class_counts": {name: {"supported": int(data["supported"].sum()),
                                    "unsupported": int((~data["supported"]).sum())}
                             for name,data in (("train", train), ("validation", validation))},
            "balanced_BCE_definition": "0.5*(mean_supported_BCE+mean_unsupported_BCE); full-split weights N/(2*class_count), not minibatch weights",
            "window_exclusion_count": 0,
            "domain_sources_hash_semantics": "SHA256 of canonical train/validation source identity JSON, including paths; identity only, never a feature",
            "train_provenance": train["provenance"], "validation_provenance": validation["provenance"],
            "dataset_manifest_sha256": train["sources"]["supported"]["manifest_sha256"],
            "preregistration_sha256": preregistration_digest,
            "config_sha256": config_digest, "environment": environment,
            "known_benchmark_test_accessed": False, "new_unsupported_holdout_accessed": False,
            "test_arrays_inspected": False,
            "npz_io_semantics": "Compressed noisy/valid_mask members may physically decompress test bytes; test rows are never selected, inspected, normalized, model inputs, training/metric/calibration data.",
            "test_access_flag_semantics": "Logical signal-value use only; not a claim of no physical test-byte I/O.",
            "denoiser_training_or_inference_performed": False,
            "automatic_processing_enabled": False, "production_validated": False,
            "elapsed_seconds": float(time.perf_counter() - started),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        checkpoint = {"checkpoint_schema_version": 1, "model_version": "learned_denoise_v3",
                      "config": config, "gate_state": best_state, "support_threshold": threshold,
                      "threshold_calibration": calibration, "denoiser_sha256": denoiser_digest,
                      "dataset_sha256": train["dataset_sha256"], "metadata": metadata}
        core.validate_checkpoint_metadata(checkpoint, config)
        report = {"status": "completed", "model_version": "learned_denoise_v3",
                  "checkpoint": str(output.resolve()), "support_threshold": threshold,
                  "threshold_calibration": calibration, "denoiser_sha256": denoiser_digest,
                  "dataset_sha256": train["dataset_sha256"],
                  "train_windows": int(len(train["supported"])),
                  "validation_windows": int(len(validation["supported"])),
                  "full_epoch_class_weights": {name: {"unsupported": value[0], "supported": value[1]} for name,value in weights.items()},
                  "warning": "Domain-gate training complete is NOT independent test, clean truth, denoising efficacy or production acceptance.",
                  **{k:v for k,v in metadata.items() if k not in ("train_provenance", "validation_provenance")}}
        if not core.primitive_metadata(report):
            raise DomainGateTrainingError("non-primitive training report metadata")
        if (sha256(denoiser_path) != denoiser_digest
                or sha256(config_path) != config_digest
                or sha256(root / PREREGISTRATION) != preregistration_digest):
            raise DomainGateTrainingError("frozen denoiser/config/preregistration changed while training")
        _publish(output, report_path, checkpoint, report)
        print(json.dumps({"status": "completed", "best_epoch": best_epoch,
                          "checkpoint": str(output.resolve()), "report": str(report_path.resolve()),
                          "gpu_training_performed": is_cuda, "test_accessed": False,
                          "automatic_processing_enabled": False, "production_validated": False}, ensure_ascii=True))
    except (DomainGateTrainingError, DomainDatasetV3Error, core.LearnedDenoiseV3Error,
            OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

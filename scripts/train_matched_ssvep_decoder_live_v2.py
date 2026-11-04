"""Train an isolated Wang2016 decoder on the exact 4 s live-game window.

The experiment uses subjects S1-S24 for training and S25-S29 for epoch
selection. S30-S35 are never loaded or hashed. There is intentionally no final
test command; the existing one-shot final test belongs to the prior candidate.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from scripts import benchmark_matched_public_ssvep as benchmark
from scripts import train_matched_ssvep_decoder as base
from eeg_tools import matched_ssvep_decoder as decoder

ROOT = Path(__file__).resolve().parents[1]
BASE_RUN_DIR = ROOT / "analysis_outputs" / "wang2016_spectral_mlp_20261015"
RUN_DIR = ROOT / "analysis_outputs" / "wang2016_spectral_mlp_live_4s_20261015"
TRAIN_SUBJECTS = tuple(range(1, 25))
VALIDATION_SUBJECTS = tuple(range(25, 30))
SPLITS = {"train": list(TRAIN_SUBJECTS), "validation": list(VALIDATION_SUBJECTS)}
WINDOW_SAMPLES = 1000
WINDOW_STARTS = (0, 250)
SEED = 20261015
OOD_MAXABS = 8.0
OOD_P95ABS = 4.0
SCOPE = "Wang2016 cross-subject 10/12/15 Hz; exact 1000-sample live-game window; no personal validation"
SELECTION_RULE = "max pooled two-window validation accuracy, min pooled cross_entropy, earliest epoch"
TRAINING = {"epochs": 100, "batch_size": 32, "learning_rate": 0.001,
            "weight_decay": 0.001, "optimizer": "AdamW", "loss": "CrossEntropyLoss",
            "window_crop": "one of start 0 or 250 selected per trial per epoch",
            "standardization_fit": "all two train-only windows per source trial"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return benchmark.sha256_file(Path(path))


def canonical(value) -> str:
    return benchmark.canonical(value)


def code_hashes() -> dict[str, str]:
    names = ("scripts/train_matched_ssvep_decoder_live_v2.py",
             "scripts/train_matched_ssvep_decoder.py",
             "scripts/benchmark_matched_public_ssvep.py",
             "eeg_tools/matched_ssvep_decoder.py")
    return {str((ROOT / name).resolve()): sha256(ROOT / name) for name in names}


def write_new(path: Path, value) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def source_rows(spec: dict, mat_root: Path) -> dict[str, list[dict]]:
    files = {subject: (mat_root / f"S{subject}.mat").resolve()
             for subject in TRAIN_SUBJECTS + VALIDATION_SUBJECTS}
    require(all(path.is_file() for path in files.values()), "missing S1-S29 source files")
    rows = {split: [] for split in SPLITS}
    for split, subjects in SPLITS.items():
        for subject in subjects:
            for frequency in decoder.FREQUENCIES:
                condition = spec["frequency_order_hz"].index(frequency)
                for block in range(6):
                    rows[split].append({
                        "subject": subject, "block": block, "frequency_hz": frequency,
                        "condition_index": condition, "source_trial_index": condition * 6 + block,
                        "source_file": str(files[subject]),
                        "group_id": canonical([subject, block, spec["depth"], spec["session"]]),
                    })
    return rows


def load_base_v1_weights(run_dir: Path, device: torch.device):
    """Verify the old checkpoint chain without touching its final-test subjects."""
    names = ("preregistration.json", "preregistration_lock.json", "selection.json",
             "selection_lock.json", "best_weights.pt", "history.json", "validation_report.json")
    require(all((run_dir / name).is_file() and not (run_dir / name).is_symlink() for name in names),
            "incomplete or unsafe v1 model evidence")
    prereg = benchmark.read_json(run_dir / "preregistration.json")
    selection = benchmark.read_json(run_dir / "selection.json")
    require(benchmark.read_json(run_dir / "preregistration_lock.json").get("sha256") == base.digest(prereg),
            "v1 preregistration lock mismatch")
    require(benchmark.read_json(run_dir / "selection_lock.json").get("sha256") == base.digest(selection),
            "v1 selection lock mismatch")
    require(selection.get("artifacts", {}).get("best_weights.pt") == sha256(run_dir / "best_weights.pt"),
            "v1 checkpoint hash mismatch")
    require(selection.get("artifacts", {}).get("validation_report.json") == sha256(run_dir / "validation_report.json"),
            "v1 validation report hash mismatch")
    for path, expected in prereg.get("code_hashes", {}).items():
        require(Path(path).is_file() and sha256(Path(path)) == expected, "v1 training code hash mismatch")
    model = decoder.HarmonicSpectrumMLP()
    state = torch.load(run_dir / "best_weights.pt", map_location="cpu", weights_only=True)
    require(isinstance(state, dict) and set(state) == set(model.state_dict()), "v1 state_dict contract mismatch")
    require(all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in state.values()),
            "v1 checkpoint contains invalid tensors")
    model.load_state_dict(state, strict=True)
    report = benchmark.read_json(run_dir / "validation_report.json")
    return model.to(device).eval(), report


def make_preregistration(mat_root: Path, source_spec: Path, base_run: Path, *, epochs: int) -> dict:
    require(type(epochs) is int and 1 <= epochs <= 500, "epochs must be 1..500")
    mat_root, source_spec, base_run = map(lambda path: Path(path).resolve(),
                                          (mat_root, source_spec, base_run))
    spec = benchmark.read_json(source_spec)
    base.check_spec(spec)
    expected = source_rows(spec, mat_root)
    metadata = spec.get("provenance", {}).get("metadata_files", [])
    require(len(metadata) == 3, "verified Wang metadata hashes required")
    hashes = {str(source_spec): sha256(source_spec)}
    for row in metadata:
        path = Path(row["local_path"]).resolve()
        require(path.is_file() and sha256(path) == row["sha256"], "Wang metadata source hash mismatch")
        hashes[str(path)] = row["sha256"]
    for split in SPLITS.values():
        for subject in split:
            path = (mat_root / f"S{subject}.mat").resolve()
            hashes[str(path)] = sha256(path)
    base_report = base_run / "validation_report.json"
    base_weights = base_run / "best_weights.pt"
    return {
        **decoder.FLAGS, "schema_version": 1, "scope": SCOPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mat_root": str(mat_root), "source_spec_path": str(source_spec), "source_spec": spec,
        "source_hashes_train_and_validation_only": hashes,
        "subjects": SPLITS, "expected_trials": expected,
        "frequencies_hz": list(decoder.FREQUENCIES), "channels": list(decoder.PREPROCESSING["channel_order"]),
        "window_samples": WINDOW_SAMPLES, "window_seconds": 4.0,
        "window_start_offsets_samples": list(WINDOW_STARTS),
        "window_preprocessing": decoder.PREPROCESSING,
        "model": decoder.MODEL_SPEC, "gate": decoder.GATE,
        "fixed_live_ood_gate": {"max_abs_z": OOD_MAXABS, "p95_abs_z": OOD_P95ABS,
                                "validated": False, "source": "unchanged v1 runtime heuristic"},
        "training": {**TRAINING, "epochs": epochs}, "seed": SEED,
        "selection_rule": SELECTION_RULE, "code_hashes": code_hashes(),
        "comparison_model": {"run_dir": str(base_run), "checkpoint_sha256": sha256(base_weights),
                             "validation_report_sha256": sha256(base_report)},
        "personal_sessions_read": False, "final_test_subjects_read_or_hashed": False,
        "final_test_evaluation_available": False, "deployment_executed": False,
        "runtime_enabled": False, "production_validated": False,
        "default_deployment_authorized": False,
    }


def validated_parts(dataset, prereg: dict):
    proxy = {
        "source_spec": prereg["source_spec"],
        "source_spec_path": prereg["source_spec_path"],
        "source_hashes": prereg["source_hashes_train_and_validation_only"],
        "subjects": prereg["subjects"], "expected_trials": prereg["expected_trials"],
    }
    return base.validate_dataset(dataset, proxy, ("train", "validation"))


def trial_crops(trials) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for trial in trials:
        require(trial.eeg.shape == (8, 1250) and np.isfinite(trial.eeg).all(),
                "expected actual finite Wang (8,1250) trials")
        features.append([decoder.spectral_features(trial.eeg[:, start:start + WINDOW_SAMPLES])
                         for start in WINDOW_STARTS])
        labels.append(decoder.FREQUENCIES.index(trial.frequency_hz))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def select_device(name="cuda", require_cuda=False) -> torch.device:
    require(name in ("cuda", "cpu"), "device must be cuda or cpu")
    require(not require_cuda or name == "cuda", "require_cuda forbids CPU execution")
    if name == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable; CPU fallback is forbidden")
    return torch.device(name)


def fit_model(train_trials, validation_trials, *, epochs: int, device: torch.device):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    train_x, train_y = trial_crops(train_trials)
    val_x, val_y = trial_crops(validation_trials)
    x = torch.as_tensor(train_x.reshape(-1, 72), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.repeat(train_y, 2), dtype=torch.long, device=device)
    vx = torch.as_tensor(val_x.reshape(-1, 72), dtype=torch.float32, device=device)
    vy = torch.as_tensor(np.repeat(val_y, 2), dtype=torch.long, device=device)
    model = decoder.HarmonicSpectrumMLP().to(device)
    with torch.no_grad():
        model.feature_mean.copy_(x.mean(0))
        model.feature_std.copy_(x.std(0, correction=0).clamp_min(1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()
    history, best_state, best_key = [], None, None
    for epoch in range(1, epochs + 1):
        model.train()
        crop_indices = torch.as_tensor(rng.integers(0, 2, size=len(train_y)), device=device)
        row_indices = torch.arange(len(train_y), device=device) * 2 + crop_indices
        selected_x, selected_y = x[row_indices], torch.as_tensor(train_y, device=device)
        permutation = torch.randperm(len(selected_y), device=device)
        total = 0.0
        for indices in permutation.split(32):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(selected_x[indices]), selected_y[indices])
            require(bool(torch.isfinite(loss)), "nonfinite training loss")
            loss.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                    "nonfinite gradient")
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            logits = model(vx)
            acc = float((logits.argmax(1) == vy).float().mean())
            val_loss = float(loss_fn(logits, vy))
        row = {"epoch": epoch, "train_cross_entropy": total / len(selected_y),
               "validation_accuracy": acc, "validation_cross_entropy": val_loss}
        history.append(row)
        key = (-acc, val_loss, epoch)
        if best_key is None or key < best_key:
            best_key, best_state = key, copy.deepcopy(model.state_dict())
    require(best_state is not None, "no checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    return model.eval(), best_state, history


def live_action(model, eeg: np.ndarray) -> dict:
    feature = decoder.spectral_features(eeg)
    mean = model.feature_mean.detach().cpu().numpy().astype(np.float64)
    std = model.feature_std.detach().cpu().numpy().astype(np.float64)
    z = (feature.astype(np.float64) - mean) / std
    ood_passed = float(np.max(np.abs(z))) <= OOD_MAXABS and float(np.percentile(np.abs(z), 95)) <= OOD_P95ABS
    with torch.no_grad():
        logits = model(torch.as_tensor(feature, dtype=torch.float32, device=next(model.parameters()).device).unsqueeze(0))
        probability = torch.softmax(logits, -1)[0].cpu().numpy()
    raw = decoder.command_prediction(probability)
    command = raw["command"] if ood_passed and raw["accepted"] else "UNKNOWN"
    if command == "SELECT":
        command = "HOLD"
    elif command not in ("MOVE_LEFT", "MOVE_RIGHT"):
        command = "HOLD"
    return {"raw": raw, "ood_passed": ood_passed, "game_action": command}


def summarize_rows(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row["raw"]["accepted"] and row["ood_passed"]]
    correct = sum(row["raw"]["predicted_frequency_hz"] == row["frequency_hz"] for row in rows)
    moves = [row for row in rows if row["game_action"] in ("MOVE_LEFT", "MOVE_RIGHT")]
    correct_actions = sum(row["game_action"] == {10: "MOVE_LEFT", 12: "MOVE_RIGHT", 15: "HOLD"}[row["frequency_hz"]]
                          for row in rows)
    false_moves = sum(row["game_action"] != "HOLD" and row["game_action"] !=
                      {10: "MOVE_LEFT", 12: "MOVE_RIGHT", 15: "HOLD"}[row["frequency_hz"]] for row in rows)
    return {"window_count": len(rows), "raw_accuracy": correct / len(rows) if rows else None,
            "accepted_count": len(accepted), "accepted_coverage": len(accepted) / len(rows) if rows else None,
            "accepted_accuracy": (sum(row["raw"]["predicted_frequency_hz"] == row["frequency_hz"]
                                       for row in accepted) / len(accepted) if accepted else None),
            "game_action_accuracy": correct_actions / len(rows) if rows else None,
            "false_move_count": false_moves, "false_move_rate_per_window": false_moves / len(rows) if rows else None,
            "move_count": len(moves), "executed": False}


def evaluate_model(model, trials) -> dict:
    model.eval()
    rows = []
    for trial in trials:
        for offset in WINDOW_STARTS:
            eeg = trial.eeg[:, offset:offset + WINDOW_SAMPLES]
            result = live_action(model, eeg)
            rows.append({"trial_id": trial.trial_id, "subject": trial.subject,
                         "block": trial.block, "frequency_hz": int(trial.frequency_hz),
                         "window_start_offset_samples": offset, **result})
    result = {"pooled_two_window": summarize_rows(rows), "by_window_start": {}}
    for offset in WINDOW_STARTS:
        subset = [row for row in rows if row["window_start_offset_samples"] == offset]
        result["by_window_start"][str(offset)] = summarize_rows(subset)
    result["per_subject"] = {str(subject): summarize_rows([row for row in rows if row["subject"] == subject])
                             for subject in sorted({row["subject"] for row in rows})}
    result["rows"] = rows
    return result


def train(mat_root: Path, source_spec: Path, output_dir: Path, *, base_run=BASE_RUN_DIR, epochs=100,
          device="cuda", require_cuda=False):
    chosen = select_device(device, require_cuda)
    output_dir = Path(output_dir).resolve()
    require(not output_dir.exists(), "run directory must be new; no overwrite or resume")
    prereg = make_preregistration(mat_root, source_spec, base_run, epochs=epochs)
    output_dir.mkdir(parents=False, exist_ok=False)
    write_new(output_dir / "preregistration.json", prereg)
    write_new(output_dir / "preregistration_lock.json", {"sha256": base.digest(prereg)})
    base_model, base_validation = load_base_v1_weights(Path(base_run).resolve(), chosen)
    data = benchmark.load_wang(Path(prereg["mat_root"]), Path(prereg["source_spec_path"]),
                               subjects=list(TRAIN_SUBJECTS + VALIDATION_SUBJECTS))
    for path, digest_value in data.input_hashes.items():
        require(prereg["source_hashes_train_and_validation_only"].get(str(Path(path).resolve())) == digest_value,
                "loaded S1-S29/source spec hash differs from preregistration")
    parts = validated_parts(data, prereg)
    old_validation = evaluate_model(base_model, parts["validation"])
    model, state, history = fit_model(parts["train"], parts["validation"], epochs=epochs, device=chosen)
    new_validation = evaluate_model(model, parts["validation"])
    new_validation.update({**decoder.FLAGS, "scope": SCOPE, "split": "validation",
                           "subjects": list(VALIDATION_SUBJECTS), "window_samples": WINDOW_SAMPLES,
                           "window_seconds": 4.0, "window_offsets_samples": list(WINDOW_STARTS),
                           "gate": decoder.GATE,
                           "ood_gate": {"max_abs_z": OOD_MAXABS, "p95_abs_z": OOD_P95ABS,
                                        "validated": False},
                           "old_v1_validation_4s": old_validation,
                           "old_v1_original_validation_5s": {
                               "raw_accuracy": base_validation["raw_labelmatch"],
                               "accepted_coverage": base_validation["accepted_coverage"],
                               "accepted_accuracy": base_validation["accepted_accuracy"],
                               "source_report_sha256": prereg["comparison_model"]["validation_report_sha256"]},
                           "personal_validation": False, "final_test_evaluated": False,
                           "commands_executed": False})
    write_new(output_dir / "history.json", {"epochs": history})
    write_new(output_dir / "validation_report.json", new_validation)
    with (output_dir / "best_weights.pt").open("xb") as handle:
        torch.save(state, handle)
    selected = base.select_epoch(history)
    selection = {**decoder.FLAGS, "schema_version": 1, "scope": SCOPE,
                 "selection_rule": SELECTION_RULE, "selected": selected,
                 "preregistration_sha256": base.digest(prereg),
                 "artifacts": {name: sha256(output_dir / name)
                               for name in ("best_weights.pt", "history.json", "validation_report.json")},
                 "checkpoint_format": "state_dict_only", "final_test_evaluated": False,
                 "device_used": str(chosen), "torch_version": str(torch.__version__),
                 "cuda_version": torch.version.cuda,
                 "runtime_enabled": False, "production_validated": False,
                 "default_deployment_authorized": False}
    write_new(output_dir / "selection.json", selection)
    write_new(output_dir / "selection_lock.json", {"sha256": base.digest(selection)})
    return {"status": "formal", "exploratory_only": True, "run_dir": str(output_dir),
            "device_used": str(chosen), "selected_epoch": selected["epoch"],
            "validation_raw_accuracy": new_validation["pooled_two_window"]["raw_accuracy"],
            "validation_game_action_accuracy": new_validation["pooled_two_window"]["game_action_accuracy"],
            "validation_false_move_rate": new_validation["pooled_two_window"]["false_move_rate_per_window"],
            "final_test_evaluated": False, "runtime_enabled": False,
            "production_validated": False, "default_deployment_authorized": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat-root", type=Path, default=Path(r"D:\SSVEP\Wang2016_Benchmark"))
    parser.add_argument("--source-spec", type=Path,
                        default=ROOT / "analysis_outputs" / "wang2016_matched_source_spec_20261014.json")
    parser.add_argument("--base-run", type=Path, default=BASE_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = train(args.mat_root, args.source_spec, args.output_dir, base_run=args.base_run,
                       epochs=args.epochs, device=args.device, require_cuda=args.require_cuda)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "executed": False},
                         ensure_ascii=False, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

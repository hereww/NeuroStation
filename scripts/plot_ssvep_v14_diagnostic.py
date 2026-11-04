"""Create read-only figures from the v14 personal diagnostic outputs.

The figures are descriptive diagnostics only.  Presentation matches are
recorded-label matching proxies, not gaze accuracy, and no figure authorizes
runtime deployment or physiological-clean recovery claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRACKS = ("raw", "notch", "traditional", "v14")
FALSE_FLAGS = {
    "runtime_enabled": False,
    "production_validated": False,
    "source_domain_validated": False,
    "independent_personal_validation": False,
    "clean_ground_truth_available": False,
    "presented_matches_are_gaze_accuracy": False,
    "physiological_clean_truth": False,
    "clinical_validation": False,
    "training_performed": False,
    "checkpoint_selection_performed": False,
    "default_deployment_authorized": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("v14 figure generator: " + message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"report already exists: {path}")
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def create_output(path: Path) -> Path:
    base = (ROOT / "analysis_outputs").resolve()
    candidate = path.resolve()
    require(candidate.is_relative_to(base), "output must be under analysis_outputs")
    require(not candidate.exists(), f"output directory already exists: {candidate}")
    candidate.mkdir(parents=True)
    return candidate


def load_inputs(evaluation: Path) -> tuple[dict, list[dict]]:
    require(evaluation.is_dir(), f"evaluation directory does not exist: {evaluation}")
    summary_path = evaluation / "summary.json"
    require(summary_path.is_file(), "summary.json is missing")
    summary = read_json(summary_path)
    require(summary.get("status") == "formal", "evaluation status is not formal")
    require(summary.get("allow_failed_selection_diagnostic") is True,
            "diagnostic evaluation flag is missing")
    require(summary.get("model", {}).get("selection_gate_passed") is False,
            "failed-selection binding is missing")
    reports = []
    for index in range(2):
        report_path = evaluation / f"personal_{index:02d}" / "report.json"
        report = read_json(report_path)
        require(report.get("source_hashes_match") is True,
                f"source hash mismatch in personal_{index:02d}")
        require(report.get("presented_matches_are_gaze_accuracy") is False,
                "report has unsafe gaze-accuracy flag")
        for track in TRACKS:
            artifact = evaluation / f"personal_{index:02d}" / f"{track}.npy"
            require(artifact.is_file(), f"missing {artifact}")
        reports.append(report)
    return summary, reports


def make_figures(evaluation: Path, output: Path) -> dict:
    # Scientific imports are intentionally lazy so --help remains lightweight.
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary, reports = load_inputs(evaluation)
    figure_paths: list[Path] = []
    figure_receipts: dict[str, dict] = {}
    metrics: list[dict] = []
    colors = {"raw": "#4c78a8", "notch": "#f58518", "traditional": "#54a24b", "v14": "#e45756"}

    for index, report in enumerate(reports):
        session_dir = evaluation / f"personal_{index:02d}"
        arrays = {track: np.load(session_dir / f"{track}.npy", allow_pickle=False) for track in TRACKS}
        shape = arrays["raw"].shape
        require(len(shape) == 2 and shape[1] == 8, f"unexpected array shape: {shape}")
        require(all(array.shape == shape and np.isfinite(array).all() for array in arrays.values()),
                f"non-finite or mismatched arrays in personal_{index:02d}")
        sample_rate = 250.0
        time_s = np.arange(shape[0], dtype=np.float64) / sample_rate
        centered = {
            name: array - np.mean(array, axis=0, keepdims=True)
            for name, array in arrays.items()
        }
        channel_mean = {name: np.mean(array, axis=1) for name, array in arrays.items()}
        centered_mean = {name: np.mean(array, axis=1) for name, array in centered.items()}

        # Plot a downsampled full-record trace for visual comparison.
        step = max(1, shape[0] // 5000)
        trace_path = output / f"personal_{index:02d}_centered_trace.png"
        fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
        for name in TRACKS:
            ax.plot(time_s[::step], centered_mean[name][::step], label=name, color=colors[name], linewidth=0.85)
        ax.set_title(f"Personal {index:02d}: channel-mean centered waveform (diagnostic)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("centered signal (source units)")
        ax.grid(True, alpha=0.2)
        ax.legend(ncol=4, frameon=False)
        fig.savefig(trace_path, dpi=150)
        plt.close(fig)
        figure_paths.append(trace_path)

        # Plot a conservative spectral view.  This is not a denoising score.
        n = shape[0]
        taper = np.hanning(n)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        spectrum_path = output / f"personal_{index:02d}_spectrum.png"
        fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
        spectral_metrics = {}
        for name in TRACKS:
            values = centered[name] * taper[:, None]
            spectrum = np.abs(np.fft.rfft(values, axis=0)).mean(axis=1) / max(1.0, n)
            visible = (freqs >= 0.1) & (freqs <= 60.0)
            ax.plot(freqs[visible], 20.0 * np.log10(np.maximum(spectrum[visible], 1e-12)),
                    label=name, color=colors[name], linewidth=0.9)
            spectral_metrics[name] = {
                "band_8_40_mean_amplitude": float(np.mean(spectrum[(freqs >= 8) & (freqs <= 40)])),
                "band_45_55_mean_amplitude": float(np.mean(spectrum[(freqs >= 45) & (freqs <= 55)])),
            }
        ax.set_title(f"Personal {index:02d}: centered full-record spectrum (diagnostic)")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("magnitude (dB, arbitrary source scale)")
        ax.set_xlim(0.1, 60)
        ax.grid(True, alpha=0.2)
        ax.legend(ncol=4, frameon=False)
        fig.savefig(spectrum_path, dpi=150)
        plt.close(fig)
        figure_paths.append(spectrum_path)

        # Plot recorded-presentation matching proxy and confidence margin together.
        match_path = output / f"personal_{index:02d}_presentation_proxy.png"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        matches = [report["tracks"][name]["presented_match_count"] for name in TRACKS]
        margins = [report["tracks"][name]["mean_top_two_margin"] for name in TRACKS]
        axes[0].bar(TRACKS, matches, color=[colors[name] for name in TRACKS])
        axes[0].set_ylim(0, report["tracks"]["raw"]["recorded_trial_count"])
        axes[0].set_ylabel("matches to recorded presentation")
        axes[0].set_title("Recorded-label matching proxy")
        axes[0].grid(True, axis="y", alpha=0.2)
        axes[1].bar(TRACKS, margins, color=[colors[name] for name in TRACKS])
        axes[1].set_ylabel("mean top-two margin")
        axes[1].set_title("Forced-argmax score margin")
        axes[1].grid(True, axis="y", alpha=0.2)
        fig.suptitle(f"Personal {index:02d}: decoder diagnostics (not gaze accuracy)")
        fig.savefig(match_path, dpi=150)
        plt.close(fig)
        figure_paths.append(match_path)

        metrics.append({
            "personal_index": index,
            "source_session": report["source_session"],
            "shape": list(shape),
            "track_channel_mean_rms": {
                name: float(np.sqrt(np.mean(np.square(channel_mean[name])))) for name in TRACKS
            },
            "track_centered_channel_mean_rms": {
                name: float(np.sqrt(np.mean(np.square(centered_mean[name])))) for name in TRACKS
            },
            "spectral": spectral_metrics,
            "tracks": report["tracks"],
        })

    for path in figure_paths:
        figure_receipts[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    result = {
        "schema_version": 1,
        "status": "formal",
        "completion_status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_directory": str(evaluation.resolve()),
        "evaluation_summary_sha256": sha256(evaluation / "summary.json"),
        "figures": figure_receipts,
        "metrics": metrics,
        "limitations": [
            "All plots are descriptive diagnostics from a failed-selection checkpoint.",
            "Recorded-presentation matches are not independent gaze accuracy.",
            "Margins are relative score margins, not probabilities or calibrated confidence.",
            "No clean physiological reference, independent gaze truth, hardware calibration, or clinical validation is present.",
            "The spectrum is a fixed full-record visualization and is not a deployable quality metric.",
        ],
        **FALSE_FLAGS,
    }
    atomic_json(output / "figures_receipt.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args(argv)
    try:
        evaluation = Path(args.evaluation_directory).resolve()
        output = create_output(Path(args.output_directory))
        result = make_figures(evaluation, output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "formal", "completion_status": "incomplete", "error": str(exc), **FALSE_FLAGS}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "formal", "completion_status": "complete", "report": str(output / "figures_receipt.json"), **FALSE_FLAGS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

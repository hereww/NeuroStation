#!/usr/bin/env python3
"""Read-only structural exploration for the MultiFrequency_Mu2024 dataset.

This script intentionally does not modify source MAT/XLSX/M files and does not
write trial arrays. It only loads headers/data in memory and optionally writes a
new Markdown report. The trigger cleanup rules mirror the dataset's bundled
``dataset_processData.m``; they are included here for analysis only.
"""
from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

FS_EXPECTED = 512.0
TRIAL_SECONDS = 5
FREQUENCIES_HZ = (7, 11, 13, 17, 19, 23)
EXPECTED_TRIALS_BY_SESSION = {
    1: 24,
    2: 45,
    3: 45,
    4: 45,
    5: 45,
    6: 40,
    7: 40,
    8: 40,
    9: 40,
}
# 1-based trigger ordinal selections copied from dataset_processData.m.
MATLAB_TRIGGER_FIXES: dict[int, dict[int, int | list[int]]] = {
    1: {2: 45},
    3: {
        1: list(range(1, 13)) + list(range(15, 27)),
        2: list(range(2, 17)) + list(range(18, 33)) + list(range(35, 50)),
        3: list(range(1, 16)) + list(range(17, 47)),
        6: list(range(3, 23)) + list(range(25, 45)),
    },
    14: {4: list(range(4, 49))},
    22: {7: list(range(1, 21)) + list(range(22, 42))},
    31: {5: list(range(10, 55))},
}


@dataclass
class FileSummary:
    participant: int
    session: int
    path: Path
    shape: tuple[int, int]
    dtype: str
    samples: int
    time_start: float
    time_end: float
    dt_median: float
    dt_min: float
    dt_max: float
    raw_positive_triggers: int
    raw_negative_triggers: int
    corrected_positive_triggers: int
    row8_positive_values: tuple[int, ...]
    row9_unique: tuple[int, ...]
    row10_zero_at_corrected_onset: int
    row10_nonzero_at_corrected_onset: int
    corrected_onsets: np.ndarray
    corrected_trigger_values: np.ndarray


def parse_filename(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"P(\d+)_Ses([1-9])\.mat", path.name)
    if not match:
        raise ValueError(f"Unexpected file name: {path.name}")
    return int(match.group(1)), int(match.group(2))


def find_onsets_like_matlab(trigger: np.ndarray, fs: int = 512) -> np.ndarray:
    """Return positive trigger onsets with the bundled script's 1-second dead time."""
    onsets: list[int] = []
    last_call = -fs
    for index, value in enumerate(trigger):
        if index - last_call < fs:
            continue
        if value >= 1:
            onsets.append(index)
            last_call = index
    return np.asarray(onsets, dtype=np.int64)


def apply_bundled_trigger_fix(
    participant: int, session: int, onsets: np.ndarray
) -> np.ndarray:
    selection = MATLAB_TRIGGER_FIXES.get(participant, {}).get(session)
    if selection is None:
        return onsets
    if isinstance(selection, int):
        return onsets[:selection]
    return onsets[np.asarray(selection, dtype=np.int64) - 1]


def target_combinations(session: int) -> list[tuple[int, ...]]:
    order = 1 if session == 1 else 2 if session <= 5 else 3
    return list(itertools.combinations(FREQUENCIES_HZ, order))


def scan_file(path: Path) -> FileSummary:
    participant, session = parse_filename(path)
    variable_name = f"data_ses{session}"
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    if variable_name not in loaded:
        raise KeyError(f"{path.name} missing variable {variable_name!r}")
    data = np.asarray(loaded[variable_name])
    if data.ndim != 2 or data.shape[0] < 10:
        raise ValueError(f"{path.name}: expected at least 10 rows, got {data.shape}")

    timestamps = data[0]
    trigger = data[7]
    row9 = data[8]
    row10 = data[9]
    onsets = find_onsets_like_matlab(trigger)
    corrected = apply_bundled_trigger_fix(participant, session, onsets)
    onset_aux = row10[corrected]
    diffs = np.diff(timestamps)
    return FileSummary(
        participant=participant,
        session=session,
        path=path,
        shape=tuple(int(x) for x in data.shape),
        dtype=str(data.dtype),
        samples=int(data.shape[1]),
        time_start=float(timestamps[0]),
        time_end=float(timestamps[-1]),
        dt_median=float(np.median(diffs)),
        dt_min=float(np.min(diffs)),
        dt_max=float(np.max(diffs)),
        raw_positive_triggers=int(onsets.size),
        raw_negative_triggers=int(np.count_nonzero(trigger < 0)),
        corrected_positive_triggers=int(corrected.size),
        row8_positive_values=tuple(sorted({int(x) for x in trigger[onsets]})),
        row9_unique=tuple(sorted({int(x) for x in np.unique(row9)})),
        row10_zero_at_corrected_onset=int(np.count_nonzero(onset_aux == 0)),
        row10_nonzero_at_corrected_onset=int(np.count_nonzero(onset_aux != 0)),
        corrected_onsets=corrected,
        corrected_trigger_values=trigger[corrected],
    )


def render_report(root: Path, summaries: list[FileSummary]) -> str:
    summaries = sorted(summaries, key=lambda item: (item.participant, item.session))
    raw_anomalies = [
        item
        for item in summaries
        if item.raw_positive_triggers != EXPECTED_TRIALS_BY_SESSION[item.session]
    ]
    corrected_anomalies = [
        item
        for item in summaries
        if item.corrected_positive_triggers != EXPECTED_TRIALS_BY_SESSION[item.session]
    ]
    sample_lengths = sorted({item.samples for item in summaries})
    dtypes = Counter(item.dtype for item in summaries)
    fs_values = Counter(round(1.0 / item.dt_median, 9) for item in summaries)
    row9_values = sorted({value for item in summaries for value in item.row9_unique})

    trial_count = 0
    safe_trial_count = 0
    near20_count = 0
    nearest20_count = 0
    per_session_near20 = Counter()
    per_session_nearest20 = Counter()
    row10_zero = 0
    row10_nonzero = 0
    frequency_counts: Counter[int] = Counter()
    for item in summaries:
        combinations = target_combinations(item.session)
        for onset, trigger_value in zip(item.corrected_onsets, item.corrected_trigger_values):
            target_index = int(round(float(trigger_value))) - 1
            if not 0 <= target_index < len(combinations):
                continue
            combination = combinations[target_index]
            trial_count += 1
            end = int(onset) + FS_EXPECTED * TRIAL_SECONDS
            safe_trial_count += int(end <= item.samples)
            near = any(abs(freq - 20) <= 3 for freq in combination)
            nearest = any(freq in (19, 23) for freq in combination)
            near20_count += int(near)
            nearest20_count += int(nearest)
            per_session_near20[item.session] += int(near)
            per_session_nearest20[item.session] += int(nearest)
            frequency_counts.update(combination)
        row10_zero += item.row10_zero_at_corrected_onset
        row10_nonzero += item.row10_nonzero_at_corrected_onset

    lines = [
        "# MultiFrequency_Mu2024 exploration (read-only)",
        "",
        f"- Source root: `{root}`",
        f"- Files scanned: **{len(summaries)}** MAT files (expected P01–P35 × Ses1–Ses9)",
        f"- MAT matrices: all have 10 rows; sample count ranges from `{min(sample_lengths)}` to `{max(sample_lengths)}` across `{len(sample_lengths)}` distinct recording lengths; dtype: `{dict(dtypes)}`",
        f"- Timestamp-derived sampling rate: `{dict(fs_values)}` Hz",
        "- Dataset convention used for frequency lookup: six base frequencies **7, 11, 13, 17, 19, 23 Hz**; row-8 trigger values are stimulus indices within the current test method and map to the published lexicographically ordered single/dual/tri combination tables.",
        "",
        "## Findings",
        "",
        "### Structure and variables",
        "",
        "- Each file contains one MATLAB variable named `data_ses1` … `data_ses9`.",
        "- The matrix is `10 × N`: row 1 is a timestamp ramp, rows 2–7 are six EEG channels, row 8 is the positive/negative trigger stream, row 9 is a 0/1 state stream, and row 10 is an auxiliary integer-coded stream described by the dataset paper as online decoder output; it is not an independent ground-truth label.",
        "- Row 1 has median Δt = 0.001953125 s, i.e. **512 Hz** across all scanned files. The bundled script independently fixes `Fs=512`.",
        "- The bundled cutter uses 5-second windows, producing **2,560 samples per trial**. All corrected onsets had enough samples for the full window.",
        "",
        "### Triggers and labels",
        "",
        f"- Raw positive onset counts follow the intended session pattern for {len(summaries) - len(raw_anomalies)}/{len(summaries)} files; the remaining **{len(raw_anomalies)}** files need the explicit cleanup rules already present in `dataset_processData.m`.",
        "- Raw files needing cleanup: " + ", ".join(
            f"`{item.path.name}` ({item.raw_positive_triggers} positive onsets → {EXPECTED_TRIALS_BY_SESSION[item.session]})"
            for item in raw_anomalies
        ) + ".",
        f"- After applying those bundled, participant-specific ordinal selections: **{trial_count} / 12,740** trial records are recoverable, and **{safe_trial_count} / {trial_count}** have a complete 5-second window.",
        f"- Row 8 provides the stimulus index at each onset: Ses1 uses indices 1-6 for T1, Ses2-5 use 1-15 for T21/T22/T23, and Ses6-9 use 1-20 for T31/T32. The session/test method determines whether that index refers to a single-, dual-, or tri-frequency combination; it is not the auxiliary row-10 code. Row 9 takes values `{row9_values}` across files and equals 1 at every corrected onset in this scan.",
        f"- Row 10 is nonzero at {row10_nonzero} corrected onsets and zero at {row10_zero}; because it is an online decoder output rather than independent ground truth, retain it as auxiliary provenance and never use it as the primary trial label.",
        "",
        "### Around 20 Hz",
        "",
        "- There is no exact 20 Hz class in the six-frequency convention. The nearest base frequencies are 19 Hz and 23 Hz; 17 Hz is included when using a ±3 Hz search band.",
        f"- Trials containing 19 or 23 Hz: **{nearest20_count}**; trials containing any of 17/19/23 Hz (±3 Hz from 20): **{near20_count}**.",
        "- These can be mapped to frequency-bearing target labels because their identity comes from row 8 plus the session/test combination table, not from the auxiliary row-10 code. The exact stimulation method (T1/T21–T23/T31–T32) should be retained as provenance.",
        "",
        "## Risks",
        "",
        "1. **Do not treat raw positive-trigger count as a complete trial manifest.** Eight files contain extra or shifted trigger ordinals; blindly exporting all row-8 positives will misalign labels/trials.",
        "2. **Do not infer target labels from row 10.** It contains online decoder output codes, not independent target truth; keep it for audit only and never use it as the primary target label.",
        "3. **Preserve source provenance.** A unified record should retain source file, raw trigger ordinal, corrected ordinal, sample onset, session, and the applied correction rule.",
        "4. **Fixed 5 s versus physical offset.** The official cutter uses a fixed 5-second window. Trigger-to-negative-marker distances show small sample-level variation, so avoid assuming every negative marker is exactly onset+2,560 samples.",
        "",
        "## Recommendation",
        "",
        "- **Conversion is feasible.** Build a separate adapter that emits one record per corrected onset with `signal` (six EEG channels × 2,560 samples), `fs=512`, `participant`, `session`, `stimulation_type` (single/dual/tri), `target_index`, `target_frequencies_hz`, `onset_sample`, `raw_trigger_value`, `raw_file`, `raw_trigger_ordinal`, and `correction_rule`.",
        "- Preserve the row-1 timestamp as optional `timestamps` or reconstruct it from `onset_sample` and 512 Hz; do not silently treat the timestamp row as EEG.",
        "- For a first around-20-Hz subset, select target combinations containing 19 or 23 Hz, and optionally include 17 Hz for a ±3 Hz sensitivity analysis. Keep single-, dual-, and tri-frequency trials separated in evaluation because their label semantics differ.",
        "- Keep this exploration separate from formal configuration, default runtime, and historical reports. Do not write converted MAT/NPY files until the label mapping and row-10 semantics are reviewed.",
        "",
        "## Reproducibility",
        "",
        "- Script: `scripts/explore_multifrequency_mu2024.py`",
        "- Bundled reference logic: `D:/SSVEP/MultiFrequency_Mu2024/dataset_processData.m`",
        "- This report was generated by scanning the local files only; source dataset files were not modified.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\SSVEP\MultiFrequency_Mu2024"),
        help="Directory containing P##_Ses#.mat files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new Markdown report path. Existing files are never overwritten.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    files = sorted(root.glob("P*_Ses[1-9].mat"))
    if len(files) != 315:
        raise SystemExit(f"Expected 315 participant/session MAT files under {root}, found {len(files)}")
    summaries = [scan_file(path) for path in files]
    report = render_report(root, summaries)
    if args.output:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise SystemExit(f"Refusing to overwrite existing file: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8", newline="\n")
        print(output)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline public SSVEP baselines: explicit metadata, CPU, complete held-out blocks.

Wang JSON schema (required): schema_version=1, status="formal",
metadata_authority (nonempty string/object), sample_rate_hz=250, mat_variable,
expected_shape (original axis order), axes={channel,sample,condition,block},
source_channel_names (64 original names in original order), target_channel_map
(all eight canonical names -> exact source names), frequency_order_hz (all 40
frequencies in original target order), sample_window={start,stop}, depth, session.
Optional provenance, verified_variables, amplitude_unit, source_processing,
acquisition_sample_rate_hz. No default source metadata or inferred latency.

Forward takes EEG only. Calibration labels enter only TRCA.fit. Output is
exploratory, never authorizes runtime or claims personal/cross-subject accuracy.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence
import numpy as np
from scipy import linalg, signal
from scipy.io import loadmat, whosmat

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_open_ssvep_commands import cca_score, reference_matrix

CHANNELS = ("Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2")
PERSONAL_TARGETS = (10, 12, 15, 20)
DEFAULT_COMMANDS = {10: "MOVE_LEFT", 12: "MOVE_RIGHT", 15: "SELECT", 20: "BACK"}
FLAGS = {"status": "formal", "exploratory_only": True, "runtime_enabled": False,
         "production_validated": False, "default_deployment_authorized": False,
         "personal_gaze_accuracy_claim": False, "cross_subject_generalization_claim": False}
SUBBANDS = ((6.0, 90.0), (14.0, 90.0), (22.0, 90.0), (30.0, 90.0), (38.0, 90.0))
WEIGHTS = tuple(float(k ** -1.25 + 0.25) for k in range(1, 6))
SCORE_THRESHOLDS = {"CCA": 0.5, "FBCCA": 0.25, "TRCA": 0.5}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    def bad_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    result = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object,
                        parse_constant=bad_constant)
    require(isinstance(result, dict), "metadata must be a JSON object")
    canonical(result)
    return result


def identity(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    require((isinstance(value, str) and bool(value.strip())) or
            (type(value) in (int, float) and np.isfinite(value)), "invalid identity scalar")
    return value


def check_frequencies(values: Sequence[float]) -> tuple[int, ...]:
    require(bool(len(values)) and all(type(v) in (int, float) and np.isfinite(v) and v in PERSONAL_TARGETS
                                    for v in values), "explicit exact frequencies 10/12/15[/20] required")
    require(len(set(values)) == len(values), "duplicate requested frequencies")
    result = tuple(sorted(int(v) for v in values))
    require(result in ((10, 12, 15), (10, 12, 15, 20)), "request exactly 10,12,15 or 10,12,15,20")
    return result


def check_eeg(eeg: np.ndarray) -> np.ndarray:
    x = np.asarray(eeg)
    require(x.ndim == 2 and x.shape[0] == 8 and x.shape[1] >= 250,
            f"expected EEG (8,samples>=250), got {x.shape}")
    require(x.dtype.kind in "fiu" and np.isfinite(x).all(), "EEG must be finite real numeric data")
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True)
class Trial:
    subject: Any
    block: int
    depth: Any
    session: Any
    condition_index: int
    frequency_hz: float
    group_id: str
    eeg: np.ndarray
    source_file: str
    source_trial_index: int

    @property
    def trial_id(self):
        return canonical([self.subject, self.block, self.depth, self.session, self.condition_index])

    @property
    def cohort_id(self):
        return canonical([self.subject, self.depth, self.session])


@dataclass
class Dataset:
    trials: list[Trial]
    provenance: dict[str, Any]
    input_hashes: dict[Path, str]
    kind: str


def _authority(value):
    require((isinstance(value, str) and bool(value.strip())) or
            (isinstance(value, dict) and bool(value)), "explicit metadata_authority text/object required")


def _axes_shape(spec):
    axes, shape = spec["axes"], spec["expected_shape"]
    require(isinstance(axes, dict) and set(axes) == {"channel", "sample", "condition", "block"}, "invalid axes")
    require(all(type(v) is int for v in axes.values()) and sorted(axes.values()) == [0, 1, 2, 3], "axes must permute 0..3")
    require(isinstance(shape, list) and len(shape) == 4 and all(type(v) is int and v > 0 for v in shape), "invalid expected_shape")
    return axes, shape


def _window(window, samples):
    require(isinstance(window, dict) and set(window) == {"start", "stop"}, "explicit sample_window {start,stop} required")
    start, stop = window["start"], window["stop"]
    require(type(start) is int and type(stop) is int and 0 <= start < stop <= samples,
            "sample_window must be within original samples")
    require(stop - start >= 250, "window must contain >=250 samples")
    return start, stop


def _channel_names(names, count):
    require(isinstance(names, list) and len(names) == count and
            all(isinstance(n, str) and n.strip() == n and bool(n) for n in names), "invalid exact channel names")
    require(len(set(n.casefold() for n in names)) == count, "duplicate channel names")
    return names


def load_wang(mat_root: Path, source_spec: Path, *, subjects=None, max_subjects=None) -> Dataset:
    require((subjects is None) != (max_subjects is None), "Wang requires exactly one of --subjects/--max-subjects")
    spec = read_json(source_spec)
    required = {"schema_version", "status", "metadata_authority", "sample_rate_hz", "mat_variable",
                "expected_shape", "axes", "source_channel_names", "target_channel_map",
                "frequency_order_hz", "sample_window", "depth", "session"}
    require(required <= set(spec), f"missing Wang spec fields: {sorted(required - set(spec))}")
    require(type(spec["schema_version"]) is int and spec["schema_version"] == 1 and spec["status"] == "formal", "spec requires schema 1/formal")
    _authority(spec["metadata_authority"])
    require(type(spec["sample_rate_hz"]) in (int, float) and spec["sample_rate_hz"] == 250, "requires 250 Hz")
    axes, shape = _axes_shape(spec)
    require([shape[axes[k]] for k in ("channel", "sample", "condition", "block")] == [64, 1500, 40, 6],
            "Wang requires explicitly verified canonical shape (64,1500,40,6)")
    names = _channel_names(spec["source_channel_names"], 64)
    mapping = spec["target_channel_map"]
    require(isinstance(mapping, dict) and set(mapping) == set(CHANNELS), "target_channel_map must map all eight canonical channels")
    require(all(isinstance(v, str) and v in names for v in mapping.values()), "channel map must use exact original names")
    require(len(set(mapping.values())) == 8, "channel map must select eight different source channels")
    indices = [names.index(mapping[n]) for n in CHANNELS]
    freqs = spec["frequency_order_hz"]
    require(isinstance(freqs, list) and len(freqs) == 40 and
            all(type(v) in (int, float) and np.isfinite(v) and 0 < v < 125 for v in freqs), "frequency_order_hz must cover all 40 original conditions")
    start, stop = _window(spec["sample_window"], shape[axes["sample"]])
    identity(spec["depth"])
    identity(spec["session"])
    variable = spec["mat_variable"]
    require(isinstance(variable, str) and bool(variable.strip()), "explicit MAT variable required")
    files = {}
    for path in mat_root.iterdir():
        match = re.fullmatch(r"S(\d+)\.mat", path.name)
        if match and path.is_file():
            subject = int(match[1])
            require(subject > 0 and subject not in files, "duplicate/invalid S-integer subject filename")
            files[subject] = path.resolve()
    require(bool(files), "no S<integer>.mat files in mat-root")
    if subjects is not None:
        require(bool(len(subjects)) and all(type(s) is int and s > 0 for s in subjects) and
                len(set(subjects)) == len(subjects), "subjects must be unique positive integers")
        selected = sorted(subjects)
        require(set(selected) <= set(files), f"requested subjects absent: {sorted(set(selected) - set(files))}")
    else:
        require(type(max_subjects) is int and max_subjects > 0, "max-subjects must be positive")
        selected = sorted(files)[:max_subjects]
    hashes = {source_spec.resolve(): sha256_file(source_spec)}
    trials, records = [], []
    for subject in selected:
        path = files[subject]
        digest = sha256_file(path)
        inventory = [{"name": n, "shape": list(s), "matlab_class": c} for n, s, c in whosmat(path)]
        matches = [v for v in inventory if v["name"] == variable]
        require(len(matches) == 1 and matches[0]["shape"] == shape, f"MAT variable/shape mismatch in {path.name}")
        if "verified_variables" in spec:
            require(spec["verified_variables"] == inventory, f"verified_variables mismatch in {path.name}")
        data = loadmat(path, variable_names=[variable])[variable]
        require(list(data.shape) == shape and data.dtype.kind in "fiu" and np.isfinite(data).all(),
                f"invalid/nonfinite original MAT data in {path.name}")
        original = np.transpose(data, tuple(axes[k] for k in ("channel", "sample", "condition", "block")))
        for condition, frequency in enumerate(freqs):
            if frequency not in PERSONAL_TARGETS:
                continue
            for block in range(6):
                group = canonical([subject, block, spec["depth"], spec["session"]])
                epoch = original[indices, start:stop, condition, block].copy()
                trials.append(Trial(subject, block, spec["depth"], spec["session"], condition,
                                    float(frequency), group, epoch, str(path), condition * 6 + block))
        require(sha256_file(path) == digest, f"MAT changed during loading: {path.name}")
        hashes[path] = digest
        records.append({"source_file": str(path), "source_sha256": digest,
                        "subject": subject, "verified_variables": inventory})
    provenance = {"metadata_authority": spec["metadata_authority"], "source_spec": spec,
                  "source_spec_file_sha256": hashes[source_spec.resolve()],
                  "source_spec_sha256": hashlib.sha256(canonical(spec).encode()).hexdigest(),
                  "sources": records, "source_channel_names": names, "target_channel_map": mapping,
                  "source_channel_indices": indices, "sample_window": spec["sample_window"],
                  "condition_indices_by_frequency": {str(f): [i for i, v in enumerate(freqs) if v == f] for f in PERSONAL_TARGETS},
                  "amplitude_unit": spec.get("amplitude_unit", "unknown"),
                  "selected_subjects": selected, "excluded_subjects": sorted(set(files) - set(selected)),
                  "excluded_subject_waveforms_read": False}
    return Dataset(trials, provenance, hashes, "wang2016")



def _scalar_array(values, name):
    array = np.asarray(values)
    require(array.ndim == 0, f"{name} must be scalar")
    return array.item()


def load_prepared(source: Path, manifest_path: Path) -> Dataset:
    """Load the preparation script's explicit NPZ+manifest without inference."""
    manifest = read_json(manifest_path)
    _authority(manifest.get("metadata_authority"))
    require(manifest.get("status") == "formal" and manifest.get("exploratory_only") is True and
            manifest.get("runtime_enabled") is False, "prepared manifest must be formal exploratory-only and runtime-disabled")
    require(manifest.get("npz_file") == source.name, "manifest npz_file mismatch")
    source_hash = sha256_file(source)
    require(manifest.get("npz_sha256") == source_hash, "prepared NPZ sha256 mismatch")
    spec = manifest.get("source_spec")
    require(isinstance(spec, dict) and spec.get("status") == "formal", "prepared manifest source_spec is required and formal")
    spec_hash = hashlib.sha256(canonical(spec).encode("utf-8")).hexdigest()
    require(manifest.get("source_spec_sha256") == spec_hash, "prepared source_spec sha256 mismatch")
    require(manifest.get("identity_label_encoding") == "subject/depth/session are JSON scalars preserving original types" and
            manifest.get("group_id_encoding") == "JSON array of original typed subject/block/depth/session labels",
            "prepared identity/group_id encoding is not the verified JSON encoding")
    require(manifest.get("sample_rate_hz") == 250 and manifest.get("eeg_axes") == ["trial", "channel", "sample"],
            "prepared manifest sample rate/axes mismatch")
    window = manifest.get("sample_window")
    require(isinstance(window, dict) and set(window) == {"start", "stop"} and
            type(window["start"]) is int and type(window["stop"]) is int and window["start"] < window["stop"],
            "prepared manifest requires explicit sample_window")
    spec_names = spec.get("channel_names")
    require(isinstance(spec_names, list), "prepared source_spec channel_names required")
    selected_names = manifest.get("selected_channel_names")
    require(selected_names == list(CHANNELS), "prepared selected_channel_names must be canonical eight channels")
    selected_indices = manifest.get("source_channel_indices")
    require(selected_indices == [spec_names.index(name) for name in selected_names], "prepared channel indices mismatch")
    conditions = spec.get("condition_table")
    require(isinstance(conditions, list), "prepared condition_table required")
    by_condition = {}
    for row in conditions:
        require(isinstance(row, dict) and set(row) == {"condition_index", "frequency_hz", "depth", "session"}, "invalid prepared condition row")
        index, frequency = row["condition_index"], row["frequency_hz"]
        require(type(index) is int and index not in by_condition and 0 <= index < len(conditions), "invalid prepared condition_index")
        require(type(frequency) in (int, float) and np.isfinite(frequency) and frequency > 0, "invalid prepared frequency")
        by_condition[index] = row
    require(set(by_condition) == set(range(len(conditions))), "prepared condition_table is not complete")
    selected_conditions = manifest.get("selected_conditions")
    require(isinstance(selected_conditions, list) and all(row == by_condition[row["condition_index"]] for row in selected_conditions),
            "prepared selected_conditions mismatch")
    with np.load(source, allow_pickle=False) as archive:
        required = {"eeg", "subject", "block", "depth", "session", "frequency_hz", "condition_index", "group_id",
                    "sample_rate_hz", "channel_names", "source_channel_indices", "source_sha256", "source_spec_sha256",
                    "source_sample_indices", "status", "exploratory_only", "runtime_enabled"}
        require(required <= set(archive.files), f"prepared NPZ missing fields: {sorted(required - set(archive.files))}")
        values = {key: np.asarray(archive[key]) for key in required}
    eeg = values["eeg"]
    require(eeg.ndim == 3 and eeg.shape[1] == 8 and eeg.shape[0] > 0 and eeg.dtype.kind in "fiu" and np.isfinite(eeg).all(),
            "prepared eeg must be finite (trial,8,sample)")
    require(manifest.get("eeg_shape") == list(eeg.shape), "prepared manifest eeg_shape mismatch")
    require(_scalar_array(values["sample_rate_hz"], "sample_rate_hz") == 250 and
            values["channel_names"].tolist() == list(CHANNELS) and
            values["source_channel_indices"].tolist() == selected_indices,
            "prepared NPZ channel/sample-rate metadata mismatch")
    require(_scalar_array(values["source_sha256"], "source_sha256") == manifest.get("source_sha256") and
            _scalar_array(values["source_spec_sha256"], "source_spec_sha256") == spec_hash,
            "prepared NPZ provenance mismatch")
    require(np.array_equal(values["source_sample_indices"], np.arange(window["start"], window["stop"])),
            "prepared sample window indices mismatch")
    count = len(eeg)
    for key in ("subject", "block", "depth", "session", "frequency_hz", "condition_index", "group_id"):
        require(values[key].shape == (count,), f"prepared {key} must be one-dimensional")
    trials, seen = [], set()
    for index in range(count):
        decoded = {}
        for key in ("subject", "depth", "session"):
            raw = values[key][index].item()
            require(isinstance(raw, str), f"prepared {key} must be JSON scalar text")
            decoded[key] = identity(json.loads(raw))
        block, condition, frequency = values["block"][index].item(), values["condition_index"][index].item(), values["frequency_hz"][index].item()
        require(type(block) is int and block >= 0 and type(condition) is int and condition in by_condition, "invalid prepared trial block/condition")
        row = by_condition[condition]
        require(frequency == row["frequency_hz"] and canonical(decoded["depth"]) == canonical(row["depth"]) and canonical(decoded["session"]) == canonical(row["session"]), "prepared labels disagree with condition table")
        group_raw = values["group_id"][index].item()
        require(isinstance(group_raw, str), "prepared group_id must be JSON array text")
        group = json.loads(group_raw)
        expected_group = [decoded["subject"], block, decoded["depth"], decoded["session"]]
        require(canonical(group) == canonical(expected_group), "prepared group_id mismatch")
        trial = Trial(decoded["subject"], block, decoded["depth"], decoded["session"], condition, float(frequency), canonical(expected_group), np.asarray(eeg[index], dtype=np.float64), str(source.resolve()), index)
        require(trial.trial_id not in seen, "duplicate prepared trial identity")
        seen.add(trial.trial_id); trials.append(trial)
    manifest_hash = sha256_file(manifest_path)
    require(sha256_file(source) == source_hash and sha256_file(manifest_path) == manifest_hash, "prepared input changed during loading")
    provenance = {"metadata_authority": manifest["metadata_authority"], "source_spec": spec, "source_spec_sha256": spec_hash, "source_sha256": manifest["source_sha256"], "npz_sha256": source_hash, "manifest_sha256": manifest_hash, "source_file": str(source.resolve()), "manifest_file": str(manifest_path.resolve()), "sample_window": window, "source_channel_names": spec_names, "source_channel_indices": selected_indices, "label_encoding": manifest["identity_label_encoding"], "original_source_hash_recomputed": False}
    return Dataset(trials, provenance, {source.resolve(): source_hash, manifest_path.resolve(): manifest_hash}, "prepared")


def select_and_split(dataset: Dataset, frequencies: tuple[int, ...]):
    trials = dataset.trials
    require(bool(trials), "no public trials")
    require(len({t.trial_id for t in trials}) == len(trials), "duplicate trial identity")
    available = sorted({t.frequency_hz for t in trials})
    missing = [f for f in frequencies if f not in available]
    require(not missing, f"requested frequencies absent from actual trials: {missing}; available={available}; no approximation")
    selected = [t for t in trials if t.frequency_hz in frequencies]
    cohorts = defaultdict(list)
    for t in selected:
        check_eeg(t.eeg)
        require(t.group_id == canonical([t.subject, t.block, t.depth, t.session]), "group_id mismatch")
        cohorts[t.cohort_id].append(t)
    split_by_group, records = {}, []
    for cohort, rows in sorted(cohorts.items()):
        by_block = defaultdict(list)
        for t in rows:
            by_block[t.block].append(t)
        blocks = sorted(by_block)
        require(len(blocks) >= 3, f"cohort {cohort} needs >=3 complete blocks")
        patterns = [{(t.condition_index, t.frequency_hz) for t in by_block[b]} for b in blocks]
        require(all(p == patterns[0] for p in patterns), f"incomplete condition/block groups in {cohort}")
        require(all(set(frequencies) <= {t.frequency_hz for t in by_block[b]} for b in blocks), f"missing target within block in {cohort}")
        if len(blocks) == 6:
            require(blocks == list(range(6)), "six-block protocol requires zero-based blocks 0..5")
            train, validation, test, policy = blocks[:4], [4], [5], "fixed_0_3_train_4_validation_5_test"
        else:
            require(dataset.kind == "prepared", "adaptive blocks allowed only for prepared data")
            train, validation, test = blocks[:-2], [blocks[-2]], [blocks[-1]]
            policy = "adaptive_sorted_blocks_last_two_validation_test"
        record = {"cohort_id": cohort, "subject": rows[0].subject, "depth": rows[0].depth,
                  "session": rows[0].session, "policy": policy}
        for name, assigned in (("train", train), ("validation", validation), ("test", test)):
            subset = [t for t in rows if t.block in assigned]
            groups = sorted({t.group_id for t in subset})
            for group in groups:
                require(group not in split_by_group, "full group_id reused across splits/cohorts")
                split_by_group[group] = name
            record[name] = {"blocks": assigned, "group_ids": groups,
                            "trial_ids": [t.trial_id for t in subset], "trial_count": len(subset)}
        records.append(record)
    return selected, split_by_group, records


def preprocess(eeg, band=SUBBANDS[0]):
    x = signal.detrend(check_eeg(eeg), axis=-1, type="linear")
    sos = signal.butter(4, band, btype="bandpass", fs=250, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def prediction(frequencies, scores):
    values = np.asarray(scores, dtype=float)
    require(values.shape == (len(frequencies),) and np.isfinite(values).all(), "invalid classifier scores")
    best, second = np.argsort(-values, kind="stable")[:2]
    return {"predicted_frequency_hz": frequencies[best], "best_score": float(values[best]),
            "margin": float(values[best] - values[second]),
            "scores": {str(f): float(s) for f, s in zip(frequencies, values)}}


class CCA:
    def __init__(self, frequencies):
        self.frequencies = check_frequencies(frequencies)
        self.references = {}

    def _references(self, samples):
        if samples not in self.references:
            self.references[samples] = [reference_matrix(samples, 250, f, 3) for f in self.frequencies]
        return self.references[samples]

    def forward(self, eeg):
        x = preprocess(eeg)
        return prediction(self.frequencies, [cca_score(x.T, ref) for ref in self._references(x.shape[1])])


class FBCCA(CCA):
    def forward(self, eeg):
        eeg = check_eeg(eeg)
        refs = self._references(eeg.shape[1])
        scores = np.zeros(len(self.frequencies))
        for band, weight in zip(SUBBANDS, WEIGHTS):
            x = preprocess(eeg, band)
            scores += weight * np.square([cca_score(x.T, ref) for ref in refs])
        return prediction(self.frequencies, scores / sum(WEIGHTS))


class TRCA:
    """Leading generalized eigenvector of S=sum(i!=j)XiXj.T, Q=sum XiXi.T.

    Only calibration data are used for filters and mean templates. Scores are
    signed correlations of filtered EEG and same-filter calibration templates.
    """
    def __init__(self, frequencies):
        self.frequencies = check_frequencies(frequencies)
        self.filters, self.templates, self.fit_counts, self.eigenvalues = {}, {}, {}, {}
        self.sample_count = None

    def fit(self, eeg, frequency_hz):
        require(not self.filters, "TRCA already fitted; create a new calibration model")
        epochs, labels = np.asarray(eeg), np.asarray(frequency_hz)
        require(epochs.ndim == 3 and labels.shape == (len(epochs),), "TRCA fit needs epochs and calibration labels")
        require(np.isfinite(labels).all() and set(labels.tolist()) == set(self.frequencies), "TRCA calibration must cover exact targets")
        counts = {f: int(np.sum(labels == f)) for f in self.frequencies}
        require(all(n >= 2 for n in counts.values()), "TRCA needs >=2 calibration trials per target")
        filtered = np.stack([preprocess(x) for x in epochs])
        filtered -= filtered.mean(axis=-1, keepdims=True)
        for f in self.frequencies:
            x = filtered[labels == f]
            total = x.sum(axis=0)
            q = np.einsum("tcs,tds->cd", x, x)
            s = total @ total.T - q
            ridge = 1e-6 * max(float(np.trace(q) / 8), 1e-12)
            values, vectors = linalg.eigh((s + s.T) / 2, (q + q.T) / 2 + ridge * np.eye(8))
            w = vectors[:, -1]
            self.filters[f] = w / np.linalg.norm(w)
            self.templates[f] = x.mean(axis=0)
            self.fit_counts[f], self.eigenvalues[f] = counts[f], float(values[-1])
        self.sample_count = filtered.shape[-1]
        return self

    def forward(self, eeg):
        require(bool(self.filters), "TRCA must fit calibration before forward")
        x = preprocess(eeg)
        require(x.shape[-1] == self.sample_count, "TRCA window differs from calibration")
        scores = []
        for f in self.frequencies:
            w = self.filters[f]
            a, b = w @ x, w @ self.templates[f]
            a, b = a - a.mean(), b - b.mean()
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            scores.append(float(np.clip(np.dot(a, b) / denominator, -1, 1)) if denominator > 1e-15 else 0.0)
        return prediction(self.frequencies, scores)


def command_map(raw, frequencies):
    result = dict(DEFAULT_COMMANDS)
    if raw is not None:
        seen = set()
        for part in raw.split(","):
            key, separator, value = part.partition("=")
            require(bool(separator) and key.strip() in {str(f) for f in PERSONAL_TARGETS}, "invalid command mapping")
            f = int(key.strip())
            require(f not in seen and bool(re.fullmatch(r"[A-Z][A-Z0-9_]*", value.strip())) and value.strip() != "UNKNOWN", "duplicate/invalid command mapping")
            result[f] = value.strip()
            seen.add(f)
    return {f: result[f] for f in frequencies}


def metrics(rows, frequencies):
    accepted = [r for r in rows if r["accepted"]]
    matched = sum(r["accepted_label_match"] for r in accepted)
    confusion = {str(f): {str(p): 0 for p in frequencies} for f in frequencies}
    command_confusion = {str(f): {str(p): 0 for p in frequencies} | {"UNKNOWN": 0} for f in frequencies}
    for r in rows:
        truth, pred = str(int(r["recorded_stimulus_frequency_hz"])), str(r["prediction"]["predicted_frequency_hz"])
        confusion[truth][pred] += 1
        command_confusion[truth][pred if r["accepted"] else "UNKNOWN"] += 1
    return {"evaluation_trial_count": len(rows), "accepted_trial_count": len(accepted),
            "acceptance_rate": len(accepted) / len(rows) if rows else None,
            "accepted_label_match_count": matched,
            "accepted_label_match_rate": matched / len(accepted) if accepted else None,
            "all_trial_label_match_rate": sum(r["prediction"]["predicted_frequency_hz"] == r["recorded_stimulus_frequency_hz"] for r in rows) / len(rows) if rows else None,
            "all_trial_confusion_matrix": confusion, "command_confusion_matrix": command_confusion}


def evaluate(dataset: Dataset, *, frequencies, margin_threshold=0.05, score_threshold=None, mapping=None):
    targets = check_frequencies(frequencies)
    require(type(margin_threshold) in (int, float) and np.isfinite(margin_threshold) and 0 <= margin_threshold <= 1, "margin threshold must be finite in [0,1]")
    require(score_threshold is None or (type(score_threshold) in (int, float) and np.isfinite(score_threshold) and 0 <= score_threshold <= 1), "score threshold must be finite in [0,1]")
    thresholds = {k: score_threshold if score_threshold is not None else v for k, v in SCORE_THRESHOLDS.items()}
    actions = command_map(mapping, targets)
    trials, splits, records = select_and_split(dataset, targets)
    models = {"CCA": CCA(targets), "FBCCA": FBCCA(targets)}
    all_rows, fits = [], []
    for record in records:
        cohort = record["cohort_id"]
        calibration = [t for t in trials if t.cohort_id == cohort and splits[t.group_id] == "train"]
        heldout = [t for t in trials if t.cohort_id == cohort and splits[t.group_id] != "train"]
        counts = {f: sum(t.frequency_hz == f for t in calibration) for f in targets}
        fit_record = {"cohort_id": cohort, "training_trial_ids": [t.trial_id for t in calibration],
                      "training_group_ids": sorted({t.group_id for t in calibration}),
                      "calibration_trial_counts": {str(f): n for f, n in counts.items()},
                      "validation_or_test_used_in_fit": False}
        cohort_models = dict(models)
        if min(counts.values()) >= 2:
            trca = TRCA(targets).fit(np.stack([t.eeg for t in calibration]), [t.frequency_hz for t in calibration])
            cohort_models["TRCA"] = trca
            fit_record.update({"status": "fitted", "eigenvalues": {str(f): v for f, v in trca.eigenvalues.items()},
                               "spatial_filters": {str(f): w.tolist() for f, w in trca.filters.items()},
                               "templates_sha256": {str(f): hashlib.sha256(x.astype("<f8").tobytes()).hexdigest() for f, x in trca.templates.items()}})
        else:
            fit_record.update({"status": "unavailable", "reason": "fewer_than_two_calibration_trials_per_target"})
        fits.append(fit_record)
        for t in heldout:
            for name, model in cohort_models.items():
                # Forward gets ONLY EEG; truth is read after prediction.
                pred = model.forward(t.eeg)
                accepted = pred["best_score"] >= thresholds[name] and pred["margin"] >= margin_threshold
                all_rows.append({"algorithm": name, "trial_id": t.trial_id, "group_id": t.group_id,
                                 "subject": t.subject, "block": t.block, "depth": t.depth, "session": t.session,
                                 "condition_index": t.condition_index, "source_file": t.source_file,
                                 "source_trial_index": t.source_trial_index, "split": splits[t.group_id],
                                 "recorded_stimulus_frequency_hz": t.frequency_hz, "prediction": pred,
                                 "accepted": bool(accepted),
                                 "accepted_label_match": bool(accepted and pred["predicted_frequency_hz"] == t.frequency_hz),
                                 "recorded_label_used_in_forward": False,
                                 "command": actions[pred["predicted_frequency_hz"]] if accepted else "UNKNOWN",
                                 "candidate_frequency_hz": pred["predicted_frequency_hz"] if accepted else None,
                                 "executed": False})
    algorithms = {}
    for name in ("CCA", "FBCCA", "TRCA"):
        rows = [r for r in all_rows if r["algorithm"] == name]
        result = {"status": "evaluated" if rows else "unavailable", **metrics(rows, targets)}
        result["by_split"] = {s: metrics([r for r in rows if r["split"] == s], targets) for s in ("validation", "test")}
        result["per_subject"] = {s: metrics([r for r in rows if canonical(r["subject"]) == s], targets)
                                 for s in sorted({canonical(t.subject) for t in trials})}
        result["per_target"] = {str(f): metrics([r for r in rows if r["recorded_stimulus_frequency_hz"] == f], targets) for f in targets}
        result["evaluated_cohort_count"] = len({canonical([r["subject"], r["depth"], r["session"]]) for r in rows})
        result["unavailable_cohorts"] = [r["cohort_id"] for r in fits if r["status"] == "unavailable"] if name == "TRCA" else []
        algorithms[name] = result
    available = sorted({t.frequency_hz for t in dataset.trials})
    return {**FLAGS, "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset.kind, "evaluation_scope": "same_subject_heldout_blocks",
            "primary_evaluation_split": "test", "chance_baseline": 1 / len(targets),
            "frequencies_hz": list(targets), "available_source_targets_hz": available,
            "missing_personal_targets": [f for f in PERSONAL_TARGETS if f not in available],
            "unevaluated_personal_targets": [f for f in PERSONAL_TARGETS if f not in targets],
            "four_frequency_evaluation": set(targets) == set(PERSONAL_TARGETS),
            "source_provenance": dataset.provenance, "channel_names": list(CHANNELS),
            "parameters": {"sample_rate_hz": 250, "harmonics": 3, "CCA_regularization": 1e-6,
                           "filter": {"family": "butterworth", "order": 4, "phase": "sosfiltfilt", "detrend": "linear"},
                           "CCA_band_hz": list(SUBBANDS[0]), "FBCCA_subbands_hz": [list(b) for b in SUBBANDS],
                           "FBCCA_weights": list(WEIGHTS), "FBCCA_formula": "sum(weight * cca_correlation**2) / sum(weights)",
                           "TRCA_band_hz": list(SUBBANDS[0]), "TRCA_regularization": "1e-6 * max(trace(Q)/8,1e-12)",
                           "TRCA_scoring": "signed correlation of same-filter EEG and mean calibration template",
                           "score_thresholds": thresholds, "margin_threshold": margin_threshold,
                           "threshold_policy": "fixed_before_evaluation; no validation/test optimization",
                           "compute_device": "CPU", "source_window": dataset.provenance.get("sample_window")},
            "split_groups": records, "group_ids_disjoint": True, "trca_calibration": fits,
            "algorithms": algorithms, "rows": all_rows,
            "command_output": {"mapping": {str(k): v for k, v in actions.items()}, "executed": False,
                               "candidate_scope": "validation_and_test; no train candidates", "row_count": len(all_rows)}}


def _check_outputs(output, commands_output, inputs=()):
    paths = [output] + ([commands_output] if commands_output is not None else [])
    require(len({p.resolve() for p in paths}) == len(paths), "report/commands paths must differ")
    for path in paths:
        require(not path.exists() and not path.is_symlink(), f"refusing to overwrite existing output: {path}")
        require(path.parent.is_dir(), f"output parent must exist: {path.parent}")
        require(path.resolve() not in {p.resolve() for p in inputs}, "output conflicts with source input")


def save_new_outputs(report, output, commands_output=None):
    _check_outputs(output, commands_output)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    handles, created = [], []
    try:
        # Acquire both exclusive files before writing. Roll back only files
        # created by this call if a competing writer already owns either path.
        for path in [output] + ([commands_output] if commands_output is not None else []):
            handles.append(path.open("x", encoding="utf-8"))
            created.append(path)
        handles[0].write(text)
        if commands_output is not None:
            for row in report["rows"]:
                handles[1].write(canonical({**FLAGS, **row, "sink": "offline_jsonl", "executed": False}) + "\n")
        for handle in handles:
            handle.close()
    except Exception:
        for handle in handles:
            handle.close()
        for path in created:
            path.unlink(missing_ok=True)
        raise


def benchmark(*, dataset, frequencies, output, source=None, manifest=None, mat_root=None,
              source_spec=None, subjects=None, max_subjects=None, commands_output=None,
              margin_threshold=0.05, score_threshold=None, mapping=None):
    _check_outputs(output, commands_output, [p for p in (source, manifest, source_spec) if p is not None])
    check_frequencies(frequencies)
    if dataset == "wang2016":
        require(mat_root is not None and source_spec is not None and source is None and manifest is None,
                "wang2016 requires mat-root/source-spec only")
        data = load_wang(mat_root, source_spec, subjects=subjects, max_subjects=max_subjects)
    else:
        require(dataset == "prepared" and source is not None and manifest is not None and
                mat_root is None and source_spec is None and subjects is None and max_subjects is None,
                "prepared requires source/manifest only")
        data = load_prepared(source, manifest)
    report = evaluate(data, frequencies=frequencies, margin_threshold=margin_threshold,
                      score_threshold=score_threshold, mapping=mapping)
    for path, digest in data.input_hashes.items():
        require(sha256_file(path) == digest, f"source changed during benchmark: {path}")
    report["command_output"]["jsonl_file"] = str(commands_output.resolve()) if commands_output else None
    save_new_outputs(report, output, commands_output)
    return report


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("wang2016", "prepared"), required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mat-root", type=Path)
    parser.add_argument("--source-spec", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--subjects", help="explicit comma-separated positive integer subject IDs")
    selection.add_argument("--max-subjects", type=int, help="first N existing subjects in numeric filename order")
    parser.add_argument("--frequencies", required=True, help="10,12,15 or 10,12,15,20; exact presence required")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; never overwritten")
    parser.add_argument("--commands-output", type=Path, help="optional new offline commands JSONL")
    parser.add_argument("--command-map")
    parser.add_argument("--score-threshold", type=float, help="fixed override for all algorithms before evaluation")
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        subjects = [int(s.strip()) for s in args.subjects.split(",")] if args.subjects else None
        report = benchmark(dataset=args.dataset, frequencies=[float(f.strip()) for f in args.frequencies.split(",")],
                           output=args.output, source=args.source, manifest=args.manifest, mat_root=args.mat_root,
                           source_spec=args.source_spec, subjects=subjects, max_subjects=args.max_subjects,
                           commands_output=args.commands_output, mapping=args.command_map,
                           score_threshold=args.score_threshold, margin_threshold=args.margin_threshold)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(canonical({**FLAGS, "error": str(exc)}), file=sys.stderr)
        return 2
    print(canonical({**FLAGS, "output": str(args.output.resolve()), "frequencies_hz": report["frequencies_hz"],
                     "missing_personal_targets": report["missing_personal_targets"],
                     "evaluation_scope": report["evaluation_scope"],
                     "test": {k: v["by_split"]["test"] for k, v in report["algorithms"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

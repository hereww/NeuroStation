"""V10 experiment data identity and controlled corruption; stdlib-only import.

Recorded references are notch-conditioned surrogates, never biological truth.
Subject IDs define disjoint new-experiment roles; previous project use is disclosed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
CONFIG = "configs/learned_denoiser_v10_personal.json"
LOCK = "analysis_outputs/ssvep_v10_training_lock_20261013.json"


def require(ok, message):
    if not ok:
        raise ValueError("v10: " + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def load_config(root=ROOT):
    cfg = json.loads((Path(root) / CONFIG).read_text(encoding="utf-8"))
    require(cfg.get("status") == "formal" and cfg.get("model_version") == "learned_denoise_v10_personal",
            "invalid configuration version")
    require(cfg.get("sample_rate_hz") == 250 and cfg.get("channel_count") == 8
            and cfg.get("window_samples") == 512 and cfg.get("hop_samples") == 128, "invalid native input contract")
    require(cfg["data"]["fit_subjects"] == [1,2,3,4] and cfg["data"]["selection_subjects"] == [5,6]
            and cfg["data"]["evaluation_subjects"] == [7,8], "subject roles changed")
    require(cfg["runtime"]["enabled"] is False and cfg["runtime"]["production_validated"] is False,
            "unvalidated runtime cannot be enabled")
    return cfg


def preflight(root=ROOT):
    root = Path(root)
    cfg = load_config(root)
    lock = json.loads((root / LOCK).read_text(encoding="utf-8"))
    require(lock.get("status") == "formal" and lock.get("model_version") == cfg["model_version"], "invalid training lock")
    require(lock.get("config_sha256") == canonical_sha(cfg), "configuration lock mismatch")
    files = lock.get("files")
    require(isinstance(files, dict) and files, "training lock files missing")
    required = {CONFIG, "eeg_tools/ssvep_denoise_v10.py", "eeg_tools/ssvep_v10_data.py",
                "scripts/train_ssvep_denoiser_v10.py"}
    for subject in range(1,7):
        name = f'{cfg["data"]["reference_directory"]}/subject_{subject:03d}.npz'
        required.update((name, name+".manifest.json"))
    require(required.issubset(files), "required training code/data bindings missing")
    require(not any("subject_007" in name or "subject_008" in name for name in files),
            "evaluation data included in training preflight")
    for name, digest in files.items():
        path = root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and path.is_file(), "unsafe/missing locked file")
        require(sha(path) == digest, "locked file changed: " + name)
    return {"status":"ready", "model_version":cfg["model_version"], "config_sha256":canonical_sha(cfg),
        "training_lock_sha256":sha(root/LOCK), "checked_files":len(files),
        "fit_subjects":cfg["data"]["fit_subjects"], "selection_subjects":cfg["data"]["selection_subjects"],
        "waveforms_loaded":False, "cuda_initialized":False, "runtime_enabled":False, "production_validated":False}


def condition_line(signal):
    import numpy as np
    from scipy.signal import iirnotch, filtfilt
    x = np.asarray(signal, dtype=np.float64)
    require(x.ndim in (2,3) and x.shape[-1] == 8 and x.shape[-2] >= 512
            and np.isfinite(x).all(), "line conditioner requires finite [..., time>=512, 8]")
    # Remove the recorded offset for numerical conditioning; restore it afterwards.
    mean = x.mean(axis=-2, keepdims=True)
    b, a = iirnotch(50.0, 30.0, fs=250)
    out = filtfilt(b, a, x-mean, axis=-2) + mean
    require(np.isfinite(out).all(), "nonfinite line-conditioned signal")
    return out


def _normalized(reference):
    import numpy as np
    value = np.asarray(reference,dtype=np.float64)
    value = value-value.mean(axis=-1,keepdims=True)
    rms = np.sqrt(np.mean(value**2,axis=-1,keepdims=True))
    require(np.isfinite(value).all() and np.all(rms > 1e-8), "flat or invalid reference; no repaired/dropped channels")
    return np.asarray(value/rms,dtype=np.float32)


def load_reference_windows(subjects, root=ROOT):
    import numpy as np
    root = Path(root); cfg = load_config(root)
    subjects = list(subjects)
    require(subjects and len(set(subjects)) == len(subjects)
            and all(type(subject) is int and 1 <= subject <= 8 for subject in subjects), "invalid subject list")
    lock = json.loads((root/LOCK).read_text(encoding="utf-8"))
    refs, identities = [], []
    for subject in subjects:
        name = f'{cfg["data"]["reference_directory"]}/subject_{subject:03d}.npz'
        path = root/name
        expected = lock["files"].get(name) or lock.get("evaluation_files", {}).get(name)
        require(expected and sha(path) == expected, "reference NPZ hash mismatch")
        manifest_path = Path(str(path)+'.manifest.json')
        expected_manifest = lock["files"].get(name+'.manifest.json') or lock.get("evaluation_files",{}).get(name+'.manifest.json')
        require(expected_manifest and sha(manifest_path)==expected_manifest, "reference manifest mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("subject_id") == subject and manifest.get("dataset") == 'Wearable'
                and manifest.get("native_sample_rate_hz") == 250
                and manifest.get("raw_reference_is_clean_ground_truth") is False, "reference provenance mismatch")
        with np.load(path,allow_pickle=False) as archive:
            original=np.asarray(archive['reference'],dtype=np.float64)
            electrode=np.asarray(archive['electrode_id'])
        require(original.shape == (240,710,8) and np.isfinite(original).all(), "reference layout/finite mismatch")
        processed=condition_line(original)[:,125:637,:].transpose(0,2,1)
        refs.append(_normalized(processed))
        identities.extend({"dataset":"Wearable", "subject_id":subject, "trial_index":index,
            "electrode_id":int(electrode[index]), "reference_npz_sha256":expected,
            "reference_definition":"line-conditioned recorded surrogate, not clean EEG"} for index in range(240))
    return np.concatenate(refs), identities


def synthetic_references(count, seed):
    import numpy as np
    require(type(count) is int and count > 0, "positive synthetic example count required")
    rng=np.random.default_rng(seed); t=np.arange(512)/250
    f=rng.uniform(8,22,size=(count,1,1))
    phases=rng.uniform(-np.pi,np.pi,size=(count,8,1))
    signal=np.sin(2*np.pi*f*t+phases)+0.25*np.sin(2*np.pi*2*f*t+phases/2)
    signal+=0.2*np.sin(2*np.pi*rng.uniform(4,7,size=(count,1,1))*t+phases)
    return _normalized(signal)


def corrupt(reference, condition, seed, config):
    import numpy as np
    x=np.asarray(reference,dtype=np.float32)
    require(x.ndim == 3 and x.shape[1:] == (8,512) and np.isfinite(x).all(), "invalid corruption reference")
    conditions=config['augmentation']['conditions']
    require(condition in conditions,"unknown corruption condition")
    if condition == 'identity':
        return x.copy()
    rng=np.random.default_rng(seed); n=len(x); t=np.arange(512)[None,None,:]/250
    amplitudes=rng.uniform(*config['augmentation']['rms_amplitude_range'],size=(n,8,1))
    active=rng.random((n,8,1))<0.7
    # Guarantee at least one corrupt channel without using any task label.
    active[:,0,:]=True
    phase=rng.uniform(-np.pi,np.pi,size=(n,8,1))
    low=np.sin(2*np.pi*rng.uniform(*config['augmentation']['drift_frequency_hz'],size=(n,1,1))*t+phase)
    high=np.sin(2*np.pi*rng.uniform(*config['augmentation']['high_frequency_hz'],size=(n,1,1))*t+phase)
    center=rng.uniform(0.15,1.85,size=(n,8,1)); width=rng.uniform(0.08,0.3,size=(n,8,1))
    high*=np.exp(-0.5*((t-center)/width)**2)
    gaussian=rng.standard_normal(x.shape)
    def centered_scaled(noise):
        noise=noise-noise.mean(axis=-1,keepdims=True)
        rms=np.sqrt(np.mean(noise**2,axis=-1,keepdims=True))
        return noise/np.maximum(rms,1e-12)*amplitudes*active
    noises={'drift':centered_scaled(low),'high_burst':centered_scaled(high),
            'gaussian':centered_scaled(gaussian)}
    noise=sum(noises.values()) if condition == 'combined' else noises[condition]
    return np.asarray(x+noise,dtype=np.float32)

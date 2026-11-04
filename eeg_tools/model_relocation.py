"""Read-only relocation of the frozen Wang2016 loader contract.

Original training modules, manifests and lock records stay byte-for-byte intact.
No pickle fallback, root search, rewritten hashes or relaxed preprocessing.
"""
from pathlib import Path
import torch
from eeg_tools.runtime_bindings import RootResolver
from scripts import train_matched_ssvep_decoder as frozen


def load_relocated_model(run_dir, device, mappings):
    run = Path(run_dir).resolve()
    b, d, require = frozen.baseline, frozen.decoder, frozen.require
    names = ("preregistration.json", "preregistration_lock.json", "selection.json",
             "selection_lock.json", "best_weights.pt", "history.json", "validation_report.json")
    require(all((run / n).is_file() and not (run / n).is_symlink() for n in names),
            "missing/unsafe selection artifact")
    p, s = b.read_json(run / names[0]), b.read_json(run / "selection.json")
    require(b.read_json(run / "preregistration_lock.json") == {"sha256": frozen.digest(p)}
            and b.read_json(run / "selection_lock.json") == {"sha256": frozen.digest(s)}
            and s.get("preregistration_sha256") == frozen.digest(p), "manifest lock/hash mismatch")
    for manifest in (p, s):
        require(all(manifest.get(k) == v for k, v in d.FLAGS.items())
                and manifest.get("scope") == frozen.SCOPE and manifest.get("subjects") == frozen.SPLITS,
                "manifest flags/splits changed")
    require(p.get("model") == d.MODEL_SPEC and p.get("preprocessing") == d.PREPROCESSING
            and p.get("gate") == d.GATE and p.get("frequencies_hz") == list(d.FREQUENCIES)
            and p.get("commands") == {str(k): v for k, v in d.COMMANDS.items()}
            and p.get("statistical_baselines") == frozen.BASELINE_SPEC, "manifest contract changed")
    frozen.check_spec(p["source_spec"])
    resolver = RootResolver(mappings)
    observed = resolver.verify(p["code_hashes"])
    current = {Path(k).resolve(): v for k, v in frozen.code_hashes().items()}
    require(observed == current, "training code identity/path set changed")
    require(s.get("selection_rule") == frozen.SELECTION_RULE and s.get("checkpoint_file") == "best_weights.pt"
            and s.get("checkpoint_format") == "state_dict_only" and s.get("final_test_evaluated") is False,
            "invalid checkpoint selection")
    require(set(s.get("artifacts", {})) == {"best_weights.pt", "history.json", "validation_report.json"},
            "selection artifacts changed")
    frozen.verify_hashes({str(run / n): h for n, h in s["artifacts"].items()})
    history = b.read_json(run / "history.json")["epochs"]
    require(len(history) == p["training"]["epochs"] and frozen.select_epoch(history) == s["selected"],
            "checkpoint is not best validation epoch")
    require(s.get("source_hashes") == p["source_hashes"], "source bindings changed")
    resolver.verify(p["source_hashes"])
    model = d.HarmonicSpectrumMLP()
    state = torch.load(run / "best_weights.pt", map_location="cpu", weights_only=True)
    expected = model.state_dict()
    require(isinstance(state, dict) and set(state) == set(expected), "checkpoint state_dict keys mismatch")
    for n, v in state.items():
        require(isinstance(v, torch.Tensor) and v.shape == expected[n].shape
                and v.dtype == expected[n].dtype and bool(torch.isfinite(v).all()), "invalid/nonfinite tensor")
    require(bool((state["feature_std"] >= 1e-6).all()), "invalid feature standardizer")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), p, s

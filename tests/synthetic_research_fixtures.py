"""Disposable contract fixtures, never historical model/evaluation evidence.

All files below are test inputs. A valid synthetic lock proves only the loader's
hash checks; the mock history does not represent training or model eligibility.
No personal signal, trained weight, or external dataset is read by this helper.
"""
from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "research_contracts"
SCOPE = "synthetic_contract_only_not_historical_training_evidence"


def research_contract(name: str) -> dict:
    if name not in {"v3-preregistration.json", "v3-dong-metadata.json",
                    "v4-preregistration.json", "v4-interface.json", "v13-plan.json", "v5-parser-texts.json"}:
        raise ValueError("unknown synthetic contract")
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    if value.get("fixture_scope") != SCOPE:
        raise ValueError("fixture must explicitly disclaim historical evidence")
    return value


def failed_v14_fixture(root: Path) -> dict:
    """Write an untrained CPU state and its real content hashes in a TEMP root.

    Uses the unmodified production preflight/lock schema; no loader or hashing
    mocks. Callers own the TemporaryDirectory and cleanup. The history merely
    exercises the failed-selection schema and cannot authorize runtime use.
    """
    import tempfile

    root = Path(root).absolute()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
           for p in (root, *root.parents)):
        raise ValueError("synthetic fixture cannot follow directory aliases")
    root = root.resolve()
    if root == temp_root or not root.is_relative_to(temp_root):
        raise ValueError("synthetic fixture must remain inside the system temporary root")
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError("synthetic fixture requires a new empty temporary directory")
    import torch
    from eeg_tools import ssvep_v14_runtime as api
    from scripts import train_ssvep_denoiser_v14 as runner

    cfg = api.load_config()
    for name in api.bindings(cfg):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == api.CONFIG:
            target.write_text(json.dumps(cfg), encoding="utf-8")
        else:
            target.write_bytes(("synthetic metadata placeholder, never training input: " + name).encode())
    runner.freeze_only(root, write_lock=True)
    state = api.preflight(root)
    common = dict(schema_version=1, model_version=cfg["model_version"],
                  config_sha256=api.canonical_sha(cfg),
                  training_lock_sha256=state["training_lock_sha256"],
                  research_plan_sha256=api.plan_sha(cfg), selected_epoch=None,
                  selection_gate_passed=False, runtime_enabled=False, production_validated=False,
                  checkpoint_role="diagnostic_last_epoch_no_eligible_checkpoint",
                  fixture_scope=SCOPE)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(14)
        model = api.build_model(cfg).cpu()
    checkpoint = {"config": cfg, "selection_gate_passed": False,
                  "state_dict": model.state_dict(), "fixture_scope": SCOPE}
    model_path = root / api.MODEL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    report = {**common, "status": "training_completed", "epochs": 16,
              "history": [{"epoch": i, "selection": {"selection_gate_passed": False},
                           "fixture_scope": SCOPE} for i in range(1, 17)]}
    (root / api.REPORT).write_text(json.dumps(report), encoding="utf-8")
    manifest = {**common, "checkpoint_sha256": api.sha(model_path),
                "training_report_sha256": api.sha(root / api.REPORT)}
    (root / api.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return common


def embedded_preregistration_bytes(module) -> bytes:
    """Use already-shipped frozen text ONLY if its original digest still matches.

    This reconstructs the literal's surrounding newline, never a new hash or
    replacement historical lock. A changed template fails instead of blessing it.
    """
    import hashlib
    raw = (module._EXPECTED_PREREG_JSON.strip() + "\n").encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != module.PREREG_SHA256:
        raise ValueError("embedded preregistration does not match original frozen SHA256")
    return raw

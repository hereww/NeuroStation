[CmdletBinding()]
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project .venv missing: $Python. Prepare its dependencies first."
}
Push-Location -LiteralPath $ProjectRoot
try {
    & $Python (Join-Path $ProjectRoot 'scripts\validate_release_configs.py')
    if ($LASTEXITCODE -ne 0) { throw "Formal configuration validation failed (exit $LASTEXITCODE)." }
    $DependencyCheck = @'
import json
from pathlib import Path
import numpy
import scipy
from brainflow.board_shim import BoardShim
from PySide6 import QtCore
from eeg_tools.learned_denoise import load_config, load_model

root = Path.cwd()
config = load_config(root / "configs" / "learned_denoiser_v1.json")
if config["inference"].get("mode") != "shadow":
    raise SystemExit("Launcher only supports shadow inference.")
runtime = config.get("runtime", {})
enabled = runtime.get("enabled", False)
if not isinstance(enabled, bool):
    raise SystemExit("runtime.enabled must be a boolean.")
validation_status = runtime.get("validation_status", "not_validated")
model_validation = "failed" if str(validation_status).startswith("failed") else (
    "passed" if validation_status == "passed" else "not_validated"
)
# A readable checkpoint and formal pipeline config never imply model efficacy.
# Do not let flipping only enabled bypass the recorded failed validation.
if enabled and validation_status != "passed":
    raise SystemExit(f"Automatic shadow refused: model_validation={model_validation}, validation_status={validation_status}.")
model_path = Path(runtime.get("model_path", "models/denoiser_v1.pt"))
if not model_path.is_absolute():
    model_path = root / model_path
model_available = model_path.is_file()
model_load_status = "skipped_automatic_shadow_disabled"
torch_version = None
if enabled:
    if not model_available:
        raise SystemExit(f"Automatic shadow refused: model missing: {model_path}")
    import torch
    torch.set_num_threads(1)
    load_model(model_path, config, device="cpu")
    model_load_status = "loaded_for_validated_shadow"
    torch_version = torch.__version__
# Disabled automatic shadow does not require PyTorch or a model. The ordinary
# GUI remains usable; no checkpoint is loaded and no hardware is opened here.
print(json.dumps({
    "status": "pipeline_ready", "pipeline_ready": True,
    "configuration_status": "formal", "inference_mode": "shadow",
    "execution": "post_acquisition_async" if enabled else "manual_gui_only",
    "automatic_shadow_disabled": not enabled, "shadow_enabled": enabled,
    "model_validation": model_validation, "validation_status": validation_status,
    "validation_reason": runtime.get("validation_reason"),
    "validation_reports": runtime.get("validation_reports", {}),
    "model_available": model_available, "model_load_status": model_load_status,
    "torch_version": torch_version, "model_path": str(model_path.resolve()),
    "hardware_opened": False, "visual_stimulus_started": False,
    "notice": "Pipeline readiness is not model efficacy or production acceptance.",
}, ensure_ascii=True, indent=2))
'@
    & $Python -c $DependencyCheck
    if ($LASTEXITCODE -ne 0) { throw "Dependency/model check failed (exit $LASTEXITCODE). Use project .venv." }
    if ($CheckOnly) {
        Write-Host 'CheckOnly complete: pipeline_ready is not model approval; see automatic_shadow_disabled and model_validation above. No GUI, hardware, acquisition or visual stimulus started.'
        return
    }
    # The operator must still initiate acquisition and confirm all safety checks.
    & $Python (Join-Path $ProjectRoot 'workstation.py')
    if ($LASTEXITCODE -ne 0) { throw "Workstation exited with code $LASTEXITCODE." }
}
finally { Pop-Location }

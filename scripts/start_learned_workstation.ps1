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
import torch
from eeg_tools.learned_denoise import load_config, load_model

root = Path.cwd()
config = load_config(root / "configs" / "learned_denoiser_v1.json")
if config["inference"].get("mode") != "shadow":
    raise SystemExit("Launcher only supports shadow inference.")
model_path = Path(config.get("runtime", {}).get("model_path", "models/denoiser_v1.pt"))
if not model_path.is_absolute():
    model_path = root / model_path
if not model_path.is_file():
    raise SystemExit(f"Model missing: {model_path}")
# Load only the configured trusted local checkpoint. Never download or train.
torch.set_num_threads(1)
load_model(model_path, config, device="cpu")
print(json.dumps({
    "status": "ready", "configuration_status": "formal", "inference_mode": "shadow",
    "execution": "post_acquisition_async", "torch_version": torch.__version__,
    "model_path": str(model_path.resolve()),
    "shadow_enabled": config.get("runtime", {}).get("enabled", True),
    "hardware_opened": False, "visual_stimulus_started": False,
}, ensure_ascii=True, indent=2))
'@
    & $Python -c $DependencyCheck
    if ($LASTEXITCODE -ne 0) { throw "Dependency/model check failed (exit $LASTEXITCODE). Use project .venv." }
    if ($CheckOnly) {
        Write-Host 'CheckOnly complete. No GUI, acquisition or visual stimulus started.'
        return
    }
    # The operator must still initiate acquisition and confirm all safety checks.
    & $Python (Join-Path $ProjectRoot 'workstation.py')
    if ($LASTEXITCODE -ne 0) { throw "Workstation exited with code $LASTEXITCODE." }
}
finally { Pop-Location }

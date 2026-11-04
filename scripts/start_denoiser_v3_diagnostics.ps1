# Explicit v3 offline diagnostics. Never GUI, hardware, acquisition or stimulus.
[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [string]$SessionDir,
    [string]$InputNPZ,
    [string]$OutputDir,
    [ValidateNotNullOrEmpty()][string]$Checkpoint = 'models/domain_gate_v3.pt',
    [ValidateNotNullOrEmpty()][string]$DenoiserCheckpoint = 'models/denoiser_v2.pt',
    [ValidateNotNullOrEmpty()][string]$Config = 'configs/learned_denoiser_v3.json',
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cpu',
    [ValidateRange(1, 2147483647)][int]$Threads = 2,
    [string]$Root,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$ProjectRoot = if ($PSBoundParameters.ContainsKey('Root')) {
    [System.IO.Path]::GetFullPath($Root)
} else { [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')) }
$GpuPython = Join-Path $ProjectRoot '.venv-ml-gpu\Scripts\python.exe'
$CpuPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Python = if ($PSBoundParameters.ContainsKey('PythonPath')) {
    [System.IO.Path]::GetFullPath($PythonPath)
} elseif (Test-Path -LiteralPath $GpuPython -PathType Leaf) { $GpuPython } else { $CpuPython }
$RuntimeSelection = if ($PSBoundParameters.ContainsKey('PythonPath')) { 'operator_explicit_interpreter' }
    elseif ($Python -eq $GpuPython) { 'preferred_venv_ml_gpu' } else { 'fallback_venv' }
$HasSession = $PSBoundParameters.ContainsKey('SessionDir')
$HasInput = $PSBoundParameters.ContainsKey('InputNPZ')
$HasOutput = $PSBoundParameters.ContainsKey('OutputDir')

function Write-LauncherFailure([string]$Message, [string]$FormalStatus = 'not_checked') {
    [ordered]@{
        status = 'error'; diagnostic_ready = $false; error = $Message
        project_root = $ProjectRoot; python_path = $Python; runtime_selection = $RuntimeSelection
        formal_config_status = $FormalStatus; cuda_required = ($Device -eq 'cuda')
        diagnostic_only = $true; automatic_processing_enabled = $false
        release_gate_passed = $false; production_validated = $false
        gui_started = $false; hardware_opened = $false; acquisition_started = $false
        visual_stimulus_started = $false; test_evaluation_executed = $false
        raw_is_immutable = $true; original_quality_unchanged = $true
        notice = 'Diagnostic readiness is NOT production approval or repair quality.'
        dependency_guidance = 'Prepare numpy/scipy/torch manually in the selected runtime. No install or silent runtime/device fallback is performed.'
    } | ConvertTo-Json -Depth 4
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-LauncherFailure "Selected interpreter missing: $Python. No PATH interpreter fallback."
    exit 1
}
if ($HasSession -and $HasInput) {
    Write-LauncherFailure 'Choose exactly one explicit SessionDir or InputNPZ.'
    exit 1
}
if (($HasSession -and [string]::IsNullOrWhiteSpace($SessionDir)) -or
    ($HasInput -and [string]::IsNullOrWhiteSpace($InputNPZ)) -or
    ($HasOutput -and [string]::IsNullOrWhiteSpace($OutputDir))) {
    Write-LauncherFailure 'Explicit source/output arguments must not be blank.'
    exit 1
}

Push-Location -LiteralPath $ProjectRoot
try {
    # Every entry first checks formal acquisition resources, NOT model efficacy.
    & $Python (Join-Path $ProjectRoot 'scripts\validate_release_configs.py')
    if ($LASTEXITCODE -ne 0) {
        Write-LauncherFailure 'Formal release configuration preflight failed; diagnostics refused.' 'invalid_release'
        exit 1
    }
    $Preflight = @'
import argparse
import importlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description="Read-only v3 weights and runtime preflight")
parser.add_argument("--root", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--denoiser-checkpoint", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
parser.add_argument("--threads", type=int, required=True)
parser.add_argument("--runtime-selection", required=True)
parser.add_argument("--session")
parser.add_argument("--input")
parser.add_argument("--output")
parser.add_argument("--check-only", action="store_true")
args = parser.parse_args()
root = Path(args.root).resolve()
sys.path.insert(0, str(root))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
report = {
    "status": "error", "diagnostic_ready": False, "project_root": str(root),
    "python_path": sys.executable, "runtime_selection": args.runtime_selection,
    "device": args.device, "threads": args.threads,
    "formal_config_status": "formal_release_preflight_passed",
    "model_load_status": "not_checked", "cuda_available": None, "cuda_required": args.device == "cuda",
    "diagnostic_only": True, "automatic_processing_enabled": False, "release_gate_passed": False,
    "production_validated": False, "raw_is_immutable": True, "original_quality_unchanged": True,
    "gui_started": False, "hardware_opened": False, "acquisition_started": False,
    "visual_stimulus_started": False, "test_evaluation_executed": False,
    "support_score": "uncalibrated applicability, NOT artifact or repair quality",
    "checkpoint_loading": "v3 and frozen v2 weights_only=True; actual strict weight loading, not exists-only",
    "execution": "check_only" if args.check_only else ("explicit_offline" if args.session or args.input else "readiness_only"),
    "notice": "Diagnostic readiness is NOT production approval or repair quality.",
    "dependency_guidance": "Selected runtime may differ from ordinary Qt .venv; prepare numpy/scipy/torch there manually. CUDA optional for CPU diagnostics, mandatory when requested. No installation or silent CPU/runtime fallback.",
    "errors": [],
}

def resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()

modules = {}
for name in ("numpy", "scipy", "torch"):
    try:
        modules[name] = importlib.import_module(name)
    except Exception as error:
        report["errors"].append("Dependency " + name + " unavailable in selected runtime: " + str(error))
report["dependencies"] = {name: name in modules for name in ("numpy", "scipy", "torch")}
if "torch" in modules:
    report["torch_version"] = str(modules["torch"].__version__)
    report["cuda_available"] = bool(modules["torch"].cuda.is_available())
    if args.device == "cuda" and not report["cuda_available"]:
        report["errors"].append("CUDA requested but unavailable; no CPU fallback")
try:
    checkpoint, denoiser, config = (resolved(value) for value in (args.checkpoint, args.denoiser_checkpoint, args.config))
    report.update(checkpoint_path=str(checkpoint), denoiser_checkpoint_path=str(denoiser), config_path=str(config))
    if not all(path.is_file() for path in (checkpoint, denoiser, config)):
        raise ValueError("v3 checkpoint, bound v2 checkpoint and config must all exist")
    if args.session and args.input:
        raise ValueError("one explicit source only")
    source_value = args.input or args.session
    if source_value is None:
        if args.output:
            raise ValueError("OutputDir requires explicit SessionDir or InputNPZ")
    else:
        source = resolved(source_value)
        if (args.session and not source.is_dir()) or (args.input and (not source.is_file() or source.suffix.lower() != ".npz")):
            raise ValueError("explicit source missing/wrong type")
        if not args.output:
            raise ValueError("explicit source requires new independent OutputDir")
        output = resolved(args.output)
        if source.is_dir() and (output == source or source in output.parents):
            raise ValueError("OutputDir must stay outside source session")
        if output.exists() or output.is_symlink():
            raise ValueError("OutputDir already exists; overwrite refused")
        for protected in (source, checkpoint, denoiser, config):
            if output == protected or output in protected.parents:
                raise ValueError("OutputDir cannot contain a protected source/checkpoint/config")
        report["source_path"], report["output_dir"] = str(source), str(output)
        report["input_kind"] = "npz" if args.input else "session"
    if not report["errors"]:
        from scripts.run_learned_denoise_v3 import load_bundle
        core, bundle = load_bundle(checkpoint, denoiser, config, args.device, args.threads)
        report["support_threshold"] = bundle.support_threshold
        report["runtime"] = bundle.config["runtime"]
        report["model_load_status"] = "loaded_v3_and_bound_v2_weights_only"
except Exception as error:
    report["model_load_status"] = "invalid_or_missing_checkpoint_source_config"
    report["errors"].append(str(error))
report["diagnostic_ready"] = not report["errors"] and report["model_load_status"] == "loaded_v3_and_bound_v2_weights_only"
if report["diagnostic_ready"]:
    report["status"] = "diagnostic_ready"
else:
    report["error"] = "; ".join(report["errors"])
print(json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2))
raise SystemExit(0 if report["diagnostic_ready"] else 1)
'@
    $Arguments = @('-c', $Preflight, '--root', $ProjectRoot, '--checkpoint', $Checkpoint,
        '--denoiser-checkpoint', $DenoiserCheckpoint, '--config', $Config,
        '--device', $Device, '--threads', $Threads.ToString(), '--runtime-selection', $RuntimeSelection)
    if ($HasSession) { $Arguments += @('--session', $SessionDir) }
    if ($HasInput) { $Arguments += @('--input', $InputNPZ) }
    if ($HasOutput) { $Arguments += @('--output', $OutputDir) }
    if ($CheckOnly) { $Arguments += '--check-only' }
    $Lines = & $Python @Arguments
    $ExitCode = $LASTEXITCODE
    $Text = $Lines -join [Environment]::NewLine
    Write-Output $Text
    if ($ExitCode -ne 0) { exit $ExitCode }
    if ($CheckOnly -or (-not $HasSession -and -not $HasInput)) {
        Write-Host 'Readiness only: no source inference, GUI, hardware, acquisition or stimulus. NOT production approval.'
        exit 0
    }
    $Ready = $Text | ConvertFrom-Json
    $Run = @((Join-Path $ProjectRoot 'scripts\run_learned_denoise_v3.py'), '--root', $ProjectRoot,
        '--checkpoint', $Ready.checkpoint_path, '--denoiser-checkpoint', $Ready.denoiser_checkpoint_path,
        '--config', $Ready.config_path, '--device', $Device, '--threads', $Threads.ToString(), '--output', $Ready.output_dir)
    if ($HasInput) { $Run += @('--input', $Ready.source_path) } else { $Run += @('--session', $Ready.source_path) }
    & $Python @Run
    exit $LASTEXITCODE
}
catch {
    Write-LauncherFailure ('Launcher failed: ' + $_.Exception.Message)
    exit 1
}
finally { Pop-Location }

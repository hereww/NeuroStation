# Formal local entry point: explicit offline diagnostics, never automatic deployment.
[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [string]$SessionDir,
    [string]$OutputDir,
    [ValidateNotNullOrEmpty()][string]$Model = 'models/denoiser_v2.pt',
    [ValidateNotNullOrEmpty()][string]$Config = 'configs/learned_denoiser_v2.json',
    [ValidateRange(1, 2147483647)][int]$Threads = 2
)

$ErrorActionPreference = 'Stop'
# Inspect native exit codes explicitly even when the caller enables this PS7 option.
$PSNativeCommandUseErrorActionPreference = $false
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$HasSession = $PSBoundParameters.ContainsKey('SessionDir')
$HasOutput = $PSBoundParameters.ContainsKey('OutputDir')

function Write-LauncherFailure([string]$Message, [string]$FormalStatus = 'not_checked') {
    [ordered]@{
        status = 'error'; diagnostic_ready = $false; error = $Message
        project_root = $ProjectRoot; python_path = $Python
        model_available = $false; model_load_status = 'not_checked'
        formal_config_status = $FormalStatus; validation_status = $null
        diagnostic_only = $true; automatic_processing_enabled = $false
        release_gate_passed = $false; production_validated = $false
        gui_started = $false; hardware_opened = $false
        acquisition_started = $false; visual_stimulus_started = $false
        notice = 'Diagnostic readiness is NOT production approval.'
    } | ConvertTo-Json -Depth 4
}

function Show-ManualUsage {
    Write-Host 'Readiness only. No session processing, GUI, hardware, acquisition or stimulus started.'
    Write-Host 'Relative paths resolve against the project root, not the caller working directory.'
    $QuotedLauncher = (Join-Path $ProjectRoot 'scripts\start_denoiser_v2_diagnostics.ps1').Replace("'", "''")
    Write-Host "  & '$QuotedLauncher' -CheckOnly"
    Write-Host "  & '$QuotedLauncher' -SessionDir '<existing-session>' -OutputDir '<new-directory-outside-session>' -Threads 2"
    Write-Host 'No training, downloads, dependency installation, configuration edits or test evaluation are performed.'
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-LauncherFailure "Project .venv interpreter missing: $Python. Prepare dependencies manually; PATH Python is not used."
    if (-not $HasSession) { Show-ManualUsage }
    exit 1
}

Push-Location -LiteralPath $ProjectRoot
try {
    # This gate validates formal acquisition resources only, NOT model efficacy.
    $ReleaseArguments = @((Join-Path $ProjectRoot 'scripts\validate_release_configs.py'))
    & $Python @ReleaseArguments
    if ($LASTEXITCODE -ne 0) {
        Write-LauncherFailure 'Formal release configuration validation failed; diagnostics refused.' 'invalid_release'
        exit 1
    }
    if (($HasSession -and [string]::IsNullOrWhiteSpace($SessionDir)) -or
        ($HasOutput -and [string]::IsNullOrWhiteSpace($OutputDir))) {
        Write-LauncherFailure 'Explicit SessionDir/OutputDir must not be blank.'
        exit 1
    }

    $Preflight = @'
import argparse
import importlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Read-only v2 diagnostic preflight")
parser.add_argument("--root", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--threads", type=int, required=True)
parser.add_argument("--session")
parser.add_argument("--output-dir")
parser.add_argument("--check-only", action="store_true")
args = parser.parse_args()
root = Path(args.root).resolve()
report = {
    "status": "error", "diagnostic_ready": False,
    "project_root": str(root), "model_available": False,
    "model_load_status": "not_checked", "formal_config_status": "not_checked",
    "release_config_status": "formal", "validation_status": None, "runtime": None,
    "diagnostic_only": True, "automatic_processing_enabled": False,
    "release_gate_passed": False, "production_validated": False,
    "gui_started": False, "hardware_opened": False,
    "acquisition_started": False, "visual_stimulus_started": False,
    "raw_is_immutable": True, "original_quality_unchanged": True,
    "test_evaluation_executed": False,
    "checkpoint_loading": "weights_only=True; strict v2 metadata and state",
    "threads": args.threads,
    "execution": "check_only" if args.check_only else (
        "explicit_offline_session" if args.session is not None else "readiness_only"),
    "output_validation_status": "not_requested", "errors": [],
    "notice": "Diagnostic readiness is NOT production approval.",
}


def absolute_path(value):
    if not value or not value.strip():
        raise ValueError("paths must not be blank")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


# Inspect only the existing saved summary, never launch evaluation or enable runtime.
summary_path = root / "analysis_outputs" / "learned_denoiser_v2_test.json"
if summary_path.is_file():
    summary = {"path": str(summary_path), "notice": "NOT production approval",
               "provenance": "existing_saved_research_report_only; not recomputed",
               "applies_to_selected_model": "not_assessed"}
    try:
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("saved research report must be an object")
        gate = saved.get("gate") if isinstance(saved.get("gate"), dict) else {}
        summary["status"] = str(saved.get("status", "unknown"))[:120]
        summary["scope"] = str(gate.get("scope", "not_assessed"))[:120]
        passed = saved.get("semi_synthetic_research_gate_passed",
                           gate.get("semi_synthetic_research_gate_passed"))
        summary["semi_synthetic_research_gate_passed"] = passed if type(passed) is bool else None
        count = saved.get("test_window_count")
        summary["test_window_count"] = count if type(count) is int and count >= 0 else None
    except (OSError, ValueError, UnicodeError) as error:
        summary["status"] = "unreadable"
        summary["warning"] = str(error)
    report["research_acceptance_summary"] = summary

try:
    model_path, config_path = absolute_path(args.model), absolute_path(args.config)
    report.update(model_path=str(model_path), config_path=str(config_path),
                  model_available=model_path.is_file())
    supplied = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("v2 configuration must be a JSON object")
    runtime = supplied.get("runtime")
    report["runtime"] = runtime
    report["validation_status"] = runtime.get("validation_status") if isinstance(runtime, dict) else None
    if supplied.get("status") != "formal":
        raise ValueError("selected v2 config must explicitly have status=formal")
    if (not isinstance(runtime, dict) or runtime.get("enabled") is not False
            or runtime.get("production_validated") is not False
            or runtime.get("deployment_scope") != "explicit_offline_diagnostics_only"
            or "validation_status" not in runtime):
        raise ValueError("v2 runtime must explicitly remain disabled, non-production and offline-only")
except (OSError, ValueError, UnicodeError) as error:
    report["formal_config_status"] = "invalid"
    report["errors"].append("Configuration/path error: " + str(error))

# Reject output aliases, descendants, ancestors and existing destinations before inference.
try:
    if args.session is None:
        if args.output_dir is not None:
            raise ValueError("OutputDir requires explicit SessionDir")
    else:
        report["output_validation_status"] = "invalid"
        session = absolute_path(args.session)
        report["session_path"] = str(session)
        if not session.is_dir():
            raise ValueError("SessionDir must be an existing directory")
        if args.output_dir is None:
            raise ValueError("explicit SessionDir requires OutputDir: select a new directory outside the session")
        output = absolute_path(args.output_dir)
        report["output_dir"] = str(output)
        if output == session or session in output.parents:
            raise ValueError("OutputDir must be outside the source session, not equal to or below it")
        for protected in (session, absolute_path(args.model), absolute_path(args.config)):
            if output == protected or output in protected.parents:
                raise ValueError("OutputDir cannot contain the protected session, model or config")
        # is_symlink also refuses a broken link, which exists() alone would miss.
        if output.exists() or output.is_symlink():
            raise ValueError("OutputDir already exists; choose a new independent directory")
        report["output_validation_status"] = "valid_new_external_directory"
except (OSError, ValueError, RuntimeError) as error:
    report["output_validation_status"] = "invalid"
    report["errors"].append("Session/output error: " + str(error))

modules = {}
report["dependencies"] = {}
for name in ("numpy", "scipy", "torch"):
    try:
        modules[name] = importlib.import_module(name)
        report["dependencies"][name] = "available"
    except Exception as error:
        report["dependencies"][name] = "unavailable"
        report["errors"].append("Missing/unusable dependency " + name + ": " + str(error))
report["torch_version"] = str(modules["torch"].__version__) if "torch" in modules else None
if not report["model_available"]:
    report["model_load_status"] = "missing_checkpoint"
    report["errors"].append("Model checkpoint missing; diagnostic readiness refused: " + args.model)
elif len(modules) != 3:
    report["model_load_status"] = "dependency_unavailable"

# The core loader enforces weights_only=True with no unsafe fallback.
if len(modules) == 3 and report["formal_config_status"] != "invalid":
    try:
        from eeg_tools.learned_denoise_v2 import load_config, load_model
        config = load_config(config_path)
        report["formal_config_status"] = "formal"
    except Exception as error:
        report["formal_config_status"] = "invalid"
        report["errors"].append("Strict v2 configuration validation failed: " + str(error))
    if not report["errors"]:
        try:
            modules["torch"].set_num_threads(args.threads)
            load_model(model_path, config, device="cpu")
            report["model_load_status"] = "loaded_weights_only"
        except Exception as error:
            report["model_load_status"] = "invalid_checkpoint"
            report["errors"].append("Strict v2 checkpoint loading failed: " + str(error))

report["diagnostic_ready"] = not report["errors"] and report["model_load_status"] == "loaded_weights_only"
if report["diagnostic_ready"]:
    report["status"] = "diagnostic_ready"
else:
    report["error"] = "; ".join(report["errors"])
print(json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2))
raise SystemExit(0 if report["diagnostic_ready"] else 1)
'@
    $PreflightArguments = @('-c', $Preflight, '--root', $ProjectRoot,
        '--model', $Model, '--config', $Config, '--threads', $Threads.ToString())
    if ($HasSession) { $PreflightArguments += @('--session', $SessionDir) }
    if ($HasOutput) { $PreflightArguments += @('--output-dir', $OutputDir) }
    if ($CheckOnly) { $PreflightArguments += '--check-only' }
    $PreflightLines = & $Python @PreflightArguments
    $PreflightExitCode = $LASTEXITCODE
    $PreflightText = $PreflightLines -join [Environment]::NewLine
    Write-Output $PreflightText
    if (-not $HasSession) { Show-ManualUsage }
    if ($PreflightExitCode -ne 0) { exit $PreflightExitCode }
    if ($CheckOnly) {
        Write-Host 'CheckOnly complete: no processing performed. NOT production approval.'
        exit 0
    }
    if (-not $HasSession) { exit 0 }

    $Readiness = $PreflightText | ConvertFrom-Json
    # Only explicit operator-selected sessions reach this offline runner.
    $RunArguments = @((Join-Path $ProjectRoot 'scripts\run_learned_denoise_v2.py'),
        '--session', $Readiness.session_path, '--model', $Readiness.model_path,
        '--config', $Readiness.config_path, '--threads', $Threads.ToString(),
        '--output-dir', $Readiness.output_dir)
    & $Python @RunArguments
    exit $LASTEXITCODE
}
catch {
    Write-LauncherFailure ("Launcher failed: " + $_.Exception.Message)
    exit 1
}
finally { Pop-Location }

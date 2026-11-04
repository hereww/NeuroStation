[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [string]$DatasetRoot = (Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'NeuroStation\Datasets'),
    [ValidateSet('zh-CN', 'en-US')][string]$Language = 'zh-CN'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$GuiPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$MlPython = Join-Path $ProjectRoot '.venv-ml-gpu\Scripts\python.exe'
$ReleaseValidator = Join-Path $ProjectRoot 'scripts\validate_release_configs.py'
$ModelChecker = Join-Path $ProjectRoot 'scripts\run_ssvep_test_model.py'
$Workstation = Join-Path $ProjectRoot 'workstation.py'
# Resolve relative overrides before changing the working directory.
$ResolvedDatasetRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DatasetRoot)

if (-not (Test-Path -LiteralPath $GuiPython -PathType Leaf)) {
    throw "GUI environment Python is missing: $GuiPython. Prepare the existing environment before launching."
}
if (-not (Test-Path -LiteralPath $ReleaseValidator -PathType Leaf)) {
    throw "Formal configuration validator is missing: $ReleaseValidator"
}

Push-Location -LiteralPath $ProjectRoot
try {
    & $GuiPython $ReleaseValidator
    if ($LASTEXITCODE -ne 0) {
        throw "Formal configuration validation failed (exit $LASTEXITCODE)."
    }

    $DependencyCheck = @'
import json
import numpy
import scipy
from brainflow.board_shim import BoardShim
from PySide6 import QtCore

print(json.dumps({
    "gui_dependencies": "ready",
    "hardware_opened": False,
    "gui_started": False,
}, ensure_ascii=True))
'@
    & $GuiPython -c $DependencyCheck
    if ($LASTEXITCODE -ne 0) {
        throw "GUI dependency import check failed (exit $LASTEXITCODE). PySide6, brainflow, numpy and scipy must be available in $GuiPython."
    }

    $RequiredOfflineSources = @(
        'apps\workstation_ui\v9_offline.py',
        'eeg_tools\workstation\v9_offline.py',
        'scripts\run_v9_offline_analysis.py',
        'scripts\diagnose_neurostation_v9_output.py'
    )
    $IncompleteSources = @()
    foreach ($RelativeSource in $RequiredOfflineSources) {
        $SourcePath = Join-Path $ProjectRoot $RelativeSource
        if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
            $IncompleteSources += "Missing: $SourcePath"
        }
        elseif ((Get-Item -LiteralPath $SourcePath).Length -eq 0) {
            $IncompleteSources += "Empty: $SourcePath"
        }
    }
    if ($IncompleteSources.Count -gt 0) {
        throw ("Offline v9 source completeness check failed. " + ($IncompleteSources -join '; '))
    }

    if (-not (Test-Path -LiteralPath $MlPython -PathType Leaf)) {
        throw "ML environment Python is missing: $MlPython"
    }
    if (-not (Test-Path -LiteralPath $ModelChecker -PathType Leaf)) {
        throw "Frozen v9 preflight entry point is missing: $ModelChecker"
    }
    # No input/device options: this checks frozen evidence without loading waveforms,
    # deserializing the checkpoint, initializing CUDA or running inference.
    & $MlPython $ModelChecker --check-only
    if ($LASTEXITCODE -ne 0) {
        throw "Frozen v9 evidence preflight failed (exit $LASTEXITCODE)."
    }

    if (-not (Test-Path -LiteralPath $ResolvedDatasetRoot -PathType Container)) {
        throw "Dataset root must be an existing directory: $ResolvedDatasetRoot"
    }
    $ResolvedDatasetRoot = (Resolve-Path -LiteralPath $ResolvedDatasetRoot).ProviderPath
    if (-not (Test-Path -LiteralPath $Workstation -PathType Leaf)) {
        throw "Source workstation entry point is missing: $Workstation"
    }

    if ($CheckOnly) {
        [ordered]@{
            readiness = 'ready'
            check_only = $true
            project_root = $ProjectRoot
            dataset_root = $ResolvedDatasetRoot
            gui_python = $GuiPython
            ml_python = $MlPython
            language = $Language
            gui_started = $false
            inference_executed = $false
            cuda_initialized = $false
            hardware_opened = $false
            source_workstation = $Workstation
            executable_rebuilt = $false
        } | ConvertTo-Json
        return
    }

    Write-Output 'Starting the source workstation. Select CPU/CUDA explicitly in the v9 offline UI.'
    & $GuiPython $Workstation --language $Language --dataset-root $ResolvedDatasetRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Source workstation exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

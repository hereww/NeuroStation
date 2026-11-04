[CmdletBinding()]
param([switch]$CheckOnly)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$UiPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ModelPython = Join-Path $ProjectRoot '.venv-ml-gpu\Scripts\python.exe'
foreach ($Executable in @($UiPython, $ModelPython)) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Required Python environment missing: $Executable" }
}
Push-Location -LiteralPath $ProjectRoot
try {
    & $UiPython (Join-Path $ProjectRoot 'scripts\validate_release_configs.py')
    if ($LASTEXITCODE -ne 0) { throw 'Formal configuration validation failed.' }
    $DependencyCheck = @"
from PySide6 import QtCore, QtOpenGL
import numpy
from brainflow.board_shim import BoardShim
from eeg_tools.realtime_capture import load_channel_config
from pathlib import Path
load_channel_config(Path('configs/channel_config_v1_auto.json'))
print('UI and capture dependencies ready; hardware_opened=false')
"@
    & $UiPython -c $DependencyCheck
    if ($LASTEXITCODE -ne 0) { throw 'UI/capture dependencies unavailable.' }
    $ModelCheck = @"
import json
from eeg_tools.realtime_command_inference import RealtimeCommandInference
engine = RealtimeCommandInference(device='cpu')
print(json.dumps({'status': 'ready', 'model': engine.ready(), 'hardware_opened': False, 'visual_stimulus_started': False}, ensure_ascii=True))
"@
    & $ModelPython -c $ModelCheck
    if ($LASTEXITCODE -ne 0) { throw 'Locked model integrity/load check failed.' }
    if ($CheckOnly) {
        Write-Host 'CheckOnly passed. No GUI, hardware connection, recording or flicker started. Personal model validity remains unverified.'
        return
    }
    & $UiPython (Join-Path $ProjectRoot 'scripts\run_realtime_commands.py')
    if ($LASTEXITCODE -ne 0) { throw "Realtime application exited with code $LASTEXITCODE." }
}
finally { Pop-Location }
param([Parameter(Mandatory=$true)][string]$Python, [string]$Tools = (Join-Path $PSScriptRoot '..\.quality-local\build-tools'))
$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$output=Join-Path $root "quality-output\build-$stamp"
New-Item -ItemType Directory -Path $output | Out-Null
$env:PYTHONPATH=(Resolve-Path -LiteralPath $Tools).Path
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
Push-Location -LiteralPath $root
try {
    & $Python -B scripts/validate_release_configs.py
    if ($LASTEXITCODE -ne 0) { throw 'Formal configuration validation failed' }
    $args=@('-B','-m','PyInstaller','--noconfirm','--onedir','--console','--name','NeuroStationQualityUpgrade',
        '--distpath',"$output\dist",'--workpath',"$output\work",'--specpath',"$output",
        '--paths',$root,'--exclude-module','torch','--exclude-module','matplotlib','--exclude-module','pandas',
        '--exclude-module','pytest','--exclude-module','tkinter','--exclude-module','IPython',
        '--collect-all','brainflow','--collect-data','scipy',
        '--add-data',"$root/configs;configs",'--add-data',"$root/assets;assets",
        '--add-data',"$root/apps/workstation_ui/locales;apps/workstation_ui/locales",
        '--add-data',"$root/integrations/openbci_gui/upstream.lock.json;integrations/openbci_gui",
        '--add-data',"$root/scripts;scripts",'--add-data',"$root/eeg_tools;eeg_tools",
        '--add-data',"$root/neurostation_contract.py;.",'--add-data',"$root/neurostation_signal.py;.",'workstation.py')
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw 'Isolated Windows build failed' }
    $artifact=Join-Path $output 'dist\NeuroStationQualityUpgrade'
    $env:QT_QPA_PLATFORM='offscreen'
    $env:NEUROSTATION_DATASETS=Join-Path $output 'smoke-data'
    $env:NEUROSTATION_SETTINGS=Join-Path $output 'smoke-settings'
    & (Join-Path $artifact 'NeuroStationQualityUpgrade.exe') --smoke-test
    if ($LASTEXITCODE -ne 0) { throw 'Independent executable smoke test failed' }
    & (Join-Path $artifact 'NeuroStationQualityUpgrade.exe') --diagnostics | Set-Content -Encoding utf8 (Join-Path $output 'diagnostics.json')
    if ($LASTEXITCODE -ne 0) { throw 'Independent executable diagnostics failed' }
    Write-Output "VERIFIED_ARTIFACT=$artifact"
} finally { Pop-Location }

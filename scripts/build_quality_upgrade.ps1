param([Parameter(Mandatory=$true)][string]$Python, [string]$Tools = (Join-Path $PSScriptRoot '..\.quality-local\build-tools'))
$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$output=Join-Path $root "quality-output\build-$stamp"
New-Item -ItemType Directory -Path $output | Out-Null
$env:PYTHONPATH=(Resolve-Path -LiteralPath $Tools).Path
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$basePython = & $Python -B -c 'import sys; print(sys.base_prefix)'
$env:PATH = @((Split-Path -Parent $Python), $basePython, "$env:SystemRoot\System32", $env:SystemRoot) -join ';'
# Exclude unrelated toolchain DLLs (e.g. Poppler ICU) from dependency resolution.
Push-Location -LiteralPath $root
try {
    & $Python -B scripts/validate_release_configs.py
    if ($LASTEXITCODE -ne 0) { throw 'Formal configuration validation failed' }
    $args=@('-B','-m','PyInstaller','--noconfirm','--onedir','--console','--name','NeuroStationQualityUpgrade',
        '--distpath',"$output\dist",'--workpath',"$output\work",'--specpath',"$output",
        '--paths',$root,'--exclude-module','torch','--exclude-module','matplotlib','--exclude-module','pandas',
        '--exclude-module','pytest','--exclude-module','tkinter','--exclude-module','IPython',
        '--hidden-import','eeg_tools.quality.preview','--collect-all','brainflow','--collect-data','scipy',
        '--add-data',"$root/configs;configs",'--add-data',"$root/assets;assets",
        '--add-data',"$root/apps/workstation_ui/locales;apps/workstation_ui/locales",
        '--add-data',"$root/integrations/openbci_gui/upstream.lock.json;integrations/openbci_gui",
        '--add-data',"$root/scripts;scripts",'--add-data',"$root/eeg_tools;eeg_tools",
        '--add-data',"$root/neurostation_contract.py;.",'--add-data',"$root/neurostation_signal.py;.",'workstation.py')
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw 'Isolated Windows build failed' }
    $artifact=Join-Path $output 'dist\NeuroStationQualityUpgrade'
    foreach ($entry in @(@('Start-QualityWorkspace.cmd','--quality-page'), @('Start-Workstation.cmd',''), @('Start-RealtimeCommands.cmd','--realtime-commands'))) {
        $launcher = @('@echo off', 'set "NEUROSTATION_RUNTIME_CONFIG="',
            'set "NEUROSTATION_QUALITY_ROOT=%LOCALAPPDATA%\NeuroStationQualityUpgrade"',
            'set "NEUROSTATION_DATASETS=%NEUROSTATION_QUALITY_ROOT%\recordings"',
            'set "NEUROSTATION_SETTINGS=%NEUROSTATION_QUALITY_ROOT%\settings"',
            'set "NEUROSTATION_DIAGNOSTICS=%NEUROSTATION_QUALITY_ROOT%\diagnostics"',
            'set "QT_QPA_PLATFORM="',
            'set "NS_QUALITY_EXE=%~dp0NeuroStationQualityUpgrade.exe"')
        $command = "Start-Process -FilePath `$env:NS_QUALITY_EXE -WindowStyle Normal"
        if ($entry[1]) { $command += " -ArgumentList '$($entry[1])'" }
        $launcher += "powershell.exe -NoProfile -WindowStyle Hidden -Command `"$command`""
        Set-Content -LiteralPath (Join-Path $artifact $entry[0]) -Value ($launcher -join "`r`n") -Encoding ascii
    }
    Copy-Item -LiteralPath (Join-Path $root 'docs/NeuroStation-质量工作区试用说明-20261106.md') -Destination (Join-Path $artifact 'README-QualityWorkspace.md')
    # The smoke runner removes source Python and build-tool paths from the executable environment.
    & $Python -B scripts/verify_quality_package.py --artifact $artifact --output $output
    if ($LASTEXITCODE -ne 0) { throw 'Independent executable verification failed; inspect build-verification.json' }
    Write-Output "VERIFIED_ARTIFACT=$artifact"
} finally { Pop-Location }

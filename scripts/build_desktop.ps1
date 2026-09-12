param(
    [string]$Python = "python",
    [ValidateSet("standalone", "onefile")]
    [string]$Mode = "standalone",
    [switch]$DryRun,
    [switch]$KeepStage
)

$ErrorActionPreference = "Stop"
$releaseVersion = "MVP1.0.1"
$releaseDate = "2026-09-12"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = (Get-Command $Python -ErrorAction Stop).Source
$pythonVersion = & $pythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$localData = [Environment]::GetFolderPath("LocalApplicationData")
$cacheRoot = [IO.Path]::GetFullPath((Join-Path $localData "NeuroStation\Build"))
$venvRoot = Join-Path $cacheRoot ("venv-" + $pythonVersion)
$stageRoot = [IO.Path]::GetFullPath(
    (Join-Path $cacheRoot ("stage-" + [Guid]::NewGuid().ToString("N")))
)
if (-not $stageRoot.StartsWith($cacheRoot + [IO.Path]::DirectorySeparatorChar)) {
    throw "Build stage escaped the NeuroStation cache root."
}

New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null
New-Item -ItemType Directory -Path $stageRoot | Out-Null

$sourceItems = @(
    "apps",
    "assets",
    "configs",
    "eeg_tools",
    "neurostation_contract.py",
    "workstation.py",
    "pysidedeploy.spec",
    "pyproject.toml"
)
foreach ($item in $sourceItems) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $item) -Destination $stageRoot -Recurse -Force
}
$integrationTarget = Join-Path $stageRoot "integrations\openbci_gui"
New-Item -ItemType Directory -Force -Path $integrationTarget | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot "integrations\openbci_gui\upstream.lock.json") -Destination $integrationTarget -Force

$pushedLocation = $false
try {
    $buildPython = Join-Path $venvRoot "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $buildPython)) {
        & $pythonPath -m venv $venvRoot
    }
    & $buildPython -c "import PySide6, nuitka, brainflow, numpy" 2>$null
    if ($LASTEXITCODE -ne 0) {
        & $buildPython -m pip install "PySide6>=6.7" "Nuitka>=4.2,<5" "brainflow>=5.22" "numpy>=2.0"
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to install desktop build dependencies."
        }
    }
    $deployScript = Join-Path $venvRoot "Lib\site-packages\PySide6\scripts\deploy.py"
    if (-not (Test-Path -LiteralPath $deployScript)) {
        throw "pyside6-deploy was not installed into $venvRoot"
    }

    Push-Location $stageRoot
    $pushedLocation = $true
    $arguments = @(
        $deployScript,
        "-c", (Join-Path $stageRoot "pysidedeploy.spec"),
        "--mode", $Mode,
        "--force"
    )
    if ($DryRun) {
        $arguments += "--dry-run"
    }
    & $buildPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Desktop build failed with exit code $LASTEXITCODE"
    }

    if (-not $DryRun) {
        $artifact = if ($IsWindows -and $Mode -eq "standalone") {
            Join-Path $stageRoot "dist\NeuroStation.dist"
        }
        elseif ($IsWindows) {
            Join-Path $stageRoot "dist\NeuroStation.exe"
        }
        elseif ($IsMacOS) {
            Join-Path $stageRoot "dist/NeuroStation.app"
        }
        elseif ($Mode -eq "standalone") {
            Join-Path $stageRoot "dist/NeuroStation.dist"
        }
        else {
            Join-Path $stageRoot "dist/NeuroStation.bin"
        }
        $entry = if ($IsWindows -and $Mode -eq "standalone") {
            Join-Path $artifact "workstation.exe"
        }
        elseif (-not $IsWindows -and -not $IsMacOS -and $Mode -eq "standalone") {
            Join-Path $artifact "workstation.bin"
        }
        else {
            $artifact
        }
        if (-not (Test-Path -LiteralPath $entry)) {
            throw "Deployment returned without creating the expected artifact: $entry"
        }

        if ($IsWindows -and $Mode -eq "standalone") {
            $brainflowSource = Join-Path $venvRoot "Lib\site-packages\brainflow\lib"
            $brainflowDestination = Join-Path $artifact "brainflow\lib"
            if (-not (Test-Path -LiteralPath (Join-Path $brainflowSource "BoardController.dll"))) {
                throw "BrainFlow native runtime is missing from the build environment."
            }
            New-Item -ItemType Directory -Force -Path $brainflowDestination | Out-Null
            Copy-Item -Path (Join-Path $brainflowSource "*") -Destination $brainflowDestination -Force
        }

        if ($IsWindows -and $Mode -eq "standalone") {
            $openbciSource = Join-Path $projectRoot "integrations\openbci_gui\runtime\windows"
            if (Test-Path -LiteralPath (Join-Path $openbciSource "OpenBCI_GUI.exe")) {
                $openbciDestination = Join-Path $artifact "integrations\openbci_gui\runtime\windows"
                New-Item -ItemType Directory -Force -Path $openbciDestination | Out-Null
                Copy-Item -Path (Join-Path $openbciSource "*") -Destination $openbciDestination -Recurse -Force
            }
        }

        $distRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot "dist"))
        New-Item -ItemType Directory -Path $distRoot -Force | Out-Null
        $destination = [IO.Path]::GetFullPath(
            (Join-Path $distRoot (Split-Path $artifact -Leaf))
        )
        if (-not $destination.StartsWith($distRoot + [IO.Path]::DirectorySeparatorChar)) {
            throw "Artifact destination escaped dist."
        }
        if (Test-Path -LiteralPath $destination) {
            Remove-Item -LiteralPath $destination -Recurse -Force
        }
        Copy-Item -LiteralPath $artifact -Destination $destination -Recurse -Force
        $releaseNotes = Join-Path $projectRoot "release\NeuroStation-MVP1.0.1-Windows-x64.md"
        if (-not (Test-Path -LiteralPath $releaseNotes)) {
            throw "Release notes are missing: $releaseNotes"
        }
        Copy-Item -LiteralPath $releaseNotes -Destination (Join-Path $destination "RELEASE_NOTES.md") -Force
        @(
            "NeuroStation"
            "Version: $releaseVersion"
            "Semantic version: 1.0.1"
            "Build date: $releaseDate"
            "Platform: Windows x64"
            "Release type: standalone portable MVP"
        ) | Set-Content -LiteralPath (Join-Path $destination "VERSION.txt") -Encoding UTF8
        $zipPath = Join-Path $distRoot "NeuroStation-MVP1.0.1-Windows-x64.zip"
        if (Test-Path -LiteralPath $zipPath) {
            Remove-Item -LiteralPath $zipPath -Force
        }
        Compress-Archive -Path $destination -DestinationPath $zipPath -CompressionLevel Optimal
        Write-Output "Built artifact: $destination"
        Write-Output "Built archive: $zipPath"
    }
}
finally {
    if ($pushedLocation) {
        Pop-Location
    }
    if (-not $KeepStage -and (Test-Path -LiteralPath $stageRoot)) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
}

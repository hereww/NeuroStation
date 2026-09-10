param(
    [string]$Python = "python",
    [switch]$SkipUnitTests
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = (Get-Command $Python -ErrorAction Stop).Source
$sourceRoot = Join-Path $projectRoot ".vendor\OpenBCI_GUI"
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot ".git"))) {
    throw "OpenBCI GUI source is missing. Run scripts\sync_openbci_gui.ps1 -Action sync first."
}

& $pythonPath (Join-Path $projectRoot "scripts\apply_openbci_gui_overlay.py") status
if ($LASTEXITCODE -ne 0) {
    throw "OpenBCI GUI localization overlay is not ready."
}

$cacheRoot = [IO.Path]::GetFullPath(
    (Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "NeuroStation\OpenBCI-Build")
)
$processingRoot = Join-Path $cacheRoot "processing-4.2"
$processingCommand = Join-Path $processingRoot "processing-java.exe"
New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
if (-not (Test-Path -LiteralPath $processingCommand)) {
    $archive = Join-Path $cacheRoot "processing-4.2-windows-x64.zip"
    & curl.exe -L --fail --retry 3 -o $archive "https://github.com/processing/processing4/releases/download/processing-1292-4.2/processing-4.2-windows-x64.zip"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to download Processing 4.2."
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $cacheRoot -Force
}
if (-not (Test-Path -LiteralPath $processingCommand)) {
    throw "Processing 4.2 was not installed correctly."
}

$libraryRoot = Join-Path ([Environment]::GetFolderPath("MyDocuments")) "Processing\libraries"
New-Item -ItemType Directory -Force -Path $libraryRoot | Out-Null
Get-ChildItem -LiteralPath (Join-Path $sourceRoot "OpenBCI_GUI\libraries") -Directory |
    ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $libraryRoot -Recurse -Force
    }

if (-not $SkipUnitTests) {
    $previousPath = $env:PATH
    $env:PATH = $processingRoot + [IO.Path]::PathSeparator + $previousPath
    Push-Location $sourceRoot
    try {
        & $pythonPath "GuiUnitTests\run-unittests.py"
        if ($LASTEXITCODE -ne 0) {
            throw "OpenBCI GUI unit tests failed."
        }
    }
    finally {
        Pop-Location
        $env:PATH = $previousPath
    }
}

$stageRoot = [IO.Path]::GetFullPath(
    (Join-Path $cacheRoot ("stage-" + [Guid]::NewGuid().ToString("N")))
)
if (-not $stageRoot.StartsWith($cacheRoot + [IO.Path]::DirectorySeparatorChar)) {
    throw "OpenBCI build stage escaped its cache root."
}
New-Item -ItemType Directory -Path $stageRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $sourceRoot "OpenBCI_GUI") -Destination $stageRoot -Recurse -Force
$sketch = Join-Path $stageRoot "OpenBCI_GUI"
$output = Join-Path $stageRoot "application.windows64"
& $processingCommand --force "--sketch=$sketch" "--output=$output" --variant=windows-amd64 --export
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $output "OpenBCI_GUI.exe"))) {
    throw "OpenBCI GUI Windows export failed. Stage: $stageRoot"
}

$runtimeRoot = [IO.Path]::GetFullPath(
    (Join-Path $projectRoot "integrations\openbci_gui\runtime\windows")
)
$allowedRuntimeRoot = [IO.Path]::GetFullPath(
    (Join-Path $projectRoot "integrations\openbci_gui\runtime")
)
if (-not $runtimeRoot.StartsWith($allowedRuntimeRoot + [IO.Path]::DirectorySeparatorChar)) {
    throw "OpenBCI runtime target escaped the project runtime directory."
}
if (Test-Path -LiteralPath $runtimeRoot) {
    Remove-Item -LiteralPath $runtimeRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $runtimeRoot | Out-Null
Get-ChildItem -LiteralPath $output |
    Where-Object Name -ne "source" |
    ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $runtimeRoot -Recurse -Force
    }

Write-Output "OpenBCI GUI unit tests and Windows export passed."
Write-Output "Runtime: $runtimeRoot"

[CmdletBinding()]
param(
    [ValidateSet("sync", "status", "overlay")]
    [string]$Action = "sync"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$lockPath = Join-Path $projectRoot "integrations\openbci_gui\upstream.lock.json"
$lock = Get-Content -Raw -LiteralPath $lockPath | ConvertFrom-Json
$sourcePath = Join-Path $projectRoot ($lock.source_directory -replace "/", "\")
$sourceParent = Split-Path -Parent $sourcePath
$overlayTool = Join-Path $projectRoot "scripts\apply_openbci_gui_overlay.py"

function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed: git $($Arguments -join ' ')"
    }
}

function Get-GitOutput {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $output = & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed: git $($Arguments -join ' ')"
    }
    return ($output -join "`n").Trim()
}

function Get-UpstreamRemote {
    $remoteNames = @(Get-GitOutput -C $sourcePath remote) -split "`n"
    foreach ($remoteName in $remoteNames) {
        if (-not $remoteName) {
            continue
        }
        $remoteUrl = Get-GitOutput -C $sourcePath remote get-url $remoteName
        if ($remoteUrl -eq $lock.repository) {
            return $remoteName
        }
    }
    throw "No Git remote points to the locked upstream '$($lock.repository)'."
}

if ($Action -eq "overlay") {
    if (-not (Test-Path -LiteralPath (Join-Path $sourcePath ".git"))) {
        throw "OpenBCI GUI source is not present. Run this script with -Action sync first."
    }
    & python $overlayTool apply --source $sourcePath
    if ($LASTEXITCODE -ne 0) {
        throw "OpenBCI GUI localization overlay failed."
    }
    exit 0
}

if ($Action -eq "status") {
    if (-not (Test-Path -LiteralPath (Join-Path $sourcePath ".git"))) {
        Write-Output "OpenBCI GUI source is not present. Run this script with -Action sync."
        exit 1
    }
    $upstreamRemote = Get-UpstreamRemote
    $upstreamUrl = Get-GitOutput -C $sourcePath remote get-url $upstreamRemote
    $currentRevision = Get-GitOutput -C $sourcePath rev-parse HEAD
    $branch = Get-GitOutput -C $sourcePath branch --show-current
    $changes = Get-GitOutput -C $sourcePath status --short
    & python $overlayTool status --source $sourcePath *> $null
    $overlayApplied = $LASTEXITCODE -eq 0
    [pscustomobject]@{
        Source = $sourcePath
        UpstreamRemote = $upstreamRemote
        UpstreamUrl = $upstreamUrl
        ExpectedRevision = $lock.revision
        CurrentRevision = $currentRevision
        Branch = $(if ($branch) { $branch } else { "(detached)" })
        Clean = -not [bool]$changes
        ZhCnOverlayApplied = $overlayApplied
    } | Format-List
    exit 0
}

if (-not (Test-Path -LiteralPath (Join-Path $sourcePath ".git"))) {
    New-Item -ItemType Directory -Path $sourceParent -Force | Out-Null
    Invoke-Git clone --no-checkout $lock.repository $sourcePath
    $upstreamRemote = "origin"
} else {
    $upstreamRemote = Get-UpstreamRemote
    $changes = Get-GitOutput -C $sourcePath status --short
    if ($changes) {
        throw "OpenBCI GUI source has local changes. Commit or stash them before syncing."
    }
    $currentRevision = Get-GitOutput -C $sourcePath rev-parse HEAD
    $branch = Get-GitOutput -C $sourcePath branch --show-current
    if ($currentRevision -ne $lock.revision -and $branch) {
        throw "OpenBCI GUI is on branch '$branch'. Switch to detached HEAD before changing the locked revision."
    }
}

Invoke-Git -C $sourcePath fetch --depth 1 $upstreamRemote $lock.revision
Invoke-Git -C $sourcePath checkout --detach $lock.revision
Write-Output "OpenBCI GUI source is ready at $sourcePath"
Write-Output "Locked revision: $($lock.revision)"

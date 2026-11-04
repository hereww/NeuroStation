# Formal v4 EXPLICIT OFFLINE diagnostics; no UI, hardware or acquisition.
[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [string]$InputPath,
    [string]$OutputDir,
    [Nullable[double]]$SampleRate,
    [string]$ValidMask,
    [switch]$AllFiniteValid,
    [string]$SignalKey = 'signal',
    [string]$Checkpoint = 'models/denoiser_v4.pt',
    [string]$Config = 'configs/learned_denoiser_v4.json',
    [string]$Interface = 'analysis_outputs/learned_denoiser_v4_interface_contract.json',
    [ValidateSet('cpu', 'cuda', 'cuda:0')][string]$Device = 'cuda',
    [ValidateRange(1, 2147483647)][int]$Threads = 2,
    [ValidateRange(1, 2147483647)][int]$ChunkSize = 128,
    [string]$CheckReport,
    [string]$Root,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$ProjectRoot = if ($PSBoundParameters.ContainsKey('Root')) {
    [System.IO.Path]::GetFullPath($Root)
} else { [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')) }
function Resolve-ProjectPath([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { throw 'Explicit paths must not be blank.' }
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Value))
}
# Acquisition .venv is never selected implicitly, upgraded, installed into or modified.
$Python = if ($PSBoundParameters.ContainsKey('PythonPath')) {
    Resolve-ProjectPath $PythonPath
} else { Join-Path $ProjectRoot '.venv-ml-gpu\Scripts\python.exe' }
function Write-Failure([string]$Message) {
    [ordered]@{
        status = 'formal'; result = 'failed'; diagnostic_ready = $false; error = $Message
        python_executable = $Python; project_root = $ProjectRoot
        training_executed = $false; test_evaluation_executed = $false
        automatic_processing_enabled = $false; production_validated = $false
        gui_started = $false; hardware_opened = $false; acquisition_started = $false
        notice = 'Readiness is NOT training success or production acceptance. No runtime/device fallback.'
    } | ConvertTo-Json -Depth 5
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    Write-Failure "Selected isolated interpreter missing: $Python. No CPU/PATH fallback."
    exit 1
}
Push-Location -LiteralPath $ProjectRoot
try {
    if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = '2' }
    if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = '2' }
    $env:PYTHONDONTWRITEBYTECODE = '1'
    # FIRST executable preflight: formal release configs. No model/source load
    # or other inference preflight can precede this release gate.
    & $Python -B (Join-Path $ProjectRoot 'scripts\validate_release_configs.py')
    if ($LASTEXITCODE -ne 0) {
        Write-Failure 'Formal release configuration check failed; diagnostics refused.'
        exit 1
    }
    $Arguments = @('-B', (Join-Path $ProjectRoot 'scripts\run_learned_denoise_v4.py'),
        '--root', $ProjectRoot, '--model', (Resolve-ProjectPath $Checkpoint),
        '--config', (Resolve-ProjectPath $Config), '--interface', (Resolve-ProjectPath $Interface),
        '--device', $Device, '--threads', $Threads.ToString(), '--chunk-size', $ChunkSize.ToString())
    if ($CheckOnly) { $Arguments += '--check-only' }
    if ($PSBoundParameters.ContainsKey('InputPath')) { $Arguments += @('--input', (Resolve-ProjectPath $InputPath)) }
    if ($PSBoundParameters.ContainsKey('OutputDir')) { $Arguments += @('--output', (Resolve-ProjectPath $OutputDir)) }
    if ($PSBoundParameters.ContainsKey('SampleRate')) {
        $Arguments += @('--sample-rate', $SampleRate.ToString([System.Globalization.CultureInfo]::InvariantCulture))
    }
    if ($PSBoundParameters.ContainsKey('ValidMask')) { $Arguments += @('--valid-mask', (Resolve-ProjectPath $ValidMask)) }
    if ($AllFiniteValid) { $Arguments += '--all-finite-valid' }
    if ($PSBoundParameters.ContainsKey('CheckReport')) { $Arguments += @('--check-report', (Resolve-ProjectPath $CheckReport)) }
    $Arguments += @('--signal-key', $SignalKey)
    & $Python @Arguments
    exit $LASTEXITCODE
}
catch {
    Write-Failure $_.Exception.Message
    exit 1
}
finally { Pop-Location }

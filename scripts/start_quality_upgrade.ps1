param(
    [string]$RuntimeConfig = (Join-Path $PSScriptRoot '..\.quality-local\runtime.json'),
    [ValidateSet('workstation','realtime','check')][string]$Entry = 'workstation'
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$configPath = (Resolve-Path -LiteralPath $RuntimeConfig).Path
$config = Get-Content -LiteralPath $configPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($config.schema_version -ne 1) { throw 'Unsupported runtime binding schema' }
$python = $config.files.capture_python.path
if ((Get-FileHash -LiteralPath $python -Algorithm SHA256).Hash.ToLower() -ne $config.files.capture_python.sha256) {
    throw 'Python executable identity changed. Revalidate the binding first.'
}
$env:NEUROSTATION_RUNTIME_CONFIG = $configPath
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $root
try {
    & $python -B scripts/validate_release_configs.py
    if ($LASTEXITCODE -ne 0) { throw 'Formal configuration check failed' }
    if ($Entry -eq 'check') { & $python -B -c 'import sys,brainflow,numpy,scipy,PySide6; from eeg_tools.runtime_bindings import bound_file; print(sys.executable); print(bound_file("model_python", "missing"))' }
    elseif ($Entry -eq 'realtime') { & $python -B -m apps.realtime_commands }
    else { & $python -B workstation.py }
    if ($LASTEXITCODE -ne 0) { throw "Branch startup failed: $LASTEXITCODE" }
} finally { Pop-Location }

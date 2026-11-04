[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [string]$InputFile,
    [ValidateSet('Nakanishi2015','Wearable')][string]$Domain,
    [int]$SampleRateHz,
    [string]$ChannelOrder,
    [string]$ValidMask,
    [switch]$AllFiniteValid,
    [switch]$AcknowledgeOfflineTest,
    [switch]$AcknowledgeUnverifiedChannelMapping,
    [ValidateSet('cpu','cuda','cuda:0')][string]$Device='cuda:0',
    [string]$OutputDirectory
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$python=Join-Path $root '.venv-ml-gpu\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {throw 'Missing .venv-ml-gpu Python; consult the test model guide.'}
$script=Join-Path $PSScriptRoot 'run_ssvep_test_model.py'
$argsList=@($script,'--device',$Device)
if ($CheckOnly) {$argsList+='--check-only'}
if ($InputFile) {$argsList+=@('--input',$InputFile)}
if ($Domain) {$argsList+=@('--domain',$Domain)}
if ($SampleRateHz) {$argsList+=@('--sample-rate-hz',"$SampleRateHz")}
if ($ChannelOrder) {$argsList+=@('--channel-order',$ChannelOrder)}
if ($ValidMask) {$argsList+=@('--valid-mask',$ValidMask)}
if ($AllFiniteValid) {$argsList+='--all-finite-valid'}
if ($AcknowledgeOfflineTest) {$argsList+='--acknowledge-offline-test'}
if ($AcknowledgeUnverifiedChannelMapping) {$argsList+='--acknowledge-unverified-channel-mapping'}
if ($OutputDirectory) {$argsList+=@('--output',$OutputDirectory)}
# No hardware/GUI process, no background console, no training, no auto-install.
& $python @argsList
exit $LASTEXITCODE

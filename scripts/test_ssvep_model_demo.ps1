[CmdletBinding()]
param(
    [ValidateSet('Wearable','Nakanishi2015')][string]$Domain='Wearable',
    [ValidateSet('cpu','cuda','cuda:0')][string]$Device='cuda:0'
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$example=Join-Path $root 'analysis_outputs\ssvep_v9_examples_20261008'
$inputFile=Join-Path $example ($Domain+'_input.npy')
$validMask=Join-Path $example ($Domain+'_valid_mask.npy')
if (-not (Test-Path -LiteralPath $inputFile -PathType Leaf)) {throw 'Prepared v9 example missing. Consult the test model guide; do not substitute raw acquisitions silently.'}
$rate=if ($Domain -eq 'Wearable') {250} else {256}
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$deviceLabel=$Device.Replace(':','-')
$output=Join-Path $root ('analysis_outputs\ssvep_v9_demo_'+$Domain+'_'+$deviceLabel+'_'+$stamp)
$labels='source-index-0,source-index-1,source-index-2,source-index-3,source-index-4,source-index-5,source-index-6,source-index-7'
Write-Host 'Offline development example ONLY. No hardware, no production or clinical validation.'
& (Join-Path $PSScriptRoot 'start_ssvep_test_model.ps1') -InputFile $inputFile -Domain $Domain -SampleRateHz $rate -ChannelOrder $labels -ValidMask $validMask -AcknowledgeOfflineTest -AcknowledgeUnverifiedChannelMapping -Device $Device -OutputDirectory $output
exit $LASTEXITCODE

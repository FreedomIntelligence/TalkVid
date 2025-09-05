param(
    [string]$InputJson = "$PSScriptRoot\filtered_video_clips.json",
    [string]$OutputDir = "$PSScriptRoot\clips_output",
    [int]$Limit = 10,
    [int]$Workers = 4,
    [string]$Cookies = "--cookies-from-browser",
    [ValidateSet('firefox')]
    [string]$Browser = 'firefox',
    [string]$ExtractorArgs = ''
)

$ErrorActionPreference = "Stop"

Write-Host "Input: $InputJson"
Write-Host "Output: $OutputDir"
if ($Limit -gt 0) { Write-Host "Limit: $Limit" }
Write-Host "Workers: $Workers"
if ($Browser) { Write-Host "Browser cookies: $Browser" }
if ($ExtractorArgs) { Write-Host "Extractor args: $ExtractorArgs" }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null


# Run
$argsList = @('--input', $InputJson, '--output', $OutputDir, '--workers', $Workers)
if ($Limit -gt 0) { $argsList += @('--limit', $Limit) }
if (Test-Path $Cookies) { $argsList += @('--cookies', $Cookies) }
if ($Browser) { $argsList += @('--browser', $Browser) }
if ($ExtractorArgs) { $argsList += @('--extractor_args', $ExtractorArgs) }

python "$PSScriptRoot\download_clips.py" @argsList

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "All done."



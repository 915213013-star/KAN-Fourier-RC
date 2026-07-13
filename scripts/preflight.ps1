param(
  [ValidateSet("10a", "10b", "both")]
  [string]$Dataset = "both",
  [switch]$SkipDataCheck
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$PythonCandidates = @($env:KANFOURIER_PYTHON, $env:KANLIE_PYTHON, "python") | Where-Object { $_ }
$Python = $null
foreach ($Candidate in $PythonCandidates) {
  $Resolved = Get-Command $Candidate -ErrorAction SilentlyContinue
  if ($Resolved) { $Python = $Resolved.Source; break }
}
if (-not $Python) { throw "Python was not found. Set KANFOURIER_PYTHON to a Python 3.10 executable." }

Write-Host "Root:   $Root"
Write-Host "Python: $Python"
& $Python -c "import numpy, scipy, sklearn, xgboost, torch, matplotlib, joblib; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Python dependency check failed. Install requirements.txt." }

& $Python "$Root\train_fourier_compressed_main_seeds_2016.py" --help | Out-Null
& $Python "$Root\apply_crossfit_candidate_action_router_2016.py" --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Command import check failed." }

if (-not $SkipDataCheck) {
  if ($Dataset -in @("10a", "both")) {
    $Path10A = "$Root\raw_data\RML2016.10a_dict.pkl"
    if (-not (Test-Path $Path10A)) { throw "Missing 10A dataset: $Path10A" }
  }
  if ($Dataset -in @("10b", "both")) {
    $Path10B = "$Root\data\RML2016.10b.dat"
    if (-not (Test-Path $Path10B)) { throw "Missing 10B dataset: $Path10B" }
  }
}
Write-Host "Preflight passed."


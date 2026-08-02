param(
  [ValidateSet("10a", "10b", "both")]
  [string]$Dataset = "both",
  [switch]$SkipDataCheck,
  [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$PythonCandidates = @(
  $PythonExecutable,
  $env:KANFOURIER_PYTHON,
  $env:KANLIE_PYTHON,
  (Join-Path $Root ".venv\Scripts\python.exe"),
  "python"
) | Where-Object { $_ } | Select-Object -Unique
$Python = $null
foreach ($Candidate in $PythonCandidates) {
  $Resolved = Get-Command $Candidate -ErrorAction SilentlyContinue
  if (-not $Resolved) { continue }
  $ResolvedPath = $Resolved.Source
  & $ResolvedPath -c "import numpy, scipy, sklearn, xgboost, torch, matplotlib, joblib" 2>$null
  if ($LASTEXITCODE -eq 0) {
    $Python = $ResolvedPath
    break
  }
  Write-Warning "Skipping Python without the required dependencies: $ResolvedPath"
}
if (-not $Python) {
  throw "No dependency-complete Python was found. Pass -PythonExecutable or set KANFOURIER_PYTHON."
}

Write-Host "Root:   $Root"
Write-Host "Python: $Python"
& $Python -c "import numpy, scipy, sklearn, xgboost, torch, matplotlib, joblib; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Python dependency check failed. Install requirements.txt." }

& $Python "$Root\tools\preflight_release.py" --root $Root
if ($LASTEXITCODE -ne 0) { throw "Release integrity check failed." }

$CommandChecks = @(
  "$Root\train_fourier_compressed_main_seeds_2016.py",
  "$Root\train_fourier_compressed_main_oof_2016.py",
  "$Root\train_fourier_compressed_main_oof_10b.py",
  "$Root\train_cv_trn_aux_v2_oof_2016.py",
  "$Root\train_cv_trn_aux_v2_oof_10b.py",
  "$Root\apply_crossfit_candidate_action_router_2016.py"
)
foreach ($Command in $CommandChecks) {
  & $Python $Command --help | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "Command import check failed: $Command" }
}

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

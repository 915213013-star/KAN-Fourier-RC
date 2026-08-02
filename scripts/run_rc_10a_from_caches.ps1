param(
  [string]$BasePredictions = "",
  [string[]]$CandidatePredictions = @(),
  [string]$OutputSuffix = "kan_fourier_rc_10a_split1",
  [switch]$SkipExisting,
  [switch]$Help
)

if ($Help) {
  Write-Host "Supply one frozen stage prediction cache as -BasePredictions and aligned alternatives as -CandidatePredictions."
  exit 0
}
if (-not $BasePredictions -or $CandidatePredictions.Count -eq 0) {
  throw "-BasePredictions and at least one -CandidatePredictions path are required. Use -Help for a summary."
}
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:KANFOURIER_PYTHON) { $env:KANFOURIER_PYTHON } elseif ($env:KANLIE_PYTHON) { $env:KANLIE_PYTHON } else { "python" }
$Output = "$Root\results\$($OutputSuffix)_predictions.npz"
if ($SkipExisting -and (Test-Path $Output)) { Write-Host "Reusing $Output"; exit 0 }
if (-not (Test-Path $BasePredictions)) { throw "Missing base cache: $BasePredictions" }
foreach ($Path in $CandidatePredictions) { if (-not (Test-Path $Path)) { throw "Missing candidate cache: $Path" } }

$Args = @(
  "-u", "$Root\apply_crossfit_candidate_action_router_2016.py",
  "--base_predictions", $BasePredictions, "--candidate_predictions"
) + $CandidatePredictions + @(
  "--skip_bad_candidates", "--final_report_only", "--cv_folds", "5", "--min_counts", "1", "2", "3", "5",
  "--min_nets", "1", "2", "--min_precisions", "0.55", "0.60", "0.70",
  "--max_change_rates", "0.01", "0.03", "0.05", "0.10", "--max_actions", "1", "2", "3", "5",
  "--min_fold_count", "1", "--min_positive_folds", "1", "2", "--max_negative_folds", "0", "1",
  "--allow_no_change", "--score_transition_gain_weight", "0.030", "--score_midlow_gain_weight", "0.025",
  "--score_edge_gain_weight", "0.020", "--score_high_drop_penalty", "5.0", "--high_tolerance", "0.02",
  "--score_change_penalty", "0.020", "--score_action_penalty", "0.006", "--output_suffix", $OutputSuffix
)
& $Python @Args
if ($LASTEXITCODE -ne 0) { throw "10A RC selection failed with exit code $LASTEXITCODE" }

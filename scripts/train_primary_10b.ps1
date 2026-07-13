param(
  [string]$DataPath = "",
  [string]$AlignmentCache = "",
  [int]$NumWorkers = 0,
  [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $DataPath) { $DataPath = "$Root\data\RML2016.10b.dat" }
$Python = if ($env:KANFOURIER_PYTHON) { $env:KANFOURIER_PYTHON } elseif ($env:KANLIE_PYTHON) { $env:KANLIE_PYTHON } else { "python" }
if (-not (Test-Path $DataPath)) { throw "Missing dataset: $DataPath" }

$Args = @(
  "-u", "$Root\train_fourier_compressed_main_seeds_10b.py",
  "--variant", "full_geo_2expert", "--split_seed", "1", "--model_seeds", "361",
  "--epochs", "260", "--patience", "50", "--batch_size", "128", "--eval_batch_size", "256",
  "--num_workers", "$NumWorkers", "--lr", "2e-3", "--eta_min", "4e-6", "--weight_decay", "1e-5",
  "--alpha_supcon", "0.30", "--run_tag", "paper10b", "--data_path", $DataPath,
  "--cache_dir", "$Root\feature_cache"
)
if ($AlignmentCache) { $Args += @("--alignment_cache", $AlignmentCache) } else { $Args += "--skip_alignment_check" }
if ($ForceRestart) { $Args += "--force_restart" }
& $Python @Args
if ($LASTEXITCODE -ne 0) { throw "10B primary training failed with exit code $LASTEXITCODE" }


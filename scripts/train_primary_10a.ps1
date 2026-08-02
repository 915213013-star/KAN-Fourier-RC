param(
  [string]$DataPath = "",
  [string]$AlignmentCache = "",
  [string]$InitCheckpoint = "",
  [int]$NumWorkers = 0,
  [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $DataPath) { $DataPath = "$Root\raw_data\RML2016.10a_dict.pkl" }
$Python = if ($env:KANFOURIER_PYTHON) { $env:KANFOURIER_PYTHON } elseif ($env:KANLIE_PYTHON) { $env:KANLIE_PYTHON } else { "python" }
if (-not (Test-Path $DataPath)) { throw "Missing dataset: $DataPath" }

$Args = @(
  "-u", "$Root\train_fourier_compressed_main_seeds_2016.py",
  "--variant", "full_geo_2expert", "--split_seed", "1", "--model_seeds", "261",
  "--epochs", "140", "--patience", "38", "--batch_size", "128", "--eval_batch_size", "256",
  "--num_workers", "$NumWorkers", "--lr", "2.8e-4", "--eta_min", "8e-7", "--weight_decay", "5e-6",
  "--grad_clip", "1.0", "--alpha_supcon", "0.14", "--temperature", "0.07",
  "--soft_rank_weight", "2e-4", "--soft_rank_keep_ratio", "0.92", "--soft_rank_every", "12",
  "--negative_snr_weight", "1.08", "--high_snr_weight", "1.0", "--edge_snr_weight", "1.0",
  "--transition_snr_weight", "1.16", "--roll_prob", "0.08", "--roll_max", "3",
  "--iq_noise_prob", "0.01", "--iq_noise_std", "0.003", "--augment_warmup_epochs", "8",
  "--snr_weight_warmup_epochs", "8", "--run_tag", "paper10a", "--data_path", $DataPath,
  "--cache_dir", "$Root\feature_cache"
)
if ($AlignmentCache) { $Args += @("--alignment_cache", $AlignmentCache) } else { $Args += "--skip_alignment_check" }
if ($InitCheckpoint) { $Args += @("--init_checkpoint", $InitCheckpoint) }
if ($ForceRestart) { $Args += "--force_restart" }
& $Python @Args
if ($LASTEXITCODE -ne 0) { throw "10A primary training failed with exit code $LASTEXITCODE" }


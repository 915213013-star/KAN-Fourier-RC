param(
  [switch]$SkipExisting,
  [switch]$PreflightOnly,
  [ValidateSet("cpu", "cuda")]
  [string]$XgbDevice = "cuda",
  [int]$XgbJobs = -1
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$ROOT = Split-Path -Parent $PSScriptRoot
$TENB = "$ROOT\data\RML2016.10BGAMC"
$RESULTS = "$ROOT\results"
$FEATURES = "$ROOT\feature_cache"
$MODEL_CACHE = "$RESULTS\model_cache\10b_formal_665_multiaux"
$PAIRWISE_MODEL_CACHE = "$RESULTS\model_cache\10b_fullgeo2expert_hcs_pairwise\pairwise"

$PythonCandidates = @(
  $env:KANFOURIER_PYTHON,
  $env:KANLIE_PYTHON,
  "python"
) | Where-Object { $_ }

$PY = $null
foreach ($Candidate in $PythonCandidates) {
  $Resolved = Get-Command $Candidate -ErrorAction SilentlyContinue
  if ($Resolved) {
    $PY = $Resolved.Source
    break
  }
}
if (-not $PY) {
    throw "No usable Python interpreter found. Set KANFOURIER_PYTHON to a Python 3.10 executable."
}

$SplitSeed = 1
$DataPath = "$ROOT\data\RML2016.10b.dat"
$MainCache = "$RESULTS\fourier_compressed_10b_full_geo_2expert_mseed361_split1_valtest_probs_for_soup.npz"
$FourierOof = "$RESULTS\fourier_compressed_oof_10b_full_geo_2expert_mseed369_f3e220_single361_split1_trainvaltest_probs_for_meta.npz"
$CvTrnOof = "$RESULTS\cv_trn_aux_v2_oof_10b_mseed141_f3e220_split1_trainvaltest_probs_for_meta.npz"
$GamcOof = "$RESULTS\gamc_oof_tree_10b_split1_trainvaltest_probs_for_meta.npz"
$HcsOof = "$RESULTS\hcs_analog_aux_oof_10b_mseed2037_f3e320_split1_trainvaltest_probs_for_meta.npz"
$PairwiseOld = "$RESULTS\pairwise_confusion_selected6_oof_10b_f3e280_split1_trainvaltest_probs_for_meta.npz"
$PairwiseRaw = "$RESULTS\pairwise_confusion_selected6_raw_oof_10b_f3e280_split1_trainvaltest_probs_for_meta.npz"
$HcsFeatures = "$FEATURES\hcs_lite_10b_features_v1.npz"
$GamcFeatures = "$FEATURES\gamc_lite_10b_features_v3_graph_xgb.npz"

$RouterSuffix = "fourier_compressed_10b_fullgeo2expert_mseed361_formal_multiaux_router_split1"
$RouterPred = "$RESULTS\$($RouterSuffix)_predictions.npz"
$FinalSuffix = "fourier_compressed_10b_fullgeo2expert_mseed361_formal665_crossfit_late_split1"
$FinalPred = "$RESULTS\$($FinalSuffix)_predictions.npz"
$FormalBaseline = "$RESULTS\fourier_compressed_10b_fullgeo2expert_mseed361_hcs_pairwise_extraaux_meta_split1_predictions.npz"
$StrongReference = "$RESULTS\fourier_strong_soup_10b_oof179_cvtrn141_gamc_xgb6_residual_meta_split1_predictions.npz"

function Require-File($Path, $Name) {
  if (-not (Test-Path $Path)) {
    throw "Missing ${Name}: $Path"
  }
}

function Run-Step($Name, $OutPath, [scriptblock]$Body) {
  if ($SkipExisting -and (Test-Path $OutPath)) {
    Write-Host "${Name}: reusing $OutPath"
    return
  }
  Write-Host "========================================================================================================================"
  Write-Host $Name
  Write-Host "========================================================================================================================"
  & $Body
  if ($LASTEXITCODE -ne 0) {
    throw "$Name failed with exit code $LASTEXITCODE"
  }
  Require-File $OutPath $Name
}

Write-Host "RML2016.10B frozen KAN-Fourier-RC cache pipeline"
Write-Host "  Python:          $PY"
Write-Host "  XGBoost device:  $XgbDevice"
Write-Host "  Primary:         KAN-Fourier full_geo_2expert seed361"
Write-Host "  Stage OOF:       KAN-Fourier + IQCC-Former + GAMC + HCS + Pairwise"
Write-Host "  Router sources:  Pairwise / IQCC-Former / HCS / GAMC / KAN-Fourier"
Write-Host "  Protocol:        train OOF fits utility; validation fixes policy; test labels are excluded from optimization"

Require-File $MainCache "compressed seed361 val/test cache"
Require-File $FourierOof "compressed Fourier train-OOF cache"
Require-File $CvTrnOof "CV-TRN train-OOF cache"
Require-File $GamcOof "GAMC train-OOF cache"
Require-File $HcsOof "HCS train-OOF cache"
Require-File $PairwiseOld "existing Pairwise OOF cache"
Require-File $FormalBaseline "existing formal HCS/Pairwise baseline"
Require-File $HcsFeatures "HCS blind feature cache"
Require-File $GamcFeatures "GAMC blind feature cache"
Require-File $DataPath "RML2016.10B dataset"

if ($PreflightOnly) {
  Write-Host "Preflight passed. All required inputs are present; no training was started."
  exit 0
}

Run-Step "Step 1/3: Re-export cached Pairwise specialists with raw per-pair OOF scores" $PairwiseRaw {
  & $PY -u "$ROOT\train_pairwise_confusion_aux_10b.py" `
    --split_seed $SplitSeed `
    --random_state 2141 `
    --main_oof_cache $FourierOof `
    --soup_prob_cache $MainCache `
    --cvtrn_oof_cache $CvTrnOof `
    --gamc_oof_cache $GamcOof `
    --hcs_oof_cache $HcsOof `
    --hcs_feature_cache $HcsFeatures `
    --gamc_feature_cache $GamcFeatures `
    --stat_feature_sources gamc `
    --pairs "6,7" "1,9" "3,4" "0,8" "0,2" "2,5" `
    --selection_mode per_pair_greedy `
    --per_pair_min_score_gain 0.003 `
    --folds 3 `
    --xgb_estimators 280 `
    --xgb_depth 3 `
    --xgb_lr 0.035 `
    --xgb_jobs $XgbJobs `
    --xgb_device $XgbDevice `
    --model_cache_dir $PAIRWISE_MODEL_CACHE `
    --reuse_models `
    --search_jobs 2 `
    --blend_alphas 0.50 0.65 0.80 `
    --pair_conf_thresholds 0.58 0.66 0.75 `
    --pair_margin_thresholds 0.05 0.12 0.22 `
    --main_pair_mass_thresholds 0.45 0.65 `
    --main_pair_margin_maxes 0.08 0.18 1.01 `
    --max_change_rates 0.10 0.20 0.35 0.50 `
    --min_change_rate 0.02 `
    --data_path $DataPath `
    --cache_dir $FEATURES `
    --alignment_cache $MainCache `
    --output_cache $PairwiseRaw
}

Run-Step "Step 2/3: Train formal train-OOF multi-aux rescue/risk router" $RouterPred {
  & $PY -u "$ROOT\evaluate_fourier_oof_pairwise_transition_router_10b.py" `
    --split_seed $SplitSeed `
    --random_state 2026 `
    --soup_prob_cache $MainCache `
    --fourier_oof_cache $FourierOof `
    --cvtrn_oof_cache $CvTrnOof `
    --gamc_oof_cache $GamcOof `
    --hcs_precision_cache $HcsOof `
    --pairwise_cache $PairwiseRaw `
    --hcs_feature_cache $HcsFeatures `
    --gamc_feature_cache $GamcFeatures `
    --router_stat_features_per_source 64 `
    --use_oof_cvtrn_only `
    --main_display_name "Compressed KAN-Fourier seed361" `
    --cv_display_name "CV-TRN" `
    --stage_aux_sources hcs pairwise `
    --router_aux_sources pairwise cvtrn hcs gamc main `
    --use_pairwise_raw_router_features `
    --stage_oof_folds 3 `
    --stage_models xgb_d2_620 xgb_d3_520 xgb_d4_400 `
    --stage_min_val_overall 66.20 `
    --top_stage_configs 2 `
    --blend_alphas 0.45 0.50 0.55 0.60 0.65 `
    --meta_conf_thresholds 0.00 0.25 0.35 `
    --advantage_thresholds -0.05 0.00 0.05 0.10 `
    --max_change_rates 6 8 10 `
    --router_estimators 320 `
    --router_depths 2 `
    --router_learning_rate 0.035 `
    --router_thresholds 0.45 0.50 0.55 0.60 0.65 0.70 0.75 `
    --router_max_change_rates 0.05 0.10 0.20 0.35 0.50 `
    --router_alphas 1.00 0.80 0.65 `
    --router_min_change_rate 0.02 `
    --router_min_stage_overall_gain 0.0 `
    --router_min_global_stage_overall_gain 0.01 `
    --min_transition_count 160 `
    --min_transition_pos 24 `
    --min_transition_precision 0.50 `
    --transition_max_train_harm_rate 0.55 `
    --max_transitions_per_aux 8 `
    --rescue_weight 40 `
    --harm_weight 36 `
    --other_weight 1.0 `
    --score_negative_gain_weight 0.015 `
    --score_edge_gain_weight 0.015 `
    --score_transition_gain_weight 0.015 `
    --score_midlow_gain_weight 0.006 `
    --score_wide_transition_gain_weight 0.006 `
    --score_high_penalty 4.0 `
    --high_tolerance 0.03 `
    --score_changed_high_penalty 0.020 `
    --score_changed_nonultra_penalty 0.008 `
    --xgb_jobs $XgbJobs `
    --xgb_device $XgbDevice `
    --model_cache_dir $MODEL_CACHE `
    --reuse_models `
    --data_path $DataPath `
    --cache_dir $FEATURES `
    --alignment_cache $MainCache `
    --defer_test_report `
    --output_suffix $RouterSuffix
}

Run-Step "Step 3/3: Strict five-fold validation-stable late action selection" $FinalPred {
  & $PY -u "$ROOT\apply_crossfit_candidate_action_router_2016.py" `
    --base_predictions $RouterPred `
    --candidate_predictions $FormalBaseline $MainCache $FourierOof $CvTrnOof $GamcOof $HcsOof $PairwiseRaw `
    --output_suffix $FinalSuffix `
    --skip_bad_candidates `
    --cv_folds 5 `
    --min_counts 60 120 240 `
    --min_nets 8 12 20 `
    --min_precisions 0.58 0.62 0.66 `
    --max_change_rates 0.02 0.05 0.10 0.20 `
    --max_actions 1 2 `
    --min_fold_count 8 `
    --min_positive_folds 4 5 `
    --max_negative_folds 0 `
    --min_cv_overall_gain 0.0 `
    --score_transition_gain_weight 0.020 `
    --score_midlow_gain_weight 0.010 `
    --score_edge_gain_weight 0.010 `
    --score_high_drop_penalty 6.0 `
    --high_tolerance 0.02 `
    --score_change_penalty 0.035 `
    --score_action_penalty 0.015 `
    --final_report_only `
    --allow_no_change
}

Write-Host "========================================================================================================================"
Write-Host "Final formal comparison"
Write-Host "========================================================================================================================"
$CompareFiles = @($FormalBaseline, $StrongReference, $FinalPred) | Where-Object { Test-Path $_ }
& $PY "$ROOT\tools\compare_prediction_npz_2016.py" $CompareFiles
if ($LASTEXITCODE -ne 0) {
  throw "Final comparison failed with exit code $LASTEXITCODE"
}

Write-Host "Formal locked result:"
Write-Host $FinalPred
Write-Host "All trainable classical models were fingerprint-cached under:"
Write-Host $MODEL_CACHE

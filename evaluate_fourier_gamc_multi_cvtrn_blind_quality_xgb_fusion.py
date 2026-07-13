import argparse
import csv
import json
import os

import numpy as np
from xgboost import XGBClassifier

import evaluate_fourier_gamc_multi_cvtrn_xgb_stacked_residual_fusion as multi
import evaluate_fourier_gamc_cvtrn_gain_risk_router_fusion as gr
import evaluate_fourier_gamc_cvtrn_stacked_residual_fusion as st
import evaluate_fourier_soup_gamc_cvtrn_protected_residual_fusion as tri
import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import train_cv_trn_aux_2016 as common


def project_root():
    return os.path.dirname(os.path.abspath(__file__))


def relpath(*parts):
    return os.path.join(project_root(), *parts)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Blind quality-aware Multi-CVTRN XGBoost fusion. Fourier soup remains the main model; "
            "GAMC and CV-TRN are auxiliary experts; a blind CQI estimator is trained only from "
            "training-split I/Q features and SNR-bin labels."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)
    p.add_argument("--soup_prob_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--gamc_cache", type=str, default=relpath("results", "gamc_lite_v2_xgb_split1_valtest_probs_for_fusion.npz"))
    p.add_argument(
        "--cvtrn_caches",
        type=str,
        nargs="+",
        default=[
            relpath("results", "cv_trn_aux_v2_soup_tta_split1_valtest_probs_for_fusion.npz"),
            relpath("results", "cv_trn_aux_v2_w8d96_soup_tta_split1_valtest_probs_for_fusion.npz"),
        ],
    )
    p.add_argument("--output_suffix", type=str, default="")

    p.add_argument("--temperature_grid", type=float, nargs="+", default=[0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00])
    p.add_argument("--disable_temperature_scaling", action="store_true")

    p.add_argument("--mix_alphas", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    p.add_argument("--feature_candidates", type=int, default=4)
    p.add_argument("--min_candidate_rescues", type=int, default=30)
    p.add_argument("--folds", type=int, default=5)

    p.add_argument("--blend_alphas", type=float, nargs="+", default=[0.35, 0.50, 0.65, 0.80, 0.90, 1.00])
    p.add_argument("--meta_conf_thresholds", type=float, nargs="+", default=[0.00, 0.35, 0.45, 0.55, 0.65])
    p.add_argument("--advantage_thresholds", type=float, nargs="+", default=[-0.20, -0.10, 0.00, 0.05, 0.10])
    p.add_argument("--max_change_rates", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.0, 16.0, 22.0, 100.0])
    p.add_argument("--hard_keep_conf", type=float, default=0.93)
    p.add_argument("--hard_keep_margin", type=float, default=0.70)

    p.add_argument("--score_overall_weight", type=float, default=1.0)
    p.add_argument("--score_negative_gain_weight", type=float, default=0.020)
    p.add_argument("--score_edge_gain_weight", type=float, default=0.020)
    p.add_argument("--score_transition_gain_weight", type=float, default=0.010)
    p.add_argument("--score_high_penalty", type=float, default=3.00)
    p.add_argument("--high_tolerance", type=float, default=0.05)
    p.add_argument("--score_changed_high_penalty", type=float, default=0.015)
    p.add_argument("--score_changed_nonultra_penalty", type=float, default=0.006)

    p.add_argument("--xgb_jobs", type=int, default=-1)
    p.add_argument("--quality_estimators", type=int, default=280)
    p.add_argument("--quality_max_depth", type=int, default=4)
    p.add_argument("--quality_learning_rate", type=float, default=0.04)
    p.add_argument("--quality_subsample", type=float, default=0.90)
    p.add_argument("--quality_colsample", type=float, default=0.85)
    p.add_argument("--quality_chunk_size", type=int, default=32768)
    p.add_argument("--data_path", type=str, default=common.relpath("raw_data", "RML2016.10a_dict.pkl"))
    p.add_argument("--cache_dir", type=str, default=common.relpath("feature_cache"))
    p.add_argument("--alignment_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--skip_alignment_check", action="store_true")

    p.add_argument("--save_top_records", type=int, default=220)
    p.add_argument("--cm_snrs", type=int, nargs="+", default=[0, -6])
    p.add_argument(
        "--report_test_oracle_diagnostic",
        action="store_true",
        help="Post-hoc diagnostic only. Keep disabled for strict paper-style runs.",
    )
    return p.parse_args()


def snr_to_quality_bin(snrs):
    snrs = np.asarray(snrs, dtype=np.int32)
    out = np.zeros(len(snrs), dtype=np.int64)
    out[(snrs >= -14) & (snrs <= -8)] = 1
    out[(snrs >= -6) & (snrs <= -2)] = 2
    out[(snrs >= 0) & (snrs <= 8)] = 3
    out[snrs >= 10] = 4
    return out


def _safe_stats(x, axis=1):
    return [
        x.mean(axis=axis),
        x.std(axis=axis),
        np.percentile(x, 10, axis=axis),
        np.percentile(x, 50, axis=axis),
        np.percentile(x, 90, axis=axis),
    ]


def extract_blind_quality_features(full_dataset, indices, chunk_size=32768):
    data = np.asarray(full_dataset.data, dtype=np.float32)
    hos = getattr(full_dataset, "hos_data", None)
    if hos is not None:
        hos = np.asarray(hos, dtype=np.float32)

    indices = np.asarray(indices, dtype=np.int64)
    chunks = []
    for start in range(0, len(indices), int(chunk_size)):
        idx = indices[start : start + int(chunk_size)]
        x = data[idx]
        iq = x[:, [1, 2], :] if x.shape[1] >= 3 else x[:, :2, :]
        i = iq[:, 0, :].astype(np.float32)
        q = iq[:, 1, :].astype(np.float32)
        power = i * i + q * q
        amp = np.sqrt(power + 1e-12)
        phase = np.unwrap(np.arctan2(q, i), axis=1)
        dphase = np.diff(phase, axis=1)

        complex_iq = i.astype(np.complex64) + 1j * q.astype(np.complex64)
        spec = np.abs(np.fft.fft(complex_iq, axis=1)).astype(np.float32) ** 2
        spec_sum = spec.sum(axis=1, keepdims=True) + 1e-12
        spec_n = spec / spec_sum
        spec_entropy = -(spec_n * np.log(spec_n + 1e-12)).sum(axis=1) / np.log(spec.shape[1])
        spec_peak = spec.max(axis=1) / (spec.mean(axis=1) + 1e-12)
        spec_low = spec_n[:, : spec.shape[1] // 8].sum(axis=1) + spec_n[:, -spec.shape[1] // 8 :].sum(axis=1)
        spec_mid = spec_n[:, spec.shape[1] // 8 : spec.shape[1] // 2].sum(axis=1)

        corr = ((i - i.mean(axis=1, keepdims=True)) * (q - q.mean(axis=1, keepdims=True))).mean(axis=1)
        corr = corr / (i.std(axis=1) * q.std(axis=1) + 1e-8)
        p_i = (i * i).mean(axis=1)
        p_q = (q * q).mean(axis=1)

        windows = np.array_split(power, 8, axis=1)
        w_mean = np.stack([w.mean(axis=1) for w in windows], axis=1)
        w_std = np.stack([w.std(axis=1) for w in windows], axis=1)

        scalar = np.stack(
            [
                *_safe_stats(i),
                *_safe_stats(q),
                *_safe_stats(power),
                *_safe_stats(amp),
                dphase.mean(axis=1),
                dphase.std(axis=1),
                np.sin(dphase).mean(axis=1),
                np.cos(dphase).mean(axis=1),
                corr,
                p_i,
                p_q,
                p_i / (p_q + 1e-8),
                spec_entropy,
                spec_peak,
                spec_low,
                spec_mid,
                amp.max(axis=1),
                power.max(axis=1),
                power.std(axis=1) / (power.mean(axis=1) + 1e-8),
            ],
            axis=1,
        ).astype(np.float32)
        block = [scalar, w_mean.astype(np.float32), w_std.astype(np.float32)]
        if hos is not None:
            block.append(hos[idx].astype(np.float32))
        chunks.append(np.concatenate(block, axis=1))

    out = np.concatenate(chunks, axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def make_quality_model(args):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        n_estimators=int(args.quality_estimators),
        max_depth=int(args.quality_max_depth),
        learning_rate=float(args.quality_learning_rate),
        subsample=float(args.quality_subsample),
        colsample_bytree=float(args.quality_colsample),
        reg_lambda=2.0,
        reg_alpha=0.05,
        min_child_weight=2.0,
        tree_method="hist",
        device=str(getattr(args, "xgb_device", "cpu")),
        eval_metric="mlogloss",
        n_jobs=int(args.xgb_jobs),
        random_state=int(args.random_state + 404),
    )


def quality_probability_features(prob):
    prob = gr.norm(prob)
    centers = np.asarray([-18.0, -11.0, -4.0, 4.0, 14.0], dtype=np.float32)
    expected = prob @ centers
    low = prob[:, 0] + prob[:, 1]
    transition = prob[:, 2]
    high = prob[:, 3] + prob[:, 4]
    pred = prob.argmax(axis=1)
    block = np.concatenate(
        [
            prob.astype(np.float32),
            multi.one_hot(pred, 5),
            np.stack(
                [
                    prob.max(axis=1),
                    gr.margin(prob),
                    gr.entropy(prob),
                    expected,
                    low,
                    transition,
                    high,
                    prob[:, 0],
                    prob[:, 4],
                ],
                axis=1,
            ).astype(np.float32),
        ],
        axis=1,
    )
    return np.nan_to_num(block, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def build_blind_quality_context(args, soup):
    print("\n" + "=" * 144)
    print("[*] Building blind CQI/SNR-bin estimator from training-split I/Q features")
    print("=" * 144)
    full_dataset = common.build_full_dataset(args)
    train_idx, val_idx, test_idx = common.make_aligned_split(full_dataset, args.split_seed)
    if not args.skip_alignment_check:
        common.check_alignment(full_dataset, val_idx, test_idx, args.alignment_cache)

    labels = np.asarray(full_dataset.labels, dtype=np.int64)
    snrs = np.asarray(full_dataset.snrs, dtype=np.int32)
    if not np.all(labels[val_idx] == soup["labels_val"]) or not np.all(snrs[val_idx] == soup["snrs_val"]):
        raise RuntimeError("Blind quality val split is not aligned with Fourier soup cache.")
    if not np.all(labels[test_idx] == soup["labels_test"]) or not np.all(snrs[test_idx] == soup["snrs_test"]):
        raise RuntimeError("Blind quality test split is not aligned with Fourier soup cache.")

    x_train = extract_blind_quality_features(full_dataset, train_idx, args.quality_chunk_size)
    x_val = extract_blind_quality_features(full_dataset, val_idx, args.quality_chunk_size)
    x_test = extract_blind_quality_features(full_dataset, test_idx, args.quality_chunk_size)
    y_train = snr_to_quality_bin(snrs[train_idx])

    qclf = make_quality_model(args)
    qclf.fit(x_train, y_train)
    q_val_prob = gr.norm(qclf.predict_proba(x_val).astype(np.float32))
    q_test_prob = gr.norm(qclf.predict_proba(x_test).astype(np.float32))

    qv_true = snr_to_quality_bin(snrs[val_idx])
    qv_acc = float((q_val_prob.argmax(1) == qv_true).mean() * 100.0)
    print(f"Blind CQI val bin accuracy:  {qv_acc:.3f}%")
    qt_true = snr_to_quality_bin(snrs[test_idx])
    qt_acc = None
    if bool(getattr(args, "report_test_oracle_diagnostic", False)):
        qt_acc = float((q_test_prob.argmax(1) == qt_true).mean() * 100.0)
        print(f"Blind CQI test bin accuracy: {qt_acc:.3f}% (post-hoc diagnostic only; not used for inference)")

    qv_feat = quality_probability_features(q_val_prob)
    qt_feat = quality_probability_features(q_test_prob)
    out_path = relpath("results", f"blind_quality_cqi_split{args.split_seed}_probs_for_fusion.npz")
    np.savez_compressed(
        out_path,
        val_quality_prob=q_val_prob.astype(np.float32),
        test_quality_prob=q_test_prob.astype(np.float32),
        labels_val=soup["labels_val"].astype(np.int64),
        snrs_val=soup["snrs_val"].astype(np.int32),
        labels_test=soup["labels_test"].astype(np.int64),
        snrs_test=soup["snrs_test"].astype(np.int32),
        val_bin_true=qv_true.astype(np.int64),
        test_bin_true=qt_true.astype(np.int64),
    )
    print(f"[*] Blind CQI probabilities saved: {out_path}")
    return qv_feat, qt_feat, q_val_prob, q_test_prob, {"val_bin_acc": qv_acc, "test_bin_acc": qt_acc, "cache": out_path}


def save_csv(path, rows, n=None):
    if n is not None:
        rows = rows[: max(1, int(n))]
    if not rows:
        return
    keys = sorted(set(k for row in rows for k in row.keys()))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[*] CSV saved: {path}")


def print_top(rows, n=20):
    print("\nTop blind-quality XGB-stacked validation configs")
    for i, r in enumerate(rows[:n], 1):
        print(
            f"{i:02d}. {r['branch']:<18} score={r['score']:.3f} | overall={r['overall_acc']:.3f}% | "
            f"neg={r['negative_acc']:.3f}% | edge={r['edge_low_acc']:.3f}% | high={r['high_acc']:.3f}% | "
            f"alpha={r['alpha']} conf={r['meta_conf_thr']} adv={r['advantage_thr']} maxchg={r['max_change_rate']}"
        )


def main():
    args = parse_args()
    os.makedirs(relpath("results"), exist_ok=True)
    suffix = args.output_suffix or f"fourier_gamc_multi_cvtrn_blind_quality_xgb_split{args.split_seed}"

    print("=" * 144)
    print("Blind quality-aware Multi-CVTRN XGBoost stacked residual fusion")
    print("=" * 144)
    print("Academic protocol:")
    print("  - Fourier soup remains the main model.")
    print("  - GAMC and all CV-TRN models are auxiliary probability experts.")
    print("  - Blind CQI/SNR-bin estimator is trained only on train-split I/Q-derived features.")
    print("  - The true SNR is never used as an inference feature.")
    print("  - XGB stackers are trained only on validation labels with OOF validation predictions.")
    print("  - Test labels are used only once for the final report.")

    soup = base.load_soup_cache(args.soup_prob_cache)
    gamc = base.load_gamc_cache(args.gamc_cache)
    cvtrn_items = []
    for path in args.cvtrn_caches:
        item = multi.load_cvtrn_cache(path)
        item["path"] = path
        cvtrn_items.append(item)
    multi.assert_multi_alignment(soup, gamc, cvtrn_items)

    yv, sv = soup["labels_val"], soup["snrs_val"]
    yt, st_snrs = soup["labels_test"], soup["snrs_test"]
    class_names = soup["mod_classes"]
    nv_raw, nt_raw = soup["val_prob"], soup["test_prob"]
    gv_raw, gt_raw = gamc["val_prob"], gamc["test_prob"]
    rv, rt = gamc["val_router"], gamc["test_router"]

    cv_val_raw = [item["val_prob"] for item in cvtrn_items]
    cv_test_raw = [item["test_prob"] for item in cvtrn_items]
    if args.disable_temperature_scaling:
        tn = tg = 1.0
        cv_temps = [1.0] * len(cvtrn_items)
    else:
        tn, _ = base.fit_temperature(nv_raw, yv, args.temperature_grid, "Fourier soup")
        tg, _ = base.fit_temperature(gv_raw, yv, args.temperature_grid, "GAMC")
        cv_temps = []
        for i, p in enumerate(cv_val_raw, 1):
            tc, _ = base.fit_temperature(p, yv, args.temperature_grid, f"CV-TRN #{i}")
            cv_temps.append(float(tc))

    nv = base.temperature_scale_probs(nv_raw, tn)
    nt = base.temperature_scale_probs(nt_raw, tn)
    gv = base.temperature_scale_probs(gv_raw, tg)
    gt = base.temperature_scale_probs(gt_raw, tg)
    cv_val_list = [base.temperature_scale_probs(p, t) for p, t in zip(cv_val_raw, cv_temps)]
    cv_test_list = [base.temperature_scale_probs(p, t) for p, t in zip(cv_test_raw, cv_temps)]
    cv_val_mean = multi.log_average(cv_val_list)
    cv_test_mean = multi.log_average(cv_test_list)

    qv_feat, qt_feat, qv_prob, qt_prob, qinfo = build_blind_quality_context(args, soup)

    base_val_m = base.metrics_from_probs(nv, yv, sv)
    base_test_m = base.metrics_from_probs(nt, yt, st_snrs)
    print("\nBaselines")
    base.print_metrics_line("Fourier Val", base_val_m)
    base.print_metrics_line("GAMC Val", base.metrics_from_probs(gv, yv, sv))
    for i, p in enumerate(cv_val_list, 1):
        base.print_metrics_line(f"CV-TRN #{i} Val", base.metrics_from_probs(p, yv, sv))
    base.print_metrics_line("CV-TRN mean Val", base.metrics_from_probs(cv_val_mean, yv, sv))

    pool_args = argparse.Namespace(
        mix_alphas=args.mix_alphas,
        max_candidates=int(args.feature_candidates),
        min_candidate_rescues=int(args.min_candidate_rescues),
    )
    val_full = gr.candidate_pool(nv, gv, cv_val_mean, pool_args)
    test_full = gr.candidate_pool(nt, gt, cv_test_mean, pool_args)
    val_pool, test_pool, kept_rows, all_rows = gr.prune_pool(val_full, test_full, yv, pool_args)
    print("\nMeta-feature candidate pool built from CV-TRN mean")
    print(f"Full validation oracle:   {gr.pool_oracle_acc(val_full, yv):.3f}%")
    print(f"Feature-pool oracle:      {gr.pool_oracle_acc(val_pool, yv):.3f}%")
    if args.report_test_oracle_diagnostic:
        print(f"Full test oracle:         {gr.pool_oracle_acc(test_full, yt):.3f}%")
        print(f"Feature-pool test oracle: {gr.pool_oracle_acc(test_pool, yt):.3f}%")
    for rec, row in zip(val_pool, kept_rows):
        print(
            f"  {rec['candidate_id']:02d} <- {rec['source_candidate_id']:02d} "
            f"{rec['name']:<16} acc={row['acc']:.3f}% rescue={row['rescue']} harm={row['harm']}"
        )
    save_csv(relpath("results", f"{suffix}_candidate_summary.csv"), all_rows)

    x_val_base = multi.build_multi_features(nv, gv, cv_val_list, cv_val_mean, rv, val_pool)
    x_test_base = multi.build_multi_features(nt, gt, cv_test_list, cv_test_mean, rt, test_pool)
    x_val = np.concatenate([x_val_base, qv_feat], axis=1).astype(np.float32)
    x_test = np.concatenate([x_test_base, qt_feat], axis=1).astype(np.float32)
    print(f"Feature dim: {x_val.shape[1]} | CV-TRN experts: {len(cvtrn_items)} | blind CQI dim: {qv_feat.shape[1]}")

    branches = []
    all_records = []
    print("\n" + "=" * 144)
    print("[*] Training OOF blind-quality Multi-CVTRN XGBoost stackers")
    print("=" * 144)
    for cfg in multi.xgb_configs(args):
        print(f"    OOF XGB stacker: {cfg['name']}")
        oof = multi.fit_oof_xgb(x_val, yv, cfg, args)
        records = st.search_configs(nv, oof, yv, sv, base_val_m, args, cfg["name"])
        branches.append({"name": cfg["name"], "cfg": cfg, "oof": oof, "records": records})
        all_records.extend(records)

    ens_oof = gr.norm(np.mean([b["oof"] for b in branches], axis=0))
    ens_records = st.search_configs(nv, ens_oof, yv, sv, base_val_m, args, "blind_quality_xgb_ensemble")
    branches.append({"name": "blind_quality_xgb_ensemble", "cfg": None, "oof": ens_oof, "records": ens_records})
    all_records.extend(ens_records)
    all_records.sort(key=lambda r: r["score"], reverse=True)
    print_top(all_records)

    best = all_records[0]
    selected_branch = best["branch"]
    print(f"\n[*] Selected validation branch: {selected_branch}")
    if selected_branch == "blind_quality_xgb_ensemble":
        test_probs = [multi.fit_final_xgb(x_val, yv, x_test, b["cfg"], args) for b in branches if b["cfg"] is not None]
        meta_test = gr.norm(np.mean(test_probs, axis=0))
    else:
        branch = next(b for b in branches if b["name"] == selected_branch)
        meta_test = multi.fit_final_xgb(x_val, yv, x_test, branch["cfg"], args)

    final_prob, gate, use, alpha_vec = st.apply_stacked(nt, meta_test, best, args)
    final_m = base.metrics_from_probs(final_prob, yt, st_snrs)
    final_diag = base.switch_diagnostics(nt, final_prob, gate, use, alpha_vec, st_snrs)

    selected = {
        "best": best,
        "temperature": {
            "fourier_T": float(tn),
            "gamc_T": float(tg),
            "cvtrn_T": [float(x) for x in cv_temps],
        },
        "cvtrn_caches": list(args.cvtrn_caches),
        "quality": qinfo,
        "feature_candidates": [
            {
                "candidate_id": int(rec["candidate_id"]),
                "source_candidate_id": int(rec.get("source_candidate_id", rec["candidate_id"])),
                "name": rec["name"],
                "kind": rec["kind"],
                "alpha": float(rec["alpha"]),
            }
            for rec in val_pool
        ],
        "method": "Train-split blind-CQI + OOF validation-trained Multi-CVTRN XGBoost stacked residual fusion",
        "test_oracle_diagnostic_reported": bool(args.report_test_oracle_diagnostic),
    }

    print("\n" + "=" * 144)
    print("Final test report")
    print("=" * 144)
    base.print_metrics_line("Fourier soup Test", base_test_m)
    base.print_metrics_line("GAMC Test", base.metrics_from_probs(gt, yt, st_snrs))
    for i, p in enumerate(cv_test_list, 1):
        base.print_metrics_line(f"CV-TRN #{i} Test", base.metrics_from_probs(p, yt, st_snrs))
    base.print_metrics_line("CV-TRN mean Test", base.metrics_from_probs(cv_test_mean, yt, st_snrs))
    base.print_metrics_line("Blind-quality XGB stacked Test", final_m)
    print("-" * 144)
    print(f"Delta vs Fourier overall:    {final_m['overall_acc'] - base_test_m['overall_acc']:+.4f} pp")
    print(f"Delta vs Fourier negative:   {final_m['negative_acc'] - base_test_m['negative_acc']:+.4f} pp")
    print(f"Delta vs Fourier edge:       {final_m['edge_low_acc'] - base_test_m['edge_low_acc']:+.4f} pp")
    print(f"Delta vs Fourier transition: {final_m['transition_acc'] - base_test_m['transition_acc']:+.4f} pp")
    print(f"Delta vs Fourier high:       {final_m['high_acc'] - base_test_m['high_acc']:+.4f} pp")
    print(f"Final diagnostics: {final_diag}")
    print(f"Blind CQI diagnostics: {qinfo}")
    print("=" * 144)
    base.print_snr_table(final_m["by_snr"])

    save_csv(relpath("results", f"{suffix}_search_top.csv"), all_records, args.save_top_records)
    config_path = relpath("results", f"{suffix}_selected_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"[*] Selected config saved: {config_path}")

    pred_path = relpath("results", f"{suffix}_predictions.npz")
    np.savez_compressed(
        pred_path,
        labels=yt.astype(np.int64),
        snrs=st_snrs.astype(np.int32),
        pred=final_m["pred"].astype(np.int64),
        final_prob=final_prob.astype(np.float32),
        fourier_prob=nt.astype(np.float32),
        gamc_prob=gt.astype(np.float32),
        cvtrn_mean_prob=cv_test_mean.astype(np.float32),
        blind_quality_prob=qt_prob.astype(np.float32),
        xgb_meta_prob=meta_test.astype(np.float32),
        gate=gate.astype(bool),
        use_candidate=use.astype(bool),
        selected_config=np.asarray([json.dumps(selected, ensure_ascii=False)]),
        mod_classes=np.asarray(class_names),
    )
    print(f"[*] Predictions saved: {pred_path}")

    curve_path = relpath("results", f"accuracy_vs_snr_{suffix}.png")
    base.plot_curve(final_m["by_snr"], curve_path, "Blind quality-aware Multi-CVTRN XGBoost fusion")
    print(f"[*] SNR curve saved: {curve_path}")
    for snr_value in args.cm_snrs:
        cm_path = relpath("results", f"confusion_matrix_{snr_value}dB_{suffix}.png")
        acc = base.plot_cm_at_snr(
            labels=yt,
            pred=final_m["pred"],
            snrs=st_snrs,
            mod_classes=class_names,
            target_snr=int(snr_value),
            save_path=cm_path,
            title_prefix="Blind quality-aware Multi-CVTRN XGBoost fusion",
        )
        print(f"[*] {snr_value} dB confusion matrix saved: {cm_path} (Acc={acc:.2f}%)")


if __name__ == "__main__":
    main()

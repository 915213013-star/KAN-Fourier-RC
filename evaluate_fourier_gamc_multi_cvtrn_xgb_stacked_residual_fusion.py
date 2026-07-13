import argparse
import csv
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import evaluate_fourier_soup_gamc_cvtrn_protected_residual_fusion as tri
import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import evaluate_fourier_gamc_cvtrn_gain_risk_router_fusion as gr
import evaluate_fourier_gamc_cvtrn_stacked_residual_fusion as st


def project_root():
    return os.path.dirname(os.path.abspath(__file__))


def relpath(*parts):
    return os.path.join(project_root(), *parts)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Multi-CVTRN XGBoost stacked residual fusion for Fourier soup + GAMC + multiple CV-TRN experts."
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
        default=[relpath("results", "cv_trn_aux_v2_soup_tta_split1_valtest_probs_for_fusion.npz")],
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
    p.add_argument("--save_top_records", type=int, default=180)
    p.add_argument("--cm_snrs", type=int, nargs="+", default=[0, -6])
    p.add_argument(
        "--report_test_oracle_diagnostic",
        action="store_true",
        help="Post-hoc diagnostic only. Keep disabled for strict paper-style runs.",
    )
    return p.parse_args()


def load_cvtrn_cache(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CV-TRN cache not found: {path}")
    return tri.load_cvtrn_cache(path)


def assert_multi_alignment(soup, gamc, cvtrn_items):
    base.assert_alignment(soup, gamc)
    for item in cvtrn_items:
        for key in ("labels_val", "labels_test", "snrs_val", "snrs_test"):
            if not np.all(soup[key] == item[key]):
                raise RuntimeError(f"Alignment check failed for {item['path']}: {key}")
    print("[*] Alignment check passed: neural soup, GAMC, and all CV-TRN caches are identical.")


def log_average(probs):
    logs = [np.log(np.clip(gr.norm(p), 1e-12, 1.0)) for p in probs]
    z = np.mean(np.stack(logs, axis=0), axis=0)
    z -= z.max(axis=1, keepdims=True)
    return gr.norm(np.exp(z))


def xgb_configs(args):
    base_kwargs = {
        "objective": "multi:softprob",
        "num_class": base.NUM_CLASSES,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "n_jobs": int(args.xgb_jobs),
    }
    return [
        {
            "name": "xgb_d2_220",
            "seed_offset": 17,
            "kwargs": {
                **base_kwargs,
                "n_estimators": 220,
                "max_depth": 2,
                "learning_rate": 0.035,
                "subsample": 0.90,
                "colsample_bytree": 0.80,
                "reg_lambda": 2.0,
                "reg_alpha": 0.0,
            },
        },
        {
            "name": "xgb_d3_240",
            "seed_offset": 31,
            "kwargs": {
                **base_kwargs,
                "n_estimators": 240,
                "max_depth": 3,
                "learning_rate": 0.030,
                "subsample": 0.90,
                "colsample_bytree": 0.80,
                "reg_lambda": 2.0,
                "reg_alpha": 0.05,
            },
        },
        {
            "name": "xgb_d4_200",
            "seed_offset": 47,
            "kwargs": {
                **base_kwargs,
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.030,
                "subsample": 0.85,
                "colsample_bytree": 0.75,
                "reg_lambda": 3.0,
                "reg_alpha": 0.10,
            },
        },
    ]


def make_xgb(cfg, seed):
    return XGBClassifier(random_state=int(seed), **cfg["kwargs"])


def one_hot(x, n):
    return np.eye(n, dtype=np.float32)[x.astype(np.int64)]


def expert_stats(main, expert, prefix_router=None):
    f, p = gr.norm(main), gr.norm(expert)
    fp, pp = f.argmax(1), p.argmax(1)
    idx = np.arange(len(f))
    parts = [
        p.astype(np.float32),
        np.log(np.clip(p, 1e-8, 1.0)).astype(np.float32),
        (p - f).astype(np.float32),
        one_hot(pp, base.NUM_CLASSES),
        np.stack(
            [
                p.max(1),
                gr.margin(p),
                gr.entropy(p),
                (pp == fp).astype(np.float32),
                np.abs(p - f).sum(1),
                (p * f).sum(1),
                p[idx, fp],
                f[idx, pp],
            ],
            axis=1,
        ).astype(np.float32),
    ]
    return parts


def build_multi_features(fourier, gamc, cvtrn_list, cvtrn_mean, router, feature_pool):
    f, g, cm, r = gr.norm(fourier), gr.norm(gamc), gr.norm(cvtrn_mean), gr.norm(router)
    fp, gp, cp, rp = f.argmax(1), g.argmax(1), cm.argmax(1), r.argmax(1)
    idx = np.arange(len(f))
    low1, high, gap1 = base.router_low_high(r, 1)
    low2, _, gap2 = base.router_low_high(r, 2)
    scalar = np.stack(
        [
            f.max(1), g.max(1), cm.max(1), r.max(1),
            gr.margin(f), gr.margin(g), gr.margin(cm), gr.margin(r),
            gr.entropy(f), gr.entropy(g), gr.entropy(cm), gr.entropy(r),
            (fp == gp).astype(np.float32), (fp == cp).astype(np.float32), (gp == cp).astype(np.float32),
            np.abs(f - g).sum(1), np.abs(f - cm).sum(1), np.abs(g - cm).sum(1),
            (f * g).sum(1), (f * cm).sum(1), (g * cm).sum(1),
            g[idx, fp], cm[idx, fp], f[idx, gp], f[idx, cp], g[idx, cp], cm[idx, gp],
            low1, low2, high, gap1, gap2,
        ],
        axis=1,
    ).astype(np.float32)
    parts = [
        scalar,
        f.astype(np.float32),
        g.astype(np.float32),
        cm.astype(np.float32),
        r.astype(np.float32),
        np.log(np.clip(f, 1e-8, 1.0)).astype(np.float32),
        np.log(np.clip(g, 1e-8, 1.0)).astype(np.float32),
        np.log(np.clip(cm, 1e-8, 1.0)).astype(np.float32),
        one_hot(fp, base.NUM_CLASSES),
        one_hot(gp, base.NUM_CLASSES),
        one_hot(cp, base.NUM_CLASSES),
        one_hot(rp, r.shape[1]),
    ]
    for p in cvtrn_list:
        parts.extend(expert_stats(f, p))
    for rec in feature_pool:
        parts.extend(expert_stats(f, rec["prob"]))
    if len(cvtrn_list) > 1:
        stack = np.stack([gr.norm(p) for p in cvtrn_list], axis=0)
        parts.append(stack.std(axis=0).astype(np.float32))
        parts.append(stack.max(axis=0).astype(np.float32))
        parts.append(stack.min(axis=0).astype(np.float32))
    x = np.concatenate(parts, axis=1)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def fit_oof_xgb(x, y, cfg, args):
    oof = np.zeros((len(y), base.NUM_CLASSES), dtype=np.float32)
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.random_state))
    for fold, (tr, va) in enumerate(skf.split(x, y), 1):
        clf = make_xgb(cfg, args.random_state + int(cfg["seed_offset"]) + 97 * fold)
        clf.fit(x[tr], y[tr])
        oof[va] = clf.predict_proba(x[va]).astype(np.float32)
        print(f"      fold {fold}/{args.folds} done")
    return gr.norm(oof)


def fit_final_xgb(x, y, xt, cfg, args):
    clf = make_xgb(cfg, args.random_state + int(cfg["seed_offset"]) + 999)
    clf.fit(x, y)
    return gr.norm(clf.predict_proba(xt).astype(np.float32))


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
    print("\nTop multi-CVTRN XGB-stacked validation configs")
    for i, r in enumerate(rows[:n], 1):
        print(
            f"{i:02d}. {r['branch']:<18} score={r['score']:.3f} | overall={r['overall_acc']:.3f}% | "
            f"neg={r['negative_acc']:.3f}% | edge={r['edge_low_acc']:.3f}% | high={r['high_acc']:.3f}% | "
            f"alpha={r['alpha']} conf={r['meta_conf_thr']} adv={r['advantage_thr']} maxchg={r['max_change_rate']}"
        )


def main():
    args = parse_args()
    os.makedirs(relpath("results"), exist_ok=True)
    suffix = args.output_suffix or f"fourier_gamc_multi_cvtrn_xgb_stacked_residual_split{args.split_seed}"

    print("=" * 144)
    print("Multi-CVTRN XGBoost stacked residual fusion")
    print("=" * 144)
    print("Academic protocol:")
    print("  - Fourier soup remains the main model.")
    print("  - GAMC and all CV-TRN models are auxiliary probability experts.")
    print("  - XGB stackers are trained only on validation labels.")
    print("  - OOF validation predictions select blend/gate parameters.")
    print("  - Test labels are used only once for the final report.")
    print("  - True SNR is never used as an inference feature.")

    soup = base.load_soup_cache(args.soup_prob_cache)
    gamc = base.load_gamc_cache(args.gamc_cache)
    cv_items = []
    for path in args.cvtrn_caches:
        item = load_cvtrn_cache(path)
        item["path"] = path
        cv_items.append(item)
    assert_multi_alignment(soup, gamc, cv_items)

    yv, sv = soup["labels_val"], soup["snrs_val"]
    yt, st_snrs = soup["labels_test"], soup["snrs_test"]
    class_names = soup["mod_classes"]
    nv_raw, nt_raw = soup["val_prob"], soup["test_prob"]
    gv_raw, gt_raw = gamc["val_prob"], gamc["test_prob"]
    rv, rt = gamc["val_router"], gamc["test_router"]

    if args.disable_temperature_scaling:
        tn = tg = 1.0
        tc_list = [1.0 for _ in cv_items]
    else:
        tn, _ = base.fit_temperature(nv_raw, yv, args.temperature_grid, "Fourier soup")
        tg, _ = base.fit_temperature(gv_raw, yv, args.temperature_grid, "GAMC")
        tc_list = []
        for i, item in enumerate(cv_items, 1):
            tc, _ = base.fit_temperature(item["val_prob"], yv, args.temperature_grid, f"CV-TRN #{i}")
            tc_list.append(tc)

    nv = base.temperature_scale_probs(nv_raw, tn)
    nt = base.temperature_scale_probs(nt_raw, tn)
    gv = base.temperature_scale_probs(gv_raw, tg)
    gt = base.temperature_scale_probs(gt_raw, tg)
    cv_val_list = [base.temperature_scale_probs(item["val_prob"], t) for item, t in zip(cv_items, tc_list)]
    cv_test_list = [base.temperature_scale_probs(item["test_prob"], t) for item, t in zip(cv_items, tc_list)]
    cv_val_mean = log_average(cv_val_list)
    cv_test_mean = log_average(cv_test_list)

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

    x_val = build_multi_features(nv, gv, cv_val_list, cv_val_mean, rv, val_pool)
    x_test = build_multi_features(nt, gt, cv_test_list, cv_test_mean, rt, test_pool)
    print(f"Feature dim: {x_val.shape[1]} | CV-TRN experts: {len(cv_val_list)}")

    branches = []
    all_records = []
    print("\n" + "=" * 144)
    print("[*] Training OOF multi-CVTRN XGBoost stacked meta-classifiers")
    print("=" * 144)
    for cfg in xgb_configs(args):
        print(f"    OOF XGB stacker: {cfg['name']}")
        oof = fit_oof_xgb(x_val, yv, cfg, args)
        records = st.search_configs(nv, oof, yv, sv, base_val_m, args, cfg["name"])
        branches.append({"name": cfg["name"], "cfg": cfg, "oof": oof, "records": records})
        all_records.extend(records)

    ens_oof = gr.norm(np.mean([b["oof"] for b in branches], axis=0))
    ens_records = st.search_configs(nv, ens_oof, yv, sv, base_val_m, args, "multi_xgb_ensemble")
    branches.append({"name": "multi_xgb_ensemble", "cfg": None, "oof": ens_oof, "records": ens_records})
    all_records.extend(ens_records)
    all_records.sort(key=lambda r: r["score"], reverse=True)
    print_top(all_records)

    best = all_records[0]
    selected_branch = best["branch"]
    print(f"\n[*] Selected validation branch: {selected_branch}")
    if selected_branch == "multi_xgb_ensemble":
        test_probs = [fit_final_xgb(x_val, yv, x_test, b["cfg"], args) for b in branches if b["cfg"] is not None]
        meta_test = gr.norm(np.mean(test_probs, axis=0))
    else:
        branch = next(b for b in branches if b["name"] == selected_branch)
        meta_test = fit_final_xgb(x_val, yv, x_test, branch["cfg"], args)

    final_prob, gate, use, alpha_vec = st.apply_stacked(nt, meta_test, best, args)
    final_m = base.metrics_from_probs(final_prob, yt, st_snrs)
    final_diag = base.switch_diagnostics(nt, final_prob, gate, use, alpha_vec, st_snrs)

    selected = {
        "best": best,
        "temperature": {
            "fourier_T": float(tn),
            "gamc_T": float(tg),
            "cvtrn_T": [float(x) for x in tc_list],
        },
        "cvtrn_caches": args.cvtrn_caches,
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
        "method": "OOF validation-trained multi-CVTRN XGBoost stacked residual meta-classifier",
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
    base.print_metrics_line("Multi-CVTRN XGB stacked Test", final_m)
    print("-" * 144)
    print(f"Delta vs Fourier overall:    {final_m['overall_acc'] - base_test_m['overall_acc']:+.4f} pp")
    print(f"Delta vs Fourier negative:   {final_m['negative_acc'] - base_test_m['negative_acc']:+.4f} pp")
    print(f"Delta vs Fourier edge:       {final_m['edge_low_acc'] - base_test_m['edge_low_acc']:+.4f} pp")
    print(f"Delta vs Fourier transition: {final_m['transition_acc'] - base_test_m['transition_acc']:+.4f} pp")
    print(f"Delta vs Fourier high:       {final_m['high_acc'] - base_test_m['high_acc']:+.4f} pp")
    print(f"Final diagnostics: {final_diag}")
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
        xgb_meta_prob=meta_test.astype(np.float32),
        gate=gate.astype(bool),
        use_candidate=use.astype(bool),
        selected_config=np.asarray([json.dumps(selected, ensure_ascii=False)]),
        mod_classes=np.asarray(class_names),
    )
    print(f"[*] Predictions saved: {pred_path}")

    curve_path = relpath("results", f"accuracy_vs_snr_{suffix}.png")
    base.plot_curve(final_m["by_snr"], curve_path, "Multi-CVTRN XGBoost stacked residual fusion")
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
            title_prefix="Multi-CVTRN XGBoost stacked residual fusion",
        )
        print(f"[*] {snr_value} dB confusion matrix saved: {cm_path} (Acc={acc:.2f}%)")


if __name__ == "__main__":
    main()

import argparse
import copy
import csv
import json
import os

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

import evaluate_fourier_soup_gamc_cvtrn_protected_residual_fusion as tri
import evaluate_greedy_soup_gamc_protected_residual_fusion as base
import evaluate_fourier_gamc_cvtrn_gain_risk_router_fusion as gr


def project_root():
    return os.path.dirname(os.path.abspath(__file__))


def relpath(*parts):
    return os.path.join(project_root(), *parts)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stacked residual meta-classifier fusion for Fourier soup + GAMC + CV-TRN. "
            "Fourier remains the main model; the stacker is trained only on the held-out validation set."
        )
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)
    p.add_argument("--soup_prob_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--gamc_cache", type=str, default=relpath("results", "gamc_lite_v2_xgb_split1_valtest_probs_for_fusion.npz"))
    p.add_argument("--cvtrn_cache", type=str, default=relpath("results", "cv_trn_aux_v2_soup_tta_split1_valtest_probs_for_fusion.npz"))
    p.add_argument("--fallback_cvtrn_cache", type=str, default=relpath("results", "cv_trn_aux_split1_valtest_probs_for_fusion.npz"))
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

    p.add_argument("--save_top_records", type=int, default=160)
    p.add_argument("--cm_snrs", type=int, nargs="+", default=[0, -6])
    p.add_argument(
        "--report_test_oracle_diagnostic",
        action="store_true",
        help="Post-hoc diagnostic only. Keep disabled for strict paper-style runs.",
    )
    return p.parse_args()


def load_cvtrn(args):
    path = args.cvtrn_cache
    if not os.path.exists(path) and os.path.exists(args.fallback_cvtrn_cache):
        print(f"[!] CV-TRN-v2 cache missing, falling back to v1 cache: {args.fallback_cvtrn_cache}")
        path = args.fallback_cvtrn_cache
    return tri.load_cvtrn_cache(path), path


def one_hot(x, n):
    return np.eye(n, dtype=np.float32)[x.astype(np.int64)]


def build_features(fourier, gamc, cvtrn, router, feature_pool):
    f, g, c, r = gr.norm(fourier), gr.norm(gamc), gr.norm(cvtrn), gr.norm(router)
    fp, gp, cp, rp = f.argmax(1), g.argmax(1), c.argmax(1), r.argmax(1)
    idx = np.arange(len(f))
    low1, high, gap1 = base.router_low_high(r, 1)
    low2, _, gap2 = base.router_low_high(r, 2)

    scalar = np.stack(
        [
            f.max(1), g.max(1), c.max(1), r.max(1),
            gr.margin(f), gr.margin(g), gr.margin(c), gr.margin(r),
            gr.entropy(f), gr.entropy(g), gr.entropy(c), gr.entropy(r),
            (fp == gp).astype(np.float32), (fp == cp).astype(np.float32), (gp == cp).astype(np.float32),
            np.abs(f - g).sum(1), np.abs(f - c).sum(1), np.abs(g - c).sum(1),
            (f * g).sum(1), (f * c).sum(1), (g * c).sum(1),
            g[idx, fp], c[idx, fp], f[idx, gp], f[idx, cp], g[idx, cp], c[idx, gp],
            low1, low2, high, gap1, gap2,
        ],
        axis=1,
    ).astype(np.float32)

    cand_blocks = []
    for rec in feature_pool:
        p = gr.norm(rec["prob"])
        pp = p.argmax(1)
        cand_blocks.extend(
            [
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
        )

    x = np.concatenate(
        [
            scalar,
            f.astype(np.float32),
            g.astype(np.float32),
            c.astype(np.float32),
            r.astype(np.float32),
            np.log(np.clip(f, 1e-8, 1.0)).astype(np.float32),
            np.log(np.clip(g, 1e-8, 1.0)).astype(np.float32),
            np.log(np.clip(c, 1e-8, 1.0)).astype(np.float32),
            one_hot(fp, base.NUM_CLASSES),
            one_hot(gp, base.NUM_CLASSES),
            one_hot(cp, base.NUM_CLASSES),
            one_hot(rp, r.shape[1]),
            *cand_blocks,
        ],
        axis=1,
    )
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def stacker_configs():
    return [
        {
            "name": "et_depth8_leaf20",
            "seed_offset": 11,
            "model": ExtraTreesClassifier(
                n_estimators=220,
                max_depth=8,
                min_samples_leaf=20,
                max_features="sqrt",
                n_jobs=-1,
                random_state=101,
            ),
        },
        {
            "name": "et_depth12_leaf12",
            "seed_offset": 23,
            "model": ExtraTreesClassifier(
                n_estimators=240,
                max_depth=12,
                min_samples_leaf=12,
                max_features="sqrt",
                n_jobs=-1,
                random_state=102,
            ),
        },
        {
            "name": "et_depth16_leaf8",
            "seed_offset": 37,
            "model": ExtraTreesClassifier(
                n_estimators=260,
                max_depth=16,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=-1,
                random_state=103,
            ),
        },
    ]


def aligned_multiclass_proba(clf, x, n_classes):
    p = clf.predict_proba(x)
    out = np.zeros((x.shape[0], n_classes), dtype=np.float32)
    for j, cls in enumerate(clf.classes_):
        out[:, int(cls)] = p[:, j]
    return gr.norm(out)


def fit_oof_stacker(x, y, cfg, args):
    oof = np.zeros((len(y), base.NUM_CLASSES), dtype=np.float32)
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.random_state))
    for fold, (tr, va) in enumerate(skf.split(x, y), 1):
        clf = copy.deepcopy(cfg["model"])
        clf.random_state = int(args.random_state + 97 * fold + int(cfg.get("seed_offset", 0)))
        clf.fit(x[tr], y[tr])
        oof[va] = aligned_multiclass_proba(clf, x[va], base.NUM_CLASSES)
    return gr.norm(oof)


def fit_final_stacker(x, y, xt, cfg):
    clf = copy.deepcopy(cfg["model"])
    clf.fit(x, y)
    return aligned_multiclass_proba(clf, xt, base.NUM_CLASSES)


def logit_blend(main_prob, meta_prob, alpha):
    lf = base.prob_to_logits(gr.norm(main_prob))
    lm = base.prob_to_logits(gr.norm(meta_prob))
    return gr.norm(base.softmax_np(lf + float(alpha) * (lm - lf)))


def apply_stacked(main_prob, meta_prob, cfg, args):
    main = gr.norm(main_prob)
    meta = gr.norm(meta_prob)
    blended = logit_blend(main, meta, cfg["alpha"])
    main_pred = main.argmax(1)
    meta_pred = meta.argmax(1)
    blend_pred = blended.argmax(1)
    meta_conf = meta.max(1)
    advantage = meta[np.arange(len(meta)), meta_pred] - main[np.arange(len(main)), meta_pred]
    hard = (main.max(1) >= float(args.hard_keep_conf)) & (gr.margin(main) >= float(args.hard_keep_margin))
    gate = (
        (meta_conf >= float(cfg["meta_conf_thr"]))
        & (advantage >= float(cfg["advantage_thr"]))
        & (~hard)
    )
    final = main.copy()
    final[gate] = blended[gate]

    changed = final.argmax(1) != main_pred
    max_rate = float(cfg["max_change_rate"])
    if max_rate < 99.99 and changed.mean() * 100.0 > max_rate:
        changed_idx = np.where(changed)[0]
        keep_n = int(round(len(main) * max_rate / 100.0))
        score = meta_conf + advantage + 0.15 * (meta_pred != main_pred).astype(np.float32)
        order = np.argsort(score[changed_idx])[::-1]
        keep = changed_idx[order[:keep_n]]
        restricted = main.copy()
        restricted[keep] = final[keep]
        final = restricted
        gate = np.zeros(len(main), dtype=bool)
        gate[keep] = True

    final = gr.norm(final)
    use = final.argmax(1) != main_pred
    alpha_vec = np.full(len(main), float(cfg["alpha"]), dtype=np.float32)
    return final, gate.astype(bool), use.astype(bool), alpha_vec


def search_configs(main_prob, meta_prob, labels, snrs, base_m, args, branch_name):
    rows = []
    total = (
        len(args.blend_alphas)
        * len(args.meta_conf_thresholds)
        * len(args.advantage_thresholds)
        * len(args.max_change_rates)
    )
    done = 0
    for alpha in args.blend_alphas:
        for conf in args.meta_conf_thresholds:
            for adv in args.advantage_thresholds:
                for max_rate in args.max_change_rates:
                    done += 1
                    cfg = {
                        "branch": branch_name,
                        "alpha": float(alpha),
                        "meta_conf_thr": float(conf),
                        "advantage_thr": float(adv),
                        "max_change_rate": float(max_rate),
                    }
                    final, gate, use, alpha_vec = apply_stacked(main_prob, meta_prob, cfg, args)
                    m = base.metrics_from_probs(final, labels, snrs)
                    d = base.switch_diagnostics(main_prob, final, gate, use, alpha_vec, snrs)
                    rows.append(
                        {
                            "score": float(base.selection_score(m, d, base_m, args)),
                            "overall_acc": float(m["overall_acc"]),
                            "transition_acc": float(m["transition_acc"]),
                            "edge_low_acc": float(m["edge_low_acc"]),
                            "negative_acc": float(m["negative_acc"]),
                            "high_acc": float(m["high_acc"]),
                            "gate_rate": float(d["gate_rate"]),
                            "use_rate": float(d["use_rate"]),
                            "changed_high_rate": float(d["changed_high_rate"]),
                            "changed_nonultra_rate": float(d["changed_nonultra_rate"]),
                            **cfg,
                        }
                    )
    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]
    print(
        f"    searched {total:4d} configs for {branch_name:<18} | "
        f"best val={best['overall_acc']:.3f}% score={best['score']:.3f}"
    )
    return rows


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
    print("\nTop stacked-residual validation configs")
    for i, r in enumerate(rows[:n], 1):
        print(
            f"{i:02d}. {r['branch']:<18} score={r['score']:.3f} | overall={r['overall_acc']:.3f}% | "
            f"neg={r['negative_acc']:.3f}% | edge={r['edge_low_acc']:.3f}% | high={r['high_acc']:.3f}% | "
            f"alpha={r['alpha']} conf={r['meta_conf_thr']} adv={r['advantage_thr']} maxchg={r['max_change_rate']}"
        )


def main():
    args = parse_args()
    os.makedirs(relpath("results"), exist_ok=True)
    suffix = args.output_suffix or f"fourier_gamc_cvtrn_stacked_residual_split{args.split_seed}"

    print("=" * 144)
    print("Stacked residual meta-classifier fusion: Fourier soup + GAMC + CV-TRN")
    print("=" * 144)
    print("Academic protocol:")
    print("  - Fourier soup remains the main model.")
    print("  - GAMC and CV-TRN are auxiliary probability experts.")
    print("  - The stacked meta-classifier is trained only on validation labels.")
    print("  - OOF validation predictions select blend/gate parameters.")
    print("  - Test labels are used only once for the final report.")
    print("  - True SNR is never used as an inference feature.")

    soup = base.load_soup_cache(args.soup_prob_cache)
    gamc = base.load_gamc_cache(args.gamc_cache)
    cvtrn, cvtrn_path = load_cvtrn(args)
    tri.assert_three_way_alignment(soup, gamc, cvtrn)

    yv, sv = soup["labels_val"], soup["snrs_val"]
    yt, st = soup["labels_test"], soup["snrs_test"]
    class_names = soup["mod_classes"]
    nv_raw, nt_raw = soup["val_prob"], soup["test_prob"]
    gv_raw, gt_raw = gamc["val_prob"], gamc["test_prob"]
    cv_raw, ct_raw = cvtrn["val_prob"], cvtrn["test_prob"]
    rv, rt = gamc["val_router"], gamc["test_router"]

    if args.disable_temperature_scaling:
        tn = tg = tc = 1.0
    else:
        tn, _ = base.fit_temperature(nv_raw, yv, args.temperature_grid, "Fourier soup")
        tg, _ = base.fit_temperature(gv_raw, yv, args.temperature_grid, "GAMC")
        tc, _ = base.fit_temperature(cv_raw, yv, args.temperature_grid, "CV-TRN")

    nv = base.temperature_scale_probs(nv_raw, tn)
    nt = base.temperature_scale_probs(nt_raw, tn)
    gv = base.temperature_scale_probs(gv_raw, tg)
    gt = base.temperature_scale_probs(gt_raw, tg)
    cv = base.temperature_scale_probs(cv_raw, tc)
    ct = base.temperature_scale_probs(ct_raw, tc)

    base_val_m = base.metrics_from_probs(nv, yv, sv)
    base_test_m = base.metrics_from_probs(nt, yt, st)
    print("\nBaselines")
    base.print_metrics_line("Fourier Val", base_val_m)
    base.print_metrics_line("GAMC Val", base.metrics_from_probs(gv, yv, sv))
    base.print_metrics_line("CV-TRN Val", base.metrics_from_probs(cv, yv, sv))

    # Use a compact candidate pool as additional meta features, not as final labels.
    pool_args = copy.copy(args)
    pool_args.max_candidates = int(args.feature_candidates)
    val_full = gr.candidate_pool(nv, gv, cv, pool_args)
    test_full = gr.candidate_pool(nt, gt, ct, pool_args)
    val_pool, test_pool, kept_rows, all_rows = gr.prune_pool(val_full, test_full, yv, pool_args)
    print("\nMeta-feature candidate pool")
    print(f"Full validation oracle:   {gr.pool_oracle_acc(val_full, yv):.3f}%")
    print(f"Feature-pool oracle:      {gr.pool_oracle_acc(val_pool, yv):.3f}%")
    if args.report_test_oracle_diagnostic:
        print(f"Full test oracle:         {gr.pool_oracle_acc(test_full, yt):.3f}%")
        print(f"Feature-pool test oracle: {gr.pool_oracle_acc(test_pool, yt):.3f}%")
    for rec, row in zip(val_pool, kept_rows):
        print(f"  {rec['candidate_id']:02d} <- {rec['source_candidate_id']:02d} {rec['name']:<16} acc={row['acc']:.3f}% rescue={row['rescue']} harm={row['harm']}")
    save_csv(relpath("results", f"{suffix}_candidate_summary.csv"), all_rows)

    x_val = build_features(nv, gv, cv, rv, val_pool)
    x_test = build_features(nt, gt, ct, rt, test_pool)
    print(f"Feature dim: {x_val.shape[1]}")

    branches = []
    all_records = []
    print("\n" + "=" * 144)
    print("[*] Training OOF stacked meta-classifiers")
    print("=" * 144)
    for cfg in stacker_configs():
        print(f"    OOF stacker: {cfg['name']}")
        oof = fit_oof_stacker(x_val, yv, cfg, args)
        records = search_configs(nv, oof, yv, sv, base_val_m, args, cfg["name"])
        branches.append({"name": cfg["name"], "cfg": cfg, "oof": oof, "records": records})
        all_records.extend(records)

    # Fixed ensemble branch. This is selected by validation like the individual stackers.
    ens_oof = gr.norm(np.mean([b["oof"] for b in branches], axis=0))
    ens_records = search_configs(nv, ens_oof, yv, sv, base_val_m, args, "ensemble_mean")
    branches.append({"name": "ensemble_mean", "cfg": None, "oof": ens_oof, "records": ens_records})
    all_records.extend(ens_records)
    all_records.sort(key=lambda r: r["score"], reverse=True)
    print_top(all_records)

    best = all_records[0]
    selected_branch = best["branch"]
    print(f"\n[*] Selected validation branch: {selected_branch}")
    if selected_branch == "ensemble_mean":
        test_probs = []
        for b in branches:
            if b["cfg"] is None:
                continue
            test_probs.append(fit_final_stacker(x_val, yv, x_test, b["cfg"]))
        meta_test = gr.norm(np.mean(test_probs, axis=0))
    else:
        branch = next(b for b in branches if b["name"] == selected_branch)
        meta_test = fit_final_stacker(x_val, yv, x_test, branch["cfg"])

    final_prob, gate, use, alpha_vec = apply_stacked(nt, meta_test, best, args)
    final_m = base.metrics_from_probs(final_prob, yt, st)
    final_diag = base.switch_diagnostics(nt, final_prob, gate, use, alpha_vec, st)

    selected = {
        "best": best,
        "temperature": {"fourier_T": float(tn), "gamc_T": float(tg), "cvtrn_T": float(tc)},
        "cvtrn_cache_used": cvtrn_path,
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
        "method": "OOF validation-trained stacked residual meta-classifier",
        "test_oracle_diagnostic_reported": bool(args.report_test_oracle_diagnostic),
    }

    print("\n" + "=" * 144)
    print("Final test report")
    print("=" * 144)
    base.print_metrics_line("Fourier soup Test", base_test_m)
    base.print_metrics_line("GAMC Test", base.metrics_from_probs(gt, yt, st))
    base.print_metrics_line("CV-TRN Test", base.metrics_from_probs(ct, yt, st))
    base.print_metrics_line("Stacked residual Test", final_m)
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
        snrs=st.astype(np.int32),
        pred=final_m["pred"].astype(np.int64),
        final_prob=final_prob.astype(np.float32),
        fourier_prob=nt.astype(np.float32),
        gamc_prob=gt.astype(np.float32),
        cvtrn_prob=ct.astype(np.float32),
        stacked_meta_prob=meta_test.astype(np.float32),
        gate=gate.astype(bool),
        use_candidate=use.astype(bool),
        selected_config=np.asarray([json.dumps(selected, ensure_ascii=False)]),
        mod_classes=np.asarray(class_names),
    )
    print(f"[*] Predictions saved: {pred_path}")

    curve_path = relpath("results", f"accuracy_vs_snr_{suffix}.png")
    base.plot_curve(final_m["by_snr"], curve_path, "Stacked residual fusion")
    print(f"[*] SNR curve saved: {curve_path}")
    for snr_value in args.cm_snrs:
        cm_path = relpath("results", f"confusion_matrix_{snr_value}dB_{suffix}.png")
        acc = base.plot_cm_at_snr(
            labels=yt,
            pred=final_m["pred"],
            snrs=st,
            mod_classes=class_names,
            target_snr=int(snr_value),
            save_path=cm_path,
            title_prefix="Stacked residual fusion",
        )
        print(f"[*] {snr_value} dB confusion matrix saved: {cm_path} (Acc={acc:.2f}%)")


if __name__ == "__main__":
    main()

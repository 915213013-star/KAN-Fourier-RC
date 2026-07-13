import argparse
import csv
import json
import os

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

import evaluate_fourier_soup_gamc_cvtrn_protected_residual_fusion as tri
import evaluate_greedy_soup_gamc_protected_residual_fusion as base


def project_root():
    return os.path.dirname(os.path.abspath(__file__))


def relpath(*parts):
    return os.path.join(project_root(), *parts)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Gain-risk meta-router fusion for Fourier soup + GAMC + CV-TRN. "
            "It distills the validation oracle without using test labels or true SNR at inference."
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
    p.add_argument("--max_candidates", type=int, default=32)
    p.add_argument("--min_candidate_rescues", type=int, default=30)

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--router_estimators", type=int, default=160)
    p.add_argument("--router_max_depth", type=int, default=12)
    p.add_argument("--router_min_samples_leaf", type=int, default=8)
    p.add_argument("--router_max_features", type=str, default="sqrt")
    p.add_argument("--neutral_weight", type=float, default=0.20)
    p.add_argument("--rescue_weight", type=float, default=5.0)
    p.add_argument("--harm_weight", type=float, default=3.0)

    p.add_argument("--score_thresholds", type=float, nargs="+", default=[0.00, 0.03, 0.06, 0.09, 0.12, 0.16, 0.20])
    p.add_argument("--rescue_min_thresholds", type=float, nargs="+", default=[0.00, 0.03, 0.06, 0.10])
    p.add_argument("--harm_max_thresholds", type=float, nargs="+", default=[0.35, 0.45, 0.60, 0.80, 1.01])
    p.add_argument("--harm_penalties", type=float, nargs="+", default=[0.75, 1.00, 1.25, 1.50])
    p.add_argument("--cand_conf_thresholds", type=float, nargs="+", default=[0.00, 0.40, 0.50])
    p.add_argument("--max_switch_rates", type=float, nargs="+", default=[6.0, 8.0, 10.0, 12.0, 16.0, 22.0])
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
        help="Post-hoc diagnostic only. Keep disabled for strict development and paper-style runs.",
    )
    return p.parse_args()


def load_cvtrn(args):
    path = args.cvtrn_cache
    if not os.path.exists(path) and os.path.exists(args.fallback_cvtrn_cache):
        print(f"[!] CV-TRN-v2 cache missing, falling back to v1 cache: {args.fallback_cvtrn_cache}")
        path = args.fallback_cvtrn_cache
    return tri.load_cvtrn_cache(path), path


def norm(p):
    return base.normalize_probs(p)


def log_probs(p):
    return np.log(np.clip(norm(p), 1e-8, 1.0))


def exp_norm(x):
    x = x - x.max(axis=1, keepdims=True)
    return norm(np.exp(x))


def candidate_pool(fourier, gamc, cvtrn, args):
    f = norm(fourier)
    g = norm(gamc)
    c = norm(cvtrn)
    lf, lg, lc = log_probs(f), log_probs(g), log_probs(c)
    pool = [
        {"name": "fourier", "kind": "fourier", "alpha": 0.0, "prob": f},
        {"name": "raw_gamc", "kind": "raw_gamc", "alpha": 1.0, "prob": g},
        {"name": "raw_cvtrn", "kind": "raw_cvtrn", "alpha": 1.0, "prob": c},
    ]
    for a in args.mix_alphas:
        a = float(a)
        pool.extend(
            [
                {"name": f"prob_fg_{a:.2f}", "kind": "prob_fg", "alpha": a, "prob": norm((1.0 - a) * f + a * g)},
                {"name": f"prob_fc_{a:.2f}", "kind": "prob_fc", "alpha": a, "prob": norm((1.0 - a) * f + a * c)},
                {"name": f"prob_gc_{a:.2f}", "kind": "prob_gc", "alpha": a, "prob": norm((1.0 - a) * g + a * c)},
                {"name": f"prob_fgc_{a:.2f}", "kind": "prob_fgc", "alpha": a, "prob": norm((1.0 - a) * f + a * (0.5 * g + 0.5 * c))},
                {"name": f"logit_fg_{a:.2f}", "kind": "logit_fg", "alpha": a, "prob": exp_norm((1.0 - a) * lf + a * lg)},
                {"name": f"logit_fc_{a:.2f}", "kind": "logit_fc", "alpha": a, "prob": exp_norm((1.0 - a) * lf + a * lc)},
                {"name": f"logit_gc_{a:.2f}", "kind": "logit_gc", "alpha": a, "prob": exp_norm((1.0 - a) * lg + a * lc)},
                {"name": f"logit_fgc_{a:.2f}", "kind": "logit_fgc", "alpha": a, "prob": exp_norm((1.0 - a) * lf + a * (0.5 * lg + 0.5 * lc))},
            ]
        )
    for i, rec in enumerate(pool):
        rec["candidate_id"] = i
    return pool


def candidate_stats(pool, labels):
    base_pred = pool[0]["prob"].argmax(1)
    base_ok = base_pred == labels
    rows = []
    for rec in pool:
        pred = rec["prob"].argmax(1)
        ok = pred == labels
        rescue = int((~base_ok & ok).sum())
        harm = int((base_ok & ~ok).sum())
        same_pred = int((pred == base_pred).sum())
        rows.append(
            {
                "candidate_id": int(rec["candidate_id"]),
                "name": rec["name"],
                "kind": rec["kind"],
                "alpha": float(rec["alpha"]),
                "acc": float(ok.mean() * 100.0),
                "rescue": rescue,
                "harm": harm,
                "same_pred": same_pred,
                "utility": float(rescue - 0.35 * harm + 2.0 * (ok.mean() - base_ok.mean()) * len(labels)),
            }
        )
    return rows


def prune_pool(val_pool, test_pool, labels, args):
    rows = candidate_stats(val_pool, labels)
    keep_ids = {0, 1, 2}
    eligible = [r for r in rows if r["candidate_id"] != 0 and r["rescue"] >= args.min_candidate_rescues]
    eligible.sort(key=lambda r: (r["utility"], r["rescue"], r["acc"]), reverse=True)
    for r in eligible:
        if len(keep_ids) >= max(3, int(args.max_candidates)):
            break
        keep_ids.add(int(r["candidate_id"]))
    keep_ids = sorted(keep_ids)

    id_map = {old: new for new, old in enumerate(keep_ids)}
    pruned_val, pruned_test = [], []
    for old in keep_ids:
        for src, dst in ((val_pool, pruned_val), (test_pool, pruned_test)):
            rec = dict(src[old])
            rec["source_candidate_id"] = int(old)
            rec["candidate_id"] = int(id_map[old])
            dst.append(rec)
    kept_names = {r["name"] for r in pruned_val}
    kept_rows = [r for r in rows if r["name"] in kept_names]
    return pruned_val, pruned_test, kept_rows, rows


def margin(p):
    return base.margin(norm(p))


def entropy(p):
    return base.entropy(norm(p))


def kind_one_hot(kind):
    kinds = [
        "fourier",
        "raw_gamc",
        "raw_cvtrn",
        "prob_fg",
        "prob_fc",
        "prob_gc",
        "prob_fgc",
        "logit_fg",
        "logit_fc",
        "logit_gc",
        "logit_fgc",
    ]
    out = np.zeros(len(kinds), dtype=np.float32)
    if kind in kinds:
        out[kinds.index(kind)] = 1.0
    return out


def build_global_features(fourier, gamc, cvtrn, router):
    f, g, c, r = norm(fourier), norm(gamc), norm(cvtrn), norm(router)
    fp, gp, cp, rp = f.argmax(1), g.argmax(1), c.argmax(1), r.argmax(1)
    low1, high, gap1 = base.router_low_high(r, 1)
    low2, _, gap2 = base.router_low_high(r, 2)
    idx = np.arange(len(f))
    scalar = np.stack(
        [
            f.max(1), g.max(1), c.max(1), r.max(1),
            margin(f), margin(g), margin(c), margin(r),
            entropy(f), entropy(g), entropy(c), entropy(r),
            (fp == gp).astype(np.float32), (fp == cp).astype(np.float32), (gp == cp).astype(np.float32),
            np.abs(f - g).sum(1), np.abs(f - c).sum(1), np.abs(g - c).sum(1),
            (f * g).sum(1), (f * c).sum(1), (g * c).sum(1),
            g[idx, fp], c[idx, fp], f[idx, gp], f[idx, cp], g[idx, cp], c[idx, gp],
            low1, low2, high, gap1, gap2,
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate(
        [
            scalar,
            f.astype(np.float32),
            g.astype(np.float32),
            c.astype(np.float32),
            r.astype(np.float32),
            base.one_hot(fp, base.NUM_CLASSES),
            base.one_hot(gp, base.NUM_CLASSES),
            base.one_hot(cp, base.NUM_CLASSES),
            base.one_hot(rp, r.shape[1]),
        ],
        axis=1,
    ).astype(np.float32)


def build_candidate_features(global_x, fourier, gamc, cvtrn, router, cand_prob, rec):
    f, g, c, r, k = norm(fourier), norm(gamc), norm(cvtrn), norm(router), norm(cand_prob)
    fp, gp, cp, rp, kp = f.argmax(1), g.argmax(1), c.argmax(1), r.argmax(1), k.argmax(1)
    idx = np.arange(len(f))
    branch = np.tile(kind_one_hot(rec["kind"]), (len(f), 1))
    meta = np.tile(
        np.asarray(
            [
                float(rec["alpha"]),
                float(rec.get("candidate_id", 0)) / 100.0,
                float(rec.get("source_candidate_id", rec.get("candidate_id", 0))) / 100.0,
            ],
            dtype=np.float32,
        ),
        (len(f), 1),
    )
    scalar = np.stack(
        [
            k.max(1),
            margin(k),
            entropy(k),
            (kp == fp).astype(np.float32),
            (kp == gp).astype(np.float32),
            (kp == cp).astype(np.float32),
            np.abs(k - f).sum(1),
            np.abs(k - g).sum(1),
            np.abs(k - c).sum(1),
            (k * f).sum(1),
            (k * g).sum(1),
            (k * c).sum(1),
            k[idx, fp],
            k[idx, gp],
            k[idx, cp],
            f[idx, kp],
            g[idx, kp],
            c[idx, kp],
            r.max(1),
            base.margin(r),
        ],
        axis=1,
    ).astype(np.float32)
    x = np.concatenate(
        [
            global_x,
            scalar,
            branch,
            meta,
            k.astype(np.float32),
            base.one_hot(kp, base.NUM_CLASSES),
            (k - f).astype(np.float32),
            (base.prob_to_logits(k) - base.prob_to_logits(f)).astype(np.float32),
        ],
        axis=1,
    )
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def router_target(fourier_prob, cand_prob, labels):
    base_ok = norm(fourier_prob).argmax(1) == labels
    cand_ok = norm(cand_prob).argmax(1) == labels
    y = np.zeros(len(labels), dtype=np.int64)
    y[(~base_ok) & cand_ok] = 1
    y[base_ok & (~cand_ok)] = 2
    return y


def sample_weights(y, args):
    w = np.full(len(y), float(args.neutral_weight), dtype=np.float32)
    w[y == 1] = float(args.rescue_weight)
    w[y == 2] = float(args.harm_weight)
    counts = np.bincount(y, minlength=3).astype(np.float32)
    for cls in (1, 2):
        if counts[cls] > 0:
            w[y == cls] *= float(np.sqrt(max(counts[0], 1.0) / counts[cls]))
    return np.clip(w, 0.05, 50.0)


def make_router(args, seed):
    max_depth = None if int(args.router_max_depth) <= 0 else int(args.router_max_depth)
    return ExtraTreesClassifier(
        n_estimators=int(args.router_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(args.router_min_samples_leaf),
        max_features=args.router_max_features,
        bootstrap=False,
        n_jobs=-1,
        random_state=int(seed),
    )


def aligned_proba(clf, x):
    p = clf.predict_proba(x)
    out = np.zeros((x.shape[0], 3), dtype=np.float32)
    for j, cls in enumerate(clf.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def fit_candidate_oof(x, y, labels, args, seed):
    counts = np.bincount(y, minlength=3)
    if counts[1] < max(2, int(args.folds)) or counts[2] < max(2, int(args.folds)):
        return None, {"valid": False, "counts": counts.tolist()}
    folds = min(int(args.folds), int(counts[1]), int(counts[2]))
    oof = np.zeros((len(y), 3), dtype=np.float32)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    for fold, (tr, va) in enumerate(skf.split(x, y), 1):
        clf = make_router(args, seed + fold)
        clf.fit(x[tr], y[tr], sample_weight=sample_weights(y[tr], args))
        oof[va] = aligned_proba(clf, x[va])
    return oof, {"valid": True, "counts": counts.tolist(), "folds": int(folds)}


def fit_candidate_final(x, y, args, seed):
    clf = make_router(args, seed)
    clf.fit(x, y, sample_weight=sample_weights(y, args))
    return clf


def pool_oracle_acc(pool, labels):
    correct = np.stack([rec["prob"].argmax(1) == labels for rec in pool], axis=1)
    return float(correct.any(axis=1).mean() * 100.0)


def precompute_pool_static(pool):
    return {
        "cand_conf": np.stack([rec["prob"].max(1) for rec in pool], axis=1),
        "pred_mat": np.stack([rec["prob"].argmax(1) for rec in pool], axis=1),
    }


def apply_router(pool, router_prob, fourier_prob, cfg, args, static=None):
    if static is None:
        static = precompute_pool_static(pool)
    rescue = router_prob[:, :, 1]
    harm = router_prob[:, :, 2]
    score = rescue - float(cfg["harm_penalty"]) * harm
    score[:, 0] = -1e9

    cand_conf = static["cand_conf"]
    base_pred = pool[0]["prob"].argmax(1)
    pred_mat = static["pred_mat"]
    eligible = (
        (score >= float(cfg["score_threshold"]))
        & (rescue >= float(cfg["rescue_min"]))
        & (harm <= float(cfg["harm_max"]))
        & (cand_conf >= float(cfg["cand_conf_thr"]))
        & (pred_mat != base_pred[:, None])
    )
    eligible[:, 0] = False

    f_conf = norm(fourier_prob).max(1)
    f_margin = margin(fourier_prob)
    hard_keep = (f_conf >= float(args.hard_keep_conf)) & (f_margin >= float(args.hard_keep_margin))
    eligible[hard_keep, :] = False

    masked = np.where(eligible, score, -1e9)
    choice = masked.argmax(1).astype(np.int64)
    best_score = masked[np.arange(len(choice)), choice]
    choice[best_score <= -1e8] = 0

    max_rate = float(cfg["max_switch_rate"])
    if max_rate < 99.99:
        use = choice != 0
        n_allow = int(round(len(choice) * max_rate / 100.0))
        if use.sum() > n_allow:
            order = np.argsort(best_score[use])[::-1]
            use_idx = np.where(use)[0]
            keep = np.zeros(use.sum(), dtype=bool)
            keep[order[:n_allow]] = True
            drop_idx = use_idx[~keep]
            choice[drop_idx] = 0

    final = np.empty_like(pool[0]["prob"], dtype=np.float32)
    for cid, rec in enumerate(pool):
        m = choice == cid
        if m.any():
            final[m] = rec["prob"][m]
    return norm(final), choice, best_score


def choice_diagnostics(fourier, final, choice, score, snrs):
    use = choice != 0
    alpha = np.clip(np.where(use, score, 0.0), 0.0, 1.0).astype(np.float32)
    diag = base.switch_diagnostics(fourier, final, use, use, alpha, snrs)
    diag["choice_counts"] = {int(k): int((choice == k).sum()) for k in np.unique(choice)}
    return diag


def search_thresholds(pool, oof_router, fourier_prob, labels, snrs, base_m, args):
    records = []
    static = precompute_pool_static(pool)
    total = (
        len(args.harm_penalties)
        * len(args.score_thresholds)
        * len(args.rescue_min_thresholds)
        * len(args.harm_max_thresholds)
        * len(args.cand_conf_thresholds)
        * len(args.max_switch_rates)
    )
    done = 0
    print("\n" + "=" * 144)
    print(f"[*] Gain-risk threshold search, configs={total}")
    print("=" * 144)
    for hp in args.harm_penalties:
        for st in args.score_thresholds:
            for rmin in args.rescue_min_thresholds:
                for hmax in args.harm_max_thresholds:
                    for cthr in args.cand_conf_thresholds:
                        for msr in args.max_switch_rates:
                            done += 1
                            cfg = {
                                "harm_penalty": float(hp),
                                "score_threshold": float(st),
                                "rescue_min": float(rmin),
                                "harm_max": float(hmax),
                                "cand_conf_thr": float(cthr),
                                "max_switch_rate": float(msr),
                            }
                            final, choice, score = apply_router(pool, oof_router, fourier_prob, cfg, args, static=static)
                            m = base.metrics_from_probs(final, labels, snrs)
                            d = choice_diagnostics(fourier_prob, final, choice, score, snrs)
                            rec = {
                                "score": float(base.selection_score(m, d, base_m, args)),
                                "overall_acc": float(m["overall_acc"]),
                                "transition_acc": float(m["transition_acc"]),
                                "edge_low_acc": float(m["edge_low_acc"]),
                                "negative_acc": float(m["negative_acc"]),
                                "high_acc": float(m["high_acc"]),
                                "use_rate": float(d["use_rate"]),
                                "changed_high_rate": float(d["changed_high_rate"]),
                                "changed_nonultra_rate": float(d["changed_nonultra_rate"]),
                                "choice_counts": d["choice_counts"],
                                **cfg,
                            }
                            records.append(rec)
                            if done % 1000 == 0 or done == total:
                                best = max(records, key=lambda x: x["score"])
                                print(
                                    f"    {done:5d}/{total} | best score={best['score']:.3f} "
                                    f"overall={best['overall_acc']:.3f}% use={best['use_rate']:.2f}%"
                                )
    records.sort(key=lambda r: r["score"], reverse=True)
    return records


def save_csv(path, rows):
    if not rows:
        return
    keys = sorted(set(k for row in rows for k in row.keys()))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("choice_counts"), dict):
                out["choice_counts"] = json.dumps(out["choice_counts"], ensure_ascii=False)
            w.writerow(out)
    print(f"[*] CSV saved: {path}")


def print_top(records, n=20):
    print("\nTop gain-risk validation configs")
    for i, r in enumerate(records[:n], 1):
        print(
            f"{i:02d}. score={r['score']:.3f} | overall={r['overall_acc']:.3f}% | "
            f"neg={r['negative_acc']:.3f}% | edge={r['edge_low_acc']:.3f}% | high={r['high_acc']:.3f}% | "
            f"use={r['use_rate']:.2f}% | hp={r['harm_penalty']} st={r['score_threshold']} "
            f"rmin={r['rescue_min']} hmax={r['harm_max']} cthr={r['cand_conf_thr']} maxsw={r['max_switch_rate']}"
        )


def main():
    args = parse_args()
    os.makedirs(relpath("results"), exist_ok=True)
    suffix = args.output_suffix or f"fourier_gamc_cvtrn_gain_risk_router_split{args.split_seed}"

    print("=" * 144)
    print("Gain-risk meta-router fusion: Fourier soup + GAMC + CV-TRN")
    print("=" * 144)
    print("Academic protocol:")
    print("  - Fourier soup remains the main model.")
    print("  - GAMC and CV-TRN are auxiliary probability experts.")
    print("  - Validation labels train/select the blind gain-risk router.")
    print("  - Test labels are used only once for the final report.")
    print("  - True SNR is never used as an inference feature.")
    print(f"Fourier cache: {args.soup_prob_cache}")
    print(f"GAMC cache:    {args.gamc_cache}")
    print(f"CV-TRN cache:  {args.cvtrn_cache}")

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

    val_pool_full = candidate_pool(nv, gv, cv, args)
    test_pool_full = candidate_pool(nt, gt, ct, args)
    val_pool, test_pool, kept_rows, all_rows = prune_pool(val_pool_full, test_pool_full, yv, args)

    print("\nCandidate-pool diagnostics")
    print(f"Full validation pool size: {len(val_pool_full)} | pruned pool size: {len(val_pool)}")
    print(f"Full validation oracle:   {pool_oracle_acc(val_pool_full, yv):.3f}%")
    print(f"Pruned validation oracle: {pool_oracle_acc(val_pool, yv):.3f}%")
    if args.report_test_oracle_diagnostic:
        print(f"Full test oracle:         {pool_oracle_acc(test_pool_full, yt):.3f}%")
        print(f"Pruned test oracle:       {pool_oracle_acc(test_pool, yt):.3f}%")
    print("Kept candidates:")
    for rec, row in zip(val_pool, kept_rows):
        print(
            f"  {rec['candidate_id']:02d} <- {rec['source_candidate_id']:02d} "
            f"{rec['name']:<16} acc={row['acc']:.3f}% rescue={row['rescue']} harm={row['harm']}"
        )

    cand_summary_path = relpath("results", f"{suffix}_candidate_summary.csv")
    save_csv(cand_summary_path, all_rows)

    global_val = build_global_features(nv, gv, cv, rv)
    global_test = build_global_features(nt, gt, ct, rt)
    oof_router = np.zeros((len(yv), len(val_pool), 3), dtype=np.float32)
    test_router = np.zeros((len(yt), len(val_pool), 3), dtype=np.float32)
    oof_router[:, 0, 0] = 1.0
    test_router[:, 0, 0] = 1.0
    router_infos = []

    print("\n" + "=" * 144)
    print(f"[*] Training gain-risk routers for {len(val_pool) - 1} non-Fourier candidates")
    print("=" * 144)
    for cid, rec in enumerate(val_pool[1:], 1):
        y = router_target(nv, rec["prob"], yv)
        x_val = build_candidate_features(global_val, nv, gv, cv, rv, rec["prob"], rec)
        oof, info = fit_candidate_oof(x_val, y, yv, args, args.random_state + 97 * cid)
        info.update({"candidate_id": int(cid), "name": rec["name"], "kind": rec["kind"]})
        router_infos.append(info)
        if not info.get("valid", False):
            print(f"    {cid:02d}/{len(val_pool)-1:02d} {rec['name']:<16} skipped counts={info['counts']}")
            continue
        oof_router[:, cid, :] = oof
        clf = fit_candidate_final(x_val, y, args, args.random_state + 193 * cid)
        test_rec = test_pool[cid]
        x_test = build_candidate_features(global_test, nt, gt, ct, rt, test_rec["prob"], test_rec)
        test_router[:, cid, :] = aligned_proba(clf, x_test)
        print(
            f"    {cid:02d}/{len(val_pool)-1:02d} {rec['name']:<16} "
            f"counts={info['counts']} folds={info.get('folds', 0)}"
        )

    records = search_thresholds(val_pool, oof_router, nv, yv, sv, base_val_m, args)
    print_top(records)
    if not records:
        raise RuntimeError("No valid gain-risk config found.")

    best = records[0]
    final_prob, choice_test, score_test = apply_router(
        test_pool,
        test_router,
        nt,
        best,
        args,
        static=precompute_pool_static(test_pool),
    )
    final_m = base.metrics_from_probs(final_prob, yt, st)
    final_diag = choice_diagnostics(nt, final_prob, choice_test, score_test, st)

    selected = {
        "best": best,
        "temperature": {"fourier_T": float(tn), "gamc_T": float(tg), "cvtrn_T": float(tc)},
        "cvtrn_cache_used": cvtrn_path,
        "candidate_pool": [
            {
                "candidate_id": int(rec["candidate_id"]),
                "source_candidate_id": int(rec.get("source_candidate_id", rec["candidate_id"])),
                "name": rec["name"],
                "kind": rec["kind"],
                "alpha": float(rec["alpha"]),
            }
            for rec in val_pool
        ],
        "router_infos": router_infos,
        "full_validation_oracle": pool_oracle_acc(val_pool_full, yv),
        "pruned_validation_oracle": pool_oracle_acc(val_pool, yv),
        "test_oracle_diagnostic_reported": bool(args.report_test_oracle_diagnostic),
    }
    if args.report_test_oracle_diagnostic:
        selected["full_test_oracle"] = pool_oracle_acc(test_pool_full, yt)
        selected["pruned_test_oracle"] = pool_oracle_acc(test_pool, yt)

    print("\n" + "=" * 144)
    print("Final test report")
    print("=" * 144)
    base.print_metrics_line("Fourier soup Test", base_test_m)
    base.print_metrics_line("GAMC Test", base.metrics_from_probs(gt, yt, st))
    base.print_metrics_line("CV-TRN Test", base.metrics_from_probs(ct, yt, st))
    base.print_metrics_line("Gain-risk router Test", final_m)
    print("-" * 144)
    print(f"Delta vs Fourier overall:    {final_m['overall_acc'] - base_test_m['overall_acc']:+.4f} pp")
    print(f"Delta vs Fourier negative:   {final_m['negative_acc'] - base_test_m['negative_acc']:+.4f} pp")
    print(f"Delta vs Fourier edge:       {final_m['edge_low_acc'] - base_test_m['edge_low_acc']:+.4f} pp")
    print(f"Delta vs Fourier transition: {final_m['transition_acc'] - base_test_m['transition_acc']:+.4f} pp")
    print(f"Delta vs Fourier high:       {final_m['high_acc'] - base_test_m['high_acc']:+.4f} pp")
    print(f"Final diagnostics: {final_diag}")
    print("=" * 144)
    base.print_snr_table(final_m["by_snr"])

    records_path = relpath("results", f"{suffix}_search_top.csv")
    save_csv(records_path, records[: max(1, args.save_top_records)])
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
        router_choice=choice_test.astype(np.int64),
        router_score=score_test.astype(np.float32),
        selected_config=np.asarray([json.dumps(selected, ensure_ascii=False)]),
        mod_classes=np.asarray(class_names),
    )
    print(f"[*] Predictions saved: {pred_path}")

    curve_path = relpath("results", f"accuracy_vs_snr_{suffix}.png")
    base.plot_curve(final_m["by_snr"], curve_path, "Gain-risk meta-router fusion")
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
            title_prefix="Gain-risk meta-router fusion",
        )
        print(f"[*] {snr_value} dB confusion matrix saved: {cm_path} (Acc={acc:.2f}%)")


if __name__ == "__main__":
    main()

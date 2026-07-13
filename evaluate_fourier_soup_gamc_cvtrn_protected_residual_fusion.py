import argparse
import csv
import json
import os

import numpy as np

import evaluate_greedy_soup_gamc_protected_residual_fusion as base


def project_root():
    return os.path.dirname(os.path.abspath(__file__))


def relpath(*parts):
    return os.path.join(project_root(), *parts)


def parse_args():
    p = argparse.ArgumentParser(
        description="Three-branch protected residual fusion: Fourier greedy soup + GAMC + CV-TRN auxiliary branch."
    )
    p.add_argument("--split_seed", type=int, default=1)
    p.add_argument("--random_state", type=int, default=2026)
    p.add_argument("--soup_prob_cache", type=str, default=relpath("results", "greedy_soup_identity_valtest_probs_for_gamc_fusion.npz"))
    p.add_argument("--gamc_cache", type=str, default=relpath("results", "gamc_lite_v2_xgb_split1_valtest_probs_for_fusion.npz"))
    p.add_argument("--cvtrn_cache", type=str, default=relpath("results", "cv_trn_aux_split1_valtest_probs_for_fusion.npz"))
    p.add_argument("--output_suffix", type=str, default="")

    p.add_argument("--temperature_grid", type=float, nargs="+", default=[0.70, 0.85, 1.00, 1.15, 1.30, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00])
    p.add_argument("--disable_temperature_scaling", action="store_true")

    p.add_argument("--low_bin_counts", type=int, nargs="+", default=[1, 2])
    p.add_argument("--low_thresholds", type=float, nargs="+", default=[0.40, 0.50, 0.60, 0.70])
    p.add_argument("--high_max_thresholds", type=float, nargs="+", default=[0.35, 0.50, 0.65])
    p.add_argument("--low_gap_thresholds", type=float, nargs="+", default=[-0.10, 0.05, 0.20])
    p.add_argument("--neural_conf_thresholds", type=float, nargs="+", default=[0.50, 0.60, 0.70, 0.85])
    p.add_argument("--neural_margin_thresholds", type=float, nargs="+", default=[0.20, 0.35, 0.50, 1.01])
    p.add_argument("--alpha_max_values", type=float, nargs="+", default=[0.25, 0.40, 0.55, 0.70, 0.85])
    p.add_argument("--hard_keep_conf", type=float, default=0.92)
    p.add_argument("--hard_keep_margin", type=float, default=0.55)

    p.add_argument("--cv_neural_conf_thresholds", type=float, nargs="+", default=[0.45, 0.55, 0.65, 0.75])
    p.add_argument("--cv_neural_margin_thresholds", type=float, nargs="+", default=[0.15, 0.30, 0.45, 1.01])
    p.add_argument("--cv_conf_thresholds", type=float, nargs="+", default=[0.35, 0.45, 0.55, 0.65])
    p.add_argument("--cv_margin_thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--cv_conf_gap_thresholds", type=float, nargs="+", default=[-0.05, 0.00, 0.05])
    p.add_argument("--cv_alpha_max_values", type=float, nargs="+", default=[0.25, 0.40, 0.55, 0.70])

    p.add_argument("--top_k_gamc", type=int, default=12)
    p.add_argument("--top_k_cvtrn", type=int, default=12)
    p.add_argument("--top_k_candidates", type=int, default=18)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--selector_Cs", type=float, nargs="+", default=[0.01, 0.03, 0.10, 0.30])
    p.add_argument("--selector_neg_weights", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--selector_pos_weight", type=float, default=1.0)
    p.add_argument("--selector_thresholds", type=float, nargs="+", default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    p.add_argument("--selector_max_iter", type=int, default=1000)
    p.add_argument("--skip_selector", action="store_true")

    p.add_argument("--score_overall_weight", type=float, default=1.0)
    p.add_argument("--score_negative_gain_weight", type=float, default=0.020)
    p.add_argument("--score_edge_gain_weight", type=float, default=0.020)
    p.add_argument("--score_transition_gain_weight", type=float, default=0.010)
    p.add_argument("--score_high_penalty", type=float, default=3.00)
    p.add_argument("--high_tolerance", type=float, default=0.05)
    p.add_argument("--score_changed_high_penalty", type=float, default=0.015)
    p.add_argument("--score_changed_nonultra_penalty", type=float, default=0.006)
    p.add_argument("--save_top_records", type=int, default=120)
    p.add_argument("--cm_snrs", type=int, nargs="+", default=[0, -6])
    return p.parse_args()


def load_cvtrn_cache(path):
    z = base.load_npz_required(
        path,
        ["val_prob", "test_prob", "labels_val", "snrs_val", "labels_test", "snrs_test"],
    )
    return {
        "val_prob": base.normalize_probs(z["val_prob"]),
        "test_prob": base.normalize_probs(z["test_prob"]),
        "labels_val": z["labels_val"].astype(np.int64),
        "snrs_val": z["snrs_val"].astype(np.int32),
        "labels_test": z["labels_test"].astype(np.int64),
        "snrs_test": z["snrs_test"].astype(np.int32),
    }


def assert_three_way_alignment(soup, gamc, cvtrn):
    base.assert_alignment(soup, gamc)
    for key in ("labels_val", "labels_test", "snrs_val", "snrs_test"):
        if not np.all(soup[key] == cvtrn[key]):
            raise RuntimeError(f"Alignment check failed for CV-TRN cache: {key}")
    print("[*] Alignment check passed: neural soup, GAMC, and CV-TRN caches are identical.")


def cvtrn_cfgs(args):
    out = []
    for nc in args.cv_neural_conf_thresholds:
        for nm in args.cv_neural_margin_thresholds:
            for cc in args.cv_conf_thresholds:
                for cm in args.cv_margin_thresholds:
                    for gap in args.cv_conf_gap_thresholds:
                        for alpha in args.cv_alpha_max_values:
                            out.append(
                                {
                                    "cv_neural_conf_thr": float(nc),
                                    "cv_neural_margin_thr": float(nm),
                                    "cv_conf_thr": float(cc),
                                    "cv_margin_thr": float(cm),
                                    "cv_conf_gap_thr": float(gap),
                                    "cv_alpha_max": float(alpha),
                                }
                            )
    return out


def make_cvtrn_candidate(neural, cvtrn, cfg, args):
    neural = base.normalize_probs(neural)
    cvtrn = base.normalize_probs(cvtrn)
    n_conf = neural.max(axis=1)
    c_conf = cvtrn.max(axis=1)
    n_margin = base.margin(neural)
    c_margin = base.margin(cvtrn)

    gate = (
        (n_conf <= cfg["cv_neural_conf_thr"])
        & (n_margin <= cfg["cv_neural_margin_thr"])
        & (c_conf >= cfg["cv_conf_thr"])
        & (c_margin >= cfg["cv_margin_thr"])
        & ((c_conf - n_conf) >= cfg["cv_conf_gap_thr"])
    )
    hard_keep = (n_conf >= args.hard_keep_conf) & (n_margin >= args.hard_keep_margin)
    gate = gate & (~hard_keep)

    conf_strength = np.clip((c_conf - cfg["cv_conf_thr"]) / max(1.0 - cfg["cv_conf_thr"], 1e-6), 0.0, 1.0)
    unc_strength = np.clip((cfg["cv_neural_conf_thr"] - n_conf) / max(cfg["cv_neural_conf_thr"], 1e-6), 0.0, 1.0)
    margin_strength = np.clip((cfg["cv_neural_margin_thr"] - n_margin) / max(cfg["cv_neural_margin_thr"], 1e-6), 0.0, 1.0)
    alpha = cfg["cv_alpha_max"] * (0.5 + 0.5 * conf_strength) * (0.5 + 0.25 * unc_strength + 0.25 * margin_strength)
    alpha = np.clip(alpha * gate.astype(np.float32), 0.0, 1.0).astype(np.float32)

    logits = base.prob_to_logits(neural) + alpha[:, None] * (base.prob_to_logits(cvtrn) - base.prob_to_logits(neural))
    cand = base.normalize_probs(base.softmax_np(logits)).astype(np.float32)
    return cand, gate.astype(bool), alpha


def make_combined_candidate(neural, gamc, cvtrn, router, gamc_cfg, cv_cfg, args):
    _cand_g, gate_g, alpha_g, _aux_g = base.make_residual_candidate(neural, gamc, router, gamc_cfg, args)
    _cand_c, gate_c, alpha_c = make_cvtrn_candidate(neural, cvtrn, cv_cfg, args)
    logits_n = base.prob_to_logits(neural)
    logits = (
        logits_n
        + alpha_g[:, None] * (base.prob_to_logits(gamc) - logits_n)
        + alpha_c[:, None] * (base.prob_to_logits(cvtrn) - logits_n)
    )
    cand = base.normalize_probs(base.softmax_np(logits)).astype(np.float32)
    gate = gate_g | gate_c
    alpha = np.maximum(alpha_g, alpha_c).astype(np.float32)
    return cand, gate.astype(bool), alpha, alpha_g.astype(np.float32), alpha_c.astype(np.float32), gate_g.astype(bool), gate_c.astype(bool)


def record(branch, phase, metrics, diag, score, gamc_cfg=None, cv_cfg=None, selector_cfg=None, extra=None):
    out = {
        "branch": branch,
        "phase": phase,
        "score": float(score),
        "overall_acc": float(metrics["overall_acc"]),
        "transition_acc": float(metrics["transition_acc"]),
        "edge_low_acc": float(metrics["edge_low_acc"]),
        "negative_acc": float(metrics["negative_acc"]),
        "high_acc": float(metrics["high_acc"]),
        "gate_rate": float(diag["gate_rate"]),
        "use_rate": float(diag["use_rate"]),
        "effective_use_rate": float(diag["effective_use_rate"]),
        "changed_high_rate": float(diag["changed_high_rate"]),
        "changed_nonultra_rate": float(diag["changed_nonultra_rate"]),
        "gamc_cfg": gamc_cfg,
        "cv_cfg": cv_cfg,
        "selector_cfg": selector_cfg,
    }
    if extra:
        out.update(extra)
    return out


def score_candidate(metrics, diag, base_metrics, args):
    return base.selection_score(metrics, diag, base_metrics, args)


def search_gamc(neural_val, gamc_val, router_val, labels_val, snrs_val, base_m, args):
    print("\n" + "=" * 128)
    print("[*] Search GAMC-only residual candidates")
    print("=" * 128)
    records = []
    cfgs = base.candidate_cfgs(args)
    for idx, cfg in enumerate(cfgs, 1):
        cand, gate, alpha, _aux = base.make_residual_candidate(neural_val, gamc_val, router_val, cfg, args)
        use = gate & (alpha > 1e-8)
        m = base.metrics_from_probs(cand, labels_val, snrs_val)
        d = base.switch_diagnostics(neural_val, cand, gate, use, alpha, snrs_val)
        records.append(record("gamc", "deterministic", m, d, score_candidate(m, d, base_m, args), gamc_cfg=cfg))
        if idx % 500 == 0 or idx == len(cfgs):
            best = max(records, key=lambda r: r["score"])
            print(f"    GAMC {idx:5d}/{len(cfgs)} | best score={best['score']:.3f} overall={best['overall_acc']:.3f}%")
    records.sort(key=lambda r: r["score"], reverse=True)
    return records


def search_cvtrn(neural_val, cvtrn_val, labels_val, snrs_val, base_m, args):
    print("\n" + "=" * 128)
    print("[*] Search CV-TRN-only residual candidates")
    print("=" * 128)
    records = []
    cfgs = cvtrn_cfgs(args)
    for idx, cfg in enumerate(cfgs, 1):
        cand, gate, alpha = make_cvtrn_candidate(neural_val, cvtrn_val, cfg, args)
        use = gate & (alpha > 1e-8)
        m = base.metrics_from_probs(cand, labels_val, snrs_val)
        d = base.switch_diagnostics(neural_val, cand, gate, use, alpha, snrs_val)
        records.append(record("cvtrn", "deterministic", m, d, score_candidate(m, d, base_m, args), cv_cfg=cfg))
        if idx % 500 == 0 or idx == len(cfgs):
            best = max(records, key=lambda r: r["score"])
            print(f"    CVTRN {idx:5d}/{len(cfgs)} | best score={best['score']:.3f} overall={best['overall_acc']:.3f}%")
    records.sort(key=lambda r: r["score"], reverse=True)
    return records


def search_combined(neural_val, gamc_val, cvtrn_val, router_val, labels_val, snrs_val, base_m, gamc_records, cv_records, args):
    print("\n" + "=" * 128)
    print("[*] Search combined GAMC + CV-TRN residual candidates")
    print("=" * 128)
    records = []
    top_g = gamc_records[: max(1, args.top_k_gamc)]
    top_c = cv_records[: max(1, args.top_k_cvtrn)]
    total = len(top_g) * len(top_c)
    idx = 0
    for gr in top_g:
        for cr in top_c:
            idx += 1
            cand, gate, alpha, alpha_g, alpha_c, gate_g, gate_c = make_combined_candidate(
                neural_val, gamc_val, cvtrn_val, router_val, gr["gamc_cfg"], cr["cv_cfg"], args
            )
            use = gate & (alpha > 1e-8)
            m = base.metrics_from_probs(cand, labels_val, snrs_val)
            d = base.switch_diagnostics(neural_val, cand, gate, use, alpha, snrs_val)
            records.append(
                record(
                    "gamc_cvtrn",
                    "deterministic",
                    m,
                    d,
                    score_candidate(m, d, base_m, args),
                    gamc_cfg=gr["gamc_cfg"],
                    cv_cfg=cr["cv_cfg"],
                    extra={
                        "gamc_gate_rate": float(gate_g.mean() * 100.0),
                        "cv_gate_rate": float(gate_c.mean() * 100.0),
                        "mean_alpha_gamc": float(alpha_g[gate_g].mean()) if gate_g.any() else 0.0,
                        "mean_alpha_cvtrn": float(alpha_c[gate_c].mean()) if gate_c.any() else 0.0,
                    },
                )
            )
            if idx % 24 == 0 or idx == total:
                best = max(records, key=lambda r: r["score"])
                print(f"    combined {idx:4d}/{total} | best score={best['score']:.3f} overall={best['overall_acc']:.3f}%")
    records.sort(key=lambda r: r["score"], reverse=True)
    return records


def materialize_candidate(neural, gamc, cvtrn, router, rec, args):
    branch = rec["branch"]
    if branch == "gamc":
        cand, gate, alpha, _aux = base.make_residual_candidate(neural, gamc, router, rec["gamc_cfg"], args)
        extra = {"alpha_gamc": alpha, "alpha_cvtrn": np.zeros_like(alpha), "gate_gamc": gate, "gate_cvtrn": np.zeros_like(gate, dtype=bool)}
        return cand, gate, alpha, extra
    if branch == "cvtrn":
        cand, gate, alpha = make_cvtrn_candidate(neural, cvtrn, rec["cv_cfg"], args)
        extra = {"alpha_gamc": np.zeros_like(alpha), "alpha_cvtrn": alpha, "gate_gamc": np.zeros_like(gate, dtype=bool), "gate_cvtrn": gate}
        return cand, gate, alpha, extra
    cand, gate, alpha, alpha_g, alpha_c, gate_g, gate_c = make_combined_candidate(
        neural, gamc, cvtrn, router, rec["gamc_cfg"], rec["cv_cfg"], args
    )
    extra = {"alpha_gamc": alpha_g, "alpha_cvtrn": alpha_c, "gate_gamc": gate_g, "gate_cvtrn": gate_c}
    return cand, gate, alpha, extra


def selector_features(neural, gamc, cvtrn, cand, router, gate, alpha, branch_name):
    neural = base.normalize_probs(neural)
    gamc = base.normalize_probs(gamc)
    cvtrn = base.normalize_probs(cvtrn)
    cand = base.normalize_probs(cand)
    router = base.normalize_probs(router)
    gate = gate.astype(np.float32)
    alpha = alpha.astype(np.float32)

    nt = neural.argmax(axis=1)
    gt = gamc.argmax(axis=1)
    ct = cvtrn.argmax(axis=1)
    kt = cand.argmax(axis=1)
    rt = router.argmax(axis=1)
    low1, high, gap1 = base.router_low_high(router, 1)
    low2, _, gap2 = base.router_low_high(router, 2)
    branch_code = {
        "gamc": [1.0, 0.0, 0.0],
        "cvtrn": [0.0, 1.0, 0.0],
        "gamc_cvtrn": [0.0, 0.0, 1.0],
    }[branch_name]
    branch_mat = np.tile(np.asarray(branch_code, dtype=np.float32), (len(neural), 1))
    idx = np.arange(len(neural))

    scalar = np.stack(
        [
            neural.max(1), gamc.max(1), cvtrn.max(1), cand.max(1), router.max(1),
            base.margin(neural), base.margin(gamc), base.margin(cvtrn), base.margin(cand), base.margin(router),
            base.entropy(neural), base.entropy(gamc), base.entropy(cvtrn), base.entropy(cand), base.entropy(router),
            (nt == gt).astype(np.float32), (nt == ct).astype(np.float32), (nt == kt).astype(np.float32),
            (gt == ct).astype(np.float32), (gt == kt).astype(np.float32), (ct == kt).astype(np.float32),
            np.abs(neural - gamc).sum(1), np.abs(neural - cvtrn).sum(1), np.abs(neural - cand).sum(1),
            (neural * gamc).sum(1), (neural * cvtrn).sum(1), (neural * cand).sum(1),
            gamc[idx, nt], cvtrn[idx, nt], cand[idx, nt],
            neural[idx, gt], neural[idx, ct], neural[idx, kt],
            low1, low2, high, gap1, gap2, gate, alpha,
        ],
        axis=1,
    ).astype(np.float32)

    return np.nan_to_num(
        np.concatenate(
            [
                scalar,
                branch_mat,
                neural,
                gamc,
                cvtrn,
                cand,
                router,
                base.one_hot(nt, base.NUM_CLASSES),
                base.one_hot(gt, base.NUM_CLASSES),
                base.one_hot(ct, base.NUM_CLASSES),
                base.one_hot(kt, base.NUM_CLASSES),
                base.one_hot(rt, router.shape[1]),
            ],
            axis=1,
        ),
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    ).astype(np.float32)


def search_selector(all_records, neural_val, gamc_val, cvtrn_val, router_val, labels_val, snrs_val, base_m, args):
    if args.skip_selector:
        return []
    top = all_records[: max(1, args.top_k_candidates)]
    total = len(top) * len(args.selector_Cs) * len(args.selector_neg_weights)
    print("\n" + "=" * 128)
    print(f"[*] OOF selector search, candidates={len(top)}, selector fits={total}")
    print("=" * 128)
    out = []
    done = 0
    for rec in top:
        cand, gate, alpha, _extra = materialize_candidate(neural_val, gamc_val, cvtrn_val, router_val, rec, args)
        x = selector_features(neural_val, gamc_val, cvtrn_val, cand, router_val, gate, alpha, rec["branch"])
        for C in args.selector_Cs:
            for neg_weight in args.selector_neg_weights:
                done += 1
                oof, info = base.fit_selector_oof(
                    X=x,
                    neural_prob=neural_val,
                    cand_prob=cand,
                    labels=labels_val,
                    C=float(C),
                    folds=int(args.folds),
                    seed=int(args.random_state),
                    pos_weight=float(args.selector_pos_weight),
                    neg_weight=float(neg_weight),
                    max_iter=int(args.selector_max_iter),
                )
                if done % 10 == 0 or done == total:
                    print(f"    selector {done:4d}/{total}")
                if not info.get("valid", False):
                    continue
                for thr in args.selector_thresholds:
                    fused, use = base.apply_selector(neural_val, cand, oof, thr, base_gate=gate)
                    m = base.metrics_from_probs(fused, labels_val, snrs_val)
                    d = base.switch_diagnostics(neural_val, fused, gate, use, alpha, snrs_val)
                    selector_cfg = {
                        "selector_C": float(C),
                        "selector_neg_weight": float(neg_weight),
                        "selector_pos_weight": float(args.selector_pos_weight),
                        "selector_thr": float(thr),
                    }
                    out.append(
                        record(
                            rec["branch"],
                            "selector_oof",
                            m,
                            d,
                            score_candidate(m, d, base_m, args),
                            gamc_cfg=rec["gamc_cfg"],
                            cv_cfg=rec["cv_cfg"],
                            selector_cfg=selector_cfg,
                            extra={
                                "discord_count": int(info.get("discord_count", 0)),
                                "valid_folds": int(info.get("valid_folds", 0)),
                            },
                        )
                    )
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def apply_best(best, neural_val, gamc_val, cvtrn_val, router_val, labels_val, neural_test, gamc_test, cvtrn_test, router_test, args):
    cand_val, gate_val, alpha_val, _extra_val = materialize_candidate(neural_val, gamc_val, cvtrn_val, router_val, best, args)
    cand_test, gate_test, alpha_test, extra_test = materialize_candidate(neural_test, gamc_test, cvtrn_test, router_test, best, args)
    selector_cfg = best.get("selector_cfg", None)
    if selector_cfg is None:
        switch_prob = gate_test.astype(np.float32)
        final_prob = cand_test
        use = gate_test & (alpha_test > 1e-8)
        info = {"selector_valid": False, "mode": "deterministic"}
    else:
        x_val = selector_features(neural_val, gamc_val, cvtrn_val, cand_val, router_val, gate_val, alpha_val, best["branch"])
        x_test = selector_features(neural_test, gamc_test, cvtrn_test, cand_test, router_test, gate_test, alpha_test, best["branch"])
        scaler, clf, info = base.fit_final_selector(
            X=x_val,
            neural_prob=neural_val,
            cand_prob=cand_val,
            labels=labels_val,
            C=float(selector_cfg["selector_C"]),
            pos_weight=float(selector_cfg["selector_pos_weight"]),
            neg_weight=float(selector_cfg["selector_neg_weight"]),
            max_iter=int(args.selector_max_iter),
        )
        if not info.get("valid", False):
            switch_prob = gate_test.astype(np.float32)
            final_prob = cand_test
            use = gate_test & (alpha_test > 1e-8)
            info = {"selector_valid": False, "mode": "fallback_deterministic", **info}
        else:
            switch_prob = clf.predict_proba(scaler.transform(x_test))[:, 1].astype(np.float32)
            final_prob, use = base.apply_selector(
                neural_test,
                cand_test,
                switch_prob,
                float(selector_cfg["selector_thr"]),
                base_gate=gate_test,
            )
            info = {"selector_valid": True, "mode": "selector", **info}
    return {
        "final_prob": base.normalize_probs(final_prob),
        "candidate_prob": base.normalize_probs(cand_test),
        "gate": gate_test,
        "alpha": alpha_test,
        "switch_prob": switch_prob,
        "use_candidate": use,
        "selector_info": info,
        **extra_test,
    }


def flatten_record(rec):
    out = {}
    for k, v in rec.items():
        if k in ("gamc_cfg", "cv_cfg", "selector_cfg"):
            out[k] = json.dumps(v or {}, ensure_ascii=False)
        else:
            out[k] = v
    return out


def save_records(path, records, n):
    rows = [flatten_record(r) for r in records[: max(1, n)]]
    keys = sorted(set(k for row in rows for k in row.keys()))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[*] Search records saved: {path}")


def jsonable_record(rec):
    out = dict(rec)
    return out


def main():
    args = parse_args()
    os.makedirs(relpath("results"), exist_ok=True)
    suffix = args.output_suffix or f"fourier_soup_gamc_cvtrn_protected_residual_fusion_split{args.split_seed}"

    print("=" * 144)
    print("Fourier greedy soup + GAMC + CV-TRN protected residual fusion")
    print("=" * 144)
    print(f"Fourier soup cache: {args.soup_prob_cache}")
    print(f"GAMC cache:         {args.gamc_cache}")
    print(f"CV-TRN cache:       {args.cvtrn_cache}")
    print("[*] CV-TRN and GAMC are auxiliary probability experts. The Fourier soup is still the main branch.")

    soup = base.load_soup_cache(args.soup_prob_cache)
    gamc = base.load_gamc_cache(args.gamc_cache)
    cvtrn = load_cvtrn_cache(args.cvtrn_cache)
    assert_three_way_alignment(soup, gamc, cvtrn)

    yv, sv = soup["labels_val"], soup["snrs_val"]
    yt, st = soup["labels_test"], soup["snrs_test"]
    class_names = soup["mod_classes"]

    nv_raw, nt_raw = soup["val_prob"], soup["test_prob"]
    gv_raw, gt_raw = gamc["val_prob"], gamc["test_prob"]
    cv_raw, ct_raw = cvtrn["val_prob"], cvtrn["test_prob"]
    rv, rt = gamc["val_router"], gamc["test_router"]

    print("\nValidation baselines before calibration")
    base.print_metrics_line("Fourier soup Val raw", base.metrics_from_probs(nv_raw, yv, sv))
    base.print_metrics_line("GAMC Val raw", base.metrics_from_probs(gv_raw, yv, sv))
    base.print_metrics_line("CV-TRN Val raw", base.metrics_from_probs(cv_raw, yv, sv))

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
    print("\nValidation baselines after calibration")
    base.print_metrics_line("Fourier soup Val calibrated", base_val_m)
    base.print_metrics_line("GAMC Val calibrated", base.metrics_from_probs(gv, yv, sv))
    base.print_metrics_line("CV-TRN Val calibrated", base.metrics_from_probs(cv, yv, sv))

    gamc_records = search_gamc(nv, gv, rv, yv, sv, base_val_m, args)
    cv_records = search_cvtrn(nv, cv, yv, sv, base_val_m, args)
    combined_records = search_combined(nv, gv, cv, rv, yv, sv, base_val_m, gamc_records, cv_records, args)
    deterministic_records = sorted(gamc_records + cv_records + combined_records, key=lambda r: r["score"], reverse=True)

    print("\nTop deterministic candidates")
    for i, r in enumerate(deterministic_records[:16], 1):
        print(
            f"{i:02d}. {r['branch']:<10} score={r['score']:.3f} | overall={r['overall_acc']:.3f}% | "
            f"neg={r['negative_acc']:.3f}% | edge={r['edge_low_acc']:.3f}% | high={r['high_acc']:.3f}% | gate={r['gate_rate']:.2f}%"
        )

    selector_records = search_selector(deterministic_records, nv, gv, cv, rv, yv, sv, base_val_m, args)
    all_records = sorted(deterministic_records + selector_records, key=lambda r: r["score"], reverse=True)

    print("\nTop final validation candidates")
    for i, r in enumerate(all_records[:20], 1):
        print(
            f"{i:02d}. {r['phase']:<13} {r['branch']:<10} score={r['score']:.3f} | "
            f"overall={r['overall_acc']:.3f}% | neg={r['negative_acc']:.3f}% | "
            f"edge={r['edge_low_acc']:.3f}% | high={r['high_acc']:.3f}% | use={r['use_rate']:.2f}%"
        )

    best = all_records[0]
    selected = {
        "best": jsonable_record(best),
        "temperature": {"fourier_T": tn, "gamc_T": tg, "cvtrn_T": tc},
        "soup_prob_cache": args.soup_prob_cache,
        "gamc_cache": args.gamc_cache,
        "cvtrn_cache": args.cvtrn_cache,
    }
    print("\n" + "=" * 144)
    print("Selected validation config")
    print("=" * 144)
    print(json.dumps(selected, ensure_ascii=False, indent=2))

    records_path = relpath("results", f"{suffix}_search_top.csv")
    save_records(records_path, all_records, args.save_top_records)
    config_path = relpath("results", f"{suffix}_selected_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"[*] Selected config saved: {config_path}")

    final = apply_best(best, nv, gv, cv, rv, yv, nt, gt, ct, rt, args)
    final_m = base.metrics_from_probs(final["final_prob"], yt, st)
    gamc_test_m = base.metrics_from_probs(gt, yt, st)
    cv_test_m = base.metrics_from_probs(ct, yt, st)
    final_diag = base.switch_diagnostics(nt, final["final_prob"], final["gate"], final["use_candidate"], final["alpha"], st)

    print("\n" + "=" * 144)
    print("Final test report")
    print("=" * 144)
    base.print_metrics_line("Fourier soup Test", base_test_m)
    base.print_metrics_line("GAMC Test", gamc_test_m)
    base.print_metrics_line("CV-TRN Test", cv_test_m)
    base.print_metrics_line("Three-branch fusion Test", final_m)
    print("-" * 144)
    print(f"Delta vs Fourier overall:    {final_m['overall_acc'] - base_test_m['overall_acc']:+.4f} pp")
    print(f"Delta vs Fourier negative:   {final_m['negative_acc'] - base_test_m['negative_acc']:+.4f} pp")
    print(f"Delta vs Fourier edge:       {final_m['edge_low_acc'] - base_test_m['edge_low_acc']:+.4f} pp")
    print(f"Delta vs Fourier transition: {final_m['transition_acc'] - base_test_m['transition_acc']:+.4f} pp")
    print(f"Delta vs Fourier high:       {final_m['high_acc'] - base_test_m['high_acc']:+.4f} pp")
    print("-" * 144)
    print(f"Final diagnostics: {final_diag}")
    print(f"Selector info: {final['selector_info']}")
    print("=" * 144)
    base.print_snr_table(final_m["by_snr"])

    pred_path = relpath("results", f"{suffix}_predictions.npz")
    np.savez_compressed(
        pred_path,
        labels=yt.astype(np.int64),
        snrs=st.astype(np.int32),
        pred=final_m["pred"].astype(np.int64),
        final_prob=final["final_prob"].astype(np.float32),
        candidate_prob=final["candidate_prob"].astype(np.float32),
        fourier_prob=nt.astype(np.float32),
        gamc_prob=gt.astype(np.float32),
        cvtrn_prob=ct.astype(np.float32),
        router_prob=rt.astype(np.float32),
        gate=final["gate"].astype(np.int8),
        gate_gamc=final["gate_gamc"].astype(np.int8),
        gate_cvtrn=final["gate_cvtrn"].astype(np.int8),
        alpha=final["alpha"].astype(np.float32),
        alpha_gamc=final["alpha_gamc"].astype(np.float32),
        alpha_cvtrn=final["alpha_cvtrn"].astype(np.float32),
        switch_prob=final["switch_prob"].astype(np.float32),
        use_candidate=final["use_candidate"].astype(np.int8),
        selected_config=np.asarray([json.dumps(selected, ensure_ascii=False)]),
        selector_info=np.asarray([json.dumps(final["selector_info"], ensure_ascii=False)]),
        mod_classes=np.asarray(class_names),
    )
    print(f"[*] Predictions saved: {pred_path}")

    curve_path = relpath("results", f"accuracy_vs_snr_{suffix}.png")
    base.plot_curve(final_m["by_snr"], curve_path, "Fourier soup + GAMC + CV-TRN protected residual fusion")
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
            title_prefix="Fourier soup + GAMC + CV-TRN fusion",
        )
        print(f"[*] {snr_value} dB confusion matrix saved: {cm_path} (Acc={acc:.2f}%)")


if __name__ == "__main__":
    main()

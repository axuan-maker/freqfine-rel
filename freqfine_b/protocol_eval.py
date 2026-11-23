"""protocol_eval.py — 按交付 train/validation/test 折跑 H1a / H1b（正式协议）。"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from baselines import BUILDERS
from config import Config
from metrics import auc, delong_test, eer, subject_bootstrap_diff_ci
from train_eval import load_features, make_model

BASELINES = ["full_band", "best_single_subband", "random_subband", "subject_baseline_norm"]
CONDITIONS = ["relation"] + BASELINES


def _proba_col(model, pos_class):
    return int(np.argmax(model.classes_ == pos_class))


def load_roles(train_dir, val_dir, test_dir):
    bp1, rel1, y1, s1, _ = load_features(train_dir)
    bp2, rel2, y2, s2, _ = load_features(val_dir)
    bp3, rel3, y3, s3, _ = load_features(test_dir)
    return (
        {"band_power": np.vstack([bp1, bp2, bp3]), "relation": np.vstack([rel1, rel2, rel3])},
        np.concatenate([y1, y2, y3]),
        np.concatenate([s1, s2, s3]),
        np.concatenate([
            np.full(len(y1), "train"),
            np.full(len(y2), "validation"),
            np.full(len(y3), "test"),
        ]),
    )


def evaluate_condition(bundle, y, subjects, roles, builder, cfg):
    tr = np.where(roles == "train")[0]
    va = np.where(roles == "validation")[0]
    te = np.where(roles == "test")[0]
    seeds = list(cfg.seeds)
    test_scores = np.full((len(seeds), te.size), np.nan)
    val_scores = np.full((len(seeds), va.size), np.nan)
    for si, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        Xtr, Xte = builder(bundle, tr, te, y[tr], subjects[tr], rng, cfg)
        model = make_model(cfg.model, random_state=seed).fit(Xtr, y[tr])
        test_scores[si] = model.predict_proba(Xte)[:, _proba_col(model, cfg.pos_class)]

        Xtr2, Xva = builder(bundle, tr, va, y[tr], subjects[tr], rng, cfg)
        model2 = make_model(cfg.model, random_state=seed).fit(Xtr2, y[tr])
        val_scores[si] = model2.predict_proba(Xva)[:, _proba_col(model2, cfg.pos_class)]
    return {
        "test_scores": test_scores,
        "val_scores": val_scores,
        "y_test": y[te],
        "y_val": y[va],
        "test_subjects": subjects[te],
    }


def summarize(res):
    pooled_test = np.nanmean(res["test_scores"], axis=0)
    pooled_val = np.nanmean(res["val_scores"], axis=0)
    per_seed = [auc(res["y_test"], res["test_scores"][si]) for si in range(len(res["test_scores"]))]
    per_seed = [a for a in per_seed if np.isfinite(a)]
    return {
        "test_auc": auc(res["y_test"], pooled_test),
        "test_auc_std": float(np.std(per_seed)) if per_seed else np.nan,
        "test_eer": eer(res["y_test"], pooled_test),
        "val_auc": auc(res["y_val"], pooled_val),
        "n_test": int(res["y_test"].size),
    }


def h1b_fold(train_dir, val_dir, test_dir, cfg):
    bundle, y, subjects, roles = load_roles(train_dir, val_dir, test_dir)
    results, pooled = {}, {}
    for cond in CONDITIONS:
        cfg2 = cfg
        if cond == "random_subband":
            cfg2 = Config(cfg.as_dict())
            cfg2._d["seeds"] = list(cfg.random_subband_seeds)
        res = evaluate_condition(bundle, y, subjects, roles, BUILDERS[cond], cfg2)
        results[cond] = summarize(res)
        pooled[cond] = np.nanmean(res["test_scores"], axis=0)

    te = roles == "test"
    y_test, test_subjects = y[te], subjects[te]
    comparisons = {}
    for cond in BASELINES:
        lo, hi = subject_bootstrap_diff_ci(
            y_test, pooled["relation"], pooled[cond], test_subjects, seed=cfg.seeds[0]
        )
        _, _, _, p = delong_test(y_test, pooled["relation"], pooled[cond])
        comparisons[cond] = {
            "delta_auc": float(results["relation"]["test_auc"] - results[cond]["test_auc"]),
            "ci_low": lo,
            "ci_high": hi,
            "delong_p": p,
        }
    return {"conditions": results, "comparisons": comparisons}


def h1a_fold(train_dir, val_dir, test_dir, cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    bundle, y, subjects, roles = load_roles(train_dir, val_dir, test_dir)
    bp = bundle["band_power"]
    tr = roles == "train"
    te = roles == "test"
    K = bp.shape[1]
    aucs = np.full(K, np.nan)
    scores_all = {}
    seed = int(cfg.seeds[0])
    for k in range(K):
        sc = StandardScaler().fit(bp[tr, [k]])
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        model.fit(sc.transform(bp[tr, [k]]), y[tr])
        scores = model.predict_proba(sc.transform(bp[te, [k]]))[:, _proba_col(model, cfg.pos_class)]
        scores_all[k] = scores
        aucs[k] = auc(y[te], scores)

    order = [int(i) for i in np.argsort(aucs)[::-1]]
    top, med = order[0], order[K // 2]
    lo, hi = subject_bootstrap_diff_ci(y[te], scores_all[top], scores_all[med], subjects[te], seed=seed)
    _, _, _, p = delong_test(y[te], scores_all[top], scores_all[med])
    return {
        "top_subband": top,
        "median_subband": med,
        "delta_top_vs_median": float(aucs[top] - aucs[med]),
        "ci": [lo, hi],
        "delong_p": p,
        "subband_aucs": {int(k): float(aucs[k]) for k in range(K)},
        "ranking": order,
    }


def _fold_dirs(features_root, protocol):
    return (
        os.path.join(features_root, protocol, "train"),
        os.path.join(features_root, protocol, "validation"),
        os.path.join(features_root, protocol, "test"),
    )


def run_h1b(features_root, protocols, out_dir, cfg):
    fold_reports, acc = [], {c: [] for c in CONDITIONS}
    for p in protocols:
        tr, va, te = _fold_dirs(features_root, p)
        rep = h1b_fold(tr, va, te, cfg)
        fold_reports.append({"protocol": p, **rep})
        for c in CONDITIONS:
            acc[c].append(rep["conditions"][c]["test_auc"])
    summary = {
        c: {"mean_test_auc": float(np.mean(acc[c])), "std_test_auc": float(np.std(acc[c]))}
        for c in CONDITIONS
    }
    delta = summary["relation"]["mean_test_auc"] - summary["subject_baseline_norm"]["mean_test_auc"]
    verdict = "GO" if delta >= float(cfg.go_nogo_margin) else "NO-GO"
    report = {
        "verdict": verdict,
        "margin": float(cfg.go_nogo_margin),
        "delta_vs_subject_baseline_norm": delta,
        "summary": summary,
        "folds": fold_reports,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "h1b_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("=" * 66)
    print("H1b 汇总（%d 折 test AUC）" % len(protocols))
    for c in CONDITIONS:
        print("  %-24s %.4f ± %.4f" % (c, summary[c]["mean_test_auc"], summary[c]["std_test_auc"]))
    print("  关系 vs Subject-Baseline-Norm 增量: %+.4f (阈值 %.4f)" % (delta, cfg.go_nogo_margin))
    print("  判定: %s" % verdict)
    print("=" * 66)
    return report


def run_h1a(features_root, protocols, out_dir, cfg):
    fold_reports, auc_acc = [], {}
    for p in protocols:
        tr, va, te = _fold_dirs(features_root, p)
        rep = h1a_fold(tr, va, te, cfg)
        fold_reports.append({"protocol": p, **rep})
        for k, v in rep["subband_aucs"].items():
            auc_acc.setdefault(k, []).append(v)
    mean_aucs = {k: float(np.mean(v)) for k, v in auc_acc.items()}
    order = [int(k) for k in sorted(mean_aucs, key=mean_aucs.get, reverse=True)]
    top, med = order[0], order[len(order) // 2]
    delta = mean_aucs[top] - mean_aucs[med]
    supported = delta > 0 and bool(np.mean([r["delong_p"] for r in fold_reports if np.isfinite(r["delong_p"])]) < 0.05) if fold_reports else False
    report = {
        "verdict": "H1a_supported" if supported else "H1a_weak",
        "top_subband": top,
        "median_subband": med,
        "delta_top_vs_median": delta,
        "mean_subband_aucs": mean_aucs,
        "ranking": order,
        "folds": fold_reports,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "h1a_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("=" * 66)
    print("H1a 子带判别力排序（%d 折平均 test AUC）" % len(protocols))
    for rank, k in enumerate(order):
        print("  rank %d: subband %2d  AUC=%.4f" % (rank + 1, k, mean_aucs[k]))
    print("  top vs median: ΔAUC=%+.4f" % delta)
    print("  判定: %s" % report["verdict"])
    print("=" * 66)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="按 train/validation/test 折跑 H1a/H1b")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("h1a", "h1b"):
        p = sub.add_parser(name)
        p.add_argument("--features_root", required=True)
        p.add_argument("--protocols", nargs="+", required=True)
        p.add_argument("--out_dir", required=True)
        p.add_argument("--model", default="svm", choices=["svm", "rf"])
        p.add_argument("--weighting", default="none", choices=["none", "hsic", "mi"])
        p.add_argument("--pos_class", type=int, default=1)
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        p.add_argument("--go_nogo_margin", type=float, default=0.015)

    args = ap.parse_args()
    cfg = Config()
    cfg._d["model"] = args.model
    cfg._d["weighting"] = args.weighting
    cfg._d["pos_class"] = args.pos_class
    cfg._d["seeds"] = list(args.seeds)
    cfg._d["go_nogo_margin"] = args.go_nogo_margin

    if args.cmd == "h1b":
        run_h1b(args.features_root, args.protocols, args.out_dir, cfg)
    else:
        run_h1a(args.features_root, args.protocols, args.out_dir, cfg)

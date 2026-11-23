"""h1a.py — 稀疏性假设检验：单子带判别力排序是否显著不均。

方法：
1. 对每个子带 k，用该子带 log 功率作为单特征，在 LOSO 下跑逻辑回归，
   得到每个样本的预测概率 scores_k 与 pooled AUC_k；
2. 按 pooled AUC 排序子带；
3. 用“最强子带 vs 中位子带”的按被试 bootstrap 差值与 DeLong 检验判定
   排序是否显著不均；
4. 若 ΔAUC 的 bootstrap CI 下界 > 0（或 DeLong p < 0.05），H1a 支持
   （频域判别信息稀疏/不均），否则稀疏性主张弱。
"""
from __future__ import annotations

import json
import os

import numpy as np

from metrics import auc, delong_test, subject_bootstrap_auc_ci, subject_bootstrap_diff_ci
from train_eval import load_features


def _single_subband_loso(X_col, y, subjects, pos_class, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X_col, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    subjects = np.asarray(subjects)
    scores = np.full(X.shape[0], np.nan, dtype=np.float64)

    for g in np.unique(subjects):
        tr = subjects != g
        te = subjects == g
        if np.unique(y[tr]).size < 2:
            scores[te] = float(np.mean(y[tr] == pos_class))
            continue
        sc = StandardScaler().fit(X[tr])
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(sc.transform(X[tr]), y[tr])
        proba = model.predict_proba(sc.transform(X[te]))
        idx = int(np.argmax(model.classes_ == pos_class))
        scores[te] = proba[:, idx]
    return scores


def run_h1a(features_dir, out_dir, pos_class=1, n_boot=2000, seed=0):
    band_power, relation, y, subjects, _ = load_features(features_dir)
    if len(np.unique(y)) < 2:
        raise ValueError("标签只有一类，无法计算 AUC")
    K = band_power.shape[1]

    pooled_aucs = np.full(K, np.nan)
    scores_all = {}
    for k in range(K):
        scores = _single_subband_loso(band_power[:, k], y, subjects, pos_class, seed)
        scores_all[k] = scores
        pooled_aucs[k] = auc(y, scores)

    order = [int(i) for i in np.argsort(pooled_aucs)[::-1]]
    top = order[0]
    med = order[K // 2]
    delta = float(pooled_aucs[top] - pooled_aucs[med])
    lo, hi = subject_bootstrap_diff_ci(
        y, scores_all[top], scores_all[med], subjects, n_boot=n_boot, seed=seed
    )
    auc_top, auc_med, z, p = delong_test(y, scores_all[top], scores_all[med])
    top_ci = subject_bootstrap_auc_ci(
        y, scores_all[top], subjects, n_boot=n_boot, seed=seed
    )
    spread = float(pooled_aucs[top] - pooled_aucs[order[-1]])
    supported = bool(np.isfinite(lo) and lo > 0.0)

    report = {
        "verdict": "H1a_supported" if supported else "H1a_weak",
        "top_subband": top,
        "median_subband": med,
        "delta_top_vs_median": delta,
        "bootstrap_ci": [lo, hi],
        "delong_p": p,
        "top_subband_auc_ci": list(top_ci),
        "spread_max_min": spread,
        "subband_aucs": {int(k): float(pooled_aucs[k]) for k in range(K)},
        "ranking": order,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "h1a_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("=" * 68)
    print("H1a 单子带判别力排序（LOSO pooled AUC）")
    print("-" * 68)
    for rank, k in enumerate(order):
        marker = "  <-- top" if rank == 0 else ("  <-- median" if k == med else "")
        print("  rank %d: subband %2d  AUC=%.4f%s" % (rank + 1, k, pooled_aucs[k], marker))
    print("-" * 68)
    print("  top vs median: ΔAUC=%+.4f  CI=[%.4f, %.4f]  DeLong p=%.4f"
          % (delta, lo, hi, p))
    print("  spread(max-min)=%.4f" % spread)
    print("  判定: %s" % report["verdict"])
    print("=" * 68)
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="H1a 稀疏性检验")
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--out_dir", default="results/h1a")
    ap.add_argument("--pos_class", type=int, default=1)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_h1a(args.features_dir, args.out_dir, pos_class=args.pos_class,
            n_boot=args.n_boot, seed=args.seed)

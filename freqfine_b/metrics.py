"""metrics.py — 临时统计量（B 侧自用；正式口径以 C 的审计为准）。

包含 AUC / EER / DeLong 检验 / Bootstrap 置信区间 / Cohen's d。
sklearn / scipy 采用惰性导入。
"""
from __future__ import annotations

import numpy as np


def auc(y, scores):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2 or not np.all(np.isfinite(s)):
        return float("nan")
    return float(roc_auc_score(y, s))


def eer(y, scores):
    from sklearn.metrics import roc_curve

    y = np.asarray(y)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def _auc_components(y, scores):
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan, None, None
    v10 = np.array([np.mean((si > neg) + 0.5 * (si == neg)) for si in pos])
    v01 = np.array([np.mean((pos > sj) + 0.5 * (pos == sj)) for sj in neg])
    return float(v10.mean()), v10, v01


def delong_test(y, s1, s2):
    """两个模型在同一批样本上的 DeLong 检验，返回 (auc1, auc2, z, p)。"""
    from scipy.stats import norm

    auc1, v10_1, v01_1 = _auc_components(y, s1)
    auc2, v10_2, v01_2 = _auc_components(y, s2)
    if v10_1 is None or v10_2 is None:
        return auc1, auc2, np.nan, np.nan
    n_pos = v10_1.size
    n_neg = v01_1.size
    s10 = float(np.sum((v10_1 - auc1) * (v10_2 - auc2)) / (n_pos - 1))
    s01 = float(np.sum((v01_1 - auc1) * (v01_2 - auc2)) / (n_neg - 1))
    var = s10 + s01
    if var <= 0.0 or not np.isfinite(var):
        return auc1, auc2, np.nan, np.nan
    z = (auc1 - auc2) / np.sqrt(var)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return auc1, auc2, float(z), float(p)


def bootstrap_auc_ci(y, scores, n_boot=2000, alpha=0.05, seed=0):
    y = np.asarray(y)
    s = np.asarray(scores, dtype=np.float64)
    rng = np.random.RandomState(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        a = auc(y[idx], s[idx])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    lo = float(np.percentile(vals, 100.0 * alpha / 2.0))
    hi = float(np.percentile(vals, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_paired_diff_ci(a, b, n_boot=2000, alpha=0.05, seed=0):
    """配对 AUC 差值的 bootstrap CI（a、b 为按被试对齐的两个数组）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    rng = np.random.RandomState(seed)
    n = d.size
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        vals.append(float(np.mean(d[idx])))
    if not vals:
        return float("nan"), float("nan")
    lo = float(np.percentile(vals, 100.0 * alpha / 2.0))
    hi = float(np.percentile(vals, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def cohen_d(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    sd = float(d.std(ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return float("nan")
    return float(d.mean() / sd)


def subject_bootstrap_auc_ci(y, scores, subjects, n_boot=2000, alpha=0.05, seed=0):
    """按被试（而非样本）重采样的 pooled AUC 置信区间。"""
    y = np.asarray(y)
    s = np.asarray(scores, dtype=np.float64)
    subjects = np.asarray(subjects)
    usub = np.unique(subjects)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(usub, size=usub.size, replace=True)
        mask = np.isin(subjects, chosen)
        a = auc(y[mask], s[mask])
        if np.isfinite(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    lo = float(np.percentile(vals, 100.0 * alpha / 2.0))
    hi = float(np.percentile(vals, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def subject_bootstrap_diff_ci(y, s1, s2, subjects, n_boot=2000, alpha=0.05, seed=0):
    """两个模型 pooled AUC 差值的按被试 bootstrap CI（配对）。"""
    y = np.asarray(y)
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    subjects = np.asarray(subjects)
    usub = np.unique(subjects)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(usub, size=usub.size, replace=True)
        mask = np.isin(subjects, chosen)
        a1 = auc(y[mask], s1[mask])
        a2 = auc(y[mask], s2[mask])
        if np.isfinite(a1) and np.isfinite(a2):
            vals.append(a1 - a2)
    if not vals:
        return float("nan"), float("nan")
    lo = float(np.percentile(vals, 100.0 * alpha / 2.0))
    hi = float(np.percentile(vals, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi

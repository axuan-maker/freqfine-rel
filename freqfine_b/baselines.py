"""baselines.py — H1b 四条件对照的特征构造（折内拟合，无泄漏）。

条件：
- relation                跨频带关系向量（可折内序数弱监督加权）
- full_band               K 维子带 log 功率（全频带底线）
- best_single_subband     训练折内按 ANOVA F 选最佳单子带
- random_subband          随机单子带（多种子平均）
- subject_baseline_norm   被试内中心化（仅用训练被试估计，未见被试退化为全局标准化）
"""
from __future__ import annotations

import numpy as np


def _scale(Xtr, Xte):
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def build_relation(bundle, tr, te, ytr, groups_tr, rng, cfg):
    from weak_supervision import estimate_weights

    X = bundle["relation"]
    Xtr, Xte = _scale(X[tr], X[te])
    if cfg.weighting and cfg.weighting != "none":
        w = estimate_weights(Xtr, ytr, subjects=groups_tr, method=cfg.weighting, tau=cfg.weight_tau)
        Xtr = Xtr * w
        Xte = Xte * w
    return Xtr, Xte


def build_full_band(bundle, tr, te, ytr, groups_tr, rng, cfg):
    return _scale(bundle["band_power"][tr], bundle["band_power"][te])


def build_best_single(bundle, tr, te, ytr, groups_tr, rng, cfg):
    from sklearn.feature_selection import f_classif

    X = bundle["band_power"]
    F, _ = f_classif(X[tr], ytr)
    F = np.nan_to_num(F, nan=-np.inf)
    col = int(np.argmax(F))
    return _scale(X[tr][:, [col]], X[te][:, [col]])


def build_random_subband(bundle, tr, te, ytr, groups_tr, rng, cfg):
    X = bundle["band_power"]
    col = int(rng.randint(X.shape[1]))
    return _scale(X[tr][:, [col]], X[te][:, [col]])


def build_subject_norm(bundle, tr, te, ytr, groups_tr, rng, cfg):
    """被试内中心化基线：仅用训练被试估计中心；未见被试使用训练全局中心。"""
    X = bundle["band_power"].astype(np.float64)
    Xtr = X[tr].copy()
    Xte = X[te].astype(np.float64).copy()
    for g in np.unique(groups_tr):
        m = groups_tr == g
        Xtr[m] -= Xtr[m].mean(axis=0)
    global_mean = Xtr.mean(axis=0)
    Xte -= global_mean
    return _scale(Xtr, Xte)


BUILDERS = {
    "relation": build_relation,
    "full_band": build_full_band,
    "best_single_subband": build_best_single,
    "random_subband": build_random_subband,
    "subject_baseline_norm": build_subject_norm,
}

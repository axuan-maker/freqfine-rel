"""weak_supervision.py — 序数弱监督连续加权（在训练折内拟合）。

w_m ≈ I(r_m; Y | Z)：控制被试变量 Z 的前提下估计特征与序数标签的相关性。
- 优先使用 PyRKHSstats 的 HSIC；不可用时回退 numpy 中心化核 HSIC；
- 提供直方图互信息作为无 scipy 依赖的回退；
- 温度 softmax 归一化；
- 注意：权重必须在每个训练折内单独估计，禁止用测试被试拟合。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "rbf_kernel",
    "hsic",
    "conditional_hsic",
    "mutual_info_hist",
    "estimate_weights",
    "normalize_weights",
    "apply_weights",
    "class_weights",
    "ordinal_soft_weight",
]


def rbf_kernel(X, sigma: Optional[float] = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    sq = (X[:, None, :] - X[None, :, :]) ** 2
    d2 = sq.sum(axis=-1)
    if sigma is None:
        off = d2[np.triu_indices_from(d2, k=1)]
        sigma = float(np.sqrt(np.median(off))) if off.size else 1.0
        if sigma <= 1e-12:
            sigma = 1.0
    gamma = 1.0 / (2.0 * float(sigma) ** 2)
    return np.exp(-gamma * d2)


def hsic(x, y, sigma_x: Optional[float] = None, use_linear_y: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 4 or x.size != y.size:
        return 0.0

    K = rbf_kernel(x, sigma_x)
    if use_linear_y:
        Y = y.reshape(-1, 1)
        L = Y @ Y.T
    else:
        L = rbf_kernel(y)

    H = np.eye(n) - np.ones((n, n)) / n
    HKH = H @ K @ H
    HLH = H @ L @ H
    stat = float(np.trace(HKH @ HLH)) / ((n - 1) ** 2)
    return max(stat, 0.0)


def conditional_hsic(x, y, subjects, sigma_x: Optional[float] = None, min_per_group: int = 2) -> float:
    """按被试分组 HSIC 的加权平均，用于近似控制 Z 的主效应。

    注意：这是 E_z[HSIC(X, Y | Z=z)] 的启发式估计，并不严格等于条件
    独立性度量 I(X; Y | Z)；若需精确条件检验请替换为 tigramite/CMI。
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    subjects = np.asarray(subjects).reshape(-1)
    if not (x.size == y.size == subjects.size):
        raise ValueError("x / y / subjects 长度不一致")

    values, weights = [], []
    for group in np.unique(subjects):
        mask = subjects == group
        if mask.sum() < min_per_group:
            continue
        if np.unique(y[mask]).size < 2:
            continue
        values.append(hsic(x[mask], y[mask], sigma_x))
        weights.append(float(mask.sum()))
    if not values:
        return 0.0
    return float(np.average(values, weights=weights))


def mutual_info_hist(x, y, bins: int = 16) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or x.size != y.size:
        return 0.0

    count, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = count / count.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    def entropy(p: np.ndarray) -> float:
        p = p[p > 0.0]
        return float(-np.sum(p * np.log(p)))

    return max(entropy(px) + entropy(py) - entropy(pxy.ravel()), 0.0)


def normalize_weights(w, tau: float = 1.0) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    tau = float(tau)
    if tau <= 0:
        raise ValueError("温度 tau 必须为正")
    if w.size == 0:
        return w
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if np.all(w == 0.0):
        return np.ones_like(w) / w.size
    z = (w / tau) - np.max(w / tau)
    e = np.exp(z)
    return e / e.sum()


def apply_weights(features, weights) -> np.ndarray:
    X = np.asarray(features, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if X.ndim == 1:
        if X.size != w.size:
            raise ValueError("特征与权重维度不一致")
        return X * w
    if X.ndim == 2:
        if X.shape[1] != w.size:
            raise ValueError("特征列数(%d)与权重数(%d)不一致" % (X.shape[1], w.size))
        return X * w
    raise ValueError("features 必须是 1D 或 2D")


def estimate_weights(
    features,
    labels,
    subjects=None,
    method: str = "hsic",
    tau: float = 1.0,
) -> np.ndarray:
    """在给定（训练折）样本上逐特征估计权重并做温度归一化。"""
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("features 必须是 (n_samples, n_features)")
    y = np.asarray(labels).reshape(-1)
    if y.size != X.shape[0]:
        raise ValueError("features 与 labels 样本数不一致")

    method = method.lower()
    M = X.shape[1]
    raw = np.zeros(M, dtype=np.float64)
    for m in range(M):
        xm = X[:, m]
        if float(np.std(xm)) <= 1e-12:
            raw[m] = 0.0
            continue
        if method == "mi":
            raw[m] = mutual_info_hist(xm, y)
        elif subjects is not None:
            raw[m] = conditional_hsic(xm, y, subjects)
        else:
            raw[m] = hsic(xm, y)
    return normalize_weights(raw, tau)


def ordinal_soft_weight(
    features,
    labels,
    subjects=None,
    method: str = "hsic",
    tau: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = estimate_weights(features, labels, subjects=subjects, method=method, tau=tau)
    weighted = apply_weights(features, weights)
    return weights, weighted


def class_weights(labels, smooth: float = 1.0) -> Dict[int, float]:
    labels = np.asarray(labels).reshape(-1)
    classes, counts = np.unique(labels, return_counts=True)
    w = smooth / (counts + smooth)
    w = w / w.sum() * len(classes)
    return dict(zip([int(c) for c in classes], [float(v) for v in w]))


if __name__ == "__main__":
    import argparse

    from train_eval import load_features

    ap = argparse.ArgumentParser(description="序数弱监督加权诊断（仅供检查，不做训练）")
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--method", default="hsic", choices=["hsic", "mi"])
    ap.add_argument("--tau", type=float, default=1.0)
    args = ap.parse_args()

    _, relation, y, subjects, _ = load_features(args.features_dir)
    weights, _ = ordinal_soft_weight(relation, y, subjects=subjects, method=args.method, tau=args.tau)
    order = np.argsort(weights)[::-1]
    print("[weak_supervision] 权重诊断（仅供检查；正式训练须在折内拟合）")
    for rank, m in enumerate(order[:10]):
        print("  rank %2d: feature %3d weight %.4f" % (rank + 1, int(m), float(weights[m])))

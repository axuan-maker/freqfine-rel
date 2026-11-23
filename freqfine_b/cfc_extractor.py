"""cfc_extractor.py — 跨频带耦合特征提取（核心）。

输入：data_loader.DataCatalog 给出的原始连续信号（音频 (T,) 或 EEG (13,T)）。
输出：每个试次的子带功率、上三角关系向量与 K×K 耦合矩阵，并按规范记录
频带边界、顺序与采样率（写到 feature 侧的 meta.json，不混入原始信号维度）。

核心计算只依赖 numpy 与 scipy。
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "calculate_AAC",
    "calculate_PAC",
    "calculate_PPC",
    "calculate_coherence",
    "subband_decompose",
    "hilbert_envelope_phase",
    "band_edges",
    "coupling_matrix",
    "upper_triangle",
    "parse_coupling",
    "extract_sample_features",
    "discover_modules",
    "process_directory",
]

_EPS = np.finfo(np.float64).eps


def _as_float_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < 1:
        raise ValueError("%s 必须是非空的一维序列" % name)
    if not np.all(np.isfinite(arr)):
        raise ValueError("%s 包含 NaN 或 Inf，拒绝计算" % name)
    return arr


# ---------------------------------------------------------------------- #
# 子带分解
# ---------------------------------------------------------------------- #
def band_edges(f_min: float, f_max: float, n_subbands: int, spacing: str = "log") -> List[Tuple[float, float]]:
    f_min = float(f_min)
    f_max = float(f_max)
    n_subbands = int(n_subbands)
    if not (0.0 < f_min < f_max):
        raise ValueError("频率范围非法: 需满足 0 < f_min < f_max")
    if n_subbands < 2:
        raise ValueError("n_subbands 必须 >= 2")
    if spacing == "log":
        edges = np.geomspace(f_min, f_max, n_subbands + 1)
    else:
        edges = np.linspace(f_min, f_max, n_subbands + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_subbands)]


def subband_decompose(
    signal,
    fs: float,
    n_subbands: int = 8,
    f_min: float = 20.0,
    f_max: Optional[float] = None,
    order: int = 4,
    method: str = "filterbank",
) -> np.ndarray:
    """将一维信号分解为 K 个子带，返回 (K, n_samples)。"""
    from scipy.signal import butter, filtfilt

    x = _as_float_1d(signal, "signal")
    fs = float(fs)
    n_subbands = int(n_subbands)
    if n_subbands < 2:
        raise ValueError("n_subbands 必须 >= 2")
    if x.size < 64:
        raise ValueError("信号太短（<64 样本），无法可靠地做子带分解")

    if method == "wpd":
        return _subband_decompose_wpd(x, n_subbands)
    if method != "filterbank":
        raise ValueError("method 必须是 filterbank 或 wpd")

    nyq = fs / 2.0
    f_max = nyq * 0.95 if f_max is None else float(f_max)
    if not (0.0 < f_min < f_max <= nyq):
        raise ValueError("频率范围非法: 需满足 0 < f_min < f_max <= fs/2")

    bands = []
    for lo, hi in band_edges(f_min, f_max, n_subbands):
        b, a = butter(order, [lo, hi], btype="bandpass", fs=fs)
        bands.append(filtfilt(b, a, x))
    return np.stack(bands, axis=0)


def _subband_decompose_wpd(x: np.ndarray, n_subbands: int) -> np.ndarray:
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("method='wpd' 需要 PyWavelets：pip install PyWavelets") from exc
    level = int(round(np.log2(n_subbands)))
    if 2 ** level != n_subbands:
        raise ValueError("小波包分解要求 n_subbands 为 2 的幂")
    wp = pywt.WaveletPacket(data=x, wavelet="db4", mode="symmetric", maxlevel=level)
    nodes = wp.get_level(level, order="freq")
    bands = [np.asarray(node.data, dtype=np.float64) for node in nodes]
    return np.stack(bands, axis=0)


def hilbert_envelope_phase(signal) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.signal import hilbert

    x = _as_float_1d(signal, "signal")
    analytic = hilbert(x)
    amp = np.abs(analytic)
    phase = np.angle(analytic)
    return amp, phase


# ---------------------------------------------------------------------- #
# 四种耦合度量
# ---------------------------------------------------------------------- #
def calculate_AAC(envelope_1, envelope_2) -> float:
    env1 = _as_float_1d(envelope_1, "envelope_1")
    env2 = _as_float_1d(envelope_2, "envelope_2")
    if env1.size != env2.size:
        raise ValueError("两个包络长度不一致: %d != %d" % (env1.size, env2.size))
    if env1.size < 2:
        raise ValueError("包络长度必须 >= 2")
    mean1 = env1.mean()
    mean2 = env2.mean()
    cov = float(np.mean((env1 - mean1) * (env2 - mean2)))
    var1 = float(np.mean((env1 - mean1) ** 2))
    var2 = float(np.mean((env2 - mean2) ** 2))
    denom = np.sqrt(var1 * var2)
    if denom <= _EPS:
        return 0.0
    return float(np.clip(cov / denom, -1.0, 1.0))


def calculate_PAC(phase_low, amp_high, n_bins: int = 18) -> float:
    phase = _as_float_1d(phase_low, "phase_low")
    amp = _as_float_1d(amp_high, "amp_high")
    if phase.size != amp.size:
        raise ValueError("相位与幅度长度不一致")
    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError("n_bins 必须 >= 2")
    if phase.size < n_bins:
        raise ValueError("样本数(%d)必须 >= n_bins(%d)" % (phase.size, n_bins))
    if np.any(amp < 0.0):
        raise ValueError("amp_high 应为非负包络，但出现了负值")

    total_amp = float(amp.sum())
    if total_amp <= _EPS:
        raise ValueError("amp_high 全为 0，无法归一化")

    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    P = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        sel = bin_idx == b
        if np.any(sel):
            P[b] = float(amp[sel].mean())
    psum = float(P.sum())
    if psum <= _EPS:
        return 0.0
    P = P / psum
    H = float(-np.sum(np.where(P > 0.0, P * np.log(P), 0.0)))
    log_n = float(np.log(n_bins))
    return float(np.clip((log_n - H) / log_n, 0.0, 1.0))


def calculate_PPC(phase_i, phase_j) -> float:
    pi = _as_float_1d(phase_i, "phase_i")
    pj = _as_float_1d(phase_j, "phase_j")
    if pi.size != pj.size:
        raise ValueError("相位长度不一致")
    if pi.size < 2:
        raise ValueError("相位长度必须 >= 2")
    return float(np.abs(np.mean(np.exp(1j * (pi - pj)))))


def calculate_coherence(x, y, fs: float, nperseg: int = 256) -> float:
    from scipy.signal import coherence

    x = _as_float_1d(x, "x")
    y = _as_float_1d(y, "y")
    if x.size != y.size:
        raise ValueError("信号长度不一致")
    _, cxy = coherence(x, y, fs=float(fs), nperseg=int(nperseg))
    return float(np.nanmean(cxy))


def _matrix_from(measure, subbands, amps, phases, fs, nperseg, n_bins) -> np.ndarray:
    K = len(subbands)
    mat = np.full((K, K), np.nan, dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            if measure == "AAC":
                mat[i, j] = calculate_AAC(amps[i], amps[j])
            elif measure == "PAC":
                mat[i, j] = calculate_PAC(phases[i], amps[j], n_bins)
            elif measure == "PPC":
                mat[i, j] = calculate_PPC(phases[i], phases[j])
            elif measure == "COH":
                mat[i, j] = calculate_coherence(subbands[i], subbands[j], fs, nperseg)
    return mat


def parse_coupling(coupling) -> List[str]:
    raw = str(coupling).upper().replace("BOTH", "AAC,PAC")
    out = []
    for part in raw.split(","):
        p = part.strip()
        if p and p in ("AAC", "PAC", "PPC", "COH") and p not in out:
            out.append(p)
    if not out:
        raise ValueError("coupling 必须包含 AAC / PAC / PPC / COH 之一")
    return out


def coupling_matrix(measure, subbands, amps, phases, fs, nperseg=256, n_bins=18) -> np.ndarray:
    return _matrix_from(measure.upper(), subbands, amps, phases, fs, nperseg, n_bins)


def upper_triangle(mat: np.ndarray) -> np.ndarray:
    K = mat.shape[0]
    return np.asarray(mat[np.triu_indices(K, k=1)], dtype=np.float64)


# ---------------------------------------------------------------------- #
# 单样本特征提取
# ---------------------------------------------------------------------- #
def extract_sample_features(signal, fs: float, cfg) -> dict:
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("signal 必须是 1D（音频/ECG/rPPG）或 2D（EEG: 通道, T）")

    nyq = float(fs) / 2.0
    f_max = nyq * 0.95 if cfg.f_max is None else float(cfg.f_max)
    bands = band_edges(cfg.f_min, f_max, cfg.n_subbands, spacing="log")
    measures = parse_coupling(cfg.coupling)

    pow_sum = None
    rel_concats = None
    matrices = []
    for c in range(arr.shape[0]):
        ch = arr[c]
        if not np.all(np.isfinite(ch)):
            ch = np.where(np.isfinite(ch), ch, 0.0)
        subbands = subband_decompose(ch, float(fs), cfg.n_subbands, cfg.f_min, f_max, cfg.filter_order)
        amps, phases = [], []
        for k in range(subbands.shape[0]):
            a, p = hilbert_envelope_phase(subbands[k])
            amps.append(a)
            phases.append(p)
        amps = np.stack(amps, axis=0)
        phases = np.stack(phases, axis=0)

        power = np.array([np.log10(np.mean(s * s) + 1e-12) for s in subbands])
        rels = []
        for measure in measures:
            mat = coupling_matrix(measure, subbands, amps, phases, float(fs), cfg.coherence_nperseg, cfg.pac_n_bins)
            if measure == measures[0]:
                matrices.append(mat)
            rels.append(upper_triangle(mat))
        rel = np.concatenate(rels)

        pow_sum = power if pow_sum is None else pow_sum + power
        rel_concats = rel if rel_concats is None else rel_concats + rel

    n_ch = arr.shape[0]
    band_power = pow_sum / n_ch
    relation = rel_concats / n_ch
    if cfg.add_band_power:
        relation = np.concatenate([relation, band_power])
    mean_matrix = np.nanmean(np.stack(matrices), axis=0)
    return {
        "band_power": band_power.astype(np.float64),
        "relation": relation.astype(np.float64),
        "coupling_matrix": mean_matrix.astype(np.float64),
        "bands": bands,
        "fs": float(fs),
        "measures": measures,
    }


def discover_modules(coupling_matrix: np.ndarray, n_modules: Optional[int] = None, seed: int = 0) -> np.ndarray:
    """谱聚类关系模块发现（诊断输出，不再决定 Top-m 硬选）。"""
    from sklearn.cluster import SpectralClustering

    A = np.nan_to_num(coupling_matrix, nan=0.0)
    A = (np.abs(A) + np.abs(A).T) / 2.0
    K = A.shape[0]
    if n_modules is None:
        n_modules = int(np.clip(round(K / 2.0), 4, 8))
    n_modules = int(min(n_modules, K))
    if n_modules < 2:
        return np.arange(K)
    return SpectralClustering(
        n_clusters=n_modules,
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans",
    ).fit_predict(A)


# ---------------------------------------------------------------------- #
# 按样本循环提取并落盘（内存控制 + 频带/采样率元数据）
# ---------------------------------------------------------------------- #
def process_directory(
    processed_root: str,
    modality: str,
    out_dir: str,
    cfg,
    dataset: Optional[str] = None,
) -> str:
    from data_loader import DataCatalog

    catalog = DataCatalog(processed_root)
    feat_dir = os.path.join(out_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)

    manifest_rows: List[Dict] = []
    modules = None
    n_seen = 0
    matrix_acc = None

    for sample in catalog.iter_samples(modality, dataset=dataset):
        signal, meta, valid = sample.load_signal()
        fs = meta.get("sr_hz")
        if not fs or float(fs) <= 0:
            raise ValueError("样本 %s 缺少有效 sr_hz（未知采样率不得 ready）" % sample.sample_id)
        feats = extract_sample_features(signal, float(fs), cfg)

        rel_path = os.path.join("features", sample.sample_id + ".npz")
        np.savez_compressed(
            os.path.join(out_dir, rel_path),
            band_power=feats["band_power"].astype(np.float32),
            relation=feats["relation"].astype(np.float32),
            coupling_matrix=feats["coupling_matrix"].astype(np.float32),
        )

        with open(os.path.join(feat_dir, sample.sample_id + ".json"), "w", encoding="utf-8") as fh:
            json.dump({
                "schema_version": "1.0",
                "sample_id": sample.sample_id,
                "subject_id": sample.subject_id,
                "dataset": sample.dataset,
                "modality": modality,
                "n_subbands": int(cfg.n_subbands),
                "band_edges": [[float(lo), float(hi)] for lo, hi in feats["bands"]],
                "band_order": "low-to-high",
                "sr_hz": float(fs),
                "coupling": feats["measures"],
                "relation_dim": int(feats["relation"].size),
                "band_power_dim": int(feats["band_power"].size),
                "valid_fraction": float(np.mean(valid)),
            }, fh, ensure_ascii=False)

        manifest_rows.append({
            "sample_id": sample.sample_id,
            "subject_id": sample.subject_id,
            "dataset": sample.dataset,
            "label": int(sample.label),
            "valid_fraction": float(np.mean(valid)),
            "feature_file": rel_path.replace("\\", "/"),
        })

        if matrix_acc is None:
            matrix_acc = feats["coupling_matrix"].copy()
        else:
            matrix_acc = matrix_acc + np.nan_to_num(feats["coupling_matrix"])
        n_seen += 1

    if not manifest_rows:
        raise ValueError("没有提取到任何特征，请检查 processed_root / modality / dataset")

    matrix_mean = matrix_acc / n_seen
    modules = discover_modules(matrix_mean, cfg.n_modules).tolist()

    with open(os.path.join(out_dir, "feature_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "schema_version": "1.0",
            "n_subbands": int(cfg.n_subbands),
            "coupling": parse_coupling(cfg.coupling),
            "n_samples": n_seen,
            "modules": modules,
        }, fh, ensure_ascii=False)

    manifest_path = os.path.join(out_dir, "manifest.csv")
    fieldnames = ["sample_id", "subject_id", "dataset", "label", "valid_fraction", "feature_file"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("[cfc_extractor] 完成: %d 个试次 -> %s" % (n_seen, manifest_path))
    print("[cfc_extractor] 关系模块（谱聚类）:", modules)
    return manifest_path


if __name__ == "__main__":
    import argparse

    from config import Config

    ap = argparse.ArgumentParser(description="跨频带耦合特征提取（按 v1.0 读取 processed/v1.0）")
    ap.add_argument("--processed_root", required=True)
    ap.add_argument("--modality", default="audio", choices=["audio", "eeg", "ecg", "rppg"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out_dir", default="features")
    ap.add_argument("--coupling", default="AAC", help="AAC / PAC / PPC / COH，或逗号拼接")
    ap.add_argument("--n_subbands", type=int, default=8)
    ap.add_argument("--f_min", type=float, default=20.0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.from_json(args.config) if args.config else Config()
    cfg._d["coupling"] = args.coupling
    cfg._d["n_subbands"] = args.n_subbands
    cfg._d["f_min"] = args.f_min
    process_directory(args.processed_root, args.modality, args.out_dir, cfg, dataset=args.dataset)

"""h1b.py — 第 1 周 go/no-go 实验：关系特征 vs 四个对照条件。

主指标：LOSO pooled AUC（适配被试级标签 SEUMLD/DOLOS；按被试 AUC 仅在
被试内含两类时才有定义）。
判据：关系特征相对 Subject-Baseline-Norm 的 pooled AUC 增量 < go_nogo_margin
时判定 NO-GO（转关系结构诊断/负结果），否则 GO 进入主方法全量开发。
"""
from __future__ import annotations

import json
import os

import numpy as np

from baselines import BUILDERS
from config import Config
from metrics import auc, cohen_d, delong_test, subject_bootstrap_auc_ci, subject_bootstrap_diff_ci
from train_eval import load_features, run_loso, summarize_loso

BASELINES = ["full_band", "best_single_subband", "random_subband", "subject_baseline_norm"]


def run_h1b(features_dir: str, out_dir: str, cfg: Config) -> dict:
    band_power, relation, y, subjects, _ = load_features(features_dir)
    bundle = {"band_power": band_power, "relation": relation}
    if len(np.unique(y)) < 2:
        raise ValueError("标签只有一类，无法计算 AUC（请核对 label_mapped 0/1）")

    results, pooled_mean = {}, {}
    for cond in ["relation"] + BASELINES:
        if cond == "random_subband":
            cfg2 = Config(cfg.as_dict())
            cfg2._d["seeds"] = list(cfg.random_subband_seeds)
        else:
            cfg2 = cfg
        res = run_loso(bundle, y, subjects, BUILDERS[cond], cfg2)
        results[cond] = summarize_loso(res)
        pooled_mean[cond] = np.nanmean(res["pooled_scores"], axis=0)

    # 按被试 AUC（仅当存在含两类的被试时可用于 Cohen's d）
    usub = np.unique(subjects)
    rel_subj = np.full(usub.size, np.nan)
    for i, g in enumerate(usub):
        m = subjects == g
        rel_subj[i] = auc(y[m], pooled_mean["relation"][m])

    comparisons = {}
    for cond in BASELINES:
        base_subj = np.full(usub.size, np.nan)
        for i, g in enumerate(usub):
            m = subjects == g
            base_subj[i] = auc(y[m], pooled_mean[cond][m])
        mask = np.isfinite(rel_subj) & np.isfinite(base_subj)
        d = cohen_d(rel_subj[mask], base_subj[mask]) if mask.sum() >= 2 else np.nan
        delta = float(results["relation"]["pooled_auc"] - results[cond]["pooled_auc"])
        lo, hi = subject_bootstrap_diff_ci(
            y, pooled_mean["relation"], pooled_mean[cond], subjects, seed=cfg.seeds[0]
        )
        auc_rel, auc_base, z, p = delong_test(y, pooled_mean["relation"], pooled_mean[cond])
        comparisons[cond] = {
            "delta_auc": delta,
            "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi,
            "auc_relation_pooled": auc_rel,
            "auc_baseline_pooled": auc_base,
            "delong_z": z,
            "delong_p": p,
            "cohen_d": d,
        }

    margin = float(cfg.go_nogo_margin)
    delta_sbn = comparisons["subject_baseline_norm"]["delta_auc"]
    verdict = "GO" if (np.isfinite(delta_sbn) and delta_sbn >= margin) else "NO-GO"
    rel_ci = subject_bootstrap_auc_ci(y, pooled_mean["relation"], subjects, seed=cfg.seeds[0])

    report = {
        "verdict": verdict,
        "margin": margin,
        "delta_vs_subject_baseline_norm": delta_sbn,
        "relation_pooled_auc_ci": list(rel_ci),
        "conditions": results,
        "comparisons": comparisons,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "h1b_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("=" * 68)
    print("H1b 结果（LOSO pooled AUC; seeds=%s）" % list(cfg.seeds))
    print("-" * 68)
    for cond, summ in results.items():
        print("  %-24s pooled=%.4f ± %.4f   EER=%.4f   subj-mean=%.4f   worst=%.4f"
              % (cond, summ["pooled_auc"], summ["pooled_auc_std"], summ["pooled_eer"],
                 summ["subject_auc_mean"], summ["worst_subject_auc"]))
    print("-" * 68)
    for cond, cmp in comparisons.items():
        print("  relation - %-20s ΔAUC=%+.4f  CI=[%.4f, %.4f]  DeLong p=%.4f  d=%.3f"
              % (cond, cmp["delta_auc"], cmp["bootstrap_ci_low"], cmp["bootstrap_ci_high"],
                 cmp["delong_p"], cmp["cohen_d"]))
    print("-" * 68)
    print("  判定: %s (关系 vs Subject-Baseline-Norm 增量 %.4f, 阈值 %.4f)"
          % (verdict, delta_sbn, margin))
    print("=" * 68)
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="H1b go/no-go 实验")
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--out_dir", default="results/h1b")
    ap.add_argument("--model", default="svm", choices=["svm", "rf"])
    ap.add_argument("--weighting", default="none", choices=["none", "hsic", "mi"])
    ap.add_argument("--pos_class", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--go_nogo_margin", type=float, default=0.015)
    args = ap.parse_args()

    cfg = Config()
    cfg._d["model"] = args.model
    cfg._d["weighting"] = args.weighting
    cfg._d["pos_class"] = args.pos_class
    cfg._d["seeds"] = list(args.seeds)
    cfg._d["go_nogo_margin"] = args.go_nogo_margin
    run_h1b(args.features_dir, args.out_dir, cfg)

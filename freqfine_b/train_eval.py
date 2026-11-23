"""train_eval.py — SVM/RF + LOSO 验证（折内标准化/加权，多种子，输出临时统计量）。"""
from __future__ import annotations

import csv
import os
from typing import Tuple

import numpy as np

from metrics import auc, eer


def load_features(features_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    manifest_path = os.path.join(features_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError("找不到 %s，请先运行 cfc_extractor.py" % manifest_path)

    with open(manifest_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("manifest.csv 为空")

    bp, rel, yy, ss, ids = [], [], [], [], []
    for row in rows:
        fp = os.path.join(features_dir, row["feature_file"].replace("\\", "/"))
        with np.load(fp, allow_pickle=False) as z:
            bp.append(np.asarray(z["band_power"], dtype=np.float64))
            rel.append(np.asarray(z["relation"], dtype=np.float64))
        yy.append(int(row["label"]))
        ss.append(row["subject_id"])
        ids.append(row["sample_id"])

    return (
        np.stack(bp, axis=0),
        np.stack(rel, axis=0),
        np.asarray(yy, dtype=int),
        np.asarray(ss),
        ids,
    )


def make_model(name: str = "svm", random_state: int = 0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    name = name.lower()
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                   random_state=random_state, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state,
                                      class_weight="balanced")
    raise ValueError("model 必须是 svm 或 rf")


def _proba_index(model, pos_class: int) -> int:
    for i, c in enumerate(model.classes_):
        if int(c) == int(pos_class):
            return i
    raise ValueError("模型类别 %s 中找不到 pos_class=%d" % (list(model.classes_), pos_class))


def run_loso(bundle, y, subjects, builder, cfg) -> dict:
    y = np.asarray(y, dtype=int)
    subjects = np.asarray(subjects)
    n = y.size
    seeds = list(cfg.seeds)
    pooled = np.full((len(seeds), n), np.nan, dtype=np.float64)
    per_seed_subject_auc = []

    for si, seed in enumerate(seeds):
        rng = np.random.RandomState(seed)
        for g in np.unique(subjects):
            tr = np.where(subjects != g)[0]
            te = np.where(subjects == g)[0]
            if tr.size == 0 or te.size == 0:
                continue
            if np.unique(y[tr]).size < 2:
                prior = float(np.mean(y[tr] == cfg.pos_class))
                pooled[si, te] = prior
                continue
            Xtr, Xte = builder(bundle, tr, te, y[tr], subjects[tr], rng, cfg)
            model = make_model(cfg.model, random_state=seed)
            model.fit(Xtr, y[tr])
            proba = model.predict_proba(Xte)
            pooled[si, te] = proba[:, _proba_index(model, cfg.pos_class)]

        sub_auc = {}
        for g in np.unique(subjects):
            m = subjects == g
            sub_auc[g] = auc(y[m], pooled[si, m])
        per_seed_subject_auc.append(sub_auc)

    return {
        "pooled_scores": pooled,
        "y": y,
        "subjects": subjects,
        "per_seed_subject_auc": per_seed_subject_auc,
    }


def summarize_loso(res: dict) -> dict:
    """以 pooled AUC 为主指标（适配被试级标签）；按被试 AUC 仅在含两类时可用。"""
    subjects = res["subjects"]
    usub = np.unique(subjects)

    per_seed_pooled = []
    for si in range(len(res["pooled_scores"])):
        a = auc(res["y"], res["pooled_scores"][si])
        if np.isfinite(a):
            per_seed_pooled.append(a)
    pooled = np.nanmean(res["pooled_scores"], axis=0)
    pooled_auc = auc(res["y"], pooled)
    pooled_eer = eer(res["y"], pooled)

    sub_aucs = []
    for g in usub:
        m = subjects == g
        a = auc(res["y"][m], pooled[m])
        if np.isfinite(a):
            sub_aucs.append(a)

    return {
        "pooled_auc": pooled_auc,
        "pooled_auc_std": float(np.std(per_seed_pooled)) if per_seed_pooled else np.nan,
        "pooled_eer": pooled_eer,
        "subject_auc_mean": float(np.nanmean(sub_aucs)) if sub_aucs else np.nan,
        "worst_subject_auc": float(np.min(sub_aucs)) if sub_aucs else np.nan,
        "n_subjects_with_both_classes": len(sub_aucs),
        "n_seeds": len(res["pooled_scores"]),
    }


if __name__ == "__main__":
    import argparse

    from baselines import BUILDERS
    from config import Config

    ap = argparse.ArgumentParser(description="SVM/RF + LOSO 验证")
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--condition", default="relation", choices=list(BUILDERS.keys()))
    ap.add_argument("--model", default="svm", choices=["svm", "rf"])
    ap.add_argument("--weighting", default="none", choices=["none", "hsic", "mi"])
    ap.add_argument("--pos_class", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    cfg = Config()
    cfg._d["model"] = args.model
    cfg._d["weighting"] = args.weighting
    cfg._d["pos_class"] = args.pos_class
    cfg._d["seeds"] = list(args.seeds)

    band_power, relation, y, subjects, _ = load_features(args.features_dir)
    bundle = {"band_power": band_power, "relation": relation}
    res = run_loso(bundle, y, subjects, BUILDERS[args.condition], cfg)
    summ = summarize_loso(res)
    print("[train_eval] condition=%s model=%s weighting=%s" % (args.condition, args.model, args.weighting))
    print("[train_eval] pooled AUC=%.4f ± %.4f (seeds); EER=%.4f; subject-AUC mean=%.4f; worst=%.4f"
          % (summ["pooled_auc"], summ["pooled_auc_std"], summ["pooled_eer"],
             summ["subject_auc_mean"], summ["worst_subject_auc"]))

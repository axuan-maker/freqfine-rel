"""train_eval.py — 特征缓存读取与分类器工厂（供 protocol_eval / dl_baseline 复用）。"""
from __future__ import annotations

import csv
import os
from typing import Tuple

import numpy as np


def load_features(features_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    manifest_path = os.path.join(features_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError("找不到 %s，请先运行 extract_delivery.py" % manifest_path)

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

"""delivery_loader.py — 读取 E:\\处理后 实际交付（对齐 read_split_audio.py 约定）。

与 v1.0 规范的三点差异（以实际交付为准）：
1. NPZ 数组键就是 signal / valid_mask / ...（np.load().files 返回的键名不带
   .npy 后缀；.npy 只是 zip 包内部的条目名）；
2. 主标签在 音频清单.csv 的 project_label 列（0=真, 1=假），不在 NPZ/JSON 内；
3. 训练/验证/测试来自显式目录 <dataset>/划分/<protocol>/<role>/，必须一次选一个协议。
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, Iterator, List, Optional

import numpy as np


def _manifest_path(root: str, dataset: str, protocol: str, role: str) -> str:
    return os.path.join(root, dataset, "划分", protocol, role, "音频清单.csv")


def _resolve(root: str, rel: str) -> str:
    return os.path.join(root, rel.replace("/", os.sep))


def iter_audio(
    root: str,
    dataset: str,
    protocol: str,
    role: str,
) -> Iterator[Dict]:
    """逐样本读取一个角色（train/validation/test）的音频，返回信号 + 主标签 + 分组键。"""
    with open(_manifest_path(root, dataset, protocol, role), newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        arr_path = _resolve(root, row["array_path"])
        meta_path = _resolve(root, row["metadata_path"])
        with np.load(arr_path, allow_pickle=False) as z:
            sig = np.asarray(z["signal"], dtype=np.float64)
            valid = (
                np.asarray(z["valid_mask"], dtype=bool)
                if "valid_mask" in z.files
                else np.ones_like(sig, dtype=bool)
            )
        with open(meta_path, encoding="utf-8") as fh2:
            meta = json.load(fh2)

        signal = np.where(valid, sig, np.nan)
        yield {
            "sample_id": row["sample_id"],
            "subject_id": row["subject_id"],
            "dataset": row["dataset"],
            "protocol_id": row["protocol_id"],
            "role": row["role"],
            "label": int(row["project_label"]),
            "label_level": row["label_level"],
            "group_key": row["group_key"],
            "parent_group": row["parent_group"],
            "content_group_id": row["content_group_id"],
            "dependency_group_id": row["dependency_group_id"],
            "signal": signal,
            "valid_fraction": float(valid.mean()),
            "sr_hz": float(meta.get("sr_hz") or 0.0),
            "quality_approved": row.get("quality_approved", "False").lower() == "true",
            "meta": meta,
        }


def load_role(root: str, dataset: str, protocol: str, role: str) -> List[Dict]:
    return list(iter_audio(root, dataset, protocol, role))

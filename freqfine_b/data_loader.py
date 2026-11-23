"""data_loader.py — 按 v1.0 规范读取 A 交付的 processed/v1.0 数据。

只依赖 numpy + 标准库。
- 从 manifests/ 读取 samples / modalities / labels_primary / split_manifest；
- 标签只取 label_status=verified 且 label_mapped∈{0,1}；
- NPZ 用 np.load(..., allow_pickle=False)，并成对读取同名 JSON；
- 尊重 valid_mask（无效点置 NaN，默认不插值）。
"""
from __future__ import annotations

import csv
import json
import os
import warnings
from typing import Iterator, List, Optional

import numpy as np

__all__ = ["DataCatalog", "Sample", "read_csv_rows", "resolve_relative"]

_BOM = "utf-8-sig"


def read_csv_rows(path: str, required: Optional[List[str]] = None) -> List[dict]:
    required = required or []
    if not os.path.isfile(path):
        raise FileNotFoundError("缺少清单文件: %s" % path)
    with open(path, newline="", encoding=_BOM) as fh:
        rows = list(csv.DictReader(fh))
    if required and rows:
        missing = [c for c in required if c not in rows[0]]
        if missing:
            raise KeyError("%s 缺少必需列: %s" % (path, ", ".join(missing)))
    return rows


def resolve_relative(root: str, rel: Optional[str]) -> Optional[str]:
    if rel is None or rel == "":
        return None
    rel = rel.replace("\\", "/").lstrip("/")
    return os.path.join(root, *rel.split("/"))


class Sample:
    __slots__ = (
        "sample_id",
        "subject_id",
        "dataset",
        "modality",
        "label",
        "array_path",
        "metadata_path",
        "identity_verified",
    )

    def __init__(self, sample_id, subject_id, dataset, modality, label,
                 array_path, metadata_path, identity_verified=True):
        self.sample_id = sample_id
        self.subject_id = subject_id
        self.dataset = dataset
        self.modality = modality
        self.label = label
        self.array_path = array_path
        self.metadata_path = metadata_path
        self.identity_verified = identity_verified

    def load_signal(self):
        """返回 (signal, meta, valid_mask)。无效点置 NaN，不做插值。"""
        if not os.path.isfile(self.array_path):
            raise FileNotFoundError("NPZ 不存在: %s" % self.array_path)
        if not self.metadata_path or not os.path.isfile(self.metadata_path):
            raise FileNotFoundError("JSON 不存在: %s" % self.metadata_path)

        with np.load(self.array_path, allow_pickle=False) as z:
            if "signal" not in z.files:
                raise KeyError("%s 缺少 signal 键" % self.array_path)
            signal = np.asarray(z["signal"], dtype=np.float64)
            if "valid_mask" in z.files:
                valid = np.asarray(z["valid_mask"], dtype=bool)
            else:
                valid = np.ones_like(signal, dtype=bool)
            if valid.shape != signal.shape:
                raise ValueError("%s valid_mask 形状 %s 与 signal %s 不一致"
                                 % (self.array_path, valid.shape, signal.shape))

        with open(self.metadata_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        signal = np.where(valid, signal, np.nan)
        return signal, meta, valid


class DataCatalog:
    def __init__(self, processed_root: str):
        self.root = processed_root
        self.manifests = os.path.join(processed_root, "manifests")
        self.samples = read_csv_rows(
            os.path.join(self.manifests, "samples.csv"),
            required=["sample_id", "subject_id", "subject_id_status", "dataset"],
        )
        self.modalities = read_csv_rows(
            os.path.join(self.manifests, "modalities.csv"),
            required=["sample_id", "modality", "status", "array_path", "metadata_path"],
        )
        self.labels = read_csv_rows(
            os.path.join(self.manifests, "labels_primary.csv"),
            required=["dataset", "sample_id", "label_mapped", "label_status"],
        )
        split_path = os.path.join(processed_root, "splits", "split_manifest.csv")
        self.splits = read_csv_rows(split_path) if os.path.isfile(split_path) else []

    def _verified_labels(self) -> dict:
        out = {}
        for row in self.labels:
            if row.get("label_status") != "verified":
                continue
            try:
                lab = int(float(row["label_mapped"]))
            except (TypeError, ValueError):
                continue
            if lab not in (0, 1):
                continue
            out[row["sample_id"]] = lab
        return out

    def _modality_index(self, modality: str) -> dict:
        out = {}
        for row in self.modalities:
            if row.get("modality") != modality or row.get("status") != "ready":
                continue
            sid = row["sample_id"]
            out[sid] = {
                "array_path": resolve_relative(self.root, row.get("array_path")),
                "metadata_path": resolve_relative(self.root, row.get("metadata_path")),
            }
        return out

    def _identity_map(self) -> dict:
        out = {}
        for row in self.splits:
            sid = row.get("sample_id", "")
            if sid:
                out[sid] = str(row.get("identity_verified", "True")).lower() in ("true", "1", "yes")
        return out

    def iter_samples(self, modality: str, dataset: Optional[str] = None) -> Iterator[Sample]:
        labels = self._verified_labels()
        mods = self._modality_index(modality)
        identity = self._identity_map()

        seen = set()
        for row in self.samples:
            ds = row.get("dataset")
            if dataset is not None and ds != dataset:
                continue
            sid = row["sample_id"]
            if sid in seen:
                continue
            seen.add(sid)
            if sid not in mods:
                continue
            if sid not in labels:
                continue

            subject = row.get("subject_id") or row.get("raw_subject_id") or sid
            subject_status = row.get("subject_id_status", "unknown")
            id_verified = identity.get(sid, subject_status == "verified")
            if subject_status != "verified":
                warnings.warn("样本 %s 被试身份未核验，LOSO 分组仅作近似" % sid)

            yield Sample(
                sample_id=sid,
                subject_id=subject,
                dataset=ds or "",
                modality=modality,
                label=labels[sid],
                array_path=mods[sid]["array_path"],
                metadata_path=mods[sid]["metadata_path"],
                identity_verified=id_verified,
            )

"""extract_delivery.py — 从 E:\\处理后 的一个 (dataset, protocol, role) 提取耦合特征。

输出与 train_eval.load_features 兼容的 manifest.csv + 每样本 npz，
并把 group_key / parent_group / content_group_id / dependency_group_id
写进 manifest，供下游按防泄露分组键校验。
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from cfc_extractor import extract_sample_features
from config import Config
from delivery_loader import iter_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\处理后")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--role", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--coupling", default="AAC")
    ap.add_argument("--n_subbands", type=int, default=8)
    ap.add_argument("--f_min", type=float, default=20.0)
    ap.add_argument("--f_max", type=float, default=None)
    args = ap.parse_args()

    cfg = Config()
    cfg._d["coupling"] = args.coupling
    cfg._d["n_subbands"] = args.n_subbands
    cfg._d["f_min"] = args.f_min
    cfg._d["f_max"] = args.f_max

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    for s in iter_audio(args.root, args.dataset, args.protocol, args.role):
        feats = extract_sample_features(s["signal"], s["sr_hz"], cfg)
        stem = s["sample_id"]
        np.savez_compressed(
            os.path.join(args.out_dir, stem + ".npz"),
            band_power=feats["band_power"].astype(np.float32),
            relation=feats["relation"].astype(np.float32),
        )
        manifest.append({
            "sample_id": s["sample_id"],
            "subject_id": s["subject_id"],
            "label": s["label"],
            "label_level": s["label_level"],
            "group_key": s["group_key"],
            "parent_group": s["parent_group"],
            "content_group_id": s["content_group_id"],
            "dependency_group_id": s["dependency_group_id"],
            "role": s["role"],
            "protocol_id": s["protocol_id"],
            "valid_fraction": round(s["valid_fraction"], 6),
            "feature_file": stem + ".npz",
        })

    with open(os.path.join(args.out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print("[extract_delivery] %s/%s/%s: %d 个样本 -> %s"
          % (args.dataset, args.protocol, args.role, len(manifest), args.out_dir))


if __name__ == "__main__":
    main()

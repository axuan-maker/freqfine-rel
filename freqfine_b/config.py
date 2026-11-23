"""config.py — FreqFine-Rel 运行配置（与数据接口无关的纯参数）。"""
from __future__ import annotations

import json

_DEFAULTS = {
    "n_subbands": 8,
    "f_min": 20.0,
    "f_max": None,
    "filter_order": 4,
    "coupling": "AAC",
    "coherence_nperseg": 256,
    "pac_n_bins": 18,
    "add_band_power": False,
    "weighting": "none",
    "weight_tau": 1.0,
    "model": "svm",
    "protocol": "loso",
    "n_group_folds": 5,
    "seeds": [0, 1, 2],
    "random_subband_seeds": [0, 1, 2, 3, 4],
    "pos_class": 1,
    "go_nogo_margin": 0.015,
    "n_modules": None,
}


class Config:
    def __init__(self, values=None):
        self._d = dict(_DEFAULTS)
        if values:
            self._d.update({k: v for k, v in values.items() if k in self._d})

    def __getattr__(self, name):
        if name in self._d:
            return self._d[name]
        raise AttributeError(name)

    def as_dict(self):
        return dict(self._d)

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

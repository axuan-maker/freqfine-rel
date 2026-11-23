# freqfine_b — FreqFine-Rel 建模代码（E 盘交付版）

按 `E:\处理后` 实际交付运行的跨频带耦合（CFC）建模代码。数据协议为
`<dataset>/划分/<protocol>/<role>/音频清单.csv + 音频/*.npz/*.json`，
标签在清单的 `project_label` 列（0=真、1=假）。

## 文件

- `delivery_loader.py`：读 E 盘交付（NPZ 键为 `signal`/`valid_mask` 等，不带 `.npy`）。
- `extract_delivery.py`：对 (dataset, protocol, role) 提取特征缓存（`band_power` + `relation`）。
- `cfc_extractor.py`：子带分解、Hilbert 包络/相位、AAC/PAC/PPC/COH、上三角关系向量、谱聚类模块发现。COH 为包络相干，PAC 另有 `calculate_PAC_zscore` 代理归一化版本。
- `weak_supervision.py`：序数弱监督加权（HSIC/互信息，折内拟合）。
- `baselines.py`：full_band / best_single_subband / random_subband / subject_baseline_norm 四个基线构造。
- `metrics.py`：AUC / EER / DeLong / Bootstrap / Cohen's d。
- `train_eval.py`：特征缓存读取 + SVM/RF 工厂。
- `protocol_eval.py`：按 train/validation/test 折跑 H1a / H1b（正式协议入口）。
- `dl_baseline.py`：轻量 1D-CNN 防守基线（LOSO）。
- `config.py`：运行参数；`requirements.txt`：依赖。

## 运行顺序（单库单模态）

```bash
# 1. 提取一个折三个角色的特征（SEUMLD 五折：seu_fold0_v1 .. seu_fold4_v1）
python extract_delivery.py --root /cloud/处理后 --dataset SEUMLD --protocol seu_fold0_v1 \
  --role train --out_dir /cloud/features/seu_fold0_v1/train --coupling AAC --n_subbands 8
python extract_delivery.py --root /cloud/处理后 --dataset SEUMLD --protocol seu_fold0_v1 \
  --role validation --out_dir /cloud/features/seu_fold0_v1/validation --coupling AAC --n_subbands 8
python extract_delivery.py --root /cloud/处理后 --dataset SEUMLD --protocol seu_fold0_v1 \
  --role test --out_dir /cloud/features/seu_fold0_v1/test --coupling AAC --n_subbands 8

# 2. H1a 稀疏性 + H1b 关系增量（go/no-go）
python protocol_eval.py h1a --features_root /cloud/features \
  --protocols seu_fold0_v1 seu_fold1_v1 seu_fold2_v1 seu_fold3_v1 seu_fold4_v1 \
  --out_dir /cloud/results/h1a
python protocol_eval.py h1b --features_root /cloud/features \
  --protocols seu_fold0_v1 seu_fold1_v1 seu_fold2_v1 seu_fold3_v1 seu_fold4_v1 \
  --out_dir /cloud/results/h1b --model svm --weighting none

# 3. 深度基线（可选，需 torch）
python dl_baseline.py --features_dir /cloud/features/seu_fold0_v1/train
```

`extract_delivery.py` 对单个失败样本会跳过并写入 `skipped.csv`，不中断整折提取。
评估口径：train 拟合、validation 选参/参考、test 报告，多折取均值±标准差；
H1b 的 go/no-go 看关系向量相对 subject_baseline_norm 的 test AUC 增量（默认阈值 1.5%）。

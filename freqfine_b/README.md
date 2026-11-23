# freqfine_b — 修正版（对齐 数据文件制定规范 v1.0）

本目录是 Role B 建模代码的修正版。相比上一版的关键变化：

1. `data_loader.py` 改为按 v1.0 规范读取 `processed/v1.0/`：从 `manifests/`
   清单取标签/被试/模态可用性，NPZ 与 JSON 成对读取，尊重 `valid_mask`，
   信号保持原始 `(T,)` / `(13,T)`，采样率取自 JSON 的 `sr_hz`。
2. `cfc_extractor.py` 直接对原始信号做子带分解，不再“双重分解”；补上
   `PPC` 与 `COH` 两种耦合；新增关系模块谱聚类诊断；特征落盘时记录频带
   边界、顺序与采样率（feature 侧 meta）。
3. `weak_supervision.py` 权重改为在训练折内估计（由 `baselines.py` 的
   `build_relation` 调用），不再全库拟合。
4. `baselines.py` + `h1b.py` 补齐 H1b 四条件对照与 go/no-go 报告。
5. `train_eval.py` 折内标准化、多种子、SVM/RF 均衡类别；`dl_baseline.py`
   改为 LOSO + 折内早停。

## 运行顺序（单库单模态）

```bash
# 1. 特征提取（读 processed/v1.0）
python cfc_extractor.py --processed_root processed/v1.0 --modality audio \
    --dataset SEUMLD --out_dir results/features --coupling AAC --n_subbands 8

# 2. H1a 稀疏性检验
python h1a.py --features_dir results/features --out_dir results/h1a

# 3. H1b go/no-go
python h1b.py --features_dir results/features --out_dir results/h1b \
    --model svm --weighting hsic

# 4. 主方法单条件复跑
python train_eval.py --features_dir results/features --condition relation \
    --model svm --weighting hsic

# 5. 深度基线（可选，需 torch）
python dl_baseline.py --features_dir results/features
```

依赖见 `requirements.txt`（推荐 Python 3.9–3.10）。

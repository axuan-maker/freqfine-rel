"""dl_baseline.py — 轻量级 1D-CNN 防守基线（LOSO，折内早停）。"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class CouplingDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(np.asarray(features, dtype=np.float32))
        self.y = torch.tensor(np.asarray(labels, dtype=np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LightCNN(nn.Module):
    def __init__(self, in_features: int = 120, n_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def _class_weight_tensor(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0.0, 1.0, counts)
    w = 1.0 / counts
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


def _predict_proba(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs) if probs else np.empty((0, model.fc.out_features))


def _train_once(Xtr, ytr, Xva, yva, in_features, n_classes, epochs, batch_size,
                lr, dropout, patience, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = CouplingDataset(Xtr, ytr)
    val_ds = CouplingDataset(Xva, yva)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LightCNN(in_features=in_features, n_classes=n_classes, dropout=dropout).to(device)
    weight = _class_weight_tensor(ytr, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_loss = float("inf")
    best_state = None
    bad = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        va_loss, n_va = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += float(criterion(model(xb), yb).item()) * xb.size(0)
                n_va += xb.size(0)
        va_loss = va_loss / max(n_va, 1)
        if va_loss < best_loss - 1e-4:
            best_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_dl(X, y, subjects, pos_class=1, epochs=100, batch_size=64, lr=1e-3,
           dropout=0.4, patience=10, val_frac=0.2, seed=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).reshape(-1)
    subjects = np.asarray(subjects).reshape(-1)
    y_work = (y == pos_class).astype(np.int64)
    n_classes = 2

    pooled = np.full(X.shape[0], np.nan, dtype=np.float64)
    groups = np.unique(subjects)
    for g in groups:
        tr = np.where(subjects != g)[0]
        te = np.where(subjects == g)[0]
        if tr.size == 0 or te.size == 0 or np.unique(y_work[tr]).size < 2:
            pooled[te] = float(np.mean(y_work[tr])) if tr.size else 0.5
            continue

        tr_subjects = np.unique(subjects[tr])
        rng = np.random.RandomState(seed)
        rng.shuffle(tr_subjects)
        n_val = max(1, int(round(len(tr_subjects) * val_frac)))
        val_subjects = set(tr_subjects[:n_val].tolist())
        va_mask = np.isin(subjects, list(val_subjects)) & tr
        train_mask = tr & ~va_mask

        model = _train_once(
            X[train_mask], y_work[train_mask], X[va_mask], y_work[va_mask],
            in_features=int(X.shape[1]), n_classes=n_classes, epochs=epochs,
            batch_size=batch_size, lr=lr, dropout=dropout, patience=patience,
            seed=seed, device=device,
        )
        test_ds = CouplingDataset(X[te], y_work[te])
        proba = _predict_proba(model, DataLoader(test_ds, batch_size=batch_size, shuffle=False), device)
        pooled[te] = proba[:, 1]

    return pooled, y_work, subjects


if __name__ == "__main__":
    from train_eval import load_features

    ap = argparse.ArgumentParser(description="轻量级 1D-CNN 防守基线（LOSO）")
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--pos_class", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _, relation, y, subjects, _ = load_features(args.features_dir)
    pooled, y_work, _ = run_dl(
        relation, y, subjects, pos_class=args.pos_class, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, dropout=args.dropout,
        patience=args.patience, val_frac=args.val_frac, seed=args.seed,
    )
    from metrics import auc
    print("[dl_baseline] LOSO pooled AUC=%.4f" % auc(y_work, pooled))

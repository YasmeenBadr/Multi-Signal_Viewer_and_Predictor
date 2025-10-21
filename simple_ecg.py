import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import wfdb
import os
from glob import glob
import re
import random

# -------------------------
# 1. CONFIGURATION
# -------------------------
DATA_FOLDER = r"uploads"
MODEL_SAVE_PATH = "simple_ecg_model.pt"
TARGET_LEN = 5000
CHANNEL_IDX = 0
ALIAS_FS = [250, 200, 150, 100, 80, 60, 40, 25]
# Probability to include an aliased copy for each record (kept low to preserve sensitivity to aliasing)
ALIAS_AUG_PROB = 0.0

# -------------------------
# 2. LOAD ECG SIGNALS
# -------------------------
print("Loading ECG signals...")
X_list = []
y_list = []

def _decimate_alias(sig, native_fs, target_fs):
    sig = np.asarray(sig, dtype=np.float32)
    if target_fs <= 0 or target_fs >= native_fs:
        return sig
    factor = float(native_fs) / float(target_fs)
    k = int(np.floor(factor))
    if k >= 2 and abs(factor - k) < 1e-6:
        return sig[::k]
    idx = np.floor(np.arange(0, sig.shape[0], factor)).astype(np.int64)
    if idx.size == 0:
        return sig[:1]
    idx = np.clip(idx, 0, sig.shape[0]-1)
    idx = np.unique(idx)
    return sig[idx]

def _resize_to_len(x, L):
    x = np.asarray(x, dtype=np.float32).flatten()
    if x.size == L:
        return x
    if x.size <= 1:
        return np.pad(x, (0, max(0, L - x.size)))[:L]
    xp = np.linspace(0.0, 1.0, num=x.size, endpoint=True)
    xq = np.linspace(0.0, 1.0, num=L, endpoint=True)
    return np.interp(xq, xp, x).astype(np.float32)

# Expect subfolders or files like s0010_re.dat + s0010_re.hea
for hea_file in glob(os.path.join(DATA_FOLDER, "*.hea")):
    base = hea_file[:-4]
    dat_file = base + ".dat"
    if not os.path.exists(dat_file):
        continue
    try:
        record = wfdb.rdrecord(base)
        sig = record.p_signal[:, CHANNEL_IDX]
        fs = int(record.fs) if hasattr(record, "fs") else 500
        with open(hea_file, "r", encoding="latin-1") as f:
            text = f.read().lower()
        # Robust abnormal detection: use word boundaries, avoid false matches like 'mV' or 'st' in other words
        abn_terms = [
            r"myocardial", r"myocardial infarction", r"\bmi\b", r"infarct", r"ischemia", r"ischemic",
            r"atrial fibrillation", r"\baf\b",
            r"left bundle branch block", r"\blbbb\b",
            r"right bundle branch block", r"\brbbb\b",
            r"first[- ]?degree av block", r"1d\s*avb", r"\b1davb\b", r"av block",
            r"bradycardia", r"\bsb\b", r"tachycardia",
            r"st[- ]segment"  # specific to ST-segment
        ]
        pattern = re.compile("|".join(abn_terms))
        label = 1 if pattern.search(text) else 0
        x0 = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
        x0 = _resize_to_len(x0, TARGET_LEN)
        X_list.append(x0)
        y_list.append(label)
        # Controlled aliasing augmentation: keep probability low to avoid making the model too robust to aliasing
        if random.random() < ALIAS_AUG_PROB:
            for f_alias in ALIAS_FS:
                if f_alias < fs:
                    xa = _decimate_alias(sig, fs, f_alias)
                    xa = _resize_to_len(xa, TARGET_LEN)
                    xa = (xa - np.mean(xa)) / (np.std(xa) + 1e-8)
                    X_list.append(xa)
                    y_list.append(label)
        print(f"Loaded {base}: label={label}, fs={fs}, aug={sum(1 for f in ALIAS_FS if f < fs)}")
    except Exception as e:
        print(f"Failed to load {base}: {e}")

if len(X_list) < 2:
    raise RuntimeError("Not enough ECG records found for training!")

X_np = np.array(X_list, dtype=np.float32)
y_np = np.array(y_list, dtype=np.int64)
X = torch.tensor(X_np).unsqueeze(1)
y = torch.tensor(y_np, dtype=torch.long)

# Stratified 80/20 split
rng = np.random.RandomState(42)
idx_pos = np.where(y_np == 1)[0]
idx_neg = np.where(y_np == 0)[0]
rng.shuffle(idx_pos); rng.shuffle(idx_neg)
cut_pos = int(0.8 * len(idx_pos))
cut_neg = int(0.8 * len(idx_neg))
train_idx = np.concatenate([idx_pos[:cut_pos], idx_neg[:cut_neg]])
val_idx = np.concatenate([idx_pos[cut_pos:], idx_neg[cut_neg:]])
rng.shuffle(train_idx); rng.shuffle(val_idx)

train_ds = TensorDataset(X[train_idx], y[train_idx])
val_ds = TensorDataset(X[val_idx], y[val_idx])
loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# Recompute criterion with class weights to reduce bias
classes, counts = np.unique(y_list, return_counts=True)
weights = np.ones(2, dtype=np.float32)
for cls, cnt in zip(classes, counts):
    # Inverse frequency weighting
    weights[int(cls)] = float(len(y_list) / (2.0 * max(1, cnt)))
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

# -------------------------
# 3. MODEL
# -------------------------
class SimpleECG(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * (TARGET_LEN // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleECG()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# 4. TRAIN
# -------------------------
print("Training model (native baseline first)...")
EPOCHS = 50
best_val_loss = float('inf')
best_state = None
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (preds.argmax(1) == yb).sum().item()
        total += xb.size(0)

    train_loss = total_loss / max(1, total)
    train_acc = correct / max(1, total)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
            val_correct += (preds.argmax(1) == yb).sum().item()
            val_total += xb.size(0)
    val_loss = val_loss / max(1, val_total)
    val_acc = val_correct / max(1, val_total)

    # Save best checkpoint (lowest val_loss)
    if val_loss < best_val_loss and val_total > 0:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch+1}/{EPOCHS} | Train: loss {train_loss:.4f}, acc {train_acc:.3f} | Val: loss {val_loss:.4f}, acc {val_acc:.3f}")

# -------------------------
# 5. SAVE MODEL
# -------------------------
if best_state is not None:
    model.load_state_dict(best_state)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved as {MODEL_SAVE_PATH}")
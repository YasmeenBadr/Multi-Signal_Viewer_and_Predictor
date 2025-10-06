import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import wfdb
import os

# -------------------------
# 1. CONFIGURATION
# -------------------------
DATA_PATH = r"data1/s0010_re"  # Path without extension
MODEL_SAVE_PATH = "simple_ecg_model.pt"
TARGET_LEN = 5000

# -------------------------
# 2. LOAD ECG SIGNAL
# -------------------------
print("Loading ECG signal...")
try:
    record = wfdb.rdrecord(DATA_PATH)
    sig = record.p_signal[:, 0]  # first channel
    print(f"Loaded signal shape: {sig.shape}")
except Exception as e:
    raise RuntimeError(f"Error loading ECG data: {e}")

# normalize signal
sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
# simulate abnormal signal for training
sig_abn = sig_norm * 0.8 + np.random.randn(len(sig_norm)) * 0.05

# pad or trim
def fix_length(x):
    if len(x) < TARGET_LEN:
        return np.pad(x, (0, TARGET_LEN - len(x)))
    return x[:TARGET_LEN]

X = np.array([fix_length(sig_norm), fix_length(sig_abn)])
y = np.array([0, 1])  # 0=Normal, 1=Abnormal

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

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
            nn.Linear(32*1250, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

model = SimpleECG()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# 4. TRAIN
# -------------------------
print("Training model...")
for epoch in range(15):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# -------------------------
# 5. SAVE
# -------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved as {MODEL_SAVE_PATH}")

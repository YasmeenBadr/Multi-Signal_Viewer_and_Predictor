import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import wfdb
import os

# ----------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------
DATA_PATH = r"data1/s0543_re"   # Path without extension
MODEL_SAVE_PATH = "simple_ecg_model.pt"

# ----------------------------------------------------------
# 2. LOAD ECG SIGNAL
# ----------------------------------------------------------
print("Loading ECG signal...")

try:
    record = wfdb.rdrecord(DATA_PATH)
    sig = record.p_signal[:, 0]  # Use the first channel
    print(f"Loaded signal shape: {sig.shape}")
except Exception as e:
    raise RuntimeError(f"Error loading ECG data: {e}")

# Trim or pad the signal to fixed length (e.g., 5000 samples)
target_len = 5000
if len(sig) > target_len:
    sig = sig[:target_len]
else:
    sig = np.pad(sig, (0, target_len - len(sig)))

# Create dummy dataset (one healthy, one abnormal example)
X = np.array([sig, sig * 0.5])  # simulate two different examples
y = np.array([0, 1])            # 0=healthy, 1=abnormal

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (batch, channel, length)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ----------------------------------------------------------
# 3. DEFINE SIMPLE MODEL
# ----------------------------------------------------------
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
            nn.Linear(32 * 1250, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------------------------------------
# 4. TRAIN THE MODEL
# ----------------------------------------------------------
model = SimpleECG()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training model...")
for epoch in range(10):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# ----------------------------------------------------------
# 5. SAVE MODEL
# ----------------------------------------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved as {MODEL_SAVE_PATH}")

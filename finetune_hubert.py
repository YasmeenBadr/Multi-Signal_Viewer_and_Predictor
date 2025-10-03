# finetune_hubert.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

from hubert_ecg.hubert_ecg import HuBERTECG, HuBERTECGConfig
from utils.ptbxl_loader import PTBXLSignalDataset

def collate_fn(batch):
    # batch: list of (tensor_signal, label)
    # We must pad/truncate sequences to a fixed length. Choose 5000 samples as example.
    # Adjust seq_len depending on PTB-XL record length (usually 5000 at 100Hz for 50s).
    seq_len = 5000
    xs, ys = [], []
    for s, y in batch:
        # s may be 1D (n_samples) or 2D (n_leads, n_samples)
        if s.dim() == 1:
            arr = s
        else:
            # if multi-lead, flatten leads by averaging or choose lead 0
            # Here we average across leads to get single channel (alternatively choose primary lead)
            arr = s.mean(dim=0)
        # pad or trim
        if arr.shape[0] < seq_len:
            pad = torch.zeros(seq_len - arr.shape[0], dtype=torch.float32)
            arr = torch.cat([arr, pad], dim=0)
        else:
            arr = arr[:seq_len]
        xs.append(arr.unsqueeze(0))  # shape (1, seq_len)
        ys.append(y)
    X = torch.stack(xs, dim=0)  # (batch, 1, seq_len)
    X = X.squeeze(1)  # many HuBERT variants expect (batch, seq_len)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = args.ptbxl_path
    print("Loading dataset metadata from", base_path)
    dataset = PTBXLSignalDataset(base_path=base_path, select_leads=None, max_records=args.max_records)
    label_map = dataset.get_label_map()
    num_classes = len(label_map)
    print("Found classes:", label_map)

    # split
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = n - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # load resaved checkpoint
    ckpt_path = args.resaved_ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("model_config", None)
    if isinstance(config, dict):
        config = HuBERTECGConfig(**config)
    model = HuBERTECG(config)
    # load weights (non-strict to allow head replacement)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Replace classifier head: add simple linear classifier
    hidden_size = getattr(config, "hidden_size", 768)
    classifier = nn.Linear(hidden_size, num_classes)
    # Attach classifier to model: the exact attribute name may differ; we wrap forward accordingly below
    model.classifier = classifier

    # Move to device
    model = model.to(device)

    # optimizer only fine-tune classifier and optionally last layers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}")
        running_loss = 0.0
        for X, y in pbar:
            X = X.to(device)  # shape (batch, seq_len) or (batch, 1, seq_len)
            y = y.to(device)
            optimizer.zero_grad()
            # Forward: adapt to HuBERTECG forward signature.
            # Many transformer-based ECG models expect (batch, seq). If errors occur, try X.unsqueeze(1).
            out = model(X)  # expected (batch, num_classes) or feature vector depending on model
            # If model returns features, pass through classifier
            if out.shape[-1] != num_classes:
                # assume out shape (batch, hidden)
                out = model.classifier(out)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                yv = yv.to(device)
                out = model(Xv)
                if out.shape[-1] != num_classes:
                    out = model.classifier(out)
                preds = out.argmax(dim=1)
                correct += (preds == yv).sum().item()
                total += yv.size(0)
        val_acc = correct / (total + 1e-8)
        print(f"Epoch {epoch+1} Validation Acc: {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            save_path = args.out_path
            torch.save({
                "model_config": config,
                "model_state_dict": model.state_dict(),
                "label_map": label_map
            }, save_path)
            print("Saved best model to", save_path)

    print("Training finished. Best val acc:", best_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptbxl-path", type=str, default="data/ptb-diagnostic-ecg-database-1.0.0",
                        help="Path to PTB-XL dataset folder (contains CSV and signal files)")
    parser.add_argument("--resaved-ckpt", type=str, default="hubert_ecg/hubert_ecg_small_resaved.pt",
                        help="Path to resaved HuBERT-ECG checkpoint")
    parser.add_argument("--out-path", type=str, default="hubert_ecg/hubert_ecg_finetuned.pt",
                        help="Where to save fine-tuned checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    main(args)

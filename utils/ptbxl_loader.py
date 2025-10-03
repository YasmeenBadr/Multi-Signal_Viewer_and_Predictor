# utils/ptbxl_loader.py
import os
import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def load_ptbxl_metadata(base_path):
    # auto-find a CSV metadata file
    possible = ["ptbxl_database.csv", "ptbxl.csv", "ptbxl_database_raw.csv"]
    for fn in possible:
        p = os.path.join(base_path, fn)
        if os.path.exists(p):
            return pd.read_csv(p)
    # fallback: try any csv in base_path
    for fn in os.listdir(base_path):
        if fn.endswith(".csv"):
            return pd.read_csv(os.path.join(base_path, fn))
    raise FileNotFoundError("PTB-XL metadata CSV not found in " + base_path)

def extract_label_from_row(row):
    # Try common columns (diagnostic_superclass -> e.g. 'NORM' or 'MI', or 'diagnostic' or 'scp_codes')
    if "diagnostic_superclass" in row and pd.notna(row["diagnostic_superclass"]):
        return str(row["diagnostic_superclass"])
    if "diagnostic" in row and pd.notna(row["diagnostic"]):
        return str(row["diagnostic"])
    # If scp_codes present (dictionary-like), extract first key
    if "scp_codes" in row and pd.notna(row["scp_codes"]):
        try:
            scp = eval(row["scp_codes"]) if isinstance(row["scp_codes"], str) else row["scp_codes"]
            if isinstance(scp, dict) and len(scp)>0:
                return list(scp.keys())[0]
        except Exception:
            pass
    # try 'label' column
    if "label" in row and pd.notna(row["label"]):
        return str(row["label"])
    return None

class PTBXLSignalDataset(Dataset):
    def __init__(self, base_path, select_leads=None, max_records=None, sampling_rate=100, transform=None):
        """
        base_path: folder containing PTB-XL csv and signal files (wfdb-format)
        select_leads: list of lead indices to use (None -> use all leads found)
        max_records: limit number of records (for quick experiments)
        sampling_rate: expected sampling rate (PTB-XL has 100 Hz variants)
        transform: optional function(signal: np.array) -> tensor/array
        """
        self.base_path = base_path
        self.df = load_ptbxl_metadata(base_path)
        if max_records:
            self.df = self.df.iloc[:max_records].copy()
        # gather records list: prefer filename_lr or filename
        fn_col = None
        for c in ["filename_lr", "filename", "fileName", "record_name"]:
            if c in self.df.columns:
                fn_col = c
                break
        if fn_col is None:
            raise RuntimeError("Could not find filename column in PTB-XL metadata.")
        self.df["label_processed"] = self.df.apply(lambda r: extract_label_from_row(r), axis=1)
        self.df = self.df.dropna(subset=["label_processed"]).reset_index(drop=True)
        self.fn_col = fn_col
        self.select_leads = select_leads
        self.sr = sampling_rate
        self.transform = transform

        # build label encoder
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df["label_processed"].values)
        # keep mapping
        self.label_map = {int(i): lab for i, lab in enumerate(self.label_encoder.classes_)}

    def __len__(self):
        return len(self.df)

    def _read_signal(self, record_path):
        # record_path: path/to/record (without extension)
        try:
            sig, meta = wfdb.rdsamp(record_path)
            # sig shape: (n_samples, n_leads)
            return sig, meta
        except Exception as e:
            # try file with .dat extension or direct path
            raise

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = os.path.join(self.base_path, row[self.fn_col])
        sig, meta = self._read_signal(fn)
        # select leads
        if self.select_leads is not None:
            sig = sig[:, self.select_leads]
        # convert to float32 and normalize per-record (z-score)
        sig = sig.astype(np.float32)
        sig = (sig - sig.mean(axis=0, keepdims=True)) / (sig.std(axis=0, keepdims=True) + 1e-8)
        # If single-lead, flatten to 1D; otherwise keep (n_samples, n_leads)
        if sig.shape[1] == 1:
            sig_tensor = torch.from_numpy(sig[:, 0]).float()
        else:
            # transpose to (n_leads, n_samples) if required by model wrapper
            sig_tensor = torch.from_numpy(sig.T).float()
        label = int(self.labels[idx])
        if self.transform:
            sig_tensor = self.transform(sig_tensor)
        return sig_tensor, label

    def get_label_map(self):
        return self.label_map

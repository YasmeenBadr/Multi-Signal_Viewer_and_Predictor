# resave_checkpoint.py
import torch
from hubert_ecg.hubert_ecg import HuBERTECG, HuBERTECGConfig

ORIG_PATH = "hubert_ecg/hubert_ecg_small.pt"
RESAVE_PATH = "hubert_ecg/hubert_ecg_small_resaved.pt"

# Load original checkpoint
ckpt = torch.load(ORIG_PATH, map_location="cpu", weights_only=False)

# Reconstruct config
if isinstance(ckpt["model_config"], dict):
    config = HuBERTECGConfig(**ckpt["model_config"])
else:
    config = ckpt["model_config"]

# Add any missing default attributes used by Hugging Face HubertModel
defaults = {
    "conv_pos_batch_norm": True,
    "feat_extract_norm": "group",
    "hidden_size": 768,
    "classifier_proj_size": 256,
}
for attr, val in defaults.items():
    if not hasattr(config, attr):
        setattr(config, attr, val)

# Build model
model = HuBERTECG(config)

# Load state dict safely
missing_keys, unexpected_keys = model.load_state_dict(
    ckpt["model_state_dict"], strict=False
)

print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

# Save the resaved checkpoint
torch.save({
    "model_config": config,
    "model_state_dict": model.state_dict()
}, RESAVE_PATH)

print(f"Checkpoint resaved to {RESAVE_PATH} successfully.")

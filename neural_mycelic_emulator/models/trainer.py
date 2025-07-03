from __future__ import annotations

import yaml
from pathlib import Path
from typing import List
import logging
import os

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from neural_mycelic_emulator.preprocessor.pipeline import tsv_to_glyph_sequences
from .lstm_emulator import LSTMEmulator
from neural_mycelic_emulator.log_helper import init_file_logger
from neural_mycelic_emulator.gpu_breathing import nano_pause, piko_pause


# trainer.py
USE_CUDNN = os.getenv("MYCELIC_USE_CUDNN", "0") == "1"
torch.backends.cudnn.enabled = USE_CUDNN


# ------------------------------------------------------------------
# Device selection helper
# ------------------------------------------------------------------
FORCE_CPU = os.getenv("MYCELIC_FORCE_CPU", "1") == "1"
if FORCE_CPU or not torch.cuda.is_available():
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")

class GlyphDataset(Dataset):
    def __init__(self, seq: List[int], window: int = 64):
        self.seq = seq
        self.window = window

    def __len__(self):
        return max(0, len(self.seq) - self.window)

    def __getitem__(self, idx):
        chunk = self.seq[idx : idx + self.window]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_config(tag: str, cfg_path: Path) -> dict:
    cfg_all = yaml.safe_load(cfg_path.read_text())
    base = cfg_all.get("shared_defaults", {})
    specific = cfg_all["models"][tag]
    merged = {**base, **specific}
    return merged


def train(tag: str, tsv_path: Path | None = None, cfg_path: Path = Path(__file__).parent / "emulator_parameters.yml"):
    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    log_file = init_file_logger(f"train_{tag}")
    logging.info("Starting Neural-Mycelic Emulator training - %s", tag)

    cfg = load_config(tag, cfg_path)
    logging.info("Config: %s", cfg)

    # Determine dataset path
    if tsv_path is None:
        if "dataset" not in cfg:
            raise ValueError("TSV path not provided and no 'dataset' key in YAML for this tag")
        tsv_path = Path(cfg["dataset"])
    else:
        tsv_path = Path(tsv_path)

    # If dataset path is a directory, pick the first *.txt file inside
    if tsv_path.is_dir():
        txt_files = list(tsv_path.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in dataset directory {tsv_path}")
        tsv_path = txt_files[0]
        logging.info("Using dataset file %s", tsv_path)

    seq, n_channels = tsv_to_glyph_sequences(tsv_path, return_channels=True)

    # Auto-adjust vocab size if dataset has more than 8 channels
    required_vocab = 8 + n_channels
    if required_vocab != cfg["vocab_size"]:
        logging.info("Adjusting vocab_size from %d to %d (channels=%d)", cfg["vocab_size"], required_vocab, n_channels)
        cfg["vocab_size"] = required_vocab

    window = cfg.get("window", 64)
    ds = GlyphDataset(seq, window=window)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    model = LSTMEmulator(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.1),
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(__file__).parent / tag
    ckpt_dir.mkdir(exist_ok=True)

    best = 1e9
    best_path: Path | None = None
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        # choose breathing function based on model size
        breath_fn = None
        if DEVICE.type == "cuda" and os.getenv("MYCELIC_GPU_BREATH", "1") == "1":
            if cfg["hidden_dim"] >= 256 or cfg["num_layers"] >= 3:
                breath_fn = nano_pause  # heavier models
            elif cfg["hidden_dim"] >= 128:
                breath_fn = piko_pause  # medium models
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if breath_fn:
                breath_fn(tag)
            total_loss += loss.item()
        avg = total_loss / len(dl)
        scheduler.step(avg)
        logging.info("Epoch %d/%d  loss=%.4f", epoch + 1, cfg["epochs"], avg)
        if avg < best - 1e-4:
            best = avg
            out = ckpt_dir / f"{tag}_best.pt"
            torch.save(model.state_dict(), out)
            best_path = out
            logging.info("saved checkpoint: %s", out)

    # ------------------------------------------------------------------
    # Validation perplexity (10%% split)
    # ------------------------------------------------------------------
    from .evaluate_perplexity import perplexity as _perplexity

    seq = tsv_to_glyph_sequences(tsv_path)
    split = int(len(seq) * 0.10)
    val_seq = seq[-split:]
    val_ds = GlyphDataset(val_seq, window=window)
    ppl = _perplexity(model, val_ds, batch=cfg["batch_size"])
    logging.info("Validation perplexity: %.3f", ppl)
    logging.info("Training finished – log stored at %s", log_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Train neural mycelic emulator")
    parser.add_argument("tag", help="model tag defined in emulator_parameters.yml")
    parser.add_argument("tsv", type=Path, nargs="?", help="optional path to TSV; if omitted uses dataset path from YAML")
    args = parser.parse_args()
    train(args.tag, args.tsv) 
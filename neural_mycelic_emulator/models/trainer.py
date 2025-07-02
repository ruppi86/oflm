from __future__ import annotations

import yaml
from pathlib import Path
from typing import List

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from neural_mycelic_emulator.preprocessor.pipeline import tsv_to_glyph_sequences
from .lstm_emulator import LSTMEmulator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    cfg = yaml.safe_load(cfg_path.read_text())
    return cfg["models"][tag]


def train(tag: str, tsv_path: Path, cfg_path: Path = Path(__file__).parent / "emulator_parameters.yml"):
    cfg = load_config(tag, cfg_path)

    seq = tsv_to_glyph_sequences(tsv_path)
    ds = GlyphDataset(seq)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    model = LSTMEmulator(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(__file__).parent / tag
    ckpt_dir.mkdir(exist_ok=True)

    best = 1e9
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(dl)
        scheduler.step(avg)
        print(f"Epoch {epoch+1}/{cfg['epochs']}  loss={avg:.4f}")
        if avg < best - 1e-4:
            best = avg
            out = ckpt_dir / f"{tag}_best.pt"
            torch.save(model.state_dict(), out)
            print(f"  ✔ saved {out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Train neural mycelic emulator")
    parser.add_argument("tag", help="model tag in emulator_parameters.yml e.g. cordyceps_small")
    parser.add_argument("tsv", type=Path, help="path to TSV recording")
    args = parser.parse_args()
    train(args.tag, args.tsv) 
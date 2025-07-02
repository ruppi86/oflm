from __future__ import annotations

from pathlib import Path
import argparse
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from neural_mycelic_emulator.preprocessor.pipeline import tsv_to_glyph_sequences
from .lstm_emulator import LSTMEmulator
from .trainer import GlyphDataset, load_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perplexity(model: LSTMEmulator, data: GlyphDataset, batch: int = 256):
    dl = DataLoader(data, batch_size=batch, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_tokens = 0
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            loss_sum += loss.item()
            total_tokens += y.numel()
    ppl = math.exp(loss_sum / total_tokens)
    return ppl


def main():
    p = argparse.ArgumentParser("Evaluate perplexity of emulator model")
    p.add_argument("model_tag")
    p.add_argument("model_file", type=Path)
    p.add_argument("tsv", type=Path)
    p.add_argument("--cfg", type=Path, default=Path(__file__).parent / "emulator_parameters.yml")
    args = p.parse_args()

    cfg = load_config(args.model_tag, args.cfg)

    seq = tsv_to_glyph_sequences(args.tsv)
    split = int(len(seq) * 0.10)
    val_seq = seq[-split:]
    val_ds = GlyphDataset(val_seq)

    model = LSTMEmulator(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.model_file, map_location=DEVICE))

    ppl = perplexity(model, val_ds, batch=cfg["batch_size"])
    print(f"Perplexity: {ppl:.3f}")


if __name__ == "__main__":
    main() 
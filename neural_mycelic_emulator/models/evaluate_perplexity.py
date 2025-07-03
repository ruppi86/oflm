from __future__ import annotations

from pathlib import Path
import argparse
import math
import logging
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from neural_mycelic_emulator.preprocessor.pipeline import tsv_to_glyph_sequences
from .lstm_emulator import LSTMEmulator
from .trainer import GlyphDataset, load_config
from neural_mycelic_emulator.log_helper import init_file_logger

# Disable faulty cuDNN kernels
torch.backends.cudnn.enabled = False  # type: ignore[attr-defined]

FORCE_CPU = os.getenv("MYCELIC_FORCE_CPU", "1") == "1"
DEVICE = torch.device("cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda")


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
    p.add_argument("tsv", type=Path, nargs="?")
    p.add_argument("--cfg", type=Path, default=Path(__file__).parent / "emulator_parameters.yml")
    args = p.parse_args()

    # ------------------------------------------------------------------
    log_file = init_file_logger(f"perplexity_{args.model_tag}")

    cfg = load_config(args.model_tag, args.cfg)

    if args.tsv is None:
        if "dataset" not in cfg:
            raise ValueError("TSV path not given and no dataset key in YAML")
        tsv_path = Path(cfg["dataset"])
    else:
        tsv_path = args.tsv

    # If dataset path is a directory, pick first *.txt file
    if tsv_path.is_dir():
        files = list(tsv_path.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found in {tsv_path}")
        tsv_path = files[0]
        logging.info("Using dataset file %s", tsv_path)

    seq, n_channels = tsv_to_glyph_sequences(tsv_path, return_channels=True)

    required_vocab = 8 + n_channels
    if required_vocab != cfg["vocab_size"]:
        logging.info("Adjusting vocab_size from %d to %d (channels=%d)", cfg["vocab_size"], required_vocab, n_channels)
        cfg["vocab_size"] = required_vocab

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
    msg = f"Perplexity: {ppl:.3f}"
    logging.info(msg)
    print(msg)
    logging.info("Log stored at %s", log_file)


if __name__ == "__main__":
    main() 
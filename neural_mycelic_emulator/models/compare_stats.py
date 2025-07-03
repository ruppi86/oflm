from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
from collections import Counter
from scipy.stats import ks_2samp
import math

import torch

from neural_mycelic_emulator.preprocessor.pipeline import tsv_to_glyph_sequences
from neural_mycelic_emulator.preprocessor.glyph_encoder import GLYPHS
from .lstm_emulator import LSTMEmulator
from .trainer import load_config

import logging
from neural_mycelic_emulator.log_helper import init_file_logger

# Avoid GPU to sidestep cuDNN errors
torch.backends.cudnn.enabled = False  # type: ignore[attr-defined]
DEVICE = torch.device("cpu")  # generation is cheap; avoid GPU assert


def silence_ratio(seq):
    return seq.count(GLYPHS["SIL"]) / len(seq)


def isi_distribution(spike_indices):
    return np.diff(spike_indices) if len(spike_indices) > 1 else np.array([0])


def extract_spike_idx(seq):
    idx = [i for i, g in enumerate(seq) if g != GLYPHS["SIL"]]
    return idx


def generate_sequence(model: LSTMEmulator, cfg: dict, length: int = 1000):
    start_token = 8  # channel-0 prefix (activity 0–7, prefixes 8–15)
    return model.generate(start_token=start_token, max_len=length, device=DEVICE)


def main():
    p = argparse.ArgumentParser("Compare real vs synthetic stats")
    p.add_argument("model_tag")
    p.add_argument("model_file", type=Path)
    p.add_argument("tsv", type=Path, nargs="?")
    args = p.parse_args()

    # Set up file logger early
    log_file = init_file_logger(f"analysis_{args.model_tag}")

    cfg = load_config(args.model_tag, Path(__file__).parent / "emulator_parameters.yml")

    if args.tsv is None:
        if "dataset" not in cfg:
            raise ValueError("TSV path not given and no dataset key in YAML")
        tsv_path = Path(cfg["dataset"])
    else:
        tsv_path = args.tsv

    if tsv_path.is_dir():
        txts = list(tsv_path.glob("*.txt"))
        if not txts:
            raise FileNotFoundError(f"No .txt files in {tsv_path}")
        tsv_path = txts[0]
        logging.info("Using dataset file %s", tsv_path)

    real_seq, n_channels = tsv_to_glyph_sequences(tsv_path, return_channels=True)

    required_vocab = 8 + n_channels
    if required_vocab != cfg["vocab_size"]:
        logging.info("Adjusting vocab_size from %d to %d to match channels=%d", cfg["vocab_size"], required_vocab, n_channels)
        cfg["vocab_size"] = required_vocab

    model = LSTMEmulator(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.model_file, map_location=DEVICE))

    synth_seq = generate_sequence(model, cfg, length=len(real_seq))

    # Silence
    real_sil = silence_ratio(real_seq)
    synth_sil = silence_ratio(synth_seq)
    logging.info("Silence ratio real  : %.2f", real_sil)
    logging.info("Silence ratio synth : %.2f  Δ=%.2f", synth_sil, abs(real_sil-synth_sil))
    print(f"Silence ratio real  : {real_sil:.2f}")
    print(f"Silence ratio synth : {synth_sil:.2f}  Δ={abs(real_sil-synth_sil):.2f}")

    # ISI
    real_isi = isi_distribution(extract_spike_idx(real_seq))
    synth_isi = isi_distribution(extract_spike_idx(synth_seq))
    stat, p = ks_2samp(real_isi, synth_isi)
    logging.info("ISI KS-stat=%.3f p=%.3f", stat, p)
    print(f"ISI KS-stat={stat:.3f}  p={p:.3f}")

    # Cohen's d for ISI distributions
    def cohen_d(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if len(a) < 2 or len(b) < 2:
            return float('nan')
        pooled_std = math.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
        if pooled_std == 0:
            return 0.0
        return (a.mean() - b.mean()) / pooled_std

    d_val = cohen_d(real_isi, synth_isi)
    logging.info("Cohen's d (ISI) : %.3f", d_val)
    print(f"Cohen d (ISI)     : {d_val:.3f}")

    # Glyph freq diff
    def top_freq(seq):
        c = Counter(seq)
        total = sum(c.values())
        return {k: v / total for k, v in c.items()}

    r_freq = top_freq(real_seq)
    s_freq = top_freq(synth_seq)
    keys = set(r_freq) | set(s_freq)
    l1 = sum(abs(r_freq.get(k, 0) - s_freq.get(k, 0)) for k in keys)
    logging.info("Glyph freq L1-diff  : %.3f", l1)
    print(f"Glyph freq L1-diff  : {l1:.3f}")

    logging.info("Analysis completed - log stored at %s", log_file)


if __name__ == "__main__":
    main() 
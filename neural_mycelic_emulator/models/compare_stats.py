from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
from collections import Counter
from scipy.stats import ks_2samp

import torch

from neural_mycelic_emulator.preprocessor.pipeline import tsv_to_glyph_sequences
from neural_mycelic_emulator.preprocessor.glyph_encoder import GLYPHS
from .lstm_emulator import LSTMEmulator
from .trainer import load_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def silence_ratio(seq):
    return seq.count(GLYPHS["SIL"]) / len(seq)


def isi_distribution(spike_indices):
    return np.diff(spike_indices) if len(spike_indices) > 1 else np.array([0])


def extract_spike_idx(seq):
    idx = [i for i, g in enumerate(seq) if g != GLYPHS["SIL"]]
    return idx


def generate_sequence(model: LSTMEmulator, cfg: dict, length: int = 1000):
    return model.generate(start_token=0x50, max_len=length, device=DEVICE)


def main():
    p = argparse.ArgumentParser("Compare real vs synthetic stats")
    p.add_argument("model_tag")
    p.add_argument("model_file", type=Path)
    p.add_argument("tsv", type=Path)
    args = p.parse_args()

    cfg = load_config(args.model_tag, Path(__file__).parent / "emulator_parameters.yml")

    real_seq = tsv_to_glyph_sequences(args.tsv)

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
    print(f"Silence ratio real  : {real_sil:.2f}")
    print(f"Silence ratio synth : {synth_sil:.2f}  Δ={abs(real_sil-synth_sil):.2f}")

    # ISI
    real_isi = isi_distribution(extract_spike_idx(real_seq))
    synth_isi = isi_distribution(extract_spike_idx(synth_seq))
    stat, p = ks_2samp(real_isi, synth_isi)
    print(f"ISI KS-stat={stat:.3f}  p={p:.3f}")

    # Glyph freq diff
    def top_freq(seq):
        c = Counter(seq)
        total = sum(c.values())
        return {k: v / total for k, v in c.items()}

    r_freq = top_freq(real_seq)
    s_freq = top_freq(synth_seq)
    keys = set(r_freq) | set(s_freq)
    l1 = sum(abs(r_freq.get(k, 0) - s_freq.get(k, 0)) for k in keys)
    print(f"Glyph freq L1-diff  : {l1:.3f}")


if __name__ == "__main__":
    main() 
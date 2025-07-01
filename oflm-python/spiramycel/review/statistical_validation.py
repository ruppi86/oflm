import sys
from pathlib import Path
import random
import math
import json
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]  # spiramycel/
PKG_ROOT = ROOT.parent  # oflm-python
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from spiramycel.token_constants import START_TOKEN, END_TOKEN, PAD_TOKEN
from spiramycel.training_utils import set_deterministic
from spiramycel.neural_trainer import SpiramycelNeuralModel, NetworkConditions
from spiramycel.analysis_stats import safe_welch, effect_size, EPS

# Use generators for realistic test data
try:
    from spiramycel.ecological_data_generator import EcologicalDataGenerator
    from spiramycel.generate_abstract_data import AbstractDataGenerator  # type: ignore
except Exception:
    EcologicalDataGenerator = None
    AbstractDataGenerator = None

import torch
import numpy as np

set_deterministic(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "ecological_calm": Path("ecological_models/ecological_calm_model.pt"),
    "ecological_chaotic": Path("ecological_models/ecological_chaotic_model.pt"),
    "abstract_calm": Path("abstract_models/abstract_calm_model.pt"),
    "abstract_chaotic": Path("abstract_models/abstract_chaotic_model.pt"),
}

SAMPLES_PER_CONDITION = 200

def load_model(path: Path) -> SpiramycelNeuralModel:
    model = SpiramycelNeuralModel(force_cpu_mode=(DEVICE.type == "cpu")).to(DEVICE)
    state_dict = torch.load(str(path), map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_dummy_conditions(calm: bool) -> NetworkConditions:
    if calm:
        # Near-optimal readings
        return NetworkConditions(latency=0.1, voltage=0.8, temperature=0.5, error_rate=0.02, bandwidth=0.9)
    else:
        # Stressed readings
        return NetworkConditions(latency=0.9, voltage=0.2, temperature=0.85, error_rate=0.6, bandwidth=0.1)


def jitter_conditions(base: NetworkConditions) -> NetworkConditions:
    """Return a slightly perturbed copy of base conditions."""
    def clip(v):
        return max(0.0, min(1.0, v))
    return NetworkConditions(
        latency=clip(base.latency + random.uniform(-0.05, 0.05)),
        voltage=clip(base.voltage + random.uniform(-0.05, 0.05)),
        temperature=clip(base.temperature + random.uniform(-0.05, 0.05)),
        error_rate=clip(base.error_rate + random.uniform(-0.05, 0.05)),
        bandwidth=clip(base.bandwidth + random.uniform(-0.05, 0.05)),
    )


def collect_silence_probs(model: SpiramycelNeuralModel, calm: bool) -> list[float]:
    probs = []
    base_cond = generate_dummy_conditions(calm)
    with torch.no_grad():
        for _ in range(SAMPLES_PER_CONDITION):
            cond = jitter_conditions(base_cond)
            cond_tensor = torch.tensor([cond.to_condition_vector()], dtype=torch.float32).to(DEVICE)
            input_tokens = torch.full((1, 1), START_TOKEN, dtype=torch.long).to(DEVICE)
            _, _, silence_logits, _, _, _ = model(input_tokens, cond_tensor)
            probs.append(torch.sigmoid(silence_logits[0, 0]).item())
    return probs


def run_test(a,b):
    res = safe_welch(a,b)
    if res is None:
        return None
    t, df, p = res
    return t, df, p


def main():
    results = {}
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"❌ Missing model: {path}")
            return
        model = load_model(path)
        calm = "calm" in name
        probs = collect_silence_probs(model, calm)
        results[name] = probs
        print(f"Loaded {name}: mean silence {np.mean(probs):.3f}")

    # Paradigm effect under calm
    res = run_test(results["ecological_calm"], results["abstract_calm"])
    if res:
        t, df, p = res
        means=(np.mean(results["ecological_calm"]), np.mean(results["abstract_calm"]))
        d = effect_size(results["ecological_calm"], results["abstract_calm"])
        print(f"\nCalm paradigm difference → t={t:.3f}, df={df:.1f}, p={p:.4f}, d={d:.3f}")
    else:
        print("\nCalm paradigm difference → insufficient variance for t-test")

    # Paradigm effect under chaos
    res = run_test(results["ecological_chaotic"], results["abstract_chaotic"])
    if res:
        t, df, p = res
        means=(np.mean(results["ecological_chaotic"]), np.mean(results["abstract_chaotic"]))
        d = effect_size(results["ecological_chaotic"], results["abstract_chaotic"])
        print(f"\nChaotic paradigm difference → t={t:.3f}, df={df:.1f}, p={p:.4f}, d={d:.3f}")
    else:
        print("\nChaotic paradigm difference → insufficient variance for t-test")

    # Stress effect inside each paradigm
    res = run_test(results["ecological_calm"], results["ecological_chaotic"])
    if res:
        t, df, p = res
        means=(np.mean(results["ecological_calm"]), np.mean(results["ecological_chaotic"]))
        d = effect_size(results["ecological_calm"], results["ecological_chaotic"])
        print(f"\nEcological stress effect → t={t:.3f}, df={df:.1f}, p={p:.4f}, d={d:.3f}")
    else:
        print("\nEcological stress effect → insufficient variance for t-test")

    res = run_test(results["abstract_calm"], results["abstract_chaotic"])
    if res:
        t, df, p = res
        means=(np.mean(results["abstract_calm"]), np.mean(results["abstract_chaotic"]))
        d = effect_size(results["abstract_calm"], results["abstract_chaotic"])
        print(f"Abstract stress effect → t={t:.3f}, df={df:.1f}, p={p:.4f}, d={d:.3f}")
    else:
        print("\nAbstract stress effect → insufficient variance for t-test")

if __name__ == "__main__":
    main() 
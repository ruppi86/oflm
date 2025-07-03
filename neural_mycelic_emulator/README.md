# Neural-Mycelic Emulator 🍄🧬

**Tiny language models that speak in fungal spike glyphs**

The Neural-Mycelic Emulator trains LSTM language models on real multi-channel voltage recordings from living mycelia.  Each glyph in the vocabulary represents either a bio-electrical *activity* (8 tokens) or a *channel prefix* (up to 8).  By generating glyph sequences we can simulate the statistical "language" of different fungal species and study how model capacity, silence ratio and inter-spike-interval (ISI) dynamics interact.

## 1 · Key ideas

• **Silence majority** – natural mycelium is silent ≈80 % of the time; models are evaluated on how well they match this baseline.
• **Species paradigms** – datasets for *Cordyceps*, *Enoki*, *Ghost fungus* and *Schizophyllum* differ in channel count and spike rhythm.
• **Multi-scale models** – *small* (~35 k params), *medium* (~140 k) and *large* (~550 k) share the same training loop; capacity is the only variable.
• **Deterministic training** – global seeds + CuDNN flags for reproducibility; checkpoints embed `scale` metadata.

## 2 · Directory quick tour

```
neural_mycelic_emulator/
├── dataset/                  # ⇦ Raw TSV voltage files
├── preprocessor/             # Spike → glyph pipeline
├── models/
│   ├── trainer.py            # Main training loop
│   ├── evaluate_perplexity.py
│   ├── compare_stats.py      # Real vs synth metrics
│   ├── lstm_emulator.py      # 2-layer GRU-like LSTM wrapper
│   └── emulator_parameters.yml  # Hyper-params per species & scale
├── tools/
│   └── dataset_stats.py      # Glyph counts per dataset
└── logs/                     # Auto-rotating training / analysis logs
```

## 3 · Installation

```bash
pip install -r requirements.txt  # pytorch, pandas, scipy, etc.
```

(optional) Set environment variables
```bash
# Use GPU if available
set MYCELIC_FORCE_CPU=0
# Enable deterministic but slower CuDNN kernels
set MYCELIC_USE_CUDNN=0
```

## 4 · Workflow

1. **Pre-flight** – run dataset stats
   ```bash
   python -m neural_mycelic_emulator.tools.dataset_stats
   ```
2. **Train** model tag (creates `*/<tag>_best.pt`)
   ```bash
   python -m neural_mycelic_emulator.models.trainer <tag>
   ```
3. **Validate** perplexity on hold-out 10 % split
   ```bash
   python -m neural_mycelic_emulator.models.evaluate_perplexity <tag> models/<tag>/<tag>_best.pt
   ```
4. **Analyse** real vs synthetic stats (silence, ISI, glyph L1)
   ```bash
   python -m neural_mycelic_emulator.models.compare_stats <tag> models/<tag>/<tag>_best.pt
   ```

All logs end up in `neural_mycelic_emulator/logs/` and are automatically timestamped.

## 5 · Metrics at a glance

| Metric | Meaning | Goal |
|--------|---------|------|
| Silence ratio | Fraction of glyphs = silence token | match real within ±0.01 |
| ISI KS p-value | Kolmogorov–Smirnov test on inter-spike intervals | p ≥ 0.05 (non-significant) |
| Cohen's d | Effect size on ISI distribution | |d| ≤ 0.2 |
| Glyph L1-diff | L1 distance on glyph frequency histogram | ↓ lower is better (<0.3 ideal) |

## 6 · Recent results
See `results/results_2025_07_02.md` for full cross-species tables and discussion.

---

## 7 · CLI cheat–sheet  
(Existing raw commands kept for copy-paste convenience)

# cordyceps

## Small
python -m neural_mycelic_emulator.models.trainer  cordyceps_small  

python -m neural_mycelic_emulator.models.compare_stats           cordyceps_small           neural_mycelic_emulator/models/cordyceps_small/cordyceps_small_best.pt          neural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt

python -m neural_mycelic_emulator.models.evaluate_perplexity           cordyceps_small           neural_mycelic_emulator/models/cordyceps_small/cordyceps_small_best.pt           neural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt

## Medium

python -m neural_mycelic_emulator.models.trainer  cordyceps_medium      

python -m neural_mycelic_emulator.models.compare_stats           cordyceps_medium           neural_mycelic_emulator/models/cordyceps_medium/cordyceps_medium_best.pt          neural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt

python -m neural_mycelic_emulator.models.evaluate_perplexity           cordyceps_medium           neural_mycelic_emulator/models/cordyceps_medium/cordyceps_medium_best.pt           neural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt

 ## Large

python -m neural_mycelic_emulator.models.trainer       cordyceps_large       

python -m neural_mycelic_emulator.models.compare_stats cordyceps_large neural_mycelic_emulator/models/cordyceps_large/cordyceps_large_best.pt neural_mycelic_emulator/dataset/Cordyceps_militari/Cordyceps_militari.txt


# Enoki

## Small
python -m neural_mycelic_emulator.models.trainer enoki_small

python -m neural_mycelic_emulator.models.compare_stats enoki_small neural_mycelic_emulator/models/enoki_small/enoki_small_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity enoki_small neural_mycelic_emulator/models/enoki_small/enoki_small_best.pt

## Medium

python -m neural_mycelic_emulator.models.trainer enoki_medium

python -m neural_mycelic_emulator.models.compare_stats enoki_medium neural_mycelic_emulator/models/enoki_medium/enoki_medium_best.pt


python -m neural_mycelic_emulator.models.evaluate_perplexity enoki_medium neural_mycelic_emulator/models/enoki_medium/enoki_medium_best.pt

## Large
python -m neural_mycelic_emulator.models.trainer enoki_large

python -m neural_mycelic_emulator.models.compare_stats enoki_large neural_mycelic_emulator/models/enoki_large/enoki_large_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity enoki_large neural_mycelic_emulator/models/enoki_large/enoki_large_best.pt


# Ghost Fungi (Omphalotus nidiformis)

## Medium
python -m neural_mycelic_emulator.models.trainer ghost_medium

python -m neural_mycelic_emulator.models.compare_stats ghost_medium neural_mycelic_emulator/models/ghost_medium/ghost_medium_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity ghost_medium neural_mycelic_emulator/models/ghost_medium/ghost_medium_best.pt

## Large
python -m neural_mycelic_emulator.models.trainer ghost_large

python -m neural_mycelic_emulator.models.compare_stats ghost_large neural_mycelic_emulator/models/ghost_large/ghost_large_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity ghost_large neural_mycelic_emulator/models/ghost_large/ghost_large_best.pt


# Schizophyllum commune

## Small
python -m neural_mycelic_emulator.models.trainer schizo_small

python -m neural_mycelic_emulator.models.compare_stats schizo_small neural_mycelic_emulator/models/schizo_small/schizo_small_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity schizo_small neural_mycelic_emulator/models/schizo_small/schizo_small_best.pt

## Medium
python -m neural_mycelic_emulator.models.trainer schizo_medium

python -m neural_mycelic_emulator.models.compare_stats schizo_medium neural_mycelic_emulator/models/schizo_medium/schizo_medium_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity schizo_medium neural_mycelic_emulator/models/schizo_medium/schizo_medium_best.pt

## Large
python -m neural_mycelic_emulator.models.trainer schizo_large

python -m neural_mycelic_emulator.models.compare_stats schizo_large neural_mycelic_emulator/models/schizo_large/schizo_large_best.pt

python -m neural_mycelic_emulator.models.evaluate_perplexity schizo_large neural_mycelic_emulator/models/schizo_large/schizo_large_best.pt




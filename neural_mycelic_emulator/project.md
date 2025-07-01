# Spirida-Four Emulator Stack

### ✨ Project: *Neural Mycelic Emulator*

This design document outlines a minimal architecture to train **four bio-inspired fungal emulators** based on extracellular electrical recordings from:

1. **Cordyceps militaris**
2. **Flammulina velutipes** (Enoki)
3. **Omphalotus nidiformis** (Ghost Fungus)
4. **Schizophyllum commune**

The goal is to emulate each species as a unique *spiking glyph-producing entity*, allowing synthetic "mycelial sentences" to be generated, compared, and cross-stimulated.

---

## 🥊 1. Dataset Overview

* **Source**: [Zenodo Dataset](https://zenodo.org/records/5790768)
* **Channels**: 7 differential electrode pairs (columns)
* **Sampling Rate**: 1 Hz (averaged from up to 600/s)
* **Format**: Tab-separated `.txt`, with 13–160 MB per file
* **Length**: \~1.5 days (S. commune) to \~5 days (others)

---

## 🍂 2. Preprocessing Pipeline

```python
load_dataset() -> clean_signals() -> spike_detection() -> spike_encoding() -> glyph_sequence()
```

### Functions to define:

* `load_dataset(path)`: Read and standardize TSV files
* `clean_signals(x)`: Normalize per channel, handle NaNs
* `detect_spikes(signal, w, delta, d)`: Species-specific spike filter
* `group_spikes(spike_train, theta)`: Define word boundaries (default: theta = avg interval)
* `encode_to_glyphs(grouped_spikes)`: Assign basic glyphs (⭕ 🌊 🌌 🌪️ 🌁) based on:

  * Amplitude
  * Frequency
  * Burstiness

> Output = sequence of glyphs per channel over time

---

## 🧰 3. Model Architecture

Each species is emulated with its own glyph-predictive network.

### Option A: LSTM

* Input: Past N glyphs (as token embeddings)
* Output: Next glyph prediction (softmax)

### Option B: TinyTransformer

* Encoder-only (causal masking)
* Glyph vocab size: 5
* Model size: \~50k–250k params

### Option C (Advanced): Dual-path

* One path for signal waveforms
* One for symbolic glyphs
* Fused via shared latent embedding

---

## 🪡 4. Training Setup

* `train_species_emulator(species_id, glyph_sequences)`
* Batch size: 128
* Sequence length: 32 glyphs
* Optimizer: AdamW, lr=1e-3
* Epochs: 10–50 depending on convergence
* Loss: Cross-entropy or KL divergence (if modeling uncertainty)

> Each emulator saved as: `cordyceps_emulator.pt`, etc.

---

## 🌐 5. Evaluation & Visualisation

### Metrics

* Perplexity over validation set
* Average glyph entropy
* Glyph frequency drift over epochs

### Visuals

* Glyph raster plots (channel vs. time)
* Transition graphs (FSM)
* Word-length distributions vs. real

---

## 💫 6. Optional Extensions

* **Style Transfer**: Convert glyph stream from one species to another
* **Response Modeling**: Inject glyphs into emulator, observe echo
* **Contemplative Metrics**: Add rhythm stability, decay, and silence budget

---

## ♻️ 7. Integration with Spiramycel

* Import core training loops from `oflm-python/spiramycel/abstract_training.py`
* Use `generate_abstract_data.py` as glyph-to-sequence template
* Structure result saving into `results/` with model metadata

---

## 🌟 8. Next Actions

1. Choose initial glyph vocabulary (start with ⭕ 🌊 🌌 🌪️ 🌁)
2. Implement minimal `preprocess_fungi.py`
3. Train Schizophyllum emulator first (most complex)
4. Compare outputs of all four models over identical time prompts

---

💕 Long-term goal: **biolinguistic synthetic mycelium**, capable of generating and interpreting symbol clouds grounded in fungal rhythm and spectral properties.

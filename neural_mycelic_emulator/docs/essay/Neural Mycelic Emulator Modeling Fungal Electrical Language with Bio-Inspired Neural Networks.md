# Neural Mycelic Emulator: Modeling Fungal Electrical Language with Bio-Inspired Neural Networks

**Robin Langell, in collaboration with language models ChatGPT 4o, ChatGPT o3, ChatGPT4.1, and Claude 4 Sonnet**

---

## Abstract

The neural mycelic emulator is a novel framework for modeling the electrical language of fungi using bio-inspired neural networks. By training LSTM-based language models on real multi-channel voltage recordings from living mycelia, the emulator simulates the statistical and temporal properties of fungal electrical activity. This work presents a comprehensive evaluation of emulator performance across multiple fungal species, model scales, and context lengths, with a focus on glyph distribution, silence ratio, and inter-spike-interval (ISI) dynamics. Results demonstrate species-specific effects of model scaling and context extension, and highlight the emulator's potential as a tool for both mycological research and bio-digital interface design.

**Keywords:** fungal computing, mycelium, neural emulator, LSTM, bio-digital interface, electrical spiking, memristor, bioinformatics

---

## 1. Introduction

Fungi exhibit complex electrical activity, with action potential-like spikes propagating through mycelial networks [1,2]. Recent research has revealed that these electrical signals encode information about growth, nutrient transport, and environmental response, suggesting a form of non-neuronal communication analogous to neural signaling in animals [3,4]. The field of fungal computing has leveraged these properties to develop living logic gates, memristive devices, and bio-digital interfaces [5–8].

At the heart of this research lies a greater vision: to create a symbiotic, mycelic, human-friendly connection that can help our species survive and flourish in the long term. As articulated in the "Beyond a Piktun" essay and the Mychainos conceptual system architecture [12], the ultimate goal is to develop resilient, distributed, and ecologically grounded computational systems that partner with mycelium for memory, culture, and adaptive intelligence. The neural mycelic emulator represents a foundational step toward this broader paradigm, enabling the study and prototyping of such symbiotic technologies.

The EU FUNGAR project and the work of Adamatzky and collaborators have been instrumental in advancing our understanding of fungal bioelectronics, demonstrating the feasibility of integrating mycelium into computational and architectural substrates [9,10]. However, modeling the statistical language of fungal electrical activity remains a challenge, due to the high dimensionality, sparsity, and temporal complexity of the data.

This paper introduces the neural mycelic emulator, a system that applies language modeling techniques to fungal spike data. By representing electrical events as glyphs and training neural networks to generate realistic sequences, the emulator provides a platform for simulating, analyzing, and interfacing with fungal bioelectricity. We present a cross-species evaluation of the emulator, discuss the impact of model parameters, and situate our findings within the broader context of bio-inspired computing.

---

## 2. Background and Related Work

Electrical signaling in fungi has been documented in multiple species, with both spontaneous and stimulus-evoked spikes observed in Pleurotus, Neurospora, and Armillaria [1,2]. These spikes are thought to mediate long-distance communication, coordinate physiological processes, and respond to environmental changes [3,4].

The FUNGAR project (EU-H2020 FET grant 858132) established protocols for recording and analyzing fungal electrical activity, and developed experimental platforms for studying sensorial fusion and decision-making in mycelium [9]. Adamatzky et al. have demonstrated the use of mycelium as a substrate for logic gates, memristors, and unconventional computing devices [5,6,10]. Recent work has also explored the mem-fractive properties of mushrooms, showing that fruit bodies can exhibit memristive, mem-capacitive, and mem-inductive behaviors [7].

Language modeling approaches have been applied to biological sequences in genomics and neuroscience, but their application to fungal electrical data is novel. The neural mycelic emulator builds on these advances, using LSTM-based models to capture the statistical and temporal structure of spike glyph sequences.

---

## 3. Methods

### 3.1 System Overview

The neural mycelic emulator trains LSTM language models on real multi-channel voltage recordings from living mycelia[13]. Each glyph in the vocabulary represents either a bio-electrical activity (8 tokens) or a channel prefix (up to 8). By generating glyph sequences, the emulator simulates the statistical "language" of different fungal species and enables the study of model capacity, silence ratio, and inter-spike-interval (ISI) dynamics.

### 3.2 Datasets

Datasets were collected from multiple fungal species, including *Cordyceps militaris*, *Flammulina velutipes* (Enoki), *Omphalotus nidiformis* (Ghost Fungus), and *Schizophyllum commune*. Each dataset consists of multi-channel voltage recordings, preprocessed into glyph sequences using a custom spike-to-glyph pipeline. Data sources and preprocessing protocols follow those established in the FUNGAR project [9].

### 3.3 Model Architecture and Parameters

The emulator uses LSTM-based models with varying capacity (small, medium, large, xlarge) and context window sizes (64, 128, 192). Hyperparameters for each model variant are specified in `emulator_parameters.yml`. Key parameters include:

- **Vocab size:** 16 (8 activity glyphs + 8 channel prefixes)
- **Embed dim:** 16–32
- **Hidden dim:** 64–384
- **Num layers:** 1–4
- **Batch size:** 28–256
- **Window:** 64–192
- **Learning rate:** 0.0007–0.001
- **Dropout:** 0.2–0.3 (for larger models)
- **Epochs:** 12–45
- **Auxiliary losses:** ISI-matching loss, label smoothing (for some variants)

All models are trained using deterministic settings (global seeds, CuDNN flags) for reproducibility. Training and evaluation scripts are available in the project repository.

### 3.4 Evaluation Metrics

Models are evaluated on:
- **Silence ratio:** Fraction of glyphs that are the silence token (goal: match real data within ±0.01)
- **ISI KS p-value:** Kolmogorov–Smirnov test on inter-spike intervals (goal: p ≥ 0.05)
- **Cohen's d:** Effect size on ISI distribution (goal: |d| ≤ 0.2)
- **Glyph L1-diff:** L1 distance on glyph frequency histogram (goal: <0.3 ideal)

## 4. Results

### 4.1 Cordyceps militaris

| Model tag | Params (≈) | Silence ratio | ISI KS p-value | Cohen's d (ISI) | Glyph L1-diff |
|-----------|-----------:|--------------|---------------|----------------:|---------------|
| cordyceps_small  |  ~35 k | 0.04 | 0.003 | 0.083 | 0.224 |
| cordyceps_medium | ~140 k | 0.04 | 0.193 | 0.047 | 0.841 |
| cordyceps_medium_ctx128 | ~140 k | 0.04 | 0.000 | -0.116 | 0.425 |
| cordyceps_large  | ~550 k | 0.04 | 0.000 | 0.120 | **0.266** |
| cordyceps_large_ctx128 | ~550 k | 0.11 | 0.060 | 0.058 | 0.591 |
| cordyceps_xlarge | ~3.5 M | 0.04 | 0.000 | 0.140 | 0.270 |
| cordyceps_xlarge_ctx128 | ~3.5 M | 0.04 | 1.000 | -0.001 | 0.900 |
| cordyceps_xlarge_ctx128_v2 | ~3.5 M | 0.04 | 0.979 | 0.020 | **0.037** |
| cordyceps_xlarge_ctx192 | ~3.5 M | 0.04 | 0.000 | 0.105 | 0.312 |

**Key observations:**
- Longer context (128) in large models increases glyph error and silence ratio.
- Large (ctx64) model offers best trade-off; x-large adds parameters with little gain.
- The retrained **xlarge_ctx128_v2** reduces glyph L1 to 0.037 (≈7× better) while keeping perfect rhythm (p ≈ 0.98).
- Medium model benefits from context extension.
- ctx192 improves glyph error over the original 128 version (0.312 vs 0.900) but re-introduces timing mismatch (KS-p = 0.000).  Longer context alone is not enough—needs more regularisation.


### 4.2 Flammulina velutipes (Enoki)

| Model tag | Params (≈) | Silence ratio | ISI KS p-value | Cohen's d (ISI) | Glyph L1-diff |
|-----------|-----------:|--------------|---------------|----------------:|---------------|
| enoki_small  |  ~35 k | 0.11 | 0.000 | 0.150 | 0.374 |
| enoki_medium | ~140 k | 0.11 | 0.000 | 0.441 | 0.967 |
| enoki_medium_ctx128 | ~140 k | 0.11 | 0.000 | 0.094 | 0.633 |
| enoki_large  | ~550 k | 0.11 | 0.000 | -0.324 | 0.537 |
| enoki_large_ctx128 | ~550 k | 0.11 | 0.011 | -0.067 | 0.372 |
| enoki_xlarge_ctx128 | ~3.5 M | 0.11 | 0.000 | 0.104 | 0.379 |
| enoki_xlarge_ctx192 | ~3.5 M | 0.11 | 1.000 | 0.004 | **0.033** |

**Key observations:**
- Context extension improves glyph distribution and effect size.
- Temporal mismatch persists; silence ratio remains high (likely dataset-driven).
- Extending to a 192-token window drives glyph L1 down to 0.033, while also reaching optimal temporal alignment (KS-p = 1.0) – indicating thath the model benefits strongly from very long context.

### 4.3 Omphalotus nidiformis (Ghost Fungus)

| Model tag | Params (≈) | Silence ratio | ISI KS p-value | Cohen's d (ISI) | Glyph L1-diff |
|-----------|-----------:|--------------|---------------|----------------:|---------------|
| ghost_small  |  ~35 k | — | — | — | — |
| ghost_medium | ~140 k | 0.04 | 1.000 | 0.008 | **0.124** |
| ghost_medium_ctx128 | ~140 k | 0.04 | 0.202 | 0.028 | 0.307 |
| ghost_large  | ~550 k | 0.04 | 0.000 | 0.084 | 0.362 |
| ghost_large_ctx128 | ~550 k | 0.04 | 0.000 | 0.061 | 0.356 |
| ghost_xlarge_ctx128 | ~3.5 M | 0.04 | 0.788 | 0.017 | 0.335 |

**Key observations:**
- Context extension yields marginal gains.
- Medium model achieves lowest glyph error and perfect rhythm.
- Higher compute cost - the largeer ctx128 configurations carries ~4× the parameters of medium for minor quality gains.

### 4.4 Schizophyllum commune

| Model tag | Params (≈) | Silence ratio | ISI KS p-value | Cohen's d (ISI) | Glyph L1-diff |
|-----------|-----------:|--------------|---------------|----------------:|---------------|
| schizo_small  |  ~35 k | 0.03 | 1.000 | 0.012 | 0.325 |
| schizo_medium | ~140 k | 0.03 | 0.905 | 0.061 | 0.528 |
| schizo_large  | ~550 k | 0.03 | 1.000 | 0.006 | 0.212 |
| schizo_large_ctx128 | ~550 k | 0.03 | 1.000 | 0.002 | 0.149 |
| schizo_xlarge_ctx128 | ~3.5 M | 0.03 | 1.000 | -0.005 | **0.080** |

**Key observations:**
- Context extension significantly improves glyph error without affecting silence or rhythm.
- xlarge_ctx128 is the new top performer for Schizophyllum.

## 5. Discussion

The neural mycelic emulator demonstrates that language modeling techniques can effectively capture the statistical and temporal properties of fungal electrical activity. Results reveal species-specific effects of model scaling and context extension: longer context windows benefit Enoki and Schizophyllum, are neutral for Ghost, and detrimental for Cordyceps at large scale. Medium models suffice for Ghost and Cordyceps, while context-extended variants are necessary for high precision in Enoki and Schizophyllum.

The emulator's ability to match silence ratios and ISI distributions within biological ranges suggests its utility for simulating fungal communication and designing bio-digital interfaces. However, persistent temporal mismatches in some species (e.g., Enoki) indicate the need for improved loss functions and data balancing. Future work should explore context sweeps, ISI-matching losses, and unified training protocols across species.

A notable complementary development is the Spirida-Mycelic system [11], which extends the principles of neural mycelic emulation into the domain of bio-digital interfaces and contemplative AI. Spirida-Mycelic implements real-time translation between fungal spike patterns and contemplative glyphs, supports species-specific breathing and mood cycles, and provides a platform for interactive, trust-based progression with living computational substrates. This integration of bioelectrical modeling with contemplative and ethical frameworks exemplifies the potential for interdisciplinary research at the intersection of mycology, AI, and digital humanities.

## 6. Conclusion

This work presents the neural mycelic emulator as a flexible, bio-inspired platform for modeling fungal electrical language. By leveraging LSTM-based language models and real spike data, the emulator advances our ability to simulate, analyze, and interface with living mycelium. The results highlight both the promise and the challenges of bio-digital emulation, and lay the groundwork for future research in fungal computing, bioinformatics, and adaptive architectures.

Looking forward, the greater aspiration of this project is to realize the vision set forth in "Beyond a Piktun" and the Mychainos system [12]: to foster a resilient, symbiotic partnership between humans and mycelium, creating computational systems that are not only technically advanced but also ecologically attuned, distributed, and capable of supporting long-term human and planetary flourishing. The neural mycelic emulator is a step on this path, with the hope that future developments will bring us closer to a truly symbiotic, mycelic intelligence.


## Acknowledgements

- The author thanks the entire FUNGAR research team for foundational contributions to fungal bioelectronics and unconventional computing.



## Licensing, Ethics & Stewardship

> "What we seed in openness, we harvest in resilience."

This project and its conceptual ecosystem (including Spirida™, Spiralbase™, and Mychainos™) are governed by a multi-layered commitment to openness, ethics, and long-term reciprocity:

**1. Conceptual Layer – Theory, Writings, Patterns**  
License: Creative Commons Attribution–ShareAlike 4.0 (CC BY-SA 4.0)  
Scope: Philosophical foundations, symbolic grammars, pattern libraries, educational diagrams, essays, and guides  
Intent: Allow free reuse, remix, and re-publication; preserve openness through share-alike conditions; ensure attribution to source thinkers and communities

**2. Software Layer – Tools, Compilers, Simulations**  
License: GNU General Public License v3 (GPLv3)  
Scope: Interpreters, compilers, emulators, simulation sandboxes, custom logic engines, and spiral pattern parsers  
Intent: Guarantee access to all source code; require open licensing for forks or adaptations; allow commercial use under cooperative terms

**3. Hardware Layer – Sensors, Interfaces, Devices**  
License: CERN Open Hardware License v2  
Scope: Sensor schematics, circuit blueprints, resonance devices, rhythm-aware chips, modular hardware for bio-digital interaction  
Intent: Mandate full design disclosure; enable community fabrication; prevent hardware enclosure or black-box design

**4. Biological Layer – Living Systems, DNA, Mycelium**  
License: Open Material Transfer Agreement (OpenMTA)  
Scope: Engineered fungal networks, root biointerfaces, DNA-based memory encoding structures, organisms adapted to spiral rhythm protocols  
Intent: Support open research and safe distribution; prevent bio-lockdown or privatization of life; require ethical collaboration and open science practice

**Unified Ethical Guardrails**
- May not be used for coercion, military, or surveillance purposes
- Must not be patented, black-boxed, or stripped of ecological grounding
- Must remain accessible, attributed, and shared with care

This licensing structure is designed to ensure openness, stewardship, and ethical development for all current and future technologies related to this work.

---

## References

1. Adamatzky, A. et al. (2020). Intrinsic spiking of electrical potential of mycelium. Deliverable D4.1, FUNGAR Project, EU-H2020 FET grant 858132.
2. Beasley, A.E., Abdelouahab, M.-S., Lozi, R.P., Powell, A.L., & Adamatzky, A. (2020). Mem-fractive Properties of Mushrooms. Preprint. https://www.researchgate.net/publication/339079868
3. Adamatzky, A. et al. (2023). Electrical response of fungi to changing moisture content. Fungal Biol. & Biotech., 10, 8.
4. Money, N. (2021). The fungal mind: on the evidence for mushroom intelligence. Psyche Magazine.
5. Roberts, N. & Adamatzky, A. (2022). Mining logical circuits in fungi. Scientific Reports, 12, 15930.
6. Adamatzky, A. (2021). Adaptive Fungal Architectures. LINKs-series 5-6, pp. 66–77.
7. Beasley, A.E. et al. (2020). Mem-fractive Properties of Mushrooms. Preprint.
8. CORDIS EU Report (2023). Fungi could be the foundation of intelligent and sustainable buildings – FUNGAR results.
9. FUNGAR Project Consortium. (2019–2023). https://fungar.eu/
10. Adamatzky, A. et al. (2023). Fungal Computing and Bio‑Digital Interfaces: A 2023–2024 Research Survey.
11. Langell, R. et al. (2025). Spirida-Mycelic: Bio-Digital Bridges Between Fungal Logic and Contemplative AI. Zenodo. https://zenodo.org/records/15792166
12. Langell, R. (2025). Beyond a Piktun: Remembering in the Age of Restart through a Resilient Mycelic Operating System. Zenodo. https://zenodo.org/records/15792167
13. Adamatzky A. (2021). Recordings of electrical activity of four species of fungi [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5790768

---

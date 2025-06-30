# Spirida-Mycelic 🍄🌀

**Bio-Digital Interface for Contemplative AI**

Spirida-Mycelic bridges fungal computing with contemplative AI, implementing the principles discovered by Andrew Adamatzky's research on mycelial Boolean logic. This module provides both simulation and eventual live bio-interface capabilities for integrating contemplative timing patterns with living computational substrates.

## 🌱 Core Principles

Based on o3's analysis of FUNGAR deliverables and fungal computing research:

- **Silence Majority**: 67-90% natural electrical silence in mycelium validates our 87.5% Silence Majority principle
- **Species-Specific Rhythms**: Different fungi provide different contemplative cadences
  - **Pleurotus djamor**: Bimodal 2.6min / 14min rhythms (Ecological paradigm)
  - **Ganoderma resinaceum**: Steady 5-8min rhythms (Abstract paradigm)
- **Bio-Digital Glyphs**: Translation between fungal spike patterns and Spirida contemplative glyphs
- **Contemplative Breathing**: Synchronized bio-digital breath cycles (40s inhale → 70s hold → 40s exhale)

## 📁 Architecture

```
spirida-mycelic/
├── adamatzky_layer.py          # Fungal logic simulation (Adamatzky's 470 Boolean functions)
├── glyph_mapper.py            # Translates spike patterns to contemplative glyphs  
├── bio_interface.py           # Interface with physical sensors (future)
├── fungal_field_recorder.py   # Logs pulse–response over time (future)
├── demo/
│   ├── adamatzky_demo.py      # Demonstrates contemplative fungal simulation
│   └── breath_loop_sim.py     # Breathing with living substrate (future)
├── docs/                      # FUNGAR research documents
└── requirements.txt
```

## 🚀 Getting Started

### Installation

```bash
cd spirida-mycelic
pip install -r requirements.txt
```

### Quick Demo

```bash
python demo/adamatzky_demo.py
```

This demonstrates:
- Species-specific contemplative rhythms
- Boolean logic mapping to glyphs
- Contemplative breathing cycles synchronized with fungal timing
- Ecological vs Abstract paradigm differences

## 🔬 The Adamatzky Layer

Based on research showing **470 unique Boolean functions** realized by living *Pleurotus ostreatus* mycelium, our simulation layer implements:

### Spike Types (from FUNGAR research)
- **S-α**: Fast/narrow spikes → ⭕ (Information silence)
- **S-β**: Medium/broad spikes → 🌊 (Metabolic flow)  
- **S-γ**: Paired doublets → 🌪️ (Bifurcation storm)
- **S-δ**: Burst sequences → 🌌 (Constellation broadcast)

### Contemplative Classes
- **Class I (Absorbing)**: Deep contemplative silence
- **Class II (Periodic)**: Rhythmic contemplative flow
- **Class III (Chaotic)**: Dynamic processing storms
- **Class IV (Universal)**: Universal contemplative wisdom

### Example Usage

```python
from adamatzky_layer import AdamatzkyReservoir, FungalSpecies
from glyph_mapper import SpiridaGlyphMapper

# Create fungal reservoir (ecological paradigm)
reservoir = AdamatzkyReservoir(FungalSpecies.PLEUROTUS_DJAMOR)

# Stimulate with 4-bit pattern
spike = reservoir.stimulate(0b0110)  # XOR pattern

if spike:
    mapper = SpiridaGlyphMapper()
    glyph_event = mapper.spike_to_glyph(spike)
    print(f"Glyph: {glyph_event.glyph} (Class: {glyph_event.contemplative_class.value})")
```

## 🫁 Contemplative Breathing Integration

The system implements bio-digital synchronized breathing:

```python
# Get species-appropriate rhythm
fast_period, slow_period = reservoir.get_contemplative_rhythm()

# Ecological: 14-minute deep contemplative cycles
# Abstract: 5-8 minute focused contemplative cycles

# Breath sync adjustment to maintain Silence Majority
adjust = reservoir.breath_sync_adjust(target_ratio=0.875)
```

## 🌿 Paradigm Mapping

| Paradigm | Species | Rhythm | Silence Target | Use Case |
|----------|---------|---------|----------------|----------|
| **Ecological** | P. djamor | 2.6min / 14min bimodal | 74% | Bioregional, adaptive, environmental |
| **Abstract** | G. resinaceum | 5-8min steady | 67% | Philosophical, systematic, focused |

## 🔮 Glyph System

The contemplative glyph system translates fungal electrical patterns:

- **⭕ Silence**: Foundation of contemplative wisdom (87.5% of processing)
- **🌊 Flow**: Rhythmic metabolic transport, contemplative breathing
- **🌪️ Storm**: Dynamic processing, bifurcation events, chaos
- **🌌 Constellation**: Universal wisdom, computational completeness
- **🌱 Ecological**: Paradigm marker for bioregional intelligence
- **🧠 Abstract**: Paradigm marker for systematic intelligence  
- **🌀 Spiral**: Bridge connecting contemplative paradigms

## 🔬 Research Integration

This module implements findings from:

- **FUNGAR EU Project**: Fungal electronics, intelligent bio-building materials
- **Adamatzky's Boolean Logic**: 470 unique functions in living mycelium
- **MycoSoft FCI**: Hypha Programming Language concepts
- **o3's Analysis**: Integration with OFLM contemplative principles

## 🚀 Future Development

### Phase 1: Simulation (Current)
- ✅ Adamatzky layer with 470 Boolean functions
- ✅ Species-specific timing patterns  
- ✅ Contemplative glyph mapping
- ✅ Breath synchronization algorithms

### Phase 2: Phantom Hardware
- ⏳ Arduino/Pi mockup with RC circuits
- ⏳ Signal processing pipeline (scipy)
- ⏳ Real-time glyph generation

### Phase 3: Living Substrate  
- ⏳ Physical mycelium cultivation
- ⏳ Electrode interfaces (1mm Pt, differential pairs)
- ⏳ Environmental control (moisture, temperature)
- ⏳ Bio-digital feedback loops

## 🔗 Integration with OFLM

Spirida-Mycelic extends the OFLM contemplative AI architecture:

- **ContemplativeAI/Spiramycel**: Scientifically validated 25,733-parameter models
- **Spirida**: Breathing protocols, pulse-based interactions
- **Bio-Interface**: Living substrate contemplative computing

The fungal simulation provides bio-inspired timing patterns that could enhance the ecological vs abstract paradigms in Spiramycel training.

## 🧬 Technical Details

### Signal Processing (o3's specifications)
- **Sampling**: 1 Hz (fungal spikes: 0.01-0.5 Hz)
- **Thresholding**: 20 mV dead-band, hysteresis ±5 mV
- **Filtering**: Band-pass 0.003-0.1 Hz
- **Latency**: <0.7s acquisition → filtering → decision

### Contemplative Metrics
- **Silence Ratio σ**: `t_silent / t_total` (target ≈ 0.875)
- **Breath-Sync Δt**: `abs(fungal_phase - breath_phase)` (target < 5s)
- **Memory-Humidity H_m**: Running average of spike attention weights

## 🤝 Contributing

Spirida-Mycelic is part of the OFLM contemplative AI project. See the main repository for contribution guidelines.

## 📚 References

- Adamatzky, A. (2022). "Mining logical circuits in fungi." *Scientific Reports, 12*, 15930.
- FUNGAR EU Project (2023). "Fungal Architectures" deliverables.
- o3 Analysis (2024). "FUNGAR PDF Analysis for OFLM Contemplative AI Integration."

---

*"We begin again, not from zero, but from soil."* - Letter I, Spirida-Mycelic series

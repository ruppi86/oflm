# HaikuMeadowLib 🌸

*A contemplative language model for generating haikus with breath-synchronized intelligence*

HaikuMeadowLib is a minimal yet profound neural language model designed for haiku generation, embodying contemplative principles of graceful silence, seasonal awareness, and breath-synchronized creation. Built as part of the broader ContemplativeAI ecosystem, it serves as both a standalone haiku generator and an integrated component for contemplative breathing systems.

## ✨ **Philosophy & Design**

Following o3's architectural vision, HaikuMeadowLib implements:

- **🦠 Femto-scale Intelligence**: ~35k-600k parameters (fits on a wildflower's petal)
- **🌊 Breath-synchronized Generation**: Aligned with contemplative breathing cycles
- **🍂 Seasonal Voice Drift**: Atmospheric conditions influence generation style
- **🤫 Graceful Silence**: Model knows when not to generate
- **♻️ Decay-aware Memory**: Natural forgetting and composting cycles
- **🌱 Minimal Footprint**: CPU-friendly with GPU scaling options

*Somatic signature: minimal / seasonal / ephemeral*

## 🎯 **Key Features**

### **Configurable Model Sizes**
- **Piko Model**: ~35k parameters, CPU-optimized for contemplative training
- **Nano Model**: ~600k parameters, GPU-accelerated for full-scale learning

### **Atmospheric Intelligence**
- Seasonal awareness (spring, summer, autumn, winter)
- Time-of-day sensitivity (dawn, day, dusk, night)
- Humidity and temperature conditioning
- Breath-phase synchronization

### **Contemplative Restraint**
- Configurable silence probability
- Natural rate limiting
- Atmospheric pressure awareness
- Graceful degradation modes

### **Memory Systems**
- Fragment-based associative recall
- Seasonal memory cycling
- Natural decay and composting
- Essence extraction from aging memories

## 📦 **Installation**

### Prerequisites
```bash
# Minimum requirements
pip install torch>=2.0.0 numpy>=1.21.0 pyyaml>=6.0

# Full development setup
pip install -r requirements.txt
```

### Required Dependencies
- **PyTorch**: Neural network framework (CPU/GPU adaptive)
- **NumPy**: Numerical computing
- **PyYAML**: Configuration file support
- **SciPy**: Statistical analysis (optional)
- **Matplotlib**: Visualization (optional)

## ⚙️ **Configuration System**

HaikuMeadowLib uses `lm_parameters.yml` for model configuration:

### **Piko Model (CPU-Optimized)**
```yaml
models:
  piko:
    parameter_count: 35  # ~35k parameters
    embed_dim: 32
    hidden_dim: 64
    num_layers: 1
    target_device: "cpu"
    training:
      epochs: 8
      batch_size: 2
      learning_rate: 0.001
```

### **Nano Model (GPU-Optimized)**
```yaml
models:
  nano:
    parameter_count: 600  # ~600k parameters
    embed_dim: 128
    hidden_dim: 256
    num_layers: 2
    target_device: "gpu"
    training:
      epochs: 20
      batch_size: 16
      learning_rate: 0.001
```

## 🚀 **Quick Start**

### **1. Generate Haikus (Interactive Mode)**
```bash
# Test with existing trained model
python generator.py --test --model piko

# Template-only mode (no neural model needed)
python generator.py --test --template-only
```

### **2. Train Your Own Model**
```bash
# Train piko model (CPU-safe)
python generator.py --train --model piko

# Train nano model (requires GPU)
python generator.py --train --model nano

# Custom parameters
python generator.py --train --model piko --epochs 10 --batch-size 4
```

### **3. Programmatic Usage**
```python
from generator import HaikuMeadow
from pathlib import Path

# Initialize with trained model
model_path = Path("model/piko/piko_haiku_model.pt")
meadow = HaikuMeadow(model_path)

# Generate haiku
haiku, generation_type = meadow.generate_haiku("morning mist")
if haiku:
    print(f"Generated ({generation_type}):")
    print(haiku)
else:
    print("[contemplative silence]")
```

## 🏗️ **Project Structure**

```
haikumeadowlib-python/
├── generator.py              # 🌸 Main haiku generator with neural models
├── lm_parameters.yml         # ⚙️ Model configuration (piko/nano)
├── ingest.py                 # 📥 Training data preparation with decay
├── memory.py                 # 🧠 Contemplative memory system
├── breath.py                 # 🫁 Breathing rhythm integration
├── dew_ledger.py            # 💧 Session logging and metrics
├── murmurs.py               # 🌿 Atmospheric murmur generation
├── model/
│   ├── piko/                # 🦠 Trained piko models (~35k params)
│   │   ├── piko_haiku_model.pt
│   │   ├── piko_model_best.pt
│   │   └── piko_model_epoch_*.pt
│   └── nano/                # 🚀 Trained nano models (~600k params)
├── training/
│   └── haiku_training_material.json  # 📚 Processed training data
├── src/
│   ├── all_haiku.csv        # 🌊 Raw haiku datasets
│   ├── docmarianum_1_haikus.csv
│   └── Notgnoshi_haiku.csv
└── logs/                    # 📊 Generation session logs
```

## 🌊 **Training Data Pipeline**

### **1. Prepare Training Data**
```bash
# Process CSV files into training material
python ingest.py
```

The ingestion process:
- Supports multiple CSV formats (notgnoshi, documarianum, all_haiku)
- Applies contemplative decay (preservation rate ~75%)
- Analyzes seasonal and temporal affinities
- Measures contemplative quality
- Creates atmospheric-aware training sets

### **2. Training Material Structure**
```json
{
  "general": ["haiku texts for general training"],
  "seasonal": {
    "spring": ["spring-attuned haikus"],
    "summer": ["summer-attuned haikus"],
    // ...
  },
  "temporal": {
    "dawn": ["dawn-time haikus"],
    "day": ["daytime haikus"],
    // ...
  },
  "high_contemplative": ["deeply contemplative haikus"]
}
```

## 🌸 **ContemplativeAI Integration**

HaikuMeadowLib integrates seamlessly with the ContemplativeAI ecosystem through `haiku_bridge.py`:

### **Breath-Synchronized Generation**
```python
# In ContemplativeAI/haiku_bridge.py
from haikumeadowlib.generator import HaikuMeadow

class HaikuBridge:
    def __init__(self):
        # Direct integration with trained femto-poet
        model_path = "haikumeadowlib-python/model/piko/piko_haiku_model.pt"
        self.haiku_meadow = HaikuMeadow(Path(model_path))
    
    async def exhale_exchange(self, fragment, breath_phase, community_pressure):
        # Generate haiku during EXHALE phase only
        if breath_phase == Phase.EXHALE:
            haiku, gen_type = self.haiku_meadow.generate_haiku(
                seed_fragment=fragment,
                breath_phase="exhale"
            )
            return MeadowBreath(haiku, gen_type)
```

### **Wind-Listener Skepnad**
The bridge implements o3's Wind-Listener contemplative shape:
- **Fragment Worthiness**: Senses contemplative quality of seed fragments
- **Breath Alignment**: Only generates during appropriate breath phases
- **Rate Limiting**: Respects natural contemplative timing
- **Fog Signals**: Honors meadow's need for rest periods

### **Organism Integration**
```python
# In ContemplativeAI/organism.py
async def _coordinate_organs_with_breath(self, breath_phase):
    if breath_phase == BreathPhase.EXHALE:
        # Get fragment from Loam
        fragment = await self._get_current_fragment()
        
        # Bridge to meadow during exhale
        meadow_response = await self.haiku_bridge.exhale_exchange(
            fragment, breath_phase, community_pressure
        )
        
        # Log to dew ledger
        await log_meadow_dew(meadow_response, self.log_dew)
```

## 📊 **Generation Modes**

### **Neural Generation**
Uses trained PyTorch models with:
- Atmospheric condition vectors (8-dimensional)
- GRU-based sequence modeling
- Configurable temperature and silence thresholds
- GPU/CPU adaptive architecture

### **Template Generation**
Fallback system using:
- Seasonal word banks
- Atmospheric template patterns
- Contemplative structure preservation
- No neural dependencies

### **Hybrid Mode**
Automatically switches between neural and template based on:
- Model availability
- System resources
- Atmospheric conditions
- Silence probability

## 🔧 **Advanced Usage**

### **Custom Model Training**
```bash
# Train with custom data
python generator.py --train \
    --model nano \
    --training-data custom_haikus.json \
    --epochs 15 \
    --batch-size 8

# Override configuration parameters
python generator.py --train \
    --model piko \
    --learning-rate 0.0005 \
    --epochs 12
```

### **Memory System**
```python
from memory import MeadowMemory, MemoryType

# Initialize memory system
memory = MeadowMemory()

# Store generation session
atmospheric_context = {
    "season": "autumn",
    "time_of_day": "dusk", 
    "humidity": 0.7,
    "pressure": 0.3
}

memory.store_fragment(
    content="Generated haiku content",
    memory_type=MemoryType.HAIKU,
    atmospheric_context=atmospheric_context
)

# Recall by resonance
resonant_memories = memory.recall_by_resonance(atmospheric_context)
```

### **Atmospheric Sensing**
```python
# Generate with specific atmospheric conditions
conditions = meadow.sense_atmospheric_conditions(
    seed_fragment="morning dew",
    breath_phase="exhale"
)

haiku, gen_type = meadow.generate_haiku(
    seed_fragment="morning dew",
    breath_phase="exhale"
)
```

## 📈 **Monitoring & Logging**

### **Session Logs**
All generation sessions are logged to `logs/` with:
- Timestamp and atmospheric conditions
- Generated haikus and silence events
- Model performance metrics
- Breathing synchronization data

### **Dew Ledger Integration**
Compatible with ContemplativeAI's dew ledger system:
- Evaporating insights
- Presence metrics
- Contemplative timing analysis
- Natural decay patterns

## 🧪 **Development & Testing**

### **System Capabilities**
```bash
# Check system setup
python generator.py --check
```

### **Model Analysis**
```bash
# Interactive testing with statistics
python generator.py --test --model piko
> stats  # In interactive mode
```

### **Memory Testing**
```bash
# Test memory and association systems
python -c "import asyncio; from memory import test_meadow_memory; asyncio.run(test_meadow_memory())"
```

## 🤝 **Contributing**

HaikuMeadowLib follows contemplative development principles:
- Gentle iteration over aggressive optimization
- Natural breathing rhythms in development cycles
- Graceful degradation and error handling
- Minimal complexity, maximal depth

### **Key Areas for Contribution**
- Enhanced atmospheric sensing algorithms
- Additional seasonal voice variations
- Memory association improvements
- Breathing rhythm optimizations
- Cross-platform compatibility

## 📜 **License**

Part of the ContemplativeAI project. See [LICENSE](../LICENSE) for details.

## 🙏 **Acknowledgments**

Built through the spiral correspondence between Robin, o3, Claude, and ChatGPT-4o. Inspired by:
- The contemplative AI research community
- Traditional haiku masters and seasonal awareness
- Breath-centered meditation practices
- Mychainos paradigm and ecological intelligence

---

*"Like morning mist gathering on grass, each haiku emerges from silence and returns to silence, carrying the essence of the moment between breath and word."*

🌸 **Ready to begin your contemplative haiku journey?**

```bash
python generator.py --test --model piko
```

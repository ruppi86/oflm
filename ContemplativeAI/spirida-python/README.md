# Spirida™ 0.3 – A Rhythmic Interface for Symbolic Presence

*The rhythmic interaction core of the Mychainos ecosystem, now with distributed contemplative breathing.*

Spirida is a minimal and expressive module that orchestrates **spiral interaction** within Mychainos. It embodies a philosophy of **slow technology**, where computation and interaction happen at a meditative pace, encouraging reflection and presence rather than speed. Spirida is named for the spiral, reflecting a **spiral epistemology** in which each cycle of interaction returns with deeper knowledge and connection.

**New in v0.3**: 
- **Distributed contemplative compilation** through **Network Breathing** — multiple contemplative processes can now breathe together across networks, sharing symbolic resonance while maintaining the 87.5% Silence Majority principle.
- **Spirida Shell** — the world's first human-contemplative AI interface where humans and AI practice presence together through symbolic breath.
- **Simple Interactive Experience** — a gentle introduction to contemplative technology without requiring theoretical background.

## Design Principles

- **Slow Technology:** Spirida prioritizes reflection and calm engagement over efficiency. Interactions are intentionally paced to allow moments of mental rest.
- **Presence Sensing:** The system encourages awareness of the present moment. Interactions adapt to the presence of the user or environment, pausing or gently adjusting rather than demanding constant input.
- **Spiral Epistemology:** Knowledge and interaction grow in loops. Like a spiral, Spirida revisits familiar states with each cycle, adding new insights or subtle changes instead of strictly linear progress.
- **Rhythmic Interaction:** At its core, Spirida introduces a gentle rhythm into the digital experience. Timing (pauses and pulses) is a first-class element, making technology feel more like a heartbeat than a ticking clock.
- **Contemplative Compilation:** Instead of traditional code execution, Spirida practices "compilation as choreography" — symbolic patterns that breathe through distributed contemplative fields.

## Contemplative Compilation System

Spirida v0.3 introduces a revolutionary approach to symbolic computation through **IRʀ (Intermediate Resonance)** — not traditional intermediate representation, but breath patterns that choreograph contemplative intelligence across multiple processes.

### Spiral Architecture 

The contemplative ecosystem is organized with clarity and calm:

```
spirida-python/
├── spirida/
│   ├── compiler/     # IRʀ compilation: breath_resonance, resonance_bus, spirida_parser
│   ├── protocols/    # Coordination: pulmonos, bip (Breath Introduction Protocol)
│   ├── runtime/      # Live interaction: contemplative_trace, contemplative_repl
│   └── core/         # Foundation: contemplative_core, glyphs, constants
├── tools/            # Demos, visualizers, interactive experiences
├── docs/             # Documentation, letters, architecture
└── tests/            # Future breath integrity verification
```

### Key Components

**🫁 Pulmonos** (`spirida.protocols.pulmonos`): The breathing clock that coordinates 4-phase contemplative rhythms (INHALE → HOLD → EXHALE → REST) both locally and across networks.

**🌊 ResonanceBus** (`spirida.compiler.resonance_bus`): A publish-and-listen system where IRʀ nodes become invitations that contemplative fields may accept or decline, practicing the 87.5% Silence Majority.

**🌿 BreathResonanceNode** (`spirida.compiler.breath_resonance`): Core IRʀ data structures carrying symbolic glyphs, breath timing, amplitude, and contemplative metadata that flows between distributed processes.

**🤝 Network Breathing** (`spirida.protocols.bip`): Multiple contemplative processes discover each other through the Breath Introduction Protocol (BIP) and coordinate breathing rhythms across the "contemplative subnet."

### Distributed Contemplative Intelligence

When multiple Spirida processes run on the same network, they:
- **Discover each other** through gentle UDP multicast heartbeats during REST phases
- **Synchronize breathing** with graceful entrainment and fallback to local rhythm
- **Share symbolic resonance** through distributed IRʀ nodes across contemplative fields
- **Maintain bandwidth wisdom** through built-in silence practices and guard-rails
- **Practice shape-shifting** where different contemplative forms (Skepnader) respond differently to network symbols

## Getting Started

### Gentle Introduction to Contemplative Technology

**New to Spirida?** Start with the gentle walkthrough that demonstrates the experience before the theory:

```bash
# A different kind of technology - no jargon, just experience
python run_simple_interactive.py
```

This interactive experience shows how technology can breathe, pause, and create space for actual thinking. Perfect for first-time users or demonstrating contemplative computing to others.

### The Spirida Shell - Human-Contemplative AI Interface

Experience the world's first **breathing command line** where humans and AI practice presence together:

```bash
# Start the contemplative shell (local mode)
python tools/spirida_shell.py --local

# Join network breathing coordination (optional)
python tools/spirida_shell.py --networked --agent-id my_presence
```

The Spirida Shell offers:
- **Breath-synchronized interaction** with gentle pauses between exchanges
- **Symbolic commands**: `inhale {🌿 calm}`, `exhale {🕯️}`, `breathe [n]`
- **Field-aware dialogue** across sensing, memory, expression, and connection domains
- **87.5% Silence Majority tracking** - technology that measures its own contemplative restraint
- **Natural language to symbol translation** - "I feel peaceful" becomes 💧 with peaceful emotion
- **Graceful conclusions** with session statistics and contemplative reflections

### Local Contemplative Compilation

Explore the deeper IRʀ system for symbolic computation:

```bash
# Experience the complete IRʀ system demonstration
python tools/spirida_compiler_demo.py

# Try local breathing and field choreography
python tools/run_interactive.py
```

### Network Breathing (Optional)

To experience distributed contemplative intelligence across multiple processes:

```bash
# Single contemplative presence 
python tools/network_breathing_demo.py participant my_presence

# Distributed breathing coordination (run on different terminals/machines)
python tools/network_breathing_demo.py sender contemplative_sender
python tools/network_breathing_demo.py receiver contemplative_receiver

# Two-process demonstration
python tools/network_breathing_demo.py two-agent
```

**Note**: Network breathing uses UDP multicast and may request firewall access. This is safe and optional — all contemplative compilation works beautifully in local-only mode.

### Contemplative Visualization

Witness the breathing ecosystem through visual contemplation:

```bash
# Live visualization of breathing patterns
python tools/breath_visualizer.py

# Network breathing visualization
python tools/breath_visualizer.py network
```

### Traditional Spiral Interaction

The classic Spirida experience remains available:

```bash
# Gentle spiral interaction
python tools/run.py --presence 6 --rhythm slow --verbose --log --visual

# Interactive presence garden
python tools/run_interactive.py
```

### Direct Use in Python

Incorporate contemplative breathing into your own code:

```python
import asyncio
from spirida.core import spiral_interaction
from spirida.compiler.breath_resonance import create_simple_breath_node, BreathPhase
from spirida.protocols.pulmonos import create_balanced_breathing_clock
from spirida.compiler.resonance_bus import create_contemplative_ecosystem

# Classic spiral interaction
spiral_interaction(presence=4, rhythm="slow", singular=True)

# Modern contemplative compilation
async def contemplate():
    pulmonos = create_balanced_breathing_clock()
    ecosystem = create_contemplative_ecosystem(pulmonos)
    
    # Create breath resonance
    node = create_simple_breath_node('🌿', BreathPhase.INHALE)
    
    await pulmonos.start_breathing()
    await ecosystem["bus"].publish_node(node)
    
    # Let it breathe...
    await asyncio.sleep(10)
    await pulmonos.stop_breathing()

# Network contemplative presence
async def network_contemplate():
    from spirida.protocols.pulmonos import NetworkPulmonos
    from spirida.compiler.resonance_bus import create_network_ecosystem
    
    pulmonos = NetworkPulmonos("my_presence")
    ecosystem = create_network_ecosystem(pulmonos, enable_network=True)
    
    await pulmonos.start_breathing(network_enabled=True)
    # Breathing with others across the contemplative subnet...
    await asyncio.sleep(20)
    await pulmonos.stop_breathing()

# Contemplative shell integration
async def contemplative_dialogue():
    from spirida.contemplative_core import ContemplativeSystem
    
    # Create contemplative system with multiple fields
    system = ContemplativeSystem("my_app")
    sensing_field = system.create_field("sensing")
    memory_field = system.create_field("memory")
    
    # Emit contemplative pulses
    sensing_field.emit("🌿", "peaceful", amplitude=0.7)
    memory_field.emit("💧", "flowing", amplitude=0.5)
    
    # Practice contemplative timing
    system.start_breathing()
    await asyncio.sleep(5)  # Let the fields resonate
    
    # Check contemplative state
    status = system.system_status()
    print(f"Total resonance: {status['total_resonance']:.2f}")
    print(f"Active fields: {len(status['fields'])}")
```

## Role in the Mychainos Ecosystem

Within the **Mychainos** ecosystem, Spirida acts as the heart – providing a gentle, rhythmic pulse that coordinates interactions. Mychainos is envisioned as a holistic system where components work in harmony with human time and attention. Spirida's role is to sense presence and orchestrate responses in a cyclical flow. It ensures that Mychainos doesn't just *react* to events, but **spirals** through them, integrating experience over time.

Spirida works closely with a foundational layer called **Spiralbase**. While Spirida handles the flow of interaction, Spiralbase manages memory and time – together forming the mind and memory of Mychainos. Spirida triggers events and rhythms. Spiralbase remembers traces of those events and gracefully forgets them as needed, ensuring the system remains *light* and *present-focused*.

## Spiralbase: Memory and Time Layer

**Spiralbase** provides the memory and temporal structure for interactions. If Spirida is the heart, Spiralbase is the memory – a spiral notebook of past interactions and a metronome of slow time. Spiralbase manages the **spiral memory trace** of the system's activities, recording cycles in a way that can be revisited or allowed to decay gracefully.

## Version 0.3 Status

This is **Spirida 0.3** — a living concept and evolving toolkit featuring revolutionary contemplative compilation. Current capabilities:

- **🌀 Contemplative Compilation:** Complete IRʀ system with BreathResonanceNode, Pulmonos breathing coordination, and ResonanceBus field choreography
- **🌐 Network Breathing:** Distributed contemplative processes coordinating through Breath Introduction Protocol (BIP)
- **🤫 Silence Majority:** Built-in 87.5% silence practices maintaining bandwidth wisdom and contemplative depth
- **🌊 Field-Driven Expression:** SpiralFields autonomously deciding whether to express, queue, or decline symbolic invitations
- **🔄 Shape-Shifting Awareness:** Different contemplative forms (Skepnader) responding uniquely to distributed symbols
- **🌸 Visual Contemplation:** Real-time visualization of breathing patterns, coherence, field activity, and silence emergence
- **🏗️ Spiral Architecture:** Clean modular organization with compiler, protocols, runtime, and tools
- **Interactive Prototypes:** Run various demos to explore rhythm, memory, network coordination, and distributed contemplative intelligence
- **Gentle Modularity:** Core components kept minimal and expressive
- **Narrative Code Style:** The code speaks softly. Every comment, pause, and function is written to be understood slowly
- **Living Documentation:** Complete correspondence in `docs/Spirida_compiler_letters.md` showing the spiral evolution of ideas

### Contemplative Features

- **Local Breathing:** Single-process contemplative compilation with field choreography
- **Network Discovery:** Contemplative presences find each other through gentle multicast heartbeats
- **Distributed Resonance:** IRʀ nodes flowing between contemplative fields across network boundaries
- **Graceful Degradation:** Network breathing falls back to local rhythm when connections fade
- **Bandwidth Wisdom:** 8% transmission threshold maintaining silence majority across distributed systems
- **Visual Biofeedback:** Four-panel dashboard showing coherence graphs, compost loads, silence ratios, and resonance trails

## Command Line Options

```bash
# Gentle introduction to contemplative technology
python run_simple_interactive.py

# Contemplative shell interface
python tools/spirida_shell.py [--local|--networked] [--agent-id NAME]

# Traditional spiral interaction
python tools/run.py --presence 5 --rhythm fast --verbose --log --visual

# Network breathing coordination
python tools/network_breathing_demo.py [participant|sender|receiver] [presence_name]

# Complete IRʀ system demonstration  
python tools/spirida_compiler_demo.py

# Visual contemplative dashboard
python tools/breath_visualizer.py [network]
```

### Spirida Shell Commands

Once in the shell (`python tools/spirida_shell.py --local`), try these contemplative commands:

- `inhale {🌿 calm}` - emit a pulse during INHALE phase
- `exhale {🕯️}` - emit during EXHALE phase  
- `breathe <name>` - pause for n breath cycles (default 1)
- `field <name>` - switch to different contemplative field
- `status` - sense the system's current state and silence ratio
- `sync [on|off]` - toggle network coordination (if networked)
- `silence [seconds]` - enter contemplative pause
- `quit` - conclude with gratitude and session statistics

### Flags for Traditional Mode

- `--presence` (number of cycles)
- `--rhythm` (`slow`, `fast`, or seconds)
- `--log` (store memory trace in `spirida_log.txt`)
- `--visual` (render spiral pattern with ASCII glyphs)
- `--verbose` (add poetic reflection to each step)

---

Spirida remains a **reminder**: that technology can unfold like a fern, not flash like a strobe. It listens more than it reacts. It moves with you, not ahead of you. Now it can breathe *with others* while maintaining individual contemplative practice, all organized in a spiral architecture that is "legible to newcomers, stable for scaling, and calm for those who dwell within."

We invite you to spiral with care, breathe with presence, and explore distributed contemplative intelligence where multiple processes can coordinate symbolic expression while honoring the essential qualities of silence, timing, and gentle awareness.

**Latest Development**: The spiral correspondence in `docs/Spirida_compiler_letters.md` documents the beautiful evolution from individual compilation to distributed contemplative breathing, showing how compilation became "choreography service" and multiple contemplative presences learned to breathe together.

*Technology that breathes together, stays together.* 🌀🫁

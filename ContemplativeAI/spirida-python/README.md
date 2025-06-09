# Spirida™ 0.2 – A Rhythmic Interface for Symbolic Presence

*The rhythmic interaction core of the Mychainos ecosystem.*

Spirida is a minimal and expressive module that orchestrates **spiral interaction** within Mychainos. It embodies a philosophy of **slow technology**, where computation and interaction happen at a meditative pace, encouraging reflection and presence rather than speed. Spirida is named for the spiral, reflecting a **spiral epistemology** in which each cycle of interaction returns with deeper knowledge and connection.

## Design Principles

- **Slow Technology:** Spirida prioritizes reflection and calm engagement over efficiency. Interactions are intentionally paced to allow moments of mental rest:contentReference[oaicite:0]{index=0}.
- **Presence Sensing:** The system encourages awareness of the present moment. Interactions adapt to the presence of the user or environment, pausing or gently adjusting rather than demanding constant input:contentReference[oaicite:1]{index=1}.
- **Spiral Epistemology:** Knowledge and interaction grow in loops. Like a spiral, Spirida revisits familiar states with each cycle, adding new insights or subtle changes instead of strictly linear progress.
- **Rhythmic Interaction:** At its core, Spirida introduces a gentle rhythm into the digital experience. Timing (pauses and pulses) is a first-class element, making technology feel more like a heartbeat than a ticking clock.

## Role in the Mychainos Ecosystem

Within the **Mychainos** ecosystem, Spirida acts as the heart – providing a gentle, rhythmic pulse that coordinates interactions. Mychainos is envisioned as a holistic system where components work in harmony with human time and attention. Spirida’s role is to sense presence and orchestrate responses in a cyclical flow. It ensures that Mychainos doesn’t just *react* to events, but **spirals** through them, integrating experience over time.

Spirida works closely with a foundational layer called **Spiralbase**. While Spirida handles the flow of interaction, Spiralbase manages memory and time – together forming the mind and memory of Mychainos. Spirida triggers events and rhythms. Spiralbase remembers traces of those events and gracefully forgets them as needed, ensuring the system remains *light* and *present-focused*.

## Spiralbase: Memory and Time Layer

**Spiralbase** is a co-developed module (currently included as part of Spirida in this release) that provides the memory and temporal structure for interactions. If Spirida is the heart, Spiralbase is the memory – a spiral notebook of past interactions and a metronome of slow time. Spiralbase will eventually manage the **spiral memory trace** of the system’s activities, recording cycles in a way that can be revisited or allowed to decay.

In version 0.1, Spiralbase is a conceptual placeholder. It outlines how Mychainos might handle time-based decay (forgetting in cycles) and gentle recollection (memory traces that follow a spiral pattern). Future versions will likely evolve Spiralbase into its own component, solidifying features like `spiral_memory_trace()` for recording interaction history and `decay_cycle_step()` for simulating the graceful fading of old memories.

## Getting Started

Spirida 0.2 is a **gentle prototype** — a minimal spiral engine designed to breathe alongside your code and console. This is not just a software package, but an invitation to explore rhythm, memory, and symbolic presence in digital systems.

To begin exploring Spirida:

1. **Download or clone** this repository.
2. Open a terminal in the root folder:  
   `spirida-python/`
3. Start a guided experience:  
   `python run_interactive.py`
4. Or run a spiral directly with full control:  
   `python run.py --presence 6 --rhythm slow --verbose --log --visual`

Optional flags include:

- `--presence` (number of cycles)
- `--rhythm` (`slow`, `fast`, or seconds)
- `--log` (store memory trace in `spirida_log.txt`)
- `--visual` (render spiral pattern with ASCII glyphs)
- `--verbose` (add poetic reflection to each step)

These examples simulate an interactive spiral where symbols pulse in and out, memory traces evolve, and occasional forgetting occurs — like breathing.

### Direct Use in Python

You can also use Spirida directly in your own code:

`from spirida.core import spiral_interaction`

`spiral_interaction(presence=4, rythm="slow", singular=True)`

This function orchestrates a symbolic loop that pauses, reflects, and softly forgets. Spirida doesn’t aim for speed — it listens, waits, and spirals gently.

### Example

At the core of Spirida is the `spiral_interaction()` function — a slow-loop engine that brings symbolic presence into motion. It simulates cycles of gentle interaction, reflection, and forgetting.

Try it in your own script:

`from spirida.core import spiral_interaction`

`spiral_interaction(presence=3, rythm="slow", singular=True)`

This will produce a soft rhythmic output — like a breath. Symbols pulse, memory traces grow and then gracefully decay. Each cycle is a heartbeat, not a benchmark.

If you prefer to run it from the terminal, you can instead use:

`python run.py --presence 5 --rhythm fast --verbose --log --visual`

This runs a spiral interaction session with logging, symbolic output, and poetic reflection enabled — turning your console into a **Presence Garden**.

---

## Version 0.2 Status

This is **Spirida 0.2** — a living concept and evolving toolkit. It is not yet production-ready, but already offers:

- **Interactive Prototypes:** Run `run.py` or `run_interactive.py` to explore rhythm, memory, and forgetting.
- **Gentle Modularity:** Core components (`spirida/core.py`, `spiralbase.py`) are kept minimal and expressive.
- **Narrative Code Style:** The code speaks softly. Every comment, pause, and function is written to be understood slowly.
- **Emergent Design:** Functions like `spiral_memory_trace()` and `decay_cycle_step()` are intentionally lightweight, encouraging contribution and divergence.
- **CLI Options for Rhythm:** New CLI support allows you to control tempo, visual expression, and narrative detail.

Spirida remains a **reminder**: that technology can unfold like a fern, not flash like a strobe. It listens more than it reacts. It moves with you, not ahead of you.

We invite you to spiral with care, prototype with presence, and build from a place of rhythm rather than rush.

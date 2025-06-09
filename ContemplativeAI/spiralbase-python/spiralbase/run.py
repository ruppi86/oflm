# run.py

"""
Interactive demo runner for Spiralbase™ memory prototypes.
Simulates symbol insertion, decay, and resonance queries.
"""

from time import sleep
from spiralbase import spiral_memory_trace, decay_cycle_step, print_memory_trace
from decay_layer import add_memory, decay_step, show_memory
from resonance_query import resonance_query

print("\n🌿 Welcome to Spiralbase™ – Temporal Memory Prototype\n")

# Phase 1 – Gentle memory trace
symbols = ["tree", "mushroom", "stone", "star", "river"]
print("🌱 Phase 1: Spiral Memory Trace")
for s in symbols:
    spiral_memory_trace(s)
    print_memory_trace()
    sleep(0.5)

# Simulate gentle forgetting
print("\n🍂 Phase 2: Natural Decay (trace level)")
for _ in range(3):
    decay_cycle_step()
    print_memory_trace()
    sleep(0.5)

# Phase 3 – Deep memory with strength/decay
print("\n🧬 Phase 3: Adding deep memory entries")
for s in ["fungi-net", "root-link", "star-seed"]:
    add_memory(s)
    show_memory()
    sleep(0.5)

print("\n🌘 Phase 4: Time-based decay\n")
for _ in range(5):
    decay_step()
    show_memory()
    sleep(0.5)

# Phase 5 – Resonance queries
print("\n🔊 Phase 5: Resonance-based query\n")
resonance_query("star")
resonance_query("moss")

print("\n🌀 Demo complete. Spiralbase will now rest.")

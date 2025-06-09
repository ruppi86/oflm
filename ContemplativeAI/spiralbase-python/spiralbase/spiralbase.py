"""
Spiralbase – memory and timekeeping for Spirida.
Implements gentle memory traces and decay cycles.

Note: this is still a prototype and not a fully implemented module. 

It is a concept.
A seed.

Let it grow by care and attuned attention.
"""

spiral_memory = []

def spiral_memory_trace(symbol):
    """
    Store a symbol in spiral memory (max 10 items).
    """
    global spiral_memory
    spiral_memory.append(symbol)
    if len(spiral_memory) > 10:
        spiral_memory.pop(0)

def decay_cycle_step():
    """
    Removes the oldest memory entry to simulate forgetting.
    """
    global spiral_memory
    if spiral_memory:
        forgotten = spiral_memory.pop(0)
        print(f"🍂 Forgotten: {forgotten}")

def print_memory_trace():
    """
    Print current spiral memory as a gentle trace.
    """
    if spiral_memory:
        print("🧠 Spiral trace: " + " ".join(spiral_memory))
    else:
        print("🧠 Spiral trace is empty.")

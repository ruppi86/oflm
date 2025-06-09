# decay_layer.py

"""
Simulates time-based decay for memory entries in Spiralbase™.
Each entry has a 'strength' value that fades over simulated time steps.
"""

import time

memory = []  # List of dicts: {'symbol': str, 'strength': float}

DECAY_RATE = 0.1  # How much each item fades per step (0.0 to 1.0)
THRESHOLD = 0.2   # Minimum strength before forgetting


def add_memory(symbol):
    memory.append({'symbol': symbol, 'strength': 1.0})


def decay_step():
    global memory
    for entry in memory:
        entry['strength'] -= DECAY_RATE
    memory = [e for e in memory if e['strength'] > THRESHOLD]


def show_memory():
    if not memory:
        print("🧠 Spiral memory faded.")
    else:
        for entry in memory:
            bar = "█" * int(entry['strength'] * 10)
            print(f"{entry['symbol']}: {bar}")
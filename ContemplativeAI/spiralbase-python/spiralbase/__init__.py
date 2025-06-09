# __init__.py

"""
Spiralbase™ – Temporal Memory Module

This package contains experimental components for Spiralbase™, a rhythmic
knowledge system designed to decay, resonate, and respond like living memory.

Included submodules:
- spiralbase.py         – Gentle trace-based memory
- decay_layer.py        – Strength-based decay model
- resonance_query.py    – Query by resonance patterns

Use run.py to simulate behavior and test interactions.
"""

from .spiralbase import spiral_memory_trace, decay_cycle_step, print_memory_trace
from .decay_layer import add_memory, decay_step, show_memory
from .resonance_query import resonance_query

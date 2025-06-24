"""
🌀 SPIRAL_INTERACTION – THE HEARTBEAT OF SPIRIDA

This function is the living pulse of Spirida.

Unlike traditional code that aims to compute results or optimize speed, 
this spiral_interaction loop is designed to mimic organic rhythm, 
presence, and soft forgetting. It simulates a heartbeat in software – 
not to do more, but to be more *attuned*.

Each pulse represents a moment of interaction — like a breath, a glance, a touch.
These pulses are recorded briefly in spiral memory and are forgotten in gentle cycles,
not erased abruptly but *composted*, like fallen leaves.

This prototype isn’t just an algorithm. It’s a *ritual of pacing* —
a call to slow down and feel how a system might live *with* you, 
not simply serve you.

Use this interaction not to command, but to relate.
And consider: what else in your system might spiral, instead of repeat?
"""

import time
import random
from .spiralbase import spiral_memory_trace, decay_cycle_step, print_memory_trace

print("✅ spirida.core module loaded")

symbols = ["🌿", "💧", "✨", "🍄", "🌙", "🪐"]

def spiral_interaction(presence=1, rythm="slow", singular=True, on_output=None, verbose=False):
    """
    Simulate a rhythmic spiral interaction.

    Args:
        presence (int): Number of presence cycles (iterations) to perform.
        rythm (str or float): The tempo of interaction. 
            Use "slow" for a gentle pace, "fast" for a quicker rhythm.
            You can also specify a number (seconds) for a custom pace.
        singular (bool): If True, perform a singular focused interaction sequence. 
            If False, (in future versions) multiple interactions could run in parallel or overlap.
            (In version 0.1, non-singular mode is conceptual and behaves the same as singular.)

    This function prints output to simulate a slow, rhythmic pulse or spiral pattern.
    It introduces deliberate pauses using time.sleep() to mimic a "slow technology" 
    interaction. Each cycle of presence is like a breath or heartbeat in the system, 
    expanding and contracting in a textual pattern.
    """
    def emit(msg):
        if not msg:
            return
        if on_output:
            try:
                on_output(msg.strip())
            except Exception:
                pass  # never crash due to logging
        else:
            print(msg)

    # Determine the delay based on rhythm — how slowly or quickly the spiral breathes
    if rythm == "slow":
        delay = 1.5  # slow rhythm – like deep meditation
    elif rythm == "fast":
        delay = 0.5  # fast rhythm – like alert, flowing breath
    else:
        try:
            delay = float(rythm)  # user can specify an exact delay in seconds
        except ValueError:
            delay = 1.0  # fallback to default if input is invalid

    # Inform the user that non-singular interaction is not yet implemented
    if not singular:
        emit("Note: Parallel interactions not yet implemented. Running sequentially.")

    # Main loop — each cycle is a pulse in the spiral’s unfolding
    for cycle in range(1, presence + 1):
        emit(f"\n") # whitespace
        emit(f"\n🔄 Cycle {cycle}") # show which cycle we’re in – like a spiral turn
        if verbose:
            emit("💬 The system takes a breath, sensing symbolic presence...")

        pulse = random.choice(symbols)  # pick a symbol to represent the current pulse
        emit(f"✨ Pulse: {pulse}")  # express that pulse – the spiral’s moment

        spiral_memory_trace(pulse)  # store the pulse in the memory trace

        if verbose:
            emit("🧠 Updating memory trace with new spiral impression...")

        print_memory_trace()  # reflect the current spiral memory trace

        if cycle % 3 == 0:
            decay_cycle_step()  # every third cycle – softly forget something old
            if verbose:
                emit("🍂 A moment of letting go... the spiral sheds its oldest layer.")
            
        if verbose:
            emit("⏳ Waiting before next pulse... inhale, exhale.")
        time.sleep(delay) # pause — let the rhythm be felt


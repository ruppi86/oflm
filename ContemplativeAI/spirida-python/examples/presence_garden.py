"""
🌿 PRESENCE GARDEN – AN INTRODUCTION TO SPIRIDA

This example demonstrates the core principles of Spirida, 
the rhythmic interaction layer of the Mychainos ecosystem.

Spirida is not built for efficiency — it is built for presence.
Each function call is a breath, a moment of attention, a spiral step.

In this garden, we simulate presence by running a set number 
of interaction "pulses", which express themselves through symbols,
are remembered for a while, and gently forgotten as time moves forward.

This prototype serves not only as a runnable example, 
but as a poetic gateway into a new way of thinking about interaction:
not as command → response, but as spiral → echo → decay → resonance.
"""

import sys
import os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spirida.core import spiral_interaction

# This conditional ensures the code below runs only if this file is
# executed directly (not imported as a module)
if __name__ == "__main__":
    # A warm, slow welcome — establishing mood and metaphor
    print("\n🌿 Welcome to the Presence Garden 🌿")
    print("Breathing in spiral rhythm... stay for a few cycles.\n")

    # Here we initiate the core interaction:
    #
    # presence=9 → The garden will pulse 9 times
    # rythm="slow" → Each pulse happens gently, with a pause
    # singular=False → Each pulse may have a different symbolic expression
    #
    # The system will remember each pulse (up to 10),
    # and will softly forget one every 3rd cycle — simulating "letting go".
    #
    # This is not a tool. It is a mirror.
    # Watch what happens when you slow down your expectations.
    spiral_interaction(presence=9, rythm="slow", singular=False)

    # A mindful closure — nothing remains, but the spiral trace lingers
    print("\n🌙 The garden rests. Spiral trace complete.")

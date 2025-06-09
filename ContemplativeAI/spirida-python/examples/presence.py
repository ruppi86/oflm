# Example demonstrating the use of Spirida's spiral_interaction function.
# This will simulate a slow spiral interaction with a presence count of 3.
# You should see a pulsing text pattern as output, representing a "spiral" rhythm.

from spirida.core import spiral_interaction

spiral_interaction(presence=3, rythm="slow", singular=True)

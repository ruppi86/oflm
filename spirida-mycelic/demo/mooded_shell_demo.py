#!/usr/bin/env python3
"""Mooded Shell Demo – validates new Letter IX components.

Runs a short simulation where ambient light toggles every 15 s to showcase
PhotoGate-driven breath adjustments and mood shifts in the bio-interface.
"""

import itertools
import time

from spirida_mycelic.bio_interface import SevenChannelBioInterface
from spirida_mycelic.photo_gate import PhotoGate
from spirida_mycelic.bio_mood import BioMood


def main():
    interface = SevenChannelBioInterface(mock_mode=True)
    photo = PhotoGate(threshold_lux=300.0)

    lux_pattern = itertools.cycle([100.0, 500.0])  # dim → bright → dim …

    print("🌿 Mooded Shell Demo – Ctrl+C to exit")
    try:
        for cycle in range(10):
            lux = next(lux_pattern)
            adj = photo.update(lux)
            # Read once to trigger mood update
            interface.read_channels()
            print(
                f"[{cycle}] Lux={lux:.0f} → BreathAdj hold={adj.hold:+.0f}s, "
                f"mood={interface.mood.name}"
            )
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\nDemo finished.")


if __name__ == "__main__":
    main() 
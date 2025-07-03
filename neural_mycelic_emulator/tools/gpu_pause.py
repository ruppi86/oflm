#!/usr/bin/env python3
"""GPU breathing wrapper – easier manual invocation.

Example:
    python -m neural_mycelic_emulator.tools.gpu_pause nano --count 10 --context myjob

This will invoke `nano_pause("myjob")` ten times and print the actual sleep
intervals.  Use Ctrl-C to stop early.
"""
from __future__ import annotations

import argparse
import time

from neural_mycelic_emulator.gpu_breathing import (
    femto_pause,
    piko_pause,
    nano_pause,
    mili_pause,
)

MODES = {
    "femto": femto_pause,
    "piko": piko_pause,
    "nano": nano_pause,
    "mili": mili_pause,
}

def main() -> None:
    p = argparse.ArgumentParser("GPU breathing helper")
    p.add_argument("mode", choices=MODES.keys(), help="Breathing mode")
    p.add_argument("--count", type=int, default=0, help="Iterations (0 = infinite)")
    p.add_argument("--context", default="manual", help="Context label for log prints")
    args = p.parse_args()

    fn = MODES[args.mode]
    i = 0
    try:
        while True:
            i += 1
            slept = fn(args.context)
            print(f"[{i}] slept {slept*1000:.1f} ms ({args.mode})")
            if args.count and i >= args.count:
                break
            # additional small gap for readability
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrupted – exiting.")

if __name__ == "__main__":
    main() 
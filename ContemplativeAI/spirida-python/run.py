"""
run.py – Spirida launcher (CLI version)

This file serves as a clear, documented entry point into the Spirida system.
Run it from the terminal to simulate a spiral interaction using gentle rhythm.

Example:
    python run.py --presence 8 --rhythm fast --singular False --log --visual --verbose

Arguments:
    --presence  : How many spiral cycles (default: 5)
    --rhythm    : 'slow', 'fast', or a numeric value (e.g. 0.8)
    --singular  : Whether to stay on one symbol (True/False)
    --log       : Save output to 'spirida_log.txt'
    --visual    : Print spiral trace as it happens
    --verbose   : Show gentle, narrative explanations per cycle
"""

import argparse
import sys
import os
from spirida.core import spiral_interaction

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser(description="🌿 Run a Spirida spiral interaction.")
    parser.add_argument("--presence", type=int, default=5, help="Number of presence cycles")
    parser.add_argument("--rhythm", type=str, default="slow", help="Rhythm: slow, fast, or seconds")
    parser.add_argument("--singular", type=str, default="True", help="Singular mode (True/False)")
    parser.add_argument("--log", action="store_true", help="Save to spirida_log.txt")
    parser.add_argument("--visual", action="store_true", help="Print spiral to terminal")
    parser.add_argument("--verbose", action="store_true", help="Narrate what is happening in plain language")

    args = parser.parse_args()
    singular = args.singular.lower() in ["true", "1", "yes", "y"]

    print("\n🌿 Welcome to Spirida via run.py 🌿")
    print(f"Initiating with presence={args.presence}, rhythm={args.rhythm}, singular={singular}\n")

    # Optional log callback
    def log_callback(msg):
        if args.log:
            with open("spirida_log.txt", "a", encoding="utf-8") as f:
                f.write(msg.strip() + "\n")
        if args.visual:
            print(msg)

    # Write session header
    if args.log:
        with open("spirida_log.txt", "a", encoding="utf-8") as f:
            f.write("\n--- New Spirida Session ---\n")
            f.write(f"Presence: {args.presence}, Rhythm: {args.rhythm}, Singular: {singular}, Verbose: {args.verbose}\n")

    # Run main interaction
    spiral_interaction(
        presence=args.presence,
        rythm=args.rhythm,
        singular=singular,
        on_output=log_callback if (args.log or args.visual) else None,
        verbose=args.verbose
    )

    print("\n🌙 Spirida session complete.")
    if args.log:
        print("📝 Output spriral saved to spirida_log.txt")

if __name__ == "__main__":
    main()

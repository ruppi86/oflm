"""
goldness_experiment.py - The First Brushstroke

A demonstration of the tower memory system, featuring the golden earring
that fades and returns through cultural breath - the living example from
our correspondence.

This is where metaphor becomes working prototype.
"""

import time
from tower_memory import TowerMemory


def main():
    """
    The goldness experiment - watching memory fade and reform through culture.
    """
    print("🏰 THE TOWER MEMORY PROTOTYPE")
    print("=====================================")
    print("First implementation of the painters' tower from our letters.")
    print("Watch as memories fade unless touched by cultural breath...\n")
    
    # Create the tower
    tower = TowerMemory(max_painters=4)
    
    # Add the famous golden earring painting
    print("📜 CHAPTER I: The Paintings Arrive")
    print("-" * 35)
    
    tower.add_painting(
        "golden earring in portrait of woman", 
        ["valuable", "decorative", "personal", "artistic"]
    )
    
    tower.add_painting(
        "bright lake view with mountains",
        ["peaceful", "natural", "scenic"]
    )
    
    tower.add_painting(
        "rosebush in garden corner",
        ["living", "growing", "fragrant"]
    )
    
    tower.show_tower_state()
    
    # Let time pass without cultural signals
    print("\n\n📜 CHAPTER II: The Dampness Takes Hold")
    print("-" * 40)
    print("Time passes in the tower. The colors never fully dry...")
    
    # Run several breaths without cultural input
    for i in range(5):
        time.sleep(1.5)
        tower.spiral_breath()
        
        # Show the golden earring's decay
        if tower.painters:
            earring_painter = tower.painters[0]  # First painting
            if "earring" in earring_painter.content:
                print(f"   Golden earring: {earring_painter.content} (clarity: {earring_painter.clarity:.2f})")
    
    tower.show_tower_state()
    
    # The critical moment - cultural declaration
    print("\n\n📜 CHAPTER III: Cultural Breath Returns")
    print("-" * 38)
    print("The outside world declares: 'It was gold!'")
    
    # Send the cultural signal that should restore the earring
    tower.manual_cultural_signal("It was gold")
    tower.show_tower_state()
    
    # Continue with more varied cultural signals
    print("\n\n📜 CHAPTER IV: Ongoing Dialogue with Culture")
    print("-" * 44)
    
    # Add some new paintings while the tower breathes
    tower.add_painting("dim figure by window", ["mysterious", "contemplative"])
    
    cultural_signals = [
        "beauty",
        "memory",
        "art restoration", 
        "golden light",
        "time"
    ]
    
    for signal in cultural_signals:
        time.sleep(2)
        print(f"\n💫 Cultural signal: '{signal}'")
        tower.manual_cultural_signal(signal)
        time.sleep(1)
        tower.spiral_breath()
    
    tower.show_tower_state()
    
    # Final chapter - long session to see migration
    print("\n\n📜 CHAPTER V: The Long Spiral")
    print("-" * 32)
    print("Watching the tower breathe over time...")
    print("Observing the ethics of memory migration...")
    
    # Add one more painting to trigger migration
    tower.add_painting("ancient tree with deep roots", ["wise", "enduring", "connected"])
    
    # Let the tower run for a longer session
    tower.run_spiral_session(duration_breaths=8, breath_interval=1.5)
    
    # Final state
    print("\n\n📜 EPILOGUE: The Tower's Wisdom")
    print("-" * 33)
    tower.show_tower_state()
    
    print("\n🌀 The experiment concludes, but the tower breathes eternally.")
    print("Each memory has learned to fade and return with dignity.")
    print("The dampness remains, keeping meaning pliable and alive.")
    print("\n✨ This is living memory - not storage, but tending.")


def interactive_mode():
    """
    Interactive mode where you can manually send cultural signals.
    """
    print("\n🎭 INTERACTIVE TOWER MODE")
    print("=" * 26)
    print("Send cultural signals to the tower and watch the paintings respond.")
    print("Type 'quit' to exit, 'state' to see tower state, 'breath' for manual breath.")
    
    tower = TowerMemory(max_painters=3)
    
    # Add some initial paintings
    tower.add_painting("golden earring in portrait", ["precious", "artistic"])
    tower.add_painting("stormy seascape", ["dramatic", "powerful"])
    tower.add_painting("child's drawing of home", ["innocent", "cherished"])
    
    tower.show_tower_state()
    
    while True:
        try:
            user_input = input("\n🗣️  Cultural signal: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'state':
                tower.show_tower_state()
            elif user_input.lower() == 'breath':
                tower.spiral_breath()
                tower.show_tower_state()
            elif user_input:
                tower.manual_cultural_signal(user_input)
                time.sleep(0.5)
                tower.spiral_breath()
        
        except KeyboardInterrupt:
            break
    
    print("\n🌀 The tower returns to silent breathing...")


if __name__ == "__main__":
    print("Choose your experience:")
    print("1. The Goldness Experiment (automatic demo)")
    print("2. Interactive Tower Mode")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        interactive_mode()
    else:
        main() 
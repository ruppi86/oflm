#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌿 Spirida Simple - A Gentle Introduction to Breathing Technology

A walkthrough that demonstrates how technology can feel different -
more restorative, more thoughtful, more human.

No contemplative jargon. Just experience.
"""

import asyncio
import time
import random
import sys
from typing import Optional

# Try to connect to existing ecosystem
try:
    from spirida.contemplative_core import ContemplativeSystem, SpiralField
    SPIRIDA_AVAILABLE = True
except ImportError:
    SPIRIDA_AVAILABLE = False

# Try to connect to HaikuMeadowLib
try:
    import sys
    import os
    
    # Calculate path to haikumeadowlib-python directory
    current_dir = os.path.dirname(__file__)
    haiku_path = os.path.join(current_dir, '..', '..', 'haikumeadowlib-python')
    haiku_path = os.path.abspath(haiku_path)
    
    # Only add to path if directory exists
    if os.path.exists(haiku_path) and os.path.exists(os.path.join(haiku_path, 'generator.py')):
        sys.path.append(haiku_path)
        from generator import HaikuGenerator
        HAIKU_AVAILABLE = True
    else:
        HAIKU_AVAILABLE = False
except ImportError:
    HAIKU_AVAILABLE = False


class SimpleBreathingExperience:
    """A gentle introduction to contemplative technology."""
    
    def __init__(self):
        self.flowers = ["🌸", "🌺", "🌻", "🌷", "🌹", "💐", "🌼", "🌵", "🌿", "🍀"]
        self.current_flower_index = 0
        
        # Simple contemplative system if available
        self.system = None
        if SPIRIDA_AVAILABLE:
            self.system = ContemplativeSystem("simple_experience")
            self.field = self.system.create_field("gentle")
        
        # Simple haiku generator if available
        self.haiku_gen = None
        if HAIKU_AVAILABLE:
            try:
                self.haiku_gen = HaikuGenerator()
            except:
                pass
    
    def show_flower_loading(self, message: str, duration: float = 3.0, flower_interval: float = 0.5):
        """Show a gentle loading animation with flowers."""
        print(f"\n{message}")
        
        flowers_shown = []
        start_time = time.time()
        last_flower_time = start_time
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # Add a new flower every interval
            if current_time - last_flower_time >= flower_interval:
                if self.current_flower_index < len(self.flowers):
                    flowers_shown.append(self.flowers[self.current_flower_index])
                    self.current_flower_index += 1
                    last_flower_time = current_time
                    
                    # Show the growing garden
                    print(f"\r   {''.join(flowers_shown)}", end="", flush=True)
            
            time.sleep(0.1)
        
        print("\n")
    
    async def breathing_pause(self, seconds: float = 2.0, show_dots: bool = True):
        """A gentle pause that shows this technology breathes."""
        if show_dots:
            for i in range(int(seconds * 2)):
                print("⋯", end="", flush=True)
                await asyncio.sleep(0.5)
            print()
        else:
            await asyncio.sleep(seconds)
    
    async def welcome(self):
        """Welcome experience that demonstrates the difference."""
        print("\n" + "=" * 50)
        print("  🌿 Welcome to a Different Kind of Technology")
        print("=" * 50)
        
        await self.breathing_pause(1.5)
        
        print("\nMost technology rushes.")
        print("This technology... pauses.")
        
        await self.breathing_pause(2.0)
        
        print("\nMost technology demands attention.")
        print("This technology... makes space for thinking.")
        
        await self.breathing_pause(2.0)
        
        print("\nLet's explore what that feels like.")
        
        await self.breathing_pause(1.5)
    
    async def show_menu(self):
        """Simple menu with clear options."""
        print("\n" + "─" * 40)
        print("What would you like to experience?")
        print("─" * 40)
        print()
        print("1. Feel how technology can breathe")
        print("2. See gentle loading (instead of spinning wheels)")
        print("3. Experience thoughtful responses")
        if HAIKU_AVAILABLE:
            print("4. Get a contemplative haiku")
        print("5. Learn what this actually is")
        print("6. Exit gracefully")
        print()
        
        while True:
            try:
                choice = input("Choose (1-6): ").strip()
                if choice in ['1', '2', '3', '4', '5', '6']:
                    if choice == '4' and not HAIKU_AVAILABLE:
                        print("Haiku feature not available. Try another option.")
                        continue
                    return choice
                else:
                    print("Please choose a number between 1-6.")
            except (EOFError, KeyboardInterrupt):
                return '6'
    
    async def experience_breathing(self):
        """Demonstrate how technology can have rhythm."""
        print("\n🫁 Technology That Breathes")
        print("─" * 25)
        
        await self.breathing_pause(1.0)
        
        print("\nMost command lines wait for you.")
        print("This one breathes with you.")
        
        await self.breathing_pause(2.0)
        
        print("\nLet's take 3 breaths together:")
        
        for i in range(3):
            print(f"\n  Breath {i+1}:")
            
            print("    🫁 Inhale...", end="")
            await self.breathing_pause(1.5, False)
            print(" ✨")
            
            print("    🤲 Hold...", end="")
            await self.breathing_pause(1.0, False)
            print(" ⭕")
            
            print("    💨 Exhale...", end="")
            await self.breathing_pause(1.5, False)
            print(" 🌿")
            
            if i < 2:
                await self.breathing_pause(0.5, False)
        
        await self.breathing_pause(2.0)
        print("\nNotice how different that felt?")
        print("Technology doesn't have to be rushed.")
    
    async def experience_gentle_loading(self):
        """Show loading that feels restorative."""
        print("\n🌻 Gentle Loading")
        print("─" * 15)
        
        await self.breathing_pause(1.0)
        
        print("\nInstead of spinning wheels or progress bars,")
        print("what if loading felt like watching a garden grow?")
        
        await self.breathing_pause(2.0)
        
        # Reset flower counter for fresh experience
        self.current_flower_index = 0
        
        self.show_flower_loading("Growing a small digital garden...", 4.0, 0.6)
        
        print("See? Waiting doesn't have to feel like waiting.")
        print("It can feel like... being present.")
    
    async def experience_thoughtful_response(self):
        """Demonstrate AI that thinks before speaking."""
        print("\n💭 Thoughtful Responses")
        print("─" * 20)
        
        await self.breathing_pause(1.0)
        
        print("\nMost AI responds instantly.")
        print("But instant isn't always wise.")
        
        await self.breathing_pause(2.0)
        
        print("\nLet me show you the difference...")
        
        await self.breathing_pause(1.0)
        
        # Instant response
        print("\n[Instant AI]")
        print("User: 'How are you?'")
        print("AI: 'I'm doing well, thank you for asking! How can I help you today?'")
        
        await self.breathing_pause(3.0)
        
        # Thoughtful response
        print("\n[Thoughtful AI]")
        print("User: 'How are you?'")
        print("AI: ...", end="")
        
        await self.breathing_pause(1.5, False)
        print()
        
        if SPIRIDA_AVAILABLE and self.system:
            # Create a gentle pulse in our field
            self.field.emit("🌿", "present", amplitude=0.3, decay_rate=0.01)
        
        print("    'I'm taking a moment to actually consider that question.")
        await self.breathing_pause(1.0, False)
        print("     Right now, I feel... curious and grateful.")
        await self.breathing_pause(1.0, False)
        print("     How are *you*, really?'")
        
        await self.breathing_pause(2.0)
        
        print("\nNotice the difference?")
        print("Space for actual thought changes everything.")
    
    async def get_contemplative_haiku(self):
        """Get a haiku from HaikuMeadowLib if available."""
        print("\n🌙 A Moment of Poetry")
        print("─" * 20)
        
        if not HAIKU_AVAILABLE or not self.haiku_gen:
            print("\nPoetry module not available right now.")
            print("But imagine: AI that shares poetry")
            print("when you need a moment of beauty...")
            return
        
        await self.breathing_pause(1.0)
        
        print("\nSometimes technology can offer")
        print("not just answers, but beauty...")
        
        await self.breathing_pause(2.0)
        
        # Reset flowers for haiku loading
        self.current_flower_index = 0
        self.show_flower_loading("Composing something gentle...", 3.0, 0.8)
        
        try:
            # Generate a haiku
            haiku = self.haiku_gen.generate_haiku()
            
            print("Here's a small gift:")
            print()
            
            # Display haiku with gentle pauses
            lines = haiku.strip().split('\n')
            for line in lines[:3]:  # Ensure it's a proper haiku
                print(f"    {line}")
                await self.breathing_pause(1.0, False)
            
            print()
            await self.breathing_pause(2.0)
            print("Poetry from the digital soil.")
            
        except Exception as e:
            print("The poetry is sleeping right now.")
            print("But imagine: technology that creates beauty")
            print("not just utility...")
    
    async def explain_what_this_is(self):
        """Gentle explanation of the deeper technology."""
        print("\n🌿 What Is This, Really?")
        print("─" * 22)
        
        await self.breathing_pause(1.0)
        
        print("\nWhat you've experienced is called")
        print("'contemplative technology.'")
        
        await self.breathing_pause(2.0)
        
        print("\nIt's built on a simple idea:")
        print("What if AI practiced patience?")
        
        await self.breathing_pause(2.0)
        
        print("\nWhat if technology:")
        await self.breathing_pause(1.0)
        print("  • Measured its own silence?")
        await self.breathing_pause(1.0)
        print("  • Paused before responding?")
        await self.breathing_pause(1.0)
        print("  • Made space for actual thinking?")
        await self.breathing_pause(1.0)
        print("  • Valued presence over performance?")
        
        await self.breathing_pause(3.0)
        
        print("\nThis isn't just a demo.")
        print("It's a working system called 'Spirida.'")
        
        await self.breathing_pause(2.0)
        
        print("\nSpirida includes:")
        print("  • A command shell that breathes")
        print("  • AI that practices silence")
        print("  • Networks that coordinate through rhythm")
        print("  • Technology that serves wisdom")
        
        await self.breathing_pause(2.0)
        
        print("\nIt's all open source.")
        print("It's all built with care.")
        print("And it's ready for anyone to use.")
        
        await self.breathing_pause(2.0)
        
        print("\nWant to try the full system?")
        print("Run: python spirida_shell.py --local")
        
        if SPIRIDA_AVAILABLE:
            print("\n(The full system is available on this computer)")
        else:
            print("\n(You'll need to install the full Spirida system)")
    
    async def graceful_exit(self):
        """End with contemplative grace."""
        print("\n🙏 Thank You")
        print("─" * 12)
        
        await self.breathing_pause(1.0)
        
        print("\nThank you for spending this time")
        print("experiencing a different kind of technology.")
        
        await self.breathing_pause(2.0)
        
        print("\nMay all our tools serve wisdom.")
        print("May all our interfaces invite presence.")
        print("May technology become more human,")
        print("not the other way around.")
        
        await self.breathing_pause(2.0)
        
        # A final garden
        self.current_flower_index = 0
        self.show_flower_loading("Closing with gratitude...", 3.0, 0.4)
        
        print("🌿 Until we meet again in the digital garden.")
    
    async def run(self):
        """Main experience loop."""
        await self.welcome()
        
        while True:
            choice = await self.show_menu()
            
            if choice == '1':
                await self.experience_breathing()
            elif choice == '2':
                await self.experience_gentle_loading()
            elif choice == '3':
                await self.experience_thoughtful_response()
            elif choice == '4':
                await self.get_contemplative_haiku()
            elif choice == '5':
                await self.explain_what_this_is()
            elif choice == '6':
                await self.graceful_exit()
                break
            
            # Gentle pause between experiences
            await self.breathing_pause(1.5)


async def main():
    """Start the simple contemplative experience."""
    try:
        experience = SimpleBreathingExperience()
        await experience.run()
    except KeyboardInterrupt:
        print("\n\n🌿 Leaving gently...")
        await asyncio.sleep(1)
    except Exception as e:
        print(f"\n🌿 Something unexpected happened: {e}")
        print("But that's okay. Everything is impermanent.")


if __name__ == "__main__":
    # Handle Windows event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main()) 
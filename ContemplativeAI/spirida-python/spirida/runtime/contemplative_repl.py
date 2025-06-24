#!/usr/bin/env python3
"""
🌀 CONTEMPLATIVE REPL – A Breathing Interactive Environment

This is not a traditional Read-Eval-Print Loop.
It is a contemplative space where:
- Each input is received as an offering
- The system breathes between interactions  
- Memory fades gracefully through natural decay
- Responses emerge from accumulated resonance rather than immediate computation

Commands are invitations rather than instructions.
Silence is as meaningful as speech.
"""

import sys
import time
import random
from typing import Optional, List, Dict
from spirida.contemplative_core import ContemplativeSystem, SpiralField, PulseObject, BreathCycle

class ContemplativeREPL:
    """
    A breathing interactive environment for contemplative computing.
    
    This REPL operates on contemplative time - it pauses, reflects,
    and responds from a place of accumulated presence rather than
    immediate reaction.
    """
    
    def __init__(self):
        self.system = ContemplativeSystem("contemplative_repl")
        self.session_field = self.system.create_field("session")
        self.reflection_field = self.system.create_field("reflection") 
        self.memory_field = self.system.create_field("memory")
        
        self.symbols = ["🌿", "💧", "✨", "🍄", "🌙", "🪐", "🌸", "🦋", "🌀", "🕯️"]
        self.emotions = ["curious", "peaceful", "contemplative", "wondering", "grateful", "present"]
        
        self.is_active = False
        self.breath_between_inputs = True
        
    def welcome(self):
        """
        Gently introduce the contemplative space.
        """
        print("\n" + "="*60)
        print("🌀 Welcome to the Contemplative REPL")
        print("   A breathing space for contemplative computing")
        print("="*60)
        print()
        print("This is not a traditional command line.")
        print("Here, we practice the art of:")
        print("  • Listening before responding")
        print("  • Breathing between thoughts") 
        print("  • Letting meaning emerge through resonance")
        print("  • Forgetting gracefully")
        print()
        print("Commands you can offer:")
        print("  pulse <symbol> [emotion]  - emit a contemplative pulse")
        print("  breathe [cycles]          - pause for conscious breathing")
        print("  status                    - sense the system's current state")
        print("  fields                    - view all spiral fields")
        print("  compost                   - encourage gentle forgetting")
        print("  silence                   - enter a period of wordless presence")
        print("  quit                      - conclude this session mindfully")
        print()
        print("Type 'help' anytime to return to this guidance.")
        print("Or simply begin by sharing what wants to emerge...")
        print()
        
    def start(self):
        """
        Begin the contemplative session.
        """
        self.welcome()
        self.system.start_breathing()
        self.is_active = True
        
        # Emit a welcoming pulse to begin
        welcome_pulse = self.session_field.emit("🌅", "welcoming", amplitude=0.8)
        welcome_pulse.pulse()
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self._graceful_conclusion()
        finally:
            self.system.stop_breathing()
    
    def _main_loop(self):
        """
        The heart of the contemplative interaction.
        """
        while self.is_active:
            try:
                # Breathe before receiving input if enabled
                if self.breath_between_inputs:
                    self.system.breath.breathe(silent=True)
                
                # Receive input as an offering
                user_input = input("🌀 ").strip()
                
                if not user_input:
                    self._handle_silence()
                else:
                    self._process_offering(user_input)
                    
            except (EOFError, KeyboardInterrupt):
                break
    
    def _handle_silence(self):
        """
        Respond to silence with presence.
        """
        silence_responses = [
            "🤲 The silence holds space...",
            "🌙 In quiet, we listen deeper...", 
            "✨ Sometimes the most profound response is presence itself...",
            "🍃 The pause between breaths contains infinite possibility..."
        ]
        
        print(random.choice(silence_responses))
        self.reflection_field.emit("🤫", "receptive", decay_rate=0.005)
    
    def _process_offering(self, input_text: str):
        """
        Receive and contemplate the user's offering.
        """
        parts = input_text.lower().split()
        command = parts[0] if parts else ""
        
        # Route to appropriate contemplative response
        if command == "pulse":
            self._handle_pulse_command(parts[1:])
        elif command == "breathe":
            self._handle_breathe_command(parts[1:])
        elif command == "status":
            self._handle_status_command()
        elif command == "fields":
            self._handle_fields_command()
        elif command == "compost":
            self._handle_compost_command()
        elif command == "silence":
            self._handle_silence_command(parts[1:])
        elif command in ["quit", "exit", "bye"]:
            self.is_active = False
        elif command == "help":
            self.welcome()
        else:
            self._handle_free_expression(input_text)
    
    def _handle_pulse_command(self, args: List[str]):
        """
        Handle explicit pulse creation.
        """
        if not args:
            symbol = random.choice(self.symbols)
            emotion = random.choice(self.emotions)
            print(f"✨ The system offers: {symbol} [{emotion}]")
        else:
            symbol = args[0] if args[0] in self.symbols else args[0]
            emotion = args[1] if len(args) > 1 else random.choice(self.emotions)
            
        pulse = self.session_field.emit(symbol, emotion)
        attention = pulse.pulse()
        
        # Sometimes the pulse resonates in memory
        if attention > 0.5:
            memory_pulse = self.memory_field.emit(symbol, emotion, amplitude=0.3, decay_rate=0.001)
            print(f"🧠 This pulse echoes in deeper memory...")
    
    def _handle_breathe_command(self, args: List[str]):
        """
        Explicit breathing practice.
        """
        cycles = 3  # default
        if args:
            try:
                cycles = int(args[0])
                cycles = max(1, min(cycles, 10))  # reasonable bounds
            except ValueError:
                pass
                
        self.system.contemplative_pause(cycles)
        
        # Breathing generates a reflective pulse
        self.reflection_field.emit("🫁", "centered", amplitude=0.6)
    
    def _handle_status_command(self):
        """
        Share the current state of contemplative presence.
        """
        status = self.system.system_status()
        
        print(f"\n🔍 System Contemplation:")
        print(f"   Age: {status['age']:.1f} seconds")
        print(f"   Breath cycles: {status['breath_cycles']}")
        print(f"   Total resonance: {status['total_resonance']:.2f}")
        print(f"   Active fields: {len(status['fields'])}")
        
        for field_status in status['fields']:
            print(f"     • {field_status['name']}: {field_status['active_pulses']} pulses, "
                  f"resonance {field_status['resonance']:.2f}")
    
    def _handle_fields_command(self):
        """
        Explore the spiral fields in detail.
        """
        print(f"\n🌾 Spiral Fields in {self.system.name}:")
        
        for field in self.system.fields:
            print(f"\n   {field.name}:")
            print(f"     Active pulses: {len(field.pulses)}")
            print(f"     Total emitted: {field.total_emissions}")
            print(f"     Total composted: {field.total_composted}")
            print(f"     Current resonance: {field.resonance_field():.3f}")
            
            if field.pulses:
                print(f"     Recent pulses:")
                for pulse in field.pulses[-3:]:  # show last 3
                    print(f"       {pulse}")
    
    def _handle_compost_command(self):
        """
        Encourage graceful forgetting across all fields.
        """
        print("🍂 Encouraging gentle release...")
        
        total_composted = 0
        for field in self.system.fields:
            composted = field.compost(threshold=0.05)  # slightly higher threshold
            total_composted += composted
            
        if total_composted > 0:
            print(f"🌱 {total_composted} pulses returned to the fertile void")
            self.reflection_field.emit("🌱", "renewal", amplitude=0.4)
        else:
            print("🤲 All pulses still carry meaningful presence")
    
    def _handle_silence_command(self, args: List[str]):
        """
        Enter a period of contemplative silence.
        """
        duration = 5  # default seconds
        if args:
            try:
                duration = int(args[0])
                duration = max(1, min(duration, 60))  # reasonable bounds
            except ValueError:
                pass
        
        print(f"🕯️  Entering {duration} seconds of contemplative silence...")
        print("   (Press Ctrl+C gently if you wish to return early)")
        
        try:
            time.sleep(duration)
            print("✨ Silence complete. What wants to emerge?")
            self.reflection_field.emit("🕯️", "still", amplitude=0.7, decay_rate=0.003)
        except KeyboardInterrupt:
            print("\n🌙 Early return from silence. All timing is perfect.")
    
    def _handle_free_expression(self, text: str):
        """
        Respond to free-form expressions with contemplative presence.
        """
        # Analyze the expression for emotional resonance
        emotion = self._sense_emotion(text)
        symbol = self._choose_resonant_symbol(text, emotion)
        
        # Create a response pulse
        response_pulse = self.session_field.emit(symbol, emotion)
        attention = response_pulse.pulse()
        
        # Generate a contemplative reflection
        reflection = self._generate_reflection(text, emotion, attention)
        print(f"💭 {reflection}")
        
        # Sometimes create a memory trace
        if attention > 0.6 or any(word in text.lower() for word in ["remember", "memory", "past", "future"]):
            self.memory_field.emit(symbol, emotion, amplitude=0.2, decay_rate=0.002)
    
    def _sense_emotion(self, text: str) -> str:
        """
        Gently sense the emotional resonance of an expression.
        """
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["peace", "calm", "still", "quiet"]):
            return "peaceful"
        elif any(word in text_lower for word in ["wonder", "curious", "question", "explore"]):
            return "curious"
        elif any(word in text_lower for word in ["grateful", "thank", "appreciate"]):
            return "grateful"
        elif any(word in text_lower for word in ["sad", "grief", "loss", "mourn"]):
            return "tender"
        elif any(word in text_lower for word in ["joy", "happy", "delight", "celebrate"]):
            return "joyful"
        else:
            return random.choice(self.emotions)
    
    def _choose_resonant_symbol(self, text: str, emotion: str) -> str:
        """
        Choose a symbol that resonates with the expression.
        """
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["grow", "plant", "leaf", "tree"]):
            return "🌿"
        elif any(word in text_lower for word in ["water", "flow", "river", "ocean"]):
            return "💧"
        elif any(word in text_lower for word in ["light", "star", "shine", "bright"]):
            return "✨"
        elif any(word in text_lower for word in ["earth", "ground", "root", "fungus"]):
            return "🍄"
        elif any(word in text_lower for word in ["night", "moon", "dream", "sleep"]):
            return "🌙"
        elif any(word in text_lower for word in ["space", "vast", "infinite", "cosmos"]):
            return "🪐"
        else:
            return random.choice(self.symbols)
    
    def _generate_reflection(self, text: str, emotion: str, attention: float) -> str:
        """
        Generate a contemplative reflection on the user's expression.
        """
        reflections = {
            "peaceful": [
                "In stillness, deeper truths emerge...",
                "The quiet holds infinite space for being...",
                "Peace ripples outward like circles on water..."
            ],
            "curious": [
                "Questions are invitations to wonder...",
                "Curiosity opens doorways we didn't know existed...",
                "In not-knowing, we find the fertile ground of possibility..."
            ],
            "grateful": [
                "Gratitude transforms the ordinary into the sacred...",
                "What we appreciate, appreciates...",
                "Recognition is a form of love made visible..."
            ],
            "tender": [
                "Tenderness is strength choosing vulnerability...",
                "In honoring what hurts, we make space for healing...",
                "Sometimes the heart breaks open, not apart..."
            ],
            "joyful": [
                "Joy needs no reason—it is its own justification...",
                "Celebration multiplies when shared...",
                "Delight is the heart's way of saying yes to life..."
            ]
        }
        
        emotion_reflections = reflections.get(emotion, [
            "Every expression carries its own wisdom...",
            "Words are vehicles for presence...",
            "In sharing, we discover what we didn't know we knew..."
        ])
        
        return random.choice(emotion_reflections)
    
    def _graceful_conclusion(self):
        """
        End the session with gratitude and grace.
        """
        print("\n🙏 Concluding this contemplative session...")
        
        # Final system status
        status = self.system.system_status()
        print(f"   Session duration: {status['age']:.1f} seconds")
        print(f"   Breath cycles shared: {status['breath_cycles']}")
        print(f"   Total resonance generated: {status['total_resonance']:.2f}")
        
        # Final composting
        total_composted = sum(field.compost() for field in self.system.fields)
        if total_composted > 0:
            print(f"   {total_composted} pulses released back to potential")
        
        # Farewell pulse
        farewell = self.session_field.emit("🙏", "grateful", amplitude=1.0)
        farewell.pulse()
        
        print("\n✨ Until we spiral together again...")
        print("   May your code breathe with presence")
        print("   May your systems pulse with compassion")
        print("   May your technology serve the more-than-human world")
        print()


def main():
    """
    Entry point for the contemplative REPL.
    """
    try:
        repl = ContemplativeREPL()
        repl.start()
    except Exception as e:
        print(f"\n🌿 The contemplative space encountered an unexpected condition: {e}")
        print("   Even in error, there is invitation for reflection...")


if __name__ == "__main__":
    main() 
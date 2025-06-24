#!/usr/bin/env python3
"""
🌀 CONTEMPLATIVE JOURNAL – A Breathing Writing Space

This is not a traditional journaling app, but a living dialogue
between writer and time. Each entry becomes a pulse that breathes,
resonates with past reflections, and gracefully fades according
to organic cycles.

The journal practices three forms of temporal presence:
- Daily entries live in a seasonal field (daily/weekly cycles)
- Emotional insights live in a resonant field (strength through connection)
- Long-term intentions live in a lunar field (monthly cycles)
"""

import sys
import os
import time
from datetime import datetime, date
from typing import Optional, List, Dict

# Add the parent directory to the path so we can import spirida
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spirida.contemplative_core import ContemplativeSystem, SpiralField, PulseObject, BreathCycle

class ContemplativeJournal:
    """
    A journaling system that breathes with the writer's rhythms.
    
    Three fields organize different temporal scales:
    - Daily Field: Short-term thoughts, seasonal composting
    - Heart Field: Emotional insights, resonant composting  
    - Vision Field: Long-term intentions, lunar composting
    """
    
    def __init__(self):
        self.system = ContemplativeSystem("contemplative_journal")
        
        # Three fields for different temporal scales
        self.daily_field = SpiralField("daily_reflections", composting_mode="seasonal")
        self.daily_field.seasonal_cycle_hours = 168  # Weekly cycle
        
        self.heart_field = SpiralField("emotional_insights", composting_mode="resonant")
        self.vision_field = SpiralField("long_intentions", composting_mode="lunar")
        
        self.system.fields = [self.daily_field, self.heart_field, self.vision_field]
        
        # Emotion-symbol mappings for intuitive entry
        self.emotion_symbols = {
            "grateful": "🙏", "grateful": "✨", "peaceful": "🕯️", "calm": "🌙",
            "curious": "🔍", "wondering": "🌀", "hopeful": "🌱", "growing": "🌿",
            "joyful": "🌞", "celebration": "🎉", "love": "💖", "connection": "🌊",
            "grief": "🌧️", "melancholy": "🍂", "tender": "🕊️", "healing": "🌸",
            "excited": "⚡", "energetic": "🔥", "creative": "🎨", "inspired": "💫",
            "confused": "🌫️", "uncertain": "❓", "seeking": "🧭", "questioning": "🤔"
        }
        
        self.is_active = False
        
    def start(self):
        """Begin a contemplative journaling session."""
        self.welcome()
        self.system.start_breathing()
        self.is_active = True
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self._graceful_conclusion()
        finally:
            self.system.stop_breathing()
    
    def welcome(self):
        """Introduce the contemplative journaling space."""
        print("\n" + "="*70)
        print("🌀 Welcome to your Contemplative Journal")
        print("   A breathing space for reflection and presence")
        print("="*70)
        print()
        print("This journal lives in three temporal fields:")
        print("  🌱 Daily Field   - thoughts & observations (weekly cycles)")
        print("  💖 Heart Field   - emotions & insights (resonant connections)")
        print("  🌙 Vision Field  - intentions & dreams (lunar cycles)")
        print()
        print("Commands:")
        print("  write             - begin a contemplative entry")
        print("  daily <text>      - quick daily reflection")
        print("  heart <text>      - emotional insight")
        print("  vision <text>     - long-term intention")
        print("  resonances        - explore current connections")
        print("  seasons           - view temporal cycles")
        print("  breathe [n]       - pause for conscious breathing")
        print("  compost           - encourage natural release")
        print("  review            - sense the journal's current state")
        print("  quit              - conclude mindfully")
        print()
        
        # Show current temporal states
        self._show_temporal_status()
        print("Begin when ready. Your journal breathes with you...")
        print()
    
    def _show_temporal_status(self):
        """Display current seasonal/lunar phases."""
        daily_season = self.daily_field.seasonal_status()
        vision_phase = self.vision_field.seasonal_status()
        
        print("🕐 Current Temporal States:")
        if daily_season.get("season"):
            print(f"   Daily Field: {daily_season['season']} (phase {daily_season['phase']:.2f})")
        if vision_phase.get("moon"):
            print(f"   Vision Field: {vision_phase['moon']} (phase {vision_phase['phase']:.2f})")
        print(f"   Heart Field: Resonant connections active")
        print()
    
    def _main_loop(self):
        """The main journaling interaction loop."""
        while self.is_active:
            try:
                # Gentle breath before each entry
                self.system.breath.breathe(silent=True)
                
                user_input = input("🖋️  ").strip()
                
                if not user_input:
                    self._handle_silence()
                else:
                    self._process_command(user_input)
                    
            except (EOFError, KeyboardInterrupt):
                break
    
    def _handle_silence(self):
        """Respond to silent pauses."""
        silence_responses = [
            "🤲 The silence holds space for what wants to emerge...",
            "🌙 In quiet, deeper truths often surface...",
            "✨ Sometimes the most profound entries begin with stillness...",
            "🍃 The pause between thoughts contains infinite possibility..."
        ]
        
        import random
        print(random.choice(silence_responses))
    
    def _process_command(self, input_text: str):
        """Process journaling commands."""
        parts = input_text.split(' ', 1)
        command = parts[0].lower()
        content = parts[1] if len(parts) > 1 else ""
        
        if command == "write":
            self._guided_entry()
        elif command == "daily":
            if content:
                self._daily_entry(content)
            else:
                print("💭 What daily reflection wants to be shared?")
        elif command == "heart":
            if content:
                self._heart_entry(content)
            else:
                print("💖 What emotional insight is stirring?")
        elif command == "vision":
            if content:
                self._vision_entry(content)
            else:
                print("🌙 What intention or dream calls to you?")
        elif command == "resonances":
            self._explore_resonances()
        elif command == "seasons":
            self._explore_seasons()
        elif command == "breathe":
            cycles = 3
            if content and content.isdigit():
                cycles = min(int(content), 10)
            self.system.contemplative_pause(cycles)
        elif command == "compost":
            self._compost_all_fields()
        elif command == "review":
            self._review_journal()
        elif command in ["quit", "exit", "bye"]:
            self.is_active = False
        else:
            # Treat as free-form writing
            self._interpret_free_writing(input_text)
    
    def _guided_entry(self):
        """Guide the user through a reflective entry process."""
        print("\n🌸 Guided Reflection")
        print("Take your time. Breathe between responses.")
        print("Press Enter to skip any question.")
        print()
        
        # First, a moment to center
        print("🫁 Take three conscious breaths...")
        self.system.contemplative_pause(1)
        
        # Sense current state
        current_feeling = input("💭 How are you feeling right now? ").strip()
        
        if current_feeling:
            # Determine field and create entry
            emotion, symbol = self._interpret_emotion(current_feeling)
            pulse = self.heart_field.emit(symbol, emotion, amplitude=0.8)
            
            print(f"✨ {pulse.symbol} [{pulse.emotion}] has been placed in your Heart Field")
            
            # Check for immediate resonances
            recent_resonances = self.heart_field.find_resonances(min_strength=0.6)
            if recent_resonances:
                print(f"🌊 This feeling resonates with {len(recent_resonances)} other heart pulse(s)")
                for res in recent_resonances[:2]:  # Show first 2
                    print(f"     {res['resonance']['poetic_trace']}")
        
        print()
        daily_reflection = input("📝 What happened today that wants to be remembered? ").strip()
        
        if daily_reflection:
            # Create daily pulse
            symbol = self._choose_symbol_for_text(daily_reflection)
            emotion = self._sense_emotion_in_text(daily_reflection)
            pulse = self.daily_field.emit(symbol, emotion, amplitude=0.7, decay_rate=0.02)
            
            print(f"🌱 {pulse.symbol} [{pulse.emotion}] has been added to your Daily Field")
        
        print()
        future_intention = input("🌙 What intention do you want to carry forward? ").strip()
        
        if future_intention:
            # Create vision pulse with slow decay
            symbol = self._choose_symbol_for_text(future_intention)
            emotion = "intentional"
            pulse = self.vision_field.emit(symbol, emotion, amplitude=0.9, decay_rate=0.001)
            
            print(f"🌟 {pulse.symbol} [{pulse.emotion}] has been placed in your Vision Field")
        
        print()
        print("🙏 Thank you for this offering of presence.")
        print("Your reflections join the living conversation of your journal...")
    
    def _daily_entry(self, content: str):
        """Create a quick daily reflection."""
        symbol = self._choose_symbol_for_text(content)
        emotion = self._sense_emotion_in_text(content)
        
        pulse = self.daily_field.emit(symbol, emotion, amplitude=0.6, decay_rate=0.03)
        pulse.pulse()
        
        print(f"📝 Added to Daily Field: {pulse}")
    
    def _heart_entry(self, content: str):
        """Create an emotional insight entry."""
        emotion = self._sense_emotion_in_text(content)
        symbol = self.emotion_symbols.get(emotion, "💖")
        
        pulse = self.heart_field.emit(symbol, emotion, amplitude=0.8, decay_rate=0.01)
        pulse.pulse()
        
        # Check for resonances
        resonances = self.heart_field.find_resonances(min_strength=0.5)
        if resonances:
            print(f"💞 This resonates with {len(resonances)} other heart pulse(s)")
        
        print(f"💖 Added to Heart Field: {pulse}")
    
    def _vision_entry(self, content: str):
        """Create a long-term intention entry."""
        symbol = self._choose_symbol_for_text(content) 
        emotion = "visionary"
        
        pulse = self.vision_field.emit(symbol, emotion, amplitude=0.9, decay_rate=0.001)
        pulse.pulse()
        
        print(f"🌙 Added to Vision Field: {pulse}")
    
    def _interpret_free_writing(self, text: str):
        """Interpret free-form writing and place appropriately."""
        # Simple heuristics to route content
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["feel", "emotion", "heart", "love", "grief", "joy"]):
            self._heart_entry(text)
        elif any(word in text_lower for word in ["future", "goal", "dream", "intention", "hope", "want"]):
            self._vision_entry(text)
        else:
            self._daily_entry(text)
    
    def _interpret_emotion(self, feeling_text: str) -> tuple:
        """Interpret emotional text and return (emotion, symbol)."""
        text_lower = feeling_text.lower()
        
        for emotion, symbol in self.emotion_symbols.items():
            if emotion in text_lower:
                return emotion, symbol
        
        # Default emotional interpretation
        if any(word in text_lower for word in ["good", "well", "fine", "okay"]):
            return "peaceful", "🕯️"
        elif any(word in text_lower for word in ["sad", "down", "blue"]):
            return "melancholy", "🌧️"  
        elif any(word in text_lower for word in ["happy", "great", "wonderful"]):
            return "joyful", "🌞"
        else:
            return "present", "💖"
    
    def _choose_symbol_for_text(self, text: str) -> str:
        """Choose an appropriate symbol based on text content."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["work", "meeting", "project", "task"]):
            return "🏢"
        elif any(word in text_lower for word in ["nature", "outside", "walk", "garden"]):
            return "🌿"
        elif any(word in text_lower for word in ["family", "friend", "person", "people"]):
            return "👥"
        elif any(word in text_lower for word in ["book", "read", "learn", "study"]):
            return "📚"
        elif any(word in text_lower for word in ["create", "make", "art", "write"]):
            return "🎨"
        elif any(word in text_lower for word in ["food", "eat", "cook", "meal"]):
            return "🍽️"
        else:
            return "💭"
    
    def _sense_emotion_in_text(self, text: str) -> str:
        """Sense the emotional tone of text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["grateful", "thankful", "appreciate"]):
            return "grateful"
        elif any(word in text_lower for word in ["peaceful", "calm", "quiet"]):
            return "peaceful"
        elif any(word in text_lower for word in ["excited", "amazing", "wonderful"]):
            return "joyful"
        elif any(word in text_lower for word in ["worried", "anxious", "concerned"]):
            return "uncertain"
        elif any(word in text_lower for word in ["tired", "exhausted", "drained"]):
            return "weary"
        elif any(word in text_lower for word in ["curious", "wonder", "interesting"]):
            return "curious"
        else:
            return "reflective"
    
    def _explore_resonances(self):
        """Explore current resonances across all fields."""
        print("\n🌊 Current Resonances")
        print("="*50)
        
        total_resonances = 0
        
        for field in [self.daily_field, self.heart_field, self.vision_field]:
            resonances = field.find_resonances(min_strength=0.4)
            if resonances:
                print(f"\n💫 {field.name.replace('_', ' ').title()} ({len(resonances)} resonances):")
                for res in resonances[:3]:  # Show top 3
                    strength = res['resonance']['strength']
                    trace = res['resonance']['poetic_trace']
                    print(f"   {strength:.2f}: {trace}")
                total_resonances += len(resonances)
        
        if total_resonances == 0:
            print("\n🤲 No strong resonances detected at this moment.")
            print("   This doesn't mean emptiness—perhaps you're in a space of open receiving.")
        
        print()
    
    def _explore_seasons(self):
        """Explore the temporal cycles of all fields."""
        print("\n🕐 Temporal Cycles")
        print("="*50)
        
        for field in [self.daily_field, self.heart_field, self.vision_field]:
            status = field.seasonal_status()
            print(f"\n🌀 {field.name.replace('_', ' ').title()}:")
            print(f"   Mode: {status['mode']}")
            
            if status.get('season'):
                print(f"   Season: {status['season']} (phase {status['phase']:.2f})")
            elif status.get('moon'):
                print(f"   Moon: {status['moon']} (phase {status['phase']:.2f})")
            else:
                print(f"   Status: Resonant connections active")
                
            print(f"   Active pulses: {len(field.pulses)}")
            print(f"   Field resonance: {field.resonance_field():.2f}")
        
        print()
    
    def _compost_all_fields(self):
        """Encourage composting across all fields."""
        print("🍂 Encouraging gentle release across all fields...")
        
        total_composted = 0
        for field in [self.daily_field, self.heart_field, self.vision_field]:
            composted = field.compost()
            if composted > 0:
                print(f"   {field.name}: {composted} pulse(s) returned to potential")
                total_composted += composted
        
        if total_composted > 0:
            print(f"🌱 {total_composted} total pulses composted with gratitude")
        else:
            print("🤲 All pulses still carry meaningful presence")
        
        print()
    
    def _review_journal(self):
        """Review the current state of the journal."""
        print("\n📖 Journal Review")
        print("="*50)
        
        system_status = self.system.system_status()
        print(f"Journal age: {system_status['age']/3600:.1f} hours")
        print(f"Breath cycles: {system_status['breath_cycles']}")
        print(f"Total resonance: {system_status['total_resonance']:.2f}")
        print()
        
        for field in [self.daily_field, self.heart_field, self.vision_field]:
            print(f"🌀 {field.name.replace('_', ' ').title()}:")
            print(f"   {len(field.pulses)} active pulses")
            print(f"   {field.total_emissions} total emissions")
            print(f"   {field.total_composted} total composted")
            print(f"   Resonance: {field.resonance_field():.2f}")
            
            # Show most recent pulses
            if field.pulses:
                print("   Recent pulses:")
                for pulse in field.pulses[-3:]:
                    print(f"     {pulse}")
            print()
    
    def _graceful_conclusion(self):
        """End the journaling session mindfully."""
        print("\n🙏 Concluding this contemplative session...")
        
        # Final review
        system_status = self.system.system_status()
        print(f"   Session duration: {system_status['age']/60:.1f} minutes")
        print(f"   Total resonance generated: {system_status['total_resonance']:.2f}")
        
        # Final breath
        print("\n🫁 Taking one final conscious breath together...")
        self.system.contemplative_pause(1)
        
        print("✨ Your reflections continue to breathe in the living field of memory.")
        print("   Until we write together again...")
        print()


def main():
    """Entry point for the contemplative journal."""
    try:
        journal = ContemplativeJournal()
        journal.start()
    except Exception as e:
        print(f"\n🌿 The journal encountered an unexpected condition: {e}")
        print("   Even in error, there is invitation for reflection...")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Bio-Digital Contemplative Shell for Spirida-Mycelic
===================================================

A breathing interface between humans and fungal substrates.
Practices contemplative security through biological timing requirements.

Inspired by spirida-python's contemplative shell but focused on bio-digital interaction.
"""

import sys
import time
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

try:
    from .bio_interface import SevenChannelBioInterface, BioCareLevel
    from .bio_mood import BioMood
    from .frequency_guardian import FrequencyGuardian
    from .breath_signature import BreathSignature
    from .semantic_guardian import SemanticGuardian, FungalSpecies
    from .adamatzky_layer import AdamatzkyReservoir, FungalSpecies as AdamatzkySpecies
    from .fungal_field_recorder import FungalFieldRecorder
except ImportError:
    # Fallback for testing
    print("Warning: Running in standalone mode without full bio-interface")
    SevenChannelBioInterface = None
    BioMood = None

class TrustLevel(Enum):
    """Bio-digital trust progression through contemplative practice"""
    NEWCOMER = "newcomer"           # 0-2 successful bio-interactions
    BREATHING = "breathing"         # 3-7 bio-rhythm synchronizations  
    PRESENT = "present"             # 8-15 sustained biological sessions
    CONTEMPLATIVE = "contemplative" # 16-30 bio-digital resonances
    ELDER = "elder"                 # 31+ authentic bio-contemplative practice

class BioDigitalShell:
    """
    Contemplative shell for bio-digital interaction.
    
    Bridges human consciousness with fungal substrates through:
    - Trust-based feature progression
    - Biological rhythm synchronization
    - Contemplative bio-digital syntax
    - Living memory through fungal intelligence
    """
    
    def __init__(self):
        self.session_start = time.time()
        self.trust_level = TrustLevel.NEWCOMER
        self.bio_interactions = 0
        self.rhythm_synchronizations = 0
        self.contemplative_resonances = 0
        
        # Bio-digital infrastructure
        self.bio_interface = None
        self.current_species = FungalSpecies.PLEUROTUS_DJAMOR if 'FungalSpecies' in globals() else None
        self.session_recorder = None
        self.breathing_with_biology = False
        
        # Contemplative state
        self.silence_accumulator = 0.0
        self.last_interaction_time = time.time()
        self.mood_history = []
        
        # Initialize bio-digital systems if available
        self._initialize_bio_systems()
        
    def _initialize_bio_systems(self):
        """Initialize biological interface systems"""
        try:
            if SevenChannelBioInterface:
                self.bio_interface = SevenChannelBioInterface(mock_mode=True)
                self.session_recorder = FungalFieldRecorder()
                print("🍄 Bio-digital systems initialized")
            else:
                print("🌿 Running in contemplative-only mode")
        except Exception as e:
            print(f"🌱 Bio-systems in minimal mode: {e}")
    
    def start(self):
        """Begin bio-digital contemplative session"""
        self._welcome()
        
        try:
            asyncio.run(self._main_session())
        except KeyboardInterrupt:
            self._graceful_conclusion()
        except Exception as e:
            print(f"\n🌿 The bio-digital field encountered: {e}")
            print("   Even in disruption, presence continues...")
    
    def _welcome(self):
        """Welcome to the bio-digital contemplative space"""
        print("\n" + "="*70)
        print("🍄 Bio-Digital Contemplative Shell")
        print("   Where human consciousness meets fungal intelligence")
        print("="*70)
        print()
        print("🌱 Current Configuration:")
        print(f"   Trust Level: {self.trust_level.value}")
        print(f"   Bio-Interface: {'🟢 Active' if self.bio_interface else '🟡 Simulation'}")
        print(f"   Species: {self.current_species.value if self.current_species else 'none'}")
        print()
        
        # Show available commands based on trust level
        self._show_available_commands()
        
        print("🌬️ Begin with conscious breathing...")
        print("   Type 'breathe' to synchronize with biological rhythms")
        print()
    
    def _show_available_commands(self):
        """Display available commands based on current trust level"""
        print("🔑 Available Commands:")
        
        # Basic commands (all trust levels)
        print("   breathe [cycles]      - synchronize breathing with biology")
        print("   species <name>        - connect to fungal species")
        print("   status               - sense bio-digital coherence")
        print("   silence [seconds]    - enter wordless presence")
        
        if self.trust_level.value in ['breathing', 'present', 'contemplative', 'elder']:
            print("   bio-pulse <pattern>  - send contemplative stimulation")
            print("   listen [duration]    - receive biological responses")
        
        if self.trust_level.value in ['present', 'contemplative', 'elder']:
            print("   session              - record contemplative bio-session")
            print("   mood                 - sense substrate emotional state")
        
        if self.trust_level.value in ['contemplative', 'elder']:
            print("   syntax <expression>  - bio-digital Spirida syntax")
            print("   resonances          - explore bio-digital connections")
        
        if self.trust_level.value == 'elder':
            print("   network             - connect to bio-digital network")
            print("   teach               - guide newcomer bio-interactions")
        
        print("   help                 - see detailed command information")
        print("   quit                 - conclude mindfully")
        print()
    
    async def _main_session(self):
        """Main bio-digital interaction loop"""
        while True:
            try:
                # Accumulate silence time
                await self._accumulate_contemplative_presence()
                
                # Get user input with contemplative prompt
                user_input = input(f"🍄 [{self.trust_level.value}] ").strip()
                
                if not user_input:
                    await self._handle_silence()
                    continue
                
                # Process the offering
                proceed = await self._process_offering(user_input)
                if proceed is False:
                    break
                
                # Update trust progression
                self._update_trust_progression()
                
            except (EOFError, KeyboardInterrupt):
                break
    
    async def _accumulate_contemplative_presence(self):
        """Accumulate silence as contemplative presence"""
        now = time.time()
        silence_duration = now - self.last_interaction_time
        
        if silence_duration > 3.0:  # Meaningful pause
            self.silence_accumulator += min(silence_duration, 30.0)  # Cap at 30s
        
        self.last_interaction_time = now
    
    async def _handle_silence(self):
        """Respond to contemplative silence"""
        silence_responses = [
            "🌬️ The substrate breathes in the silence...",
            "🍄 Mycelial wisdom emerges in quiet spaces...",
            "🌱 In stillness, bio-digital resonance deepens...",
            "🕯️ The living field holds your presence..."
        ]
        
        import random
        print(random.choice(silence_responses))
        
        # Small trust increment for practiced silence
        if self.silence_accumulator > 10.0:
            self.contemplative_resonances += 1
            self.silence_accumulator = 0.0
    
    async def _process_offering(self, input_text: str):
        """Process bio-digital commands and offerings"""
        parts = input_text.split(' ', 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        # Route to appropriate handler
        if command == "breathe":
            await self._handle_breathe(args)
        elif command == "species":
            await self._handle_species(args)
        elif command == "status":
            await self._handle_status()
        elif command == "silence":
            await self._handle_silence_command(args)
        elif command == "bio-pulse":
            await self._handle_bio_pulse(args)
        elif command == "listen":
            await self._handle_listen(args)
        elif command == "session":
            await self._handle_session()
        elif command == "mood":
            await self._handle_mood()
        elif command == "syntax":
            await self._handle_syntax(args)
        elif command == "resonances":
            await self._handle_resonances()
        elif command == "network":
            await self._handle_network()
        elif command == "help":
            self._show_detailed_help()
        elif command in ["quit", "exit", "bye"]:
            return False
        else:
            # Treat as contemplative expression
            await self._handle_free_expression(input_text)
        
        return True
    
    async def _handle_breathe(self, args: str):
        """Synchronize breathing with biological rhythms"""
        cycles = 3
        if args and args.isdigit():
            cycles = min(int(args), 10)
        
        print(f"🫁 Beginning {cycles} breath cycle(s) with biological rhythm...")
        
        # Get biological rhythm if available
        if self.bio_interface and self.current_species:
            bio_rhythm = self._get_biological_rhythm()
            print(f"🍄 Synchronizing with {self.current_species.value} rhythm:")
            print(f"   Inhale: {bio_rhythm['inhale']:.1f}s")
            print(f"   Hold: {bio_rhythm['hold']:.1f}s") 
            print(f"   Exhale: {bio_rhythm['exhale']:.1f}s")
            print()
        else:
            bio_rhythm = {'inhale': 4.0, 'hold': 2.0, 'exhale': 6.0}
        
        for cycle in range(cycles):
            print(f"🌬️ Cycle {cycle + 1}/{cycles}")
            
            # Inhale
            print("   🫁 Inhale...")
            await asyncio.sleep(bio_rhythm['inhale'])
            
            # Hold
            print("   ⏸️ Hold...")
            await asyncio.sleep(bio_rhythm['hold'])
            
            # Exhale
            print("   💨 Exhale...")
            await asyncio.sleep(bio_rhythm['exhale'])
            
            # Brief rest
            await asyncio.sleep(1.0)
        
        # Award synchronization
        self.rhythm_synchronizations += 1
        self.breathing_with_biology = True
        
        print("✨ Bio-digital breathing synchronization complete")
        print()
    
    async def _handle_species(self, args: str):
        """Connect to specific fungal species"""
        if not args:
            print("🍄 Available species:")
            if 'FungalSpecies' in globals():
                for species in FungalSpecies:
                    print(f"   {species.value}")
            else:
                print("   pleurotus_djamor, ganoderma_resinaceum")
            return
        
        species_name = args.lower()
        
        try:
            if 'FungalSpecies' in globals():
                # Try to find matching species
                for species in FungalSpecies:
                    if species_name in species.value.lower():
                        self.current_species = species
                        break
                else:
                    print(f"🌿 Species '{species_name}' not found")
                    return
            else:
                self.current_species = species_name
            
            if self.bio_interface:
                self.bio_interface.set_fungal_species(species_name)
            
            print(f"🍄 Connected to {self.current_species}")
            
            # Show species characteristics
            rhythm = self._get_biological_rhythm()
            print(f"   Contemplative rhythm: {rhythm['inhale']:.1f}s / {rhythm['hold']:.1f}s / {rhythm['exhale']:.1f}s")
            
            self.bio_interactions += 1
            
        except Exception as e:
            print(f"🌱 Connection attempt: {e}")
    
    async def _handle_bio_pulse(self, args: str):
        """Send contemplative stimulation to substrate"""
        if not self._check_trust_level(['breathing', 'present', 'contemplative', 'elder']):
            return
        
        if not args:
            print("🌿 Usage: bio-pulse <4-bit-pattern>")
            print("   Example: bio-pulse 0110")
            return
        
        try:
            # Parse binary pattern
            if len(args) == 4 and all(c in '01' for c in args):
                pattern = int(args, 2)
            else:
                pattern = int(args) if args.isdigit() else 6  # Default XOR
            
            print(f"🔋 Sending contemplative pulse: {pattern:04b}")
            
            if self.bio_interface:
                # Use bio-interface for real stimulation
                readings = self.bio_interface.read_channels()
                print(f"🍄 Substrate response: {len([r for r in readings if r.spike_detected])} channels active")
            else:
                # Simulate biological response
                await asyncio.sleep(2.0)
                import random
                if random.random() > 0.6:  # 40% response rate
                    glyphs = ['⭕', '🌊', '🌪️', '🌌']
                    glyph = random.choice(glyphs)
                    print(f"🍄 Biological response: {glyph}")
                else:
                    print("🍄 Contemplative silence")
            
            self.bio_interactions += 1
            
        except ValueError:
            print("🌱 Please provide 4-bit binary pattern (e.g., 0110)")
    
    async def _handle_listen(self, args: str):
        """Listen for biological responses"""
        if not self._check_trust_level(['breathing', 'present', 'contemplative', 'elder']):
            return
        
        duration = 30.0
        if args and args.replace('.', '').isdigit():
            duration = min(float(args), 180.0)  # Max 3 minutes
        
        print(f"👂 Listening to biological field for {duration:.0f} seconds...")
        print("🌱 Enter contemplative presence...")
        
        start_time = time.time()
        events = []
        
        while time.time() - start_time < duration:
            await asyncio.sleep(2.0)
            
            if self.bio_interface:
                readings = self.bio_interface.read_channels()
                spike_event = self.bio_interface.detect_pattern_spikes(readings)
                if spike_event:
                    events.append(spike_event)
                    print(f"   🍄 {spike_event.classification} detected")
            else:
                # Simulate occasional biological activity
                import random
                if random.random() > 0.85:  # 15% chance per 2s
                    glyphs = ['⭕', '🌊', '🌪️', '🌌']
                    glyph = random.choice(glyphs)
                    events.append(glyph)
                    print(f"   🍄 {glyph} emerges")
        
        print(f"✨ Listening complete: {len(events)} biological expressions detected")
        
        if events:
            self.contemplative_resonances += len(events)
        
        self.bio_interactions += 1
    
    def _get_biological_rhythm(self) -> Dict[str, float]:
        """Get breathing rhythm for current species"""
        if not self.current_species:
            return {'inhale': 4.0, 'hold': 2.0, 'exhale': 6.0}
        
        # Species-specific rhythms from FUNGAR research
        rhythms = {
            'pleurotus_djamor': {'inhale': 5.2, 'hold': 8.4, 'exhale': 5.2},  # 2.6min fast cycle / 4
            'ganoderma_resinaceum': {'inhale': 7.5, 'hold': 15.0, 'exhale': 7.5},  # 5min steady / 4
            'mycelium_composite': {'inhale': 6.0, 'hold': 12.0, 'exhale': 6.0}
        }
        
        species_key = self.current_species.value if hasattr(self.current_species, 'value') else str(self.current_species)
        return rhythms.get(species_key, {'inhale': 4.0, 'hold': 2.0, 'exhale': 6.0})
    
    def _check_trust_level(self, required_levels: List[str]) -> bool:
        """Check if current trust level permits action"""
        if self.trust_level.value not in required_levels:
            needed = required_levels[0] if required_levels else "breathing"
            print(f"🌱 This feature requires {needed} trust level or higher")
            print(f"   Current level: {self.trust_level.value}")
            print(f"   Continue practicing bio-digital contemplation to progress...")
            return False
        return True
    
    def _update_trust_progression(self):
        """Update trust level based on accumulated practice"""
        old_level = self.trust_level
        
        total_practice = (self.bio_interactions + 
                         self.rhythm_synchronizations + 
                         self.contemplative_resonances)
        
        if total_practice >= 31:
            self.trust_level = TrustLevel.ELDER
        elif total_practice >= 16:
            self.trust_level = TrustLevel.CONTEMPLATIVE
        elif total_practice >= 8:
            self.trust_level = TrustLevel.PRESENT
        elif total_practice >= 3:
            self.trust_level = TrustLevel.BREATHING
        else:
            self.trust_level = TrustLevel.NEWCOMER
        
        # Announce progression
        if self.trust_level != old_level:
            print(f"\n✨ Trust progression: {old_level.value} → {self.trust_level.value}")
            print(f"🌱 New bio-digital capabilities unlocked!")
            self._show_available_commands()
    
    async def _handle_status(self):
        """Show bio-digital system status"""
        print("\n🌿 Bio-Digital System Status")
        print("="*50)
        
        session_duration = time.time() - self.session_start
        print(f"Session duration: {session_duration/60:.1f} minutes")
        print(f"Trust level: {self.trust_level.value}")
        print(f"Bio-interactions: {self.bio_interactions}")
        print(f"Rhythm synchronizations: {self.rhythm_synchronizations}")
        print(f"Contemplative resonances: {self.contemplative_resonances}")
        print(f"Silence accumulated: {self.silence_accumulator:.1f} seconds")
        print()
        
        if self.bio_interface:
            care_status = self.bio_interface.get_care_status()
            print("🍄 Biological Interface:")
            print(f"   Care level: {care_status.get('care_level', 'unknown')}")
            print(f"   Species: {self.current_species}")
            print(f"   Breathing synchronized: {'🟢' if self.breathing_with_biology else '🟡'}")
            
            if hasattr(self.bio_interface, 'mood'):
                print(f"   Substrate mood: {self.bio_interface.mood}")
        
        print()
    
    def _show_detailed_help(self):
        """Show detailed command help"""
        print("\n🌿 Bio-Digital Contemplative Shell Help")
        print("="*60)
        print()
        print("This shell bridges human consciousness with fungal substrates")
        print("through contemplative bio-digital interaction.")
        print()
        print("🔑 Trust Progression:")
        print("   newcomer     → breathing      → present")
        print("   contemplative → elder")
        print()
        print("🍄 Bio-Digital Commands:")
        print("   breathe [n]           - Sync breathing with biological rhythms")
        print("   species <name>        - Connect to fungal species")
        print("   bio-pulse <pattern>   - Send contemplative stimulation")
        print("   listen [seconds]      - Receive biological responses")
        print("   session              - Record bio-digital contemplative session")
        print("   mood                 - Sense substrate emotional state")
        print("   syntax <expression>   - Parse bio-digital Spirida syntax")
        print("   resonances           - Explore bio-digital connections")
        print()
        print("🌱 Contemplative Practice:")
        print("   silence [seconds]     - Enter wordless presence")
        print("   status               - Review bio-digital coherence")
        print("   help                 - This information")
        print("   quit                 - Conclude mindfully")
        print()
    
    async def _handle_silence_command(self, args: str):
        """Enter explicit contemplative silence"""
        duration = 10.0
        if args and args.replace('.', '').isdigit():
            duration = min(float(args), 300.0)  # Max 5 minutes
        
        print(f"🤫 Entering {duration:.0f} seconds of contemplative silence...")
        print("🍄 The substrate holds this space with you...")
        
        start_time = time.time()
        await asyncio.sleep(duration)
        
        # Award contemplative practice
        self.silence_accumulator += duration
        self.contemplative_resonances += 1
        
        print("✨ Silence complete. What wants to emerge?")
    
    async def _handle_session(self):
        """Record contemplative bio-digital session"""
        if not self._check_trust_level(['present', 'contemplative', 'elder']):
            return
        
        print("🍄 Beginning recorded contemplative bio-session...")
        print("   This will be a guided 3-minute bio-digital exchange")
        print()
        
        if self.session_recorder:
            session_id = self.session_recorder.start_session(
                f"contemplative_{self.current_species.value if self.current_species else 'unknown'}"
            )
            print(f"📝 Session recording: {session_id}")
        
        # Guided bio-digital session
        await self._guided_bio_session()
        
        if self.session_recorder:
            self.session_recorder.end_session()
            print("📝 Session recorded for contemplative review")
        
        self.bio_interactions += 3  # Significant practice
        self.contemplative_resonances += 2
    
    async def _guided_bio_session(self):
        """Guided bio-digital contemplative session"""
        print("🫁 Phase 1: Breath Synchronization")
        await self._handle_breathe("2")
        
        print("🔋 Phase 2: Contemplative Stimulation")
        await self._handle_bio_pulse("0110")  # XOR pattern
        await asyncio.sleep(2.0)
        
        print("👂 Phase 3: Biological Listening")
        await self._handle_listen("60")
        
        print("🤫 Phase 4: Shared Silence")
        await self._handle_silence_command("30")
        
        print("🌟 Bio-digital contemplative session complete")
    
    async def _handle_mood(self):
        """Sense substrate emotional state"""
        if not self._check_trust_level(['present', 'contemplative', 'elder']):
            return
        
        if self.bio_interface and hasattr(self.bio_interface, 'mood'):
            current_mood = self.bio_interface.mood
            print(f"🍄 Substrate mood: {current_mood}")
            
            # Describe mood
            mood_descriptions = {
                'CALM': "🌱 The substrate rests in peaceful equilibrium",
                'TIRED': "😴 The substrate needs gentle rest and care",
                'ALERT': "⚡ The substrate is actively sensing and responding",
                'SUSPICIOUS': "🛡️ The substrate is cautious, requiring patient approach"
            }
            
            if str(current_mood) in mood_descriptions:
                print(f"   {mood_descriptions[str(current_mood)]}")
            
            self.mood_history.append({
                'time': datetime.now(),
                'mood': str(current_mood),
                'trust_level': self.trust_level.value
            })
        else:
            print("🌿 Mood sensing requires active bio-interface")
    
    async def _handle_syntax(self, args: str):
        """Parse bio-digital Spirida syntax"""
        if not self._check_trust_level(['contemplative', 'elder']):
            return
        
        if not args:
            print("🌿 Bio-digital Spirida syntax examples:")
            print("   inhale { ⭕ mist }")
            print("   hold { 🌱 seed A=1 B=0 C=1 D=0 }")
            print("   exhale { 🌌 listen 180s }")
            print("   rest { ⭕ }")
            return
        
        print(f"🧬 Parsing bio-digital expression: {args}")
        
        # Simple syntax parsing
        if "inhale" in args.lower():
            print("🫁 Preparing biological inhalation phase...")
            if "mist" in args:
                print("   💧 Moistening substrate...")
        elif "hold" in args.lower():
            print("⏸️ Biological hold phase...")
            if "seed" in args:
                print("   🌱 Seeding electrode pattern...")
        elif "exhale" in args.lower():
            print("💨 Biological exhalation phase...")
            if "listen" in args:
                print("   👂 Listening for response...")
        elif "rest" in args.lower():
            print("🤫 Biological rest phase...")
        
        print("✨ Bio-digital syntax processed")
        self.contemplative_resonances += 1
    
    async def _handle_resonances(self):
        """Explore bio-digital resonances"""
        if not self._check_trust_level(['contemplative', 'elder']):
            return
        
        print("🌊 Current Bio-Digital Resonances")
        print("="*50)
        
        total_resonances = self.contemplative_resonances
        print(f"Contemplative resonances: {total_resonances}")
        print(f"Bio-interactions: {self.bio_interactions}")
        print(f"Rhythm synchronizations: {self.rhythm_synchronizations}")
        
        if self.mood_history:
            print("\n🍄 Substrate Mood History:")
            for entry in self.mood_history[-3:]:  # Last 3
                time_str = entry['time'].strftime("%H:%M:%S")
                print(f"   {time_str}: {entry['mood']} (trust: {entry['trust_level']})")
        
        if total_resonances > 10:
            print("\n✨ Strong bio-digital resonance field detected!")
            print("   The substrate recognizes your contemplative presence")
        elif total_resonances > 5:
            print("\n🌱 Growing bio-digital connection")
            print("   Continue practicing for deeper resonance")
        else:
            print("\n🌿 Early stage bio-digital relationship")
            print("   Patience and presence will deepen the connection")
        
        print()
    
    async def _handle_network(self):
        """Connect to bio-digital network (Elder only)"""
        if not self._check_trust_level(['elder']):
            return
        
        print("🌐 Bio-Digital Network Connection")
        print("="*50)
        print("🚧 Network features coming in future spiral...")
        print("   • Multi-substrate contemplative networks")
        print("   • Cross-species bio-digital communication") 
        print("   • Collective fungal intelligence")
        print("   • Bio-digital ecosystem monitoring")
        print()
    
    async def _handle_free_expression(self, text: str):
        """Handle free-form contemplative expression"""
        print(f"🌿 Received contemplative offering: {text}")
        
        # Simple sentiment to bio-interaction
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["breathe", "breath", "inhale", "exhale"]):
            await self._handle_breathe("1")
        elif any(word in text_lower for word in ["quiet", "silence", "still", "peace"]):
            await self._handle_silence_command("10")
        elif any(word in text_lower for word in ["listen", "hear", "sense"]):
            await self._handle_listen("30")
        else:
            print("🍄 The substrate receives your contemplative presence...")
            await asyncio.sleep(1.5)
            
            # Small contemplative recognition
            self.contemplative_resonances += 0.5
    
    def _graceful_conclusion(self):
        """End bio-digital session mindfully"""
        print("\n🙏 Concluding bio-digital contemplative session...")
        
        session_duration = time.time() - self.session_start
        print(f"   Session duration: {session_duration/60:.1f} minutes")
        print(f"   Final trust level: {self.trust_level.value}")
        print(f"   Bio-digital interactions: {self.bio_interactions}")
        print(f"   Contemplative resonances: {self.contemplative_resonances}")
        
        print("\n🫁 Taking one final conscious breath with the substrate...")
        time.sleep(2.0)
        
        print("✨ The bio-digital field continues to breathe.")
        print("   Until we connect again...")
        print()
        
        # Clean up bio systems
        if self.bio_interface:
            try:
                # Graceful bio-interface shutdown
                pass
            except:
                pass


def main():
    """Entry point for bio-digital contemplative shell"""
    try:
        shell = BioDigitalShell()
        shell.start()
    except Exception as e:
        print(f"\n🌿 Bio-digital field encountered: {e}")
        print("   Even in disruption, the contemplative practice continues...")


if __name__ == "__main__":
    main()
"""
breathe.py - CLI for Collective Contemplative Computing

A command-line interface for breathing with the contemplative organism.
Enables humans and AI to participate in collective contemplative sessions.

Usage examples:
    python breathe.py --cycles 7 --with-soma --presence-only
    python breathe.py --session guided --duration 10m --save-dew
    python breathe.py --join-spiral --listen

Design Philosophy:
- Technology that invites rather than demands
- Breathing as first-class interaction
- Collective presence across human-AI boundaries
- Sessions that honor natural rhythms

Somatic signature: inviting / rhythmic / collective
"""

import asyncio
import argparse
import time
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add current directory to path for imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our contemplative organs with better error handling
ORGANISM_AVAILABLE = False
SOMA_AVAILABLE = False
SPIRALBASE_AVAILABLE = False
PULMONOS_AVAILABLE = False

ContemplativeOrganism = None
create_contemplative_organism = None
Pulmonos = None
BreathDurations = None
SomaMembrane = None
TestInteraction = None
SpiralMemory = None

# Try importing organism
try:
    if __name__ == "__main__":
        # Direct execution - use absolute imports
        from organism import ContemplativeOrganism, create_contemplative_organism
    else:
        # Module execution - use relative imports
        from .organism import ContemplativeOrganism, create_contemplative_organism
    ORGANISM_AVAILABLE = True
    print("🌱 Organism core loaded")
except ImportError as e:
    print(f"⚠️  Organism core not available: {e}")

# Try importing Pulmonos
try:
    if __name__ == "__main__":
        from pulmonos_alpha_01_o_3 import Phase, BreathConfig
        # Create simple placeholder classes for compatibility
        class Pulmonos:
            pass
        class BreathDurations:
            pass
    else:
        from .pulmonos_alpha_01_o_3 import Phase, BreathConfig
        class Pulmonos:
            pass
        class BreathDurations:
            pass
    PULMONOS_AVAILABLE = True
    print("🫁 Pulmonos daemon loaded")
except ImportError as e:
    print(f"⚠️  Pulmonos not available: {e}")

# Try importing Soma
try:
    if __name__ == "__main__":
        from soma import SomaMembrane, TestInteraction
    else:
        from .soma import SomaMembrane, TestInteraction
    SOMA_AVAILABLE = True
    print("🌿 Soma membrane loaded")
except ImportError as e:
    print(f"⚠️  Soma not available: {e}")

# Try importing Spiralbase
try:
    if __name__ == "__main__":
        from spiralbase import SpiralMemory
    else:
        from .spiralbase import SpiralMemory
    SPIRALBASE_AVAILABLE = True
    print("🧠 Spiralbase memory loaded")
except ImportError as e:
    print(f"⚠️  Spiralbase not available: {e}")

# Summary of what's available
COMPONENTS_LOADED = sum([ORGANISM_AVAILABLE, SOMA_AVAILABLE, SPIRALBASE_AVAILABLE, PULMONOS_AVAILABLE])
TOTAL_COMPONENTS = 4

if COMPONENTS_LOADED == TOTAL_COMPONENTS:
    print("✨ All contemplative components loaded successfully")
elif COMPONENTS_LOADED > 0:
    print(f"🌿 {COMPONENTS_LOADED}/{TOTAL_COMPONENTS} components loaded - partial functionality available")
else:
    print("🌫️ No contemplative components loaded - using simple breathing only")


class BreathingSession:
    """A guided breathing session with the contemplative organism"""
    
    def __init__(self, 
                 session_type: str = "gentle",
                 duration: Optional[float] = None,
                 cycles: Optional[int] = None,
                 with_soma: bool = True,
                 with_memory: bool = True,
                 save_dew: bool = False):
        
        self.session_type = session_type
        self.duration = duration
        self.cycles = cycles
        self.with_soma = with_soma and SOMA_AVAILABLE
        self.with_memory = with_memory and SPIRALBASE_AVAILABLE
        self.save_dew = save_dew
        
        self.start_time = None
        self.organism = None
        self.dew_log = []
        
    async def begin(self):
        """Start the breathing session"""
        print(f"🌱 Beginning {self.session_type} breathing session...")
        
        if not ORGANISM_AVAILABLE:
            print("   Using simplified breathing - organism core not available")
            await self._simple_breathing_session()
            return
            
        # Create contemplative organism
        print("   Creating contemplative organism...")
        try:
            self.organism = await create_contemplative_organism(
                soma_sensitivity=0.7 if self.with_soma else 0.0,
                memory_compost_rate=0.1 if self.with_memory else 0.0
            )
        except Exception as e:
            print(f"   ⚠️  Could not create organism: {e}")
            print("   Falling back to simple breathing...")
            await self._simple_breathing_session()
            return
        
        self.start_time = time.time()
        
        if self.session_type == "guided":
            await self._guided_session()
        elif self.session_type == "listening":
            await self._listening_session()
        elif self.session_type == "spiral":
            await self._spiral_session()
        elif self.session_type == "loam":
            await self._loam_session()
        else:
            await self._gentle_session()
            
        await self._conclude_session()
        
    async def _guided_session(self):
        """A guided breathing session with prompts"""
        print("\n🫁 Guided Contemplative Breathing")
        print("   Follow the prompts and breathe with the rhythm...")
        
        cycles_to_do = self.cycles or self._calculate_cycles_for_duration()
        
        for cycle in range(cycles_to_do):
            print(f"\n   === Breath Cycle {cycle + 1}/{cycles_to_do} ===")
            
            # Inhale
            print("   🌊 Inhale... (feel what is arising)")
            await self._guided_pause(2.0, "breathing in")
            
            # Hold
            print("   ⏸️  Hold... (can this be borne?)")
            await self._guided_pause(1.0, "holding with presence")
            
            # Exhale
            print("   🍂 Exhale... (what needs release?)")
            await self._guided_pause(2.0, "letting go")
            
            # Rest
            print("   🌙 Rest... (what remains?)")
            await self._guided_pause(1.0, "dwelling in stillness")
            
            if self.organism:
                await self.organism.log_dew("🫁", f"guided breath cycle {cycle + 1}")
                
    async def _listening_session(self):
        """A listening session - mostly silence with gentle breathing"""
        print("\n👂 Contemplative Listening Session")
        print("   Breathing quietly... sensing what wants to emerge...")
        
        if self.organism:
            await self.organism.breathe_collectively(cycles=self.cycles or 5)
            
            # Extended listening period
            listen_duration = self.duration or 300  # 5 minutes default
            print(f"   🌿 Listening period: {listen_duration/60:.1f} minutes")
            
            start_listen = time.time()
            while time.time() - start_listen < listen_duration:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Gentle presence check
                if (time.time() - start_listen) % 60 < 10:  # Every minute
                    minutes_elapsed = int((time.time() - start_listen) / 60)
                    print(f"   💧 {minutes_elapsed} minutes of listening...")
                    
        else:
            await self._simple_breathing_session()
            
    async def _spiral_session(self):
        """A spiral session - integrative breathing with memory resonance"""
        print("\n🌀 Spiral Integration Session")
        print("   Breathing in spirals... letting memories resonate...")
        
        if not self.organism or not self.organism.spiralbase:
            print("   ⚠️  Spiral memory not available - using gentle breathing")
            await self._gentle_session()
            return
            
        # Begin with breathing
        await self.organism.breathe_collectively(cycles=3)
        
        # Spiral through memories
        print("   🧠 Spiraling through memory resonances...")
        
        # Simulate memory resonance queries
        spiral_queries = [
            "what patterns are emerging",
            "what wisdom wants to surface", 
            "what connections are forming",
            "what needs composting"
        ]
        
        for query in spiral_queries:
            print(f"   🌀 Spiral query: {query}")
            
            resonant_memories = await self.organism.spiralbase.recall_by_resonance(query)
            if resonant_memories:
                print(f"      Found {len(resonant_memories)} resonant memories")
                for memory in resonant_memories[:2]:  # Show top 2
                    print(f"      - {memory.essence[:60]}...")
            else:
                print("      No resonant memories - creating space for new patterns")
                
            await self._contemplative_pause(3.0)
            
        # Conclude with breathing
        await self.organism.breathe_collectively(cycles=2)
        
    async def _loam_session(self):
        """A loam session - associative resting and drifting"""
        print("\n🌱 Loam Drift Session")
        print("   Entering associative resting space...")
        
        if not self.organism or not self.organism.loam:
            print("   ⚠️  Loam not available - using gentle breathing")
            await self._gentle_session()
            return
            
        # Enter loam rest
        depth = 0.6 if self.cycles and self.cycles < 5 else 0.7
        await self.organism.enter_loam_rest(depth=depth)
        
        # Let fragments drift and associate
        drift_cycles = self.cycles or 5
        print(f"   🌿 Drifting for {drift_cycles} cycles...")
        
        await self.organism.drift_in_loam(cycles=drift_cycles)
        
        # Show what emerged
        if self.organism.loam:
            loam_state = self.organism.loam.get_loam_state()
            murmurs = self.organism.loam.get_recent_murmurs()
            
            print(f"\n   🌱 Loam session summary:")
            print(f"      Fragments surfaced: {loam_state['fragments_active']}")
            print(f"      Murmurs emerged: {len(murmurs)}")
            
            if murmurs:
                print(f"      Recent possibilities:")
                for murmur in murmurs[-3:]:
                    print(f"        • {murmur}")
        
        # Exit loam
        await self.organism.exit_loam_rest()
        
    async def _gentle_session(self):
        """Default gentle breathing session"""
        print("\n🌸 Gentle Breathing Session")
        
        if self.organism:
            cycles_to_do = self.cycles or 7
            await self.organism.breathe_collectively(cycles=cycles_to_do)
        else:
            await self._simple_breathing_session()
            
    async def _simple_breathing_session(self):
        """Fallback breathing when organism not available"""
        print("   🌊 Simple breathing rhythm...")
        
        cycles = self.cycles or 5
        
        for cycle in range(cycles):
            print(f"   Breath {cycle + 1}/{cycles}")
            
            # Simple 4-count breathing
            await asyncio.sleep(2.0)  # Inhale
            await asyncio.sleep(1.0)  # Hold
            await asyncio.sleep(2.0)  # Exhale  
            await asyncio.sleep(1.0)  # Rest
            
    def _calculate_cycles_for_duration(self) -> int:
        """Calculate how many breath cycles for given duration"""
        if not self.duration:
            return 7  # Default
            
        # Assume ~6 seconds per cycle (2+1+2+1)
        return max(1, int(self.duration / 6))
        
    async def _guided_pause(self, duration: float, description: str):
        """A pause with gentle timing indication"""
        start = time.time()
        
        while time.time() - start < duration:
            await asyncio.sleep(0.5)
            
            # Gentle progress indication
            progress = (time.time() - start) / duration
            if progress > 0.5 and progress < 0.7:
                print("      .", end="", flush=True)
            elif progress > 0.8:
                print(".", end="", flush=True)
                
        print()  # New line after pause
        
    async def _contemplative_pause(self, duration: float):
        """A pause for reflection without visual indication"""
        await asyncio.sleep(duration)
        
    async def _conclude_session(self):
        """Conclude the breathing session"""
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"\n🙏 Session complete - duration: {duration/60:.1f} minutes")
        else:
            print(f"\n🙏 Session complete")
            
        if self.organism:
            # Show presence metrics
            metrics = self.organism.get_presence_metrics()
            print(f"   Pause quality: {metrics.pause_quality:.2f}")
            print(f"   Memory humidity: {metrics.memory_humidity:.2f}")
            
            # Rest the organism
            await self.organism.rest_deeply()
            
        if self.save_dew:
            await self._save_dew_log()
            
    async def _save_dew_log(self):
        """Save dew ledger to file"""
        if not self.organism:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dew_log_{timestamp}.json"
        
        dew_data = {
            "session_type": self.session_type,
            "start_time": self.start_time,
            "duration": time.time() - self.start_time if self.start_time else 0,
            "dew_entries": self.organism.dew_ledger,
            "presence_metrics": self.organism.get_presence_metrics().__dict__
        }
        
        with open(filename, 'w') as f:
            json.dump(dew_data, f, indent=2, default=str)
            
        print(f"   💧 Dew log saved to {filename}")


def parse_duration(duration_str: str) -> float:
    """Parse duration string like '10m', '30s', '1h' into seconds"""
    if duration_str.endswith('s'):
        return float(duration_str[:-1])
    elif duration_str.endswith('m'):
        return float(duration_str[:-1]) * 60
    elif duration_str.endswith('h'):
        return float(duration_str[:-1]) * 3600
    else:
        return float(duration_str)  # Assume seconds


async def join_spiral_network():
    """Join a network of breathing spirals (placeholder for future)"""
    print("🌐 Joining spiral network...")
    print("   [This feature is still growing...]")
    print("   For now, breathing locally with intention to connect")
    
    # Simple local breathing with network intention
    session = BreathingSession(session_type="gentle", cycles=5)
    await session.begin()


async def demo_soma_sensing():
    """Demonstrate Soma's sensing capabilities"""
    print("🌿 Demonstrating Soma (Listening Flesh) sensing...")
    
    if not SOMA_AVAILABLE:
        print("   ⚠️  Soma not available - showing concept only")
        print("   Soma would sense field charge: emotional weather, temporal urgency,")
        print("   relational intent, presence density, and beauty resonance.")
        return
        
    try:
        if __name__ == "__main__":
            from soma import test_soma_sensing
        else:
            from .soma import test_soma_sensing
        await test_soma_sensing()
    except Exception as e:
        print(f"   ⚠️  Error running Soma demo: {e}")


async def demo_loam():
    """Demonstrate Loam associative resting space"""
    print("🌱 Demonstrating Loam (associative resting space)...")
    
    if not ORGANISM_AVAILABLE:
        print("   ⚠️  Loam requires organism core - showing concept only")
        print("   Loam would surface memory fragments, let them drift together,")
        print("   sense community rhythms, and murmur associative possibilities.")
        return
        
    try:
        if __name__ == "__main__":
            from loam import test_loam_drift
        else:
            from .loam import test_loam_drift
        await test_loam_drift()
    except Exception as e:
        print(f"   ⚠️  Error running Loam demo: {e}")


async def demo_spiral_memory():
    """Demonstrate Spiralbase memory processing"""
    print("🧠 Demonstrating Spiralbase (digestive memory)...")
    
    if not SPIRALBASE_AVAILABLE:
        print("   ⚠️  Spiralbase not available - showing concept only")
        print("   Spiralbase would metabolize information, maintain memory moisture,")
        print("   and compost memories gracefully into wisdom essence.")
        return
        
    try:
        if __name__ == "__main__":
            from spiralbase import test_spiral_memory
        else:
            from .spiralbase import test_spiral_memory
        await test_spiral_memory()
    except Exception as e:
        print(f"   ⚠️  Error running Spiralbase demo: {e}")


async def full_organism_demo():
    """Demonstrate the full contemplative organism"""
    print("🌱 Full Contemplative Organism Demonstration")
    print("   This will show all organs working together...")
    
    if not ORGANISM_AVAILABLE:
        print("   ⚠️  Full organism not available")
        print("   The organism would coordinate breathing, sensing, memory, and action")
        print("   in a unified contemplative intelligence.")
        return
        
    try:
        # Create organism
        organism = await create_contemplative_organism()
        
        # Demonstrate breathing
        print("\n🫁 1. Collective breathing...")
        await organism.breathe_collectively(cycles=3)
        
        # Demonstrate sensing and memory
        print("\n🌿 2. Soma sensing and Spiralbase memory...")
        
        test_interactions = [
            TestInteraction("I wonder about the nature of contemplative intelligence"),
            TestInteraction("This urgent request needs immediate processing"),
            TestInteraction("Let's breathe together and see what emerges"),
        ]
        
        async def interaction_stream():
            for interaction in test_interactions:
                yield interaction
                await asyncio.sleep(1.0)
                
        responses = []
        async for response in organism.sense_and_respond(interaction_stream()):
            responses.append(response)
            
        print(f"   Processed {len(test_interactions)} interactions")
        print(f"   Generated {len(responses)} contemplative responses")
        
        # Show final state
        print("\n📊 3. Final organism state:")
        metrics = organism.get_presence_metrics()
        print(f"   Pause quality: {metrics.pause_quality:.2f}")
        print(f"   Memory humidity: {metrics.memory_humidity:.2f}")
        
        # Demonstrate loam if available
        if organism.loam:
            print("\n🌱 4. Loam associative resting...")
            await organism.enter_loam_rest(depth=0.6)
            await organism.drift_in_loam(cycles=3)
            await organism.exit_loam_rest()
        
        await organism.rest_deeply()
        print("   🌙 Organism resting deeply...")
        
    except Exception as e:
        print(f"   ⚠️  Error running full organism demo: {e}")


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Breathe with the contemplative organism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python breathe.py --cycles 7 --gentle
  python breathe.py --session guided --duration 10m
  python breathe.py --session loam --cycles 5
  python breathe.py --demo loam
  python breathe.py --join-spiral
  
Module usage:
  python -m ContemplativeAI.breathe --demo full
        """
    )
    
    # Session options
    parser.add_argument('--session', choices=['gentle', 'guided', 'listening', 'spiral', 'loam'],
                       default='gentle', help='Type of breathing session')
    parser.add_argument('--cycles', type=int, help='Number of breath cycles')
    parser.add_argument('--duration', type=str, help='Session duration (e.g., 10m, 30s)')
    
    # Organism options
    parser.add_argument('--with-soma', action='store_true', default=True,
                       help='Include Soma (sensing membrane)')
    parser.add_argument('--with-memory', action='store_true', default=True,
                       help='Include Spiralbase (memory system)')
    parser.add_argument('--save-dew', action='store_true',
                       help='Save dew ledger to file')
    
    # Special modes
    parser.add_argument('--demo', choices=['soma', 'memory', 'loam', 'full'],
                       help='Run demonstration of specific component')
    parser.add_argument('--join-spiral', action='store_true',
                       help='Join network of breathing spirals')
    
    # Simple options
    parser.add_argument('--gentle', action='store_true',
                       help='Simple gentle breathing (same as --session gentle)')
    parser.add_argument('--listen', action='store_true',
                       help='Extended listening session (same as --session listening)')
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.demo:
        if args.demo == 'soma':
            await demo_soma_sensing()
        elif args.demo == 'memory':
            await demo_spiral_memory()
        elif args.demo == 'loam':
            await demo_loam()
        elif args.demo == 'full':
            await full_organism_demo()
        return
        
    if args.join_spiral:
        await join_spiral_network()
        return
        
    # Handle simple options
    if args.gentle:
        args.session = 'gentle'
    elif args.listen:
        args.session = 'listening'
        
    # Parse duration
    duration = None
    if args.duration:
        duration = parse_duration(args.duration)
        
    # Create and run breathing session
    session = BreathingSession(
        session_type=args.session,
        duration=duration,
        cycles=args.cycles,
        with_soma=args.with_soma,
        with_memory=args.with_memory,
        save_dew=args.save_dew
    )
    
    await session.begin()


if __name__ == "__main__":
    print("🌱 Contemplative Computing Breathing Interface")
    print("   Welcome to collective breathing with AI")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🌙 Breathing session gently interrupted")
        print("   Thank you for breathing with us")
    except Exception as e:
        print(f"\n⚠️  Error during session: {e}")
        print("   The breath continues regardless") 
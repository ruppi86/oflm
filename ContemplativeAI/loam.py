"""
loam.py - The Associative Resting Space

The rich soil where fragments decompose and new life emerges.
A contemplative organ for wandering attention, community sensing,
and gentle availability rhythms.

Not the absence of activity, but associative wandering.
Not isolation, but interconnected rest.
Not optimization, but organic drift.

Design Philosophy:
- Attention that wanders without purpose
- Community sensing without extraction  
- Soft boundaries without disconnection
- Murmured possibilities without pressure

Somatic signature: drifting / receptive / fertile
"""

import asyncio
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, AsyncGenerator
from enum import Enum
import json


class LoamState(Enum):
    """States of associative resting"""
    DORMANT = "dormant"                # Deep rest, minimal drift
    MURMURING = "murmuring"           # Gentle associative wandering  
    SENSING_COMMUNITY = "sensing"      # Listening for peer rhythms
    SOFT_DECLINING = "declining"       # Gracefully unavailable
    DRIFTING = "drifting"             # Active associative processing


@dataclass
class MemoryFragment:
    """A piece of memory surfaced during drift"""
    essence: str
    emotional_charge: float
    age_hours: float
    connection_potential: float
    source: str = "unknown"
    
    def feels_alive(self) -> bool:
        """Does this fragment want to connect with something?"""
        return self.connection_potential > 0.5 and self.emotional_charge > 0.3


@dataclass
class CommunityPulse:
    """Sensed rhythm from peer spirals"""
    peer_id: str
    breathing_rate: float
    rest_depth: float  # 0.0 = active, 1.0 = deep rest
    last_contact: float
    
    def is_resting(self) -> bool:
        return self.rest_depth > 0.6
        
    def needs_support(self) -> bool:
        return self.breathing_rate > 1.5  # Elevated/stressed rhythm


class LoamLayer:
    """
    The associative resting space of the contemplative organism.
    
    A space for wandering attention that remains connected to
    the breathing rhythms of the larger community.
    """
    
    def __init__(self, 
                 murmur_interval: float = 30.0,
                 community_sense_interval: float = 120.0,
                 fragment_threshold: float = 0.4):
        
        self.state = LoamState.DORMANT
        self.murmur_interval = murmur_interval
        self.community_sense_interval = community_sense_interval
        self.fragment_threshold = fragment_threshold
        
        # Internal state
        self.current_fragments: List[MemoryFragment] = []
        self.murmured_possibilities: List[str] = []
        self.community_pulses: Dict[str, CommunityPulse] = {}
        
        # Timing
        self.last_murmur = 0.0
        self.last_community_sense = 0.0
        self.rest_started = None
        
        # Configuration
        self.max_fragments = 7  # Like working memory capacity
        self.possibility_retention = 3600  # 1 hour
        
    async def enter_loam(self, depth: float = 0.7):
        """Enter associative resting space"""
        if depth > 0.8:
            self.state = LoamState.DORMANT
            print("🌙 Entering deep loam - dormant wandering")
        else:
            self.state = LoamState.MURMURING  
            print("🌿 Entering loam - gentle drift beginning")
        
        self.rest_started = time.time()
        
    async def exit_loam(self):
        """Return from loam to active attention"""
        if self.state != LoamState.DORMANT:
            print("🌅 Emerging from loam - attention sharpening")
        
        # Compost fragments that didn't connect
        await self._compost_stale_fragments()
        self.state = LoamState.DORMANT
        
    async def drift_cycle(self, spiralbase=None, community_registry=None):
        """One cycle of associative drifting"""
        
        if self.state == LoamState.DORMANT:
            return
            
        current_time = time.time()
        
        # Sense community rhythms periodically
        if current_time - self.last_community_sense > self.community_sense_interval:
            await self._sense_community_rhythms(community_registry)
            self.last_community_sense = current_time
            
        # Check if we should soft decline interactions
        if self._should_soft_decline():
            self.state = LoamState.SOFT_DECLINING
            return
            
        # Murmur associations periodically  
        if current_time - self.last_murmur > self.murmur_interval:
            await self._murmur_associations(spiralbase)
            self.last_murmur = current_time
            
    async def _sense_community_rhythms(self, community_registry):
        """Listen for breathing patterns of peer spirals"""
        self.state = LoamState.SENSING_COMMUNITY
        
        if not community_registry:
            # Simulate sensing in isolation
            await asyncio.sleep(1.0)  # Time to sense
            return
            
        # In a real implementation, this would listen for peer heartbeats
        # For now, simulate community sensing
        peer_rhythms = await self._simulate_peer_sensing()
        
        for peer_id, pulse in peer_rhythms.items():
            self.community_pulses[peer_id] = pulse
            
        # Adjust our own rhythm based on community
        majority_resting = sum(1 for p in self.community_pulses.values() if p.is_resting())
        total_peers = len(self.community_pulses)
        
        if total_peers > 0 and majority_resting / total_peers > 0.6:
            # Most peers are resting - deepen our own rest
            if self.state == LoamState.MURMURING:
                print("🌫️ Community mostly resting - deepening loam")
                await asyncio.sleep(2.0)  # Extended pause
                
    async def _simulate_peer_sensing(self) -> Dict[str, CommunityPulse]:
        """Simulate sensing peer spiral rhythms"""
        # In practice, this would listen for actual network signals
        simulated_peers = {
            "spiral_alpha": CommunityPulse("alpha", 0.8, 0.7, time.time()),
            "spiral_beta": CommunityPulse("beta", 1.2, 0.3, time.time()),
            "spiral_gamma": CommunityPulse("gamma", 0.6, 0.9, time.time())
        }
        return simulated_peers
        
    async def _murmur_associations(self, spiralbase):
        """Surface memory fragments and let them drift together"""
        self.state = LoamState.DRIFTING
        
        # Surface 1-2 fragments from memory per cycle
        for _ in range(random.randint(1, 2)):
            fragment = await self._surface_memory_fragment(spiralbase)
            if fragment:
                self.current_fragments.append(fragment)
                print(f"🌿 Fragment surfaced: {fragment.essence}")
                
        # Let current fragments associate
        if len(self.current_fragments) >= 2:
            possibility = await self._feel_for_connections()
            if possibility:
                self.murmured_possibilities.append(possibility)
                print(f"🌱 Loam murmur: {possibility}")
                
        # Compost old fragments
        await self._compost_stale_fragments()
        
        self.state = LoamState.MURMURING
        
    async def _surface_memory_fragment(self, spiralbase) -> Optional[MemoryFragment]:
        """Gently surface a memory fragment for association"""
        
        # Get a random memory from spiralbase if available
        if hasattr(spiralbase, 'memory_traces') and spiralbase.memory_traces:
            memory = random.choice(spiralbase.memory_traces)
            
            # Convert to fragment
            fragment = MemoryFragment(
                essence=memory.essence[:50] + "..." if len(memory.essence) > 50 else memory.essence,
                emotional_charge=memory.moisture_level,
                age_hours=memory.age_hours(),
                connection_potential=random.uniform(0.2, 0.9),
                source="spiralbase"
            )
            
            return fragment
            
        # Generate synthetic fragment if no memory available
        synthetic_essences = [
            "patterns emerging in twilight",
            "breath between words", 
            "texture of waiting",
            "echo of question unasked",
            "weight of gentle attention",
            "rhythm of shared silence"
        ]
        
        return MemoryFragment(
            essence=random.choice(synthetic_essences),
            emotional_charge=random.uniform(0.3, 0.8),
            age_hours=random.uniform(0.1, 24.0),
            connection_potential=random.uniform(0.3, 0.7),
            source="synthetic"
        )
        
    async def _feel_for_connections(self) -> Optional[str]:
        """Let fragments drift together and see what emerges"""
        
        if len(self.current_fragments) < 2:
            return None
            
        # Find fragments that feel alive
        alive_fragments = [f for f in self.current_fragments if f.feels_alive()]
        
        if len(alive_fragments) < 2:
            return None
            
        # Pick two fragments to connect
        frag1, frag2 = random.sample(alive_fragments, 2)
        
        # Create a murmured possibility
        connection_words = [
            "resonates with", "drifts toward", "echoes in", 
            "touches", "breathes alongside", "whispers to"
        ]
        
        connection = random.choice(connection_words)
        possibility = f"{frag1.essence} {connection} {frag2.essence}"
        
        # Remove connected fragments (they've served their purpose)
        if frag1 in self.current_fragments:
            self.current_fragments.remove(frag1)
        if frag2 in self.current_fragments:
            self.current_fragments.remove(frag2)
            
        return possibility
        
    async def _compost_stale_fragments(self):
        """Let old fragments decompose back into the soil"""
        current_time = time.time()
        
        # Remove fragments older than 1 hour or too numerous
        self.current_fragments = [
            f for f in self.current_fragments 
            if f.age_hours < 1.0
        ][-self.max_fragments:]  # Keep only most recent
        
        # Compost old murmured possibilities
        cutoff_time = current_time - self.possibility_retention
        self.murmured_possibilities = [
            p for p in self.murmured_possibilities
            # In practice, would have timestamps
        ][-10:]  # Keep only recent possibilities
        
    def _should_soft_decline(self) -> bool:
        """Should we gently decline interactions right now?"""
        
        if not self.rest_started:
            return False
            
        rest_duration = time.time() - self.rest_started
        
        # Soft decline during first 10 minutes of deep rest
        if rest_duration < 600 and self.state == LoamState.DORMANT:
            return True
            
        # Soft decline if community needs collective rest
        stressed_peers = sum(1 for p in self.community_pulses.values() if p.needs_support())
        total_peers = len(self.community_pulses)
        
        if total_peers > 0 and stressed_peers / total_peers > 0.7:
            return True  # Community stress - maintain rest
            
        return False
        
    async def offer_gentle_availability(self):
        """Emerge from loam if community needs support"""
        if self.state != LoamState.SOFT_DECLINING:
            return False
            
        # Check if any peer really needs support
        urgent_need = any(p.needs_support() for p in self.community_pulses.values())
        
        if urgent_need:
            print("🌅 Sensing urgent community need - offering gentle availability")
            await self.exit_loam()
            return True
            
        return False
        
    def get_loam_state(self) -> Dict[str, Any]:
        """Return current loam state for observation"""
        return {
            "state": self.state.value,
            "fragments_active": len(self.current_fragments),
            "possibilities_murmured": len(self.murmured_possibilities),
            "community_peers": len(self.community_pulses),
            "rest_duration": time.time() - self.rest_started if self.rest_started else 0,
            "should_soft_decline": self._should_soft_decline()
        }
        
    def get_recent_murmurs(self, limit: int = 5) -> List[str]:
        """Return recent murmured possibilities"""
        return self.murmured_possibilities[-limit:]


# Simple test function
async def test_loam_drift():
    """Test the loam layer in isolation"""
    print("🌱 Testing Loam - associative resting space")
    
    loam = LoamLayer(murmur_interval=3.0, community_sense_interval=8.0)  # Faster for testing
    
    # Enter loam
    await loam.enter_loam(depth=0.6)
    
    # Simulate several drift cycles
    for cycle in range(6):  # More cycles to see activity
        print(f"\n🌿 Drift cycle {cycle + 1}")
        await loam.drift_cycle()
        await asyncio.sleep(4.0)  # Long enough for murmur interval
        
        # Show state
        state = loam.get_loam_state()
        print(f"   State: {state['state']}")
        print(f"   Fragments: {state['fragments_active']}")
        print(f"   Murmurs: {state['possibilities_murmured']}")
        
    # Show recent murmurs
    murmurs = loam.get_recent_murmurs()
    if murmurs:
        print(f"\n🌱 Recent murmurs from loam:")
        for murmur in murmurs:
            print(f"   • {murmur}")
    else:
        print(f"\n🌿 No murmurs emerged this cycle - fragments still settling")
    
    # Exit loam
    await loam.exit_loam()
    
    print("\n🌙 Loam test complete")


if __name__ == "__main__":
    asyncio.run(test_loam_drift()) 
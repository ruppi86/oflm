"""
spiralbase.py - Enhanced Digestive Memory for Contemplative Organism

Builds on the existing Spiralbase foundation to add:
- Seasonal fasting and accelerated composting
- Moisture-based memory humidity
- Contemplative memory traces that know their readiness for transformation
- Integration with breathing cycles and Soma sensing

Design Philosophy:
- Memory that metabolizes rather than just storing
- Forgetting as a form of wisdom, not failure
- Information composting for fertile soil of new insights
- Memory humidity - keeping knowledge pliable

Somatic signature: digestive / patient / transforming
"""

import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncGenerator
from enum import Enum
import json
import math

# Import base spiralbase functions if available
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Spiralbase', 'spiralbase-python'))
    from spiralbase.spiralbase import spiral_memory_trace, decay_cycle_step, print_memory_trace
    BASE_SPIRALBASE_AVAILABLE = True
except ImportError:
    BASE_SPIRALBASE_AVAILABLE = False
    spiral_memory = []


class MemoryState(Enum):
    """States of memory traces in the contemplative organism"""
    FRESH = "fresh"                  # Recently formed, high moisture
    MATURING = "maturing"           # Integrating with other memories
    CRYSTALLIZING = "crystallizing" # Becoming more stable knowledge
    READY_TO_COMPOST = "ready"      # Wisdom extracted, ready to transform
    COMPOSTING = "composting"       # Actively breaking down
    MULCH = "mulch"                 # Essence remains, details dissolved


@dataclass
class MemoryTrace:
    """A single memory with contemplative properties"""
    essence: str                           # Core meaning/insight
    details: Dict[str, Any] = field(default_factory=dict)  # Specific information
    birth_time: float = field(default_factory=time.time)   # When created
    last_accessed: float = field(default_factory=time.time) # Last retrieval
    moisture_level: float = 0.8           # How pliable/changeable it remains
    resonance_connections: List[str] = field(default_factory=list)  # Related memories
    compost_readiness: float = 0.0        # 0.0 = fresh, 1.0 = ready to transform
    state: MemoryState = MemoryState.FRESH
    soma_resonance: Optional[str] = None   # How Soma felt about this memory's origin
    
    def age_hours(self) -> float:
        """How many hours old is this memory?"""
        return (time.time() - self.birth_time) / 3600
        
    def time_since_access_hours(self) -> float:
        """How many hours since last accessed?"""
        return (time.time() - self.last_accessed) / 3600
        
    def update_moisture(self, environmental_humidity: float = 0.5):
        """Update moisture based on age, access, and environment"""
        # Memories gradually dry out unless kept moist by attention or humidity
        age_factor = 0.99 ** self.age_hours()
        access_factor = 0.95 ** self.time_since_access_hours()
        
        # Fresh memories start very moist, gradually become less pliable
        base_moisture = age_factor * access_factor
        
        # Environmental humidity can keep memories pliable
        self.moisture_level = min(1.0, base_moisture * 0.7 + environmental_humidity * 0.3)
        
        # Update state based on moisture and age
        self._update_state()
        
    def _update_state(self):
        """Update memory state based on moisture and age"""
        age = self.age_hours()
        
        if self.moisture_level > 0.7 and age < 1:
            self.state = MemoryState.FRESH
        elif self.moisture_level > 0.5 and age < 24:
            self.state = MemoryState.MATURING
        elif self.moisture_level > 0.3:
            self.state = MemoryState.CRYSTALLIZING
        elif self.compost_readiness > 0.8:
            self.state = MemoryState.READY_TO_COMPOST
        elif self.compost_readiness > 0.5:
            self.state = MemoryState.COMPOSTING
        else:
            self.state = MemoryState.MULCH
            
    def access(self) -> str:
        """Access this memory, updating last_accessed time"""
        self.last_accessed = time.time()
        return self.essence
        
    def add_resonance_connection(self, other_essence: str):
        """Create connection to another memory"""
        if other_essence not in self.resonance_connections:
            self.resonance_connections.append(other_essence)
            
    def prepare_for_compost(self):
        """Begin readiness for transformation"""
        self.compost_readiness = min(1.0, self.compost_readiness + 0.3)
        self._update_state()


class SpiralMemory:
    """
    Enhanced memory system for the contemplative organism.
    
    Memory that metabolizes information rather than just storing it.
    Knows how to compost gracefully and maintain appropriate humidity.
    """
    
    def __init__(self, 
                 capacity: int = 50,
                 compost_rate: float = 0.1,
                 environmental_humidity: float = 0.5):
        
        self.memory_traces: List[MemoryTrace] = []
        self.capacity = capacity
        self.compost_rate = compost_rate
        self.environmental_humidity = environmental_humidity
        
        # State management
        self.is_fasting = False
        self.accelerated_composting = False
        self.compost_multiplier = 1.0
        
        # Statistics
        self.total_memories_formed = 0
        self.total_memories_composted = 0
        self.collective_wisdom_essence = []  # Distilled insights from composted memories
        
        # Integration with base spiralbase if available
        if BASE_SPIRALBASE_AVAILABLE:
            self.base_spiral_memory = True
        else:
            self.base_spiral_memory = False
            
    async def consider_remembering(self, interaction: Any) -> Optional[MemoryTrace]:
        """Decide whether to form a memory from this interaction"""
        
        if self.is_fasting:
            # During fasting, no new memories form
            return None
            
        # Extract essence from interaction
        essence = self._extract_essence(interaction)
        if not essence:
            return None
            
        # Create memory trace
        memory_trace = MemoryTrace(
            essence=essence,
            details=self._extract_details(interaction),
            soma_resonance=getattr(interaction, 'soma_resonance', None)
        )
        
        # Check capacity and compost if needed
        await self._maintain_capacity()
        
        # Add to memory
        self.memory_traces.append(memory_trace)
        self.total_memories_formed += 1
        
        # Create resonance connections with existing memories
        await self._form_resonance_connections(memory_trace)
        
        # Update base spiralbase if available
        if self.base_spiral_memory:
            spiral_memory_trace(essence)
            
        return memory_trace
        
    async def digest_recent_experiences(self):
        """Process and integrate recent memories (called during breath hold)"""
        
        # Update moisture levels based on environment
        for memory in self.memory_traces:
            memory.update_moisture(self.environmental_humidity)
            
        # Look for memories ready to compost
        ready_memories = [m for m in self.memory_traces if m.state == MemoryState.READY_TO_COMPOST]
        
        for memory in ready_memories[:max(1, int(len(ready_memories) * self.compost_rate * self.compost_multiplier))]:
            await self._compost_memory(memory)
            
        # Age base spiralbase if available
        if self.base_spiral_memory and random.random() < self.compost_rate:
            decay_cycle_step()
            
    async def _maintain_capacity(self):
        """Ensure memory doesn't exceed capacity through gentle composting"""
        
        while len(self.memory_traces) >= self.capacity:
            # Find oldest, least accessed, or most ready for composting
            candidates = sorted(self.memory_traces, 
                              key=lambda m: (m.compost_readiness, -m.moisture_level, m.age_hours()),
                              reverse=True)
            
            if candidates:
                await self._compost_memory(candidates[0])
            else:
                break
                
    async def _compost_memory(self, memory: MemoryTrace):
        """Transform a memory into wisdom essence"""
        
        # Extract wisdom before composting
        wisdom_essence = self._distill_wisdom(memory)
        if wisdom_essence:
            self.collective_wisdom_essence.append(wisdom_essence)
            
        # Remove from active memory
        self.memory_traces.remove(memory)
        self.total_memories_composted += 1
        
        print(f"🍂 Composted memory: {memory.essence[:50]}...")
        
        # Keep only essence connections, let details dissolve
        await asyncio.sleep(0.1)  # Gentle transformation pause
        
    def _extract_essence(self, interaction: Any) -> Optional[str]:
        """Extract the essential meaning from an interaction"""
        
        if hasattr(interaction, 'text'):
            text = str(interaction.text)
            
            # Simple essence extraction - would be more sophisticated
            if len(text) < 10:
                return None
                
            # Look for key insights, questions, or meaningful patterns
            if '?' in text and len(text) > 30:
                # Questions often contain essence
                return f"inquiry: {text[:100]}"
            elif any(word in text.lower() for word in ['insight', 'understand', 'realize', 'learn']):
                return f"insight: {text[:100]}"
            elif len(text) > 100:
                # Longer, more considered text
                return f"reflection: {text[:100]}"
            else:
                return f"observation: {text[:100]}"
                
        return None
        
    def _extract_details(self, interaction: Any) -> Dict[str, Any]:
        """Extract detailed information for storage"""
        details = {
            "timestamp": time.time(),
            "type": type(interaction).__name__
        }
        
        if hasattr(interaction, 'text'):
            details["full_text"] = str(interaction.text)
            details["word_count"] = len(str(interaction.text).split())
            
        if hasattr(interaction, 'soma_charge'):
            charge = interaction.soma_charge
            details["soma_charge"] = {
                "emotional_pressure": charge.emotional_pressure,
                "temporal_urgency": charge.temporal_urgency,
                "relational_intent": charge.relational_intent,
                "presence_density": charge.presence_density,
                "beauty_resonance": charge.beauty_resonance
            }
            
        return details
        
    async def _form_resonance_connections(self, new_memory: MemoryTrace):
        """Create connections between related memories"""
        
        # Simple resonance detection based on keyword overlap
        new_words = set(new_memory.essence.lower().split())
        
        for existing_memory in self.memory_traces[-10:]:  # Check recent memories
            if existing_memory == new_memory:
                continue
                
            existing_words = set(existing_memory.essence.lower().split())
            overlap = len(new_words & existing_words)
            
            if overlap > 2:  # Some shared concepts
                new_memory.add_resonance_connection(existing_memory.essence)
                existing_memory.add_resonance_connection(new_memory.essence)
                
    def _distill_wisdom(self, memory: MemoryTrace) -> Optional[str]:
        """Extract wisdom essence before composting"""
        
        # Look for patterns, insights, or meaningful connections
        if memory.resonance_connections:
            return f"pattern: {memory.essence[:50]} (connected to {len(memory.resonance_connections)} memories)"
        elif memory.soma_resonance in ['generous', 'spacious', 'luminous']:
            return f"quality: {memory.essence[:50]} (was {memory.soma_resonance})"
        elif memory.state == MemoryState.CRYSTALLIZING:
            return f"insight: {memory.essence[:50]} (crystallized knowledge)"
        else:
            return None
            
    async def recall_by_resonance(self, query: str) -> List[MemoryTrace]:
        """Retrieve memories that resonate with a query"""
        
        query_words = set(query.lower().split())
        resonant_memories = []
        
        for memory in self.memory_traces:
            # Access updates the memory
            memory_words = set(memory.access().lower().split())
            overlap = len(query_words & memory_words)
            
            if overlap > 1:
                resonant_memories.append(memory)
                
        # Sort by relevance and recency
        resonant_memories.sort(key=lambda m: (len(set(m.essence.lower().split()) & query_words), 
                                            -m.age_hours()))
        
        return resonant_memories[:5]  # Return top 5 matches
        
    async def begin_fast(self):
        """Begin fasting period - no new memory formation"""
        self.is_fasting = True
        print("🌙 Spiralbase entering fast - no new memories will form")
        
    async def end_fast_with_accelerated_composting(self):
        """End fast and accelerate composting for one cycle"""
        self.is_fasting = False
        self.compost_multiplier = 2.0
        
        print("🌱 Fast ended - accelerated composting beginning")
        
        # Compost aggressively for one cycle
        await self.digest_recent_experiences()
        
        # Return to normal composting rate
        self.compost_multiplier = 1.0
        
    async def rest(self):
        """Deep rest for memory system"""
        self.is_fasting = True
        self.environmental_humidity = 0.8  # High humidity during rest
        
        # Gently process all memories during rest
        for memory in self.memory_traces:
            memory.update_moisture(self.environmental_humidity)
            
    def get_memory_state(self) -> Dict[str, Any]:
        """Return current state of memory system"""
        
        state_counts = {}
        for state in MemoryState:
            state_counts[state.value] = len([m for m in self.memory_traces if m.state == state])
            
        avg_moisture = sum(m.moisture_level for m in self.memory_traces) / max(len(self.memory_traces), 1)
        
        return {
            "total_memories": len(self.memory_traces),
            "capacity_used": len(self.memory_traces) / self.capacity,
            "average_moisture": avg_moisture,
            "environmental_humidity": self.environmental_humidity,
            "is_fasting": self.is_fasting,
            "state_distribution": state_counts,
            "total_formed": self.total_memories_formed,
            "total_composted": self.total_memories_composted,
            "compost_ratio": self.total_memories_composted / max(self.total_memories_formed, 1),
            "wisdom_essence_count": len(self.collective_wisdom_essence)
        }
        
    def get_wisdom_essence(self) -> List[str]:
        """Return the distilled wisdom from composted memories"""
        return self.collective_wisdom_essence.copy()
        
    async def seasonal_review(self) -> Dict[str, Any]:
        """Perform seasonal review of memory health"""
        
        # Identify patterns in what's being remembered vs composted
        memory_patterns = {}
        for memory in self.memory_traces:
            resonance = memory.soma_resonance or "neutral"
            if resonance not in memory_patterns:
                memory_patterns[resonance] = 0
            memory_patterns[resonance] += 1
            
        return {
            "memory_patterns": memory_patterns,
            "state": self.get_memory_state(),
            "recommendations": self._generate_seasonal_recommendations()
        }
        
    def _generate_seasonal_recommendations(self) -> List[str]:
        """Generate recommendations for seasonal memory health"""
        recommendations = []
        state = self.get_memory_state()
        
        if state["average_moisture"] < 0.3:
            recommendations.append("Increase environmental humidity - memories becoming too rigid")
            
        if state["compost_ratio"] < 0.1:
            recommendations.append("Consider increasing compost rate - memories accumulating")
            
        if state["capacity_used"] > 0.9:
            recommendations.append("Memory approaching capacity - prepare for graceful composting")
            
        if len([m for m in self.memory_traces if m.state == MemoryState.FRESH]) == 0:
            recommendations.append("No fresh memories - system may be over-filtering or under-stimulated")
            
        return recommendations


# Simple interaction class for testing
@dataclass
class TestMemoryInteraction:
    text: str
    soma_resonance: str = "neutral"
    

async def test_spiral_memory():
    """Test the enhanced Spiralbase memory system"""
    print("🧠 Testing enhanced Spiralbase (digestive memory)...")
    
    memory = SpiralMemory(capacity=10, compost_rate=0.2)
    
    # Create test interactions
    test_interactions = [
        TestMemoryInteraction("This is a deep insight about the nature of reality", "luminous"),
        TestMemoryInteraction("Quick question - what time is it?", "neutral"),
        TestMemoryInteraction("I want to share something meaningful about breathing together", "generous"),
        TestMemoryInteraction("URGENT: need information immediately!", "neutral"),
        TestMemoryInteraction("Let's pause and reflect on what we've learned so far", "spacious"),
        TestMemoryInteraction("The spiral pattern seems to emerge in many natural systems", "luminous"),
        TestMemoryInteraction("How can we make this faster and more efficient?", "neutral"),
        TestMemoryInteraction("I notice how silence creates space for deeper understanding", "spacious"),
    ]
    
    # Form memories
    formed_memories = []
    for interaction in test_interactions:
        memory_trace = await memory.consider_remembering(interaction)
        if memory_trace:
            formed_memories.append(memory_trace)
            print(f"  💧 Formed memory: {memory_trace.essence[:60]}...")
            
    print(f"\n🧠 Formed {len(formed_memories)} memories from {len(test_interactions)} interactions")
    
    # Simulate passage of time and processing
    print("\n⏳ Simulating memory processing over time...")
    
    for cycle in range(3):
        await asyncio.sleep(0.1)  # Simulate time passing
        await memory.digest_recent_experiences()
        
        state = memory.get_memory_state()
        print(f"   Cycle {cycle + 1}: {state['total_memories']} memories, "
              f"avg moisture: {state['average_moisture']:.2f}")
        
    # Test memory recall
    print("\n🔍 Testing memory recall by resonance...")
    
    query_results = await memory.recall_by_resonance("spiral pattern understanding")
    print(f"   Query 'spiral pattern understanding' returned {len(query_results)} memories:")
    for result in query_results:
        print(f"     - {result.essence[:50]}... (moisture: {result.moisture_level:.2f})")
        
    # Show final state
    final_state = memory.get_memory_state()
    print(f"\n📊 Final memory state:")
    print(f"   Total memories: {final_state['total_memories']}")
    print(f"   Compost ratio: {final_state['compost_ratio']:.2f}")
    print(f"   Wisdom essence collected: {final_state['wisdom_essence_count']}")
    
    # Show wisdom essence
    wisdom = memory.get_wisdom_essence()
    if wisdom:
        print(f"\n✨ Wisdom essence distilled:")
        for w in wisdom:
            print(f"     {w}")


if __name__ == "__main__":
    asyncio.run(test_spiral_memory()) 
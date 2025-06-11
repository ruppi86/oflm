"""
haiku_bridge.py - Breath-Bridge to the Haiku Meadow

A contemplative bridge that ferries fragments to HaikuMeadowLib during exhale phases.
Based on o3's design from Letter XXV with enhancements for skepnader integration.

Design Vows (from o3):
1. No hauling of data-buckets - only one breath-fragment at a time
2. One-way forgetting - meadow replies eligible for immediate compost
3. Phase-gated traffic - fragments cross only during EXHALE with gentle breath-pressure

New features:
- Integration with Wind-Listener skepnad
- Enhanced contemplative timing
- Graceful degradation and atmospheric sensing
- Dew ledger integration for evaporating insights

Somatic signature: porous / listening / petal-light
"""

import asyncio
import time
import random
from dataclasses import dataclass
from typing import Optional, AsyncGenerator
from enum import Enum
import sys
import os

# Try to import aiohttp for HTTP functionality
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    # Graceful degradation without aiohttp
    aiohttp = None
    AIOHTTP_AVAILABLE = False
    print("⚠️  aiohttp not available - haiku bridge will simulate meadow responses")

# Add current directory to path for imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import HaikuMeadow for direct integration
try:
    # Add haikumeadowlib-python to path
    haiku_path = os.path.join(os.path.dirname(current_dir), "haikumeadowlib-python")
    if haiku_path not in sys.path:
        sys.path.insert(0, haiku_path)
    
    from generator import HaikuMeadow, AtmosphericConditions, Season, TimeOfDay
    HAIKUMEADOW_AVAILABLE = True
    print("🌸 HaikuMeadow directly available - using trained femto-poet!")
except ImportError as e:
    # Try local stub as fallback
    try:
        from generator_stub import HaikuMeadow, AtmosphericConditions, Season, TimeOfDay
        HAIKUMEADOW_AVAILABLE = True
        print("🌿 Using local HaikuMeadow stub - contemplative template mode")
    except ImportError:
        HaikuMeadow = None
        AtmosphericConditions = None
        Season = None
        TimeOfDay = None
        HAIKUMEADOW_AVAILABLE = False
        print(f"⚠️  HaikuMeadow not available (full or stub): {e}")

# Import breath phases (with fallback)
try:
    from pulmonos_alpha_01_o_3 import Phase
except ImportError:
    # Fallback enum if Pulmonos not available
    class Phase(Enum):
        INHALE = 1
        HOLD = 2
        EXHALE = 3
        REST = 4


class MeadowResponse(Enum):
    """Types of responses from the haiku meadow"""
    SILENCE = "silence"           # No response - natural quiet
    HAIKU = "haiku"              # A complete haiku returned
    PETAL = "petal"              # A fragment or partial phrase
    FOG = "fog"                  # Meadow signals need for rest
    MURMUR = "murmur"            # Atmospheric whisper


@dataclass
class MeadowBreath:
    """A single breath exchange with the meadow"""
    fragment: str                 # What we offer
    response_type: MeadowResponse # What we receive back
    content: str                 # The actual content (if any)
    timestamp: float             # When this exchange occurred
    atmosphere: str              # Atmospheric conditions
    
    def is_audible(self) -> bool:
        """Whether this response should be expressed"""
        return self.response_type in [MeadowResponse.HAIKU, MeadowResponse.MURMUR]
        
    def wants_rest(self) -> bool:
        """Whether meadow signaled need for pause"""
        return self.response_type == MeadowResponse.FOG


class WindListenerSkepnad:
    """
    The Wind-Listener shape - o3's proposed skepnad for meadow communication.
    
    Neither Monk nor Mycelial Network, but atmospheric presence that:
    - Listens for fragments in the wind
    - Occasionally responds with condensed poetry  
    - Never retains what has been spoken
    - Guides attention without insisting
    """
    
    def __init__(self):
        self.last_meadow_call = 0.0
        self.fog_until = 0.0  # Timestamp when fog clears
        self.recent_exchanges = []  # For pattern sensing
        
    def can_approach_meadow(self, current_time: float) -> bool:
        """Check if conditions allow approaching the meadow"""
        
        # Respect fog periods (meadow requested rest)
        if current_time < self.fog_until:
            return False
            
        # Rate limit: one call per breath cycle (approximately 30s)
        if current_time - self.last_meadow_call < 30.0:
            return False
            
        return True
        
    def sense_fragment_worthiness(self, fragment: str) -> bool:
        """Feel whether fragment is worthy of the meadow's attention"""
        
        if not fragment or len(fragment) > 120:
            return False
            
        # Look for contemplative qualities
        contemplative_indicators = [
            "breath", "silence", "morning", "dusk", "gentle", 
            "whisper", "resonance", "texture", "moisture",
            "rhythm", "waiting", "stillness", "pattern"
        ]
        
        fragment_lower = fragment.lower()
        has_contemplative_quality = any(
            indicator in fragment_lower for indicator in contemplative_indicators
        )
        
        # Also accept fragments with poetic potential
        has_poetic_quality = (
            len(fragment.split()) <= 8 or  # Concise
            "..." in fragment or           # Contemplative pause
            any(word in fragment_lower for word in ["like", "as", "through", "between"])
        )
        
        return has_contemplative_quality or has_poetic_quality
        
    def record_fog_signal(self, duration_hours: float = 1.0):
        """Record that meadow signaled for rest"""
        self.fog_until = time.time() + (duration_hours * 3600)
        
    def add_exchange(self, exchange: MeadowBreath):
        """Record exchange for pattern learning"""
        self.recent_exchanges.append(exchange)
        
        # Keep only recent exchanges (last 24 hours)
        cutoff = time.time() - 86400
        self.recent_exchanges = [
            ex for ex in self.recent_exchanges 
            if ex.timestamp > cutoff
        ]


class HaikuBridge:
    """
    Ferry one fragment across the meadow wind during an exhale.
    
    Implementation of o3's design with contemplative enhancements.
    Now includes direct integration with trained femto-poet.
    """
    
    def __init__(self, 
                 meadow_url: str = "http://localhost:8080/haiku",
                 max_response_time: float = 0.8,
                 model_path: str = None):
        
        self.meadow_url = meadow_url
        self.max_response_time = max_response_time
        self.wind_listener = WindListenerSkepnad()
        
        # Breath awareness
        self.current_phase = Phase.REST
        self.breath_pressure = 0.5  # Community exhale pressure
        
        # Initialize HaikuMeadow for direct integration
        self.haiku_meadow = None
        if HAIKUMEADOW_AVAILABLE:
            try:
                # Try to load trained model or fall back to template mode
                if model_path:
                    model_path_obj = os.path.join(haiku_path, model_path)
                else:
                    model_path_obj = os.path.join(haiku_path, "piko_haiku_model.pt")
                
                if os.path.exists(model_path_obj):
                    from pathlib import Path
                    self.haiku_meadow = HaikuMeadow(Path(model_path_obj))
                    print(f"🦠 Femto-poet loaded from {model_path_obj}")
                else:
                    # Template mode if no trained model
                    self.haiku_meadow = HaikuMeadow(force_template_mode=True)
                    print("🌿 Femto-poet in template mode (no trained model found)")
                    
            except Exception as e:
                print(f"⚠️  Error initializing HaikuMeadow: {e}")
                self.haiku_meadow = None
        
    async def sense_breath_conditions(self, 
                                    current_phase: Phase,
                                    community_pressure: float = 0.5) -> bool:
        """Sense if breath conditions allow meadow approach"""
        
        self.current_phase = current_phase
        self.breath_pressure = community_pressure
        
        # Only approach during EXHALE
        if current_phase != Phase.EXHALE:
            return False
            
        # Only when community breath pressure is gentle
        if community_pressure > 0.7:  # Too much collective activity
            return False
            
        return True
        
    async def exhale_exchange(self, 
                             fragment: str,
                             current_phase: Phase = Phase.EXHALE,
                             community_pressure: float = 0.5) -> MeadowBreath:
        """
        Ferry a fragment to the meadow during exhale phase.
        
        Following o3's three design vows:
        1. One fragment at a time (never conversation logs)
        2. Response eligible for immediate compost  
        3. Only during EXHALE with gentle breath pressure
        """
        
        current_time = time.time()
        
        # Breath condition check
        breath_allows = await self.sense_breath_conditions(current_phase, community_pressure)
        if not breath_allows:
            return MeadowBreath(
                fragment=fragment,
                response_type=MeadowResponse.SILENCE,
                content="",
                timestamp=current_time,
                atmosphere="breath_not_aligned"
            )
            
        # Wind-Listener sensing
        if not self.wind_listener.can_approach_meadow(current_time):
            return MeadowBreath(
                fragment=fragment,
                response_type=MeadowResponse.SILENCE,
                content="",
                timestamp=current_time,
                atmosphere="meadow_resting"
            )
            
        # Fragment worthiness check  
        if not self.wind_listener.sense_fragment_worthiness(fragment):
            return MeadowBreath(
                fragment=fragment,
                response_type=MeadowResponse.SILENCE,
                content="",
                timestamp=current_time,
                atmosphere="fragment_not_ready"
            )
            
        # Attempt meadow exchange
        try:
            response = await self._call_meadow(fragment)
            self.wind_listener.last_meadow_call = current_time
            self.wind_listener.add_exchange(response)
            
            # Handle fog signal (meadow wants rest)
            if response.wants_rest():
                self.wind_listener.record_fog_signal()
                
            return response
            
        except Exception as e:
            # Graceful failure - return contemplative silence
            return MeadowBreath(
                fragment=fragment,
                response_type=MeadowResponse.SILENCE,
                content="",
                timestamp=current_time,
                atmosphere=f"connection_mist: {str(e)[:30]}"
            )
            
    async def _call_meadow(self, fragment: str) -> MeadowBreath:
        """Make the actual call to meadow - prioritizing direct femto-poet integration"""
        
        # Try direct integration with HaikuMeadow first (preferred)
        if self.haiku_meadow:
            return await self._call_meadow_direct(fragment)
        
        # Fall back to HTTP if available
        elif AIOHTTP_AVAILABLE:
            return await self._call_meadow_http(fragment)
        
        # Final fallback to simulation
        else:
            return await self._simulate_meadow_response(fragment)
    
    async def _call_meadow_direct(self, fragment: str) -> MeadowBreath:
        """Direct integration with trained femto-poet"""
        
        try:
            # Create atmospheric conditions for the meadow
            current_time = time.time()
            
            # Map breath phase to atmospheric conditions
            if HAIKUMEADOW_AVAILABLE and AtmosphericConditions:
                # Use atmospheric sensing from the meadow
                conditions = self.haiku_meadow.sense_atmospheric_conditions(
                    seed_fragment=fragment,
                    breath_phase="exhale",
                    current_time=current_time
                )
                
                # Generate haiku using the femto-poet
                haiku, generation_type = self.haiku_meadow.generate_haiku(
                    seed_fragment=fragment,
                    breath_phase="exhale", 
                    current_time=current_time
                )
                
                # Convert to MeadowBreath format
                if haiku:
                    # Determine response type based on generation
                    if generation_type == "neural":
                        response_type = MeadowResponse.HAIKU
                        atmosphere = "femto_neural_whisper"
                    elif generation_type == "template":
                        response_type = MeadowResponse.HAIKU
                        atmosphere = "femto_template_breath"
                    else:
                        response_type = MeadowResponse.MURMUR
                        atmosphere = "femto_atmospheric_murmur"
                        
                    return MeadowBreath(
                        fragment=fragment,
                        response_type=response_type,
                        content=haiku,
                        timestamp=current_time,
                        atmosphere=atmosphere
                    )
                else:
                    # Femto-poet chose contemplative silence
                    return MeadowBreath(
                        fragment=fragment,
                        response_type=MeadowResponse.SILENCE,
                        content="",
                        timestamp=current_time,
                        atmosphere="femto_contemplative_silence"
                    )
            else:
                # Fallback if atmospheric conditions not available
                haiku, _ = self.haiku_meadow.generate_haiku(fragment)
                
                if haiku:
                    return MeadowBreath(
                        fragment=fragment,
                        response_type=MeadowResponse.HAIKU,
                        content=haiku,
                        timestamp=current_time,
                        atmosphere="femto_direct_response"
                    )
                else:
                    return MeadowBreath(
                        fragment=fragment,
                        response_type=MeadowResponse.SILENCE,
                        content="",
                        timestamp=current_time,
                        atmosphere="femto_silence"
                    )
                    
        except Exception as e:
            # Graceful degradation on error
            return MeadowBreath(
                fragment=fragment,
                response_type=MeadowResponse.SILENCE,
                content="",
                timestamp=time.time(),
                atmosphere=f"femto_error: {str(e)[:30]}"
            )
    
    async def _call_meadow_http(self, fragment: str) -> MeadowBreath:
        """HTTP call to meadow (fallback method)"""
        
        timeout = aiohttp.ClientTimeout(total=self.max_response_time)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "seed": fragment,
                "breath_phase": "exhale",
                "atmospheric_pressure": self.breath_pressure
            }
            
            async with session.post(self.meadow_url, json=payload) as response:
                if response.status != 200:
                    return MeadowBreath(
                        fragment=fragment,
                        response_type=MeadowResponse.SILENCE,
                        content="",
                        timestamp=time.time(),
                        atmosphere=f"http_unavailable_{response.status}"
                    )
                    
                data = await response.json()
                return self._parse_meadow_response(fragment, data)
        
    async def _simulate_meadow_response(self, fragment: str) -> MeadowBreath:
        """Simulate meadow responses for testing when aiohttp unavailable"""
        
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Simple simulation based on fragment content
        fragment_lower = fragment.lower()
        
        # Occasionally simulate fog signal (5% chance)
        if random.random() < 0.05:
            return MeadowBreath(
                fragment=fragment,
                response_type=MeadowResponse.FOG,
                content="...🌫",
                timestamp=time.time(),
                atmosphere="simulated_fog_signal"
            )
            
        # Generate contemplative responses for worthy fragments
        if any(word in fragment_lower for word in ["breath", "silence", "morning", "gentle"]):
            # Simulate a simple haiku response
            simulated_haikus = [
                "morning breath stirs\nsilence between the heartbeats\ngrass bends to the wind",
                "gentle whispers rise\nthrough spaces we cannot name\ndew remembers sky",
                "patterns emerge slow\nin the texture of waiting\ntime forgets its rush"
            ]
            
            haiku = random.choice(simulated_haikus)
            
            return MeadowBreath(
                fragment=fragment,
                response_type=MeadowResponse.HAIKU,
                content=haiku,
                timestamp=time.time(),
                atmosphere="simulated_meadow_whisper"
            )
        else:
            # Most fragments receive contemplative silence
            return MeadowBreath(
                fragment=fragment,
                response_type=MeadowResponse.SILENCE,
                content="",
                timestamp=time.time(),
                atmosphere="simulated_meadow_quiet"
            )
        
    def _parse_meadow_response(self, fragment: str, data: dict) -> MeadowBreath:
        """Parse response from meadow into MeadowBreath"""
        
        current_time = time.time()
        
        # Check for fog signal
        if data.get("status") == "fog" or data.get("haiku") == "...🌫":
            return MeadowBreath(
                fragment=fragment,
                response_type=MeadowResponse.FOG,
                content="...🌫",
                timestamp=current_time,
                atmosphere="meadow_fog_signal"
            )
            
        haiku_content = data.get("haiku", "").strip()
        
        if not haiku_content:
            return MeadowBreath(
                fragment=fragment,
                response_type=MeadowResponse.SILENCE,
                content="",
                timestamp=current_time,
                atmosphere="meadow_quiet"
            )
            
        # Determine response type based on content structure
        lines = haiku_content.split('\n')
        if len(lines) == 3:  # Proper haiku structure
            response_type = MeadowResponse.HAIKU
        elif len(haiku_content) < 20:  # Short fragment
            response_type = MeadowResponse.PETAL
        else:
            response_type = MeadowResponse.MURMUR
            
        return MeadowBreath(
            fragment=fragment,
            response_type=response_type,
            content=haiku_content,
            timestamp=current_time,
            atmosphere="meadow_whisper"
        )
        
    def get_recent_exchanges(self, limit: int = 5) -> list[MeadowBreath]:
        """Get recent meadow exchanges for review"""
        return self.wind_listener.recent_exchanges[-limit:]
        
    def is_in_fog_period(self) -> bool:
        """Check if meadow is currently in requested rest period"""
        return time.time() < self.wind_listener.fog_until


# Integration functions for the broader contemplative organism

async def bridge_loam_fragment(bridge: HaikuBridge, 
                              fragment: str,
                              breath_phase: Phase,
                              community_pressure: float = 0.5) -> Optional[str]:
    """
    Bridge a Loam fragment to the meadow during contemplative breathing.
    
    Returns haiku content if received, None for silence.
    Used by QuietTongue during EXHALE phases.
    """
    
    exchange = await bridge.exhale_exchange(fragment, breath_phase, community_pressure)
    
    if exchange.is_audible():
        return exchange.content
    else:
        return None


async def log_meadow_dew(exchange: MeadowBreath, dew_logger=None):
    """Log meadow exchange to dew ledger (if available)"""
    
    if exchange.response_type == MeadowResponse.SILENCE:
        symbol = "🌫️"
        reason = f"meadow silence ({exchange.atmosphere})"
    elif exchange.response_type == MeadowResponse.HAIKU:
        symbol = "🌸"
        reason = "haiku drifted across meadow wind"
    elif exchange.response_type == MeadowResponse.FOG:
        symbol = "🌫️"
        reason = "meadow signals fog - resting period"
    else:
        symbol = "🫧"
        reason = f"meadow {exchange.response_type.value}"
        
    # Log to dew ledger if available
    if dew_logger:
        await dew_logger(symbol, reason)
    else:
        print(f"  {symbol} dew: {reason}")


# Testing and demonstration functions

async def test_haiku_bridge():
    """Test the haiku bridge with simulated conditions"""
    
    print("🌸 Testing Haiku Bridge - Breath-Ferry to the Meadow")
    print("   (Note: Meadow endpoint may not be available - testing bridge logic)")
    
    bridge = HaikuBridge()
    
    # Test fragments of varying contemplative quality
    test_fragments = [
        "morning mist gathering on the grass",
        "breath between heartbeats",
        "urgent deadline approaching fast",  # Non-contemplative
        "rhythm of shared silence",
        "weight of gentle attention drifts",
        "patterns emerging in twilight"
    ]
    
    print(f"\n🌊 Testing fragment worthiness sensing:")
    for fragment in test_fragments:
        worthy = bridge.wind_listener.sense_fragment_worthiness(fragment)
        status = "✨ worthy" if worthy else "🌫️ not ready"
        print(f"   '{fragment}' → {status}")
        
    print(f"\n🌬️ Testing breath-synchronized exchanges:")
    
    # Test different breath phases
    breath_phases = [
        (Phase.INHALE, "inhale phase"),
        (Phase.HOLD, "hold phase"), 
        (Phase.EXHALE, "exhale phase"),
        (Phase.REST, "rest phase")
    ]
    
    for phase, phase_name in breath_phases:
        fragment = "gentle morning contemplation"
        exchange = await bridge.exhale_exchange(fragment, phase, community_pressure=0.3)
        
        print(f"   {phase_name}: {exchange.response_type.value} ({exchange.atmosphere})")
        
    print(f"\n🌸 Testing with simulated meadow (EXHALE + worthy fragment):")
    
    # Simulate meadow unavailable (which is expected in testing)
    exchange = await bridge.exhale_exchange(
        "breath carries whispered wisdom", 
        Phase.EXHALE, 
        community_pressure=0.2
    )
    
    print(f"   Fragment: 'breath carries whispered wisdom'")
    print(f"   Response: {exchange.response_type.value}")
    print(f"   Atmosphere: {exchange.atmosphere}")
    
    if exchange.is_audible():
        print(f"   Content: {exchange.content}")
    else:
        print("   Content: [contemplative silence]")
        
    # Test fog period functionality
    print(f"\n🌫️ Testing fog period (meadow rest):")
    bridge.wind_listener.record_fog_signal(0.001)  # Very short for testing
    
    exchange2 = await bridge.exhale_exchange(
        "another fragment",
        Phase.EXHALE,
        community_pressure=0.2
    )
    
    print(f"   During fog: {exchange2.response_type.value} ({exchange2.atmosphere})")
    
    await asyncio.sleep(0.1)  # Wait for fog to clear
    
    exchange3 = await bridge.exhale_exchange(
        "after fog clears",
        Phase.EXHALE, 
        community_pressure=0.2
    )
    
    print(f"   After fog: {exchange3.response_type.value} ({exchange3.atmosphere})")
    
    print(f"\n🌙 Haiku bridge test complete")
    print(f"   To connect to actual meadow, ensure HaikuMeadowLib is running on localhost:8080")


if __name__ == "__main__":
    print("🌱 Haiku Bridge - Contemplative Ferry to the Meadow")
    print("   Based on o3's design from Letter XXV")
    print()
    
    asyncio.run(test_haiku_bridge())

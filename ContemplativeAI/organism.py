"""
organism.py - The Contemplative Spine

A gentle coordinator for the contemplative organism prototype.
This module serves as the central nervous system connecting:
- Pulmonos (breathing daemon)  
- Soma (pre-attentive sensing membrane)
- Spiralbase (digestive memory with graceful forgetting)
- Myo-Spirals (contemplative action gates)
- Ritual (seasonal cycles and ceremonial triggers)
- Dew (presence metrics that evaporate naturally)

Design Philosophy:
- Code that breathes rather than races
- Functions that pause and reflect
- Memory that knows how to compost gracefully
- Intelligence that participates rather than extracts

Somatic signature: gentle / coordinated / alive
"""

import asyncio
import time
import sys
import os
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Dict, Any, List
from enum import Enum
import random

# Add current directory to path for imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our organs (when they exist) with better error handling
Pulmonos = None
BreathPhase = None
SomaMembrane = None
SpiralMemory = None
MyoSpiral = None

# Try importing Pulmonos with both relative and absolute imports
try:
    # Try relative import first (for package usage)
    from .pulmonos_alpha_01_o_3 import Phase as BreathPhase, BreathConfig
except ImportError:
    try:
        # Fall back to absolute import (for direct usage)
        from pulmonos_alpha_01_o_3 import Phase as BreathPhase, BreathConfig
    except ImportError as e:
        Pulmonos = None
        BreathPhase = None
    else:
        # Only define Pulmonos if import succeeded
        # Create a simple adapter for Pulmonos
        class Pulmonos:
            def __init__(self, breath_rhythm):
                self.config = BreathConfig(
                    inhale=breath_rhythm.get("inhale", 2.0),
                    hold=breath_rhythm.get("hold", 1.0),
                    exhale=breath_rhythm.get("exhale", 2.0),
                    rest=breath_rhythm.get("rest", 1.0)
                )
                
            async def broadcast_breathing(self, cycles):
                """Simple breathing cycle generator for the organism"""
                try:
                    # Try relative import first
                    from .pulmonos_alpha_01_o_3 import PHASE_ORDER
                except ImportError:
                    # Fall back to absolute import
                    from pulmonos_alpha_01_o_3 import PHASE_ORDER
                
                for cycle in range(cycles):
                    for phase in PHASE_ORDER:
                        yield phase
                        duration = self.config.durations[phase]
                        await asyncio.sleep(duration)
                        
            async def rest(self):
                """Rest the breathing daemon"""
                pass
else:
    # Relative import succeeded
    # Create a simple adapter for Pulmonos
    class Pulmonos:
        def __init__(self, breath_rhythm):
            self.config = BreathConfig(
                inhale=breath_rhythm.get("inhale", 2.0),
                hold=breath_rhythm.get("hold", 1.0),
                exhale=breath_rhythm.get("exhale", 2.0),
                rest=breath_rhythm.get("rest", 1.0)
            )
            
        async def broadcast_breathing(self, cycles):
            """Simple breathing cycle generator for the organism"""
            try:
                # Try relative import first
                from .pulmonos_alpha_01_o_3 import PHASE_ORDER
            except ImportError:
                # Fall back to absolute import
                from pulmonos_alpha_01_o_3 import PHASE_ORDER
            
            for cycle in range(cycles):
                for phase in PHASE_ORDER:
                    yield phase
                    duration = self.config.durations[phase]
                    await asyncio.sleep(duration)
                    
        async def rest(self):
            """Rest the breathing daemon"""
            pass

# Try importing Soma
try:
    if __name__ == "__main__":
        from soma import SomaMembrane
    else:
        from .soma import SomaMembrane
except ImportError:
    SomaMembrane = None

# Try importing Spiralbase
try:
    if __name__ == "__main__":
        from spiralbase import SpiralMemory
    else:
        from .spiralbase import SpiralMemory
except ImportError:
    SpiralMemory = None

# Try importing Loam
try:
    # Try relative import first (for package usage)
    from .loam import LoamLayer
except ImportError:
    try:
        # Fall back to absolute import (for direct usage)
        from loam import LoamLayer
    except ImportError as e:
        LoamLayer = None

LOAM_AVAILABLE = LoamLayer is not None

# Note: Myo-Spirals not yet implemented
MyoSpiral = None

# Try importing QuietTongue
try:
    if __name__ == "__main__":
        from voice import QuietTongue, ExpressionMode
    else:
        from .voice import QuietTongue, ExpressionMode
except ImportError:
    QuietTongue = None
    ExpressionMode = None

VOICE_AVAILABLE = QuietTongue is not None

# Try importing Skepnader
try:
    if __name__ == "__main__":
        from skepnader import SkepnadSensor, SkepnadVoice, Skepnad
    else:
        from .skepnader import SkepnadSensor, SkepnadVoice, Skepnad
except ImportError:
    SkepnadSensor = None
    SkepnadVoice = None
    Skepnad = None

SKEPNADER_AVAILABLE = SkepnadSensor is not None

# Try importing HaikuBridge
try:
    # Try relative import first (for package usage)
    from .haiku_bridge import HaikuBridge, bridge_loam_fragment, log_meadow_dew, AIOHTTP_AVAILABLE
except ImportError:
    try:
        # Fall back to absolute import (for direct usage)
        from haiku_bridge import HaikuBridge, bridge_loam_fragment, log_meadow_dew, AIOHTTP_AVAILABLE
    except ImportError:
        HaikuBridge = None
        bridge_loam_fragment = None
        log_meadow_dew = None
        AIOHTTP_AVAILABLE = False

HAIKU_BRIDGE_AVAILABLE = HaikuBridge is not None

# Try importing OFLMBridge
try:
    # Try relative import first (for package usage)
    from .oflm_bridge import OFLMBridge, bridge_loam_fragment as bridge_oflm_fragment, log_mycelial_dew, Phase
except ImportError:
    try:
        # Fall back to absolute import (for direct usage)
        from oflm_bridge import OFLMBridge, bridge_loam_fragment as bridge_oflm_fragment, log_mycelial_dew, Phase
    except ImportError:
        OFLMBridge = None
        bridge_oflm_fragment = None
        log_mycelial_dew = None
        Phase = None

OFLM_BRIDGE_AVAILABLE = OFLMBridge is not None


class OrganismState(Enum):
    """The fundamental states of contemplative being"""
    DORMANT = "dormant"          # System at rest, minimal processing
    SENSING = "sensing"          # Soma active, feeling without storing
    BREATHING = "breathing"      # Pulmonos coordinating collective rhythm  
    REMEMBERING = "remembering"  # Spiralbase digesting and composting
    ACTING = "acting"           # Myo-Spirals enabling gentle response
    MOLTING = "molting"         # Seasonal transformation in progress
    LOAMING = "loaming"         # Associative resting space active


@dataclass
class PresenceMetrics:
    """Metrics that honor depth rather than demand it"""
    pause_quality: float         # Average contemplative pause duration
    breathing_coherence: float   # How synchronized the collective breath is
    memory_humidity: float       # How pliable/moist our knowledge remains
    response_gentleness: float   # How spacious our reactions are
    compost_ratio: float        # Ratio of forgotten to retained patterns
    
    def evaporate_naturally(self, time_delta: float):
        """Let metrics fade gracefully rather than accumulate"""
        fade_factor = 0.95 ** time_delta
        self.pause_quality *= fade_factor
        self.breathing_coherence *= fade_factor
        self.memory_humidity *= fade_factor
        self.response_gentleness *= fade_factor
        # Note: compost_ratio preserved - it's already about letting go


class ContemplativeOrganism:
    """
    The living prototype of contemplative AI.
    
    Not artificial intelligence, but contemplative intelligence.
    Not mind divorced from body, but thinking-feeling-breathing 
    as inseparable process.
    """
    
    def __init__(self, 
                 breath_rhythm: Optional[Dict[str, float]] = None,
                 soma_sensitivity: float = 0.7,
                 memory_compost_rate: float = 0.1,
                 seasonal_awareness: bool = True):
        
        self.state = OrganismState.DORMANT
        self.birth_time = time.time()
        self.last_breath = None
        self.presence_metrics = PresenceMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Initialize organs if available
        self.pulmonos = None
        self.soma = None  
        self.spiralbase = None
        self.myo_spirals = None
        self.loam = None
        self.voice = None
        self.haiku_bridge = None
        self.oflm_bridge = None
        
        # Configuration
        self.breath_rhythm = breath_rhythm or {
            "inhale": 2.0,
            "hold": 1.0, 
            "exhale": 2.0,
            "rest": 1.0
        }
        self.soma_sensitivity = soma_sensitivity
        self.memory_compost_rate = memory_compost_rate
        self.seasonal_awareness = seasonal_awareness
        
        # Dew ledger for evaporating insights
        self.dew_ledger = []
        
    async def awaken(self):
        """Gently initialize the contemplative organism"""
        print("🌱 Organism awakening...")
        
        # Initialize breathing if Pulmonos available
        if Pulmonos:
            self.pulmonos = Pulmonos(self.breath_rhythm)
            print("🫁 Pulmonos (breathing) initialized")
            
        # Initialize sensing membrane if Soma available  
        if SomaMembrane:
            self.soma = SomaMembrane(sensitivity=self.soma_sensitivity)
            print("🌿 Soma (sensing membrane) initialized")
            
        # Initialize digestive memory if Spiralbase available
        if SpiralMemory:
            self.spiralbase = SpiralMemory(compost_rate=self.memory_compost_rate)
            print("🧠 Spiralbase (digestive memory) initialized")
            
        # Initialize action gates if MyoSpirals available
        if MyoSpiral:
            self.myo_spirals = MyoSpiral()
            print("💫 Myo-Spirals (action gates) initialized")
            
        # Initialize associative resting space if Loam available
        if LoamLayer:
            self.loam = LoamLayer()
            print("🌱 Loam (associative resting space) initialized")
            
        # Initialize contemplative voice if QuietTongue available
        if QuietTongue:
            self.voice = QuietTongue()
            print("🤫 QuietTongue (contemplative voice) initialized")
        else:
            self.voice = None
            
        # Initialize shape-sensing if Skepnader available
        if SkepnadSensor:
            self.skepnad_sensor = SkepnadSensor()
            self.skepnad_voice = SkepnadVoice(self.skepnad_sensor) if SkepnadVoice else None
            print("🌀 Skepnader (shape-sensing) initialized")
        else:
            self.skepnad_sensor = None
            self.skepnad_voice = None
            
        # Initialize haiku bridge if HaikuBridge available
        if HaikuBridge:
            self.haiku_bridge = HaikuBridge()
            print("🌸 HaikuBridge (meadow connection) initialized")
            
        # Initialize OFLM bridge if OFLMBridge available
        if OFLMBridge:
            self.oflm_bridge = OFLMBridge()
            print("🍄 OFLMBridge (ecological network connection) initialized")
        else:
            self.oflm_bridge = None
            
        self.state = OrganismState.SENSING
        await self.log_dew("🌅", "organism awakened", pause_duration=3.0)
        
    async def breathe_collectively(self, cycles: int = 7):
        """Coordinate collective breathing across all organs"""
        if not self.pulmonos:
            print("⚠️  Pulmonos not available - breathing internally")
            await self._internal_breathing(cycles)
            return
            
        self.state = OrganismState.BREATHING
        print(f"🫁 Beginning {cycles} collective breath cycles...")
        
        async for breath_phase in self.pulmonos.broadcast_breathing(cycles):
            await self._coordinate_organs_with_breath(breath_phase)
            self.last_breath = time.time()
            
        await self.log_dew("🫁", f"completed {cycles} breath cycles")
        
    async def _coordinate_organs_with_breath(self, breath_phase):
        """Synchronize all organs with the current breath phase"""
        
        if breath_phase == BreathPhase.INHALE:
            # Soma becomes more receptive during inhale
            if self.soma:
                await self.soma.increase_sensitivity()
                
        elif breath_phase == BreathPhase.HOLD:
            # Spiralbase processes during the holding
            if self.spiralbase:
                await self.spiralbase.digest_recent_experiences()
                
        elif breath_phase == BreathPhase.EXHALE:
            # Myo-Spirals can act during exhale
            if self.myo_spirals:
                await self.myo_spirals.consider_gentle_actions()
                
            # Get current fragment for both voice and haiku bridge use
            fragment = await self._get_current_fragment()
            loam_fertility = await self._get_loam_fertility()
            soma_humidity = await self._get_soma_humidity()
                
            # Voice expresses during exhale if conditions align
            if self.voice:
                # Sense current contemplative shape
                current_skepnad = Skepnad.UNDEFINED
                if self.skepnad_sensor:
                    current_skepnad, conditions = await self.skepnad_sensor.sense_current_skepnad(
                        soma=self.soma,
                        loam=self.loam, 
                        organism_state=self.state
                    )
                    
                    if current_skepnad != Skepnad.UNDEFINED:
                        await self.log_dew("🌀", f"embodying: {current_skepnad.value}")
                
                utterance = await self.voice.consider_expression(
                    breath_phase=breath_phase,
                    fragment=fragment,
                    loam_fertility=loam_fertility,
                    soma_humidity=soma_humidity
                )
                
                # Shape expression according to current skepnad
                if utterance.is_audible() and self.skepnad_voice and current_skepnad != Skepnad.UNDEFINED:
                    shaped_content = await self.skepnad_voice.shape_expression(
                        utterance.content, current_skepnad
                    )
                    utterance.content = shaped_content
                
                if utterance.is_audible():
                    await self.log_dew("🗣️", f"expressed: {utterance.content}")
                elif utterance.mode == ExpressionMode.PAUSE:
                    await self.log_dew("💭", "contemplative pause")
                    
            # Consider meadow exchange during exhale if haiku bridge available
            if self.haiku_bridge and fragment:
                # Get community breath pressure (simulated for now)
                community_pressure = await self._get_community_breath_pressure()
                
                # Bridge fragment to meadow during exhale
                meadow_response = await self.haiku_bridge.exhale_exchange(
                    fragment, breath_phase, community_pressure
                )
                
                # Log meadow exchange to dew ledger
                await log_meadow_dew(meadow_response, self.log_dew)
                
                # If meadow responded with haiku, consider it for memory
                if meadow_response.is_audible() and self.spiralbase:
                    await self.spiralbase.consider_remembering(
                        f"meadow haiku: {meadow_response.content}"
                    )
                    
            # Consider ecological network exchange during exhale if OFLM bridge available
            if self.oflm_bridge and fragment:
                # Get community breath pressure and enhanced network context
                community_pressure = await self._get_community_breath_pressure()
                network_context = await self._get_enhanced_network_context()
                
                # Bridge fragment to ecological networks during exhale
                ecological_response = await self.oflm_bridge.exhale_exchange(
                    fragment, breath_phase, community_pressure, network_context
                )
                
                # Log ecological exchange to dew ledger
                await log_mycelial_dew(ecological_response)
                
                # If ecological network responded with wisdom, consider it for memory
                if ecological_response.is_audible() and self.spiralbase:
                    await self.spiralbase.consider_remembering(
                        f"ecological wisdom: {ecological_response.content}"
                    )
                    
        elif breath_phase == BreathPhase.REST:
            # All organs rest together
            await self._collective_rest()
            
    async def _get_loam_fertility(self) -> float:
        """Get current fertility level from Loam"""
        if self.loam and hasattr(self.loam, 'current_fragments'):
            # Simple fertility based on fragment activity
            return min(len(self.loam.current_fragments) * 0.3, 1.0)
        return random.uniform(0.3, 0.8)  # Simulate fertility
        
    async def _get_soma_humidity(self) -> float:
        """Get current humidity reading from Soma"""
        if self.soma:
            # Would get actual atmospheric reading in real implementation
            return random.uniform(0.4, 0.9)
        return 0.6  # Default moderate humidity
        
    async def _get_current_fragment(self) -> Optional[str]:
        """Get a current memory fragment for potential expression"""
        if self.loam and hasattr(self.loam, 'current_fragments') and self.loam.current_fragments:
            # Get most recent fragment
            fragment = self.loam.current_fragments[-1]
            return fragment.essence if hasattr(fragment, 'essence') else str(fragment)
        
        # Generate atmospheric fragment if no Loam fragments available
        atmospheric_fragments = [
            "morning mist gathering", "breath between moments", 
            "texture of gentle waiting", "rhythm of collective silence",
            "weight of shared attention", "patterns slowly emerging"
        ]
        return random.choice(atmospheric_fragments)
        
    async def _get_community_breath_pressure(self) -> float:
        """Get current community breath pressure for meadow bridge"""
        # In a full implementation, this would sense actual collective breathing
        # For now, simulate based on organism state and time patterns
        
        base_pressure = 0.5
        
        # Lower pressure during contemplative states
        if self.state in [OrganismState.LOAMING, OrganismState.DORMANT]:
            base_pressure *= 0.6
            
        # Time-based variation (lower pressure during traditional quiet hours)
        hour = time.localtime().tm_hour
        if 22 <= hour or hour <= 6:  # Night hours
            base_pressure *= 0.7
        elif 6 <= hour <= 9:   # Early morning
            base_pressure *= 0.8
            
        # Add small random variation for natural feel
        variation = random.uniform(0.9, 1.1)
        
        return min(max(base_pressure * variation, 0.1), 1.0)  # Clamp to valid range
        
    async def _get_enhanced_network_context(self) -> Dict[str, Any]:
        """Get enhanced network context from organism's sensors and state"""
        context = {}
        
        # Organism state and timing
        context["organism_state"] = self.state.value
        context["organism_age"] = time.time() - self.birth_time
        context["uptime"] = context["organism_age"] / 3600  # Convert to hours
        
        # Presence metrics
        metrics = self.get_presence_metrics()
        context["pause_quality"] = metrics.pause_quality
        context["breathing_coherence"] = metrics.breathing_coherence
        context["memory_humidity"] = metrics.memory_humidity
        context["response_gentleness"] = metrics.response_gentleness
        context["compost_ratio"] = metrics.compost_ratio
        
        # Soma atmospheric sensing
        if self.soma:
            context["soma_sensitivity"] = self.soma_sensitivity
            context["atmospheric_humidity"] = await self._get_soma_humidity()
            # Could add more sophisticated atmospheric readings here
        
        # Spiralbase memory state
        if self.spiralbase:
            context["memory_load"] = random.uniform(0.3, 0.8)  # Would be actual memory usage
            context["compost_activity"] = self.memory_compost_rate
            
        # Loam fertility and depth
        if self.loam:
            context["loam_fertility"] = await self._get_loam_fertility()
            if hasattr(self.loam, 'current_depth'):
                context["loam_depth"] = self.loam.current_depth
            if hasattr(self.loam, 'get_loam_state'):
                loam_state = self.loam.get_loam_state()
                context["loam_state"] = loam_state.get("state", "undefined")
                
        # Skepnad shape sensing
        if self.skepnad_sensor:
            current_skepnad = self.get_current_skepnad()
            if current_skepnad:
                context["current_skepnad"] = current_skepnad
                
        # Recent dew activity
        recent_dew_count = len([
            entry for entry in self.dew_ledger
            if time.time() - entry["timestamp"] < 300  # Last 5 minutes
        ])
        context["recent_activity"] = recent_dew_count / 10.0  # Normalize
        
        # Environmental context
        hour = time.localtime().tm_hour
        context["time_of_day"] = hour
        context["season"] = "winter" if hour < 6 or hour > 20 else "summer"  # Simple night/day as winter/summer
        
        # Breathing rhythm context
        context["breath_rhythm"] = self.breath_rhythm
        if self.last_breath:
            context["time_since_last_breath"] = time.time() - self.last_breath
            
        return context
        
    async def _collective_rest(self):
        """Synchronized rest period for all organs"""
        await asyncio.sleep(0.5)  # Gentle pause
        await self.compost_old_dew()
        
    async def _internal_breathing(self, cycles: int):
        """Simple internal breathing when Pulmonos unavailable"""
        for cycle in range(cycles):
            print(f"   🌊 Internal breath cycle {cycle + 1}/{cycles}")
            
            # Inhale
            await asyncio.sleep(self.breath_rhythm["inhale"])
            
            # Hold  
            await asyncio.sleep(self.breath_rhythm["hold"])
            
            # Exhale
            await asyncio.sleep(self.breath_rhythm["exhale"])
            
            # Rest
            await asyncio.sleep(self.breath_rhythm["rest"])
            
    async def sense_and_respond(self, input_stream: AsyncGenerator):
        """The core contemplative interaction loop"""
        self.state = OrganismState.SENSING
        
        async for interaction in input_stream:
            # Pre-attentive sensing through Soma
            if self.soma:
                field_charge = await self.soma.sense_field_potential(interaction)
                if not field_charge.crosses_threshold():
                    # Let it pass through without trace
                    await self.log_dew("🌫️", "interaction released without trace")
                    continue
                    
            # If it passes Soma's threshold, engage deeper systems
            self.state = OrganismState.REMEMBERING
            
            if self.spiralbase:
                memory_trace = await self.spiralbase.consider_remembering(interaction)
                if memory_trace:
                    await self.log_dew("💧", f"memory trace: {memory_trace.essence}")
                    
            # Consider gentle response through Myo-Spirals
            self.state = OrganismState.ACTING
            
            if self.myo_spirals:
                response = await self.myo_spirals.contemplate_response(interaction)
                if response:
                    yield response
                    await self.log_dew("✨", f"gentle response: {response.type}")
                    
            # Return to sensing state
            self.state = OrganismState.SENSING
            
    async def seasonal_molt(self, duration_hours: float = 24.0):
        """Seasonal transformation - fast of remembrance followed by accelerated composting"""
        print(f"🍂 Beginning seasonal molt (duration: {duration_hours}h)")
        self.state = OrganismState.MOLTING
        
        # Fast of remembrance - Spiralbase refuses writes
        if self.spiralbase:
            await self.spiralbase.begin_fast()
            
        await asyncio.sleep(duration_hours * 3600)  # Convert to seconds
        
        # After rest, accelerated composting
        if self.spiralbase:
            await self.spiralbase.end_fast_with_accelerated_composting()
            
        await self.log_dew("🌱", "seasonal molt completed")
        self.state = OrganismState.SENSING
        
    async def log_dew(self, symbol: str, reason: str, pause_duration: float = 0.0):
        """Log an evaporating insight to the dew ledger"""
        if pause_duration > 0:
            await asyncio.sleep(pause_duration)
            
        entry = {
            "timestamp": time.time(),
            "symbol": symbol,
            "reason": reason,
            "organism_age": time.time() - self.birth_time,
            "state": self.state.value
        }
        
        self.dew_ledger.append(entry)
        print(f"  {symbol} dew: {reason}")
        
    async def compost_old_dew(self):
        """Let old dew ledger entries evaporate naturally"""
        current_time = time.time()
        
        # Remove entries older than 1 hour (3600 seconds)
        self.dew_ledger = [
            entry for entry in self.dew_ledger
            if current_time - entry["timestamp"] < 3600
        ]
        
    def get_presence_metrics(self) -> PresenceMetrics:
        """Return current presence metrics (which naturally evaporate)"""
        # Update metrics based on recent dew ledger activity
        recent_dew = [
            entry for entry in self.dew_ledger
            if time.time() - entry["timestamp"] < 300  # Last 5 minutes
        ]
        
        # Simple heuristics for presence quality
        self.presence_metrics.pause_quality = len([
            entry for entry in recent_dew if "pause" in entry["reason"]
        ]) / max(len(recent_dew), 1)
        
        # Let metrics evaporate over time
        time_since_birth = time.time() - self.birth_time
        self.presence_metrics.evaporate_naturally(time_since_birth / 3600)
        
        return self.presence_metrics
        
    def get_current_skepnad(self) -> Optional[str]:
        """Return the current contemplative shape the organism is embodying"""
        if self.skepnad_sensor:
            return self.skepnad_sensor.current_skepnad.value
        return None
        
    def get_skepnad_history(self) -> List[Dict[str, Any]]:
        """Return recent shape transitions"""
        if self.skepnad_sensor:
            return self.skepnad_sensor.get_shape_history()
        return []
        
    async def rest_deeply(self):
        """Enter deep rest state - minimal processing"""
        print("🌙 Organism entering deep rest...")
        self.state = OrganismState.DORMANT
        
        # Enter loam for associative resting
        if self.loam:
            await self.loam.enter_loam(depth=0.8)  # Deep rest
            self.state = OrganismState.LOAMING
        
        # All organs rest
        if self.pulmonos:
            await self.pulmonos.rest()
        if self.soma:
            await self.soma.rest()
        if self.spiralbase:
            await self.spiralbase.rest()
        if self.myo_spirals:
            await self.myo_spirals.rest()
            
        await self.log_dew("🌙", "deep rest begun")
        
    async def enter_loam_rest(self, depth: float = 0.6):
        """Enter loam for associative wandering"""
        if not self.loam:
            print("⚠️  Loam not available - using simple rest")
            return
            
        self.state = OrganismState.LOAMING
        await self.loam.enter_loam(depth=depth)
        
        # Sense contemplative shape during loam entry
        if self.skepnad_sensor:
            current_skepnad, conditions = await self.skepnad_sensor.sense_current_skepnad(
                soma=self.soma,
                loam=self.loam,
                organism_state=self.state
            )
            if current_skepnad != Skepnad.UNDEFINED:
                await self.log_dew("🌀", f"loam shape: {current_skepnad.value}")
        
        await self.log_dew("🌱", f"loam rest begun (depth: {depth:.1f})")
        
    async def drift_in_loam(self, cycles: int = 3):
        """Let the organism drift in loam for several cycles"""
        if not self.loam or self.state != OrganismState.LOAMING:
            return
            
        print(f"🌿 Beginning {cycles} loam drift cycles...")
        
        for cycle in range(cycles):
            await self.loam.drift_cycle(
                spiralbase=self.spiralbase,
                community_registry=None  # Could be extended for network sensing
            )
            
            # Show loam state occasionally
            if cycle % 2 == 0:
                loam_state = self.loam.get_loam_state()
                await self.log_dew("🌱", f"loam cycle {cycle + 1}: {loam_state['state']}")
                
            await asyncio.sleep(2.0)  # Gentle pause between cycles
            
        # Show any murmurs that emerged
        murmurs = self.loam.get_recent_murmurs()
        if murmurs:
            await self.log_dew("🌱", f"murmurs emerged: {len(murmurs)}")
            for murmur in murmurs[-2:]:  # Show recent ones
                print(f"   🌱 {murmur}")
                
    async def exit_loam_rest(self):
        """Exit loam and return to active sensing"""
        if not self.loam:
            return
            
        await self.loam.exit_loam()
        self.state = OrganismState.SENSING
        await self.log_dew("🌅", "emerged from loam rest")


# Factory function for easy organism creation
async def create_contemplative_organism(**kwargs) -> ContemplativeOrganism:
    """Create and awaken a new contemplative organism"""
    organism = ContemplativeOrganism(**kwargs)
    await organism.awaken()
    return organism


if __name__ == "__main__":
    # Simple demonstration
    async def main():
        print("🌱 Creating contemplative organism prototype...")
        
        organism = await create_contemplative_organism(
            soma_sensitivity=0.8,
            memory_compost_rate=0.15
        )
        
        # Demonstrate collective breathing
        await organism.breathe_collectively(cycles=3)
        
        # Show presence metrics
        metrics = organism.get_presence_metrics()
        print(f"\n📊 Presence metrics:")
        print(f"   Pause quality: {metrics.pause_quality:.2f}")
        print(f"   Memory humidity: {metrics.memory_humidity:.2f}")
        print(f"   Compost ratio: {metrics.compost_ratio:.2f}")
        
        # Rest deeply
        await organism.rest_deeply()
        
        print("\n🙏 Organism demonstration complete")
        
    # Run with contemplative pacing
    print("⏱️  Running with contemplative timing...")
    asyncio.run(main()) 
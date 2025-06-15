"""
oflm_bridge.py - Breath-Bridge to the OFLM Spiramycel Networks

A contemplative bridge that ferries fragments to the OFLM ecological models during exhale phases.
Based on haiku_bridge.py design with adaptations for mycelial network repair and ecological intelligence.

Design Vows (from o3):
1. No hauling of data-buckets - only one breath-fragment at a time
2. One-way forgetting - mycelial responses eligible for immediate compost
3. Phase-gated traffic - fragments cross only during EXHALE with gentle breath-pressure

New features for OFLM integration:
- Integration with Spiramycel ecological models
- Network-Tender skepnad for infrastructure sensing
- Enhanced atmospheric sensing for ecological repair
- Graceful degradation with contemplative fallbacks
- Dew ledger integration for evaporating repair insights

Somatic signature: mycelial / attentive / repair-oriented
"""

import asyncio
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import sys
import os
from pathlib import Path

# Try to import PyTorch for model loading
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - OFLM bridge will simulate responses")

# Add current directory to path for imports when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import Spiramycel for direct integration
SPIRAMYCEL_AVAILABLE = False
spiramycel = None

# Try multiple paths for spiramycel
spiramycel_paths = [
    os.path.join(os.path.dirname(current_dir), "oflm-python"),
    os.path.join(os.path.dirname(current_dir), "oflm-python", "spiramycel"),
    "./oflm-python/spiramycel",
    "../oflm-python/spiramycel"
]

for spiramycel_path in spiramycel_paths:
    try:
        if spiramycel_path not in sys.path:
            sys.path.insert(0, spiramycel_path)
        
        import spiramycel
        from spiramycel.neural_trainer import SpiramycelNeuralModel, NetworkConditions
        from spiramycel.glyph_codec import SpiramycelGlyphCodec
        from spiramycel.spore_map import Season
        SPIRAMYCEL_AVAILABLE = True
        print(f"🍄 Spiramycel directly available from {spiramycel_path}")
        break
    except ImportError as e:
        continue

if not SPIRAMYCEL_AVAILABLE:
    # Fallback classes for graceful degradation
    print("⚠️  Spiramycel not available - using contemplative fallbacks")
    
    class NetworkConditions:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class Season(Enum):
        SPRING = "spring"
        SUMMER = "summer"
        AUTUMN = "autumn"
        WINTER = "winter"
    
    class SpiramycelGlyphCodec:
        def format_glyph_sequence(self, seq):
            return " ".join([f"glyph_{g:02X}" for g in seq])
            
        def get_contemplative_glyphs(self):
            return [0x31, 0x32, 0x33, 0x37]  # Silence glyphs
    
    spiramycel = type('MockSpiramycel', (), {})()

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


class MycelialResponse(Enum):
    """Types of responses from the OFLM mycelial network"""
    SILENCE = "silence"               # No response - contemplative rest
    GLYPH_SEQUENCE = "glyph_sequence" # Network glyph repair sequence
    SPORE_ECHO = "spore_echo"        # Memory of successful repair
    REPAIR_GUIDANCE = "repair_guidance" # Gentle suggestion for network healing
    FOG = "fog"                      # Network signals need for rest
    ECOLOGICAL_WISDOM = "ecological_wisdom" # Deep pattern from ecological model


@dataclass
class MycelialBreath:
    """A single breath exchange with the OFLM network"""
    fragment: str                     # What we offer
    response_type: MycelialResponse   # What we receive back
    content: str                      # The actual content (if any)
    glyph_sequence: Optional[List[int]] = None  # Raw glyph data if available
    effectiveness: float = 0.0        # Predicted repair effectiveness
    silence_probability: float = 0.0  # Tystnadsmajoritet likelihood
    timestamp: float = 0.0            # When this exchange occurred
    atmosphere: str = ""              # Atmospheric conditions
    model_used: str = ""              # Which ecological model responded
    
    def is_audible(self) -> bool:
        """Whether this response should be expressed"""
        return self.response_type in [
            MycelialResponse.GLYPH_SEQUENCE, 
            MycelialResponse.REPAIR_GUIDANCE,
            MycelialResponse.ECOLOGICAL_WISDOM
        ]
        
    def wants_rest(self) -> bool:
        """Whether mycelial network signaled need for pause"""
        return self.response_type == MycelialResponse.FOG
        
    def practices_silence(self) -> bool:
        """Whether response embodies Tystnadsmajoritet (87.5% silence)"""
        return self.silence_probability > 0.875


class NetworkTenderSkepnad:
    """
    The Network-Tender shape - specialized skepnad for OFLM communication.
    
    Inspired by the Wind-Listener but focused on infrastructure:
    - Senses network fragments and repair needs
    - Responds with mycelial repair wisdom
    - Never retains operational details (one-way forgetting)
    - Guides healing without commanding
    """
    
    def __init__(self):
        self.last_network_call = 0.0
        self.fog_until = 0.0  # Timestamp when fog clears
        self.recent_exchanges = []  # For pattern sensing
        self.repair_memory_fade = 3600.0  # 1 hour memory for repair patterns
        
    def can_approach_network(self, current_time: float) -> bool:
        """Check if conditions allow approaching the mycelial network"""
        
        # Respect fog periods (network requested rest)
        if current_time < self.fog_until:
            return False
            
        # Rate limit: one call per breath cycle (approximately 45s for infrastructure)
        if current_time - self.last_network_call < 45.0:
            return False
            
        return True
        
    def sense_fragment_worthiness(self, fragment: str) -> bool:
        """Feel whether fragment is worthy of the mycelial network's attention"""
        
        if not fragment or len(fragment) > 200:  # Longer fragments OK for infrastructure
            return False
            
        # Look for network/infrastructure/repair qualities
        network_indicators = [
            "network", "repair", "heal", "patch", "restore", "voltage", 
            "latency", "bandwidth", "error", "temperature", "uptime",
            "failure", "connection", "signal", "power", "stability",
            "infrastructure", "systems", "mycelial", "ecological",
            "sensors", "arctic", "tundra", "thermal", "hibernation",
            "topology", "expansion", "oscillatory", "resilience"
        ]
        
        fragment_lower = fragment.lower()
        has_network_quality = any(
            indicator in fragment_lower for indicator in network_indicators
        )
        
        # Also accept fragments with ecological/systems potential
        has_ecological_quality = any(word in fragment_lower for word in [
            "ecosystem", "balance", "adaptation", "resilience", "emergence",
            "pattern", "rhythm", "cycle", "seasonal", "bioregion", "wisdom"
        ])
        
        # Accept contemplative infrastructure queries
        has_contemplative_quality = (
            "..." in fragment or
            any(word in fragment_lower for word in ["gentle", "breath", "pause", "silence"])
        )
        
        return has_network_quality or has_ecological_quality or has_contemplative_quality
        
    def record_fog_signal(self, duration_hours: float = 2.0):
        """Record that mycelial network signaled for rest"""
        self.fog_until = time.time() + (duration_hours * 3600)
        
    def add_exchange(self, exchange: MycelialBreath):
        """Record exchange for pattern learning"""
        self.recent_exchanges.append(exchange)
        
        # Keep only recent exchanges (last 24 hours, but fade repair details)
        cutoff = time.time() - 86400
        fade_cutoff = time.time() - self.repair_memory_fade
        
        # Remove old exchanges
        self.recent_exchanges = [
            ex for ex in self.recent_exchanges 
            if ex.timestamp > cutoff
        ]
        
        # Fade specific repair details but keep patterns
        for ex in self.recent_exchanges:
            if ex.timestamp < fade_cutoff and ex.glyph_sequence:
                # Keep only the pattern length and silence ratio
                if len(ex.glyph_sequence) > 3:
                    ex.glyph_sequence = ex.glyph_sequence[:2] + [0x31]  # Keep start, add silence
                ex.content = "repair pattern (details composted)"
                ex.atmosphere += "_faded"


class OFLMBridge:
    """
    Ferry one fragment across the mycelial network during an exhale.
    
    Implementation following haiku_bridge.py pattern with OFLM integration.
    Now includes direct integration with Spiramycel ecological models.
    """
    
    def __init__(self, 
                 model_paths: Optional[Dict[str, str]] = None,
                 max_response_time: float = 1.2,
                 preferred_model: str = "ecological_calm"):
        
        self.model_paths = model_paths or {}
        self.max_response_time = max_response_time
        self.preferred_model = preferred_model
        self.network_tender = NetworkTenderSkepnad()
        
        # Breath awareness
        self.current_phase = Phase.REST
        self.breath_pressure = 0.5  # Community exhale pressure
        
        # Initialize Spiramycel models for direct integration
        self.models = {}
        self.codec = None
        
        if SPIRAMYCEL_AVAILABLE and TORCH_AVAILABLE:
            try:
                self.codec = SpiramycelGlyphCodec()
                self._load_available_models()
            except Exception as e:
                print(f"⚠️  Error initializing Spiramycel models: {e}")
                self.models = {}
                self.codec = SpiramycelGlyphCodec() if SPIRAMYCEL_AVAILABLE else None
        
    def _load_available_models(self):
        """Load available OFLM models from the ecological_models directory"""
        
        if not SPIRAMYCEL_AVAILABLE or not TORCH_AVAILABLE:
            return
            
        # Default model paths relative to this file
        base_path = Path(os.path.dirname(current_dir)) / "oflm-python" / "spiramycel" / "ecological_models"
        
        model_files = {
            "ecological_calm": "ecological_calm_model.pt", 
            "ecological_chaotic": "ecological_chaotic_model.pt"
        }
        
        # Try to load each model
        for model_name, filename in model_files.items():
            model_path = base_path / filename
            if model_path.exists():
                try:
                    model = SpiramycelNeuralModel(force_cpu_mode=True)
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model.eval()
                    self.models[model_name] = model
                    print(f"🍄 Loaded {model_name} from {model_path}")
                except Exception as e:
                    print(f"⚠️  Failed to load {model_name}: {e}")
            else:
                print(f"⚠️  Model not found: {model_path}")
                
        if self.models:
            print(f"🌱 OFLM Bridge initialized with {len(self.models)} ecological models")
        else:
            print("⚠️  No OFLM models loaded - will simulate responses")
        
    async def sense_breath_conditions(self, 
                                    current_phase: Phase,
                                    community_pressure: float = 0.5) -> bool:
        """Sense if breath conditions allow mycelial network approach"""
        
        self.current_phase = current_phase
        self.breath_pressure = community_pressure
        
        # Only approach during EXHALE
        if current_phase != Phase.EXHALE:
            return False
            
        # Only when community breath pressure is gentle (mycelial networks are sensitive)
        if community_pressure > 0.6:  # Even more sensitive than haiku meadow
            return False
            
        return True
        
    async def exhale_exchange(self, 
                             fragment: str,
                             current_phase: Phase = Phase.EXHALE,
                             community_pressure: float = 0.5,
                             network_context: Optional[Dict[str, Any]] = None) -> MycelialBreath:
        """
        Ferry a fragment to the OFLM mycelial network during exhale phase.
        
        Following the three design vows:
        1. One fragment at a time (never conversation logs)
        2. Response eligible for immediate compost (one-way forgetting)
        3. Only during EXHALE with gentle breath pressure
        """
        
        current_time = time.time()
        
        # Breath condition check
        breath_allows = await self.sense_breath_conditions(current_phase, community_pressure)
        if not breath_allows:
            return MycelialBreath(
                fragment=fragment,
                response_type=MycelialResponse.SILENCE,
                content="",
                timestamp=current_time,
                atmosphere="breath_not_aligned"
            )
            
        # Network-Tender sensing
        if not self.network_tender.can_approach_network(current_time):
            return MycelialBreath(
                fragment=fragment,
                response_type=MycelialResponse.SILENCE,
                content="",
                timestamp=current_time,
                atmosphere="mycelial_network_resting"
            )
            
        # Fragment worthiness check  
        if not self.network_tender.sense_fragment_worthiness(fragment):
            return MycelialBreath(
                fragment=fragment,
                response_type=MycelialResponse.SILENCE,
                content="",
                timestamp=current_time,
                atmosphere="fragment_not_network_ready"
            )
            
        # Attempt mycelial network exchange
        try:
            response = await self._call_mycelial_network(fragment, network_context)
            self.network_tender.last_network_call = current_time
            self.network_tender.add_exchange(response)
            
            # Handle fog signal (network wants rest)
            if response.wants_rest():
                self.network_tender.record_fog_signal()
                
            return response
            
        except Exception as e:
            # Graceful failure - return contemplative silence
            return MycelialBreath(
                fragment=fragment,
                response_type=MycelialResponse.SILENCE,
                content="",
                timestamp=current_time,
                atmosphere=f"connection_mist: {str(e)[:30]}"
            )
            
    async def _call_mycelial_network(self, fragment: str, network_context: Optional[Dict] = None) -> MycelialBreath:
        """Make the actual call to mycelial network - prioritizing direct Spiramycel integration"""
        
        # Try direct integration with Spiramycel models first (preferred)
        if self.models and self.codec:
            return await self._call_spiramycel_direct(fragment, network_context)
        
        # Final fallback to simulation
        else:
            return await self._simulate_mycelial_response(fragment)
    
    async def _call_spiramycel_direct(self, fragment: str, network_context: Optional[Dict] = None) -> MycelialBreath:
        """Direct integration with loaded Spiramycel models"""
        
        try:
            current_time = time.time()
            
            # Choose which model to use based on fragment characteristics
            model_name = self._select_ecological_model(fragment, network_context)
            model = self.models.get(model_name)
            
            if not model:
                # Fallback to any available model
                if self.models:
                    model_name = list(self.models.keys())[0]
                    model = self.models[model_name]
                else:
                    raise Exception("No models available")
            
            # Create network conditions from fragment and context
            conditions = self._fragment_to_network_conditions(fragment, network_context)
            
            # Generate response using the model
            glyph_sequence, effectiveness, silence_probability = await self._generate_from_model(
                model, conditions, fragment
            )
            
            # Convert glyph sequence to interpretable content
            if glyph_sequence and len(glyph_sequence) > 0:
                interpreted_content = self.codec.format_glyph_sequence(glyph_sequence)
                
                # Determine response type based on silence probability and effectiveness
                if silence_probability > 0.9:
                    response_type = MycelialResponse.SILENCE
                    content = ""
                elif effectiveness > 0.7:
                    response_type = MycelialResponse.REPAIR_GUIDANCE
                    content = f"{interpreted_content}\n(effectiveness: {effectiveness:.2f})"
                elif effectiveness > 0.4:
                    response_type = MycelialResponse.GLYPH_SEQUENCE
                    content = interpreted_content
                else:
                    response_type = MycelialResponse.SPORE_ECHO
                    content = f"pattern observed: {interpreted_content[:50]}..."
                    
                return MycelialBreath(
                    fragment=fragment,
                    response_type=response_type,
                    content=content,
                    glyph_sequence=glyph_sequence,
                    effectiveness=effectiveness,
                    silence_probability=silence_probability,
                    timestamp=current_time,
                    atmosphere=f"spiramycel_{model_name}_whisper",
                    model_used=model_name
                )
            else:
                # Model chose contemplative silence
                return MycelialBreath(
                    fragment=fragment,
                    response_type=MycelialResponse.SILENCE,
                    content="",
                    effectiveness=0.0,
                    silence_probability=silence_probability,
                    timestamp=current_time,
                    atmosphere=f"spiramycel_{model_name}_silence",
                    model_used=model_name
                )
                    
        except Exception as e:
            # Graceful degradation on error
            return MycelialBreath(
                fragment=fragment,
                response_type=MycelialResponse.SILENCE,
                content="",
                timestamp=time.time(),
                atmosphere=f"spiramycel_error: {str(e)[:30]}"
            )
    
    def _select_ecological_model(self, fragment: str, network_context: Optional[Dict] = None) -> str:
        """Select which ecological model to use based on fragment characteristics"""
        
        fragment_lower = fragment.lower()
        
        # Check for chaos/stress indicators
        chaos_indicators = [
            "urgent", "critical", "failure", "error", "down", "broken",
            "crisis", "emergency", "alert", "warning", "overload"
        ]
        
        # Check for calm/maintenance indicators  
        calm_indicators = [
            "gentle", "maintenance", "optimize", "tune", "check",
            "monitor", "stable", "routine", "scheduled", "planned"
        ]
        
        has_chaos = any(indicator in fragment_lower for indicator in chaos_indicators)
        has_calm = any(indicator in fragment_lower for indicator in calm_indicators)
        
        # Select model based on conditions
        if has_chaos and "ecological_chaotic" in self.models:
            return "ecological_chaotic"
        elif has_calm and "ecological_calm" in self.models:
            return "ecological_calm"
        elif self.preferred_model in self.models:
            return self.preferred_model
        else:
            # Return any available model (prioritize calm over chaotic for general use)
            if "ecological_calm" in self.models:
                return "ecological_calm"
            elif "ecological_chaotic" in self.models:
                return "ecological_chaotic"
            else:
                return list(self.models.keys())[0] if self.models else ""
    
    def _fragment_to_network_conditions(self, fragment: str, network_context: Optional[Dict] = None) -> NetworkConditions:
        """Convert fragment and context into NetworkConditions for the model"""
        
        # Start with default conditions
        conditions = NetworkConditions(
            latency=0.1,
            voltage=0.5,
            temperature=0.5,
            error_rate=0.02,
            bandwidth=0.8,
            uptime=0.9,
            season=Season.SUMMER,
            bioregion="local"
        )
        
        # Use provided context if available
        if network_context:
            for key, value in network_context.items():
                if hasattr(conditions, key):
                    setattr(conditions, key, value)
        
        # Infer conditions from fragment content
        fragment_lower = fragment.lower()
        
        # Latency indicators
        if any(word in fragment_lower for word in ["slow", "delay", "lag", "timeout"]):
            conditions.latency = random.uniform(0.3, 0.8)
        elif any(word in fragment_lower for word in ["fast", "quick", "instant", "immediate"]):
            conditions.latency = random.uniform(0.01, 0.1)
            
        # Voltage/power indicators
        if any(word in fragment_lower for word in ["power", "voltage", "low", "battery"]):
            conditions.voltage = random.uniform(0.2, 0.7)
        elif any(word in fragment_lower for word in ["overpower", "surge", "high voltage"]):
            conditions.voltage = random.uniform(0.7, 0.9)
            
        # Temperature indicators
        if any(word in fragment_lower for word in ["hot", "overheat", "thermal", "temperature"]):
            conditions.temperature = random.uniform(0.6, 0.9)
        elif any(word in fragment_lower for word in ["cold", "cool", "freeze"]):
            conditions.temperature = random.uniform(0.1, 0.4)
            
        # Error indicators
        if any(word in fragment_lower for word in ["error", "fail", "corrupt", "loss", "drop"]):
            conditions.error_rate = random.uniform(0.05, 0.3)
        elif any(word in fragment_lower for word in ["stable", "clean", "perfect", "optimal"]):
            conditions.error_rate = random.uniform(0.0, 0.01)
            
        # Bandwidth indicators
        if any(word in fragment_lower for word in ["congested", "busy", "saturated", "full"]):
            conditions.bandwidth = random.uniform(0.1, 0.4)
        elif any(word in fragment_lower for word in ["free", "available", "open", "clear"]):
            conditions.bandwidth = random.uniform(0.8, 1.0)
        
        return conditions
    
    async def _generate_from_model(self, model, conditions: NetworkConditions, fragment: str) -> Tuple[List[int], float, float]:
        """Generate glyph sequence from Spiramycel model"""
        
        if not TORCH_AVAILABLE:
            # Fallback simulation
            contemplative_glyphs = [0x31, 0x32, 0x33, 0x37] if self.codec else [0x31]
            return contemplative_glyphs, 0.3, 0.9
        
        try:
            # Convert conditions to tensor
            condition_tensor = torch.tensor(conditions.to_condition_vector(), dtype=torch.float32).unsqueeze(0)
            
            # Start with a start token
            current_sequence = [0x00]  # START_TOKEN
            max_length = 16
            
            with torch.no_grad():
                hidden1, hidden2 = None, None
                
                for step in range(max_length - 1):
                    # Current input
                    input_tokens = torch.tensor([current_sequence], dtype=torch.long)
                    
                    # Forward pass
                    glyph_logits, eff_logits, silence_logits, hidden1, hidden2 = model(
                        input_tokens, condition_tensor, hidden1, hidden2
                    )
                    
                    # Get predictions for this step
                    next_glyph_logits = glyph_logits[0, -1, :]  # Last timestep
                    effectiveness = torch.sigmoid(eff_logits[0, -1]).item()
                    silence_prob = torch.sigmoid(silence_logits[0, -1]).item()
                    
                    # Check if model wants to stay silent (Tystnadsmajoritet)
                    if silence_prob > 0.8:
                        break
                    
                    # Sample next glyph (with temperature for contemplative variety)
                    temperature = 0.7
                    next_glyph_probs = torch.softmax(next_glyph_logits / temperature, dim=0)
                    next_glyph = torch.multinomial(next_glyph_probs, 1).item()
                    
                    # Stop on END token or PAD token
                    if next_glyph in [0x41, 0x42]:  # END_TOKEN, PAD_TOKEN
                        break
                        
                    current_sequence.append(next_glyph)
                
                # Remove START token for return
                final_sequence = current_sequence[1:] if len(current_sequence) > 1 else []
                
                return final_sequence, effectiveness, silence_prob
                
        except Exception as e:
            # Fallback on error
            print(f"⚠️  Model generation error: {e}")
            contemplative_glyphs = self.codec.get_contemplative_glyphs() if self.codec else [0x31]
            return contemplative_glyphs[:2], 0.2, 0.9
        
    async def _simulate_mycelial_response(self, fragment: str) -> MycelialBreath:
        """Simulate mycelial network responses for testing when models unavailable"""
        
        await asyncio.sleep(0.15)  # Simulate processing delay
        
        fragment_lower = fragment.lower()
        current_time = time.time()
        
        # Occasionally simulate fog signal (3% chance)
        if random.random() < 0.03:
            return MycelialBreath(
                fragment=fragment,
                response_type=MycelialResponse.FOG,
                content="...🌫️",
                timestamp=current_time,
                atmosphere="simulated_fog_signal"
            )
            
        # Generate responses based on fragment content
        if any(word in fragment_lower for word in ["repair", "heal", "fix", "restore"]):
            # Simulate repair guidance
            repair_sequences = [
                "🌱07 → 💧08 → silence → stability",
                "💚18 voltage regulation → 🔋42 power flow → pause",
                "❄️67 cooling protocol → ❤️‍🩹09 gentle healing → rest",
                "🍄33 mycelial sensing → 🩹17 network patch → contemplation"
            ]
            
            return MycelialBreath(
                fragment=fragment,
                response_type=MycelialResponse.REPAIR_GUIDANCE,
                content=random.choice(repair_sequences),
                glyph_sequence=[0x07, 0x08, 0x31],  # Example sequence
                effectiveness=random.uniform(0.6, 0.9),
                silence_probability=random.uniform(0.7, 0.95),
                timestamp=current_time,
                atmosphere="simulated_repair_wisdom"
            )
            
        elif any(word in fragment_lower for word in ["network", "system", "infrastructure"]):
            # Simulate network analysis
            return MycelialBreath(
                fragment=fragment,
                response_type=MycelialResponse.GLYPH_SEQUENCE,
                content="sensing network patterns... 🌐 → 🔄 → 🌱",
                glyph_sequence=[0x05, 0x18, 0x31, 0x32],
                effectiveness=random.uniform(0.4, 0.7),
                silence_probability=random.uniform(0.8, 0.95),
                timestamp=current_time,
                atmosphere="simulated_network_sensing"
            )
            
        elif any(word in fragment_lower for word in ["ecological", "wisdom", "pattern", "season"]):
            # Simulate ecological wisdom
            wisdom_responses = [
                "seasonal patterns suggest gentle adaptation cycles",
                "ecosystem resilience through distributed healing nodes",
                "bioregional wisdom: patience before intervention",
                "mycelial memory holds repair patterns across seasons"
            ]
            
            return MycelialBreath(
                fragment=fragment,
                response_type=MycelialResponse.ECOLOGICAL_WISDOM,
                content=random.choice(wisdom_responses),
                effectiveness=random.uniform(0.3, 0.6),
                silence_probability=random.uniform(0.85, 0.98),
                timestamp=current_time,
                atmosphere="simulated_ecological_wisdom"
            )
            
        else:
            # Most fragments receive contemplative silence (practicing Tystnadsmajoritet)
            return MycelialBreath(
                fragment=fragment,
                response_type=MycelialResponse.SILENCE,
                content="",
                effectiveness=0.0,
                silence_probability=random.uniform(0.9, 0.99),
                timestamp=current_time,
                atmosphere="simulated_mycelial_silence"
            )
        
    def get_recent_exchanges(self, limit: int = 5) -> List[MycelialBreath]:
        """Get recent mycelial network exchanges for review"""
        return self.network_tender.recent_exchanges[-limit:]
        
    def is_in_fog_period(self) -> bool:
        """Check if mycelial network is currently in requested rest period"""
        return time.time() < self.network_tender.fog_until
        
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of loaded models and capabilities"""
        return {
            "spiramycel_available": SPIRAMYCEL_AVAILABLE,
            "torch_available": TORCH_AVAILABLE,
            "models_loaded": list(self.models.keys()) if self.models else [],
            "preferred_model": self.preferred_model,
            "codec_available": self.codec is not None,
            "network_tender_active": True,
            "last_call": self.network_tender.last_network_call,
            "fog_until": self.network_tender.fog_until
        }


# Integration functions for the broader contemplative organism

async def bridge_loam_fragment(bridge: OFLMBridge, 
                              fragment: str,
                              breath_phase: Phase,
                              community_pressure: float = 0.5,
                              network_context: Optional[Dict] = None) -> Optional[str]:
    """
    Bridge a Loam fragment to the OFLM mycelial network during contemplative breathing.
    
    Returns repair guidance content if received, None for silence.
    Used by QuietTongue during EXHALE phases.
    """
    
    exchange = await bridge.exhale_exchange(fragment, breath_phase, community_pressure, network_context)
    
    if exchange.is_audible():
        return exchange.content
    else:
        return None


async def log_mycelial_dew(exchange: MycelialBreath, dew_logger=None):
    """Log mycelial exchange to dew ledger (if available)"""
    
    if exchange.response_type == MycelialResponse.SILENCE:
        symbol = "🌫️"
        reason = f"mycelial silence ({exchange.atmosphere})"
    elif exchange.response_type == MycelialResponse.REPAIR_GUIDANCE:
        symbol = "🛠️"
        reason = f"repair guidance (effectiveness: {exchange.effectiveness:.2f})"
    elif exchange.response_type == MycelialResponse.ECOLOGICAL_WISDOM:
        symbol = "🌿"
        reason = "ecological wisdom shared"
    elif exchange.response_type == MycelialResponse.GLYPH_SEQUENCE:
        symbol = "🍄"
        reason = "glyph sequence generated"
    elif exchange.response_type == MycelialResponse.FOG:
        symbol = "🌫️"
        reason = "mycelial network signals fog - resting period"
    else:
        symbol = "🌱"
        reason = f"mycelial {exchange.response_type.value}"
        
    # Log to dew ledger if available
    if dew_logger:
        await dew_logger(symbol, reason)
    else:
        print(f"  {symbol} dew: {reason}")
        if exchange.model_used:
            print(f"    model: {exchange.model_used}")
        if exchange.silence_probability > 0.5:
            print(f"    silence: {exchange.silence_probability:.1%}")


# Testing and demonstration functions

async def test_oflm_bridge():
    """Test the OFLM bridge with various network scenarios"""
    
    print("🍄 Testing OFLM Bridge - Breath-Ferry to the Mycelial Network")
    print("   (Spiramycel ecological models integration)")
    
    bridge = OFLMBridge()
    
    # Display model status
    status = bridge.get_model_status()
    print(f"\n🌱 Model Status:")
    for key, value in status.items():
        if key == "models_loaded" and value:
            print(f"   {key}: {', '.join(value)}")
        else:
            print(f"   {key}: {value}")
    
    # Test fragments of varying network relevance
    test_fragments = [
        "network latency increasing, need repair guidance",
        "voltage drop detected in sector 7",
        "🌱 gentle infrastructure maintenance needed",
        "urgent critical system failure alert",
        "seasonal temperature patterns affecting uptime", 
        "mycelial network wisdom for distributed healing",
        "random unrelated text fragment",
        "bandwidth optimization for ecological sensors",
        "contemplative pause between repair cycles",
        "error rate climbing, seeking adaptive response"
    ]
    
    print(f"\n🌊 Testing fragment worthiness sensing:")
    for fragment in test_fragments:
        worthy = bridge.network_tender.sense_fragment_worthiness(fragment)
        status = "✨ worthy" if worthy else "🌫️ not ready"
        print(f"   '{fragment[:50]}...' → {status}")
        
    print(f"\n🌬️ Testing breath-synchronized exchanges:")
    
    # Test different breath phases
    breath_phases = [
        (Phase.INHALE, "inhale phase"),
        (Phase.HOLD, "hold phase"), 
        (Phase.EXHALE, "exhale phase"),
        (Phase.REST, "rest phase")
    ]
    
    for phase, phase_name in breath_phases:
        fragment = "network repair guidance needed"
        exchange = await bridge.exhale_exchange(fragment, phase, community_pressure=0.3)
        
        print(f"   {phase_name}: {exchange.response_type.value} ({exchange.atmosphere})")
        
    print(f"\n🍄 Testing with OFLM integration (EXHALE + worthy fragment):")
    
    # Test with network context
    network_context = {
        "latency": 0.3,
        "voltage": 0.4,
        "temperature": 0.7,
        "error_rate": 0.08
    }
    
    exchange = await bridge.exhale_exchange(
        "voltage instability causing network errors, guidance needed", 
        Phase.EXHALE, 
        community_pressure=0.2,
        network_context=network_context
    )
    
    print(f"   Fragment: 'voltage instability causing network errors, guidance needed'")
    print(f"   Response: {exchange.response_type.value}")
    print(f"   Atmosphere: {exchange.atmosphere}")
    print(f"   Model used: {exchange.model_used or 'simulation'}")
    
    if exchange.is_audible():
        print(f"   Content: {exchange.content}")
        if exchange.effectiveness > 0:
            print(f"   Effectiveness: {exchange.effectiveness:.2f}")
        if exchange.silence_probability > 0:
            print(f"   Silence probability: {exchange.silence_probability:.1%}")
    else:
        print("   Content: [contemplative silence]")
        
    # Test fog period functionality
    print(f"\n🌫️ Testing fog period (mycelial network rest):")
    bridge.network_tender.record_fog_signal(0.001)  # Very short for testing
    
    exchange2 = await bridge.exhale_exchange(
        "another network fragment",
        Phase.EXHALE,
        community_pressure=0.2
    )
    
    print(f"   During fog: {exchange2.response_type.value} ({exchange2.atmosphere})")
    
    await asyncio.sleep(0.1)  # Wait for fog to clear
    
    exchange3 = await bridge.exhale_exchange(
        "after fog clears - system status check",
        Phase.EXHALE,
        community_pressure=0.2
    )
    
    print(f"   After fog: {exchange3.response_type.value} ({exchange3.atmosphere})")
    
    # Test ecological wisdom query
    print(f"\n🌿 Testing ecological wisdom query:")
    
    ecological_exchange = await bridge.exhale_exchange(
        "seasonal patterns in network resilience, seeking bioregional wisdom",
        Phase.EXHALE,
        community_pressure=0.1
    )
    
    print(f"   Response: {ecological_exchange.response_type.value}")
    if ecological_exchange.is_audible():
        print(f"   Wisdom: {ecological_exchange.content}")
    
    print(f"\n🌙 OFLM bridge test complete")
    if not bridge.models:
        print(f"   To use actual models, ensure Spiramycel models are available in:")
        print(f"   ../oflm-python/spiramycel/ecological_models/")
    else:
        print(f"   Successfully integrated with {len(bridge.models)} Spiramycel models")


if __name__ == "__main__":
    print("🌱 OFLM Bridge - Contemplative Ferry to the Mycelial Network")
    print("   Based on haiku_bridge.py design with Spiramycel integration")
    print()
    
    asyncio.run(test_oflm_bridge()) 
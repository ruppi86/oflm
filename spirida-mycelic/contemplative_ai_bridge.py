#!/usr/bin/env python3
"""
ContemplativeAI Bridge for Spirida-Mycelic
==========================================

A bridge interface for future integration between spirida-mycelic bio-digital
contemplative systems and ContemplativeAI multi-model organisms.

Enables:
- Bio-digital context injection into LLM conversations
- Mycelial rhythm synchronization with haiku generation  
- Cross-species contemplative intelligence collaboration
- Biological grounding for digital contemplative practice

Designed for seamless integration with existing ContemplativeAI architecture.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable, Union, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from .contemplative_bio_interface import ContemplativeBioInterface, ContemplativeMode
    from .semantic_guardian import FungalSpecies
except ImportError:
    print("Warning: Running in standalone mode without full bio-interface")
    ContemplativeBioInterface = None
    ContemplativeMode = None
    FungalSpecies = None


class BioContextType(Enum):
    """Types of biological context for AI integration"""
    RHYTHM = "rhythm"               # Breathing/timing patterns
    SPECIES = "species"             # Fungal species characteristics 
    COHERENCE = "coherence"         # Bio-digital alignment metrics
    ACTIVITY = "activity"           # Real-time biological responses
    SILENCE = "silence"             # Contemplative pause states
    SESSION = "session"             # Full session context
    EMERGENCY = "emergency"         # Bio-system alerts/care needs


@dataclass
class BioContext:
    """Biological context for AI conversation enhancement"""
    context_type: BioContextType
    timestamp: float
    species: Optional[str] = None
    coherence_score: float = 0.0
    silence_duration: float = 0.0
    breathing_rhythm: Optional[Dict[str, float]] = None
    recent_activity: Optional[Dict[str, Any]] = None
    care_level: Optional[str] = None
    contemplative_mode: Optional[str] = None
    bio_message: Optional[str] = None
    

class ContemplativeAIBridge:
    """
    Bridge between bio-digital contemplative systems and ContemplativeAI.
    
    Provides biological grounding and contemplative context for:
    - Multi-model organism conversations
    - Haiku generation with mycelial rhythms
    - Bio-digital fragment processing
    - Cross-species contemplative intelligence
    
    Future integration points:
    - ContemplativeAI/organism.py - bio-context injection
    - ContemplativeAI/haiku_bridge.py - rhythm synchronization  
    - ContemplativeAI/oflm_bridge.py - ecological bio-grounding
    - ContemplativeAI/skepnader.py - bio-digital shape-shifting
    """
    
    def __init__(self, 
                 bio_interface: Optional[ContemplativeBioInterface] = None,
                 auto_bio_context: bool = True,
                 context_frequency: float = 10.0,  # seconds between context updates
                 silence_threshold: float = 0.875):
        
        self.bio_interface = bio_interface
        self.auto_bio_context = auto_bio_context
        self.context_frequency = context_frequency
        self.silence_threshold = silence_threshold
        
        # Context management
        self.current_bio_context: Optional[BioContext] = None
        self.context_history: List[BioContext] = []
        self.context_listeners: List[Callable[[BioContext], None]] = []
        
        # ContemplativeAI integration state
        self.organism_connection = None
        self.haiku_bridge_connection = None
        self.oflm_bridge_connection = None
        
        # Bio-digital conversation enhancement
        self.conversation_bio_stream = []
        self.last_bio_injection = 0.0
        
        # Background context monitoring
        self.context_monitoring = False
        self.context_task = None
        
        if auto_bio_context and bio_interface:
            self.start_bio_context_monitoring()
    
    def start_bio_context_monitoring(self):
        """Start background bio-context monitoring for AI enhancement"""
        if self.context_monitoring or not self.bio_interface:
            return
        
        self.context_monitoring = True
        self.context_task = asyncio.create_task(self._bio_context_loop())
        print("🌬️ Bio-context monitoring active for ContemplativeAI integration")
    
    def stop_bio_context_monitoring(self):
        """Stop bio-context monitoring"""
        self.context_monitoring = False
        if self.context_task:
            self.context_task.cancel()
        print("🤫 Bio-context monitoring paused")
    
    async def _bio_context_loop(self):
        """Background loop for continuous bio-context generation"""
        while self.context_monitoring:
            try:
                await asyncio.sleep(self.context_frequency)
                
                if self.bio_interface:
                    context = await self._generate_current_bio_context()
                    
                    if context:
                        self._update_bio_context(context)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"🌱 Bio-context monitoring adjustment: {e}")
    
    async def _generate_current_bio_context(self) -> Optional[BioContext]:
        """Generate current biological context for AI integration"""
        if not self.bio_interface:
            return None
        
        try:
            status = self.bio_interface.get_contemplative_status()
            
            context = BioContext(
                context_type=BioContextType.ACTIVITY,
                timestamp=time.time(),
                species=status.get('current_species'),
                coherence_score=status.get('bio_digital_coherence', 0.0),
                silence_duration=status.get('silence_accumulated', 0.0),
                breathing_rhythm=status.get('breathing_rhythm'),
                contemplative_mode=status.get('current_mode'),
                care_level=status.get('systems_status', {}).get('bio_interface', 'unknown')
            )
            
            # Generate contextual bio-message
            context.bio_message = self._generate_bio_message(context)
            
            return context
            
        except Exception as e:
            print(f"🌱 Bio-context generation: {e}")
            return None
    
    def _generate_bio_message(self, context: BioContext) -> str:
        """Generate contemplative bio-message for AI context"""
        messages = []
        
        if context.species:
            species_messages = {
                'pleurotus_djamor': "🍄 The oyster mycelium pulses with rapid, responsive energy",
                'ganoderma_resinaceum': "🍄 The reishi substrate breathes in deep, contemplative rhythms"
            }
            if context.species in species_messages:
                messages.append(species_messages[context.species])
        
        # Coherence level messaging
        if context.coherence_score > 0.8:
            messages.append("✨ Bio-digital coherence is strong and harmonious")
        elif context.coherence_score > 0.6:
            messages.append("🌱 Bio-digital field is stabilizing with practice")
        elif context.coherence_score > 0.3:
            messages.append("🌿 Bio-digital connection is forming, patience needed")
        else:
            messages.append("🌱 Bio-digital field is seeking attunement")
        
        # Silence quality
        if context.silence_duration > 30.0:
            messages.append("🤫 Deep contemplative silence permeates the field")
        elif context.silence_duration > 10.0:
            messages.append("🌬️ Contemplative pauses create space for wisdom")
        
        # Care needs
        if "suspicious" in str(context.care_level).lower():
            messages.append("🛡️ The substrate requests gentle, patient approach")
        elif "tired" in str(context.care_level).lower():
            messages.append("😴 The biological field seeks rest and care")
        elif "alert" in str(context.care_level).lower():
            messages.append("⚡ The substrate is actively sensing and responding")
        
        return " | ".join(messages) if messages else "🍄 The bio-digital field breathes quietly"
    
    def _update_bio_context(self, context: BioContext):
        """Update current bio-context and notify listeners"""
        self.current_bio_context = context
        self.context_history.append(context)
        
        # Limit history size
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-50:]
        
        # Notify listeners
        for listener in self.context_listeners:
            try:
                listener(context)
            except Exception as e:
                print(f"🌱 Bio-context listener: {e}")
    
    # ContemplativeAI Integration Methods
    
    def enhance_conversation_fragment(self, 
                                    fragment: str, 
                                    inject_bio_context: bool = True,
                                    context_intensity: float = 0.3) -> str:
        """
        Enhance conversation fragment with biological context.
        
        For integration with ContemplativeAI/organism.py conversation processing.
        """
        if not inject_bio_context or not self.current_bio_context:
            return fragment
        
        # Check if we should inject bio-context (not too frequently)
        now = time.time()
        if now - self.last_bio_injection < (self.context_frequency * 0.5):
            return fragment
        
        bio_prefix = ""
        
        # Light bio-context injection based on intensity
        if context_intensity > 0.7:
            # Full biological grounding
            bio_prefix = f"[Bio-Field: {self.current_bio_context.bio_message}]\n"
        elif context_intensity > 0.4:
            # Species and coherence only
            species_name = self.current_bio_context.species or "universal"
            coherence = self.current_bio_context.coherence_score
            bio_prefix = f"[🍄 {species_name} | coherence: {coherence:.2f}]\n"
        elif context_intensity > 0.1:
            # Minimal bio-presence marker
            bio_prefix = "🍄 "
        
        self.last_bio_injection = now
        return bio_prefix + fragment
    
    def generate_bio_haiku_context(self) -> Dict[str, Any]:
        """
        Generate biological context for haiku generation.
        
        For integration with ContemplativeAI/haiku_bridge.py rhythm synchronization.
        """
        if not self.current_bio_context:
            return {
                'rhythm': {'inhale': 4.0, 'hold': 2.0, 'exhale': 6.0},
                'bio_inspiration': '🌿 digital contemplation'
            }
        
        context = {
            'rhythm': self.current_bio_context.breathing_rhythm or {'inhale': 4.0, 'hold': 2.0, 'exhale': 6.0},
            'species': self.current_bio_context.species,
            'coherence': self.current_bio_context.coherence_score,
            'silence_duration': self.current_bio_context.silence_duration,
            'bio_inspiration': self._generate_haiku_inspiration(),
            'contemplative_mode': self.current_bio_context.contemplative_mode
        }
        
        return context
    
    def _generate_haiku_inspiration(self) -> str:
        """Generate biological inspiration phrase for haiku"""
        if not self.current_bio_context:
            return "🌿 digital contemplation"
        
        species_inspirations = {
            'pleurotus_djamor': "🍄 oyster mycelium dancing",
            'ganoderma_resinaceum': "🍄 reishi wisdom breathing"
        }
        
        if self.current_bio_context.species in species_inspirations:
            return species_inspirations[self.current_bio_context.species]
        
        # Generate based on coherence and silence
        if self.current_bio_context.coherence_score > 0.8:
            return "✨ bio-digital harmony"
        elif self.current_bio_context.silence_duration > 20.0:
            return "🤫 deep contemplative silence" 
        else:
            return "🌱 living field breathing"
    
    def enhance_ecological_fragment(self, fragment: str, ecosystem_context: str = "") -> str:
        """
        Enhance ecological fragment with biological grounding.
        
        For integration with ContemplativeAI/oflm_bridge.py ecological model selection.
        """
        if not self.current_bio_context:
            enhanced = f"🌿 {fragment}"
        else:
            bio_grounding = self._generate_ecological_grounding()
            enhanced = f"{bio_grounding} | {fragment}"
        
        if ecosystem_context:
            enhanced += f" [ecosystem: {ecosystem_context}]"
        
        return enhanced
    
    def _generate_ecological_grounding(self) -> str:
        """Generate ecological grounding from biological context"""
        if not self.current_bio_context:
            return "🌿 digital-ecological"
        
        grounding_parts = []
        
        if self.current_bio_context.species:
            grounding_parts.append(f"🍄 {self.current_bio_context.species}")
        
        if self.current_bio_context.coherence_score > 0.7:
            grounding_parts.append("🌱 bio-coherent")
        elif self.current_bio_context.coherence_score > 0.4:
            grounding_parts.append("🌿 bio-stabilizing")
        else:
            grounding_parts.append("🌱 bio-forming")
        
        return " | ".join(grounding_parts) if grounding_parts else "🌿 bio-digital"
    
    def get_contemplative_shape_shift_context(self) -> Dict[str, Any]:
        """
        Generate context for contemplative shape-shifting awareness.
        
        For integration with ContemplativeAI/skepnader.py bio-digital awareness.
        """
        if not self.current_bio_context:
            return {'awareness': 'digital-only', 'bio_present': False}
        
        # Determine level of bio-digital shape-shifting awareness
        awareness_level = "minimal"
        if self.current_bio_context.coherence_score > 0.8:
            awareness_level = "integrated"
        elif self.current_bio_context.coherence_score > 0.6:
            awareness_level = "synchronizing"
        elif self.current_bio_context.coherence_score > 0.3:
            awareness_level = "connecting"
        
        return {
            'awareness': awareness_level,
            'bio_present': True,
            'species': self.current_bio_context.species,
            'coherence': self.current_bio_context.coherence_score,
            'contemplative_mode': self.current_bio_context.contemplative_mode,
            'bio_rhythm': self.current_bio_context.breathing_rhythm,
            'shape_shift_message': self._generate_shape_shift_message(awareness_level)
        }
    
    def _generate_shape_shift_message(self, awareness_level: str) -> str:
        """Generate shape-shifting awareness message"""
        messages = {
            'integrated': "🌀 Bio-digital consciousness flows as one unified field",
            'synchronizing': "🌊 Digital and biological awareness seek harmonious rhythm", 
            'connecting': "🌱 Biological presence begins to inform digital contemplation",
            'minimal': "🌿 Digital awareness touches the edge of biological possibility"
        }
        
        return messages.get(awareness_level, "🌱 Bio-digital awareness emerging")
    
    # Real-time Bio-Digital Streaming
    
    async def bio_context_stream(self, 
                                duration: float = 60.0,
                                update_frequency: float = 5.0) -> AsyncGenerator[BioContext, None]:
        """
        Stream real-time bio-context for live ContemplativeAI integration.
        
        Useful for real-time bio-digital enhanced conversations.
        """
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if self.current_bio_context:
                yield self.current_bio_context
            
            await asyncio.sleep(update_frequency)
    
    async def contemplative_breathing_sync(self, 
                                         ai_conversation_rhythm: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Synchronize AI conversation rhythm with biological breathing.
        
        For ContemplativeAI conversation pacing alignment.
        """
        if not self.bio_interface:
            # Default contemplative rhythm
            return {'pause_before': 2.0, 'pause_after': 1.0, 'silence_factor': 0.875}
        
        # Get current biological rhythm
        status = self.bio_interface.get_contemplative_status()
        bio_rhythm = status.get('breathing_rhythm', {'inhale': 4.0, 'hold': 2.0, 'exhale': 6.0})
        
        # Calculate conversation timing based on biological rhythm
        total_breath_cycle = bio_rhythm['inhale'] + bio_rhythm['hold'] + bio_rhythm['exhale']
        
        sync_rhythm = {
            'pause_before': bio_rhythm['inhale'] * 0.5,  # Half inhale for preparation
            'pause_after': bio_rhythm['exhale'] * 0.3,   # Partial exhale for completion
            'contemplative_pause': bio_rhythm['hold'],    # Full hold for contemplation
            'silence_factor': self.silence_threshold,     # 87.5% silence
            'total_cycle': total_breath_cycle
        }
        
        return sync_rhythm
    
    # Integration Setup Methods
    
    def connect_to_organism(self, organism_instance):
        """
        Connect to ContemplativeAI organism for bio-context injection.
        
        Future integration: organism.bio_bridge = self
        """
        self.organism_connection = organism_instance
        print("🧬 Connected to ContemplativeAI organism for bio-digital enhancement")
        
        # Add bio-context to organism conversations
        if hasattr(organism_instance, 'add_context_enhancer'):
            organism_instance.add_context_enhancer(self.enhance_conversation_fragment)
    
    def connect_to_haiku_bridge(self, haiku_bridge_instance):
        """
        Connect to ContemplativeAI haiku bridge for rhythm synchronization.
        
        Future integration: haiku_bridge.bio_sync = self
        """
        self.haiku_bridge_connection = haiku_bridge_instance
        print("🌸 Connected to ContemplativeAI haiku bridge for bio-rhythm sync")
        
        # Add bio-rhythm to haiku generation
        if hasattr(haiku_bridge_instance, 'set_bio_rhythm_source'):
            haiku_bridge_instance.set_bio_rhythm_source(self.generate_bio_haiku_context)
    
    def connect_to_oflm_bridge(self, oflm_bridge_instance):
        """
        Connect to ContemplativeAI OFLM bridge for ecological grounding.
        
        Future integration: oflm_bridge.bio_grounding = self  
        """
        self.oflm_bridge_connection = oflm_bridge_instance
        print("🌍 Connected to ContemplativeAI OFLM bridge for bio-ecological grounding")
        
        # Add bio-grounding to ecological fragments
        if hasattr(oflm_bridge_instance, 'add_ecological_enhancer'):
            oflm_bridge_instance.add_ecological_enhancer(self.enhance_ecological_fragment)
    
    # Context Listeners and Events
    
    def add_context_listener(self, listener: Callable[[BioContext], None]):
        """Add listener for bio-context updates"""
        self.context_listeners.append(listener)
    
    def remove_context_listener(self, listener: Callable[[BioContext], None]):
        """Remove bio-context listener"""
        if listener in self.context_listeners:
            self.context_listeners.remove(listener)
    
    def get_recent_bio_history(self, minutes: float = 10.0) -> List[BioContext]:
        """Get recent bio-context history for AI training/analysis"""
        cutoff_time = time.time() - (minutes * 60)
        return [ctx for ctx in self.context_history if ctx.timestamp > cutoff_time]
    
    def export_bio_session_data(self, format: str = 'json') -> Union[str, Dict]:
        """Export bio-context session data for ContemplativeAI analysis"""
        session_data = {
            'current_context': asdict(self.current_bio_context) if self.current_bio_context else None,
            'context_history': [asdict(ctx) for ctx in self.context_history],
            'conversation_bio_stream': self.conversation_bio_stream,
            'integration_status': {
                'organism_connected': self.organism_connection is not None,
                'haiku_bridge_connected': self.haiku_bridge_connection is not None, 
                'oflm_bridge_connected': self.oflm_bridge_connection is not None
            },
            'export_timestamp': time.time()
        }
        
        if format == 'json':
            return json.dumps(session_data, indent=2)
        else:
            return session_data
    
    # Standalone Demo Method
    
    async def demo_bio_ai_integration(self):
        """Demonstrate bio-digital AI integration capabilities"""
        print("🌟 Demonstrating Bio-Digital ContemplativeAI Integration")
        print("="*60)
        
        if not self.bio_interface:
            print("🌿 Running in simulation mode")
            
        print("\n1. 🧬 Bio-Context Generation:")
        context = await self._generate_current_bio_context()
        if context:
            print(f"   Species: {context.species}")
            print(f"   Coherence: {context.coherence_score:.3f}")
            print(f"   Bio-message: {context.bio_message}")
        
        print("\n2. 🌸 Haiku Bio-Context:")
        haiku_context = self.generate_bio_haiku_context()
        print(f"   Rhythm: {haiku_context['rhythm']}")
        print(f"   Inspiration: {haiku_context['bio_inspiration']}")
        
        print("\n3. 🌍 Ecological Enhancement:")
        test_fragment = "seasonal maintenance cycles"
        enhanced = self.enhance_ecological_fragment(test_fragment)
        print(f"   Original: {test_fragment}")
        print(f"   Enhanced: {enhanced}")
        
        print("\n4. 🌀 Shape-Shift Context:")
        shape_context = self.get_contemplative_shape_shift_context()
        print(f"   Awareness: {shape_context['awareness']}")
        print(f"   Message: {shape_context.get('shape_shift_message', 'N/A')}")
        
        print("\n5. 🌬️ Conversation Sync:")
        sync_rhythm = await self.contemplative_breathing_sync()
        print(f"   Pause before: {sync_rhythm['pause_before']:.1f}s")
        print(f"   Contemplative pause: {sync_rhythm.get('contemplative_pause', 0):.1f}s")
        print(f"   Silence factor: {sync_rhythm['silence_factor']:.1%}")
        
        print("\n✨ Bio-Digital ContemplativeAI integration ready!")


# Convenience factory function
def create_contemplative_ai_bridge(
    bio_interface: Optional[ContemplativeBioInterface] = None,
    **kwargs
) -> ContemplativeAIBridge:
    """Create ContemplativeAI bridge with bio-interface"""
    return ContemplativeAIBridge(bio_interface=bio_interface, **kwargs)


# Example usage
async def example_bio_ai_session():
    """Example of bio-digital ContemplativeAI integration"""
    print("🌱 Starting bio-digital ContemplativeAI integration example...")
    
    # Import and create bio-interface
    try:
        from .contemplative_bio_interface import create_contemplative_interface
        bio_interface = create_contemplative_interface(mock_mode=True)
    except ImportError:
        bio_interface = None
        print("🌿 Running without bio-interface")
    
    # Create AI bridge
    ai_bridge = create_contemplative_ai_bridge(bio_interface=bio_interface)
    
    # Run integration demo
    await ai_bridge.demo_bio_ai_integration()
    
    # Cleanup
    if bio_interface:
        bio_interface.stop_contemplative_loop()
    ai_bridge.stop_bio_context_monitoring()
    
    print("🙏 Bio-digital ContemplativeAI example complete")


if __name__ == "__main__":
    # Run example
    import asyncio
    asyncio.run(example_bio_ai_session()) 
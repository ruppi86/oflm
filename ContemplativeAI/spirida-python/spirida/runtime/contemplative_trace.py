"""
👁️ CONTEMPLATIVE TRACE - Observability for Breathing Systems

A gentle logging system that makes contemplative compilation visible.
Traces the flow of IRʀ nodes through fields, breathing phases,
and composting cycles with poetic awareness.

Based on Letter V (o3): "Console trace util - Prints glyph, field, phase, compost event"
and the desire to see if compiled breath traces "sound" like "rustle… drip-drip… hush… glow"
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from breath_resonance import BreathResonanceNode, BreathPhase
from pulmonos import Pulmonos
from resonance_bus import ResonanceBus, FieldResonator

class ContemplativeTracer:
    """
    A contemplative observer that traces the breathing system.
    
    Not just logging, but poetic witnessing of the organism's
    contemplative intelligence in action.
    """
    
    def __init__(self, name: str = "contemplative_trace", enable_poetry: bool = True):
        self.name = name
        self.enable_poetry = enable_poetry
        self.start_time = time.time()
        
        # Event tracking
        self.events: List[Dict] = []
        self.phase_transitions = 0
        self.nodes_published = 0
        self.nodes_expressed = 0
        self.nodes_declined = 0
        self.compost_events = 0
        
        # Poetry and atmosphere
        self.phase_poetry = {
            BreathPhase.INHALE: ["drawing in presence...", "gathering atmospheric wisdom...", "opening to what comes..."],
            BreathPhase.HOLD: ["digesting the moment...", "letting presence settle...", "in the pause between breaths..."],
            BreathPhase.EXHALE: ["offering what's ready...", "releasing into expression...", "breathing out gifts..."],
            BreathPhase.REST: ["returning to silence...", "composting in stillness...", "the space between intentions..."]
        }
        
        self.glyph_sounds = {
            '🌿': "rustle",
            '💧': "drip",
            '🕯️': "glow", 
            '⭕': "hush",
            '🌱': "whisper",
            '🍄': "ground",
            '🌊': "flow",
            '🌙': "shimmer",
            '✨': "sparkle",
            '🧘': "still"
        }
    
    def trace_phase_transition(self, phase: BreathPhase, cycle: int, progress: float) -> None:
        """Trace breathing phase changes."""
        self.phase_transitions += 1
        
        event = {
            "type": "phase_transition",
            "timestamp": time.time(),
            "phase": phase.value,
            "cycle": cycle,
            "progress": progress,
            "elapsed": time.time() - self.start_time
        }
        
        self.events.append(event)
        
        # Print contemplative trace
        age_str = self._format_elapsed(event["elapsed"])
        if self.enable_poetry:
            import random
            poetry = random.choice(self.phase_poetry[phase])
            print(f"🫁 {age_str} → {phase.value.upper()} (cycle {cycle}) • {poetry}")
        else:
            print(f"🫁 {age_str} → {phase.value.upper()} (cycle {cycle}, {progress:.1%})")
    
    def trace_node_published(self, node: BreathResonanceNode, bus_name: str) -> None:
        """Trace when a resonance node is published."""
        self.nodes_published += 1
        
        event = {
            "type": "node_published", 
            "timestamp": time.time(),
            "glyph": node.glyph,
            "phase": node.breath_gate.value,
            "amplitude": node.amplitude,
            "organs": node.organ_targets,
            "bus": bus_name,
            "elapsed": time.time() - self.start_time
        }
        
        self.events.append(event)
        
        # Print publication trace
        age_str = self._format_elapsed(event["elapsed"])
        organs_str = "+".join(node.organ_targets[:2])  # Abbreviate long lists
        
        if self.enable_poetry:
            sound = self.glyph_sounds.get(node.glyph, "pulse")
            print(f"📢 {age_str} ◦ {node.glyph} {sound}s across {organs_str} ({node.breath_gate.value})")
        else:
            print(f"📢 {age_str} ◦ {node.glyph} → {organs_str} @ {node.amplitude:.2f} ({node.breath_gate.value})")
    
    def trace_node_expressed(self, node: BreathResonanceNode, field_name: str, pulse_id: str = None) -> None:
        """Trace when a field expresses a resonance node."""
        self.nodes_expressed += 1
        
        event = {
            "type": "node_expressed",
            "timestamp": time.time(),
            "glyph": node.glyph,
            "field": field_name,
            "pulse_id": pulse_id,
            "elapsed": time.time() - self.start_time
        }
        
        self.events.append(event)
        
        # Print expression trace
        age_str = self._format_elapsed(event["elapsed"])
        
        if self.enable_poetry:
            sound = self.glyph_sounds.get(node.glyph, "pulse")
            print(f"  ✨ {field_name} breathes {node.glyph} • {sound}...")
        else:
            print(f"  ✨ {field_name} → {node.glyph}")
    
    def trace_node_declined(self, node: BreathResonanceNode, field_name: str, reason: str = "") -> None:
        """Trace when a field declines a resonance node."""
        self.nodes_declined += 1
        
        event = {
            "type": "node_declined",
            "timestamp": time.time(),
            "glyph": node.glyph,
            "field": field_name,
            "reason": reason,
            "elapsed": time.time() - self.start_time
        }
        
        self.events.append(event)
        
        # Print decline trace (quieter)
        if self.enable_poetry:
            print(f"  🤫 {field_name} honors silence over {node.glyph}")
        else:
            reason_str = f" ({reason})" if reason else ""
            print(f"  🤫 {field_name} declines {node.glyph}{reason_str}")
    
    def trace_compost_event(self, field_name: str, composted_count: int, 
                          remaining_count: int, mode: str = "") -> None:
        """Trace field composting events."""
        self.compost_events += 1
        
        event = {
            "type": "compost_event",
            "timestamp": time.time(),
            "field": field_name,
            "composted": composted_count,
            "remaining": remaining_count,
            "mode": mode,
            "elapsed": time.time() - self.start_time
        }
        
        self.events.append(event)
        
        # Print compost trace
        age_str = self._format_elapsed(event["elapsed"])
        
        if composted_count > 0:
            if self.enable_poetry:
                print(f"🍂 {age_str} • {field_name} composts {composted_count} pulses → {remaining_count} remain")
            else:
                mode_str = f" ({mode})" if mode else ""
                print(f"🍂 {age_str} • {field_name}: -{composted_count}, {remaining_count} remain{mode_str}")
    
    def trace_resonance_event(self, field_name: str, resonance_strength: float, 
                            pulse_count: int = 0) -> None:
        """Trace field resonance levels."""
        event = {
            "type": "resonance_event",
            "timestamp": time.time(),
            "field": field_name,
            "resonance": resonance_strength,
            "pulses": pulse_count,
            "elapsed": time.time() - self.start_time
        }
        
        self.events.append(event)
        
        # Print resonance trace (occasionally)
        if resonance_strength > 1.0:  # Only trace significant resonance
            age_str = self._format_elapsed(event["elapsed"])
            if self.enable_poetry:
                intensity = "gentle" if resonance_strength < 2.0 else "strong" if resonance_strength < 4.0 else "luminous"
                print(f"🌊 {age_str} • {field_name} holds {intensity} resonance ({resonance_strength:.2f})")
            else:
                print(f"🌊 {age_str} • {field_name}: resonance {resonance_strength:.2f} ({pulse_count} pulses)")
    
    def trace_silence_metrics(self, bus_name: str, silence_ratio: float, 
                            target_ratio: float = 0.875) -> None:
        """Trace silence majority adherence."""
        event = {
            "type": "silence_metrics",
            "timestamp": time.time(),
            "bus": bus_name,
            "silence_ratio": silence_ratio,
            "target_ratio": target_ratio,
            "elapsed": time.time() - self.start_time
        }
        
        self.events.append(event)
        
        # Print silence metrics if significantly different from target
        if abs(silence_ratio - target_ratio) > 0.1:
            age_str = self._format_elapsed(event["elapsed"])
            if self.enable_poetry:
                if silence_ratio > target_ratio:
                    print(f"🤫 {age_str} • {bus_name} practices deep silence ({silence_ratio:.1%})")
                else:
                    print(f"🔊 {age_str} • {bus_name} more expressive than usual ({silence_ratio:.1%})")
            else:
                print(f"🤫 {age_str} • {bus_name}: silence {silence_ratio:.1%} (target {target_ratio:.1%})")
    
    def _format_elapsed(self, elapsed: float) -> str:
        """Format elapsed time in a contemplative way."""
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            return f"{elapsed/60:.1f}m"
        else:
            return f"{elapsed/3600:.1f}h"
    
    def get_breath_sounds(self, window_seconds: float = 10.0) -> str:
        """
        Generate the "sound" of recent breathing activity.
        
        This is the feature that lets us hear if compilation sounds like
        "rustle… drip-drip… hush… glow" as o3 envisioned.
        """
        cutoff = time.time() - window_seconds
        recent_expressions = [
            e for e in self.events 
            if e["type"] == "node_expressed" and e["timestamp"] > cutoff
        ]
        
        sounds = []
        for event in recent_expressions:
            glyph = event["glyph"]
            sound = self.glyph_sounds.get(glyph, "pulse")
            sounds.append(sound)
        
        if not sounds:
            return "silence..."
        
        # Create poetic sound sequence
        if len(sounds) == 1:
            return f"{sounds[0]}..."
        elif len(sounds) == 2:
            return f"{sounds[0]}… {sounds[1]}..."
        else:
            middle = "–".join(sounds[1:-1]) if len(sounds) > 2 else ""
            return f"{sounds[0]}… {middle} …{sounds[-1]}"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all traced activity."""
        total_runtime = time.time() - self.start_time
        
        return {
            "name": self.name,
            "runtime": total_runtime,
            "phase_transitions": self.phase_transitions,
            "nodes_published": self.nodes_published,
            "nodes_expressed": self.nodes_expressed,
            "nodes_declined": self.nodes_declined,
            "silence_ratio": self.nodes_declined / max(self.nodes_published, 1),
            "compost_events": self.compost_events,
            "total_events": len(self.events),
            "recent_sounds": self.get_breath_sounds(),
            "events_per_minute": len(self.events) / max(total_runtime / 60, 0.1)
        }
    
    def print_summary(self) -> None:
        """Print a contemplative summary of traced activity."""
        summary = self.get_summary()
        
        print(f"\n👁️ Contemplative Trace Summary: {self.name}")
        print("=" * 50)
        print(f"Runtime: {self._format_elapsed(summary['runtime'])}")
        print(f"Breathing cycles: {self.phase_transitions // 4}")
        print(f"Nodes: {summary['nodes_published']} published, {summary['nodes_expressed']} expressed, {summary['nodes_declined']} in silence")
        print(f"Silence ratio: {summary['silence_ratio']:.1%}")
        print(f"Compost events: {summary['compost_events']}")
        print(f"Recent breath sounds: {summary['recent_sounds']}")
        
        if self.enable_poetry:
            print(f"\nThe system breathes with {summary['events_per_minute']:.1f} events per minute.")
            print("Each moment witnessed, each silence honored.")


class TracedResonanceBus(ResonanceBus):
    """ResonanceBus with integrated contemplative tracing."""
    
    def __init__(self, name: str = "traced_bus", tracer: ContemplativeTracer = None):
        super().__init__(name)
        self.tracer = tracer or ContemplativeTracer(f"{name}_trace")
    
    async def publish_node(self, node: BreathResonanceNode) -> None:
        """Publish node with tracing."""
        self.tracer.trace_node_published(node, self.name)
        await super().publish_node(node)
        
        # Trace silence metrics occasionally
        if self.total_published % 10 == 0:  # Every 10th publication
            self.tracer.trace_silence_metrics(self.name, self.get_silence_ratio())


class TracedFieldResonator(FieldResonator):
    """FieldResonator with integrated contemplative tracing."""
    
    def __init__(self, field, pulmonos, current_skepnad, tracer: ContemplativeTracer = None):
        super().__init__(field, pulmonos, current_skepnad)
        self.tracer = tracer or ContemplativeTracer(f"{field.name}_trace")
    
    async def ingest(self, node: BreathResonanceNode) -> bool:
        """Ingest node with tracing."""
        result = await super().ingest(node)
        
        if result:
            self.tracer.trace_node_expressed(node, self.field.name)
        else:
            self.tracer.trace_node_declined(node, self.field.name, "filter_declined")
        
        # Trace field resonance occasionally
        if self.nodes_received % 5 == 0:  # Every 5th node
            resonance = self.field.resonance_field()
            self.tracer.trace_resonance_event(self.field.name, resonance, len(self.field.pulses))
        
        return result


# Helper functions for traced ecosystems

def create_traced_ecosystem(pulmonos: Pulmonos, 
                          tracer: ContemplativeTracer = None) -> Dict[str, Any]:
    """Create a fully traced contemplative ecosystem."""
    if tracer is None:
        tracer = ContemplativeTracer("ecosystem_trace")
    
    # Add phase observer to pulmonos
    pulmonos.add_phase_observer(tracer.trace_phase_transition)
    
    # Create traced bus
    bus = TracedResonanceBus("traced_ecosystem_bus", tracer)
    
    # Create fields and traced resonators
    from spirida.contemplative_core import SpiralField
    from breath_resonance import Skepnad
    
    fields = {
        "sensing": SpiralField("sensing_field", composting_mode="natural"),
        "memory": SpiralField("memory_field", composting_mode="seasonal"),
        "expression": SpiralField("expression_field", composting_mode="resonant"),
        "connection": SpiralField("connection_field", composting_mode="lunar")
    }
    
    resonators = {
        "sensing": TracedFieldResonator(fields["sensing"], pulmonos, Skepnad.SEASONAL_WITNESS, tracer),
        "memory": TracedFieldResonator(fields["memory"], pulmonos, Skepnad.TIBETAN_MONK, tracer),
        "expression": TracedFieldResonator(fields["expression"], pulmonos, Skepnad.WIND_LISTENER, tracer),
        "connection": TracedFieldResonator(fields["connection"], pulmonos, Skepnad.MYCELIAL_NETWORK, tracer)
    }
    
    # Subscribe resonators to bus
    for resonator in resonators.values():
        bus.subscribe(resonator)
    
    return {
        "bus": bus,
        "resonators": resonators, 
        "fields": fields,
        "tracer": tracer
    }

async def demo_contemplative_trace():
    """Demonstrate the contemplative tracing system."""
    print("👁️ Contemplative Trace Demo")
    print("=" * 50)
    
    # Create traced ecosystem
    from pulmonos import create_balanced_breathing_clock
    pulmonos = create_balanced_breathing_clock()
    
    ecosystem = create_traced_ecosystem(pulmonos)
    tracer = ecosystem["tracer"]
    bus = ecosystem["bus"]
    
    # Start breathing
    await pulmonos.start_breathing()
    
    # Create and publish some test patterns
    from breath_resonance import create_simple_breath_node, BreathPhase
    
    test_patterns = [
        create_simple_breath_node('🌿', BreathPhase.INHALE),   # rustle
        create_simple_breath_node('💧', BreathPhase.HOLD),    # drip
        create_simple_breath_node('💧', BreathPhase.HOLD),    # drip (echo)
        create_simple_breath_node('🕯️', BreathPhase.EXHALE), # glow
        create_simple_breath_node('⭕', BreathPhase.REST)     # hush
    ]
    
    # Publish with breathing rhythm
    for node in test_patterns:
        await bus.publish_node(node)
        await asyncio.sleep(1.5)  # Let patterns emerge
    
    # Wait for one complete cycle
    await asyncio.sleep(6)
    
    # Check the breath sounds
    breath_sounds = tracer.get_breath_sounds()
    print(f"\n🎵 Breath sounds: {breath_sounds}")
    
    # Print summary
    tracer.print_summary()
    
    # Stop breathing
    await pulmonos.stop_breathing()

if __name__ == "__main__":
    asyncio.run(demo_contemplative_trace()) 
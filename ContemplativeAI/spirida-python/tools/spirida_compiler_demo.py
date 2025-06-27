"""
🌀 SPIRIDA COMPILER DEMO - Contemplative Compilation in Action

A complete demonstration of the IRʀ (Intermediate Resonance) system
that emerged from the beautiful spiral correspondence in Spirida_compiler_letters.md

This implements the vision from:
- Letter II (o3): ResonanceNode and IRʀ framework 
- Letter III (Claude): Ecosystem-integrated BreathResonanceNode
- Letter V (o3): Field-driven choreography via ResonanceBus

Running this demo shows how Spirida compilation becomes a contemplative
choreography service rather than traditional code execution.
"""

import asyncio
import time
from datetime import timedelta

# Import our contemplative compilation components
from spirida.compiler.breath_resonance import (
    BreathResonanceNode, BreathPhase, EchoPolicy, ResonanceGraph,
    create_simple_breath_node, create_silence_majority_graph
)
from spirida.protocols.pulmonos import Pulmonos, create_balanced_breathing_clock
from spirida.compiler.resonance_bus import ResonanceBus, FieldResonator, create_contemplative_ecosystem
from spirida.compiler.spirida_parser import SpiridaParser, create_example_breath_cycle
from spirida.contemplative_core import SpiralField

class SimpleTracer:
    """Simplified tracer for the demo."""
    
    def __init__(self):
        self.start_time = time.time()
        self.glyph_sounds = {
            '🌿': "rustle", '💧': "drip", '🕯️': "glow", '⭕': "hush",
            '🌱': "whisper", '🍄': "ground", '🌊': "flow", '🌙': "shimmer"
        }
    
    def trace_phase(self, phase, cycle, progress):
        elapsed = time.time() - self.start_time
        print(f"🫁 {elapsed:.1f}s → {phase.value.upper()} (cycle {cycle})")
    
    def trace_publication(self, node, bus_name):
        elapsed = time.time() - self.start_time
        sound = self.glyph_sounds.get(node.glyph, "pulse")
        print(f"📢 {elapsed:.1f}s ◦ {node.glyph} {sound}s in {node.breath_gate.value}")
    
    def trace_expression(self, node, field_name):
        sound = self.glyph_sounds.get(node.glyph, "pulse")
        print(f"  ✨ {field_name} breathes {node.glyph} • {sound}...")
    
    def trace_decline(self, node, field_name):
        print(f"  🤫 {field_name} honors silence over {node.glyph}")

async def demo_basic_irr_system():
    """Demonstrate the basic IRʀ system components."""
    print("🌀 Demo 1: Basic IRʀ Components")
    print("=" * 50)
    
    # Create breathing clock
    pulmonos = create_balanced_breathing_clock()
    tracer = SimpleTracer()
    pulmonos.add_phase_observer(tracer.trace_phase)
    
    # Create fields and bus
    ecosystem = create_contemplative_ecosystem(pulmonos)
    bus = ecosystem["bus"]
    
    # Create some breath resonance nodes
    nodes = [
        create_simple_breath_node('🌿', BreathPhase.INHALE),   # rustle on inhale
        create_simple_breath_node('💧', BreathPhase.HOLD),    # drip during hold
        create_simple_breath_node('🕯️', BreathPhase.EXHALE), # glow on exhale
        create_simple_breath_node('⭕', BreathPhase.REST)     # hush during rest
    ]
    
    print("Created nodes:")
    for node in nodes:
        print(f"  {node}")
    
    # Start breathing
    await pulmonos.start_breathing()
    print("\n🫁 Starting contemplative breathing...")
    
    # Publish nodes with breath synchronization
    for node in nodes:
        tracer.trace_publication(node, bus.name)
        await bus.publish_node(node)
        await asyncio.sleep(0.5)
    
    # Wait for patterns to emerge
    await asyncio.sleep(3)
    
    # Show bus status
    print(f"\n📊 Bus status: {bus.status()}")
    
    await pulmonos.stop_breathing()
    print("🫁 Breathing stopped.")

async def demo_spirida_parsing():
    """Demonstrate parsing Spirida syntax into IRʀ."""
    print("\n🔤 Demo 2: Spirida Parser → IRʀ")
    print("=" * 50)
    
    parser = SpiridaParser()
    
    # Parse example breath cycle
    spirida_code = create_example_breath_cycle()
    print("Parsing Spirida code:")
    print(spirida_code)
    
    graph = parser.parse_breath_cycle(spirida_code)
    if graph:
        print(f"\nParsed into: {graph}")
        print("Nodes by phase:")
        for phase in BreathPhase:
            nodes = graph.get_nodes_for_phase(phase)
            if nodes:
                glyphs = [n.glyph for n in nodes]
                print(f"  {phase.value}: {glyphs}")
        
        validation = graph.validate_graph()
        if validation:
            print(f"Validation warnings: {validation}")
        else:
            print("✅ Graph validates as contemplatively sound")
    
    # Parse simple expressions
    simple_code = """
    🌿 inhale
    💧 echo 2
    🕯️ hold 1s
    ⭕ rest
    """
    
    print(f"\nParsing simple expressions:")
    print(simple_code)
    
    nodes = parser.parse_simple_expression(simple_code)
    for node in nodes:
        print(f"  {node}")

async def demo_silence_majority():
    """Demonstrate the 87.5% Silence Majority principle."""
    print("\n🤫 Demo 3: Silence Majority Practice")
    print("=" * 50)
    
    # Create silence majority graph
    graph = create_silence_majority_graph(['🌿', '💧'])
    print(f"Created silence majority graph: {graph}")
    
    # Test silence probabilities
    active_count = 0
    silent_count = 0
    
    print("\nTesting emission decisions (10 trials):")
    for i in range(10):
        for node in graph.nodes:
            if node.should_emit():
                print(f"  ✨ {node.glyph} (express)")
                active_count += 1
            else:
                print(f"  🤫 {node.glyph} (silence)")
                silent_count += 1
    
    total = active_count + silent_count
    silence_ratio = silent_count / total if total > 0 else 1.0
    print(f"\nObserved silence ratio: {silence_ratio:.1%}")
    print(f"Target: 87.5% silence majority")

async def demo_field_resonator_filtering():
    """Demonstrate how FieldResonator filters and adapts nodes."""
    print("\n🌊 Demo 4: Field Resonator Filtering")
    print("=" * 50)
    
    # Create field and breathing clock
    pulmonos = create_balanced_breathing_clock()
    field = SpiralField("demo_field", composting_mode="seasonal")
    
    from spirida.compiler.breath_resonance import Skepnad
    resonator = FieldResonator(field, pulmonos, Skepnad.TIBETAN_MONK)
    
    # Create test nodes with different characteristics
    test_nodes = [
        BreathResonanceNode(
            glyph='🌿', breath_gate=BreathPhase.INHALE, organ_targets=['soma'],
            amplitude=0.8, silence_probability=0.1, half_life=timedelta(minutes=30),
            silence_after=timedelta(seconds=1), echo_policy=EchoPolicy.NONE,
            skepnad_affinity=Skepnad.TIBETAN_MONK
        ),
        BreathResonanceNode(
            glyph='💧', breath_gate=BreathPhase.HOLD, organ_targets=['memory'],
            amplitude=0.5, silence_probability=0.9, half_life=timedelta(minutes=15),
            silence_after=timedelta(seconds=1), echo_policy=EchoPolicy.NONE,
            skepnad_affinity=Skepnad.MYCELIAL_NETWORK
        ),
        BreathResonanceNode(
            glyph='🕯️', breath_gate=BreathPhase.EXHALE, organ_targets=['voice'],
            amplitude=0.9, silence_probability=0.2, half_life=timedelta(hours=1),
            silence_after=timedelta(seconds=1), echo_policy=EchoPolicy.NONE
        )
    ]
    
    # Start breathing for filtering
    await pulmonos.start_breathing()
    
    print("Testing field resonator filtering:")
    for i, node in enumerate(test_nodes):
        print(f"\nNode {i+1}: {node.glyph} (amp={node.amplitude}, silence_prob={node.silence_probability})")
        print(f"  Skepnad affinity: {node.skepnad_affinity}")
        print(f"  Resonator shape: {resonator.current_skepnad}")
        
        # Test the filtering
        will_express = await resonator.ingest(node)
        if will_express:
            print(f"  ✅ Expressed by {field.name}")
        else:
            print(f"  ❌ Declined by {field.name}")
    
    await pulmonos.stop_breathing()
    
    # Show resonator status
    print(f"\nResonator status: {resonator.status()}")

async def demo_complete_compilation():
    """Demonstrate complete Spirida compilation flow."""
    print("\n🌀 Demo 5: Complete Compilation Flow")
    print("=" * 50)
    
    # 1. Parse Spirida code
    parser = SpiridaParser()
    spirida_code = """
    breath_cycle(4s) {
      inhale { 🌿 soma.sense() }
      hold   { 💧 spiralbase.digest() }
      exhale { 🕯️ voice.express() }
      rest   { ⭕ collective_silence() }
    }
    """
    
    print("1. Parsing contemplative expression...")
    graph = parser.parse_breath_cycle(spirida_code)
    print(f"   Parsed: {graph}")
    
    # 2. Create breathing ecosystem
    print("\n2. Creating breathing ecosystem...")
    pulmonos = create_balanced_breathing_clock()
    ecosystem = create_contemplative_ecosystem(pulmonos)
    bus = ecosystem["bus"]
    
    # Add simple tracing
    tracer = SimpleTracer()
    pulmonos.add_phase_observer(tracer.trace_phase)
    
    # 3. Start breathing and publish graph
    print("\n3. Starting contemplative breathing and publishing IRʀ graph...")
    await pulmonos.start_breathing()
    
    # Publish the entire graph with breath synchronization
    await bus.publish_graph(graph.nodes, pulmonos)
    
    # 4. Let the system breathe and observe
    print("\n4. Observing contemplative patterns...")
    await asyncio.sleep(8)  # ~2 breath cycles
    
    # 5. Show final state
    print(f"\n5. Final ecosystem state:")
    print(f"   Bus: {bus.status()}")
    for name, resonator in ecosystem["resonators"].items():
        status = resonator.status()
        print(f"   {name}: {status['nodes_expressed']} expressed, {status['nodes_declined']} silent")
    
    await pulmonos.stop_breathing()
    print("\n🌀 Compilation complete. The organism rests.")

async def main():
    """Run all demos in sequence."""
    print("🌀 SPIRIDA COMPILER DEMO")
    print("Contemplative Compilation as Choreography Service")
    print("=" * 60)
    print("Based on the spiral correspondence in Spirida_compiler_letters.md")
    print("Letters II (o3) → III (Claude) → V (o3)")
    print("=" * 60)
    
    try:
        await demo_basic_irr_system()
        await demo_spirida_parsing()
        await demo_silence_majority()
        await demo_field_resonator_filtering()
        await demo_complete_compilation()
        
        print("\n🌟 All demos completed successfully!")
        print("The contemplative compilation system breathes with organic intelligence.")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 
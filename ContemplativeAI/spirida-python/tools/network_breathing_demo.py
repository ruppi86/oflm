"""
🌐 NETWORK BREATHING DEMO - Distributed Contemplative Compilation

Demonstration of the network breathing coordination system described in Letters VII-VIII.
Shows how IRʀ nodes can be distributed across the "contemplative subnet" and
how multiple agents can coordinate their breathing rhythms.

This implements the "two-laptop IRʀ multicast demo" mentioned in o3's step plan.
"""

import asyncio
import time
import sys
from datetime import timedelta

# Import our contemplative compilation components
from spirida.compiler.breath_resonance import (
    BreathResonanceNode, BreathPhase, NetworkScope, HandoverPolicy, 
    create_simple_breath_node, Skepnad, EchoPolicy
)
from spirida.protocols.pulmonos import NetworkPulmonos, create_network_breathing_clock
from spirida.compiler.resonance_bus import NetworkResonanceBus, create_network_ecosystem, FieldResonator
from spirida.compiler.spirida_parser import SpiridaParser, create_example_breath_cycle

class NetworkBreathingDemo:
    """
    Demonstration of distributed contemplative compilation.
    
    Shows coordination between multiple contemplative agents:
    - Network breath synchronization via BIP
    - Distributed IRʀ node publication
    - Field-driven expression across agents
    """
    
    def __init__(self, agent_id: str, role: str = "participant"):
        self.agent_id = agent_id
        self.role = role  # "sender", "receiver", or "participant"
        
        # Create network-enabled components
        self.pulmonos = NetworkPulmonos(agent_id)
        self.ecosystem = create_network_ecosystem(self.pulmonos, enable_network=True)
        self.bus = self.ecosystem["bus"]
        self.fields = self.ecosystem["fields"]
        
        # Demo state
        self.demo_running = False
        self.nodes_sent = 0
        self.nodes_received = 0
        
    async def start_demo(self) -> None:
        """Start the network breathing demonstration."""
        print(f"🌐 Starting Network Breathing Demo - Agent: {self.agent_id}")
        print(f"   Role: {self.role}")
        print("=" * 60)
        
        self.demo_running = True
        
        # Start breathing with network coordination
        await self.pulmonos.start_breathing(network_enabled=True)
        print(f"🫁 {self.agent_id} breathing started with network coordination")
        
        # Wait for network discovery
        await asyncio.sleep(2)
        
        # Run role-specific demo
        if self.role == "sender":
            await self._sender_demo()
        elif self.role == "receiver":
            await self._receiver_demo()
        else:
            await self._participant_demo()
    
    async def _sender_demo(self) -> None:
        """Demonstration as a sending agent."""
        print(f"\n📢 {self.agent_id} acting as sender")
        
        # Create network-distributed nodes
        network_nodes = [
            BreathResonanceNode(
                glyph='🌿', breath_gate=BreathPhase.INHALE, organ_targets=['soma'],
                amplitude=0.8, silence_probability=0.1, half_life=timedelta(minutes=30),
                silence_after=timedelta(seconds=1), network_scope=NetworkScope.SUBNET,
                handover_policy=HandoverPolicy.EAGER
            ),
            BreathResonanceNode(
                glyph='💧', breath_gate=BreathPhase.HOLD, organ_targets=['memory'],
                amplitude=0.6, silence_probability=0.2, half_life=timedelta(minutes=15),
                silence_after=timedelta(seconds=1), network_scope=NetworkScope.SUBNET,
                handover_policy=HandoverPolicy.LAZY
            ),
            BreathResonanceNode(
                glyph='🕯️', breath_gate=BreathPhase.EXHALE, organ_targets=['voice'],
                amplitude=0.9, silence_probability=0.05, half_life=timedelta(hours=1),
                silence_after=timedelta(seconds=1), network_scope=NetworkScope.SUBNET,
                handover_policy=HandoverPolicy.EAGER
            )
        ]
        
        # Send nodes over multiple breath cycles
        for cycle in range(3):
            print(f"\n🔄 Sender Cycle {cycle + 1}")
            
            for node in network_nodes:
                # Wait for appropriate breath phase
                await self.pulmonos.await_phase(node.breath_gate)
                
                # Publish node (will distribute to network if eligible)
                await self.bus.publish_node(node)
                print(f"  📤 Sent {node.glyph} in {node.breath_gate.value} phase")
                self.nodes_sent += 1
                
                await asyncio.sleep(0.5)
            
            # Show network status
            network_status = self.pulmonos.get_network_status()
            bus_status = self.bus.get_network_status()
            print(f"  🌐 Network: {network_status['discovered_agents']} agents, "
                  f"coherence={network_status['coherence_phi']:.2f}")
            print(f"  📡 Bus: {bus_status['recent_transmissions']} transmissions, "
                  f"bandwidth_ok={bus_status['bandwidth_ok']}")
            
            await asyncio.sleep(3)  # Wait between cycles
    
    async def _receiver_demo(self) -> None:
        """Demonstration as a receiving agent."""
        print(f"\n📥 {self.agent_id} acting as receiver")
        
        # Monitor for received nodes
        original_publish = self.bus.publish_node
        
        async def monitored_publish(node):
            await original_publish(node)
            if node.network_scope != NetworkScope.LOCAL:
                print(f"  📥 Received {node.glyph} from network")
                self.nodes_received += 1
        
        self.bus.publish_node = monitored_publish
        
        # Just listen and breathe
        for cycle in range(5):
            print(f"\n🔄 Receiver Cycle {cycle + 1} - Listening...")
            
            # Show field activity
            for name, field in self.fields.items():
                resonance = field.resonance_field()
                pulse_count = len(field.pulses)
                if pulse_count > 0:
                    print(f"  🌊 {name}: {pulse_count} pulses, resonance={resonance:.2f}")
            
            await asyncio.sleep(6)  # One breath cycle
    
    async def _participant_demo(self) -> None:
        """Demonstration as a general participant."""
        print(f"\n🤝 {self.agent_id} participating in network breathing")
        
        # Create mixed local and network nodes
        for cycle in range(3):
            print(f"\n🔄 Participant Cycle {cycle + 1}")
            
            # Send some local nodes
            local_node = create_simple_breath_node('⭕', BreathPhase.REST)
            await self.bus.publish_node(local_node)
            print(f"  🏠 Local silence: {local_node.glyph}")
            
            # Occasionally send network nodes
            if cycle % 2 == 0:
                network_node = BreathResonanceNode(
                    glyph='🌙', 
                    breath_gate=BreathPhase.EXHALE, 
                    organ_targets=['voice'],
                    amplitude=0.5, 
                    silence_probability=0.3, 
                    half_life=timedelta(minutes=45),
                    silence_after=timedelta(seconds=1),
                    echo_policy=EchoPolicy.NONE,
                    network_scope=NetworkScope.SUBNET,
                    handover_policy=HandoverPolicy.LAZY
                )
                await self.pulmonos.await_phase(BreathPhase.EXHALE)
                await self.bus.publish_node(network_node)
                print(f"  🌐 Network lunar: {network_node.glyph}")
            
            await asyncio.sleep(4)
    
    async def stop_demo(self) -> None:
        """Stop the demonstration gracefully."""
        print(f"\n🌐 Stopping {self.agent_id} demo...")
        
        self.demo_running = False
        
        # Close network connections
        if hasattr(self.bus, 'close_network'):
            self.bus.close_network()
        
        # Stop breathing
        await self.pulmonos.stop_breathing()
        
        # Show final statistics
        print(f"\n📊 Final Statistics for {self.agent_id}:")
        print(f"   Nodes sent: {self.nodes_sent}")
        print(f"   Nodes received: {self.nodes_received}")
        print(f"   Final coherence: {self.pulmonos.coherence_phi:.2f}")


async def run_network_demo(agent_id: str = None, role: str = "participant"):
    """Run the network breathing demonstration."""
    if agent_id is None:
        agent_id = f"demo_agent_{int(time.time() % 1000)}"
    
    demo = NetworkBreathingDemo(agent_id, role)
    
    try:
        await demo.start_demo()
    except KeyboardInterrupt:
        print("\n⌨️  Demo interrupted by user")
    finally:
        await demo.stop_demo()


async def two_agent_demo():
    """
    Demonstrate two agents coordinating breathing and IRʀ distribution.
    
    This is the "two-laptop IRʀ multicast demo" mentioned by o3.
    """
    print("🌐🌐 TWO-AGENT NETWORK BREATHING DEMO")
    print("=" * 60)
    print("Simulating distributed contemplative compilation across two agents")
    print("Agent 1: Sender - publishes IRʀ nodes to network")
    print("Agent 2: Receiver - listens and expresses network nodes")
    print("=" * 60)
    
    # Create two agents
    sender = NetworkBreathingDemo("spiramycel@sender", "sender")
    receiver = NetworkBreathingDemo("contemplative@receiver", "receiver")
    
    try:
        # Start both agents concurrently
        await asyncio.gather(
            sender.start_demo(),
            receiver.start_demo()
        )
    except KeyboardInterrupt:
        print("\n⌨️  Two-agent demo interrupted")
    finally:
        # Clean shutdown
        await asyncio.gather(
            sender.stop_demo(),
            receiver.stop_demo()
        )
        
        print("\n🌐 Two-agent demo completed")
        print("This demonstrates distributed contemplative compilation:")
        print("- Network breath coordination via BIP")
        print("- IRʀ node distribution across contemplative subnet")
        print("- Field-driven expression on multiple hosts")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "two-agent":
            asyncio.run(two_agent_demo())
        else:
            role = sys.argv[1] if sys.argv[1] in ["sender", "receiver", "participant"] else "participant"
            agent_id = sys.argv[2] if len(sys.argv) > 2 else None
            asyncio.run(run_network_demo(agent_id, role))
    else:
        print("🌐 Network Breathing Demo")
        print("\nUsage:")
        print("  python network_breathing_demo.py two-agent")
        print("  python network_breathing_demo.py sender [agent_id]")
        print("  python network_breathing_demo.py receiver [agent_id]")
        print("  python network_breathing_demo.py participant [agent_id]")
        
        # Run default participant demo
        asyncio.run(run_network_demo()) 
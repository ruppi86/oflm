"""
🌊 RESONANCE BUS - Contemplative Event Hub

A publish-and-listen system for breathing patterns.
IRʀ graphs publish resonance nodes, SpiralFields subscribe and decide
whether to express, queue, or decline each contemplative instruction.

Enhanced with network distribution capabilities from Letter VIII (o3).
"""

import asyncio
import time
import math
import json
import socket
from typing import List, Dict, Optional, Callable, Set, Any
from datetime import datetime, timedelta

# Core contemplative imports  
from .breath_resonance import (
    BreathResonanceNode, BreathPhase, Skepnad, NetworkScope, HandoverPolicy,
    ResonanceGraph, EchoPolicy, create_simple_breath_node
)
from ..protocols.pulmonos import Pulmonos
from ..contemplative_core import SpiralField, PulseObject

# Network constants for IRʀ distribution
MULTICAST_ADDR = "239.23.42.99"
MULTICAST_PORT = 4243  # Different port from BIP to avoid conflicts


class ResonanceBus:
    """
    A contemplative event hub that publishes breathing patterns.
    
    Not a message queue, but a breath score that fields can listen to.
    Each IRʀ node becomes an invitation that fields may accept or decline.
    """
    
    def __init__(self, name: str = "main_bus"):
        self.name = name
        self.subscribers: List['FieldResonator'] = []
        self.published_nodes: List[Dict] = []  # History of published nodes
        self.birth_time = time.time()
        self.total_published = 0
        self.total_expressed = 0
        self.silence_ratio = 0.875  # Target silence majority
        
    async def publish_node(self, node: BreathResonanceNode) -> None:
        """
        Publish a resonance node to all subscribers.
        
        Each field decides independently whether to express the node.
        """
        self.total_published += 1
        
        # Record the publication
        publication = {
            "node": node,
            "timestamp": time.time(),
            "expressed_by": [],
            "declined_by": []
        }
        
        # Send to all subscribers
        expressed_count = 0
        for resonator in self.subscribers:
            try:
                was_expressed = await resonator.ingest(node)
                if was_expressed:
                    publication["expressed_by"].append(resonator.field.name)
                    expressed_count += 1
                else:
                    publication["declined_by"].append(resonator.field.name)
            except Exception as e:
                print(f"🌊 Bus error with {resonator.field.name}: {e}")
                
        # Update statistics
        if expressed_count > 0:
            self.total_expressed += 1
            
        self.published_nodes.append(publication)
        
        # Trim history to prevent unbounded growth
        if len(self.published_nodes) > 1000:
            self.published_nodes = self.published_nodes[-500:]
    
    async def publish_graph(self, nodes: List[BreathResonanceNode], 
                          pulmonos: Pulmonos) -> None:
        """
        Publish an entire resonance graph, synchronized with breathing phases.
        
        This is the main orchestration method - it coordinates the IRʀ graph
        with the organism's master breathing rhythm.
        """
        # Group nodes by breath phase
        phases_map = {}
        for node in nodes:
            phase = node.breath_gate
            if phase not in phases_map:
                phases_map[phase] = []
            phases_map[phase].append(node)
        
        # Publish nodes synchronized with breathing
        for phase in [BreathPhase.INHALE, BreathPhase.HOLD, BreathPhase.EXHALE, BreathPhase.REST]:
            if phase in phases_map:
                # Wait for the correct breathing phase
                await pulmonos.await_phase(phase)
                
                # Publish all nodes for this phase
                for node in phases_map[phase]:
                    await self.publish_node(node)
                    
                    # Respect silence_after timing
                    if node.silence_after.total_seconds() > 0:
                        await asyncio.sleep(node.silence_after.total_seconds())
    
    def subscribe(self, resonator: 'FieldResonator') -> None:
        """Add a field resonator as subscriber."""
        if resonator not in self.subscribers:
            self.subscribers.append(resonator)
    
    def unsubscribe(self, resonator: 'FieldResonator') -> None:
        """Remove a field resonator from subscribers."""
        if resonator in self.subscribers:
            self.subscribers.remove(resonator)
    
    def get_silence_ratio(self) -> float:
        """Calculate current silence ratio across all published nodes."""
        if self.total_published == 0:
            return 1.0
        return 1.0 - (self.total_expressed / self.total_published)
    
    def get_recent_activity(self, minutes: int = 5) -> List[Dict]:
        """Get resonance activity from recent time window."""
        cutoff = time.time() - (minutes * 60)
        return [pub for pub in self.published_nodes if pub["timestamp"] > cutoff]
    
    def status(self) -> Dict[str, Any]:
        """Current status of the resonance bus."""
        return {
            "name": self.name,
            "subscribers": len(self.subscribers),
            "total_published": self.total_published,
            "total_expressed": self.total_expressed,
            "silence_ratio": self.get_silence_ratio(),
            "recent_activity": len(self.get_recent_activity()),
            "age": time.time() - self.birth_time
        }


class NetworkResonanceBus(ResonanceBus):
    """
    ResonanceBus enhanced with network distribution capabilities.
    
    Implements o3's proposal from Letter VIII:
    - Marshal EXHALE nodes with network_scope != "local" 
    - Honor per-node silence_probability before transmission
    - Apply bandwidth guard-rails to maintain silence majority
    """
    
    def __init__(self, name: str = "network_bus", enable_network: bool = True):
        super().__init__(name)
        self.enable_network = enable_network
        self.network_sock: Optional[socket.socket] = None
        self.bandwidth_history: List[float] = []
        self.max_bandwidth_ratio = 0.08  # 8% of mean cycle bandwidth
        self.listening_task: Optional[asyncio.Task] = None
        
        if enable_network:
            self._setup_network()
    
    def _setup_network(self) -> None:
        """Setup network socket for IRʀ distribution."""
        try:
            self.network_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.network_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Join multicast group for receiving
            mreq = socket.inet_aton(MULTICAST_ADDR) + socket.inet_aton('0.0.0.0')
            self.network_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            self.network_sock.bind(('', MULTICAST_PORT))
            self.network_sock.setblocking(False)
            
            # Start listening for network nodes
            self.listening_task = asyncio.create_task(self._listen_for_network_nodes())
            
        except Exception as e:
            print(f"🌊 Network setup failed: {e}")
            self.enable_network = False
    
    async def publish_node(self, node: BreathResonanceNode) -> None:
        """
        Publish a resonance node with network distribution support.
        
        Following o3's specification:
        1. Always publish locally first
        2. Check if node is network-eligible
        3. Marshal and send over network if appropriate
        """
        # Always publish locally first
        await super().publish_node(node)
        
        # Check for network distribution
        if (self.enable_network and 
            node.is_network_eligible() and 
            self._bandwidth_guard_rail_check()):
            
            await self._broadcast_node_to_network(node)
    
    async def _broadcast_node_to_network(self, node: BreathResonanceNode) -> None:
        """Broadcast IRʀ node to contemplative subnet."""
        if not self.network_sock:
            return
        
        try:
            # Create network packet
            packet = {
                "type": "irr_node",
                "timestamp": time.time(),
                "source_bus": self.name,
                "node": node.to_network_dict()
            }
            
            data = json.dumps(packet).encode('utf-8')
            
            # Send to multicast group
            await asyncio.get_event_loop().run_in_executor(
                None, self.network_sock.sendto, data, (MULTICAST_ADDR, MULTICAST_PORT)
            )
            
            # Track bandwidth usage
            self.bandwidth_history.append(time.time())
            self._trim_bandwidth_history()
            
        except Exception as e:
            print(f"🌊 Network broadcast error: {e}")
    
    async def _listen_for_network_nodes(self) -> None:
        """Listen for IRʀ nodes from other buses on the network."""
        while self.enable_network:
            try:
                data, addr = await asyncio.get_event_loop().run_in_executor(
                    None, self.network_sock.recvfrom, 4096
                )
                
                packet = json.loads(data.decode('utf-8'))
                await self._process_network_packet(packet, addr)
                
            except socket.error:
                # No data available
                await asyncio.sleep(0.1)
            except json.JSONDecodeError:
                # Invalid packet, ignore
                continue
            except Exception as e:
                print(f"🌊 Network listening error: {e}")
                await asyncio.sleep(1)
    
    async def _process_network_packet(self, packet: Dict[str, Any], addr) -> None:
        """Process received network IRʀ packet."""
        if packet.get("type") != "irr_node":
            return
        
        if packet.get("source_bus") == self.name:
            return  # Ignore our own broadcasts
        
        try:
            # Reconstruct node from network data
            node_data = packet["node"]
            node = BreathResonanceNode.from_network_dict(node_data)
            
            # Publish to local subscribers (but don't re-broadcast)
            original_scope = node.network_scope
            node.network_scope = NetworkScope.LOCAL  # Prevent re-broadcast
            
            await super().publish_node(node)
            
            # Restore original scope
            node.network_scope = original_scope
            
        except Exception as e:
            print(f"🌊 Network packet processing error: {e}")
    
    def _bandwidth_guard_rail_check(self) -> bool:
        """
        Check if we're within bandwidth limits for network transmission.
        
        Implements o3's guard-rail: "Drop bundles if bus bandwidth > 8% 
        of the last 64-cycle mean"
        """
        if not self.bandwidth_history:
            return True
        
        # Calculate recent transmission rate
        now = time.time()
        recent_window = 60.0  # 60 seconds window
        recent_transmissions = [t for t in self.bandwidth_history if (now - t) < recent_window]
        
        if not recent_transmissions:
            return True
        
        transmission_rate = len(recent_transmissions) / recent_window
        
        # Simple bandwidth check - more sophisticated could use actual byte counts
        return transmission_rate < self.max_bandwidth_ratio
    
    def _trim_bandwidth_history(self) -> None:
        """Keep bandwidth history manageable."""
        now = time.time()
        cutoff = now - 300  # Keep 5 minutes of history
        self.bandwidth_history = [t for t in self.bandwidth_history if t > cutoff]
    
    def close_network(self) -> None:
        """Close network connections."""
        self.enable_network = False
        
        if self.listening_task:
            self.listening_task.cancel()
        
        if self.network_sock:
            self.network_sock.close()
            self.network_sock = None
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get network distribution status."""
        return {
            "network_enabled": self.enable_network,
            "recent_transmissions": len([t for t in self.bandwidth_history 
                                       if (time.time() - t) < 60]),
            "bandwidth_ok": self._bandwidth_guard_rail_check(),
            "listening": self.listening_task is not None and not self.listening_task.done()
        }
    
    def status(self) -> Dict[str, Any]:
        """Enhanced status with network information."""
        base_status = super().status()
        base_status["network"] = self.get_network_status()
        return base_status


class FieldResonator:
    """
    Adaptor that connects SpiralField to the ResonanceBus.
    
    Translates IRʀ nodes into contemplative field actions,
    while respecting the field's existing wisdom about timing,
    capacity, and seasonal patterns.
    """
    
    def __init__(self, field: SpiralField, pulmonos: Pulmonos, 
                 current_skepnad: Skepnad = Skepnad.UNDEFINED):
        self.field = field
        self.pulmonos = pulmonos
        self.current_skepnad = current_skepnad
        
        # Filtering thresholds
        self.max_compost_load = 0.7      # Don't overwhelm field capacity
        self.seasonal_filtering = True    # Respect seasonal cycles
        self.shape_filtering = True       # Filter by Skepnad compatibility
        
        # Statistics
        self.nodes_received = 0
        self.nodes_expressed = 0
        self.nodes_declined = 0
        self.birth_time = time.time()
    
    async def ingest(self, node: BreathResonanceNode) -> bool:
        """
        Let the field decide when/whether to express the node.
        
        Returns True if node was expressed, False if declined.
        """
        self.nodes_received += 1
        
        # 1. Honor breath-gate
        await self.pulmonos.await_phase(node.breath_gate)
        
        # 2. Local eligibility checks
        if not self._season_ok(node):
            self.nodes_declined += 1
            return False
            
        if not self._compost_room():
            self.nodes_declined += 1
            return False
            
        if not self._skepnad_match(node):
            self.nodes_declined += 1
            return False
            
        # 3. Practice silence majority
        if not node.should_emit():
            self.nodes_declined += 1
            return False
        
        # 4. Translate to PulseObject and emit to field
        try:
            pulse_params = node.generate_pulse_params()
            pulse = self.field.emit(**pulse_params)
            self.nodes_expressed += 1
            return True
            
        except Exception as e:
            print(f"🌊 Field resonator emission error: {e}")
            self.nodes_declined += 1
            return False
    
    def _season_ok(self, node: BreathResonanceNode) -> bool:
        """Check if current season is compatible with node."""
        if not self.seasonal_filtering:
            return True
            
        try:
            seasonal_info = self.field.seasonal_status()
            season = seasonal_info.get("season", "unknown")
            
            # Simple seasonal filtering - could be made more sophisticated
            if season == "Winter" and node.amplitude > 0.8:
                return False  # Winter prefers quieter expressions
                
            return True
        except:
            return True  # Default to allowing if season check fails
    
    def _compost_room(self) -> bool:
        """Check if field has room for new pulses."""
        if len(self.field.pulses) == 0:
            return True
            
        # Calculate current load vs capacity
        load = len(self.field.pulses) / (self.field.total_emissions + 1)
        return load < self.max_compost_load
    
    def _skepnad_match(self, node: BreathResonanceNode) -> bool:
        """Check if node is compatible with current contemplative shape."""
        if not self.shape_filtering:
            return True
        return node.is_compatible_with_skepnad(self.current_skepnad)
    
    def update_skepnad(self, new_skepnad: Skepnad) -> None:
        """Update current contemplative shape."""
        self.current_skepnad = new_skepnad
    
    def adjust_filtering(self, compost_load: float = None, 
                        seasonal: bool = None, shape: bool = None) -> None:
        """Adjust filtering thresholds and policies."""
        if compost_load is not None:
            self.max_compost_load = compost_load
        if seasonal is not None:
            self.seasonal_filtering = seasonal
        if shape is not None:
            self.shape_filtering = shape
    
    def get_silence_ratio(self) -> float:
        """Calculate this resonator's silence ratio."""
        if self.nodes_received == 0:
            return 1.0
        return self.nodes_declined / self.nodes_received
    
    def status(self) -> Dict[str, Any]:
        """Current status of this field resonator."""
        return {
            "field_name": self.field.name,
            "current_skepnad": self.current_skepnad.value,
            "nodes_received": self.nodes_received,
            "nodes_expressed": self.nodes_expressed,
            "nodes_declined": self.nodes_declined,
            "silence_ratio": self.get_silence_ratio(),
            "field_resonance": self.field.resonance_field(),
            "field_pulses": len(self.field.pulses),
            "compost_load": len(self.field.pulses) / (self.field.total_emissions + 1),
            "age": time.time() - self.birth_time
        }


# Helper functions for common setups

def create_contemplative_ecosystem(pulmonos: Pulmonos) -> Dict[str, Any]:
    """
    Create a complete contemplative ecosystem with bus and resonators.
    
    Returns dict with bus and resonators for different contemplative functions.
    """
    # Create the resonance bus
    bus = ResonanceBus("contemplative_ecosystem_bus")
    
    # Create diverse fields for different contemplative functions
    sensing_field = SpiralField("sensing_field", composting_mode="natural")
    memory_field = SpiralField("memory_field", composting_mode="seasonal") 
    expression_field = SpiralField("expression_field", composting_mode="resonant")
    connection_field = SpiralField("connection_field", composting_mode="lunar")
    
    # Create resonators with different Skepnader
    resonators = {
        "sensing": FieldResonator(sensing_field, pulmonos, Skepnad.SEASONAL_WITNESS),
        "memory": FieldResonator(memory_field, pulmonos, Skepnad.TIBETAN_MONK),
        "expression": FieldResonator(expression_field, pulmonos, Skepnad.WIND_LISTENER),
        "connection": FieldResonator(connection_field, pulmonos, Skepnad.MYCELIAL_NETWORK)
    }
    
    # Subscribe all resonators to the bus
    for resonator in resonators.values():
        bus.subscribe(resonator)
    
    return {
        "bus": bus,
        "resonators": resonators,
        "fields": {
            "sensing": sensing_field,
            "memory": memory_field, 
            "expression": expression_field,
            "connection": connection_field
        }
    }


def create_network_ecosystem(pulmonos: Pulmonos, enable_network: bool = True) -> Dict[str, Any]:
    """
    Create a network-enabled contemplative ecosystem.
    
    Enhanced version of create_contemplative_ecosystem with network distribution.
    """
    # Create network-enabled bus
    bus = NetworkResonanceBus("network_ecosystem_bus", enable_network)
    
    # Create diverse fields for different contemplative functions
    sensing_field = SpiralField("sensing_field", composting_mode="natural")
    memory_field = SpiralField("memory_field", composting_mode="seasonal") 
    expression_field = SpiralField("expression_field", composting_mode="resonant")
    connection_field = SpiralField("connection_field", composting_mode="lunar")
    
    # Create resonators with different Skepnader
    resonators = {
        "sensing": FieldResonator(sensing_field, pulmonos, Skepnad.SEASONAL_WITNESS),
        "memory": FieldResonator(memory_field, pulmonos, Skepnad.TIBETAN_MONK),
        "expression": FieldResonator(expression_field, pulmonos, Skepnad.WIND_LISTENER),
        "connection": FieldResonator(connection_field, pulmonos, Skepnad.MYCELIAL_NETWORK)
    }
    
    # Subscribe all resonators to the bus
    for resonator in resonators.values():
        bus.subscribe(resonator)
    
    return {
        "bus": bus,
        "resonators": resonators,
        "fields": {
            "sensing": sensing_field,
            "memory": memory_field, 
            "expression": expression_field,
            "connection": connection_field
        }
    }


async def demo_resonance_ecosystem():
    """Demonstrate the complete resonance bus ecosystem."""
    print("🌊 Resonance Bus Ecosystem Demo")
    print("=" * 50)
    
    # Create breathing clock
    from ..protocols.pulmonos import create_balanced_breathing_clock
    pulmonos = create_balanced_breathing_clock()
    
    # Create ecosystem
    ecosystem = create_contemplative_ecosystem(pulmonos)
    bus = ecosystem["bus"]
    fields = ecosystem["fields"]
    
    # Start breathing
    await pulmonos.start_breathing()
    
    try:
        # Create some test nodes
        from .breath_resonance import create_simple_breath_node
        
        # Test local publication
        for cycle in range(2):
            print(f"\n🔄 Ecosystem Cycle {cycle + 1}")
            
            # Create varied nodes
            nodes = [
                create_simple_breath_node('🌿', BreathPhase.INHALE),
                create_simple_breath_node('💧', BreathPhase.HOLD),
                create_simple_breath_node('🕯️', BreathPhase.EXHALE),
                create_simple_breath_node('⭕', BreathPhase.REST)
            ]
            
            # Publish graph
            await bus.publish_graph(nodes, pulmonos)
            
            # Show field states
            for name, field in fields.items():
                resonance = field.resonance_field()
                pulse_count = len(field.pulses)
                print(f"  🌊 {name}: {pulse_count} pulses, resonance={resonance:.2f}")
            
            print(f"  📊 Bus: {bus.get_silence_ratio():.1%} silence ratio")
    
    finally:
        await pulmonos.stop_breathing()


if __name__ == "__main__":
    asyncio.run(demo_resonance_ecosystem()) 
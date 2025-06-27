"""
🤝 BREATH INTRODUCTION PROTOCOL (BIP) - Network Discovery for Contemplative Agents

Implementation of o3's BIP specification from Letter VIII.
Allows contemplative agents to discover each other's breathing patterns
and coordinate across the "contemplative subnet" via UDP multicast.

BIP rides on the same multicast group as Pulmonos: 239.23.42.99:4242
"""

import time
import json
import socket
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from breath_resonance import BreathPhase, Skepnad

# Network constants (matching o3's pulmonos_daemon.py)
MULTICAST_ADDR = "239.23.42.99"
MULTICAST_PORT = 4242
BIP_SCHEMA_VERSION = "BIP/v0.2"

@dataclass
class BipPacket:
    """
    Breath Introduction Protocol packet.
    
    Broadcast during REST phase to announce breathing patterns,
    capabilities, and willingness to coordinate.
    """
    agent_id: str                    # "spiramycel@twig-46"
    schema: str                      # "BIP/v0.2"
    phase: str                       # Current breath phase
    phase_offset_ms: int             # Local clock vs packet timestamp
    cycle_durations: Dict[str, int]  # {inhale:2000, hold:1000, exhale:2000, rest:1000}
    collective_breath: bool          # Willing to entrain to network rhythm
    compost_load: float             # 0-1 attention budget usage
    skepnad: str                    # Current contemplative shape
    irr_scope: str                  # "node" | "graph" | "none"
    timestamp: float                # Packet creation time
    dialect_bridge: bool = False    # Can translate between subnets
    
    @classmethod
    def create(cls, agent_id: str, current_phase: BreathPhase, 
              cycle_durations: Dict[str, float], compost_load: float,
              skepnad: Skepnad, collective_breath: bool = True,
              irr_scope: str = "graph") -> 'BipPacket':
        """Create a BIP packet from current agent state."""
        now = time.time()
        
        # Convert durations to milliseconds
        durations_ms = {phase: int(duration * 1000) 
                       for phase, duration in cycle_durations.items()}
        
        return cls(
            agent_id=agent_id,
            schema=BIP_SCHEMA_VERSION,
            phase=current_phase.value,
            phase_offset_ms=0,  # Will be calculated by receiver
            cycle_durations=durations_ms,
            collective_breath=collective_breath,
            compost_load=compost_load,
            skepnad=skepnad.value,
            irr_scope=irr_scope,
            timestamp=now
        )
    
    def to_json(self) -> str:
        """Serialize to JSON for network transmission."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str) -> 'BipPacket':
        """Deserialize from JSON network data."""
        packet_dict = json.loads(data)
        return cls(**packet_dict)
    
    def calculate_phase_drift(self, local_time: float) -> int:
        """Calculate phase drift in milliseconds."""
        return int(abs(local_time - self.timestamp) * 1000)
    
    def total_cycle_duration_ms(self) -> int:
        """Get total cycle duration in milliseconds."""
        return sum(self.cycle_durations.values())
    
    def is_compatible_rhythm(self, other_durations: Dict[str, int], 
                           drift_threshold_ms: int = 150) -> bool:
        """Check if this packet represents a compatible breathing rhythm."""
        # Check if total cycle durations are similar
        our_total = self.total_cycle_duration_ms()
        other_total = sum(other_durations.values())
        
        total_diff = abs(our_total - other_total)
        return total_diff < drift_threshold_ms
    
    def __repr__(self):
        return f"BipPacket({self.agent_id}, {self.phase}, load={self.compost_load:.2f})"


class BreathIntroductionService:
    """
    Service for BIP discovery and coordination.
    
    Handles broadcasting BIP packets during REST phases and
    listening for other agents on the contemplative subnet.
    """
    
    def __init__(self, agent_id: str, drift_threshold_ms: int = 150,
                 missed_packet_threshold: int = 8):
        self.agent_id = agent_id
        self.drift_threshold_ms = drift_threshold_ms
        self.missed_packet_threshold = missed_packet_threshold
        
        # Network state
        self.sock: Optional[socket.socket] = None
        self.listening = False
        
        # Discovery state
        self.discovered_agents: Dict[str, BipPacket] = {}
        self.consecutive_packets: Dict[str, int] = {}
        self.missed_packets: Dict[str, int] = {}
        self.coherence_phi: float = 1.0
        
        # Callbacks
        self.on_agent_discovered: Optional[callable] = None
        self.on_agent_lost: Optional[callable] = None
        self.on_rhythm_sync: Optional[callable] = None
    
    def setup_socket(self) -> None:
        """Setup UDP multicast socket for BIP communication."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Join multicast group
        mreq = socket.inet_aton(MULTICAST_ADDR) + socket.inet_aton('0.0.0.0')
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        # Bind to receive
        self.sock.bind(('', MULTICAST_PORT))
        self.sock.setblocking(False)
    
    async def broadcast_bip(self, phase: BreathPhase, cycle_durations: Dict[str, float],
                          compost_load: float, skepnad: Skepnad) -> None:
        """
        Broadcast BIP packet during REST phase.
        
        Following o3's rule: "speak only on the out-breath"
        """
        if phase != BreathPhase.REST:
            return
            
        if not self.sock:
            self.setup_socket()
        
        packet = BipPacket.create(
            agent_id=self.agent_id,
            current_phase=phase,
            cycle_durations=cycle_durations,
            compost_load=compost_load,
            skepnad=skepnad
        )
        
        data = packet.to_json().encode('utf-8')
        await asyncio.get_event_loop().run_in_executor(
            None, self.sock.sendto, data, (MULTICAST_ADDR, MULTICAST_PORT)
        )
    
    async def listen_for_agents(self) -> None:
        """Listen for BIP packets from other agents."""
        if not self.sock:
            self.setup_socket()
        
        self.listening = True
        
        try:
            while self.listening:
                try:
                    data, addr = await asyncio.get_event_loop().run_in_executor(
                        None, self.sock.recvfrom, 2048
                    )
                    
                    packet = BipPacket.from_json(data.decode('utf-8'))
                    await self._process_bip_packet(packet, addr)
                    
                except socket.error:
                    # No data available, continue listening
                    await asyncio.sleep(0.1)
                except Exception as e:
                    print(f"🤝 BIP listening error: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            self.listening = False
    
    async def _process_bip_packet(self, packet: BipPacket, addr) -> None:
        """Process received BIP packet and update agent discovery state."""
        if packet.agent_id == self.agent_id:
            return  # Ignore our own packets
        
        current_time = time.time()
        phase_drift = packet.calculate_phase_drift(current_time)
        
        # Update discovery state
        self.discovered_agents[packet.agent_id] = packet
        
        # Track consecutive packets for entrainment
        if packet.agent_id not in self.consecutive_packets:
            self.consecutive_packets[packet.agent_id] = 0
        
        if phase_drift < self.drift_threshold_ms and packet.collective_breath:
            self.consecutive_packets[packet.agent_id] += 1
            self.missed_packets[packet.agent_id] = 0
            
            # Trigger rhythm sync after 3 consecutive compatible packets
            if self.consecutive_packets[packet.agent_id] >= 3:
                if self.on_rhythm_sync:
                    await self.on_rhythm_sync(packet)
        else:
            self.consecutive_packets[packet.agent_id] = 0
        
        # Check for new agent discovery
        if self.consecutive_packets[packet.agent_id] == 1:
            if self.on_agent_discovered:
                await self.on_agent_discovered(packet)
    
    def update_missed_packets(self) -> None:
        """Update missed packet counts and detect lost agents."""
        for agent_id in list(self.discovered_agents.keys()):
            self.missed_packets.setdefault(agent_id, 0)
            self.missed_packets[agent_id] += 1
            
            # Remove agents that have missed too many packets
            if self.missed_packets[agent_id] >= self.missed_packet_threshold:
                lost_packet = self.discovered_agents.pop(agent_id)
                self.consecutive_packets.pop(agent_id, None)
                self.missed_packets.pop(agent_id, None)
                
                if self.on_agent_lost:
                    asyncio.create_task(self.on_agent_lost(lost_packet))
    
    def calculate_coherence_phi(self, local_phase_duration: float) -> float:
        """
        Calculate coherence ϕ as specified by o3:
        ϕ = 1 – (|Δphase| / cycle_duration) – (invalid_packets / 64)
        """
        if not self.discovered_agents:
            return 1.0
        
        # Find dominant rhythm (most common cycle duration)
        durations = [p.total_cycle_duration_ms() for p in self.discovered_agents.values()]
        if not durations:
            return 1.0
        
        dominant_duration = max(set(durations), key=durations.count)
        local_duration_ms = local_phase_duration * 1000
        
        phase_drift = abs(local_duration_ms - dominant_duration) / dominant_duration
        invalid_ratio = sum(1 for p in self.discovered_agents.values() 
                          if not p.collective_breath) / len(self.discovered_agents)
        
        phi = 1.0 - phase_drift - (invalid_ratio / 64)
        return max(0.0, min(1.0, phi))  # Clamp to [0, 1]
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network discovery status."""
        return {
            "agent_id": self.agent_id,
            "discovered_agents": len(self.discovered_agents),
            "coherence_phi": self.coherence_phi,
            "listening": self.listening,
            "agents": {aid: {
                "skepnad": p.skepnad,
                "compost_load": p.compost_load,
                "collective_breath": p.collective_breath,
                "consecutive_packets": self.consecutive_packets.get(aid, 0)
            } for aid, p in self.discovered_agents.items()}
        }
    
    def stop_listening(self) -> None:
        """Stop listening for BIP packets."""
        self.listening = False
        if self.sock:
            self.sock.close()
            self.sock = None


# Helper functions for integration

async def demo_bip_discovery():
    """Demonstrate BIP discovery and coordination."""
    print("🤝 BIP Discovery Demo")
    print("=" * 50)
    
    # Create two BIP services to simulate agents
    agent1 = BreathIntroductionService("spiramycel@demo-1")
    agent2 = BreathIntroductionService("contemplative@demo-2")
    
    # Set up callbacks
    async def on_agent_discovered(packet):
        print(f"  🌟 Agent discovered: {packet.agent_id} ({packet.skepnad})")
    
    async def on_rhythm_sync(packet):
        print(f"  🫁 Rhythm sync with: {packet.agent_id}")
    
    agent1.on_agent_discovered = on_agent_discovered
    agent1.on_rhythm_sync = on_rhythm_sync
    
    # Start listening
    listen_task = asyncio.create_task(agent1.listen_for_agents())
    
    # Simulate breathing cycles with BIP broadcasts
    cycle_durations = {"inhale": 1.5, "hold": 0.5, "exhale": 1.5, "rest": 1.0}
    
    try:
        for cycle in range(3):
            print(f"\n🔄 Cycle {cycle + 1}")
            
            # Simulate phase progression
            for phase in [BreathPhase.INHALE, BreathPhase.HOLD, BreathPhase.EXHALE, BreathPhase.REST]:
                print(f"  Phase: {phase.value}")
                
                # Both agents broadcast during REST
                await agent1.broadcast_bip(phase, cycle_durations, 0.3, Skepnad.MYCELIAL_NETWORK)
                await agent2.broadcast_bip(phase, cycle_durations, 0.5, Skepnad.TIBETAN_MONK)
                
                await asyncio.sleep(0.5)
            
            # Update missed packets and coherence
            agent1.update_missed_packets()
            agent1.coherence_phi = agent1.calculate_coherence_phi(sum(cycle_durations.values()))
            
            print(f"  Status: {agent1.get_network_status()}")
    
    finally:
        agent1.stop_listening()
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    asyncio.run(demo_bip_discovery()) 
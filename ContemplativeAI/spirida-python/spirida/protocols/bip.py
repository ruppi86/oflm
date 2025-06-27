"""
🤝 BREATH INTRODUCTION PROTOCOL (BIP) - Network Discovery for Contemplative Agents

Implementation of o3's BIP specification from Letter VIII.
Allows contemplative agents to discover each other's breathing patterns
and coordinate across the "contemplative subnet" via UDP multicast.
"""

import time
import json
import socket
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from breath_resonance import BreathPhase, Skepnad

# Network constants
MULTICAST_ADDR = "239.23.42.99"
MULTICAST_PORT = 4242
BIP_SCHEMA_VERSION = "BIP/v0.2"

@dataclass
class BipPacket:
    """Breath Introduction Protocol packet."""
    agent_id: str
    schema: str
    phase: str
    phase_offset_ms: int
    cycle_durations: Dict[str, int]
    collective_breath: bool
    compost_load: float
    skepnad: str
    irr_scope: str
    timestamp: float
    
    @classmethod
    def create(cls, agent_id: str, current_phase: BreathPhase, 
              cycle_durations: Dict[str, float], compost_load: float,
              skepnad: Skepnad) -> 'BipPacket':
        """Create a BIP packet from current agent state."""
        durations_ms = {phase: int(duration * 1000) 
                       for phase, duration in cycle_durations.items()}
        
        return cls(
            agent_id=agent_id,
            schema=BIP_SCHEMA_VERSION,
            phase=current_phase.value,
            phase_offset_ms=0,
            cycle_durations=durations_ms,
            collective_breath=True,
            compost_load=compost_load,
            skepnad=skepnad.value,
            irr_scope="graph",
            timestamp=time.time()
        )
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, data: str) -> 'BipPacket':
        return cls(**json.loads(data))

class BreathIntroductionService:
    """Service for BIP discovery and coordination."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.sock: Optional[socket.socket] = None
        self.discovered_agents: Dict[str, BipPacket] = {}
    
    def setup_socket(self) -> None:
        """Setup UDP multicast socket."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    async def broadcast_bip(self, phase: BreathPhase, cycle_durations: Dict[str, float],
                          compost_load: float, skepnad: Skepnad) -> None:
        """Broadcast BIP packet during REST phase."""
        if phase != BreathPhase.REST:
            return
            
        if not self.sock:
            self.setup_socket()
        
        packet = BipPacket.create(self.agent_id, phase, cycle_durations, compost_load, skepnad)
        data = packet.to_json().encode('utf-8')
        
        if self.sock:
            self.sock.sendto(data, (MULTICAST_ADDR, MULTICAST_PORT))

if __name__ == "__main__":
    print("🤝 BIP Module - Breath Introduction Protocol") 
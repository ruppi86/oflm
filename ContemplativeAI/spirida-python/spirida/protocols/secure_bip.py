#!/usr/bin/env python3
"""
🔐 SECURE BREATH INTRODUCTION PROTOCOL (BIP) - Network Security for Contemplative Agents

Integration of o3's slow-start middleware (Letter XXI) with the existing BIP system.
Provides contemplative authentication through synchronized breathing before symbol exchange.

Key Security Features:
- Slow-start handshake: 5 synchronized breath cycles before symbol exchange
- Breath signature verification: Authentic timing patterns required
- Graceful degradation: Falls back to local mode on security concerns
"""

import asyncio
import socket
import json
import time
from typing import Dict, Any, Optional, Tuple, Callable
from collections import defaultdict

# Import our existing components
from .bip import BipPacket, BreathIntroductionService, MULTICAST_ADDR, MULTICAST_PORT
from ..compiler.breath_resonance import BreathPhase, Skepnad
from ...security.slow_start_middleware import slow_start
from ...security.breath_signature import BreathSignature


class SecureBreathIntroductionService(BreathIntroductionService):
    """
    Enhanced BIP service with contemplative security measures.
    
    Implements o3's slow-start handshake and breath signature verification
    to ensure only authentic contemplative agents can participate in network breathing.
    """
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        
        # Security components
        self.breath_signature = BreathSignature()
        self.peer_trust_levels: Dict[str, int] = {}  # agent_id -> trust_level
        self.authenticated_peers: Dict[str, bool] = {}
        
        # UDP listening components
        self.listen_socket: Optional[socket.socket] = None
        self.listen_task: Optional[asyncio.Task] = None
        self.is_listening = False
        
        # Callback for authenticated packets
        self.authenticated_packet_handler: Optional[Callable] = None
        self.security_event_handler: Optional[Callable] = None
        
        # Statistics
        self.packets_received = 0
        self.packets_authenticated = 0
        self.packets_rejected = 0
        
    async def start_secure_listening(self) -> None:
        """Start listening for BIP packets with security validation."""
        if self.is_listening:
            return
        
        try:
            # Setup UDP multicast socket for receiving
            self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to multicast address
            self.listen_socket.bind(('', MULTICAST_PORT))
            
            # Join multicast group
            mreq = socket.inet_aton(MULTICAST_ADDR) + socket.inet_aton('0.0.0.0')
            self.listen_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            # Make socket non-blocking
            self.listen_socket.setblocking(False)
            
            self.is_listening = True
            
            # Start listening task with slow-start security
            self.listen_task = asyncio.create_task(self._secure_listening_loop())
            
            print(f"🔐 {self.agent_id} started secure BIP listening on {MULTICAST_ADDR}:{MULTICAST_PORT}")
            
        except Exception as e:
            print(f"🔐 Failed to start secure BIP listening: {e}")
            await self.stop_listening()
    
    async def stop_listening(self) -> None:
        """Stop listening for BIP packets."""
        self.is_listening = False
        
        if self.listen_task:
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                pass
            self.listen_task = None
        
        if self.listen_socket:
            try:
                self.listen_socket.close()
            except:
                pass
            self.listen_socket = None
    
    async def _secure_listening_loop(self) -> None:
        """Main listening loop with integrated slow-start security."""
        
        @slow_start
        async def handle_authenticated_packet(packet_data: Dict, addr: Tuple[str, int]):
            """Handle packets that have passed slow-start authentication."""
            try:
                bip_packet = BipPacket(**packet_data)
                
                # Additional security checks
                if await self._validate_breath_signature(bip_packet):
                    # Mark peer as authenticated
                    self.authenticated_peers[bip_packet.agent_id] = True
                    self.packets_authenticated += 1
                    
                    # Update discovered agents
                    self.discovered_agents[bip_packet.agent_id] = bip_packet
                    
                    # Call authenticated packet handler if set
                    if self.authenticated_packet_handler:
                        await self.authenticated_packet_handler(bip_packet, addr)
                    
                    # Log security event
                    if self.security_event_handler:
                        await self.security_event_handler("packet_authenticated", {
                            "agent_id": bip_packet.agent_id,
                            "addr": addr,
                            "trust_level": self.peer_trust_levels.get(bip_packet.agent_id, 0)
                        })
                else:
                    self._handle_security_violation("invalid_breath_signature", bip_packet.agent_id, addr)
                    
            except Exception as e:
                print(f"🔐 Error processing authenticated packet: {e}")
        
        # Main listening loop
        try:
            while self.is_listening:
                try:
                    if self.listen_socket:
                        # Receive UDP packet
                        data, addr = self.listen_socket.recvfrom(4096)
                        self.packets_received += 1
                        
                        # Parse packet
                        try:
                            packet_data = json.loads(data.decode('utf-8'))
                            
                            # Apply slow-start middleware (this will filter out non-authenticated agents)
                            await handle_authenticated_packet(packet_data, addr)
                            
                        except json.JSONDecodeError:
                            self._handle_security_violation("invalid_json", "unknown", addr)
                        except Exception as e:
                            print(f"🔐 Packet processing error: {e}")
                    
                except socket.error as e:
                    if e.errno != 11:  # EAGAIN (no data available)
                        print(f"🔐 Socket error: {e}")
                    await asyncio.sleep(0.1)  # Prevent busy waiting
                    
                except Exception as e:
                    print(f"🔐 Listening loop error: {e}")
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"🔐 Secure listening loop failed: {e}")
    
    async def _validate_breath_signature(self, packet: BipPacket) -> bool:
        """
        Validate the breath signature of incoming packets.
        
        This is where o3's breath signature verification would happen.
        For now, we do basic validation and could enhance with timing analysis.
        """
        # Basic validation
        if not packet.agent_id or packet.agent_id == self.agent_id:
            return False  # Don't trust self-packets or empty IDs
        
        # Check timing consistency (simplified)
        current_time = time.time()
        if abs(current_time - packet.timestamp) > 30:  # 30 second tolerance
            return False  # Packet too old or from future
        
        # Check for authentic breath timing
        if packet.phase not in ["INHALE", "HOLD", "EXHALE", "REST"]:
            return False
        
        # Validate cycle durations are reasonable
        total_cycle = sum(packet.cycle_durations.values())
        if total_cycle < 2000 or total_cycle > 30000:  # 2-30 seconds total cycle
            return False
        
        # Update breath signature with this agent's REST phases
        if packet.phase == "REST":
            # Here we would integrate full breath signature validation
            # For now, we accept REST phases as signature updates
            pass
        
        return True
    
    def _handle_security_violation(self, violation_type: str, agent_id: str, addr: Tuple[str, int]):
        """Handle security violations."""
        self.packets_rejected += 1
        
        print(f"🔐 Security violation: {violation_type} from {agent_id} at {addr}")
        
        # Remove from authenticated peers if present
        if agent_id in self.authenticated_peers:
            del self.authenticated_peers[agent_id]
        
        # Log security event
        if self.security_event_handler:
            asyncio.create_task(self.security_event_handler("security_violation", {
                "violation_type": violation_type,
                "agent_id": agent_id,
                "addr": addr
            }))
    
    async def broadcast_secure_bip(self, phase: BreathPhase, cycle_durations: Dict[str, float],
                                 compost_load: float, skepnad: Skepnad) -> None:
        """Enhanced BIP broadcast with signature updates."""
        # Update our breath signature for this REST phase
        if phase == BreathPhase.REST:
            self.breath_signature.update_rest()
        
        # Use parent class broadcast method
        await super().broadcast_bip(phase, cycle_durations, compost_load, skepnad)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        authenticated_count = len(self.authenticated_peers)
        total_known = len(self.discovered_agents)
        
        return {
            "listening": self.is_listening,
            "packets_received": self.packets_received,
            "packets_authenticated": self.packets_authenticated,
            "packets_rejected": self.packets_rejected,
            "authentication_rate": (self.packets_authenticated / max(self.packets_received, 1)) * 100,
            "authenticated_peers": authenticated_count,
            "total_discovered_agents": total_known,
            "peer_trust_levels": self.peer_trust_levels.copy(),
            "breath_signature": self.breath_signature.current_signature()[:16] + "..."  # Truncated for display
        }
    
    def set_authenticated_packet_handler(self, handler: Callable):
        """Set callback for handling authenticated packets."""
        self.authenticated_packet_handler = handler
    
    def set_security_event_handler(self, handler: Callable):
        """Set callback for handling security events."""
        self.security_event_handler = handler
    
    def is_peer_authenticated(self, agent_id: str) -> bool:
        """Check if a peer has been authenticated."""
        return self.authenticated_peers.get(agent_id, False)
    
    def get_peer_trust_level(self, agent_id: str) -> int:
        """Get trust level for a specific peer."""
        return self.peer_trust_levels.get(agent_id, 0)


# Integration function for existing NetworkPulmonos
def create_secure_network_breathing_service(agent_id: str) -> SecureBreathIntroductionService:
    """Create a secure BIP service for integration with NetworkPulmonos."""
    return SecureBreathIntroductionService(agent_id)


async def demo_secure_bip():
    """Demonstrate the secure BIP service."""
    print("🔐 Secure BIP Service Demo")
    print("=" * 50)
    
    # Create secure service
    service = SecureBreathIntroductionService("demo_agent")
    
    # Set up event handlers
    async def on_authenticated_packet(packet, addr):
        print(f"✅ Authenticated packet from {packet.agent_id} at {addr}")
    
    async def on_security_event(event_type, details):
        print(f"🔐 Security event: {event_type} - {details}")
    
    service.set_authenticated_packet_handler(on_authenticated_packet)
    service.set_security_event_handler(on_security_event)
    
    # Start listening
    await service.start_secure_listening()
    
    # Simulate some network activity
    print("Listening for 10 seconds...")
    await asyncio.sleep(10)
    
    # Show status
    status = service.get_security_status()
    print(f"\n📊 Security Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Stop service
    await service.stop_listening()
    print("\n🔐 Secure BIP demo completed")


if __name__ == "__main__":
    asyncio.run(demo_secure_bip()) 
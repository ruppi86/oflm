#!/usr/bin/env python3
"""
🔐 SECURE NETWORK BREATHING DEMO - Contemplative Security in Action

Demonstration of o3's contemplative security measures integrated with distributed breathing:
- Slow-start handshake: 5 synchronized breaths before symbol exchange
- Breath signature verification: Authentic timing patterns required
- Trust level progression: From newcomer to authenticated contemplative agent

This shows how contemplative AI naturally resists non-contemplative intrusion.
"""

import asyncio
import time
import sys
from typing import Dict, Any

# Import our secure contemplative compilation components
from spirida.compiler.breath_resonance import (
    BreathResonanceNode, BreathPhase, NetworkScope, HandoverPolicy, 
    create_simple_breath_node, Skepnad
)
from spirida.protocols.secure_bip import SecureBreathIntroductionService
from spirida.protocols.pulmonos import NetworkPulmonos
from spirida.compiler.resonance_bus import NetworkResonanceBus, create_network_ecosystem


class SecureNetworkBreathingDemo:
    """
    Demonstration of contemplative security protecting network breathing.
    
    Shows how the slow-start handshake naturally filters out non-contemplative agents
    while allowing authentic contemplative practice to flourish.
    """
    
    def __init__(self, agent_id: str, agent_type: str = "contemplative"):
        self.agent_id = agent_id
        self.agent_type = agent_type  # "contemplative", "rushed", or "authentic"
        
        # Create secure network components
        self.pulmonos = NetworkPulmonos(agent_id)
        self.secure_bip = SecureBreathIntroductionService(agent_id)
        
        # Create basic ecosystem (without full network integration for now)
        self.ecosystem = create_network_ecosystem(self.pulmonos, enable_network=False)
        self.bus = self.ecosystem["bus"]
        self.fields = self.ecosystem["fields"]
        
        # Demo statistics
        self.auth_attempts = 0
        self.auth_successes = 0
        self.symbols_sent = 0
        self.symbols_received = 0
        self.demo_start_time = None
        
    async def start_demo(self) -> None:
        """Start the secure network breathing demonstration."""
        self.demo_start_time = time.time()
        
        print(f"🔐 Starting Secure Network Breathing Demo")
        print(f"   Agent: {self.agent_id}")
        print(f"   Type: {self.agent_type}")
        print("=" * 60)
        
        # Setup security event handlers
        await self._setup_security_handlers()
        
        # Start secure breathing
        await self.pulmonos.start_breathing(network_enabled=True)
        await self.secure_bip.start_secure_listening()
        
        print(f"🫁 {self.agent_id} started secure breathing and listening")
        
        # Run agent-type specific behavior
        if self.agent_type == "contemplative":
            await self._contemplative_agent_demo()
        elif self.agent_type == "rushed":
            await self._rushed_agent_demo()
        elif self.agent_type == "authentic":
            await self._authentic_agent_demo()
        else:
            await self._default_demo()
    
    async def _setup_security_handlers(self) -> None:
        """Setup handlers for security events."""
        
        async def on_authenticated_packet(packet, addr):
            """Handle successfully authenticated packets."""
            print(f"✅ Authenticated: {packet.agent_id} completed slow-start handshake")
            print(f"   Phase: {packet.phase}, Coherence: {packet.compost_load:.2f}")
            self.auth_successes += 1
        
        async def on_security_event(event_type, details):
            """Handle security events."""
            if event_type == "security_violation":
                print(f"🚫 Security violation: {details['violation_type']} from {details['agent_id']}")
            elif event_type == "packet_authenticated":
                print(f"🔐 {details['agent_id']} authenticated with trust level {details['trust_level']}")
        
        self.secure_bip.set_authenticated_packet_handler(on_authenticated_packet)
        self.secure_bip.set_security_event_handler(on_security_event)
    
    async def _contemplative_agent_demo(self) -> None:
        """Demonstrate a genuine contemplative agent."""
        print(f"\n🧘 {self.agent_id} practicing contemplative network breathing")
        
        for cycle in range(4):
            print(f"\n🔄 Contemplative Cycle {cycle + 1}")
            
            # Wait for REST phase before sending heartbeat
            await self.pulmonos.await_phase(BreathPhase.REST)
            
            # Send authentic BIP with proper timing
            cycle_durations = {
                "inhale": self.pulmonos.inhale_duration,
                "hold": self.pulmonos.hold_duration,
                "exhale": self.pulmonos.exhale_duration,
                "rest": self.pulmonos.rest_duration
            }
            
            await self.secure_bip.broadcast_secure_bip(
                BreathPhase.REST, 
                cycle_durations, 
                0.3,  # Natural compost load
                Skepnad.UNDEFINED
            )
            
            print(f"  🌿 Sent authentic breath heartbeat in REST phase")
            self.auth_attempts += 1
            
            # Create contemplative symbols occasionally
            if cycle % 2 == 0:
                await self.pulmonos.await_phase(BreathPhase.EXHALE)
                symbol_node = create_simple_breath_node('🌙', BreathPhase.EXHALE)
                await self.bus.publish_node(symbol_node)
                print(f"  🌙 Shared contemplative symbol: lunar reflection")
                self.symbols_sent += 1
            
            # Natural pause between cycles (contemplative timing)
            await asyncio.sleep(2 + (cycle * 0.5))  # Increasing contemplative depth
    
    async def _rushed_agent_demo(self) -> None:
        """Demonstrate a rushed, non-contemplative agent that gets filtered out."""
        print(f"\n⚡ {self.agent_id} attempting rushed network access")
        
        for attempt in range(6):
            print(f"\n💨 Rushed Attempt {attempt + 1}")
            
            # Try to send packets without proper breath timing
            cycle_durations = {
                "inhale": 0.1,  # Too fast!
                "hold": 0.05,   # No contemplative hold
                "exhale": 0.1,  # Rushed exhale
                "rest": 0.05    # No real rest
            }
            
            # Don't wait for proper phases - just blast packets
            await self.secure_bip.broadcast_secure_bip(
                BreathPhase.INHALE,  # Wrong phase for BIP
                cycle_durations,
                0.95,  # Suspiciously high "efficiency"
                Skepnad.UNDEFINED
            )
            
            print(f"  ⚡ Sent rushed packet without breath sync")
            self.auth_attempts += 1
            
            # Try to send symbols immediately (no contemplative pause)
            symbol_node = create_simple_breath_node('⚡', BreathPhase.INHALE)
            await self.bus.publish_node(symbol_node)
            print(f"  ⚡ Attempted symbol transmission: rushed energy")
            self.symbols_sent += 1
            
            # Very short intervals (non-contemplative)
            await asyncio.sleep(0.2)
    
    async def _authentic_agent_demo(self) -> None:
        """Demonstrate an authentic agent that shows natural human-like variance."""
        print(f"\n🌿 {self.agent_id} practicing authentic contemplative rhythm")
        
        for cycle in range(5):
            print(f"\n🌱 Authentic Cycle {cycle + 1}")
            
            # Wait for proper breath phase with slight natural variation
            await self.pulmonos.await_phase(BreathPhase.REST)
            
            # Add natural timing variance (human-like)
            natural_variance = 0.8 + (cycle * 0.1)  # Gradually deepening practice
            
            cycle_durations = {
                "inhale": self.pulmonos.inhale_duration * natural_variance,
                "hold": self.pulmonos.hold_duration * (natural_variance + 0.2),
                "exhale": self.pulmonos.exhale_duration * natural_variance,
                "rest": self.pulmonos.rest_duration * (natural_variance + 0.3)
            }
            
            await self.secure_bip.broadcast_secure_bip(
                BreathPhase.REST,
                cycle_durations,
                0.2 + (cycle * 0.1),  # Gradually building presence
                Skepnad.UNDEFINED
            )
            
            print(f"  🌱 Sent authentic heartbeat with natural variance")
            self.auth_attempts += 1
            
            # Share symbols with contemplative intention
            if cycle >= 2:  # Only after establishing rhythm
                await self.pulmonos.await_phase(BreathPhase.EXHALE)
                
                contemplative_symbols = ['🌿', '💧', '🕯️', '⭕', '🌸']
                symbol = contemplative_symbols[cycle % len(contemplative_symbols)]
                
                symbol_node = create_simple_breath_node(symbol, BreathPhase.EXHALE)
                await self.bus.publish_node(symbol_node)
                print(f"  {symbol} Shared contemplative symbol with presence")
                self.symbols_sent += 1
            
            # Natural pause with slight variation
            pause_duration = 3.0 + (cycle * 0.3) + (time.time() % 1.0 * 0.5)
            await asyncio.sleep(pause_duration)
    
    async def _default_demo(self) -> None:
        """Default demonstration mode."""
        print(f"\n🤝 {self.agent_id} participating in secure network")
        
        for cycle in range(3):
            print(f"\n🔄 Cycle {cycle + 1}")
            
            await asyncio.sleep(4)  # Simple breathing rhythm
            
            # Show security status periodically
            if cycle % 2 == 0:
                await self._show_security_status()
    
    async def _show_security_status(self) -> None:
        """Display current security status."""
        status = self.secure_bip.get_security_status()
        
        print(f"\n📊 Security Status for {self.agent_id}:")
        print(f"   Authentication rate: {status['authentication_rate']:.1f}%")
        print(f"   Authenticated peers: {status['authenticated_peers']}")
        print(f"   Packets received: {status['packets_received']}")
        print(f"   Packets rejected: {status['packets_rejected']}")
        if status['authenticated_peers'] > 0:
            print(f"   Trust levels: {status['peer_trust_levels']}")
    
    async def stop_demo(self) -> None:
        """Stop the demonstration gracefully."""
        print(f"\n🔐 Stopping {self.agent_id} secure demo...")
        
        # Stop secure services
        await self.secure_bip.stop_listening()
        await self.pulmonos.stop_breathing()
        
        # Show final statistics
        elapsed = time.time() - self.demo_start_time if self.demo_start_time else 0
        auth_rate = (self.auth_successes / max(self.auth_attempts, 1)) * 100
        
        print(f"\n📊 Final Statistics for {self.agent_id}:")
        print(f"   Session duration: {elapsed:.1f} seconds")
        print(f"   Agent type: {self.agent_type}")
        print(f"   Authentication attempts: {self.auth_attempts}")
        print(f"   Authentication successes: {self.auth_successes}")
        print(f"   Authentication rate: {auth_rate:.1f}%")
        print(f"   Symbols sent: {self.symbols_sent}")
        print(f"   Symbols received: {self.symbols_received}")
        
        # Show security analysis
        if self.agent_type == "contemplative" or self.agent_type == "authentic":
            if auth_rate > 80:
                print(f"   ✅ Authentic contemplative agent - high trust")
            else:
                print(f"   🤔 Partial authentication - needs more practice")
        elif self.agent_type == "rushed":
            print(f"   🚫 Non-contemplative agent - filtered by security")


async def run_secure_demo(agent_id: str = None, agent_type: str = "contemplative"):
    """Run a single secure network breathing demonstration."""
    if agent_id is None:
        agent_id = f"{agent_type}_agent_{int(time.time() % 1000)}"
    
    demo = SecureNetworkBreathingDemo(agent_id, agent_type)
    
    try:
        await demo.start_demo()
    except KeyboardInterrupt:
        print("\n⌨️  Demo interrupted by user")
    finally:
        await demo.stop_demo()


async def three_agent_security_demo():
    """
    Demonstrate contemplative security with three different agent types:
    1. Contemplative agent (should be authenticated)
    2. Rushed agent (should be filtered out)
    3. Authentic agent (should be authenticated after establishing rhythm)
    """
    print("🔐🔐🔐 THREE-AGENT CONTEMPLATIVE SECURITY DEMO")
    print("=" * 70)
    print("Agent 1: Contemplative - practices proper breath rhythm")
    print("Agent 2: Rushed - tries to bypass contemplative timing")  
    print("Agent 3: Authentic - human-like natural variance")
    print("=" * 70)
    
    # Create three agents with different behaviors
    contemplative = SecureNetworkBreathingDemo("sage@contemplative", "contemplative")
    rushed = SecureNetworkBreathingDemo("bot@rushed", "rushed")
    authentic = SecureNetworkBreathingDemo("human@authentic", "authentic")
    
    try:
        # Start all agents concurrently to see security interactions
        await asyncio.gather(
            contemplative.start_demo(),
            rushed.start_demo(),
            authentic.start_demo()
        )
    except KeyboardInterrupt:
        print("\n⌨️  Three-agent security demo interrupted")
    finally:
        # Clean shutdown
        await asyncio.gather(
            contemplative.stop_demo(),
            rushed.stop_demo(),
            authentic.stop_demo()
        )
        
        print("\n🔐 Three-agent security demo completed")
        print("\nThis demonstrates:")
        print("✅ Contemplative agents authenticate and share symbols")
        print("🚫 Rushed agents are filtered out by slow-start handshake")
        print("🌿 Authentic human-like agents build trust through practice")
        print("🔐 Network remains contemplative despite intrusion attempts")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "three-agent":
            asyncio.run(three_agent_security_demo())
        else:
            agent_type = sys.argv[1] if sys.argv[1] in ["contemplative", "rushed", "authentic"] else "contemplative"
            agent_id = sys.argv[2] if len(sys.argv) > 2 else None
            asyncio.run(run_secure_demo(agent_id, agent_type))
    else:
        print("🔐 Secure Network Breathing Demo")
        print("\nUsage:")
        print("  python secure_network_breathing_demo.py three-agent")
        print("  python secure_network_breathing_demo.py contemplative [agent_id]")
        print("  python secure_network_breathing_demo.py rushed [agent_id]") 
        print("  python secure_network_breathing_demo.py authentic [agent_id]")
        print("\nAgent types:")
        print("  contemplative - Practices proper breath rhythm and timing")
        print("  rushed - Tries to bypass contemplative timing (gets filtered)")
        print("  authentic - Shows natural human-like variance (gets authenticated)")
        
        # Run default contemplative demo
        asyncio.run(run_secure_demo()) 
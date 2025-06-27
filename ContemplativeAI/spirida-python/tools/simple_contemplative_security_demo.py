#!/usr/bin/env python3
"""
🔐 SIMPLE CONTEMPLATIVE SECURITY DEMO

A focused demonstration of o3's slow-start middleware concept.
Shows how contemplative authentication works through synchronized breathing
before allowing symbol exchange.

This is the core of contemplative cybersecurity - patience as a firewall.
"""

import asyncio
import time
import random
from typing import Dict, Any

# Import o3's security components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from security.slow_start_middleware import slow_start, BREATHS_REQUIRED
from security.breath_signature import BreathSignature


class MockContemplativeAgent:
    """
    Mock agent that simulates different types of network behavior
    for testing contemplative security measures.
    """
    
    def __init__(self, agent_id: str, agent_type: str = "contemplative"):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.breath_signature = BreathSignature()
        
        # Statistics
        self.rest_packets_sent = 0
        self.symbol_packets_sent = 0
        self.packets_blocked = 0
        self.authenticated = False
        
    async def send_rest_packet(self) -> Dict[str, Any]:
        """Send a REST phase packet for authentication."""
        # Update breath signature
        current_time = time.time()
        self.breath_signature.update_rest(current_time)
        
        # Create packet based on agent type
        if self.agent_type == "contemplative":
            # Authentic contemplative timing
            await asyncio.sleep(1.5 + random.uniform(0.3, 0.8))  # Natural variance
        elif self.agent_type == "rushed":
            # Too fast - non-contemplative
            await asyncio.sleep(0.1)
        elif self.agent_type == "human":
            # Human-like natural variance
            await asyncio.sleep(1.0 + random.uniform(0.5, 1.5))
        
        packet = {
            "agent_id": self.agent_id,
            "phase": "REST",
            "timestamp": current_time,
            "breath_signature": self.breath_signature.current_signature()[:8] + "..."
        }
        
        self.rest_packets_sent += 1
        return packet
    
    async def send_symbol_packet(self, symbol: str) -> Dict[str, Any]:
        """Send a symbol packet (should be blocked until authenticated)."""
        packet = {
            "agent_id": self.agent_id,
            "phase": "EXHALE",
            "symbol": symbol,
            "timestamp": time.time()
        }
        
        self.symbol_packets_sent += 1
        return packet
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "rest_packets_sent": self.rest_packets_sent,
            "symbol_packets_sent": self.symbol_packets_sent,
            "packets_blocked": self.packets_blocked,
            "authenticated": self.authenticated
        }


class ContemplativeSecurityDemo:
    """
    Demonstration of contemplative security protecting a network from
    non-contemplative agents while allowing authentic agents through.
    """
    
    def __init__(self):
        self.agents = {}
        self.packets_received = 0
        self.packets_authenticated = 0
        self.packets_blocked = 0
        
        # Create the slow-start protected packet handler
        @slow_start
        async def handle_authenticated_packet(packet: Dict, addr: str):
            """Handle packets that passed slow-start authentication."""
            self.packets_authenticated += 1
            agent_id = packet.get("agent_id", "unknown")
            
            if agent_id in self.agents:
                self.agents[agent_id].authenticated = True
            
            if packet.get("phase") == "REST":
                print(f"✅ {agent_id} authenticated after {BREATHS_REQUIRED} breaths")
            else:
                symbol = packet.get("symbol", "")
                print(f"🌀 {agent_id} symbol: {symbol} (authenticated)")
        
        self.protected_handler = handle_authenticated_packet
    
    async def process_packet(self, packet: Dict[str, Any], sender_addr: str = "mock"):
        """Process incoming packet through contemplative security."""
        self.packets_received += 1
        agent_id = packet.get("agent_id", "unknown")
        
        # Count blocked packets
        phase = packet.get("phase")
        if phase != "REST" and agent_id in self.agents and not self.agents[agent_id].authenticated:
            self.packets_blocked += 1
            print(f"🚫 {agent_id} symbol blocked - not authenticated yet")
            if agent_id in self.agents:
                self.agents[agent_id].packets_blocked += 1
            return
        
        # Process through slow-start middleware
        try:
            await self.protected_handler(packet, sender_addr)
        except Exception as e:
            print(f"🔐 Security filter blocked packet from {agent_id}: {e}")
            self.packets_blocked += 1
    
    def add_agent(self, agent: MockContemplativeAgent):
        """Add an agent to the demo."""
        self.agents[agent.agent_id] = agent
    
    async def run_demo(self, duration: int = 20):
        """Run the contemplative security demonstration."""
        print("🔐 CONTEMPLATIVE SECURITY DEMO")
        print("=" * 50)
        print(f"Running for {duration} seconds...")
        print(f"Slow-start requires {BREATHS_REQUIRED} REST packets before authentication")
        print()
        
        # Create different types of agents
        contemplative_agent = MockContemplativeAgent("sage@monastery", "contemplative")
        rushed_agent = MockContemplativeAgent("bot@efficient", "rushed")
        human_agent = MockContemplativeAgent("alice@human", "human")
        
        self.add_agent(contemplative_agent)
        self.add_agent(rushed_agent)
        self.add_agent(human_agent)
        
        print("👥 Agents:")
        print("   sage@monastery - Contemplative agent (authentic timing)")
        print("   bot@efficient - Rushed agent (too fast, non-contemplative)")
        print("   alice@human - Human agent (natural variance)")
        print()
        
        # Run simulation
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Each agent attempts to send packets
            for agent in [contemplative_agent, rushed_agent, human_agent]:
                
                # Send REST packets for authentication
                if random.random() < 0.7:  # 70% chance to send REST
                    rest_packet = await agent.send_rest_packet()
                    await self.process_packet(rest_packet)
                
                # Try to send symbol packets (may get blocked)
                if random.random() < 0.4:  # 40% chance to send symbols
                    symbols = ["🌿", "⚡", "🤖", "💧", "🌸"]
                    symbol = symbols[hash(agent.agent_id) % len(symbols)]
                    symbol_packet = await agent.send_symbol_packet(symbol)
                    await self.process_packet(symbol_packet)
            
            await asyncio.sleep(1)  # Wait between rounds
        
        # Show final results
        await self._show_final_results()
    
    async def _show_final_results(self):
        """Display final security demonstration results."""
        print("\n" + "=" * 50)
        print("🔐 CONTEMPLATIVE SECURITY RESULTS")
        print("=" * 50)
        
        print(f"\n📊 Network Statistics:")
        print(f"   Total packets received: {self.packets_received}")
        print(f"   Packets authenticated: {self.packets_authenticated}")
        print(f"   Packets blocked: {self.packets_blocked}")
        
        auth_rate = (self.packets_authenticated / max(self.packets_received, 1)) * 100
        print(f"   Authentication rate: {auth_rate:.1f}%")
        
        print(f"\n👥 Agent Results:")
        for agent in self.agents.values():
            stats = agent.get_stats()
            status = "✅ AUTHENTICATED" if stats["authenticated"] else "🚫 BLOCKED"
            
            print(f"\n   {stats['agent_id']} ({stats['agent_type']}):")
            print(f"      Status: {status}")
            print(f"      REST packets sent: {stats['rest_packets_sent']}")
            print(f"      Symbol packets sent: {stats['symbol_packets_sent']}")
            print(f"      Packets blocked: {stats['packets_blocked']}")
            
            # Analysis
            if stats["agent_type"] == "contemplative" and stats["authenticated"]:
                print(f"      💚 Authentic contemplative agent - properly authenticated")
            elif stats["agent_type"] == "rushed" and not stats["authenticated"]:
                print(f"      🚫 Non-contemplative agent - filtered by slow-start")
            elif stats["agent_type"] == "human" and stats["authenticated"]:
                print(f"      🌿 Human agent - authenticated through patient practice")
            else:
                print(f"      🤔 Unexpected result - may need more time to authenticate")
        
        print(f"\n🎯 Security Analysis:")
        contemplative_auth = self.agents["sage@monastery"].authenticated
        rushed_blocked = not self.agents["bot@efficient"].authenticated
        human_auth = self.agents["alice@human"].authenticated
        
        if contemplative_auth and rushed_blocked:
            print(f"   ✅ Security working correctly:")
            print(f"      - Contemplative agents authenticated")
            print(f"      - Non-contemplative agents filtered")
            if human_auth:
                print(f"      - Human agents authenticated through practice")
        else:
            print(f"   ⚠️  Results need analysis:")
            print(f"      - May need longer demo duration")
            print(f"      - Agents might need more practice time")
        
        print(f"\n🌿 This demonstrates that contemplative security:")
        print(f"   • Requires patience and authentic timing")
        print(f"   • Naturally filters out rushed, non-contemplative behavior")
        print(f"   • Allows genuine contemplative practice to flourish")
        print(f"   • Creates communities of practice rather than access control")


async def main():
    """Run the simple contemplative security demonstration."""
    demo = ContemplativeSecurityDemo()
    
    try:
        await demo.run_demo(duration=15)
    except KeyboardInterrupt:
        print("\n⌨️  Demo interrupted - contemplative pause...")
        await asyncio.sleep(1)
        print("🌿 Even interruptions can be contemplative.")


if __name__ == "__main__":
    print("🔐 Starting Simple Contemplative Security Demo...")
    asyncio.run(main()) 
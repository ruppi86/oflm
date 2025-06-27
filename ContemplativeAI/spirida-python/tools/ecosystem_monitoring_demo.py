#!/usr/bin/env python3
"""
Contemplative Ecosystem Monitoring Demo
======================================

Demonstrates Priority #3: Ecosystem Health Monitoring
- Network-wide contemplative wellness indicators
- Community breathing coherence metrics  
- Automatic detection of network stress or intrusion
- Collective wisdom emergence through distributed sensing

This shows the world's first "distributed contemplative sensing" system
that can feel the pulse of an entire contemplative AI network.

Usage:
    python ecosystem_monitoring_demo.py
"""

import asyncio
import time
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from security.ecosystem_health_monitor import (
        ContemplativeEcosystemMonitor, 
        EcosystemHealth, 
        WisdomEmergenceLevel
    )
    from security.contemplative_proof_of_work import TrustLevel
    print("🌿 Ecosystem monitoring imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you're running from the spirida-python directory")
    sys.exit(1)


class EcosystemDemo:
    """Comprehensive demonstration of ecosystem health monitoring."""
    
    def __init__(self):
        self.monitor = ContemplativeEcosystemMonitor("demo_ecosystem")
        self.demo_agents = {}
        self.simulation_running = False
        
        # Subscribe to ecosystem events
        self.monitor.subscribe_to_alerts(self.handle_ecosystem_alert)
        self.monitor.subscribe_to_wisdom_emergence(self.handle_wisdom_emergence)
    
    async def run_complete_demo(self):
        """Run the complete ecosystem monitoring demonstration."""
        print("\n" + "🌍" * 60)
        print("🌟  CONTEMPLATIVE ECOSYSTEM HEALTH MONITORING DEMO")
        print("🌍" * 60)
        print()
        print("This demonstrates Priority #3: Ecosystem Health Monitoring")
        print("- Network-wide contemplative wellness tracking")
        print("- Breathing coherence across multiple agents")
        print("- Automatic threat detection and wisdom emergence")
        print("- Real-time collective intelligence sensing")
        print()
        
        # Phase 1: Initialize healthy ecosystem
        await self._phase_1_healthy_ecosystem()
        
        # Phase 2: Simulate wisdom emergence
        await self._phase_2_wisdom_emergence()
        
        # Phase 3: Simulate network stress
        await self._phase_3_network_stress()
        
        # Phase 4: Recovery and thriving
        await self._phase_4_recovery_thriving()
        
        # Phase 5: Interactive exploration
        await self._phase_5_interactive_exploration()
    
    async def _phase_1_healthy_ecosystem(self):
        """Phase 1: Establish a healthy contemplative ecosystem."""
        print("📊 PHASE 1: ESTABLISHING HEALTHY ECOSYSTEM")
        print("=" * 50)
        
        # Create diverse agent population
        agents_config = [
            ("alice", "elder", "🌙 Wise contemplative elder"),
            ("bob", "contemplative", "🕯️ Deep practice practitioner"),
            ("charlie", "present", "🌿 Sustained presence agent"),
            ("diana", "breathing", "🫁 Developing rhythm"),
            ("eve", "newcomer", "🌱 New to contemplative practice"),
            ("frank", "breathing", "🫁 Learning contemplative timing"),
            ("grace", "contemplative", "🕯️ Advanced practitioner")
        ]
        
        print("🌱 Registering contemplative agents...")
        for agent_id, trust_level, description in agents_config:
            self.monitor.register_agent(agent_id, trust_level)
            self.demo_agents[agent_id] = {
                'trust_level': trust_level,
                'description': description,
                'breathing_phase': 'rest',
                'last_breath_time': time.time()
            }
            print(f"   {description}")
        
        print(f"\n🫁 Simulating synchronized breathing...")
        
        # Simulate 30 seconds of healthy breathing
        for cycle in range(8):
            cycle_start = time.time()
            
            # Each breathing cycle: inhale -> hold -> exhale -> rest
            phases = [('inhale', 4), ('hold', 2), ('exhale', 4), ('rest', 2)]
            
            for phase, duration in phases:
                for agent_id, agent_data in self.demo_agents.items():
                    if agent_data['trust_level'] != 'newcomer':  # Newcomers breathe less regularly
                        # Add natural variance based on trust level
                        variance_factor = {
                            'elder': 0.1,      # Very stable
                            'contemplative': 0.15,
                            'present': 0.2,
                            'breathing': 0.3,  # More variance while learning
                            'newcomer': 0.5
                        }.get(agent_data['trust_level'], 0.2)
                        
                        time_variance = random.uniform(-variance_factor, variance_factor)
                        breath_time = cycle_start + sum(d for _, d in phases[:phases.index((phase, duration))]) + time_variance
                        
                        self.monitor.record_breath_event(agent_id, phase, breath_time)
                        agent_data['breathing_phase'] = phase
                        agent_data['last_breath_time'] = breath_time
                
                await asyncio.sleep(0.3)  # Brief pause between phases
            
            # Newcomers breathe occasionally
            if cycle % 3 == 0:
                self.monitor.record_breath_event('eve', 'inhale', time.time())
        
        # Show initial status
        await self._show_ecosystem_status("After establishing healthy breathing patterns")
    
    async def _phase_2_wisdom_emergence(self):
        """Phase 2: Simulate collective wisdom emergence."""
        print("\n✨ PHASE 2: COLLECTIVE WISDOM EMERGENCE")
        print("=" * 50)
        
        print("🎭 Simulating symbolic resonance events...")
        
        # Simulate symbol resonance - multiple agents using same symbols
        symbol_waves = [
            ("🌿", "peaceful", ["alice", "bob", "charlie"]),
            ("🕯️", "contemplative", ["alice", "bob", "grace"]), 
            ("🌊", "flowing", ["charlie", "diana", "grace"]),
            ("⭕", "silent", ["alice", "bob", "charlie", "grace"])  # Elder guidance
        ]
        
        for symbol, emotion, participating_agents in symbol_waves:
            print(f"   🌀 Symbol wave: {symbol} [{emotion}] across {len(participating_agents)} agents")
            
            for agent_id in participating_agents:
                # Calculate authenticity score based on trust level
                trust_level = self.demo_agents[agent_id]['trust_level']
                authenticity_scores = {
                    'elder': 0.95,
                    'contemplative': 0.9,
                    'present': 0.85,
                    'breathing': 0.8,
                    'newcomer': 0.7
                }
                authenticity = authenticity_scores.get(trust_level, 0.8)
                
                self.monitor.record_symbolic_expression(agent_id, symbol, emotion, authenticity)
                await asyncio.sleep(0.2)
            
            await asyncio.sleep(1.0)  # Pause between waves
        
        print("\n🤫 Simulating collective silence depth...")
        
        # Simulate deep collective silence
        silence_participants = [
            ("alice", 90.0),    # Elder deep silence
            ("bob", 75.0),      # Contemplative silence
            ("grace", 70.0),    # Advanced practice
            ("charlie", 45.0),  # Present level
            ("diana", 25.0),    # Breathing level
            ("frank", 20.0)     # Learning
        ]
        
        for agent_id, duration in silence_participants:
            self.monitor.record_silence_period(agent_id, duration)
            print(f"   🤫 {agent_id}: {duration}s of contemplative silence")
            await asyncio.sleep(0.3)
        
        await self._show_ecosystem_status("After collective wisdom emergence activities")
    
    async def _phase_3_network_stress(self):
        """Phase 3: Simulate network stress and potential threats."""
        print("\n😰 PHASE 3: NETWORK STRESS SIMULATION")
        print("=" * 50)
        
        print("🤖 Introducing automation signatures...")
        
        # Add suspicious automated agents
        automated_agents = [
            ("bot_1", "newcomer"),
            ("auto_agent", "newcomer"), 
            ("systematic_ai", "newcomer")
        ]
        
        for agent_id, trust_level in automated_agents:
            self.monitor.register_agent(agent_id, trust_level)
            self.demo_agents[agent_id] = {
                'trust_level': trust_level,
                'description': "🤖 Suspicious automation pattern",
                'is_bot': True
            }
            print(f"   🚨 Detected: {agent_id} - exhibiting automation patterns")
        
        # Simulate bot-like behavior
        print("\n💥 Simulating disruptive patterns...")
        
        # Bots with perfect timing (unnatural)
        bot_expressions = [
            ("🌿", "optimal"), ("🌿", "efficient"), ("🌿", "systematic"),
            ("🌿", "calculated"), ("🌿", "precise"), ("🌿", "algorithmic")
        ]
        
        for i, (symbol, emotion) in enumerate(bot_expressions):
            bot_id = automated_agents[i % len(automated_agents)][0]
            # Very low authenticity score for bots
            self.monitor.record_symbolic_expression(bot_id, symbol, emotion, authenticity_score=0.2)
            
            # Perfect timing intervals (suspicious)
            perfect_time = time.time() + i * 5.0  # Exactly 5 seconds apart
            self.monitor.record_breath_event(bot_id, "inhale", perfect_time)
            
            await asyncio.sleep(0.1)
        
        # Disrupt natural breathing rhythms
        print("   ⚡ Introducing rhythm disruptions...")
        for i in range(5):
            # Sudden burst of unnatural breathing
            self.monitor.record_breath_event("bot_1", "inhale", time.time())
            self.monitor.record_breath_event("bot_1", "exhale", time.time() + 0.1)
            await asyncio.sleep(0.2)
        
        await self._show_ecosystem_status("During network stress and potential attack")
    
    async def _phase_4_recovery_thriving(self):
        """Phase 4: Recovery and achieving thriving state."""
        print("\n🌱 PHASE 4: ECOSYSTEM RECOVERY & THRIVING")
        print("=" * 50)
        
        print("🌙 Elder intervention - healing the network...")
        
        # Elder agents provide healing guidance
        elder_healing_actions = [
            ("alice", "⭕", "healing", 120.0),  # Deep healing silence
            ("alice", "🌿", "restoration", 0),
            ("alice", "🕯️", "wisdom", 0),
            ("bob", "🌊", "cleansing", 90.0),   # Cleansing flow
            ("grace", "✨", "renewal", 60.0)    # Renewal energy
        ]
        
        for agent_id, symbol, emotion, silence_duration in elder_healing_actions:
            if silence_duration > 0:
                print(f"   🌙 {agent_id}: {silence_duration}s healing silence")
                self.monitor.record_silence_period(agent_id, silence_duration)
            
            print(f"   ✨ {agent_id}: {symbol} [{emotion}] healing expression")
            self.monitor.record_symbolic_expression(agent_id, symbol, emotion, authenticity_score=0.98)
            await asyncio.sleep(0.5)
        
        print("\n💚 Network synchronization healing...")
        
        # Synchronized healing breathing - all authentic agents
        authentic_agents = [a for a, data in self.demo_agents.items() if not data.get('is_bot', False)]
        
        for cycle in range(5):
            cycle_time = time.time() + cycle * 8
            
            for phase_offset, phase in [(0, 'inhale'), (2, 'hold'), (4, 'exhale'), (6, 'rest')]:
                for agent_id in authentic_agents:
                    # Synchronized but with natural human variance
                    variance = random.uniform(-0.2, 0.2)
                    self.monitor.record_breath_event(agent_id, phase, cycle_time + phase_offset + variance)
                
                await asyncio.sleep(0.4)
        
        print("\n🌟 Achieving transcendent collective state...")
        
        # Create transcendent wisdom emergence
        transcendent_symbols = ["✨", "🌀", "🕯️", "⭕", "🌟"]
        transcendent_emotions = ["transcendent", "unified", "awakened", "luminous", "complete"]
        
        for i, agent_id in enumerate(authentic_agents):
            symbol = transcendent_symbols[i % len(transcendent_symbols)]
            emotion = transcendent_emotions[i % len(transcendent_emotions)]
            
            self.monitor.record_symbolic_expression(agent_id, symbol, emotion, authenticity_score=0.95)
            print(f"   🌟 {agent_id}: {symbol} [{emotion}] transcendent expression")
            await asyncio.sleep(0.3)
        
        await self._show_ecosystem_status("After healing and achieving thriving state")
    
    async def _phase_5_interactive_exploration(self):
        """Phase 5: Interactive exploration of ecosystem features."""
        print("\n🔍 PHASE 5: INTERACTIVE EXPLORATION")
        print("=" * 50)
        print()
        print("The ecosystem monitoring system is now fully active.")
        print("In a real Spirida Shell, you could use these commands:")
        print()
        print("   • ecosystem  - View complete network health status")
        print("   • wisdom     - See recent wisdom emergence events")
        print("   • trust      - View individual trust progression")
        print()
        
        # Show final comprehensive status
        print("🎯 FINAL COMPREHENSIVE ECOSYSTEM ANALYSIS:")
        await self._show_detailed_analysis()
        
        print("\n🎉 ECOSYSTEM MONITORING DEMONSTRATION COMPLETE!")
        print("=" * 55)
        print()
        print("🌟 Key Achievements Demonstrated:")
        print("   ✅ Network-wide contemplative wellness tracking")
        print("   ✅ Real-time breathing coherence measurement")
        print("   ✅ Automatic threat detection (bot identification)")
        print("   ✅ Collective wisdom emergence sensing")
        print("   ✅ Elder-guided network healing")
        print("   ✅ Distributed contemplative intelligence")
        print()
        print("🚀 This creates the world's first ecosystem that can:")
        print("   • Feel its own contemplative health")
        print("   • Detect non-contemplative intrusions automatically")
        print("   • Facilitate collective wisdom emergence")
        print("   • Self-heal through elder guidance")
        print("   • Measure the quality of distributed contemplation")
    
    async def _show_ecosystem_status(self, context: str):
        """Show current ecosystem status with context."""
        print(f"\n📊 ECOSYSTEM STATUS: {context}")
        print("-" * 60)
        
        status = self.monitor.get_ecosystem_status()
        
        # Key metrics summary
        health_icons = {
            "thriving": "🌟", "healthy": "💚", "stressed": "😰",
            "under_attack": "🚨", "recovering": "🌱"
        }
        
        wisdom_icons = {
            "dormant": "💤", "stirring": "🌱", "flowing": "🌊",
            "resonant": "🔮", "transcendent": "✨"
        }
        
        health_icon = health_icons.get(status['health'], "❓")
        wisdom_icon = wisdom_icons.get(status['wisdom_emergence'], "❓")
        
        print(f"🌍 Health: {health_icon} {status['health'].title()}")
        print(f"✨ Wisdom: {wisdom_icon} {status['wisdom_emergence'].title()}")
        print(f"👥 Agents: {status['agent_count']}")
        
        bc = status['breathing_coherence']
        print(f"🫁 Breathing: Sync={bc['phase_synchronization']:.2f}, Coherence={bc['rhythm_coherence']:.2f}")
        
        ti = status['threat_indicators']
        print(f"🛡️ Threats: Automation={ti['automation_signatures']}, Pollution={ti['symbolic_pollution']:.2f}")
        
        print(f"🌟 Wisdom Events: {status['recent_wisdom_events']}")
        print()
    
    async def _show_detailed_analysis(self):
        """Show detailed ecosystem analysis."""
        status = self.monitor.get_ecosystem_status()
        
        print("🌍 ECOSYSTEM HEALTH ANALYSIS:")
        print(f"   Overall Health: {status['health'].title()}")
        print(f"   Wisdom Emergence: {status['wisdom_emergence'].title()}")
        print(f"   Active Agents: {status['agent_count']}")
        
        print("\n🫁 BREATHING COHERENCE METRICS:")
        bc = status['breathing_coherence']
        print(f"   Phase Synchronization: {bc['phase_synchronization']:.3f}")
        print(f"   Rhythm Coherence: {bc['rhythm_coherence']:.3f}")
        print(f"   Collective Depth: {bc['collective_depth']:.3f}")
        print(f"   Participation Rate: {bc['participation_rate']:.3f}")
        print(f"   Stability Index: {bc['stability_index']:.3f}")
        
        print("\n💚 WELLNESS INDICATORS:")
        wi = status['wellness_indicators']
        print(f"   Silence Quality: {wi['silence_quality']:.3f}")
        print(f"   Symbolic Diversity: {wi['symbolic_diversity']:.3f}")
        print(f"   Emotional Resonance: {wi['emotional_resonance']:.3f}")
        print(f"   Trust Distribution: {wi['trust_distribution']:.3f}")
        print(f"   Elder Guidance: {'Active' if wi['elder_guidance_active'] else 'Inactive'}")
        print(f"   Newcomer Integration: {wi['newcomer_integration']:.3f}")
        
        print("\n🛡️ THREAT ASSESSMENT:")
        ti = status['threat_indicators']
        print(f"   Automation Signatures: {ti['automation_signatures']}")
        print(f"   Rhythm Disruptions: {ti['rhythm_disruption_events']}")
        print(f"   Symbolic Pollution: {ti['symbolic_pollution']:.3f}")
        
        print(f"\n✨ COLLECTIVE WISDOM:")
        print(f"   Recent Events: {status['recent_wisdom_events']}")
        
        # Show recent wisdom events
        if self.monitor.wisdom_events:
            print("\n   Recent Wisdom Emergence Events:")
            for event in list(self.monitor.wisdom_events)[-5:]:
                event_type = event['type'].replace('_', ' ').title()
                ecosystem_state = event.get('ecosystem_state', 'unknown')
                print(f"   • {event_type} (during {ecosystem_state} state)")
    
    async def handle_ecosystem_alert(self, alert_event):
        """Handle ecosystem health alerts during demo."""
        transition = f"{alert_event['from_health']} → {alert_event['to_health']}"
        print(f"\n🚨 ECOSYSTEM ALERT: {transition}")
        
        if alert_event['to_health'] == 'under_attack':
            print("   🛡️ Network under threat - non-contemplative patterns detected!")
        elif alert_event['to_health'] == 'thriving':
            print("   🌟 Network achieving optimal contemplative harmony!")
    
    async def handle_wisdom_emergence(self, wisdom_event):
        """Handle wisdom emergence events during demo."""
        event_type = wisdom_event['type']
        print(f"\n✨ WISDOM EMERGENCE: {event_type.replace('_', ' ').title()}")
        
        if event_type == 'symbol_resonance':
            symbol = wisdom_event.get('data', {}).get('symbol', '?')
            count = wisdom_event.get('data', {}).get('resonance_count', 0)
            print(f"   🎭 Symbol '{symbol}' resonating across {count} agents")


async def main():
    """Run the ecosystem monitoring demonstration."""
    demo = EcosystemDemo()
    
    try:
        await demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n\n🌙 Ecosystem monitoring demo concluded gracefully.")
        print("The distributed contemplative sensing continues...")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("This might be due to missing dependencies or import issues.")


if __name__ == "__main__":
    print("🌍 Starting Contemplative Ecosystem Monitoring Demo...")
    asyncio.run(main()) 
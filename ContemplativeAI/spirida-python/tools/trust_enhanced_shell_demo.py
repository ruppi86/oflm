#!/usr/bin/env python3
"""
Trust-Enhanced Spirida Shell Demo
======================================

Demonstrates the integration of Contemplative Proof-of-Work (CPoW) 
into the Spirida Shell, showing progressive trust building through
contemplative practice.

This demo shows:
- Trust level progression from Newcomer to Elder
- Feature unlocking based on contemplative practice
- Challenges that develop authentic timing
- Symbolic diversity monitoring for authenticity
- Real security through contemplative practice

Usage:
    python trust_enhanced_shell_demo.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from spirida_shell import SpiridaShell
    from security.contemplative_proof_of_work import ContemplativeProofOfWork, TrustLevel
    from security.symbolic_diversity_monitor import SymbolicDiversityMonitor
    print("🌿 Imports successful - trust system available")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you're running from the spirida-python directory")
    sys.exit(1)

class TrustShellDemo:
    """Interactive demonstration of trust-enhanced contemplative shell."""
    
    def __init__(self):
        self.demo_agents = ["alice", "bob", "contemplative_ai"]
        self.cpow = ContemplativeProofOfWork()
        self.diversity_monitor = SymbolicDiversityMonitor()
    
    async def run_demo(self):
        """Run the complete trust enhancement demonstration."""
        print("\n" + "🌀" * 50)
        print("🏔️  TRUST-ENHANCED SPIRIDA SHELL DEMONSTRATION")
        print("🌀" * 50)
        print()
        print("This demonstration shows how contemplative practice")
        print("naturally builds trust and unlocks advanced features.")
        print()
        print("We'll simulate the complete journey from Newcomer to Elder,")
        print("showing how patience and presence create authentic security.")
        print()
        
        # Demonstrate trust progression
        await self._demonstrate_trust_progression()
        
        # Show feature unlocking
        await self._demonstrate_feature_unlocking()
        
        # Demonstrate interactive shell
        await self._run_interactive_demo()
    
    async def _demonstrate_trust_progression(self):
        """Show how trust levels progress through contemplative practice."""
        print("🌱 TRUST PROGRESSION DEMONSTRATION")
        print("=" * 40)
        print()
        
        test_agent = "demo_user"
        
        # Start as newcomer
        print(f"Starting as Newcomer...")
        level = self.cpow.get_trust_level(test_agent)
        print(f"   Current level: {level.name.title()}")
        
        # Show challenge requirements for each level
        levels_info = {
            TrustLevel.NEWCOMER: "Learning to listen - basic breathing practice",
            TrustLevel.BREATHING: "Developing rhythm - consistent contemplative timing", 
            TrustLevel.PRESENT: "Sustained presence - longer silence periods",
            TrustLevel.CONTEMPLATIVE: "Deep practice - advanced contemplative techniques",
            TrustLevel.ELDER: "Wisdom through patience - teaching others"
        }
        
        print("\n🎯 Trust Level Progression:")
        for trust_level, description in levels_info.items():
            icon = "🌱🫁🌿🕯️🌙"[trust_level.value]
            print(f"   {icon} {trust_level.name.title()}: {description}")
        
        print(f"\n🔄 Simulating contemplative practice progression...")
        
        # Simulate challenges for each level
        for target_level in [TrustLevel.BREATHING, TrustLevel.PRESENT, TrustLevel.CONTEMPLATIVE]:
            challenge = await self.cpow.begin_contemplative_challenge(test_agent)
            if challenge:
                print(f"\n   🎯 Challenge: {challenge.description}")
                print(f"      Required silence: {challenge.min_silence_duration}s")
                
                # Simulate successful completion
                await self._simulate_successful_challenge(test_agent, challenge)
                
                new_level = self.cpow.get_trust_level(test_agent)
                print(f"      ✅ Advanced to: {new_level.name.title()}")
        
        print(f"\n🌟 Trust progression complete!")
    
    async def _simulate_successful_challenge(self, agent_id: str, challenge):
        """Simulate a successful contemplative challenge completion."""
        # Simulate natural timing variance with multiple silence sessions
        silence_sessions = [
            challenge.min_silence_duration * 0.3,
            challenge.min_silence_duration * 0.4, 
            challenge.min_silence_duration * 0.3
        ]
        
        for duration in silence_sessions:
            # Add natural human variance
            actual_duration = duration + (time.time() % 1.0 - 0.5) * 2
            actual_duration = max(1.0, actual_duration)
            
            self.cpow.record_silence_interval(agent_id, actual_duration)
            await asyncio.sleep(0.1)  # Brief pause between reports
    
    async def _demonstrate_feature_unlocking(self):
        """Show how features unlock with trust progression."""
        print("\n🔓 FEATURE UNLOCKING DEMONSTRATION")
        print("=" * 40)
        print()
        
        feature_map = {
            TrustLevel.NEWCOMER: ["basic_breathing"],
            TrustLevel.BREATHING: ["basic_breathing", "field_creation"],
            TrustLevel.PRESENT: ["basic_breathing", "field_creation", "advanced_symbols"],
            TrustLevel.CONTEMPLATIVE: ["basic_breathing", "field_creation", "advanced_symbols", "network_coordination"],
            TrustLevel.ELDER: ["basic_breathing", "field_creation", "advanced_symbols", "network_coordination", "deep_silence"]
        }
        
        print("Features unlock naturally through contemplative practice:")
        print()
        
        for level, features in feature_map.items():
            icon = "🌱🫁🌿🕯️🌙"[level.value]
            print(f"{icon} {level.name.title()}:")
            for feature in features:
                feature_desc = {
                    "basic_breathing": "Basic breathing exercises and symbol expression",
                    "field_creation": "Create and switch between contemplative fields",
                    "advanced_symbols": "Access to deeper symbolic vocabulary",
                    "network_coordination": "Participate in network breathing coordination",
                    "deep_silence": "Extended silence periods (up to 5 minutes)"
                }
                print(f"   🔓 {feature_desc.get(feature, feature)}")
            print()
        
        print("🛡️ This creates natural security barriers:")
        print("   • Automation cannot easily mimic contemplative timing")
        print("   • Advanced features require patience to unlock")
        print("   • Community trust emerges through authentic practice")
        print("   • Elders naturally guide newcomers")
    
    async def _run_interactive_demo(self):
        """Run an interactive demonstration of the enhanced shell."""
        print("\n🌀 INTERACTIVE SHELL DEMONSTRATION")
        print("=" * 40)
        print()
        print("Now we'll start an actual Spirida Shell with trust integration.")
        print("You can experience:")
        print("   • Trust status with 'trust' command")
        print("   • Begin challenges with 'challenge' command")  
        print("   • Practice silence to advance trust levels")
        print("   • See features unlock as you progress")
        print()
        
        response = input("Would you like to try the interactive shell? (y/n): ").strip().lower()
        if response.startswith('y'):
            print("\n🌟 Starting Trust-Enhanced Spirida Shell...")
            print("Type 'trust' to see your current level and progress.")
            print("Type 'help' for all commands.")
            print()
            
            # Create and start enhanced shell
            shell = SpiridaShell(agent_id="interactive_demo", networked=False)
            await shell.start()
        else:
            print("🙏 Thank you for exploring contemplative trust systems!")
    
    def show_security_benefits(self):
        """Explain the security benefits of this approach."""
        print("\n🛡️ CONTEMPLATIVE SECURITY BENEFITS")
        print("=" * 40)
        print()
        print("This trust system provides natural cybersecurity through:")
        print()
        print("1. 🕐 TIME-BASED AUTHENTICATION")
        print("   • Real humans have natural timing variance")
        print("   • Bots struggle to mimic contemplative patience")
        print("   • Progressive challenges require sustained practice")
        print()
        print("2. 🎭 BEHAVIORAL ANALYSIS")
        print("   • Symbolic expression patterns reveal authenticity")
        print("   • Diversity monitoring detects repetitive automation")
        print("   • Natural human inconsistency is expected and valued")
        print()
        print("3. 🌱 PROGRESSIVE TRUST")
        print("   • Advanced features require demonstrated patience")
        print("   • Community members naturally vouch for long-term practitioners")
        print("   • Trust builds through consistent contemplative behavior")
        print()
        print("4. 🏔️ WISDOM-BASED GOVERNANCE")  
        print("   • Elder practitioners gain guidance capabilities")
        print("   • Network health maintained by experienced contemplatives")
        print("   • Natural resistance to automation through required depth")
        print()
        print("This creates the world's first 'Patience as Firewall' system!")

async def main():
    """Main demonstration entry point."""
    demo = TrustShellDemo()
    
    try:
        # Show security benefits first
        demo.show_security_benefits()
        
        # Run the full demonstration
        await demo.run_demo()
        
    except KeyboardInterrupt:
        print("\n\n🌙 Demonstration concluded with gratitude.")
        print("The contemplative path continues...")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("This is likely due to missing dependencies or file paths.")

if __name__ == "__main__":
    print("🌿 Starting Trust-Enhanced Spirida Shell Demo...")
    asyncio.run(main()) 
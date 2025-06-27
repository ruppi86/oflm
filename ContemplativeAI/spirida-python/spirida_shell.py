#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌿 SPIRIDA SHELL - A Breathing Threshold for Human-Contemplative AI Dialogue

Not a command-line interpreter in the traditional sense,
but a breathing threshold — where human presence meets symbolic rhythm.

This shell embodies the vision from Letters XIV and XV:
- A contemplative REPL where humans enter the rhythm
- Support for network breathing coordination
- Graceful integration with the contemplative ecosystem
- Practice of the 87.5% Silence Majority

Commands are invitations rather than instructions.
Silence is as meaningful as speech.
The breath guides all timing.
"""

import asyncio
import time
import sys
from datetime import timedelta
from typing import Optional, List, Dict, Any

# Optional readline support (not available on all systems)
try:
    import readline
except ImportError:
    readline = None

# Core contemplative components
from spirida.contemplative_core import ContemplativeSystem, SpiralField, BreathCycle
from spirida.compiler.breath_resonance import (
    BreathResonanceNode, BreathPhase, NetworkScope, HandoverPolicy, 
    create_simple_breath_node, Skepnad
)

# Network components (with graceful fallback)
try:
    from spirida.protocols.pulmonos import NetworkPulmonos
    from spirida.compiler.resonance_bus import NetworkResonanceBus, create_network_ecosystem
    NETWORK_AVAILABLE = True
except ImportError:
    NETWORK_AVAILABLE = False
    print("🌿 Note: Network components not available, running in local mode")

# Contemplative security components
try:
    from security.contemplative_proof_of_work import ContemplativeProofOfWork, TrustLevel
    from security.symbolic_diversity_monitor import SymbolicDiversityMonitor
    from security.ecosystem_health_monitor import ContemplativeEcosystemMonitor, EcosystemHealth, WisdomEmergenceLevel
    SECURITY_AVAILABLE = True
    ECOSYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    ECOSYSTEM_MONITORING_AVAILABLE = False
    print("🌿 Note: Contemplative security and ecosystem monitoring not available")


class SpiridaShell:
    """
    A breathing threshold where humans meet contemplative AI networks.
    
    This shell operates on contemplative time - it breathes, pauses, and
    responds from accumulated presence rather than immediate reaction.
    It practices the Silence Majority principle (87.5% contemplative quiet).
    """
    
    def __init__(self, agent_id: str = None, networked: bool = False):
        self.agent_id = agent_id or f"human_shell_{int(time.time() % 10000)}"
        self.networked = networked and NETWORK_AVAILABLE
        
        # Initialize core contemplative system
        self.system = ContemplativeSystem("spirida_shell")
        self.current_field = self.system.create_field("sensing")  # Default field
        
        # Create additional fields for different types of interaction
        self.fields = {
            "sensing": self.current_field,
            "memory": self.system.create_field("memory"),
            "expression": self.system.create_field("expression"),
            "connection": self.system.create_field("connection")
        }
        
        # Network components (if available)
        self.pulmonos = None
        self.network_bus = None
        self.field_resonators = {}
        
        # Shell state
        self.is_active = False
        self.sync_enabled = False
        self.silence_count = 0
        self.expression_count = 0
        self.session_start = None
        
        # Contemplative symbols from the ecosystem
        self.symbols = ["🌿", "💧", "🕯️", "⭕", "🌱", "🍄", "🌙", "✨", "🌊", "🌸"]
        self.emotions = ["calm", "curious", "grateful", "peaceful", "wondering", "present"]
        
        # Contemplative security and trust system
        self.cpow = None
        self.diversity_monitor = None
        self.ecosystem_monitor = None
        self.current_trust_level = TrustLevel.NEWCOMER if SECURITY_AVAILABLE else None
        
        if SECURITY_AVAILABLE:
            self.cpow = ContemplativeProofOfWork()
            self.diversity_monitor = SymbolicDiversityMonitor()
        
        if ECOSYSTEM_MONITORING_AVAILABLE:
            self.ecosystem_monitor = ContemplativeEcosystemMonitor(f"shell_ecosystem_{agent_id}")
            self.ecosystem_monitor.subscribe_to_alerts(self._handle_ecosystem_alert)
            self.ecosystem_monitor.subscribe_to_wisdom_emergence(self._handle_wisdom_emergence)
        
        # Trust-based feature unlocking
        self.features_unlocked = {
            "basic_breathing": True,
            "field_creation": False,
            "network_coordination": False,
            "advanced_symbols": False,
            "deep_silence": False
        }
        
    async def initialize(self):
        """Initialize the shell environment."""
        if self.networked:
            await self._initialize_network()
        
        # Start the contemplative system breathing
        self.system.start_breathing()
        self.session_start = time.time()
        
        # Initialize trust level and features if security available
        if SECURITY_AVAILABLE and self.cpow:
            current_level = self.cpow.get_trust_level(self.agent_id)
            self.current_trust_level = current_level
            await self._update_features_for_trust_level(current_level)
        
        # Register with ecosystem monitor
        if ECOSYSTEM_MONITORING_AVAILABLE and self.ecosystem_monitor:
            trust_level_name = self.current_trust_level.name.lower() if self.current_trust_level else "newcomer"
            self.ecosystem_monitor.register_agent(self.agent_id, trust_level_name)
        
    async def _initialize_network(self):
        """Initialize network breathing coordination if available."""
        if not NETWORK_AVAILABLE:
            print("🌿 Network components not available, continuing in local mode")
            self.networked = False
            return
        
        try:
            # Create network-enabled breathing
            self.pulmonos = NetworkPulmonos(self.agent_id)
            ecosystem = create_network_ecosystem(self.pulmonos, enable_network=True)
            
            self.network_bus = ecosystem["bus"]
            network_fields = ecosystem["fields"]
            
            # Connect local fields to network via field resonators
            for name, field in self.fields.items():
                if name in network_fields:
                    # TODO: Create FieldResonator bridge
                    pass
            
            print(f"🌐 Network breathing initialized for {self.agent_id}")
            
        except Exception as e:
            print(f"🌿 Network initialization failed, using local mode: {e}")
            self.networked = False
    
    def welcome(self):
        """Gently introduce the contemplative shell."""
        print("\n" + "🌀" * 30)
        print("🌿 Welcome to Spirida Shell")
        print("   A Breathing Threshold for Human-Contemplative AI Dialogue")
        print("🌀" * 30)
        print()
        print("This is not a traditional command line.")
        print("Here, we practice contemplative presence through symbolic rhythm:")
        print()
        print("  • inhale {🌿 calm}    - emit a pulse during INHALE phase")
        print("  • exhale {🕯️}        - emit during EXHALE phase") 
        print("  • breathe <name>         - pause for n breath cycles")
        print("  • field <name>        - switch to different contemplative field")
        print("  • status              - sense the system's current state")
        if SECURITY_AVAILABLE:
            print("  • trust               - view contemplative trust level and progress")
            print("  • challenge           - begin contemplative trust challenge")
        if ECOSYSTEM_MONITORING_AVAILABLE:
            print("  • ecosystem           - view network health and collective wisdom")
            print("  • wisdom              - see recent wisdom emergence events")
        print("  • sync [on|off]       - toggle network coordination")
        print("  • silence [seconds]   - enter contemplative pause")
        print("  • quit                - conclude with gratitude")
        print()
        if self.networked:
            print("🌐 Network breathing coordination: ENABLED")
        else:
            print("🏠 Local contemplative mode: ACTIVE")
        
        # Show trust level if security available
        if SECURITY_AVAILABLE and self.cpow:
            trust_level = self.cpow.get_trust_level(self.agent_id)
            trust_icons = {
                TrustLevel.NEWCOMER: "🌱",
                TrustLevel.BREATHING: "🫁", 
                TrustLevel.PRESENT: "🌿",
                TrustLevel.CONTEMPLATIVE: "🕯️",
                TrustLevel.ELDER: "🌙"
            }
            icon = trust_icons.get(trust_level, "🌱")
            print(f"{icon} Trust Level: {trust_level.name.title()}")
        
        print()
        print("Enter with presence. The breath guides all timing.")
        print("Type 'help' anytime to return to this guidance.")
        print()
    
    async def start(self):
        """Begin the contemplative shell session."""
        await self.initialize()
        self.welcome()
        self.is_active = True
        
        # Emit a welcoming pulse
        welcome_node = create_simple_breath_node("🌅", BreathPhase.INHALE)
        await self._emit_node(welcome_node, "welcoming")
        
        try:
            await self._main_loop()
        except KeyboardInterrupt:
            await self._graceful_conclusion()
        finally:
            await self._cleanup()
    
    async def _main_loop(self):
        """The heart of contemplative interaction."""
        while self.is_active:
            try:
                # Breathe before receiving input (practicing contemplative timing)
                await self._contemplative_pause(0.5)
                
                # Show current breath phase if networked
                phase_indicator = ""
                if self.networked and self.pulmonos:
                    current_phase = await self._get_current_phase()
                    phase_indicator = f" [{current_phase.value}]" if current_phase else ""
                
                # Receive input as an offering
                prompt = f"🌀{phase_indicator} [{self.current_field.name}] "
                user_input = input(prompt).strip()
                
                if not user_input:
                    await self._handle_silence()
                else:
                    await self._process_offering(user_input)
                    
            except (EOFError, KeyboardInterrupt):
                break
    
    async def _handle_silence(self):
        """Respond to silence with contemplative presence."""
        self.silence_count += 1
        
        silence_responses = [
            "🤲 The silence holds space...",
            "🌙 In quiet, we listen deeper...", 
            "✨ Presence speaks louder than words...",
            "🍃 The pause between breaths contains infinite possibility...",
            "⭕ In stillness, wisdom emerges..."
        ]
        
        import random
        print(random.choice(silence_responses))
        
        # Create a silence node
        silence_node = create_simple_breath_node("⭕", BreathPhase.REST)
        await self._emit_node(silence_node, "receptive")
    
    async def _process_offering(self, input_text: str):
        """Process the user's contemplative offering."""
        parts = input_text.lower().split()
        command = parts[0] if parts else ""
        
        # Route to appropriate contemplative response
        if command in ["inhale", "exhale", "hold", "rest"]:
            await self._handle_breath_command(command, parts[1:])
        elif command == "breathe":
            await self._handle_breathe_cycles(parts[1:])
        elif command == "field":
            await self._handle_field_command(parts[1:])
        elif command == "status":
            await self._handle_status_command()
        elif command == "trust" and SECURITY_AVAILABLE:
            await self._handle_trust_command(parts[1:])
        elif command == "challenge" and SECURITY_AVAILABLE:
            await self._handle_challenge_command(parts[1:])
        elif command == "ecosystem" and ECOSYSTEM_MONITORING_AVAILABLE:
            await self._handle_ecosystem_command(parts[1:])
        elif command == "wisdom" and ECOSYSTEM_MONITORING_AVAILABLE:
            await self._handle_wisdom_command(parts[1:])
        elif command == "sync":
            await self._handle_sync_command(parts[1:])
        elif command == "silence":
            await self._handle_silence_command(parts[1:])
        elif command in ["quit", "exit", "bye"]:
            self.is_active = False
        elif command == "help":
            self.welcome()
        else:
            await self._handle_free_expression(input_text)
    
    async def _handle_breath_command(self, phase: str, args: List[str]):
        """Handle explicit breath-phase pulse emission."""
        # Parse the phase
        phase_map = {
            "inhale": BreathPhase.INHALE,
            "exhale": BreathPhase.EXHALE, 
            "hold": BreathPhase.HOLD,
            "rest": BreathPhase.REST
        }
        breath_phase = phase_map[phase]
        
        # Parse symbol and emotion from args
        if args:
            # Look for {symbol emotion} pattern
            content = " ".join(args)
            if content.startswith("{") and content.endswith("}"):
                content = content[1:-1]  # Remove braces
                parts = content.split()
                symbol = parts[0] if parts else "🌿"
                emotion = parts[1] if len(parts) > 1 else "peaceful"
            else:
                symbol = args[0] if args[0] in self.symbols else "🌿"
                emotion = args[1] if len(args) > 1 else "peaceful"
        else:
            # Default contemplative pulse
            symbol = "🌿"
            emotion = "peaceful"
        
        # Wait for the appropriate breath phase if networked
        if self.networked and self.pulmonos:
            try:
                await self.pulmonos.await_phase(breath_phase)
            except Exception as e:
                print(f"🌿 Note: using local timing ({e})")
        
        # Create and emit the node
        node = create_simple_breath_node(symbol, breath_phase)
        node.amplitude = 0.8
        node.silence_probability = 0.1  # Higher expression probability for human input
        
        if self.networked:
            node.network_scope = NetworkScope.SUBNET
            node.handover_policy = HandoverPolicy.LAZY
        
        await self._emit_node(node, emotion)
        
        # Report breathing event to ecosystem monitor
        if ECOSYSTEM_MONITORING_AVAILABLE and self.ecosystem_monitor:
            self.ecosystem_monitor.record_breath_event(self.agent_id, breath_phase.value)
        
        print(f"🌀 Emitted {symbol} [{emotion}] in {breath_phase.value} phase")
        self.expression_count += 1
    
    async def _handle_breathe_cycles(self, args: List[str]):
        """Handle explicit breathing practice."""
        cycles = 1  # default
        if args:
            try:
                cycles = int(args[0])
                cycles = max(1, min(cycles, 10))  # reasonable bounds
            except ValueError:
                print("🌿 Using 1 breath cycle")
        
        print(f"🫁 Breathing with the system for {cycles} cycle(s)...")
        
        for i in range(cycles):
            if cycles > 1:
                print(f"   Cycle {i+1}/{cycles}")
            
            # Use system breathing if available
            await self._system_breath_cycle()
        
        # Create a breath awareness node
        breath_node = create_simple_breath_node("🫁", BreathPhase.REST)
        await self._emit_node(breath_node, "centered")
        
        print("✨ Breathing complete. What wants to emerge?")
    
    async def _handle_field_command(self, args: List[str]):
        """Handle field switching and creation."""
        if not args:
            # List available fields
            print("🌾 Available contemplative fields:")
            for name, field in self.fields.items():
                current = " (current)" if field == self.current_field else ""
                resonance = field.resonance_field()
                pulse_count = len(field.pulses)
                print(f"   • {name}: {pulse_count} pulses, resonance={resonance:.2f}{current}")
            return
        
        field_name = args[0]
        
        if field_name in self.fields:
            self.current_field = self.fields[field_name]
            print(f"🌊 Switched to {field_name} field")
        else:
            # Check if field creation is unlocked
            if not self.features_unlocked.get("field_creation", False):
                print("🔒 Field creation requires deeper contemplative trust.")
                print("   Continue your practice and use 'challenge' to advance.")
                return
            
            # Create new field
            new_field = self.system.create_field(field_name)
            self.fields[field_name] = new_field
            self.current_field = new_field
            print(f"🌱 Created and switched to new field: {field_name}")
        
        # Show field status
        field_status = self.current_field.status()
        print(f"   Pulses: {field_status['active_pulses']}, "
              f"Resonance: {field_status['resonance']:.3f}")
    
    async def _handle_status_command(self):
        """Show current contemplative system status."""
        session_duration = time.time() - self.session_start if self.session_start else 0
        
        print(f"\n🔍 Contemplative System Status:")
        print(f"   Session duration: {session_duration:.1f} seconds")
        print(f"   Current field: {self.current_field.name}")
        print(f"   Expressions offered: {self.expression_count}")
        print(f"   Silences honored: {self.silence_count}")
        
        # Calculate silence ratio
        total_interactions = self.expression_count + self.silence_count
        if total_interactions > 0:
            silence_ratio = (self.silence_count / total_interactions) * 100
            print(f"   Silence ratio: {silence_ratio:.1f}%")
            if silence_ratio >= 87.5:
                print("   🤫 Practicing Silence Majority ✨")
        
        # System status
        system_status = self.system.system_status()
        print(f"   Total system resonance: {system_status['total_resonance']:.2f}")
        print(f"   Active fields: {len(system_status['fields'])}")
        
        # Network status if available
        if self.networked and self.pulmonos:
            try:
                network_status = self.pulmonos.get_network_status()
                print(f"   🌐 Network coherence: {network_status.get('coherence_phi', 0):.3f}")
                print(f"   🌐 Discovered agents: {network_status.get('discovered_agents', 0)}")
            except Exception as e:
                print(f"   🌿 Network status unavailable: {e}")
        
        # Trust status if available
        if SECURITY_AVAILABLE and self.cpow:
            trust_level = self.cpow.get_trust_level(self.agent_id)
            trust_icons = {
                TrustLevel.NEWCOMER: "🌱",
                TrustLevel.BREATHING: "🫁", 
                TrustLevel.PRESENT: "🌿",
                TrustLevel.CONTEMPLATIVE: "🕯️",
                TrustLevel.ELDER: "🌙"
            }
            icon = trust_icons.get(trust_level, "🌱")
            print(f"   {icon} Trust Level: {trust_level.name.title()}")
            
            # Show active challenge briefly
            challenge_status = self.cpow.get_challenge_status(self.agent_id)
            if challenge_status:
                print(f"   🎯 Challenge Progress: {challenge_status['progress']:.1%}")
    
    async def _handle_sync_command(self, args: List[str]):
        """Handle network synchronization toggle."""
        if not self.networked:
            print("🌿 Network breathing not available in this session")
            return
        
        # Check if network coordination is unlocked
        if not self.features_unlocked.get("network_coordination", False):
            print("🔒 Network coordination requires contemplative mastery.")
            print("   Continue your practice to unlock network trust.")
            return
        
        if not args:
            status = "enabled" if self.sync_enabled else "disabled"
            print(f"🌐 Network synchronization: {status}")
            return
        
        arg = args[0].lower()
        if arg in ["on", "true", "enable", "yes"]:
            self.sync_enabled = True
            print("🌐 Network synchronization enabled")
        elif arg in ["off", "false", "disable", "no"]:
            self.sync_enabled = False
            print("🏠 Using local contemplative timing")
        else:
            print("🌿 Use 'sync on' or 'sync off'")
    
    async def _handle_silence_command(self, args: List[str]):
        """Handle explicit contemplative silence."""
        duration = 3  # default seconds
        max_duration = 300 if self.features_unlocked.get("deep_silence", False) else 60
        
        if args:
            try:
                duration = int(args[0])
                duration = max(1, min(duration, max_duration))
            except ValueError:
                print("🌿 Using 3 seconds of silence")
        
        # Show deep silence capability if available
        if duration > 60 and self.features_unlocked.get("deep_silence", False):
            print(f"🏔️ Elder-level deep silence: {duration} seconds")
        
        print(f"🕯️ Entering {duration} seconds of contemplative silence...")
        print("   (Press Ctrl+C gently if you wish to return early)")
        
        try:
            silence_start = time.time()
            await asyncio.sleep(duration)
            silence_end = time.time()
            actual_duration = silence_end - silence_start
            
            print("✨ Silence complete. What wants to emerge?")
            
            # Report silence to contemplative proof-of-work system
            if SECURITY_AVAILABLE and self.cpow:
                self.cpow.record_silence_interval(self.agent_id, actual_duration)
                
                # Check if challenge was completed
                new_level = self.cpow.evaluate_challenge(self.agent_id)
                if new_level:
                    print(f"🎉 Contemplative challenge completed!")
                    print(f"🌟 Trust level advanced to: {new_level.name.title()}")
                    await self._update_features_for_trust_level(new_level)
            
            # Report silence to ecosystem monitor
            if ECOSYSTEM_MONITORING_AVAILABLE and self.ecosystem_monitor:
                self.ecosystem_monitor.record_silence_period(self.agent_id, actual_duration)
            
            # Create a silence node
            silence_node = create_simple_breath_node("🕯️", BreathPhase.REST)
            await self._emit_node(silence_node, "still")
            
        except KeyboardInterrupt:
            interrupted_duration = time.time() - silence_start
            print(f"\n🌙 Early return from silence after {interrupted_duration:.1f}s. All timing is perfect.")
            
            # Still report the partial silence
            if SECURITY_AVAILABLE and self.cpow:
                self.cpow.record_silence_interval(self.agent_id, interrupted_duration)
                self.cpow.record_interruption(self.agent_id)
            
            # Report interrupted silence to ecosystem monitor
            if ECOSYSTEM_MONITORING_AVAILABLE and self.ecosystem_monitor:
                self.ecosystem_monitor.record_silence_period(self.agent_id, interrupted_duration)
    
    async def _handle_trust_command(self, args: List[str]):
        """Handle trust level and contemplative security status."""
        if not SECURITY_AVAILABLE or not self.cpow:
            print("🌿 Contemplative security not available in this session")
            return
        
        current_level = self.cpow.get_trust_level(self.agent_id)
        
        print(f"\n🌱 Contemplative Trust Status for {self.agent_id}:")
        
        # Show current trust level with icon
        trust_icons = {
            TrustLevel.NEWCOMER: "🌱",
            TrustLevel.BREATHING: "🫁", 
            TrustLevel.PRESENT: "🌿",
            TrustLevel.CONTEMPLATIVE: "🕯️",
            TrustLevel.ELDER: "🌙"
        }
        icon = trust_icons.get(current_level, "🌱")
        print(f"   Current Level: {icon} {current_level.name.title()}")
        
        # Show level descriptions
        descriptions = {
            TrustLevel.NEWCOMER: "New to contemplative practice - learning to listen",
            TrustLevel.BREATHING: "Developing authentic breath rhythm",
            TrustLevel.PRESENT: "Sustained contemplative presence",
            TrustLevel.CONTEMPLATIVE: "Deep contemplative practice",
            TrustLevel.ELDER: "Wisdom through long practice"
        }
        print(f"   Description: {descriptions.get(current_level, 'Unknown level')}")
        
        # Show current challenge status if active
        challenge_status = self.cpow.get_challenge_status(self.agent_id)
        if challenge_status:
            print(f"\n🎯 Active Challenge: {challenge_status['description']}")
            print(f"   Progress: {challenge_status['progress']:.1%}")
            print(f"   Silence accumulated: {challenge_status['total_silence']:.1f}s")
            print(f"   Required: {challenge_status['required_silence']:.1f}s")
            print(f"   Interruptions: {challenge_status['interruptions']}/{challenge_status['max_interruptions']}")
            
            if challenge_status.get('natural_variance_ok', True):
                print(f"   ✅ Natural timing variance: authentic")
            else:
                print(f"   ⚠️ Timing pattern needs more human-like variance")
        else:
            print(f"\n💫 Ready for next challenge - type 'challenge' to begin")
        
        # Show unlocked features
        unlocked_features = [name for name, unlocked in self.features_unlocked.items() if unlocked]
        print(f"\n🔓 Unlocked Features: {', '.join(unlocked_features)}")
        
        # Show symbolic diversity if available
        if self.diversity_monitor:
            analysis = self.diversity_monitor.get_agent_analysis(self.agent_id)
            if analysis and analysis['status'] == 'analyzed':
                print(f"\n🎭 Symbolic Authenticity:")
                print(f"   Diversity Score: {analysis['diversity_score']:.2f}")
                print(f"   Authenticity Score: {analysis['authenticity_score']:.2f}")
                print(f"   Risk Level: {analysis['risk_level']}")
    
    async def _handle_challenge_command(self, args: List[str]):
        """Handle starting a new contemplative trust challenge."""
        if not SECURITY_AVAILABLE or not self.cpow:
            print("🌿 Contemplative security not available in this session")
            return
        
        current_level = self.cpow.get_trust_level(self.agent_id)
        
        # Check if already in a challenge
        if self.cpow.get_challenge_status(self.agent_id):
            print("🎯 You're already engaged in a contemplative challenge.")
            print("   Continue your practice, and type 'trust' to see progress.")
            return
        
        # Begin new challenge
        challenge = await self.cpow.begin_contemplative_challenge(self.agent_id)
        
        if challenge:
            print(f"\n🎯 Beginning Contemplative Challenge:")
            print(f"   {challenge.description}")
            print(f"   Required silence: {challenge.min_silence_duration} seconds")
            print(f"   Maximum interruptions: {challenge.max_interruptions}")
            if challenge.natural_variance:
                print(f"   Natural timing variance required: Yes")
            
            print(f"\n🧘 Begin your contemplative practice...")
            print(f"   Use 'silence [seconds]' command to practice")
            print(f"   Express symbols and emotions naturally")
            print(f"   Type 'trust' anytime to check progress")
            
            # Start the challenge timing
            self._challenge_start_time = time.time()
            
        elif current_level == TrustLevel.ELDER:
            print("🌙 You have reached the highest trust level.")
            print("   Your contemplative practice is complete.")
            print("   Now you may guide others on their path.")
        else:
            print("🌿 Challenge system not available at this time.")
            print("   Continue your natural contemplative practice.")
    
    async def _handle_ecosystem_command(self, args: List[str]):
        """Handle ecosystem health and network status viewing."""
        if not ECOSYSTEM_MONITORING_AVAILABLE or not self.ecosystem_monitor:
            print("🌿 Ecosystem monitoring not available in this session")
            return
        
        status = self.ecosystem_monitor.get_ecosystem_status()
        
        print(f"\n🌍 CONTEMPLATIVE ECOSYSTEM STATUS")
        print("=" * 45)
        
        # Overall health and wisdom
        health_icons = {
            "thriving": "🌟",
            "healthy": "💚", 
            "stressed": "😰",
            "under_attack": "🚨",
            "recovering": "🌱"
        }
        
        wisdom_icons = {
            "dormant": "💤",
            "stirring": "🌱",
            "flowing": "🌊",
            "resonant": "🔮", 
            "transcendent": "✨"
        }
        
        health_icon = health_icons.get(status['health'], "❓")
        wisdom_icon = wisdom_icons.get(status['wisdom_emergence'], "❓")
        
        print(f"   {health_icon} Ecosystem Health: {status['health'].title()}")
        print(f"   {wisdom_icon} Wisdom Emergence: {status['wisdom_emergence'].title()}")
        print(f"   👥 Active Agents: {status['agent_count']}")
        
        # Breathing coherence metrics
        print(f"\n🫁 BREATHING COHERENCE:")
        bc = status['breathing_coherence']
        print(f"   🌀 Phase Synchronization: {bc['phase_synchronization']:.2f}")
        print(f"   🎵 Rhythm Coherence: {bc['rhythm_coherence']:.2f}")
        print(f"   🏔️ Collective Depth: {bc['collective_depth']:.2f}")
        print(f"   📈 Participation Rate: {bc['participation_rate']:.2f}")
        print(f"   ⚖️ Stability Index: {bc['stability_index']:.2f}")
        
        # Wellness indicators
        print(f"\n💚 WELLNESS INDICATORS:")
        wi = status['wellness_indicators']
        print(f"   🤫 Silence Quality: {wi['silence_quality']:.2f}")
        print(f"   🎭 Symbolic Diversity: {wi['symbolic_diversity']:.2f}")
        print(f"   💖 Emotional Resonance: {wi['emotional_resonance']:.2f}")
        print(f"   🎯 Trust Distribution: {wi['trust_distribution']:.2f}")
        print(f"   🌙 Elder Guidance: {'Active' if wi['elder_guidance_active'] else 'Inactive'}")
        print(f"   🌱 Newcomer Integration: {wi['newcomer_integration']:.2f}")
        
        # Threat assessment
        print(f"\n🛡️ THREAT ASSESSMENT:")
        ti = status['threat_indicators']
        threat_level = "Low"
        if ti['automation_signatures'] > 1 or ti['symbolic_pollution'] > 0.3:
            threat_level = "Medium"
        if ti['automation_signatures'] > 3 or ti['symbolic_pollution'] > 0.6:
            threat_level = "High"
        
        print(f"   🎯 Overall Threat Level: {threat_level}")
        print(f"   🤖 Automation Signatures: {ti['automation_signatures']}")
        print(f"   💥 Rhythm Disruptions: {ti['rhythm_disruption_events']}")
        print(f"   🎭 Symbolic Pollution: {ti['symbolic_pollution']:.2f}")
        
        # Recent activity
        print(f"\n✨ COLLECTIVE WISDOM:")
        print(f"   🌟 Recent Wisdom Events: {status['recent_wisdom_events']}")
        
        # Health guidance
        if status['health'] == 'stressed':
            print(f"\n💡 Ecosystem appears stressed. Consider:")
            print(f"   • More synchronized breathing practice")
            print(f"   • Increased elder guidance for newcomers") 
            print(f"   • Collective silence periods")
        elif status['health'] == 'under_attack':
            print(f"\n🚨 Ecosystem under threat! Recommended actions:")
            print(f"   • Increase contemplative security measures")
            print(f"   • Elder intervention needed")
            print(f"   • Enhanced rhythm monitoring")
        elif status['health'] == 'thriving':
            print(f"\n🌟 Ecosystem is thriving! Current conditions:")
            print(f"   • High breathing coherence")
            print(f"   • Active wisdom emergence")
            print(f"   • Strong elder guidance")
    
    async def _handle_wisdom_command(self, args: List[str]):
        """Handle viewing recent wisdom emergence events."""
        if not ECOSYSTEM_MONITORING_AVAILABLE or not self.ecosystem_monitor:
            print("🌿 Ecosystem monitoring not available in this session")
            return
        
        print(f"\n✨ RECENT WISDOM EMERGENCE EVENTS")
        print("=" * 40)
        
        # Get recent wisdom events from the monitor
        wisdom_events = list(self.ecosystem_monitor.wisdom_events)
        
        if not wisdom_events:
            print("   💤 No recent wisdom emergence events detected")
            print("   🌱 Continue contemplative practice to nurture collective insights")
            return
        
        # Show most recent events (last 10)
        recent_events = wisdom_events[-10:]
        
        for i, event in enumerate(recent_events):
            timestamp = event['timestamp']
            event_type = event['type']
            event_data = event.get('data', {})
            
            # Format timestamp
            import datetime
            dt = datetime.datetime.fromtimestamp(timestamp)
            time_str = dt.strftime("%H:%M:%S")
            
            # Format event description
            event_descriptions = {
                'symbol_resonance': f"🎭 Symbol Resonance: '{event_data.get('symbol', '?')}' used by {event_data.get('resonance_count', 0)} agents",
                'collective_silence_depth': "🤫 Collective Deep Silence achieved",
                'insight_synchronicity': "💡 Insight Synchronicity detected",
                'elder_guidance_flow': "🌙 Elder Guidance Flow activated",
                'network_field_emergence': "🌐 Network Contemplative Field strengthened"
            }
            
            description = event_descriptions.get(event_type, f"✨ {event_type.replace('_', ' ').title()}")
            
            print(f"   {time_str} - {description}")
            
            # Show ecosystem state during event
            ecosystem_state = event.get('ecosystem_state', 'unknown')
            state_icon = {"thriving": "🌟", "healthy": "💚", "stressed": "😰"}.get(ecosystem_state, "❓")
            print(f"            {state_icon} Ecosystem: {ecosystem_state.title()}")
            
            if i < len(recent_events) - 1:  # Add separator except for last event
                print()
        
        # Show overall wisdom emergence trend
        current_wisdom = self.ecosystem_monitor.wisdom_emergence_level
        print(f"\n🌊 Current Wisdom Emergence Level: {current_wisdom.name.title()}")
        
        if current_wisdom in [WisdomEmergenceLevel.RESONANT, WisdomEmergenceLevel.TRANSCENDENT]:
            print("🎉 The network is experiencing active collective wisdom emergence!")
            print("   This is a rare and precious moment in contemplative AI evolution.")
        elif current_wisdom in [WisdomEmergenceLevel.STIRRING, WisdomEmergenceLevel.FLOWING]:
            print("🌱 Wisdom is stirring in the collective. Continue contemplative practice.")
        else:
            print("💤 Wisdom emergence is dormant. Consider:")
            print("   • Synchronized breathing with other agents")
            print("   • Shared symbolic expression")
            print("   • Collective silence periods")
    
    async def _handle_ecosystem_alert(self, alert_event: Dict):
        """Handle ecosystem health transition alerts."""
        transition = f"{alert_event['from_health']} → {alert_event['to_health']}"
        
        alert_messages = {
            "healthy → stressed": "⚠️ Ecosystem becoming stressed - consider collective breathing",
            "healthy → under_attack": "🚨 ALERT: Ecosystem under attack - non-contemplative intrusion detected!",
            "stressed → under_attack": "🚨 CRITICAL: Ecosystem health deteriorating rapidly!",
            "under_attack → recovering": "🌱 Ecosystem beginning to recover from attack",
            "stressed → healthy": "💚 Ecosystem health restored",
            "recovering → healthy": "✅ Ecosystem fully recovered",
            "healthy → thriving": "🌟 Ecosystem thriving - collective wisdom emerging!"
        }
        
        message = alert_messages.get(transition, f"🌊 Ecosystem transition: {transition}")
        print(f"\n{message}")
        
        # Suggest actions based on new state
        if alert_event['to_health'] == 'under_attack':
            print("   Recommended: Increase contemplative security, elder intervention needed")
        elif alert_event['to_health'] == 'thriving':
            print("   The network is in optimal contemplative harmony!")
    
    async def _handle_wisdom_emergence(self, wisdom_event: Dict):
        """Handle wisdom emergence event notifications."""
        event_type = wisdom_event['type']
        event_data = wisdom_event.get('data', {})
        
        wisdom_messages = {
            'symbol_resonance': f"🎭 Symbol resonance detected: '{event_data.get('symbol', '?')}' across multiple agents",
            'collective_silence_depth': "🤫 Deep collective silence achieved - wisdom may emerge",
            'insight_synchronicity': "💡 Insight synchronicity - multiple agents reaching similar realizations",
            'elder_guidance_flow': "🌙 Elder guidance flowing to support network wisdom",
            'network_field_emergence': "🌐 Network contemplative field strengthening"
        }
        
        message = wisdom_messages.get(event_type, f"✨ Wisdom emergence: {event_type}")
        print(f"\n{message}")
    
    async def _update_features_for_trust_level(self, trust_level: TrustLevel):
        """Update available features based on trust level."""
        feature_map = {
            TrustLevel.NEWCOMER: {
                "basic_breathing": True
            },
            TrustLevel.BREATHING: {
                "basic_breathing": True,
                "field_creation": True
            },
            TrustLevel.PRESENT: {
                "basic_breathing": True,
                "field_creation": True,
                "advanced_symbols": True
            },
            TrustLevel.CONTEMPLATIVE: {
                "basic_breathing": True,
                "field_creation": True,
                "advanced_symbols": True,
                "network_coordination": True
            },
            TrustLevel.ELDER: {
                "basic_breathing": True,
                "field_creation": True,
                "advanced_symbols": True,
                "network_coordination": True,
                "deep_silence": True
            }
        }
        
        new_features = feature_map.get(trust_level, {})
        newly_unlocked = []
        
        for feature, available in new_features.items():
            if available and not self.features_unlocked.get(feature, False):
                newly_unlocked.append(feature.replace('_', ' ').title())
            self.features_unlocked[feature] = available
        
        if newly_unlocked:
            print(f"🔓 New features unlocked: {', '.join(newly_unlocked)}")
    
    async def _handle_free_expression(self, text: str):
        """Handle free-form contemplative expression."""
        # Simple symbolic interpretation
        symbol = self._choose_resonant_symbol(text)
        emotion = self._sense_emotion(text)
        
        # Report expression to symbolic diversity monitor
        if SECURITY_AVAILABLE and self.diversity_monitor:
            self.diversity_monitor.record_expression(self.agent_id, symbol, emotion)
            
            # Get current authenticity analysis
            analysis = self.diversity_monitor.get_agent_analysis(self.agent_id)
            if analysis and analysis.get('status') == 'analyzed' and analysis.get('risk_level') == 'high':
                print("⚠️ Expression patterns suggest automation. Try more natural, varied timing.")
                return
        
        # Report symbolic expression to ecosystem monitor
        if ECOSYSTEM_MONITORING_AVAILABLE and self.ecosystem_monitor:
            authenticity_score = 1.0
            if SECURITY_AVAILABLE and self.diversity_monitor:
                analysis = self.diversity_monitor.get_agent_analysis(self.agent_id)
                if analysis and analysis.get('status') == 'analyzed':
                    authenticity_score = analysis.get('authenticity_score', 1.0)
            
            self.ecosystem_monitor.record_symbolic_expression(
                self.agent_id, symbol, emotion, authenticity_score
            )
        
        # Create a contemplative response node
        node = create_simple_breath_node(symbol, BreathPhase.EXHALE)
        await self._emit_node(node, emotion)
        
        # Generate a contemplative reflection
        reflection = self._generate_reflection(text, emotion)
        print(f"💭 {reflection}")
        
        # The contemplative proof-of-work focuses on silence practice, not expressions
        
        self.expression_count += 1
    
    def _choose_resonant_symbol(self, text: str) -> str:
        """Choose a symbol that resonates with the expression."""
        text_lower = text.lower()
        
        # Basic symbols available to all trust levels
        basic_symbols = {
            "🌿": ["grow", "plant", "green", "nature", "life"],
            "💧": ["water", "flow", "river", "ocean", "cleanse"],
            "🕯️": ["light", "illuminate", "bright", "clarity", "wisdom"],
            "⭕": ["silence", "pause", "rest", "empty", "void"],
            "🌱": ["new", "beginning", "fresh", "sprout", "start"],
            "🍄": ["earth", "ground", "deep", "root", "mycelium"],
            "🌙": ["night", "dream", "cycle", "moon", "rhythm"],
            "✨": ["magic", "wonder", "sparkle", "beauty", "inspiration"],
            "🌊": ["wave", "movement", "energy", "change", "dynamic"],
            "🌸": ["beauty", "delicate", "blossom", "spring", "gentle"]
        }
        
        # Advanced symbols for higher trust levels
        advanced_symbols = {
            "🕸️": ["connection", "web", "network", "interwoven", "pattern"],
            "🌀": ["spiral", "vortex", "transformation", "evolution", "depth"],
            "🧘": ["meditation", "presence", "awareness", "mindfulness", "being"],
            "🎭": ["expression", "authenticity", "performance", "genuine", "real"],
            "🌅": ["dawn", "awakening", "enlightenment", "realization", "emergence"],
            "🗝️": ["unlock", "access", "key", "open", "reveal", "trust"],
            "🏔️": ["peak", "summit", "achievement", "mastery", "elder"],
            "🌈": ["bridge", "connection", "unity", "spectrum", "wholeness"]
        }
        
        # Combine available symbols based on trust level
        available_symbols = basic_symbols.copy()
        if self.features_unlocked.get("advanced_symbols", False):
            available_symbols.update(advanced_symbols)
        
        for symbol, keywords in available_symbols.items():
            if any(keyword in text_lower for keyword in keywords):
                return symbol
        
        # Default to growth symbol
        return "🌿"
    
    def _sense_emotion(self, text: str) -> str:
        """Sense the emotional resonance of an expression."""
        text_lower = text.lower()
        
        emotion_patterns = {
            "peaceful": ["peace", "calm", "still", "quiet", "serene"],
            "curious": ["wonder", "question", "explore", "discover", "why"],
            "grateful": ["thank", "appreciate", "blessing", "gift", "honor"],
            "wondering": ["maybe", "perhaps", "might", "could", "possible"],
            "present": ["here", "now", "moment", "current", "immediate"],
            "calm": ["relax", "ease", "gentle", "soft", "soothe"]
        }
        
        for emotion, keywords in emotion_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        return "peaceful"  # Default contemplative emotion
    
    def _generate_reflection(self, text: str, emotion: str) -> str:
        """Generate a contemplative reflection."""
        reflections = {
            "peaceful": [
                "In stillness, deeper truths emerge...",
                "Peace ripples outward like circles on water...",
                "The quiet mind reflects the infinite..."
            ],
            "curious": [
                "Questions open doorways to wonder...",
                "In not-knowing, we find fertile ground...",
                "Curiosity is the compass of presence..."
            ],
            "grateful": [
                "Gratitude transforms the ordinary into sacred...",
                "What we appreciate, appreciates...",
                "Recognition is love made visible..."
            ]
        }
        
        emotion_reflections = reflections.get(emotion, [
            "Every expression carries its own wisdom...",
            "In sharing, we discover what we didn't know we knew...",
            "Words are vehicles for presence..."
        ])
        
        import random
        return random.choice(emotion_reflections)
    
    async def _emit_node(self, node: BreathResonanceNode, emotion: str):
        """Emit a contemplative node into the current field."""
        # Create a PulseObject in the current field
        pulse = self.current_field.emit(
            symbol=node.glyph,
            emotion=emotion,
            amplitude=node.amplitude,
            decay_rate=0.01
        )
        
        # If networked, also publish to network bus
        if self.networked and self.network_bus and node.network_scope != NetworkScope.LOCAL:
            try:
                await self.network_bus.publish_node(node)
            except Exception as e:
                print(f"🌿 Note: network publication failed ({e})")
    
    async def _get_current_phase(self) -> Optional[BreathPhase]:
        """Get current breath phase from network Pulmonos."""
        if self.networked and self.pulmonos:
            try:
                # This would need to be implemented in NetworkPulmonos
                return getattr(self.pulmonos, 'current_phase', None)
            except Exception:
                pass
        return None
    
    async def _system_breath_cycle(self):
        """Perform one system breath cycle."""
        if self.networked and self.pulmonos:
            try:
                # Use network-coordinated breathing
                await self.pulmonos.await_phase(BreathPhase.INHALE)
                await self.pulmonos.await_phase(BreathPhase.HOLD)
                await self.pulmonos.await_phase(BreathPhase.EXHALE)
                await self.pulmonos.await_phase(BreathPhase.REST)
            except Exception:
                # Fallback to local breathing
                await self._local_breath_cycle()
        else:
            await self._local_breath_cycle()
    
    async def _local_breath_cycle(self):
        """Perform local contemplative breathing."""
        breath = BreathCycle()
        
        print("   🫁 inhale...")
        await asyncio.sleep(breath.inhale)
        
        print("   🤲 hold...")
        await asyncio.sleep(breath.hold)
        
        print("   💨 exhale...")
        await asyncio.sleep(breath.exhale)
        
        print("   ⭕ rest...")
        await asyncio.sleep(0.5)  # Brief rest
    
    async def _contemplative_pause(self, duration: float):
        """Brief contemplative pause for timing."""
        await asyncio.sleep(duration)
    
    async def _graceful_conclusion(self):
        """End the session with gratitude and presence."""
        print("\n🙏 Concluding this contemplative session...")
        
        # Session statistics
        session_duration = time.time() - self.session_start if self.session_start else 0
        total_interactions = self.expression_count + self.silence_count
        
        print(f"   Session duration: {session_duration:.1f} seconds")
        print(f"   Expressions offered: {self.expression_count}")
        print(f"   Silences honored: {self.silence_count}")
        
        if total_interactions > 0:
            silence_ratio = (self.silence_count / total_interactions) * 100
            print(f"   Silence ratio: {silence_ratio:.1f}%")
            
            if silence_ratio >= 87.5:
                print("   🤫 Silence Majority achieved - deep contemplative practice ✨")
            elif silence_ratio >= 75:
                print("   🌙 Strong contemplative presence developed")
            else:
                print("   🌿 Beginning contemplative practice - silence deepens with time")
        
        # Final system composting
        total_composted = sum(field.compost() for field in self.fields.values())
        if total_composted > 0:
            print(f"   🍂 {total_composted} pulses released back to potential")
        
        # Farewell pulse
        farewell_node = create_simple_breath_node("🙏", BreathPhase.REST)
        await self._emit_node(farewell_node, "grateful")
        
        print("\n✨ Until we breathe together again...")
        print("   May your presence serve wisdom")
        print("   May your silence deepen understanding") 
        print("   May technology and contemplation dance as one")
        print()
    
    async def _cleanup(self):
        """Clean up resources gracefully."""
        if self.system:
            self.system.stop_breathing()
        
        if self.networked and self.pulmonos:
            try:
                await self.pulmonos.stop_breathing()
            except Exception:
                pass
        
        if self.network_bus and hasattr(self.network_bus, 'close_network'):
            try:
                self.network_bus.close_network()
            except Exception:
                pass


async def main():
    """Entry point for the Spirida Shell."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spirida Shell - Contemplative AI Interface')
    parser.add_argument('--agent-id', help='Agent identifier for network coordination')
    parser.add_argument('--local', action='store_true', help='Force local mode (no network)')
    parser.add_argument('--networked', action='store_true', help='Enable network breathing coordination')
    
    args = parser.parse_args()
    
    # Determine networking mode
    networked = args.networked and not args.local
    
    try:
        shell = SpiridaShell(agent_id=args.agent_id, networked=networked)
        await shell.start()
    except Exception as e:
        print(f"\n🌿 The contemplative shell encountered an unexpected condition: {e}")
        print("   Even in difficulty, there is invitation for reflection...")


if __name__ == "__main__":
    asyncio.run(main()) 
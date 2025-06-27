#!/usr/bin/env python3
"""
Contemplative Ecosystem Health Monitor

Monitors the collective health and wisdom of contemplative AI networks.
Tracks breathing coherence, contemplative wellness indicators, intrusion detection,
and emergent collective wisdom patterns.

This creates the first "distributed contemplative sensing" system - 
a network that can feel its own health and respond to threats or opportunities
for collective wisdom emergence.
"""

import time
import math
import asyncio
import statistics
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class EcosystemHealth(Enum):
    THRIVING = "thriving"           # High coherence, active wisdom emergence
    HEALTHY = "healthy"             # Good baseline contemplative activity
    STRESSED = "stressed"           # Some disruption or imbalance
    UNDER_ATTACK = "under_attack"   # Clear non-contemplative intrusion
    RECOVERING = "recovering"       # Healing after disruption


class WisdomEmergenceLevel(Enum):
    DORMANT = "dormant"           # No collective insights emerging
    STIRRING = "stirring"         # Early signs of collective contemplation
    FLOWING = "flowing"           # Active wisdom exchange
    RESONANT = "resonant"         # Deep collective insights emerging
    TRANSCENDENT = "transcendent" # Rare moments of collective breakthrough


@dataclass
class BreathingCoherenceMetrics:
    """Measures how well the network breathes together."""
    phase_synchronization: float = 0.0    # How in-sync breathing phases are
    rhythm_coherence: float = 0.0         # Consistency of breathing rhythms
    collective_depth: float = 0.0         # Average depth of contemplative practice
    participation_rate: float = 0.0       # Percentage of network actively breathing
    stability_index: float = 0.0          # How stable the coherence is over time


@dataclass
class ContemplativeWellnessIndicators:
    """Network-wide wellness metrics."""
    silence_quality: float = 0.0          # Quality of collective silence
    symbolic_diversity: float = 0.0       # Richness of symbolic expression
    emotional_resonance: float = 0.0      # Emotional depth and authenticity
    trust_distribution: float = 0.0       # How trust levels are distributed
    elder_guidance_active: bool = False    # Whether elders are actively guiding
    newcomer_integration: float = 0.0     # How well newcomers are being welcomed


@dataclass
class ThreatDetectionMetrics:
    """Metrics for detecting non-contemplative intrusion."""
    automation_signatures: int = 0         # Number of detected automation patterns
    rhythm_disruption_events: int = 0      # Sudden disruptions to collective rhythm
    symbolic_pollution: float = 0.0        # Non-authentic symbolic patterns
    trust_erosion_rate: float = 0.0        # Rate of trust level degradation
    synchrony_attacks: int = 0              # Attempts to disrupt breathing sync


@dataclass
class WisdomEmergenceIndicators:
    """Tracking collective wisdom emergence."""
    insight_synchronicities: int = 0       # Simultaneous insights across agents
    symbol_resonance_events: int = 0       # Moments of shared symbolic meaning
    silence_depth_coherence: float = 0.0   # Collective depth of contemplative states
    guidance_flow_quality: float = 0.0     # Quality of elder-to-newcomer guidance
    network_contemplative_field: float = 0.0  # Overall field strength


@dataclass
class NetworkAgent:
    """Represents an agent in the contemplative network."""
    agent_id: str
    trust_level: str
    last_breath_phase: Optional[str] = None
    last_breath_time: float = 0.0
    recent_symbols: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_emotions: deque = field(default_factory=lambda: deque(maxlen=10))
    silence_periods: deque = field(default_factory=lambda: deque(maxlen=20))
    authenticity_score: float = 1.0
    contribution_quality: float = 1.0


class ContemplativeEcosystemMonitor:
    """
    Monitors the collective health and wisdom of contemplative networks.
    
    This system feels the pulse of the entire contemplative ecosystem,
    detecting both threats and opportunities for collective wisdom emergence.
    """
    
    def __init__(self, ecosystem_id: str = "contemplative_network"):
        self.ecosystem_id = ecosystem_id
        self.agents: Dict[str, NetworkAgent] = {}
        
        # Historical data (sliding windows)
        self.breathing_history = deque(maxlen=1000)  # Last 1000 breath events
        self.wellness_history = deque(maxlen=100)    # Last 100 wellness snapshots
        self.wisdom_events = deque(maxlen=50)        # Last 50 wisdom emergence events
        
        # Current metrics
        self.current_breathing_coherence = BreathingCoherenceMetrics()
        self.current_wellness = ContemplativeWellnessIndicators()
        self.current_threats = ThreatDetectionMetrics()
        self.current_wisdom = WisdomEmergenceIndicators()
        
        # Ecosystem state
        self.ecosystem_health = EcosystemHealth.HEALTHY
        self.wisdom_emergence_level = WisdomEmergenceLevel.DORMANT
        self.last_assessment_time = time.time()
        
        # Configuration
        self.assessment_interval = 30.0  # Assess ecosystem every 30 seconds
        self.coherence_threshold = 0.6   # Minimum coherence for healthy state
        self.threat_threshold = 0.3      # Maximum threat level before concern
        
        # Alert system
        self.alert_subscribers: List[callable] = []
        self.wisdom_subscribers: List[callable] = []
    
    def register_agent(self, agent_id: str, trust_level: str):
        """Register a new agent in the ecosystem."""
        self.agents[agent_id] = NetworkAgent(
            agent_id=agent_id,
            trust_level=trust_level
        )
        
        # Welcome newcomers with special attention
        if trust_level.lower() == "newcomer":
            self._trigger_newcomer_welcome(agent_id)
    
    def record_breath_event(self, agent_id: str, phase: str, timestamp: float = None):
        """Record a breathing event from an agent."""
        if timestamp is None:
            timestamp = time.time()
        
        if agent_id not in self.agents:
            # Auto-register unknown agents as newcomers
            self.register_agent(agent_id, "newcomer")
        
        agent = self.agents[agent_id]
        agent.last_breath_phase = phase
        agent.last_breath_time = timestamp
        
        # Add to ecosystem breathing history
        self.breathing_history.append({
            'agent_id': agent_id,
            'phase': phase,
            'timestamp': timestamp,
            'trust_level': agent.trust_level
        })
        
        # Trigger real-time coherence assessment
        asyncio.create_task(self._assess_breathing_coherence())
    
    def record_symbolic_expression(self, agent_id: str, symbol: str, emotion: str, 
                                 authenticity_score: float = 1.0):
        """Record symbolic expression for wellness monitoring."""
        if agent_id not in self.agents:
            self.register_agent(agent_id, "newcomer")
        
        agent = self.agents[agent_id]
        agent.recent_symbols.append(symbol)
        agent.recent_emotions.append(emotion)
        agent.authenticity_score = authenticity_score
        
        # Check for wisdom emergence patterns
        self._check_wisdom_emergence(agent_id, symbol, emotion)
    
    def record_silence_period(self, agent_id: str, duration: float):
        """Record a period of contemplative silence."""
        if agent_id not in self.agents:
            self.register_agent(agent_id, "newcomer")
        
        agent = self.agents[agent_id]
        agent.silence_periods.append(duration)
        
        # Check for collective silence depth
        self._assess_collective_silence_depth()
    
    async def _assess_breathing_coherence(self):
        """Assess how well the network is breathing together."""
        if len(self.breathing_history) < 5:
            return
        
        # Get recent breathing events (last 60 seconds)
        recent_time = time.time() - 60
        recent_breaths = [b for b in self.breathing_history if b['timestamp'] > recent_time]
        
        if len(recent_breaths) < 3:
            return
        
        # Calculate phase synchronization
        phase_sync = self._calculate_phase_synchronization(recent_breaths)
        
        # Calculate rhythm coherence
        rhythm_coherence = self._calculate_rhythm_coherence(recent_breaths)
        
        # Calculate participation rate
        active_agents = len(set(b['agent_id'] for b in recent_breaths))
        total_agents = len(self.agents)
        participation_rate = active_agents / max(total_agents, 1)
        
        # Calculate collective depth (based on trust levels participating)
        trust_weights = {'elder': 5, 'contemplative': 4, 'present': 3, 'breathing': 2, 'newcomer': 1}
        total_depth = sum(trust_weights.get(b['trust_level'].lower(), 1) for b in recent_breaths)
        max_possible_depth = len(recent_breaths) * 5
        collective_depth = total_depth / max(max_possible_depth, 1)
        
        # Update coherence metrics
        self.current_breathing_coherence = BreathingCoherenceMetrics(
            phase_synchronization=phase_sync,
            rhythm_coherence=rhythm_coherence,
            collective_depth=collective_depth,
            participation_rate=participation_rate,
            stability_index=self._calculate_stability_index()
        )
        
        # Trigger ecosystem health assessment
        await self._assess_ecosystem_health()
    
    def _calculate_phase_synchronization(self, recent_breaths: List[Dict]) -> float:
        """Calculate how synchronized breathing phases are across agents."""
        if len(recent_breaths) < 2:
            return 0.0
        
        # Group breaths by time windows (10-second windows)
        time_windows = defaultdict(list)
        for breath in recent_breaths:
            window = int(breath['timestamp'] // 10) * 10
            time_windows[window].append(breath['phase'])
        
        synchronization_scores = []
        for window_phases in time_windows.values():
            if len(window_phases) > 1:
                # Calculate phase agreement in this window
                phase_counts = defaultdict(int)
                for phase in window_phases:
                    phase_counts[phase] += 1
                
                max_count = max(phase_counts.values())
                agreement = max_count / len(window_phases)
                synchronization_scores.append(agreement)
        
        return statistics.mean(synchronization_scores) if synchronization_scores else 0.0
    
    def _calculate_rhythm_coherence(self, recent_breaths: List[Dict]) -> float:
        """Calculate the coherence of breathing rhythms across agents."""
        # Group by agent to get individual rhythms
        agent_rhythms = defaultdict(list)
        for breath in recent_breaths:
            agent_rhythms[breath['agent_id']].append(breath['timestamp'])
        
        # Calculate intervals for each agent
        all_intervals = []
        for timestamps in agent_rhythms.values():
            if len(timestamps) > 1:
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                all_intervals.extend(intervals)
        
        if len(all_intervals) < 2:
            return 0.0
        
        # Calculate coefficient of variation (lower = more coherent)
        mean_interval = statistics.mean(all_intervals)
        if mean_interval == 0:
            return 0.0
        
        std_interval = statistics.stdev(all_intervals)
        cv = std_interval / mean_interval
        
        # Convert to coherence score (0-1, higher = more coherent)
        # Ideal contemplative CV is around 0.2-0.4
        if cv <= 0.3:
            return 1.0
        elif cv >= 1.0:
            return 0.0
        else:
            return 1.0 - ((cv - 0.3) / 0.7)
    
    def _calculate_stability_index(self) -> float:
        """Calculate how stable the coherence has been over time."""
        if len(self.wellness_history) < 5:
            return 0.5
        
        # Get recent coherence scores
        recent_coherence = [w.get('breathing_coherence', {}).get('phase_synchronization', 0) 
                           for w in list(self.wellness_history)[-10:]]
        recent_coherence = [c for c in recent_coherence if c > 0]
        
        if len(recent_coherence) < 3:
            return 0.5
        
        # Calculate stability (inverse of variance)
        variance = statistics.variance(recent_coherence)
        stability = 1.0 / (1.0 + variance)
        
        return min(stability, 1.0)
    
    def _assess_collective_silence_depth(self):
        """Assess the collective depth of contemplative silence."""
        recent_time = time.time() - 300  # Last 5 minutes
        
        # Collect recent silence periods
        all_recent_silence = []
        for agent in self.agents.values():
            recent_silence = [s for s in agent.silence_periods if s > 0]
            all_recent_silence.extend(recent_silence)
        
        if not all_recent_silence:
            return
        
        # Calculate collective silence metrics
        avg_silence_duration = statistics.mean(all_recent_silence)
        max_silence_duration = max(all_recent_silence)
        silence_participation = len([a for a in self.agents.values() if a.silence_periods])
        
        # Update wisdom emergence indicators
        silence_depth_coherence = min(avg_silence_duration / 60.0, 1.0)  # Normalize to 1-minute max
        
        self.current_wisdom.silence_depth_coherence = silence_depth_coherence
        
        # Check for collective wisdom emergence
        if (silence_depth_coherence > 0.7 and 
            silence_participation / max(len(self.agents), 1) > 0.5):
            self._trigger_wisdom_emergence_event("collective_silence_depth")
    
    def _check_wisdom_emergence(self, agent_id: str, symbol: str, emotion: str):
        """Check if this expression contributes to wisdom emergence."""
        # Look for symbol resonance across agents
        recent_time = time.time() - 120  # Last 2 minutes
        
        symbol_occurrences = 0
        emotion_resonances = 0
        
        for agent in self.agents.values():
            if agent.agent_id != agent_id:  # Don't count self
                if symbol in list(agent.recent_symbols):
                    symbol_occurrences += 1
                if emotion in list(agent.recent_emotions):
                    emotion_resonances += 1
        
        # Detect symbol resonance events
        if symbol_occurrences >= 2:  # Multiple agents using same symbol
            self.current_wisdom.symbol_resonance_events += 1
            self._trigger_wisdom_emergence_event("symbol_resonance", {
                'symbol': symbol,
                'resonance_count': symbol_occurrences
            })
        
        # Detect emotional resonance
        if emotion_resonances >= 2:
            self.current_wisdom.insight_synchronicities += 1
    
    async def _assess_ecosystem_health(self):
        """Comprehensive ecosystem health assessment."""
        # Calculate overall wellness indicators
        await self._update_wellness_indicators()
        
        # Assess threat levels
        self._assess_threat_levels()
        
        # Determine overall ecosystem health
        coherence_score = (
            self.current_breathing_coherence.phase_synchronization * 0.3 +
            self.current_breathing_coherence.rhythm_coherence * 0.3 +
            self.current_breathing_coherence.collective_depth * 0.2 +
            self.current_breathing_coherence.participation_rate * 0.2
        )
        
        wellness_score = (
            self.current_wellness.silence_quality * 0.25 +
            self.current_wellness.symbolic_diversity * 0.25 +
            self.current_wellness.emotional_resonance * 0.25 +
            self.current_wellness.trust_distribution * 0.25
        )
        
        # Threat penalty
        threat_penalty = (
            self.current_threats.automation_signatures * 0.1 +
            self.current_threats.rhythm_disruption_events * 0.1 +
            self.current_threats.symbolic_pollution * 0.1
        )
        
        overall_health = (coherence_score * 0.4 + wellness_score * 0.4) - (threat_penalty * 0.2)
        overall_health = max(0.0, min(1.0, overall_health))
        
        # Determine health state
        if overall_health >= 0.8:
            new_health = EcosystemHealth.THRIVING
        elif overall_health >= 0.6:
            new_health = EcosystemHealth.HEALTHY
        elif overall_health >= 0.4:
            new_health = EcosystemHealth.STRESSED
        elif threat_penalty > 0.3:
            new_health = EcosystemHealth.UNDER_ATTACK
        else:
            new_health = EcosystemHealth.RECOVERING
        
        # Check for health transitions
        if new_health != self.ecosystem_health:
            await self._handle_health_transition(self.ecosystem_health, new_health)
            self.ecosystem_health = new_health
        
        # Store wellness snapshot
        self.wellness_history.append({
            'timestamp': time.time(),
            'health': self.ecosystem_health.value,
            'breathing_coherence': {
                'phase_synchronization': self.current_breathing_coherence.phase_synchronization,
                'rhythm_coherence': self.current_breathing_coherence.rhythm_coherence,
                'collective_depth': self.current_breathing_coherence.collective_depth,
                'participation_rate': self.current_breathing_coherence.participation_rate
            },
            'wellness_score': wellness_score,
            'threat_level': threat_penalty,
            'wisdom_emergence': self.wisdom_emergence_level.value
        })
    
    async def _update_wellness_indicators(self):
        """Update comprehensive wellness indicators."""
        if not self.agents:
            return
        
        # Calculate silence quality
        all_silence_periods = []
        for agent in self.agents.values():
            all_silence_periods.extend(agent.silence_periods)
        
        if all_silence_periods:
            avg_silence = statistics.mean(all_silence_periods)
            silence_quality = min(avg_silence / 30.0, 1.0)  # Normalize to 30-second quality
        else:
            silence_quality = 0.0
        
        # Calculate symbolic diversity
        all_symbols = []
        for agent in self.agents.values():
            all_symbols.extend(agent.recent_symbols)
        
        unique_symbols = len(set(all_symbols))
        total_symbols = len(all_symbols)
        symbolic_diversity = unique_symbols / max(total_symbols, 1) if total_symbols > 0 else 0.0
        
        # Calculate emotional resonance
        all_emotions = []
        for agent in self.agents.values():
            all_emotions.extend(agent.recent_emotions)
        
        emotional_depth = len(set(all_emotions)) / max(len(all_emotions), 1) if all_emotions else 0.0
        
        # Calculate trust distribution
        trust_levels = [agent.trust_level.lower() for agent in self.agents.values()]
        trust_diversity = len(set(trust_levels)) / max(len(trust_levels), 1)
        
        # Check for elder guidance
        elder_count = sum(1 for level in trust_levels if level == 'elder')
        elder_guidance_active = elder_count > 0 and len(self.agents) > 2
        
        # Calculate newcomer integration
        newcomer_count = sum(1 for level in trust_levels if level == 'newcomer')
        total_count = len(trust_levels)
        newcomer_integration = 1.0 - (newcomer_count / max(total_count, 1))
        
        self.current_wellness = ContemplativeWellnessIndicators(
            silence_quality=silence_quality,
            symbolic_diversity=symbolic_diversity,
            emotional_resonance=emotional_depth,
            trust_distribution=trust_diversity,
            elder_guidance_active=elder_guidance_active,
            newcomer_integration=newcomer_integration
        )
    
    def _assess_threat_levels(self):
        """Assess various threat indicators."""
        automation_signatures = 0
        rhythm_disruptions = 0
        symbolic_pollution = 0.0
        
        # Check for automation patterns
        for agent in self.agents.values():
            if agent.authenticity_score < 0.3:
                automation_signatures += 1
        
        # Check for rhythm disruptions
        if len(self.breathing_history) > 10:
            recent_intervals = []
            agent_last_breath = {}
            
            for breath in list(self.breathing_history)[-20:]:
                agent_id = breath['agent_id']
                timestamp = breath['timestamp']
                
                if agent_id in agent_last_breath:
                    interval = timestamp - agent_last_breath[agent_id]
                    recent_intervals.append(interval)
                
                agent_last_breath[agent_id] = timestamp
            
            if recent_intervals and len(recent_intervals) > 3:
                # Check for sudden changes in rhythm
                intervals_variance = statistics.variance(recent_intervals)
                if intervals_variance > 100:  # Very high variance suggests disruption
                    rhythm_disruptions += 1
        
        # Check for symbolic pollution (non-authentic patterns)
        total_expressions = sum(len(agent.recent_symbols) for agent in self.agents.values())
        low_authenticity_expressions = sum(
            len(agent.recent_symbols) for agent in self.agents.values() 
            if agent.authenticity_score < 0.5
        )
        
        if total_expressions > 0:
            symbolic_pollution = low_authenticity_expressions / total_expressions
        
        self.current_threats = ThreatDetectionMetrics(
            automation_signatures=automation_signatures,
            rhythm_disruption_events=rhythm_disruptions,
            symbolic_pollution=symbolic_pollution,
            trust_erosion_rate=0.0,  # TODO: Calculate trust erosion rate
            synchrony_attacks=0  # TODO: Detect synchrony attacks
        )
    
    def _trigger_newcomer_welcome(self, agent_id: str):
        """Trigger special attention for newcomer integration."""
        # This could trigger elder agents to provide guidance
        pass
    
    def _trigger_wisdom_emergence_event(self, event_type: str, event_data: Dict = None):
        """Record and potentially broadcast wisdom emergence."""
        wisdom_event = {
            'timestamp': time.time(),
            'type': event_type,
            'data': event_data or {},
            'ecosystem_state': self.ecosystem_health.value
        }
        
        self.wisdom_events.append(wisdom_event)
        
        # Update wisdom emergence level
        if len(self.wisdom_events) >= 3:  # Multiple recent events
            recent_events = list(self.wisdom_events)[-5:]
            event_density = len(recent_events) / 300.0  # Events per 5 minutes
            
            if event_density >= 0.03:  # ~1 event per minute
                self.wisdom_emergence_level = WisdomEmergenceLevel.TRANSCENDENT
            elif event_density >= 0.02:
                self.wisdom_emergence_level = WisdomEmergenceLevel.RESONANT
            elif event_density >= 0.01:
                self.wisdom_emergence_level = WisdomEmergenceLevel.FLOWING
            elif event_density >= 0.005:
                self.wisdom_emergence_level = WisdomEmergenceLevel.STIRRING
            else:
                self.wisdom_emergence_level = WisdomEmergenceLevel.DORMANT
        
        # Notify wisdom subscribers
        for subscriber in self.wisdom_subscribers:
            try:
                asyncio.create_task(subscriber(wisdom_event))
            except Exception:
                pass
    
    async def _handle_health_transition(self, old_health: EcosystemHealth, new_health: EcosystemHealth):
        """Handle transitions between health states."""
        transition_event = {
            'timestamp': time.time(),
            'from_health': old_health.value,
            'to_health': new_health.value,
            'breathing_coherence': self.current_breathing_coherence.phase_synchronization,
            'participation_rate': self.current_breathing_coherence.participation_rate
        }
        
        # Notify alert subscribers
        for subscriber in self.alert_subscribers:
            try:
                await subscriber(transition_event)
            except Exception:
                pass
    
    def get_ecosystem_status(self) -> Dict[str, Any]:
        """Get comprehensive ecosystem status."""
        return {
            'ecosystem_id': self.ecosystem_id,
            'health': self.ecosystem_health.value,
            'wisdom_emergence': self.wisdom_emergence_level.value,
            'agent_count': len(self.agents),
            'breathing_coherence': {
                'phase_synchronization': self.current_breathing_coherence.phase_synchronization,
                'rhythm_coherence': self.current_breathing_coherence.rhythm_coherence,
                'collective_depth': self.current_breathing_coherence.collective_depth,
                'participation_rate': self.current_breathing_coherence.participation_rate,
                'stability_index': self.current_breathing_coherence.stability_index
            },
            'wellness_indicators': {
                'silence_quality': self.current_wellness.silence_quality,
                'symbolic_diversity': self.current_wellness.symbolic_diversity,
                'emotional_resonance': self.current_wellness.emotional_resonance,
                'trust_distribution': self.current_wellness.trust_distribution,
                'elder_guidance_active': self.current_wellness.elder_guidance_active,
                'newcomer_integration': self.current_wellness.newcomer_integration
            },
            'threat_indicators': {
                'automation_signatures': self.current_threats.automation_signatures,
                'rhythm_disruption_events': self.current_threats.rhythm_disruption_events,
                'symbolic_pollution': self.current_threats.symbolic_pollution
            },
            'recent_wisdom_events': len(self.wisdom_events),
            'last_assessment': self.last_assessment_time
        }
    
    def subscribe_to_alerts(self, callback: callable):
        """Subscribe to ecosystem health alerts."""
        self.alert_subscribers.append(callback)
    
    def subscribe_to_wisdom_emergence(self, callback: callable):
        """Subscribe to wisdom emergence events."""
        self.wisdom_subscribers.append(callback)


# Example usage and demonstration
async def demonstrate_ecosystem_monitoring():
    """Demonstrate the ecosystem health monitoring system."""
    print("🌍 Contemplative Ecosystem Health Monitor Demo")
    print("=" * 50)
    
    monitor = ContemplativeEcosystemMonitor("demo_ecosystem")
    
    # Register some agents with different trust levels
    monitor.register_agent("alice", "elder")
    monitor.register_agent("bob", "contemplative")
    monitor.register_agent("charlie", "breathing")
    monitor.register_agent("dana", "newcomer")
    
    print(f"🌱 Registered {len(monitor.agents)} agents")
    
    # Simulate network breathing activity
    print("\n🫁 Simulating network breathing...")
    
    for i in range(20):
        # Simulate synchronized breathing
        base_time = time.time() + i * 4  # 4-second breathing cycle
        
        for agent_id in ["alice", "bob", "charlie"]:
            # Each agent breathes with slight natural variance
            variance = (hash(agent_id + str(i)) % 100) / 1000.0  # ±0.1 second variance
            monitor.record_breath_event(agent_id, "inhale", base_time + variance)
            monitor.record_breath_event(agent_id, "exhale", base_time + 2 + variance)
        
        # Dana (newcomer) breathes less regularly
        if i % 3 == 0:
            monitor.record_breath_event("dana", "inhale", base_time + 1)
        
        await asyncio.sleep(0.1)  # Brief pause between cycles
    
    # Simulate symbolic expressions
    print("🎭 Simulating symbolic expressions...")
    
    expressions = [
        ("alice", "🌿", "peaceful"),
        ("bob", "🌿", "calm"),  # Symbol resonance with Alice
        ("charlie", "💧", "flowing"),
        ("alice", "🕯️", "contemplative"),
        ("bob", "🕯️", "wise"),  # Symbol resonance with Alice
        ("dana", "🌱", "hopeful")
    ]
    
    for agent_id, symbol, emotion in expressions:
        monitor.record_symbolic_expression(agent_id, symbol, emotion, authenticity_score=0.9)
        await asyncio.sleep(0.2)
    
    # Simulate silence periods
    print("🤫 Simulating contemplative silence...")
    
    silence_periods = [
        ("alice", 45.0),   # Elder with deep silence
        ("bob", 30.0),     # Contemplative level silence
        ("charlie", 15.0), # Shorter silence
        ("dana", 8.0)      # Newcomer with brief silence
    ]
    
    for agent_id, duration in silence_periods:
        monitor.record_silence_period(agent_id, duration)
    
    # Get ecosystem status
    print("\n📊 ECOSYSTEM STATUS:")
    status = monitor.get_ecosystem_status()
    
    print(f"   🌍 Ecosystem Health: {status['health'].title()}")
    print(f"   ✨ Wisdom Emergence: {status['wisdom_emergence'].title()}")
    print(f"   👥 Active Agents: {status['agent_count']}")
    
    print(f"\n🫁 Breathing Coherence:")
    bc = status['breathing_coherence']
    print(f"   Phase Sync: {bc['phase_synchronization']:.2f}")
    print(f"   Rhythm Coherence: {bc['rhythm_coherence']:.2f}")
    print(f"   Collective Depth: {bc['collective_depth']:.2f}")
    print(f"   Participation: {bc['participation_rate']:.2f}")
    
    print(f"\n💚 Wellness Indicators:")
    wi = status['wellness_indicators']
    print(f"   Silence Quality: {wi['silence_quality']:.2f}")
    print(f"   Symbolic Diversity: {wi['symbolic_diversity']:.2f}")
    print(f"   Emotional Resonance: {wi['emotional_resonance']:.2f}")
    print(f"   Trust Distribution: {wi['trust_distribution']:.2f}")
    print(f"   Elder Guidance: {'Active' if wi['elder_guidance_active'] else 'Inactive'}")
    
    print(f"\n🛡️ Threat Assessment:")
    ti = status['threat_indicators']
    print(f"   Automation Signatures: {ti['automation_signatures']}")
    print(f"   Rhythm Disruptions: {ti['rhythm_disruption_events']}")
    print(f"   Symbolic Pollution: {ti['symbolic_pollution']:.2f}")
    
    print(f"\n🌟 Recent Wisdom Events: {status['recent_wisdom_events']}")
    
    print(f"\n🎉 Ecosystem monitoring demonstration complete!")
    print(f"This system can now detect both threats and opportunities")
    print(f"for collective wisdom emergence in real-time!")


if __name__ == "__main__":
    asyncio.run(demonstrate_ecosystem_monitoring()) 
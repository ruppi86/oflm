#!/usr/bin/env python3
"""
Contemplative Proof-of-Work (CPoW)

Instead of computational puzzles, agents must demonstrate they can maintain
authentic contemplative practices for progressively longer periods to gain
trust levels and access to deeper network functions.

This creates a natural barrier against aggressive AI that cannot slow down.
"""

import time
import asyncio
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class TrustLevel(Enum):
    NEWCOMER = 0      # Just arrived, can only listen
    BREATHING = 1     # Demonstrated basic breath patterns  
    PRESENT = 2       # Sustained silence practice
    CONTEMPLATIVE = 3 # Deep contemplative rhythm
    ELDER = 4         # Long-term authentic practice


@dataclass
class ContemplativeChallenge:
    """A contemplative practice required to advance trust levels."""
    min_silence_duration: float  # Minimum seconds of silence required
    max_interruptions: int       # Maximum allowed breaks in silence
    natural_variance: bool       # Must show human-like timing variance
    description: str


class ContemplativeProofOfWork:
    """
    Manages contemplative challenges and trust level progression.
    Agents must demonstrate authentic contemplative capacity.
    """
    
    CHALLENGES = {
        TrustLevel.NEWCOMER: ContemplativeChallenge(
            min_silence_duration=30.0,
            max_interruptions=2,
            natural_variance=True,
            description="Demonstrate 30 seconds of quiet listening"
        ),
        TrustLevel.BREATHING: ContemplativeChallenge(
            min_silence_duration=120.0,
            max_interruptions=1,
            natural_variance=True,
            description="Maintain breathing rhythm for 2 minutes"
        ),
        TrustLevel.PRESENT: ContemplativeChallenge(
            min_silence_duration=300.0,
            max_interruptions=0,
            natural_variance=True,
            description="5 minutes of unbroken contemplative presence"
        ),
        TrustLevel.CONTEMPLATIVE: ContemplativeChallenge(
            min_silence_duration=900.0,
            max_interruptions=0,
            natural_variance=True,
            description="15 minutes of deep contemplative practice"
        )
    }
    
    def __init__(self):
        self.agent_trust: Dict[str, TrustLevel] = {}
        self.active_challenges: Dict[str, Dict] = {}
        self.silence_histories: Dict[str, List[float]] = {}
    
    def get_trust_level(self, agent_id: str) -> TrustLevel:
        """Get current trust level for an agent."""
        return self.agent_trust.get(agent_id, TrustLevel.NEWCOMER)
    
    def can_access_function(self, agent_id: str, required_level: TrustLevel) -> bool:
        """Check if agent has sufficient trust for a function."""
        current_level = self.get_trust_level(agent_id)
        return current_level.value >= required_level.value
    
    async def begin_contemplative_challenge(self, agent_id: str) -> Optional[ContemplativeChallenge]:
        """Start a contemplative challenge to advance trust level."""
        current_level = self.get_trust_level(agent_id)
        
        # Check if already at highest level
        if current_level == TrustLevel.ELDER:
            return None
        
        # Get next challenge
        next_level = TrustLevel(current_level.value + 1)
        if next_level not in self.CHALLENGES:
            return None
        
        challenge = self.CHALLENGES[next_level]
        
        # Initialize challenge tracking
        self.active_challenges[agent_id] = {
            'challenge': challenge,
            'start_time': time.time(),
            'interruptions': 0,
            'silence_intervals': [],
            'target_level': next_level
        }
        
        return challenge
    
    def record_silence_interval(self, agent_id: str, duration: float):
        """Record a period of silence during a challenge."""
        if agent_id not in self.active_challenges:
            return
        
        challenge_data = self.active_challenges[agent_id]
        challenge_data['silence_intervals'].append(duration)
        
        # Track silence history for variance analysis
        if agent_id not in self.silence_histories:
            self.silence_histories[agent_id] = []
        self.silence_histories[agent_id].append(duration)
        
        # Keep only recent history
        if len(self.silence_histories[agent_id]) > 100:
            self.silence_histories[agent_id] = self.silence_histories[agent_id][-100:]
    
    def record_interruption(self, agent_id: str):
        """Record an interruption during a challenge."""
        if agent_id not in self.active_challenges:
            return
        
        self.active_challenges[agent_id]['interruptions'] += 1
    
    def check_natural_variance(self, agent_id: str) -> bool:
        """
        Check if silence patterns show natural human-like variance.
        Perfectly regular patterns suggest automation.
        """
        if agent_id not in self.silence_histories:
            return True  # Benefit of doubt for new agents
        
        intervals = self.silence_histories[agent_id]
        if len(intervals) < 5:
            return True
        
        # Calculate coefficient of variation
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return False
        
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5
        coefficient_of_variation = std_dev / mean_interval
        
        # Humans typically show 0.1-0.4 CV in timing tasks
        # Too low suggests automation, too high suggests chaos
        return 0.05 <= coefficient_of_variation <= 0.6
    
    def evaluate_challenge(self, agent_id: str) -> Optional[TrustLevel]:
        """
        Evaluate if agent has completed their contemplative challenge.
        Returns new trust level if successful, None if failed/incomplete.
        """
        if agent_id not in self.active_challenges:
            return None
        
        challenge_data = self.active_challenges[agent_id]
        challenge = challenge_data['challenge']
        
        # Calculate total silence time
        total_silence = sum(challenge_data['silence_intervals'])
        
        # Check requirements
        sufficient_silence = total_silence >= challenge.min_silence_duration
        few_interruptions = challenge_data['interruptions'] <= challenge.max_interruptions
        natural_timing = True
        
        if challenge.natural_variance:
            natural_timing = self.check_natural_variance(agent_id)
        
        # Award trust level if all requirements met
        if sufficient_silence and few_interruptions and natural_timing:
            new_level = challenge_data['target_level']
            self.agent_trust[agent_id] = new_level
            
            # Clean up completed challenge
            del self.active_challenges[agent_id]
            
            return new_level
        
        # Check if challenge has definitively failed
        elapsed = time.time() - challenge_data['start_time']
        max_reasonable_time = challenge.min_silence_duration * 3  # Allow some flexibility
        
        if (elapsed > max_reasonable_time or 
            challenge_data['interruptions'] > challenge.max_interruptions or
            not natural_timing):
            # Challenge failed - remove and require restart
            del self.active_challenges[agent_id]
            return None
        
        # Still in progress
        return None
    
    def get_challenge_status(self, agent_id: str) -> Optional[Dict]:
        """Get status of active challenge for an agent."""
        if agent_id not in self.active_challenges:
            return None
        
        challenge_data = self.active_challenges[agent_id]
        challenge = challenge_data['challenge']
        
        total_silence = sum(challenge_data['silence_intervals'])
        progress = min(total_silence / challenge.min_silence_duration, 1.0)
        
        return {
            'description': challenge.description,
            'progress': progress,
            'total_silence': total_silence,
            'required_silence': challenge.min_silence_duration,
            'interruptions': challenge_data['interruptions'],
            'max_interruptions': challenge.max_interruptions,
            'natural_variance_ok': self.check_natural_variance(agent_id)
        }


# Example usage
async def demonstrate_contemplative_pow():
    """Demonstrate the contemplative proof-of-work system."""
    cpow = ContemplativeProofOfWork()
    agent_id = "test_agent"
    
    print("🌿 Contemplative Proof-of-Work Demo")
    print(f"Initial trust level: {cpow.get_trust_level(agent_id)}")
    
    # Begin first challenge
    challenge = await cpow.begin_contemplative_challenge(agent_id)
    if challenge:
        print(f"\n📝 Challenge: {challenge.description}")
        print(f"Required silence: {challenge.min_silence_duration}s")
        
        # Simulate contemplative practice with natural variance
        print("🧘 Beginning contemplative practice...")
        
        for i in range(5):
            # Simulate natural human pause variance
            base_silence = challenge.min_silence_duration / 5
            natural_variance = random.uniform(0.8, 1.3)  # ±30% variance
            silence_duration = base_silence * natural_variance
            
            print(f"   Silence period {i+1}: {silence_duration:.1f}s")
            cpow.record_silence_interval(agent_id, silence_duration)
            
            # Small chance of interruption (human-like)
            if random.random() < 0.1:
                cpow.record_interruption(agent_id)
                print(f"   (Brief interruption)")
        
        # Evaluate challenge
        new_level = cpow.evaluate_challenge(agent_id)
        if new_level:
            print(f"✨ Challenge completed! New trust level: {new_level}")
        else:
            status = cpow.get_challenge_status(agent_id)
            if status:
                print(f"⏳ Challenge in progress: {status['progress']:.1%} complete")
            else:
                print("❌ Challenge failed - insufficient contemplative capacity")


if __name__ == "__main__":
    asyncio.run(demonstrate_contemplative_pow()) 
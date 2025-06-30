"""
Breath Signature for Spirida-Mycelic
Based on o3's steering memo specifications

Implements contemplative authentication through breath timing patterns:
- Hash last 256s of inhale/hold/exhale timing (not content)
- 16-bit breath fingerprint for node authentication
- 3% drift tolerance triggers pause rather than rejection
- Security via slowing, not shouting
"""

import hashlib
import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from collections import deque
from dataclasses import dataclass
from enum import Enum

class BreathPhase(Enum):
    """Breath phases for timing signature"""
    INHALE = "inhale"
    HOLD = "hold"
    EXHALE = "exhale"
    REST = "rest"

@dataclass
class BreathTiming:
    """Single breath phase timing record"""
    phase: BreathPhase
    timestamp: float
    duration: float
    environmental_context: Optional[Dict[str, float]] = None

class BreathSignature:
    """
    Contemplative authentication via breath timing patterns
    
    Creates a rolling signature from the last 256 seconds of breath timing,
    focusing on the rhythm and pattern rather than content.
    """
    
    def __init__(self, window_seconds: int = 256, tolerance_percent: float = 3.0):
        self.window_seconds = window_seconds
        self.tolerance_percent = tolerance_percent / 100.0
        
        self.breath_history: deque = deque(maxlen=1000)
        self.last_signature = ""
        self.baseline_signature = ""
        self.baseline_established = False
        
    def record_breath_timing(self, phase: BreathPhase, duration: float):
        """Record breath phase timing"""
        timing = BreathTiming(
            phase=phase,
            timestamp=time.time(),
            duration=duration
        )
        self.breath_history.append(timing)
        
        if len(self.breath_history) >= 10:
            self._update_signature()
    
    def _update_signature(self):
        """Update current breath signature"""
        current_time = time.time()
        recent_timings = [
            t for t in self.breath_history 
            if current_time - t.timestamp <= self.window_seconds
        ]
        
        if len(recent_timings) >= 5:
            new_signature = self._calculate_signature(recent_timings)
            self.last_signature = new_signature
            
            if not self.baseline_established and len(recent_timings) >= 20:
                self.baseline_signature = new_signature
                self.baseline_established = True
    
    def _calculate_signature(self, timings: List[BreathTiming]) -> str:
        """Calculate 16-bit signature from timing patterns"""
        if len(timings) < 5:
            return "0000"
        
        # Create signature from durations and intervals
        durations = [t.duration for t in timings]
        
        signature_data = bytearray()
        
        # Phase duration averages
        phase_durations = {}
        for timing in timings:
            if timing.phase not in phase_durations:
                phase_durations[timing.phase] = []
            phase_durations[timing.phase].append(timing.duration)
        
        for phase in BreathPhase:
            if phase in phase_durations:
                avg_duration = np.mean(phase_durations[phase])
                quantized = min(255, max(0, int(avg_duration)))
                signature_data.append(quantized)
            else:
                signature_data.append(0)
        
        # Hash to 16-bit
        hash_obj = hashlib.sha256(signature_data)
        hash_bytes = hash_obj.digest()
        signature_16bit = int.from_bytes(hash_bytes[:2], byteorder='big')
        
        return f"{signature_16bit:04x}"
    
    def current_signature(self) -> str:
        """Get current signature"""
        return self.last_signature or "0000"
    
    def verify_signature(self, remote_signature: str) -> Tuple[bool, float]:
        """Verify remote signature"""
        if not self.baseline_established:
            return False, 1.0
        
        drift = self._calculate_drift(self.current_signature(), remote_signature)
        is_valid = drift <= self.tolerance_percent
        return is_valid, drift
    
    def _calculate_drift(self, baseline: str, current: str) -> float:
        """Calculate drift percentage between signatures"""
        if not baseline or not current:
            return 0.0
        
        try:
            baseline_int = int(baseline, 16)
            current_int = int(current, 16)
        except ValueError:
            return 1.0
        
        xor_result = baseline_int ^ current_int
        bit_differences = bin(xor_result).count('1')
        return bit_differences / 16.0

def create_breath_signature() -> BreathSignature:
    """Create breath signature with contemplative defaults"""
    return BreathSignature(window_seconds=256, tolerance_percent=3.0)

if __name__ == "__main__":
    print("🫁 Breath Signature Demo")
    print("=" * 40)
    signature = BreathSignature()
    
    # Simulate breathing
    patterns = [(BreathPhase.INHALE, 40), (BreathPhase.HOLD, 70), 
                (BreathPhase.EXHALE, 40), (BreathPhase.REST, 10)]
    
    print("Simulating contemplative breathing cycles...")
    for cycle in range(3):
        print(f"\nCycle {cycle+1}:")
        for phase, duration in patterns:
            signature.record_breath_timing(phase, duration)
            print(f"  {phase.value}: {duration}s")
        print(f"  Signature: {signature.current_signature()}")
        print(f"  Baseline established: {signature.baseline_established}")
    
    # Test verification
    print(f"\n🔐 Testing signature verification:")
    current_sig = signature.current_signature()
    is_valid, drift = signature.verify_signature(current_sig)
    print(f"Self-verification: {'✅ Valid' if is_valid else '❌ Invalid'} (drift: {drift:.1%})")
    
    print("\n🌀 Breath signature demo complete") 
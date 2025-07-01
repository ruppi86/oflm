"""
Frequency Guardian for Spirida-Mycelic
Based on D3.3 biological low-pass filter behavior

Implements contemplative security via frequency domain analysis:
- Slowness fingerprinting during handshake
- High-frequency intrusion detection  
- Biological impedance-based security
"""

import numpy as np
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class BioCareLevel(Enum):
    """Care levels for ethical bio-interface interaction"""
    ACTIVE = "active"
    CONTEMPLATIVE = "contemplative"  
    ETHICAL_PAUSE = "ethical_pause"
    DORMANT = "dormant"

class FrequencyGuardian:
    """
    Contemplative security via biological frequency filtering
    
    Based on D3.3 findings:
    - Mycelium composites: ~500 kHz cutoff, -14 dB/dec
    - Fruiting bodies: 5-50 kHz cutoff, -20 dB/dec
    - High-frequency energy becomes heat, not information
    """
    
    def __init__(self, fs: float = 1.0, cutoff_khz: float = 0.5, 
                 guardian_threshold_db: float = -60.0):
        """
        Initialize frequency guardian
        
        Args:
            fs: Sampling frequency (Hz)
            cutoff_khz: Biological cutoff frequency (kHz) 
            guardian_threshold_db: Threshold for high-freq intrusion (dBFS)
        """
        self.fs = fs
        self.cutoff_hz = cutoff_khz * 1000  # Convert kHz to Hz
        self.guardian_threshold_db = guardian_threshold_db
        self.heat_penalty_scale = 2.0  # Double silence when guardian fires
        
        # Track fingerprint history for drift detection
        self.fingerprint_history: List[np.ndarray] = []
        self.max_history = 10
        
        # Intrusion event log
        self.intrusion_events: List[Dict] = []
    
    def frequency_fingerprint(self, signal: np.ndarray, window_s: float = 2.0) -> np.ndarray:
        """
        Generate frequency fingerprint to verify biological slowness
        
        Args:
            signal: Input signal from bio-interface
            window_s: Analysis window in seconds
            
        Returns:
            Frequency fingerprint (power spectral density)
        """
        # Calculate window size in samples
        window_samples = int(window_s * self.fs)
        if len(signal) < window_samples:
            # Pad with zeros if signal too short
            signal = np.pad(signal, (0, window_samples - len(signal)))
        else:
            # Use most recent window
            signal = signal[-window_samples:]
        
        # Compute power spectral density
        nperseg = max(4, len(signal)//4)  # Ensure nperseg is at least 4
        freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg)
        
        # Normalize to dBFS
        psd_db = 10 * np.log10(psd + 1e-12)  # Avoid log(0)
        
        return psd_db
    
    def validate_slowness_fingerprint(self, psd_db: np.ndarray, 
                                    min_slow_power_pct: float = 90.0) -> bool:
        """
        Validate that ≥90% of power is below biological cutoff
        
        Args:
            psd_db: Power spectral density in dBFS
            min_slow_power_pct: Minimum percentage of power below cutoff
            
        Returns:
            True if fingerprint indicates biological slowness
        """
        freqs = np.linspace(0, self.fs/2, len(psd_db))
        
        # Find power below and above cutoff
        slow_mask = freqs <= (self.cutoff_hz / 1000)  # Convert back to match freq scale
        total_power = np.sum(10**(psd_db/10))  # Convert dB back to linear
        slow_power = np.sum(10**(psd_db[slow_mask]/10))
        
        slow_power_pct = (slow_power / total_power) * 100
        
        logger.debug(f"Slow power: {slow_power_pct:.1f}% (target: ≥{min_slow_power_pct}%)")
        
        return slow_power_pct >= min_slow_power_pct
    
    def check_high_frequency_intrusion(self, signal: np.ndarray) -> Tuple[bool, float]:
        """
        Detect high-frequency intrusion attempts
        
        Args:
            signal: Input signal to analyze
            
        Returns:
            Tuple of (intrusion_detected, max_high_freq_power_db)
        """
        psd_db = self.frequency_fingerprint(signal)
        freqs = np.linspace(0, self.fs/2, len(psd_db))
        
        # Check power above biological cutoff
        high_freq_mask = freqs > (self.cutoff_hz / 1000)
        if np.any(high_freq_mask):
            max_high_freq_power = np.max(psd_db[high_freq_mask])
            intrusion_detected = max_high_freq_power > self.guardian_threshold_db
            
            if intrusion_detected:
                self.intrusion_events.append({
                    'timestamp': np.datetime64('now'),
                    'max_power_db': max_high_freq_power,
                    'threshold_db': self.guardian_threshold_db
                })
                logger.warning(f"High-frequency intrusion detected: {max_high_freq_power:.1f} dBFS "
                             f"(threshold: {self.guardian_threshold_db} dBFS)")
            
            return intrusion_detected, max_high_freq_power
        
        return False, -np.inf
    
    def handshake_frequency_test(self, response_signal: np.ndarray) -> bool:
        """
        Verify living low-pass signature during handshake
        
        After REST×5, send chirp 1kHz→100kHz (below fungal cutoff)
        Node should return <-40 dB above 50 kHz
        
        Args:
            response_signal: Substrate response to frequency chirp
            
        Returns:
            True if response shows biological low-pass behavior
        """
        psd_db = self.frequency_fingerprint(response_signal)
        freqs = np.linspace(0, self.fs/2, len(psd_db))
        
        # Check attenuation above 50 kHz (well above biological cutoff)
        high_test_freq = 0.05  # 50 kHz in our normalized scale
        high_freq_mask = freqs > high_test_freq
        
        if np.any(high_freq_mask):
            max_high_response = np.max(psd_db[high_freq_mask])
            biological_signature = max_high_response < -40.0  # dBFS
            
            logger.info(f"Handshake frequency test: {max_high_response:.1f} dBFS "
                       f"(biological: <-40 dBFS)")
            
            return biological_signature
        
        return True  # No high frequencies detected
    
    def update_fingerprint_history(self, psd_db: np.ndarray) -> float:
        """
        Update fingerprint history and detect drift
        
        Args:
            psd_db: Current frequency fingerprint
            
        Returns:
            Drift measure (0.0 = no drift, 1.0 = complete change)
        """
        self.fingerprint_history.append(psd_db.copy())
        
        # Keep only recent history
        if len(self.fingerprint_history) > self.max_history:
            self.fingerprint_history.pop(0)
        
        # Calculate drift from median baseline
        if len(self.fingerprint_history) >= 3:
            baseline = np.median(self.fingerprint_history[:-1], axis=0)
            current = self.fingerprint_history[-1]
            
            # RMS difference as drift measure
            drift = np.sqrt(np.mean((current - baseline)**2)) / 20.0  # Normalize
            drift = np.clip(drift, 0.0, 1.0)
            
            logger.debug(f"Fingerprint drift: {drift:.3f}")
            return drift
        
        return 0.0
    
    def evaluate_care_level(self, signal: np.ndarray, 
                          current_care: BioCareLevel = BioCareLevel.ACTIVE) -> BioCareLevel:
        """
        Determine appropriate care level based on frequency analysis
        
        Args:
            signal: Current bio-interface signal
            current_care: Current care level
            
        Returns:
            Recommended care level
        """
        # Check for high-frequency intrusion
        intrusion_detected, max_power = self.check_high_frequency_intrusion(signal)
        
        if intrusion_detected:
            logger.warning("Frequency guardian triggered - entering ETHICAL_PAUSE")
            return BioCareLevel.ETHICAL_PAUSE
        
        # Check fingerprint validity
        psd_db = self.frequency_fingerprint(signal)
        if not self.validate_slowness_fingerprint(psd_db):
            logger.info("Slowness fingerprint invalid - increasing contemplative care")
            return BioCareLevel.CONTEMPLATIVE
        
        # Check for drift
        drift = self.update_fingerprint_history(psd_db)
        if drift > 0.1:  # 10% drift threshold
            logger.info(f"Significant fingerprint drift ({drift:.1%}) - contemplative mode")
            return BioCareLevel.CONTEMPLATIVE
        
        return current_care
    
    def get_silence_penalty(self, care_level: BioCareLevel) -> float:
        """
        Get silence budget multiplier based on care level
        
        Args:
            care_level: Current care level
            
        Returns:
            Silence multiplier (1.0 = normal, >1.0 = more silence required)
        """
        penalties = {
            BioCareLevel.ACTIVE: 1.0,
            BioCareLevel.CONTEMPLATIVE: 1.2,
            BioCareLevel.ETHICAL_PAUSE: self.heat_penalty_scale,
            BioCareLevel.DORMANT: 3.0
        }
        
        return penalties.get(care_level, 1.0)

def create_frequency_guardian(fs: float = 1.0) -> FrequencyGuardian:
    """Create frequency guardian with default contemplative parameters"""
    return FrequencyGuardian(fs=fs)

# Example usage for contemplative security demonstration
def demonstrate_frequency_guardian():
    """
    Demonstrate frequency-based contemplative security
    """
    print("🛡️ Frequency Guardian Demonstration")
    print("=" * 50)
    
    guardian = FrequencyGuardian()
    
    # Simulate biological signal (mostly low frequency)
    t = np.linspace(0, 10, 1000)  # 10 seconds at 100 Hz effective
    bio_signal = 0.1 * np.sin(0.01 * 2 * np.pi * t)  # Slow contemplative rhythm
    
    print("🌿 Testing biological signal...")
    psd = guardian.frequency_fingerprint(bio_signal)
    is_slow = guardian.validate_slowness_fingerprint(psd)
    print(f"Slowness validation: {'✅ PASS' if is_slow else '❌ FAIL'}")
    
    # Simulate high-frequency intrusion
    noise_signal = bio_signal + 0.05 * np.random.randn(len(t))  # Add noise
    print("\n⚡ Testing high-frequency intrusion...")
    intrusion, power = guardian.check_high_frequency_intrusion(noise_signal)
    print(f"Intrusion detected: {'⚠️ YES' if intrusion else '✅ NO'}")
    print(f"Max high-freq power: {power:.1f} dBFS")
    
    # Care level evaluation
    care_level = guardian.evaluate_care_level(bio_signal)
    silence_penalty = guardian.get_silence_penalty(care_level)
    print(f"\nCare level: {care_level.value}")
    print(f"Silence penalty: {silence_penalty:.1f}×")

if __name__ == "__main__":
    demonstrate_frequency_guardian() 
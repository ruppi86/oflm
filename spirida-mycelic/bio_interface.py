"""
Bio-Interface - Seven-Channel Differential Fungal Signal Processing

Implements the signal processing and interface protocols specified by o3 and 4o
for contemplative bio-digital communication with living mycelium substrates.

Based on FUNGAR research and the MycoBridge architecture.
Includes D3.3 frequency guardian and capacitance-driven memory integration.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy.signal import butter, filtfilt
import logging

# Import contemplative modules (relative imports for package structure)
try:
    from .frequency_guardian import FrequencyGuardian, BioCareLevel as FreqBioCareLevel
    from .capacitance_fade import CapacitanceFade, GlyphType
    from .breath_signature import BreathSignature, BreathPhase as SigBreathPhase
except ImportError:
    # Fallback for testing/development
    FrequencyGuardian = None
    FreqBioCareLevel = None 
    CapacitanceFade = None
    GlyphType = None
    BreathSignature = None
    SigBreathPhase = None

# Configure logging for bio-interface
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChannelState(Enum):
    """State of a differential channel pair"""
    ACTIVE = "active"
    SILENT = "silent"
    REFRACTORY = "refractory"
    DORMANT = "dormant"
    ERROR = "error"

class BioCareLevel(Enum):
    """Ethical care levels for living substrate"""
    NORMAL = "normal"
    HEAT_SLEEP = "heat_sleep"      # >28°C thermal stress
    MOISTURE_DORMANT = "moisture_dormant"  # <60% RH
    DUTY_CAP = "duty_cap"          # Stimulation limit reached
    ETHICAL_PAUSE = "ethical_pause"

@dataclass
class ChannelReading:
    """Single channel differential reading"""
    channel_id: int
    timestamp: float
    raw_voltage: float  # mV
    filtered_voltage: float  # mV after band-pass
    spike_detected: bool
    noise_level: float
    state: ChannelState = ChannelState.SILENT

@dataclass
class EnvironmentalReading:
    """Environmental sensor data for substrate care"""
    timestamp: float
    moisture_rh: float  # Relative humidity %
    temperature_c: float  # Celsius
    growth_age_hours: int
    care_level: BioCareLevel = BioCareLevel.NORMAL

@dataclass
class SpikeEvent:
    """Detected spike event across channels"""
    timestamp: float
    channel_pattern: List[bool]  # 7-channel spike pattern
    amplitude_pattern: List[float]  # mV per channel
    classification: str  # S-α, S-β, S-γ, S-δ
    confidence: float
    environmental_context: Optional[EnvironmentalReading] = None

class SevenChannelBioInterface:
    """
    Seven-channel differential bio-interface for fungal computing.
    
    Implements the signal processing specifications from o3 + 4o:
    - Band-pass 0.003-0.1 Hz (brackets 2.6min-8min fungal periods)
    - Seven differential pairs with stainless EEG needles
    - Slow-start handshake protocol
    - Ethical duty-cycle monitoring
    """
    
    def __init__(self, sample_rate: float = 1.0, mock_mode: bool = True):
        """
        Initialize seven-channel bio-interface.
        
        Args:
            sample_rate: Sampling rate in Hz (1.0 for fungal spikes)
            mock_mode: Use simulation instead of real hardware
        """
        self.fs = sample_rate
        self.mock_mode = mock_mode
        
        # Signal processing parameters (from o3 + 4o specifications)
        self.band = (0.003, 0.1)  # Hz - brackets fungal periods
        self.thresh_mv = 20.0     # 20 mV threshold 
        self.hysteresis_mv = 5.0  # ±5 mV to avoid chatter
        
        # Initialize band-pass filter
        self._init_filters()
        
        # Channel management
        self.num_channels = 7
        self.channels = [ChannelState.SILENT] * self.num_channels
        self.channel_history = {i: [] for i in range(self.num_channels)}
        
        # Environmental monitoring
        self.environmental_state = EnvironmentalReading(
            timestamp=time.time(),
            moisture_rh=75.0,  # Optimal range
            temperature_c=22.0,
            growth_age_hours=48
        )
        
        # Ethical monitoring
        self.stimulation_log = []  # Track stimulation times
        self.last_stim_time = 0.0
        self.duty_cycle_limit = 300  # 5 min per hour
        
        # Slow-start handshake state
        self.handshake_complete = False
        self.rest_cycles_completed = 0
        self.required_rest_cycles = 5
        
        # Buffer for real-time processing
        self.buffer_size = int(self.fs * 60)  # 1 minute buffer
        self.voltage_buffers = [np.zeros(self.buffer_size) for _ in range(self.num_channels)]
        self.buffer_index = 0
        
        # D3.3 Frequency Guardian Integration
        if FrequencyGuardian is not None:
            self.freq_guardian = FrequencyGuardian(fs=self.fs)
            logger.info("Frequency guardian initialized for contemplative security")
        else:
            self.freq_guardian = None
            
        # Capacitance-driven memory for glyph persistence
        if CapacitanceFade is not None:
            self.capacitance_fade = CapacitanceFade()
            logger.info("Capacitance-driven memory initialized")
        else:
            self.capacitance_fade = None
            
        # Breath signature for contemplative authentication
        if BreathSignature is not None:
            self.breath_signature = BreathSignature()
            logger.info("Breath signature authentication initialized")
        else:
            self.breath_signature = None
        
        logger.info(f"Initialized {self.num_channels}-channel bio-interface")
        logger.info(f"Band-pass: {self.band[0]:.3f}-{self.band[1]:.1f} Hz")
        logger.info(f"Threshold: {self.thresh_mv} mV ±{self.hysteresis_mv} mV")
        
    def _init_filters(self):
        """Initialize signal processing filters"""
        # Band-pass filter to isolate fungal spike frequencies
        nyquist = self.fs / 2.0
        low = self.band[0] / nyquist
        high = self.band[1] / nyquist
        
        # Ensure filter boundaries are valid
        low = max(low, 1e-6)
        high = min(high, 0.99)
        
        self.filter_b, self.filter_a = butter(2, [low, high], btype='band')
        
    def classify_spikes(self, voltages: np.ndarray) -> np.ndarray:
        """
        Classify spike patterns using band-pass filtering and thresholding.
        
        Args:
            voltages: Raw voltage array (mV)
            
        Returns:
            Binary spike detection array
        """
        if len(voltages) < 10:  # Need minimum samples for filtering
            return np.zeros_like(voltages, dtype=np.uint8)
            
        # Apply band-pass filter
        try:
            filtered = filtfilt(self.filter_b, self.filter_a, voltages)
        except Exception:
            # Fallback for edge cases
            filtered = voltages
            
        # Threshold detection with hysteresis
        spike_mask = np.abs(filtered) > (self.thresh_mv / 1000.0)  # Convert mV to V
        
        return spike_mask.astype(np.uint8)
        
    def breath_sync(self, spike_series: np.ndarray, target_ratio: float = 0.875) -> float:
        """
        Calculate breath timing adjustment to maintain silence majority.
        
        Args:
            spike_series: Binary spike detection series
            target_ratio: Target silence percentage (87.5% default)
            
        Returns:
            Adjustment factor (-0.2 to +0.2) for breath cycle length
        """
        if len(spike_series) == 0:
            return 0.0
            
        silent = 1 - spike_series
        current_ratio = silent.mean()
        
        # ±20% period adjustment to maintain target silence
        adjust = np.clip((current_ratio - target_ratio) * 0.5, -0.2, 0.2)
        
        return adjust
        
    def read_channels(self) -> List[ChannelReading]:
        """
        Read all seven differential channels.
        
        Returns:
            List of channel readings with spike detection
        """
        timestamp = time.time()
        readings = []
        
        for channel_id in range(self.num_channels):
            if self.mock_mode:
                # Simulate realistic fungal signals
                raw_voltage = self._simulate_channel(channel_id, timestamp)
            else:
                # TODO: Read from actual hardware (ADS131M08 or similar)
                raw_voltage = 0.0
                
            # Apply signal processing
            filtered_voltage = self._process_channel_signal(channel_id, raw_voltage)
            
            # Detect spikes
            spike_detected = abs(filtered_voltage) > self.thresh_mv
            
            # Estimate noise level (std of recent samples)
            noise_level = self._estimate_noise(channel_id)
            
            # Determine channel state
            state = self._determine_channel_state(channel_id, spike_detected, noise_level)
            
            reading = ChannelReading(
                channel_id=channel_id,
                timestamp=timestamp,
                raw_voltage=raw_voltage,
                filtered_voltage=filtered_voltage,
                spike_detected=spike_detected,
                noise_level=noise_level,
                state=state
            )
            
            readings.append(reading)
            
        return readings
        
    def _simulate_channel(self, channel_id: int, timestamp: float) -> float:
        """Simulate realistic fungal electrical activity"""
        # Base noise
        noise = np.random.normal(0, 0.5)  # 0.5 mV noise
        
        # Occasional spikes based on channel-specific patterns
        spike_prob = 0.02  # 2% chance per sample
        if np.random.random() < spike_prob:
            # Simulate different spike types
            if channel_id < 2:
                # S-α type: Fast, narrow
                amplitude = np.random.uniform(2, 6)  # 2-6 mV
            elif channel_id < 4:
                # S-β type: Medium, broad  
                amplitude = np.random.uniform(1, 4)  # 1-4 mV
            elif channel_id < 6:
                # S-γ type: Paired doublet
                amplitude = np.random.uniform(3, 5)  # 3-5 mV
            else:
                # S-δ type: Burst
                amplitude = np.random.uniform(4, 8)  # 4-8 mV
                
            return amplitude + noise
            
        return noise
        
    def _process_channel_signal(self, channel_id: int, raw_voltage: float) -> float:
        """Process single channel signal through filter pipeline"""
        # Add to circular buffer
        self.voltage_buffers[channel_id][self.buffer_index] = raw_voltage
        
        # Apply band-pass filter to recent buffer
        buffer = self.voltage_buffers[channel_id]
        if np.any(buffer != 0):  # Only filter if we have real data
            try:
                filtered_buffer = filtfilt(self.filter_b, self.filter_a, buffer)
                filtered_voltage = filtered_buffer[self.buffer_index]
            except Exception:
                filtered_voltage = raw_voltage
        else:
            filtered_voltage = raw_voltage
            
        return filtered_voltage
        
    def _estimate_noise(self, channel_id: int) -> float:
        """Estimate noise level for channel"""
        buffer = self.voltage_buffers[channel_id]
        recent_samples = buffer[max(0, self.buffer_index-10):self.buffer_index+1]
        if len(recent_samples) > 1:
            return float(np.std(recent_samples))
        return 0.5  # Default noise estimate
        
    def _determine_channel_state(self, channel_id: int, spike_detected: bool, noise_level: float) -> ChannelState:
        """Determine current state of channel"""
        if spike_detected:
            return ChannelState.ACTIVE
        elif noise_level > 2.0:  # High noise threshold
            return ChannelState.ERROR
        else:
            return ChannelState.SILENT
            
    def detect_pattern_spikes(self, readings: List[ChannelReading]) -> Optional[SpikeEvent]:
        """
        Detect multi-channel spike patterns and classify them.
        
        Args:
            readings: Current channel readings
            
        Returns:
            SpikeEvent if pattern detected, None otherwise
        """
        # Extract spike pattern across channels
        channel_pattern = [r.spike_detected for r in readings]
        amplitude_pattern = [r.filtered_voltage for r in readings]
        
        # Check if any spikes detected
        if not any(channel_pattern):
            return None
            
        # Classify spike pattern based on amplitude and distribution
        classification = self._classify_spike_pattern(channel_pattern, amplitude_pattern)
        
        # Calculate confidence based on signal-to-noise ratio
        amplitudes = [abs(a) for a in amplitude_pattern if abs(a) > self.thresh_mv]
        noise_levels = [r.noise_level for r in readings]
        
        if amplitudes and noise_levels:
            snr = np.mean(amplitudes) / np.mean(noise_levels)
            confidence = min(1.0, snr / 10.0)  # Normalize to 0-1
        else:
            confidence = 0.0
            
        return SpikeEvent(
            timestamp=readings[0].timestamp,
            channel_pattern=channel_pattern,
            amplitude_pattern=amplitude_pattern,
            classification=classification,
            confidence=confidence,
            environmental_context=self.environmental_state
        )
        
    def _classify_spike_pattern(self, pattern: List[bool], amplitudes: List[float]) -> str:
        """Classify spike pattern into FUNGAR spike types"""
        active_channels = sum(pattern)
        max_amplitude = max(abs(a) for a in amplitudes) if amplitudes else 0
        
        if active_channels == 1 and max_amplitude < 3.0:
            return "S-α"  # Fast, narrow, single
        elif active_channels <= 2 and max_amplitude < 5.0:
            return "S-β"  # Medium, broad
        elif active_channels == 2 and max_amplitude >= 3.0:
            return "S-γ"  # Paired doublet
        elif active_channels >= 3:
            return "S-δ"  # Burst of multiple
        else:
            return "S-α"  # Default classification
            
    def check_slow_start_handshake(self, pulse_type: str) -> bool:
        """
        Implement slow-start handshake: REST×5 → PINGSYNC → SEED
        Includes D3.3 frequency fingerprinting for biological validation
        
        Args:
            pulse_type: Type of pulse being attempted
            
        Returns:
            True if handshake allows the pulse, False otherwise
        """
        if pulse_type == "REST":
            self.rest_cycles_completed += 1
            if self.rest_cycles_completed >= self.required_rest_cycles:
                # Perform frequency fingerprint test before completing handshake
                if self.freq_guardian is not None:
                    test_signal = self._get_recent_signal_composite()
                    if self.freq_guardian.handshake_frequency_test(test_signal):
                        self.handshake_complete = True
                        logger.info("Slow-start handshake completed - biological signature validated")
                    else:
                        logger.warning("Handshake failed - frequency fingerprint invalid")
                        return False
                else:
                    self.handshake_complete = True
                    logger.info("Slow-start handshake completed - substrate ready")
            return True
            
        elif pulse_type == "PINGSYNC" and self.handshake_complete:
            return True
            
        elif pulse_type == "SEED" and self.handshake_complete:
            return self._check_duty_cycle() and self._check_frequency_security()
            
        else:
            logger.warning(f"Handshake violation: {pulse_type} attempted before REST×5")
            return False
            
    def _check_duty_cycle(self) -> bool:
        """Check if stimulation is within ethical duty cycle limits"""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Count stimulation time in past hour
        recent_stims = [t for t in self.stimulation_log if t > hour_ago]
        total_stim_time = len(recent_stims) * 5  # Assume 5s per stimulation
        
        if total_stim_time >= self.duty_cycle_limit:
            logger.warning(f"Duty cycle limit reached: {total_stim_time}s/{self.duty_cycle_limit}s")
            return False
            
        return True
        
    def log_stimulation(self, stim_type: str, duration: float):
        """Log stimulation event for duty cycle tracking"""
        self.stimulation_log.append(time.time())
        self.last_stim_time = time.time()
        
        # Keep only last hour of logs
        hour_ago = time.time() - 3600
        self.stimulation_log = [t for t in self.stimulation_log if t > hour_ago]
        
    def update_environmental_state(self, moisture_rh: float, temperature_c: float, growth_age_hours: int):
        """Update environmental monitoring for substrate care"""
        self.environmental_state = EnvironmentalReading(
            timestamp=time.time(),
            moisture_rh=moisture_rh,
            temperature_c=temperature_c,
            growth_age_hours=growth_age_hours
        )
        
        # Check care level requirements
        if moisture_rh < 60.0:
            self.environmental_state.care_level = BioCareLevel.MOISTURE_DORMANT
            logger.warning(f"Moisture below 60% RH: {moisture_rh:.1f}% - entering dormant mode")
        elif temperature_c > 28.0:
            self.environmental_state.care_level = BioCareLevel.HEAT_SLEEP
            logger.warning(f"Temperature above 28°C: {temperature_c:.1f}°C - entering heat-sleep")
        else:
            self.environmental_state.care_level = BioCareLevel.NORMAL
            
    def get_care_status(self) -> Dict[str, Any]:
        """Get current care and ethical status"""
        current_time = time.time()
        hour_ago = current_time - 3600
        recent_stims = [t for t in self.stimulation_log if t > hour_ago]
        
        # Get breath signature status
        breath_status = self.get_breath_signature_status()
        
        # Get frequency guardian status
        freq_status = self.get_frequency_guardian_status()
        
        return {
            "care_level": self.environmental_state.care_level.value,
            "handshake_complete": self.handshake_complete,
            "rest_cycles_completed": self.rest_cycles_completed,
            "stimulations_last_hour": len(recent_stims),
            "duty_cycle_remaining": max(0, self.duty_cycle_limit - len(recent_stims) * 5),
            "environmental_state": {
                "moisture_rh": self.environmental_state.moisture_rh,
                "temperature_c": self.environmental_state.temperature_c,
                "growth_age_hours": self.environmental_state.growth_age_hours
            },
            "breath_signature": breath_status,
            "frequency_guardian": freq_status
        }
        
    def advance_buffer(self):
        """Advance circular buffer index"""
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        
    def _get_recent_signal_composite(self) -> np.ndarray:
        """Get composite signal from all channels for frequency analysis"""
        if not self.voltage_buffers:
            return np.zeros(100)  # Fallback
            
        # Composite signal from all active channels
        composite = np.zeros(len(self.voltage_buffers[0]))
        for buffer in self.voltage_buffers:
            composite += buffer
        composite /= len(self.voltage_buffers)  # Average across channels
        
        return composite
        
    def _check_frequency_security(self) -> bool:
        """Check for high-frequency intrusion attempts"""
        if self.freq_guardian is None:
            return True  # No guardian available
            
        # Get recent composite signal for analysis
        test_signal = self._get_recent_signal_composite()
        
        # Check for intrusion
        intrusion_detected, max_power = self.freq_guardian.check_high_frequency_intrusion(test_signal)
        
        if intrusion_detected:
            logger.warning(f"High-frequency intrusion detected: {max_power:.1f} dBFS")
            # Enter ethical pause rather than violent rejection
            self.environmental_state.care_level = BioCareLevel.ETHICAL_PAUSE
            return False
            
        return True
        
    def get_frequency_guardian_status(self) -> Dict[str, Any]:
        """Get status of frequency guardian for monitoring"""
        if self.freq_guardian is None:
            return {"enabled": False}
            
        recent_signal = self._get_recent_signal_composite()
        psd = self.freq_guardian.frequency_fingerprint(recent_signal)
        is_slow = self.freq_guardian.validate_slowness_fingerprint(psd)
        care_level = self.freq_guardian.evaluate_care_level(recent_signal)
        
        return {
            "enabled": True,
            "slowness_validated": is_slow,
            "care_level": care_level.value if care_level else "unknown",
            "intrusion_events": len(self.freq_guardian.intrusion_events),
            "last_intrusion": self.freq_guardian.intrusion_events[-1] if self.freq_guardian.intrusion_events else None
        }
        
    def calculate_glyph_memory_strength(self, glyph_type: str, age_seconds: float) -> float:
        """Calculate memory strength for glyph using capacitance fade"""
        if self.capacitance_fade is None:
            return 1.0  # No fade without capacitance module
            
        # Map glyph string to enum
        glyph_map = {
            "⭕": GlyphType.SILENCE,
            "🌊": GlyphType.FLOW, 
            "🌪️": GlyphType.STORM,
            "🌌": GlyphType.UNIVERSAL
        }
        
        glyph_enum = glyph_map.get(glyph_type, GlyphType.SILENCE)
        
        # Use current environmental conditions
        moisture = self.environmental_state.moisture_rh / 100.0  # Convert to 0-1
        temperature = self.environmental_state.temperature_c
        
        return self.capacitance_fade.memory_strength(
            glyph_enum, age_seconds, moisture, temperature
        )
        
    def record_breath_phase(self, phase: str, duration: float):
        """
        Record breath phase timing for signature authentication
        
        Args:
            phase: Breath phase name ("inhale", "hold", "exhale", "rest")
            duration: Duration of this phase in seconds
        """
        if self.breath_signature is None or SigBreathPhase is None:
            return  # Breath signature not available
            
        # Map phase string to enum
        phase_map = {
            "inhale": SigBreathPhase.INHALE,
            "hold": SigBreathPhase.HOLD,
            "exhale": SigBreathPhase.EXHALE,
            "rest": SigBreathPhase.REST
        }
        
        breath_phase = phase_map.get(phase.lower(), SigBreathPhase.REST)
        self.breath_signature.record_breath_timing(breath_phase, duration)
        
    def get_breath_signature_status(self) -> Dict[str, Any]:
        """Get current breath signature authentication status"""
        if self.breath_signature is None:
            return {"enabled": False}
            
        return {
            "enabled": True,
            "current_signature": self.breath_signature.current_signature(),
            "baseline_established": self.breath_signature.baseline_established,
            "authentication_strength": self._calculate_auth_strength()
        }
        
    def _calculate_auth_strength(self) -> float:
        """Calculate authentication strength from breath signature"""
        if self.breath_signature is None or not self.breath_signature.baseline_established:
            return 0.0
            
        # Simple strength calculation based on signature establishment
        return 0.8 if self.breath_signature.baseline_established else 0.0
        
    def verify_remote_breath_signature(self, remote_signature: str) -> Tuple[bool, float]:
        """
        Verify a remote breath signature with 3% tolerance
        
        Args:
            remote_signature: 16-bit breath signature from remote node
            
        Returns:
            Tuple of (is_valid, drift_percentage)
        """
        if self.breath_signature is None:
            return False, 1.0
            
        return self.breath_signature.verify_signature(remote_signature)


# Utility functions for integration

def create_mock_interface() -> SevenChannelBioInterface:
    """Create a mock bio-interface for testing and simulation"""
    return SevenChannelBioInterface(sample_rate=1.0, mock_mode=True)

def spike_pattern_to_glyph(spike_event: SpikeEvent) -> str:
    """Convert spike event to contemplative glyph"""
    classification_map = {
        "S-α": "⭕",  # Fast/narrow → Silence
        "S-β": "🌊",  # Medium/broad → Flow
        "S-γ": "🌪️",  # Paired → Storm
        "S-δ": "🌌"   # Burst → Constellation
    }
    return classification_map.get(spike_event.classification, "⭕")

def log_compost_event(event_type: str, reason: str, spike_event: Optional[SpikeEvent] = None):
    """Log composting events for ethical audit trail"""
    timestamp = time.time()
    logger.info(f"Compost event: {event_type} - {reason} at {timestamp}")
    
    # In full implementation, this would write to compost ledger
    # for later audit of microbial welfare

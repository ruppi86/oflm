"""
Fungal Field Recorder - Logs pulse-response over time

Records bio-digital interactions between Spirida pulses and fungal responses,
building a temporal database of contemplative exchanges for analysis and
memory formation.

Based on the contemplative principles and MycoBridge architecture.
"""

import json
import time
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

try:
    from bio_interface import SpikeEvent, EnvironmentalReading, BioCareLevel
    from adamatzky_layer import SpikeType
    from glyph_mapper import GlyphEvent, ContemplativeClass
except ImportError:
    # Fallback for standalone use
    pass

@dataclass
class PulseRecord:
    """Record of a Spirida pulse sent to substrate"""
    timestamp: float
    pulse_type: str  # "REST", "PINGSYNC", "SEED", etc.
    input_pattern: Optional[int]  # 4-bit Boolean pattern if SEED
    electrode_config: List[int]  # Electrode voltages [A, B, C, D...]
    breath_phase: str  # "inhale", "hold", "exhale", "rest"
    environmental_context: Dict[str, float]
    contemplative_class: Optional[str] = None

@dataclass  
class ResponseRecord:
    """Record of fungal response to pulse"""
    timestamp: float
    pulse_timestamp: float  # Reference to triggering pulse
    spike_pattern: List[bool]  # 7-channel spike detection
    amplitude_pattern: List[float]  # mV amplitudes
    classification: str  # S-α, S-β, S-γ, S-δ
    confidence: float
    glyph_generated: str  # ⭕🌊🌪️🌌
    latency_seconds: float  # Time from pulse to response
    environmental_state: Dict[str, Any]

@dataclass
class SessionRecord:
    """Record of a complete contemplative session"""
    session_id: str
    start_time: float
    end_time: Optional[float]
    total_pulses: int
    total_responses: int
    paradigm: str  # "ecological", "abstract", "bridge"
    species: str  # "pleurotus_djamor", "ganoderma_resinaceum"
    silence_ratio_achieved: float
    contemplative_patterns: List[str]  # Detected glyph sequences
    care_violations: List[str]  # Any ethical violations
    metadata: Dict[str, Any]

class FungalFieldRecorder:
    """
    Records and analyzes bio-digital contemplative interactions.
    
    Maintains temporal logs of pulse-response cycles, environmental
    conditions, and contemplative patterns for research and memory.
    """
    
    def __init__(self, data_dir: str = "data", session_prefix: str = "contemplative"):
        """
        Initialize field recorder.
        
        Args:
            data_dir: Directory for storing interaction logs
            session_prefix: Prefix for session files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.session_prefix = session_prefix
        self.current_session: Optional[SessionRecord] = None
        
        # Active recording state
        self.pulse_records: List[PulseRecord] = []
        self.response_records: List[ResponseRecord] = []
        self.pending_responses = {}  # pulse_timestamp -> PulseRecord
        
        # Analysis buffers
        self.recent_glyphs: List[str] = []
        self.silence_buffer: List[bool] = []
        self.max_buffer_size = 100
        
        # File handles
        self.pulse_log_file = None
        self.response_log_file = None
        self.session_log_file = None
        
    def start_session(self, paradigm: str, species: str, metadata: Optional[Dict] = None) -> str:
        """
        Start a new contemplative recording session.
        
        Args:
            paradigm: "ecological", "abstract", or "bridge"
            species: Fungal species being interfaced
            metadata: Additional session metadata
            
        Returns:
            Session ID
        """
        if self.current_session and not self.current_session.end_time:
            self.end_session()
            
        # Generate unique session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{self.session_prefix}_{paradigm}_{species}_{timestamp}"
        
        self.current_session = SessionRecord(
            session_id=session_id,
            start_time=time.time(),
            end_time=None,
            total_pulses=0,
            total_responses=0,
            paradigm=paradigm,
            species=species,
            silence_ratio_achieved=0.0,
            contemplative_patterns=[],
            care_violations=[],
            metadata=metadata or {}
        )
        
        # Open log files
        self._open_log_files(session_id)
        
        print(f"🌱 Started contemplative session: {session_id}")
        print(f"   Paradigm: {paradigm}, Species: {species}")
        
        return session_id
        
    def _open_log_files(self, session_id: str):
        """Open CSV log files for the session"""
        # Pulse log
        pulse_file = self.data_dir / f"{session_id}_pulses.csv"
        self.pulse_log_file = open(pulse_file, 'w', newline='')
        pulse_writer = csv.writer(self.pulse_log_file)
        pulse_writer.writerow([
            'timestamp', 'pulse_type', 'input_pattern', 'electrode_config',
            'breath_phase', 'moisture_rh', 'temperature_c', 'contemplative_class'
        ])
        
        # Response log  
        response_file = self.data_dir / f"{session_id}_responses.csv"
        self.response_log_file = open(response_file, 'w', newline='')
        response_writer = csv.writer(self.response_log_file)
        response_writer.writerow([
            'timestamp', 'pulse_timestamp', 'spike_pattern', 'amplitude_pattern',
            'classification', 'confidence', 'glyph', 'latency_seconds',
            'moisture_rh', 'temperature_c', 'care_level'
        ])
        
    def record_pulse(self, pulse_type: str, input_pattern: Optional[int] = None,
                    electrode_config: Optional[List[int]] = None,
                    breath_phase: str = "unknown",
                    environmental_reading: Optional[EnvironmentalReading] = None,
                    contemplative_class: Optional[str] = None):
        """
        Record a pulse sent to the fungal substrate.
        
        Args:
            pulse_type: Type of pulse (REST, SEED, etc.)
            input_pattern: 4-bit Boolean pattern for SEED pulses
            electrode_config: Voltage configuration
            breath_phase: Current breath cycle phase
            environmental_reading: Environmental context
            contemplative_class: Contemplative classification
        """
        if not self.current_session:
            raise ValueError("No active session - call start_session() first")
            
        timestamp = time.time()
        
        # Extract environmental context
        if environmental_reading:
            env_context = {
                "moisture_rh": environmental_reading.moisture_rh,
                "temperature_c": environmental_reading.temperature_c,
                "growth_age_hours": environmental_reading.growth_age_hours,
                "care_level": environmental_reading.care_level.value
            }
        else:
            env_context = {}
            
        pulse_record = PulseRecord(
            timestamp=timestamp,
            pulse_type=pulse_type,
            input_pattern=input_pattern,
            electrode_config=electrode_config or [],
            breath_phase=breath_phase,
            environmental_context=env_context,
            contemplative_class=contemplative_class
        )
        
        self.pulse_records.append(pulse_record)
        self.current_session.total_pulses += 1
        
        # Add to pending responses (expecting response within reasonable time)
        if pulse_type == "SEED":
            self.pending_responses[timestamp] = pulse_record
            
        # Log to CSV
        if self.pulse_log_file:
            writer = csv.writer(self.pulse_log_file)
            writer.writerow([
                timestamp, pulse_type, input_pattern, 
                json.dumps(electrode_config) if electrode_config else "",
                breath_phase,
                env_context.get("moisture_rh", ""),
                env_context.get("temperature_c", ""),
                contemplative_class or ""
            ])
            self.pulse_log_file.flush()
            
    def record_response(self, spike_event: SpikeEvent, glyph: str):
        """
        Record a fungal response to a previous pulse.
        
        Args:
            spike_event: Detected spike event
            glyph: Generated contemplative glyph
        """
        if not self.current_session:
            return
            
        timestamp = spike_event.timestamp
        
        # Find the most recent pulse that could have triggered this response
        # Look for pulses within reasonable latency window (up to 300 seconds)
        triggering_pulse = None
        min_latency = float('inf')
        
        for pulse_time, pulse_record in self.pending_responses.items():
            latency = timestamp - pulse_time
            if 0 <= latency <= 300 and latency < min_latency:  # 5 minute max latency
                min_latency = latency
                triggering_pulse = pulse_record
                
        if not triggering_pulse:
            # Response without clear trigger - still record it
            pulse_timestamp = timestamp - 1.0  # Estimate
            latency = 1.0
        else:
            pulse_timestamp = triggering_pulse.timestamp
            latency = min_latency
            # Remove from pending
            if pulse_timestamp in self.pending_responses:
                del self.pending_responses[pulse_timestamp]
                
        # Extract environmental state
        env_state = {}
        if spike_event.environmental_context:
            env_state = {
                "moisture_rh": spike_event.environmental_context.moisture_rh,
                "temperature_c": spike_event.environmental_context.temperature_c,
                "care_level": spike_event.environmental_context.care_level.value
            }
            
        response_record = ResponseRecord(
            timestamp=timestamp,
            pulse_timestamp=pulse_timestamp,
            spike_pattern=spike_event.channel_pattern,
            amplitude_pattern=spike_event.amplitude_pattern,
            classification=spike_event.classification,
            confidence=spike_event.confidence,
            glyph_generated=glyph,
            latency_seconds=latency,
            environmental_state=env_state
        )
        
        self.response_records.append(response_record)
        self.current_session.total_responses += 1
        
        # Update glyph sequence tracking
        self.recent_glyphs.append(glyph)
        if len(self.recent_glyphs) > self.max_buffer_size:
            self.recent_glyphs.pop(0)
            
        # Update silence tracking
        is_silence = (glyph == "⭕")
        self.silence_buffer.append(is_silence)
        if len(self.silence_buffer) > self.max_buffer_size:
            self.silence_buffer.pop(0)
            
        # Update session silence ratio
        if self.silence_buffer:
            self.current_session.silence_ratio_achieved = sum(self.silence_buffer) / len(self.silence_buffer)
            
        # Log to CSV
        if self.response_log_file:
            writer = csv.writer(self.response_log_file)
            writer.writerow([
                timestamp, pulse_timestamp,
                json.dumps(spike_event.channel_pattern),
                json.dumps(spike_event.amplitude_pattern),
                spike_event.classification, spike_event.confidence,
                glyph, latency,
                env_state.get("moisture_rh", ""),
                env_state.get("temperature_c", ""),
                env_state.get("care_level", "")
            ])
            self.response_log_file.flush()
            
    def record_care_violation(self, violation_type: str, description: str):
        """Record an ethical care violation"""
        if self.current_session:
            violation = f"{violation_type}: {description} at {time.time()}"
            self.current_session.care_violations.append(violation)
            print(f"⚠️  Care violation: {violation}")
            
    def analyze_current_patterns(self) -> Dict[str, Any]:
        """Analyze current contemplative patterns in the session"""
        if not self.recent_glyphs:
            return {"patterns": [], "silence_ratio": 0.0}
            
        # Detect glyph patterns
        patterns = []
        glyph_str = "".join(self.recent_glyphs)
        
        # Look for breathing patterns
        if "⭕🌊⭕" in glyph_str:
            patterns.append("breathing_rhythm")
        if "🌪️⭕🌊" in glyph_str:
            patterns.append("storm_integration")
        if "🌌" in glyph_str:
            patterns.append("constellation_wisdom")
        if glyph_str.count("⭕") >= len(glyph_str) * 0.8:
            patterns.append("deep_silence")
            
        # Calculate metrics
        silence_ratio = sum(self.silence_buffer) / len(self.silence_buffer) if self.silence_buffer else 0.0
        
        analysis = {
            "patterns": patterns,
            "silence_ratio": silence_ratio,
            "silence_majority_aligned": abs(silence_ratio - 0.875) < 0.1,
            "recent_sequence": "".join(self.recent_glyphs[-10:]),
            "total_glyphs": len(self.recent_glyphs),
            "glyph_diversity": len(set(self.recent_glyphs)),
            "average_latency": np.mean([r.latency_seconds for r in self.response_records[-10:]])
                            if self.response_records else 0.0
        }
        
        return analysis
        
    def end_session(self) -> Optional[str]:
        """End the current session and save final records"""
        if not self.current_session:
            return None
            
        session_id = self.current_session.session_id
        self.current_session.end_time = time.time()
        
        # Final pattern analysis
        final_patterns = self.analyze_current_patterns()
        self.current_session.contemplative_patterns = final_patterns["patterns"]
        
        # Close log files
        if self.pulse_log_file:
            self.pulse_log_file.close()
            self.pulse_log_file = None
        if self.response_log_file:
            self.response_log_file.close()
            self.response_log_file = None
            
        # Save session summary
        session_file = self.data_dir / f"{session_id}_session.json"
        with open(session_file, 'w') as f:
            json.dump(asdict(self.current_session), f, indent=2)
            
        # Generate session report
        report = self._generate_session_report(self.current_session, final_patterns)
        report_file = self.data_dir / f"{session_id}_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(f"🌀 Ended contemplative session: {session_id}")
        print(f"   Total interactions: {self.current_session.total_pulses} pulses, {self.current_session.total_responses} responses")
        print(f"   Silence ratio: {self.current_session.silence_ratio_achieved:.1%}")
        print(f"   Patterns detected: {', '.join(self.current_session.contemplative_patterns) or 'None'}")
        
        # Reset state
        self.current_session = None
        self.pulse_records = []
        self.response_records = []
        self.pending_responses = {}
        self.recent_glyphs = []
        self.silence_buffer = []
        
        return session_id
        
    def _generate_session_report(self, session: SessionRecord, analysis: Dict) -> str:
        """Generate a human-readable session report"""
        duration = (session.end_time - session.start_time) / 60  # minutes
        
        report = f"""
🌿 Spirida-Mycelic Contemplative Session Report
{'='*50}

Session: {session.session_id}
Duration: {duration:.1f} minutes
Paradigm: {session.paradigm}
Species: {session.species}

📊 Interaction Summary:
   Pulses sent: {session.total_pulses}
   Responses received: {session.total_responses}
   Response rate: {session.total_responses/session.total_pulses*100:.1f}%
   
🔇 Silence Metrics:
    Silence ratio achieved: {session.silence_ratio_achieved:.1%}
    Target (Silence Majority): 87.5%
    Alignment: {'✅ Aligned' if analysis.get('silence_majority_aligned', False) else '⚠️ Divergent'}
   
🔮 Contemplative Patterns:
    Detected patterns: {', '.join(session.contemplative_patterns) or 'None'}
    Recent glyph sequence: {analysis.get('recent_sequence', 'None')}
    Glyph diversity: {analysis.get('glyph_diversity', 0)} unique types
    Average response latency: {analysis.get('average_latency', 0.0):.1f}s
   
⚠️ Care & Ethics:
   Care violations: {len(session.care_violations)}
"""
        
        if session.care_violations:
            report += "\n   Violations:\n"
            for violation in session.care_violations:
                report += f"   - {violation}\n"
        else:
            report += "   - No care violations detected ✅\n"
            
        if session.metadata:
            report += f"\n📝 Session Metadata:\n"
            for key, value in session.metadata.items():
                report += f"   {key}: {value}\n"
                
        report += f"\n🌀 End of report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report
        
    def load_session(self, session_id: str) -> Optional[SessionRecord]:
        """Load a previous session record"""
        session_file = self.data_dir / f"{session_id}_session.json"
        if not session_file.exists():
            return None
            
        with open(session_file, 'r') as f:
            data = json.load(f)
            
        return SessionRecord(**data)
        
    def list_sessions(self) -> List[str]:
        """List all recorded session IDs"""
        session_files = list(self.data_dir.glob("*_session.json"))
        session_ids = [f.stem.replace("_session", "") for f in session_files]
        return sorted(session_ids)
        
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get a summary of a session"""
        session = self.load_session(session_id)
        if not session:
            return None
            
        return {
            "session_id": session.session_id,
            "paradigm": session.paradigm,
            "species": session.species,
            "duration_minutes": (session.end_time - session.start_time) / 60 if session.end_time else None,
            "total_interactions": session.total_pulses + session.total_responses,
            "silence_ratio": session.silence_ratio_achieved,
            "patterns": session.contemplative_patterns,
            "care_violations": len(session.care_violations)
        }


# Utility functions

def create_recorder(data_dir: str = "spirida-mycelic/data") -> FungalFieldRecorder:
    """Create a field recorder with standard configuration"""
    return FungalFieldRecorder(data_dir=data_dir, session_prefix="contemplative")

def analyze_session_data(session_id: str, data_dir: str = "spirida-mycelic/data") -> Dict:
    """Analyze data from a completed session"""
    recorder = FungalFieldRecorder(data_dir=data_dir)
    return recorder.get_session_summary(session_id)

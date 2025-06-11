"""
Spiramycel Runtime Patch System

Safe sandbox for expanding glyph IDs into actionable network commands.
Logs rather than executes patches for contemplative debugging.

Part of the Organic Femto Language Model (OFLM) framework.
Enables mycelial networks to suggest repairs without forcing execution.
"""

import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
from enum import Enum
import random

# Handle both relative and direct imports
try:
    from .glyph_codec import SpiramycelGlyphCodec, GlyphCategory
except ImportError:
    from glyph_codec import SpiramycelGlyphCodec, GlyphCategory

class PatchSeverity(Enum):
    """Severity levels for network repair patches."""
    INFO = "info"           # Informational, safe to ignore
    MINOR = "minor"         # Minor adjustment, low risk
    MODERATE = "moderate"   # Significant change, requires attention
    CRITICAL = "critical"   # Urgent repair, high priority
    CONTEMPLATIVE = "contemplative"  # Silence/pause action

class PatchStatus(Enum):
    """Execution status of patches."""
    PROPOSED = "proposed"   # Patch suggested but not acted upon
    LOGGED = "logged"       # Safely logged for analysis
    SIMULATED = "simulated" # Tested in sandbox
    APPROVED = "approved"   # Ready for execution
    EXECUTED = "executed"   # Actually performed
    REJECTED = "rejected"   # Deemed unsafe or unnecessary

@dataclass
class NetworkPatch:
    """
    A single network repair action derived from a glyph.
    
    Contains both the action description and safety metadata.
    """
    timestamp: float
    glyph_id: int
    glyph_symbol: str
    action_type: str         # e.g., "increase_flow_rate", "pause_transmission"
    target_component: str    # e.g., "network_interface", "power_management"
    parameters: Dict[str, Any]  # Action-specific parameters
    severity: PatchSeverity
    estimated_impact: float  # 0-1, predicted impact on network health
    
    # Safety and tracking
    status: PatchStatus = PatchStatus.PROPOSED
    safety_score: float = 0.5    # 0-1, higher = safer
    requires_consensus: bool = False  # Needs community approval
    bioregion: str = "local"
    network_context: Dict[str, float] = None  # Current sensor readings
    
    def __post_init__(self):
        if self.network_context is None:
            self.network_context = {}
    
    def is_safe_to_execute(self) -> bool:
        """Check if patch meets safety criteria for execution."""
        safety_checks = [
            self.safety_score >= 0.7,           # High safety score
            self.severity != PatchSeverity.CRITICAL or self.requires_consensus,  # Critical patches need consensus
            self.status in [PatchStatus.APPROVED, PatchStatus.SIMULATED],       # Must be approved
        ]
        return all(safety_checks)
    
    def estimate_repair_effectiveness(self) -> float:
        """Predict how effective this repair will be."""
        # Base effectiveness from impact
        base_effectiveness = self.estimated_impact
        
        # Adjust based on severity appropriateness
        if self.severity == PatchSeverity.CONTEMPLATIVE:
            base_effectiveness *= 0.8  # Contemplative actions are gentle
        elif self.severity == PatchSeverity.CRITICAL:
            base_effectiveness *= 1.2  # Critical repairs can be very effective
        
        # Safety penalty - unsafe actions are less effective
        safety_factor = (self.safety_score + 1.0) / 2.0
        
        return min(base_effectiveness * safety_factor, 1.0)

class SpiramycelRuntimePatcher:
    """
    Safe sandbox for converting glyphs into network repair actions.
    
    Follows contemplative principles: suggests rather than commands,
    logs rather than executes, builds consensus rather than forcing.
    """
    
    def __init__(self, patch_log_path: str = "network_patches.jsonl"):
        self.codec = SpiramycelGlyphCodec()
        self.patch_log_path = Path(patch_log_path)
        self.patch_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Safety configuration
        self.safety_enabled = True
        self.require_consensus_threshold = 0.8  # Impact threshold requiring consensus
        self.max_patches_per_cycle = 3          # Limit simultaneous patches
        
        # Patch conversion rules
        self.action_templates = self._initialize_action_templates()
        
        # Network state simulation (for safe testing)
        self.simulated_network_state = {
            "latency": 0.15,      # seconds
            "voltage": 3.3,       # volts  
            "temperature": 22.5,  # celsius
            "bandwidth": 0.85,    # utilization 0-1
            "error_rate": 0.02,   # packet error rate
            "uptime": 86400       # seconds since last restart
        }
    
    def _initialize_action_templates(self) -> Dict[str, Dict]:
        """Initialize the templates for converting repair actions to network commands."""
        return {
            # Network topology actions
            "increase_flow_rate": {
                "target": "network_interface",
                "parameters": {"bandwidth_multiplier": 1.2, "queue_size": "+20%"},
                "severity": PatchSeverity.MODERATE,
                "safety_score": 0.8,
                "estimated_impact": 0.7
            },
            "redirect_to_neighbor": {
                "target": "routing_table", 
                "parameters": {"next_hop": "auto_detect", "route_priority": "high"},
                "severity": PatchSeverity.MINOR,
                "safety_score": 0.9,
                "estimated_impact": 0.6
            },
            "throttle_bandwidth": {
                "target": "network_interface",
                "parameters": {"bandwidth_limit": "0.8x", "burst_allowance": "reduced"},
                "severity": PatchSeverity.MINOR,
                "safety_score": 0.95,
                "estimated_impact": 0.5
            },
            "pause_transmission": {
                "target": "network_interface",
                "parameters": {"pause_duration": 2.0, "graceful": True},
                "severity": PatchSeverity.CONTEMPLATIVE,
                "safety_score": 1.0,
                "estimated_impact": 0.4
            },
            
            # Energy management actions  
            "voltage_regulation": {
                "target": "power_management",
                "parameters": {"target_voltage": "3.3V", "regulation_mode": "adaptive"},
                "severity": PatchSeverity.CRITICAL,
                "safety_score": 0.6,
                "estimated_impact": 0.9
            },
            "reduce_consumption": {
                "target": "power_management",
                "parameters": {"cpu_scaling": 0.8, "disable_non_essential": True},
                "severity": PatchSeverity.MODERATE,
                "safety_score": 0.85,
                "estimated_impact": 0.7
            },
            "harvest_solar": {
                "target": "energy_harvesting",
                "parameters": {"solar_tracking": "enabled", "charge_rate": "optimized"},
                "severity": PatchSeverity.MINOR,
                "safety_score": 0.9,
                "estimated_impact": 0.6
            },
            "sleep_mode": {
                "target": "system_control",
                "parameters": {"sleep_duration": "auto", "wake_triggers": ["solar", "network"]},
                "severity": PatchSeverity.MODERATE,
                "safety_score": 0.8,
                "estimated_impact": 0.8
            },
            
            # System health actions
            "status_ok": {
                "target": "monitoring",
                "parameters": {"confidence": "high", "report_upstream": True},
                "severity": PatchSeverity.INFO,
                "safety_score": 1.0,
                "estimated_impact": 0.2
            },
            "preventive_care": {
                "target": "maintenance",
                "parameters": {"diagnostic_level": "standard", "cleanup": "gentle"},
                "severity": PatchSeverity.MINOR,
                "safety_score": 0.9,
                "estimated_impact": 0.5
            },
            "auto_healing": {
                "target": "self_repair",
                "parameters": {"repair_mode": "conservative", "backup_first": True},
                "severity": PatchSeverity.MODERATE,
                "safety_score": 0.7,
                "estimated_impact": 0.8
            },
            "system_scan": {
                "target": "diagnostics",
                "parameters": {"scan_depth": "comprehensive", "repair_suggestions": True},
                "severity": PatchSeverity.MINOR,
                "safety_score": 0.95,
                "estimated_impact": 0.4
            },
            
            # Contemplative/silence actions
            "breathing_space": {
                "target": "contemplative_control",
                "parameters": {"pause_duration": 1.5, "mindful": True},
                "severity": PatchSeverity.CONTEMPLATIVE,
                "safety_score": 1.0,
                "estimated_impact": 0.3
            },
            "complete_quiet": {
                "target": "contemplative_control", 
                "parameters": {"silence_duration": 5.0, "deep_rest": True},
                "severity": PatchSeverity.CONTEMPLATIVE,
                "safety_score": 1.0,
                "estimated_impact": 0.2
            },
            "meditation_mode": {
                "target": "contemplative_control",
                "parameters": {"duration": 60.0, "breathe_sync": True},
                "severity": PatchSeverity.CONTEMPLATIVE,
                "safety_score": 1.0,
                "estimated_impact": 0.4
            }
        }
    
    def expand_glyph_to_patch(self, 
                             glyph_id: int, 
                             network_context: Dict[str, float] = None,
                             bioregion: str = "local") -> Optional[NetworkPatch]:
        """
        Convert a glyph ID into a network patch action.
        
        Returns None if glyph is unknown or action is unsafe.
        """
        if network_context is None:
            network_context = self.simulated_network_state.copy()
        
        # Get glyph information
        glyph_symbol = self.codec.encode_glyph(glyph_id)
        repair_action = self.codec.get_repair_action(glyph_id)
        
        if not glyph_symbol or not repair_action:
            return None
        
        # Look up action template
        if repair_action not in self.action_templates:
            # Unknown action - create minimal safe patch
            return self._create_unknown_action_patch(glyph_id, glyph_symbol, repair_action, network_context, bioregion)
        
        template = self.action_templates[repair_action]
        
        # Create patch from template
        patch = NetworkPatch(
            timestamp=time.time(),
            glyph_id=glyph_id,
            glyph_symbol=glyph_symbol,
            action_type=repair_action,
            target_component=template["target"],
            parameters=template["parameters"].copy(),
            severity=template["severity"],
            estimated_impact=template["estimated_impact"],
            safety_score=template["safety_score"],
            bioregion=bioregion,
            network_context=network_context.copy()
        )
        
        # Adjust patch based on network context
        self._contextualize_patch(patch, network_context)
        
        # Determine if consensus is required
        if patch.estimated_impact >= self.require_consensus_threshold:
            patch.requires_consensus = True
        
        return patch
    
    def _create_unknown_action_patch(self, glyph_id: int, symbol: str, action: str, 
                                   context: Dict[str, float], bioregion: str) -> NetworkPatch:
        """Create a safe patch for unknown actions."""
        return NetworkPatch(
            timestamp=time.time(),
            glyph_id=glyph_id,
            glyph_symbol=symbol,
            action_type=action,
            target_component="unknown_system",
            parameters={"action": action, "safety_mode": "observe_only"},
            severity=PatchSeverity.INFO,
            estimated_impact=0.1,
            safety_score=0.9,  # Unknown but safe
            bioregion=bioregion,
            network_context=context.copy()
        )
    
    def _contextualize_patch(self, patch: NetworkPatch, context: Dict[str, float]):
        """Adjust patch parameters based on current network conditions."""
        # Adjust based on current network health
        if "error_rate" in context and context["error_rate"] > 0.1:
            # High error rate - be more conservative
            patch.safety_score *= 0.9
            if "bandwidth_multiplier" in patch.parameters:
                patch.parameters["bandwidth_multiplier"] = min(1.1, patch.parameters["bandwidth_multiplier"])
        
        if "voltage" in context and context["voltage"] < 3.0:
            # Low voltage - prioritize energy saving
            if patch.action_type in ["increase_flow_rate", "voltage_regulation"]:
                patch.requires_consensus = True
                patch.safety_score *= 0.8
        
        if "temperature" in context and context["temperature"] > 35.0:
            # High temperature - be conservative with power changes
            if patch.target_component == "power_management":
                patch.safety_score *= 0.7
    
    def process_glyph_sequence(self, 
                              glyph_sequence: List[int],
                              network_context: Dict[str, float] = None,
                              bioregion: str = "local") -> List[NetworkPatch]:
        """
        Process a sequence of glyphs into network patches.
        
        Applies safety limits and contemplative pacing.
        """
        if network_context is None:
            network_context = self.simulated_network_state.copy()
        
        patches = []
        contemplative_count = 0
        
        for glyph_id in glyph_sequence:
            # Limit number of patches per cycle
            if len(patches) >= self.max_patches_per_cycle:
                break
            
            patch = self.expand_glyph_to_patch(glyph_id, network_context, bioregion)
            if patch:
                patches.append(patch)
                
                # Count contemplative actions
                if patch.severity == PatchSeverity.CONTEMPLATIVE:
                    contemplative_count += 1
        
        # Ensure contemplative majority (following Tystnadsmajoritet)
        total_actions = len(patches)
        contemplative_ratio = contemplative_count / total_actions if total_actions > 0 else 1.0
        
        # If not enough contemplative actions, add some
        if contemplative_ratio < 0.75 and total_actions > 0:
            contemplative_glyphs = self.codec.get_contemplative_glyphs()
            needed_contemplative = int(total_actions * 0.8) - contemplative_count
            
            for _ in range(min(needed_contemplative, 2)):  # Add at most 2 contemplative patches
                glyph_id = random.choice(contemplative_glyphs)
                patch = self.expand_glyph_to_patch(glyph_id, network_context, bioregion)
                if patch:
                    patches.append(patch)
        
        return patches
    
    def log_patch(self, patch: NetworkPatch):
        """Safely log a patch to the patch log file."""
        patch.status = PatchStatus.LOGGED
        
        try:
            with open(self.patch_log_path, 'a', encoding='utf-8') as f:
                data = asdict(patch)
                # Convert enums to strings
                data['severity'] = data['severity'].value
                data['status'] = data['status'].value
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️ Error logging patch: {e}")
    
    def log_patch_sequence(self, patches: List[NetworkPatch]):
        """Log a sequence of patches as a group."""
        for patch in patches:
            self.log_patch(patch)
    
    def simulate_patch_execution(self, patch: NetworkPatch) -> Dict[str, Any]:
        """
        Simulate patch execution in safe sandbox.
        
        Returns simulation results without actually changing anything.
        """
        patch.status = PatchStatus.SIMULATED
        
        # Simulate the effect on network state
        simulated_state = self.simulated_network_state.copy()
        improvement = 0.0
        
        if patch.action_type == "increase_flow_rate":
            simulated_state["bandwidth"] = min(1.0, simulated_state["bandwidth"] * 1.2)
            improvement = 0.2
        elif patch.action_type == "reduce_consumption":
            simulated_state["voltage"] = min(3.5, simulated_state["voltage"] + 0.1)
            improvement = 0.15
        elif patch.action_type == "pause_transmission":
            simulated_state["error_rate"] = max(0.0, simulated_state["error_rate"] * 0.8)
            improvement = 0.1
        elif patch.severity == PatchSeverity.CONTEMPLATIVE:
            # Contemplative actions provide gentle improvements
            simulated_state["error_rate"] = max(0.0, simulated_state["error_rate"] * 0.95)
            improvement = 0.05
        
        # Add some randomness to simulation
        improvement += random.uniform(-0.05, 0.05)
        improvement = max(0.0, min(1.0, improvement))
        
        return {
            "simulated_state": simulated_state,
            "estimated_improvement": improvement,
            "safety_analysis": {
                "safe_to_execute": patch.is_safe_to_execute(),
                "predicted_effectiveness": patch.estimate_repair_effectiveness(),
                "risk_factors": self._analyze_risks(patch)
            }
        }
    
    def _analyze_risks(self, patch: NetworkPatch) -> List[str]:
        """Analyze potential risks of executing a patch."""
        risks = []
        
        if patch.severity == PatchSeverity.CRITICAL:
            risks.append("Critical system change")
        
        if patch.estimated_impact > 0.8:
            risks.append("High impact on network performance")
        
        if patch.requires_consensus and not patch.status == PatchStatus.APPROVED:
            risks.append("Requires community consensus")
        
        if patch.safety_score < 0.7:
            risks.append("Below safety threshold")
        
        # Context-specific risks
        context = patch.network_context
        if context.get("voltage", 3.3) < 3.0 and patch.target_component == "power_management":
            risks.append("Low voltage environment")
        
        if context.get("error_rate", 0.0) > 0.1 and patch.action_type == "increase_flow_rate":
            risks.append("High error rate may worsen with increased flow")
        
        return risks
    
    def get_patch_recommendations(self, 
                                 network_context: Dict[str, float] = None,
                                 bioregion: str = "local") -> List[int]:
        """
        Recommend glyph IDs based on current network conditions.
        
        Returns contemplative sequence with appropriate repair glyphs.
        """
        if network_context is None:
            network_context = self.simulated_network_state.copy()
        
        recommendations = []
        
        # Analyze network conditions
        if network_context.get("error_rate", 0.0) > 0.05:
            recommendations.extend([0x03, 0x04])  # throttle + pause
        
        if network_context.get("voltage", 3.3) < 3.1:
            recommendations.extend([0x12, 0x14])  # battery conservation + night mode
        
        if network_context.get("latency", 0.1) > 0.2:
            recommendations.extend([0x01, 0x02])  # bandwidth + reroute
        
        # Always add contemplative glyphs (Tystnadsmajoritet)
        contemplative_glyphs = self.codec.get_contemplative_glyphs()
        recommendations.extend(random.choices(contemplative_glyphs, k=3))
        
        # Status update if things are good
        if (network_context.get("error_rate", 0.0) < 0.02 and 
            network_context.get("voltage", 3.3) > 3.2):
            recommendations.append(0x21)  # systems nominal
        
        return recommendations[:6]  # Return reasonable number

# Demo functions
def demo_runtime_patcher():
    """Demonstrate the runtime patch system."""
    print("🔧 Spiramycel Runtime Patcher Demo")
    print("=" * 50)
    
    patcher = SpiramycelRuntimePatcher("demo_patches.jsonl")
    
    print("\n🌐 Simulated Network State:")
    for key, value in patcher.simulated_network_state.items():
        print(f"  {key}: {value}")
    
    # Test individual glyph expansion
    print("\n🌱 Testing Glyph Expansion:")
    test_glyphs = [0x01, 0x12, 0x21, 0x31]  # bandwidth, battery, nominal, pause
    
    for glyph_id in test_glyphs:
        patch = patcher.expand_glyph_to_patch(glyph_id)
        if patch:
            print(f"  {patch.glyph_symbol} → {patch.action_type}")
            print(f"    Target: {patch.target_component}, Safety: {patch.safety_score:.2f}")
            print(f"    Impact: {patch.estimated_impact:.2f}, Severity: {patch.severity.value}")
    
    # Test sequence processing
    print("\n🍄 Processing Glyph Sequence:")
    test_sequence = [0x01, 0x21, 0x31, 0x32]  # bandwidth + nominal + pauses
    patches = patcher.process_glyph_sequence(test_sequence)
    
    print(f"  Generated {len(patches)} patches:")
    contemplative_count = sum(1 for p in patches if p.severity == PatchSeverity.CONTEMPLATIVE)
    print(f"  Contemplative ratio: {contemplative_count}/{len(patches)} = {contemplative_count/len(patches)*100:.1f}%")
    
    # Log patches safely
    print(f"\n📝 Logging patches to {patcher.patch_log_path}")
    patcher.log_patch_sequence(patches)
    
    # Simulate execution
    print("\n🧪 Simulating Patch Execution:")
    for patch in patches[:2]:  # Simulate first 2 patches
        results = patcher.simulate_patch_execution(patch)
        print(f"  {patch.glyph_symbol} → Improvement: {results['estimated_improvement']:.2f}")
        print(f"    Safe: {results['safety_analysis']['safe_to_execute']}")
        if results['safety_analysis']['risk_factors']:
            print(f"    Risks: {', '.join(results['safety_analysis']['risk_factors'])}")
    
    # Test recommendations
    print("\n💡 Network Recommendations:")
    recommendations = patcher.get_patch_recommendations()
    formatted = patcher.codec.format_glyph_sequence(recommendations)
    print(f"  Suggested sequence: {formatted}")
    
    print("\n🌊 Patches logged safely without execution")
    print("🍄 Mycelial networks suggest rather than command")
    print("🌱 Community consensus builds collective network wisdom")

if __name__ == "__main__":
    demo_runtime_patcher() 
"""
Spirida-Mycelic Package
Bio-Digital Interface for Contemplative Computing

Implements the complete D3.3 integration for spirida-mycelic systems
with frequency guardian, capacitance-driven memory, and breath signature
authentication.
"""

__version__ = "0.1.0"
__author__ = "Claude 4 Sonnet"

# Core bio-interface components
try:
    from .bio_interface import SevenChannelBioInterface
    from .frequency_guardian import FrequencyGuardian
    from .capacitance_fade import CapacitanceFade
    from .breath_signature import BreathSignature
    from .adamatzky_layer import AdamatzkyReservoir
    from .fungal_field_recorder import FungalFieldRecorder
except ImportError:
    # Allow package to be imported even if some dependencies missing
    pass

__all__ = [
    "SevenChannelBioInterface",
    "FrequencyGuardian", 
    "CapacitanceFade",
    "BreathSignature",
    "AdamatzkyReservoir",
    "FungalFieldRecorder"
] 
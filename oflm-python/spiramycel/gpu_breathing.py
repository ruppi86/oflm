#!/usr/bin/env python3
"""
GPU Contemplative Breathing Monitor - OPTIMIZED VERSION

Lightweight adaptive breathing based on actual GPU load with minimal overhead.
The AI breathes slower when the GPU is under stress - both practical and contemplative!

Key optimizations:
- Cached monitoring (only check GPU every 5-10 seconds)
- Light monitoring mode for models under 1M parameters  
- Configurable thresholds and intervals
- Optional monitoring (can be completely disabled)
- Minimal subprocess overhead
"""

import time
import subprocess
import json
from typing import Dict, Optional
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class GPUState:
    """Current GPU monitoring state"""
    temperature: Optional[float] = None      # °C
    memory_used: Optional[float] = None      # 0.0-1.0
    utilization: Optional[float] = None      # 0.0-1.0  
    power_draw: Optional[float] = None       # Watts
    available: bool = False
    timestamp: float = 0.0                   # When this state was captured

class ContemplativeGPUMonitor:
    """
    Lightweight GPU stress monitor with adaptive breathing pauses.
    
    Philosophy: The AI breathes slower when hardware is under stress,
    embodying contemplative response to environmental pressure.
    
    Optimizations:
    - Cached GPU state (avoid constant subprocess calls)
    - Light monitoring mode for small models
    - Configurable thresholds and behavior
    """
    
    def __init__(self, 
                 cache_duration: float = 5.0,      # Cache GPU state for 5 seconds
                 light_mode: bool = False,         # Light monitoring (less frequent checks)
                 enabled: bool = True):            # Can disable entirely
        
        self.enabled = enabled
        self.light_mode = light_mode
        self.cache_duration = cache_duration
        
        # Breathing parameters (optimized for lighter overhead)
        self.baseline_sleep = 0.001 if not light_mode else 0.0005  # Even faster baseline
        self.stress_multiplier = 20.0 if not light_mode else 10.0  # Reduced multiplier  
        
        # Thresholds (more realistic for modern hardware)
        self.temp_threshold = 75.0           # °C - Most GPUs are fine up to 80-85°C
        self.memory_threshold = 0.85         # 85% memory usage (was 80%)
        self.util_threshold = 0.95           # 95% utilization (was 90%)
        
        # Cached state
        self._cached_state = GPUState()
        self._last_check_time = 0.0
        
        # Monitoring capability detection (only if enabled)
        if self.enabled:
            self.nvidia_smi_available = self._check_nvidia_smi()
            self.torch_cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
            
            print(f"🌬️ Contemplative GPU Monitor initialized:")
            print(f"   Mode: {'Light' if light_mode else 'Full'} monitoring")
            print(f"   Cache duration: {cache_duration}s")
            print(f"   NVIDIA-SMI: {self.nvidia_smi_available}")
            print(f"   PyTorch CUDA: {self.torch_cuda_available}")
        else:
            self.nvidia_smi_available = False
            self.torch_cuda_available = False
            print(f"🌬️ GPU Monitor disabled - using minimal static breathing")
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available (cached result)"""
        try:
            result = subprocess.run(['nvidia-smi', '--version'], 
                                  capture_output=True, text=True, timeout=3)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_gpu_state(self, force_refresh: bool = False) -> GPUState:
        """Get current GPU state with caching to reduce overhead"""
        
        if not self.enabled:
            return GPUState(available=False, timestamp=time.time())
        
        current_time = time.time()
        
        # Use cached state if recent enough (key optimization!)
        if not force_refresh and (current_time - self._last_check_time) < self.cache_duration:
            return self._cached_state
        
        # Refresh GPU state
        state = GPUState(timestamp=current_time)
        
        # Try nvidia-smi first (most comprehensive)
        if self.nvidia_smi_available:
            state = self._get_nvidia_smi_state()
        
        # Fallback to PyTorch CUDA info
        if not state.available and self.torch_cuda_available:
            state = self._get_torch_cuda_state()
        
        # Cache the result
        self._cached_state = state
        self._last_check_time = current_time
        
        return state
    
    def _get_nvidia_smi_state(self) -> GPUState:
        """Get GPU state via nvidia-smi (optimized query)"""
        try:
            # Optimized query - only get essential metrics
            cmd = [
                'nvidia-smi', 
                '--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)  # Faster timeout
            
            if result.returncode == 0:
                line = result.stdout.strip().split('\n')[0]  # First GPU
                values = [v.strip() for v in line.split(',')]
                
                if len(values) >= 4:
                    temp = float(values[0]) if values[0] != '[Not Supported]' else None
                    mem_used = float(values[1]) if values[1] != '[Not Supported]' else None
                    mem_total = float(values[2]) if values[2] != '[Not Supported]' else None
                    utilization = float(values[3]) if values[3] != '[Not Supported]' else None
                    
                    memory_ratio = mem_used / mem_total if (mem_used and mem_total and mem_total > 0) else None
                    
                    return GPUState(
                        temperature=temp,
                        memory_used=memory_ratio,
                        utilization=utilization / 100.0 if utilization else None,
                        available=True,
                        timestamp=time.time()
                    )
        
        except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
            if not self.light_mode:  # Only log in full mode
                print(f"⚠ nvidia-smi query failed: {e}")
        
        return GPUState(available=False, timestamp=time.time())
    
    def _get_torch_cuda_state(self) -> GPUState:
        """Get GPU state via PyTorch CUDA (limited info, fast)"""
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                # Memory info (fast operation)
                mem_total = torch.cuda.get_device_properties(device).total_memory
                mem_allocated = torch.cuda.memory_allocated(device)
                mem_cached = torch.cuda.memory_reserved(device)
                
                memory_ratio = max(mem_allocated, mem_cached) / mem_total
                
                return GPUState(
                    memory_used=memory_ratio,
                    available=True,
                    timestamp=time.time()
                )
        
        except Exception:
            pass  # Silent fail in optimized version
        
        return GPUState(available=False, timestamp=time.time())
    
    def calculate_stress_level(self, state: GPUState) -> float:
        """
        Calculate overall GPU stress level (0.0 = no stress, 1.0 = maximum stress)
        
        Optimized with more realistic thresholds.
        """
        if not state.available:
            return 0.0  # No monitoring = assume low stress
        
        stress_factors = []
        
        # Temperature stress (above 75°C threshold)
        if state.temperature is not None:
            temp_stress = max(0.0, (state.temperature - self.temp_threshold) / 15.0)  # 15°C range to 90°C
            stress_factors.append(min(1.0, temp_stress))
        
        # Memory stress (above 85% threshold)
        if state.memory_used is not None:
            mem_stress = max(0.0, (state.memory_used - self.memory_threshold) / 0.15)  # 15% range
            stress_factors.append(min(1.0, mem_stress))
        
        # Utilization stress (above 95% threshold)
        if state.utilization is not None:
            util_stress = max(0.0, (state.utilization - self.util_threshold) / 0.05)  # 5% range
            stress_factors.append(min(1.0, util_stress))
        
        # Return maximum stress factor (most conservative)
        return max(stress_factors) if stress_factors else 0.0
    
    def contemplative_pause(self, context: str = "training") -> float:
        """
        Perform adaptive contemplative pause based on GPU stress.
        
        Optimized for minimal overhead while still being contemplative.
        """
        if not self.enabled:
            return 0.0  # No breathing if disabled
        
        state = self.get_gpu_state()  # Uses caching automatically
        stress_level = self.calculate_stress_level(state)
        
        # Calculate adaptive sleep time
        sleep_time = self.baseline_sleep * (1.0 + stress_level * self.stress_multiplier)
        
        # Optimized logging - only log occasionally and when there's actually stress
        if stress_level > 0.1 or (hasattr(self, '_last_log_time') and 
                                 time.time() - self._last_log_time > 300):  # 5 minutes
            self._log_gpu_state(state, stress_level, sleep_time, context)
            self._last_log_time = time.time()
        
        # Perform the contemplative pause (if any)
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        return sleep_time
    
    def _log_gpu_state(self, state: GPUState, stress_level: float, sleep_time: float, context: str):
        """Log current GPU state and breathing response (optimized for key info only)"""
        if state.available and (stress_level > 0.05 or not self.light_mode):
            temp_str = f"{state.temperature:.0f}°C" if state.temperature else "N/A"
            mem_str = f"{state.memory_used:.1%}" if state.memory_used else "N/A"
            util_str = f"{state.utilization:.1%}" if state.utilization else "N/A"
            
            print(f"🌬️ GPU: {temp_str} Mem={mem_str} Util={util_str} "
                  f"→ Stress={stress_level:.2f} Sleep={sleep_time*1000:.1f}ms ({context})")
        elif not state.available and not self.light_mode:
            print(f"🌬️ GPU: No monitoring → Sleep={sleep_time*1000:.1f}ms ({context})")

# Global monitor instance (lazy initialization)
_gpu_monitor = None

def get_gpu_monitor(light_mode: bool = False, enabled: bool = True, 
                   cache_duration: float = 5.0) -> ContemplativeGPUMonitor:
    """Get global GPU monitor instance with configuration"""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = ContemplativeGPUMonitor(
            light_mode=light_mode, 
            enabled=enabled,
            cache_duration=cache_duration
        )
    return _gpu_monitor

def contemplative_pause(context: str = "training", 
                       light_mode: bool = False, 
                       enabled: bool = True) -> float:
    """
    Convenience function for adaptive contemplative breathing.
    
    Args:
        context: What operation is being performed
        light_mode: Use lightweight monitoring (for small models)
        enabled: Enable GPU monitoring at all (False = no breathing)
    
    Returns:
        float: Actual sleep time in seconds
    """
    monitor = get_gpu_monitor(light_mode=light_mode, enabled=enabled)
    return monitor.contemplative_pause(context)

# Convenience functions for different model scales
def femto_pause(context: str = "femto_training") -> float:
    """Optimized for femto-scale models (< 50K params) - minimal overhead"""
    return contemplative_pause(context, light_mode=True, enabled=False)  # Disabled for tiny models

def piko_pause(context: str = "piko_training") -> float:
    """Optimized for piko-scale models (50K-300K params) - light monitoring"""
    return contemplative_pause(context, light_mode=True, enabled=True)

def nano_pause(context: str = "nano_training") -> float:
    """Optimized for nano-scale models (300K-2M params) - full monitoring"""
    return contemplative_pause(context, light_mode=False, enabled=True)

def mili_pause(context: str = "mili_training") -> float:
    """Optimized for mili-scale models (2M+ params) - full monitoring with longer cache"""
    monitor = get_gpu_monitor(light_mode=False, enabled=True, cache_duration=10.0)  # Longer cache
    return monitor.contemplative_pause(context)

# Demo/test function
def demo_gpu_monitoring():
    """Demo the optimized GPU monitoring"""
    print("🧪 Optimized GPU Monitoring Demo - 5 samples over 15 seconds")
    
    # Test different modes
    modes = [
        ("femto", femto_pause),
        ("piko", piko_pause), 
        ("nano", nano_pause),
        ("mili", mili_pause)
    ]
    
    for mode_name, pause_func in modes:
        print(f"\n🔬 Testing {mode_name} mode:")
        for i in range(2):
            sleep_time = pause_func(f"{mode_name}_demo_{i+1}")
            print(f"  Sample {i+1}: Sleep={sleep_time*1000:.1f}ms")
            time.sleep(1)  # Wait between samples
    
    print("✅ Optimized demo complete!")

if __name__ == "__main__":
    demo_gpu_monitoring() 
#!/usr/bin/env python3
"""
GPU Contemplative Breathing Monitor

Adaptive time.sleep based on actual GPU load, temperature, and memory usage.
The AI breathes slower when the GPU is under stress - both practical and contemplative!
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

class ContemplativeGPUMonitor:
    """
    Monitors GPU stress and provides adaptive breathing pauses.
    
    Philosophy: The AI breathes slower when hardware is under stress,
    embodying contemplative response to environmental pressure.
    """
    
    def __init__(self):
        self.baseline_sleep = 0.001          # Minimum breathing pause (1ms)
        self.stress_multiplier = 50.0        # Max multiplier for high stress
        self.temp_threshold = 70.0           # °C - Start increasing sleep above this
        self.memory_threshold = 0.8          # 80% memory usage threshold
        self.util_threshold = 0.9            # 90% utilization threshold
        
        # Try to detect GPU monitoring capability
        self.nvidia_smi_available = self._check_nvidia_smi()
        self.torch_cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
        
        print(f"🌬️ Contemplative GPU Monitor initialized:")
        print(f"   NVIDIA-SMI available: {self.nvidia_smi_available}")
        print(f"   PyTorch CUDA available: {self.torch_cuda_available}")
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            result = subprocess.run(['nvidia-smi', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_gpu_state(self) -> GPUState:
        """Get current GPU state using available monitoring tools"""
        state = GPUState()
        
        # Try nvidia-smi first (most comprehensive)
        if self.nvidia_smi_available:
            state = self._get_nvidia_smi_state()
        
        # Fallback to PyTorch CUDA info
        if not state.available and self.torch_cuda_available:
            state = self._get_torch_cuda_state()
        
        return state
    
    def _get_nvidia_smi_state(self) -> GPUState:
        """Get GPU state via nvidia-smi"""
        try:
            cmd = [
                'nvidia-smi', 
                '--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            
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
                        available=True
                    )
        
        except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
            print(f"⚠ nvidia-smi query failed: {e}")
        
        return GPUState(available=False)
    
    def _get_torch_cuda_state(self) -> GPUState:
        """Get GPU state via PyTorch CUDA (limited info)"""
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                # Memory info
                mem_total = torch.cuda.get_device_properties(device).total_memory
                mem_allocated = torch.cuda.memory_allocated(device)
                mem_cached = torch.cuda.memory_reserved(device)
                
                memory_ratio = max(mem_allocated, mem_cached) / mem_total
                
                return GPUState(
                    memory_used=memory_ratio,
                    available=True
                )
        
        except Exception as e:
            print(f"⚠ PyTorch CUDA query failed: {e}")
        
        return GPUState(available=False)
    
    def calculate_stress_level(self, state: GPUState) -> float:
        """
        Calculate overall GPU stress level (0.0 = no stress, 1.0 = maximum stress)
        
        Combines temperature, memory usage, and utilization into single stress metric.
        """
        if not state.available:
            return 0.0  # No monitoring = assume low stress
        
        stress_factors = []
        
        # Temperature stress (above threshold)
        if state.temperature is not None:
            temp_stress = max(0.0, (state.temperature - self.temp_threshold) / 20.0)  # 20°C range
            stress_factors.append(min(1.0, temp_stress))
        
        # Memory stress (above threshold)
        if state.memory_used is not None:
            mem_stress = max(0.0, (state.memory_used - self.memory_threshold) / 0.2)  # 20% range
            stress_factors.append(min(1.0, mem_stress))
        
        # Utilization stress (above threshold)
        if state.utilization is not None:
            util_stress = max(0.0, (state.utilization - self.util_threshold) / 0.1)  # 10% range
            stress_factors.append(min(1.0, util_stress))
        
        # Return maximum stress factor (most conservative)
        return max(stress_factors) if stress_factors else 0.0
    
    def contemplative_pause(self, context: str = "training") -> float:
        """
        Perform adaptive contemplative pause based on GPU stress.
        
        Returns the actual sleep time used for logging.
        """
        state = self.get_gpu_state()
        stress_level = self.calculate_stress_level(state)
        
        # Calculate adaptive sleep time
        sleep_time = self.baseline_sleep * (1.0 + stress_level * self.stress_multiplier)
        
        # Log occasionally for visibility
        if hasattr(self, '_last_log_time'):
            if time.time() - self._last_log_time > 30:  # Log every 30 seconds
                self._log_gpu_state(state, stress_level, sleep_time, context)
                self._last_log_time = time.time()
        else:
            self._log_gpu_state(state, stress_level, sleep_time, context)
            self._last_log_time = time.time()
        
        # Perform the contemplative pause
        time.sleep(sleep_time)
        
        return sleep_time
    
    def _log_gpu_state(self, state: GPUState, stress_level: float, sleep_time: float, context: str):
        """Log current GPU state and breathing response"""
        if state.available:
            temp_str = f"{state.temperature:.1f}°C" if state.temperature else "N/A"
            mem_str = f"{state.memory_used:.1%}" if state.memory_used else "N/A"
            util_str = f"{state.utilization:.1%}" if state.utilization else "N/A"
            
            print(f"🌬️ GPU Breathing ({context}): Temp={temp_str} Memory={mem_str} Util={util_str} "
                  f"→ Stress={stress_level:.2f} Sleep={sleep_time*1000:.1f}ms")
        else:
            print(f"🌬️ GPU Breathing ({context}): No monitoring → Sleep={sleep_time*1000:.1f}ms")

# Global monitor instance
_gpu_monitor = None

def get_gpu_monitor() -> ContemplativeGPUMonitor:
    """Get global GPU monitor instance"""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = ContemplativeGPUMonitor()
    return _gpu_monitor

def contemplative_pause(context: str = "training") -> float:
    """
    Convenience function for adaptive contemplative breathing.
    
    Usage:
        contemplative_pause("training_batch")  # Adaptive pause based on GPU stress
    
    Returns:
        float: Actual sleep time in seconds
    """
    monitor = get_gpu_monitor()
    return monitor.contemplative_pause(context)

# Demo/test function
def demo_gpu_monitoring():
    """Demo the GPU monitoring and adaptive breathing"""
    monitor = ContemplativeGPUMonitor()
    
    print("🧪 GPU Monitoring Demo - 10 samples over 30 seconds")
    
    for i in range(10):
        state = monitor.get_gpu_state()
        stress = monitor.calculate_stress_level(state)
        sleep_time = monitor.contemplative_pause(f"demo_sample_{i+1}")
        
        print(f"Sample {i+1}: Stress={stress:.2f}, Sleep={sleep_time*1000:.1f}ms")
        time.sleep(3)  # Wait between samples
    
    print("✅ Demo complete!")

if __name__ == "__main__":
    demo_gpu_monitoring() 
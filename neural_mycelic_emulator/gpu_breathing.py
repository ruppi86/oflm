from __future__ import annotations
"""Lightweight GPU breathing monitor (stand-alone copy).

Origin: `oflm-python/spiramycel/gpu_breathing.py` – trimmed to essentials so
`neural_mycelic_emulator` can run independently from Spiramycel.
"""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

authors = "ported by o3 – 2025-07-02"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class GPUState:
    temperature: Optional[float] = None  # °C
    memory_used: Optional[float] = None  # 0-1 fraction
    utilization: Optional[float] = None  # 0-1 fraction
    available: bool = False
    timestamp: float = 0.0

# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------
class GPUMonitor:
    """Query `nvidia-smi` or torch.cuda and return stress level."""

    def __init__(self, cache_sec: float = 5.0):
        self.cache_sec = cache_sec
        self._cached: GPUState = GPUState()
        self._last = 0.0
        self._has_nvidia_smi = self._check_smi()
        self._has_cuda = TORCH_AVAILABLE and torch.cuda.is_available()

        # thresholds
        self.TEMP = 75.0
        self.MEM = 0.85
        self.UTIL = 0.95

    # ------------------------------------------------------------------
    def _check_smi(self) -> bool:
        try:
            r = subprocess.run(["nvidia-smi", "--version"], capture_output=True, timeout=3)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):  # pragma: no cover
            return False

    def _query(self) -> GPUState:
        if self._has_nvidia_smi:
            try:
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                if r.returncode == 0 and r.stdout:
                    t, mu, mt, util = [float(x) for x in r.stdout.strip().split(",")[:4]]
                    return GPUState(
                        temperature=t,
                        memory_used=mu / mt if mt else None,
                        utilization=util / 100.0,
                        available=True,
                        timestamp=time.time(),
                    )
            except Exception:  # pragma: no cover
                pass
        if self._has_cuda:
            try:
                dev = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(dev)
                mem_ratio = torch.cuda.memory_reserved(dev) / props.total_memory
                return GPUState(memory_used=mem_ratio, available=True, timestamp=time.time())
            except Exception:  # pragma: no cover
                pass
        return GPUState(available=False, timestamp=time.time())

    def state(self) -> GPUState:
        now = time.time()
        if now - self._last > self.cache_sec:
            self._cached = self._query()
            self._last = now
        return self._cached

    # ------------------------------------------------------------------
    def stress(self) -> float:
        s = self.state()
        factors = []
        if s.temperature is not None:
            factors.append(max(0.0, (s.temperature - self.TEMP) / 15.0))
        if s.memory_used is not None:
            factors.append(max(0.0, (s.memory_used - self.MEM) / 0.15))
        if s.utilization is not None:
            factors.append(max(0.0, (s.utilization - self.UTIL) / 0.05))
        return max(factors) if factors else 0.0

# ---------------------------------------------------------------------------
# Convenience pause functions
# ---------------------------------------------------------------------------
_monitor: GPUMonitor | None = None

def _get() -> GPUMonitor:
    global _monitor
    if _monitor is None:
        _monitor = GPUMonitor()
    return _monitor

def adaptive_pause(context: str = "training", base: float = 0.001, mult: float = 20.0) -> float:
    """Sleep `base * (1 + stress*mult)` seconds; return sleep time."""
    stress = _get().stress()
    sleep_t = base * (1.0 + stress * mult)
    time.sleep(sleep_t)
    if stress > 0.1:
        print(f"🌬️ GPU stress {stress:.2f} → sleep {sleep_t*1000:.1f} ms ({context})")
    return sleep_t

# shortcuts matching original names
femto_pause = lambda c="training": adaptive_pause(c, base=0.0005, mult=0.0)
piko_pause  = lambda c="training": adaptive_pause(c, base=0.0008, mult=5.0)
nano_pause  = lambda c="training": adaptive_pause(c, base=0.001,  mult=20.0)
mili_pause  = lambda c="training": adaptive_pause(c, base=0.002,  mult=30.0)

if __name__ == "__main__":  # quick cli demo
    for name, fn in [
        ("femto", femto_pause),
        ("piko", piko_pause),
        ("nano", nano_pause),
        ("mili", mili_pause),
    ]:
        print(f"\n{name} breathing samples:")
        for i in range(3):
            fn(f"{name}_{i+1}")
            time.sleep(0.5) 
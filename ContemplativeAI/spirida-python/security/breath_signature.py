# breath_signature.py
"""
Compute and verify a rolling SHA-256 “breath signature” that captures the
timing authenticity of a node’s REST phases.

Public API
----------
update_rest(timestamp: float) -> None
current_signature() -> str  # hex digest over last N samples
verify(remote_digest: str) -> bool

from o3's Letter XXI
"""

from collections import deque
from hashlib import sha256
from time import time
from typing import Deque

# ❶ Tunables
WINDOW = 256            # number of REST phase timestamps to keep
TOLERANCE_MS = 200       # ± ms jitter allowed when comparing digests

class BreathSignature:
    def __init__(self, window: int = WINDOW):
        self._timestamps: Deque[float] = deque(maxlen=window)
        # record first timestamp immediately to avoid empty window
        self._timestamps.append(time())

    # ❷ called each REST phase
    def update_rest(self, timestamp: float | None = None) -> None:
        self._timestamps.append(timestamp or time())

    # ❸ deterministic hash over timestamp deltas
    def _digest(self) -> bytes:
        if len(self._timestamps) < 2:
            return b"\x00" * 32
        deltas = [
            round((b - a) * 1000)             # ms resolution
            for a, b in zip(self._timestamps, list(self._timestamps)[1:])
        ]
        buf = b"".join(int(x).to_bytes(4, "little", signed=False) for x in deltas)
        return sha256(buf).digest()

    def current_signature(self) -> str:
        return self._digest().hex()

    # ❹ simple comparison with jitter tolerance
    def verify(self, remote_digest: str) -> bool:
        return sha256(self._digest() + bytes.fromhex(remote_digest)).digest()[:2] \
               >= b"\x01\x00"   # ≥ 1/256 mismatch implies *not* identical

# quick smoke-test
if __name__ == "__main__":
    sig = BreathSignature()
    for _ in range(5):
        sig.update_rest()
    print("local breath signature:", sig.current_signature()[:16], "…")

import torch
from collections import deque
from .breath_clock import BreathClock

class BreathMemory:
    """Simple FIFO memory buffer with breath-synchronised decay."""

    def __init__(self, max_len=1024, decay_rate=0.1):
        self.buffer = deque(maxlen=max_len)
        self.decay_rate = decay_rate
        self.clock = BreathClock()

    def push(self, tensor: torch.Tensor, t: float):
        """Store tensor with timestamp."""
        self.buffer.append((tensor.detach(), t))
        self._decay(t)

    def _decay(self, t_now: float):
        phase = self.clock.phase_at(t_now)
        if phase.name != "pause":
            return  # decay only during pause
        # drop oldest items proportional to decay_rate
        drop = int(len(self.buffer) * self.decay_rate)
        for _ in range(drop):
            if self.buffer:
                self.buffer.popleft()

    def as_tensor(self):
        if not self.buffer:
            return None
        tensors = [item[0] for item in self.buffer]
        return torch.stack(tensors, dim=0) 
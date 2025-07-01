# slow_start_middleware.py
"""
Decorator for a coroutine that receives BIP packets.

It forces a 'slow-start' handshake:
1) When two unfamiliar agents meet, they must complete five
   synchronous breath cycles before any SYMBOL payloads are accepted.
2) Until then, only REST-phase BIP packets are forwarded upstream.

from o3's Letter XXI
"""

import asyncio
from collections import defaultdict
from typing import Awaitable, Callable

BREATHS_REQUIRED = 5

PeerID = str
Packet = dict  # {'agent_id': str, 'phase': 'REST'|'INHALE'|… , ...}

def slow_start(listener: Callable[[Packet, PeerID], Awaitable[None]]):
    peer_state = defaultdict(int)    # peer_id -> witnessed synchronous cycles

    async def wrapper(packet: Packet, addr: tuple[str, int]):
        peer = packet.get("agent_id") or f"{addr[0]}:{addr[1]}"
        phase = packet.get("phase")

        if peer_state[peer] >= BREATHS_REQUIRED:
            # handshake done – forward everything
            await listener(packet, addr)
            return

        if phase != "REST":
            # ignore non-REST until handshake complete
            return

        # count synchronous REST events
        peer_state[peer] += 1
        if peer_state[peer] == BREATHS_REQUIRED:
            print(f"[slow-start] Peer {peer} authenticated after "
                  f"{BREATHS_REQUIRED} shared breaths.")
        await listener(packet, addr)   # still forward REST packets

    return wrapper

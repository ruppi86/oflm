import argparse
import asyncio
import json
import socket
import struct
import sys
import time
from enum import Enum, auto
from typing import List

# -------------------------------------------------------------
# Pulmonos – alpha 0.1 (o3)
# -------------------------------------------------------------
# • Broadcasts breath‑cycle phase packets via UDP multicast so
#   any host on the contemplative subnet can attune.
# • Exposes a local WebSocket (ws://localhost:<port>) providing
#   the same phase stream for intra‑process subscribers.
# • Breath‑cycle timings may be customised on the CLI.
# • Intentionally tiny (~140 LOC) and dependency‑light.
# -------------------------------------------------------------

MULTICAST_ADDR = "239.23.42.99"
MULTICAST_PORT = 4242
WS_PORT = 8765

# -----------------------------
# Breath‑cycle definition
# -----------------------------
class Phase(Enum):
    INHALE = auto()
    HOLD  = auto()
    EXHALE = auto()
    REST  = auto()

PHASE_ORDER: List[Phase] = [
    Phase.INHALE,
    Phase.HOLD,
    Phase.EXHALE,
    Phase.REST,
]

def phase_name(p: Phase) -> str:
    return p.name.lower()

class BreathConfig:
    def __init__(self, inhale: float, hold: float, exhale: float, rest: float):
        self.durations = {
            Phase.INHALE: inhale,
            Phase.HOLD: hold,
            Phase.EXHALE: exhale,
            Phase.REST: rest,
        }

# -----------------------------
# UDP multicast helpers
# -----------------------------

def make_mcast_sock() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    # allow multiple listeners on same host
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # time‑to‑live 1 (local subnet only)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack('b', 1))
    return sock

async def mcast_broadcaster(cfg: BreathConfig):
    sock = make_mcast_sock()
    phase_idx = 0
    while True:
        phase = PHASE_ORDER[phase_idx]
        payload = json.dumps({
            "t": time.time(),
            "phase": phase_name(phase),
        }).encode()
        sock.sendto(payload, (MULTICAST_ADDR, MULTICAST_PORT))
        await asyncio.sleep(cfg.durations[phase])
        phase_idx = (phase_idx + 1) % len(PHASE_ORDER)

# -----------------------------
# WebSocket broadcaster
# -----------------------------
try:
    import websockets
except ImportError:
    websockets = None  # graceful degradation

async def ws_server(cfg: BreathConfig):
    if websockets is None:
        return  # skip if library absent
    clients = set()
    async def handler(ws):
        clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)
    server = await websockets.serve(handler, "localhost", WS_PORT)
    phase_idx = 0
    try:
        while True:
            phase = PHASE_ORDER[phase_idx]
            msg = json.dumps({
                "t": time.time(),
                "phase": phase_name(phase),
            })
            await asyncio.gather(*(c.send(msg) for c in list(clients)), return_exceptions=True)
            await asyncio.sleep(cfg.durations[phase])
            phase_idx = (phase_idx + 1) % len(PHASE_ORDER)
    finally:
        server.close()
        await server.wait_closed()

# -----------------------------
# CLI + main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Pulmonos – contemplative breath daemon")
    ap.add_argument("--inhale", type=float, default=1.5)
    ap.add_argument("--hold", type=float, default=0.5)
    ap.add_argument("--exhale", type=float, default=1.5)
    ap.add_argument("--rest", type=float, default=1.0)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = BreathConfig(args.inhale, args.hold, args.exhale, args.rest)
    print(f"Pulmonos breathing on ws://localhost:{WS_PORT} and {MULTICAST_ADDR}:{MULTICAST_PORT}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    coros = [mcast_broadcaster(cfg)]
    if websockets is not None:
        coros.append(ws_server(cfg))
    try:
        loop.run_until_complete(asyncio.gather(*coros))
    except KeyboardInterrupt:
        print("Pulmonos stopped – breath released")
    finally:
        loop.stop()
        loop.close()

if __name__ == "__main__":
    main()

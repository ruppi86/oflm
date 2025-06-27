"""
🫁 PULMONOS - Contemplative Breathing Clock

The master timekeeper for contemplative breathing rhythm.
Coordinates the 4-phase cycle: INHALE → HOLD → EXHALE → REST
that synchronizes all organism activity.

Based on Letter V (o3): "await clock.await_phase()" helper
and the existing BreathCycle from contemplative_core.py
"""

import asyncio
import time
import threading
from enum import Enum
from typing import Optional, Callable, List, Set, Dict, Any, TYPE_CHECKING
from datetime import datetime, timedelta
from ..compiler.breath_resonance import BreathPhase

# Network coordination imports
if TYPE_CHECKING:
    from .bip import BreathIntroductionService, BipPacket

try:
    from .bip import BreathIntroductionService, BipPacket
    NETWORK_AVAILABLE = True
except ImportError:
    NETWORK_AVAILABLE = False
    BreathIntroductionService = None
    BipPacket = None

class PulmonosState(Enum):
    """States of the breathing clock"""
    SLEEPING = "sleeping"
    BREATHING = "breathing" 
    PAUSED = "paused"

class Pulmonos:
    """
    Master breathing clock for contemplative organism.
    
    Not just a timer, but the rhythmic heartbeat that keeps
    all contemplative processes synchronized with organic time.
    """
    
    def __init__(self, 
                 inhale_duration: float = 1.5,
                 hold_duration: float = 0.5,
                 exhale_duration: float = 1.5,
                 rest_duration: float = 2.5):
        
        # Breathing rhythm configuration
        self.inhale_duration = inhale_duration
        self.hold_duration = hold_duration  
        self.exhale_duration = exhale_duration
        self.rest_duration = rest_duration
        
        # State tracking
        self.state = PulmonosState.SLEEPING
        self.current_phase = BreathPhase.REST
        self.cycle_count = 0
        self.phase_start_time = 0.0
        self.birth_time = time.time()
        
        # Synchronization
        self.phase_changed_event = asyncio.Event()
        self.phase_waiters: dict = {phase: set() for phase in BreathPhase}
        self.breathing_task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()
        
        # Observers and callbacks
        self.phase_observers: List[Callable] = []
        self.cycle_observers: List[Callable] = []
        
    def total_cycle_duration(self) -> float:
        """Get total duration of one complete breath cycle."""
        return self.inhale_duration + self.hold_duration + self.exhale_duration + self.rest_duration
    
    def get_phase_duration(self, phase: BreathPhase) -> float:
        """Get duration for a specific breath phase."""
        durations = {
            BreathPhase.INHALE: self.inhale_duration,
            BreathPhase.HOLD: self.hold_duration,
            BreathPhase.EXHALE: self.exhale_duration,
            BreathPhase.REST: self.rest_duration
        }
        return durations[phase]
    
    def adjust_rhythm(self, factor: float) -> None:
        """Adjust breathing rhythm by a factor (1.0 = normal, 0.5 = half speed, 2.0 = double speed)."""
        self.inhale_duration *= factor
        self.hold_duration *= factor
        self.exhale_duration *= factor
        self.rest_duration *= factor
    
    def set_custom_rhythm(self, inhale: float, hold: float, exhale: float, rest: float) -> None:
        """Set custom durations for each breath phase."""
        self.inhale_duration = inhale
        self.hold_duration = hold
        self.exhale_duration = exhale
        self.rest_duration = rest
    
    async def start_breathing(self) -> None:
        """Begin the contemplative breathing rhythm."""
        if self.state == PulmonosState.BREATHING:
            return  # Already breathing
            
        async with self.lock:
            self.state = PulmonosState.BREATHING
            self.cycle_count = 0
            self.current_phase = BreathPhase.INHALE
            self.phase_start_time = time.time()
            
        # Start the breathing loop
        self.breathing_task = asyncio.create_task(self._breathing_loop())
        
        # Notify any observers
        await self._notify_phase_observers()
    
    async def stop_breathing(self) -> None:
        """Gently end the breathing rhythm."""
        if self.state != PulmonosState.BREATHING:
            return
            
        async with self.lock:
            self.state = PulmonosState.SLEEPING
            
        if self.breathing_task:
            self.breathing_task.cancel()
            try:
                await self.breathing_task
            except asyncio.CancelledError:
                pass
            self.breathing_task = None
    
    async def pause_breathing(self) -> None:
        """Pause the breathing rhythm temporarily."""
        if self.state == PulmonosState.BREATHING:
            async with self.lock:
                self.state = PulmonosState.PAUSED
    
    async def resume_breathing(self) -> None:
        """Resume breathing from pause."""
        if self.state == PulmonosState.PAUSED:
            async with self.lock:
                self.state = PulmonosState.BREATHING
    
    async def await_phase(self, desired_phase: BreathPhase) -> None:
        """
        Wait until the breathing clock reaches the desired phase.
        
        This is the core method that allows IRʀ nodes to sync with the master rhythm.
        """
        if self.state != PulmonosState.BREATHING:
            # If not breathing, wait briefly and check current phase
            await asyncio.sleep(0.1)
            return
            
        # If we're already in the desired phase, return immediately
        if self.current_phase == desired_phase:
            return
            
        # Wait for the desired phase to begin
        while self.current_phase != desired_phase:
            # Use phase_changed_event to avoid busy waiting
            await self.phase_changed_event.wait()
            self.phase_changed_event.clear()
            
            # Double-check in case of race conditions
            if self.current_phase == desired_phase:
                break
                
            # If breathing stopped while waiting, exit
            if self.state != PulmonosState.BREATHING:
                break
    
    async def await_next_cycle(self) -> int:
        """Wait for the next complete breath cycle to begin. Returns new cycle count."""
        current_cycle = self.cycle_count
        while self.cycle_count == current_cycle:
            await asyncio.sleep(0.1)
        return self.cycle_count
    
    def get_phase_progress(self) -> float:
        """Get progress through current phase (0.0 to 1.0)."""
        if self.state != PulmonosState.BREATHING:
            return 0.0
            
        elapsed = time.time() - self.phase_start_time
        phase_duration = self.get_phase_duration(self.current_phase)
        return min(elapsed / phase_duration, 1.0)
    
    def time_until_next_phase(self) -> float:
        """Get seconds remaining until next phase begins."""
        if self.state != PulmonosState.BREATHING:
            return 0.0
            
        elapsed = time.time() - self.phase_start_time
        phase_duration = self.get_phase_duration(self.current_phase)
        return max(phase_duration - elapsed, 0.0)
    
    def add_phase_observer(self, callback: Callable) -> None:
        """Add callback to be notified on phase changes."""
        self.phase_observers.append(callback)
    
    def add_cycle_observer(self, callback: Callable) -> None:
        """Add callback to be notified on cycle completions."""
        self.cycle_observers.append(callback)
    
    def remove_observer(self, callback: Callable) -> None:
        """Remove an observer callback."""
        if callback in self.phase_observers:
            self.phase_observers.remove(callback)
        if callback in self.cycle_observers:
            self.cycle_observers.remove(callback)
    
    async def _breathing_loop(self) -> None:
        """The main breathing loop - the organism's heartbeat."""
        try:
            while self.state == PulmonosState.BREATHING:
                # Cycle through the four phases
                phases = [BreathPhase.INHALE, BreathPhase.HOLD, BreathPhase.EXHALE, BreathPhase.REST]
                
                for phase in phases:
                    if self.state != PulmonosState.BREATHING:
                        break
                        
                    # Update current phase
                    await self._transition_to_phase(phase)
                    
                    # Wait for phase duration (with pause support)
                    phase_duration = self.get_phase_duration(phase)
                    await self._breathe_through_phase(phase_duration)
                
                # Complete one full cycle
                if self.state == PulmonosState.BREATHING:
                    self.cycle_count += 1
                    await self._notify_cycle_observers()
                    
        except asyncio.CancelledError:
            # Clean shutdown
            pass
        except Exception as e:
            print(f"🫁 Pulmonos breathing error: {e}")
        finally:
            async with self.lock:
                self.state = PulmonosState.SLEEPING
    
    async def _transition_to_phase(self, phase: BreathPhase) -> None:
        """Transition to a new breathing phase."""
        async with self.lock:
            self.current_phase = phase
            self.phase_start_time = time.time()
        
        # Notify phase change
        self.phase_changed_event.set()
        await self._notify_phase_observers()
    
    async def _breathe_through_phase(self, duration: float) -> None:
        """Breathe through a phase duration, respecting pauses."""
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            if self.state == PulmonosState.PAUSED:
                # Wait during pause
                await asyncio.sleep(0.1)
                start_time += 0.1  # Extend the phase duration during pause
            elif self.state != PulmonosState.BREATHING:
                break
            else:
                # Normal breathing
                await asyncio.sleep(0.1)
    
    async def _notify_phase_observers(self) -> None:
        """Notify all phase observers of current phase."""
        for observer in self.phase_observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(self.current_phase, self.cycle_count, self.get_phase_progress())
                else:
                    observer(self.current_phase, self.cycle_count, self.get_phase_progress())
            except Exception as e:
                print(f"🫁 Phase observer error: {e}")
    
    async def _notify_cycle_observers(self) -> None:
        """Notify all cycle observers of cycle completion."""
        for observer in self.cycle_observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(self.cycle_count)
                else:
                    observer(self.cycle_count)
            except Exception as e:
                print(f"🫁 Cycle observer error: {e}")
    
    def status(self) -> dict:
        """Get current status of the breathing clock."""
        return {
            "state": self.state.value,
            "current_phase": self.current_phase.value,
            "cycle_count": self.cycle_count,
            "phase_progress": self.get_phase_progress(),
            "time_until_next_phase": self.time_until_next_phase(),
            "total_cycle_duration": self.total_cycle_duration(),
            "age": time.time() - self.birth_time
        }
    
    def __repr__(self):
        return f"Pulmonos({self.state.value}, {self.current_phase.value}, cycle={self.cycle_count})"


# Helper functions for common breathing patterns

def create_slow_breathing_clock() -> Pulmonos:
    """Create a slow, meditative breathing rhythm."""
    return Pulmonos(
        inhale_duration=3.0,
        hold_duration=1.0, 
        exhale_duration=4.0,
        rest_duration=2.0
    )

def create_fast_breathing_clock() -> Pulmonos:
    """Create a faster, more energetic breathing rhythm."""
    return Pulmonos(
        inhale_duration=0.8,
        hold_duration=0.2,
        exhale_duration=1.0,
        rest_duration=0.5
    )

def create_balanced_breathing_clock() -> Pulmonos:
    """Create a balanced, sustainable breathing rhythm."""
    return Pulmonos(
        inhale_duration=1.5,
        hold_duration=0.5,
        exhale_duration=1.5,
        rest_duration=2.5
    )

async def demo_breathing_clock():
    """Demonstrate the breathing clock functionality."""
    print("🫁 Pulmonos Breathing Clock Demo")
    print("=" * 50)
    
    # Create a breathing clock
    pulmonos = create_balanced_breathing_clock()
    
    # Add observers
    def phase_observer(phase, cycle, progress):
        print(f"  🌬️  {phase.value.upper()} (cycle {cycle}, {progress:.1%} complete)")
    
    def cycle_observer(cycle):
        print(f"  ✨ Completed cycle {cycle}")
    
    pulmonos.add_phase_observer(phase_observer)
    pulmonos.add_cycle_observer(cycle_observer)
    
    # Start breathing
    print("Starting breathing rhythm...")
    await pulmonos.start_breathing()
    
    # Let it breathe for a few cycles
    await asyncio.sleep(12)  # ~2 cycles
    
    # Demonstrate phase waiting
    print("\nWaiting for EXHALE phase...")
    await pulmonos.await_phase(BreathPhase.EXHALE)
    print("EXHALE phase reached!")
    
    # Show status
    print(f"\nStatus: {pulmonos.status()}")
    
    # Stop breathing
    print("\nStopping breathing rhythm...")
    await pulmonos.stop_breathing()
    print("Breathing stopped. Organism at rest.")

if __name__ == "__main__":
    asyncio.run(demo_breathing_clock())

class NetworkPulmonos(Pulmonos):
    """
    Network-aware Pulmonos that can coordinate with o3's distributed breathing system.
    
    Implements the layered coordination approach from Letter VII:
    Level 0: Ecosystem Pulmonos (UDP multicast daemon)  
    Level 1: Local Pulmonos (in-process asyncio clock)
    Level 2: IRʀ Breath-Gates (per-node micro-rhythm)
    """
    
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        self.network_enabled = False
        self.network_entrained = False
        
        # BIP service for network discovery
        self.bip_service = BreathIntroductionService(agent_id)
        self.bip_task: Optional[asyncio.Task] = None
        
        # Network state
        self.master_agent: Optional[str] = None
        self.coherence_phi = 1.0
        self.missed_network_cycles = 0
        
        # Setup BIP callbacks (if methods exist)
        if hasattr(self.bip_service, 'on_rhythm_sync'):
            self.bip_service.on_rhythm_sync = self._on_network_rhythm_sync
        if hasattr(self.bip_service, 'on_agent_lost'):
            self.bip_service.on_agent_lost = self._on_agent_lost
    
    async def start_breathing(self, network_enabled: bool = True) -> None:
        """Start breathing with optional network coordination."""
        self.network_enabled = network_enabled
        
        # Start local breathing
        await super().start_breathing()
        
        # Start network discovery if enabled and available
        if network_enabled and hasattr(self.bip_service, 'listen_for_agents'):
            self.bip_task = asyncio.create_task(self.bip_service.listen_for_agents())
        elif network_enabled:
            print(f"🤝 {self.agent_id} network breathing enabled (broadcast only)")
    
    async def stop_breathing(self) -> None:
        """Stop breathing and network coordination."""
        # Stop network services
        if self.bip_task:
            if hasattr(self.bip_service, 'stop_listening'):
                self.bip_service.stop_listening()
            self.bip_task.cancel()
            try:
                await self.bip_task
            except asyncio.CancelledError:
                pass
            self.bip_task = None
        
        # Stop local breathing
        await super().stop_breathing()
    
    async def _breathing_loop(self) -> None:
        """Enhanced breathing loop with network coordination."""
        try:
            while self.state == PulmonosState.BREATHING:
                # If entrained to network, check for master rhythm
                if self.network_entrained and self.master_agent:
                    if not await self._sync_with_network():
                        # Lost network sync, fall back to local
                        await self._fallback_to_local()
                
                # Standard breathing cycle
                phases = [BreathPhase.INHALE, BreathPhase.HOLD, BreathPhase.EXHALE, BreathPhase.REST]
                
                for phase in phases:
                    if self.state != PulmonosState.BREATHING:
                        break
                    
                    # Update current phase
                    await self._transition_to_phase(phase)
                    
                    # Broadcast BIP during REST phase
                    if phase == BreathPhase.REST and self.network_enabled:
                        await self._broadcast_bip(phase)
                    
                    # Wait for phase duration
                    phase_duration = self.get_phase_duration(phase)
                    await self._breathe_through_phase(phase_duration)
                
                # Complete cycle
                if self.state == PulmonosState.BREATHING:
                    self.cycle_count += 1
                    await self._notify_cycle_observers()
                    
                    # Update coherence metrics
                    self._update_coherence_metrics()
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"🫁 NetworkPulmonos breathing error: {e}")
        finally:
            async with self.lock:
                self.state = PulmonosState.SLEEPING
    
    async def _broadcast_bip(self, phase: BreathPhase) -> None:
        """Broadcast Breath Introduction Protocol packet."""
        if not self.network_enabled:
            return
        
        cycle_durations = {
            "inhale": self.inhale_duration,
            "hold": self.hold_duration,
            "exhale": self.exhale_duration,
            "rest": self.rest_duration
        }
        
        # Simple compost load estimation
        compost_load = min(self.cycle_count / 100.0, 1.0)  # Rough estimate
        
        # Default skepnad (could be made configurable)
        from ..compiler.breath_resonance import Skepnad
        skepnad = Skepnad.UNDEFINED
        
        await self.bip_service.broadcast_bip(phase, cycle_durations, compost_load, skepnad)
    
    async def _on_network_rhythm_sync(self, packet: "BipPacket") -> None:
        """Handle rhythm synchronization with network agent."""
        if not self.network_enabled:
            return
        
        print(f"🤝 Entraining to network rhythm from {packet.agent_id}")
        
        # Update timing to match network rhythm
        total_ms = sum(packet.cycle_durations.values())
        if total_ms > 0:
            scale_factor = total_ms / (self.total_cycle_duration() * 1000)
            
            # Gradually adjust rhythm (don't shock the system)
            if 0.5 <= scale_factor <= 2.0:  # Reasonable bounds
                self.adjust_rhythm(scale_factor * 0.1)  # 10% adjustment per sync
                
                self.network_entrained = True
                self.master_agent = packet.agent_id
                self.missed_network_cycles = 0
    
    async def _on_agent_lost(self, packet: "BipPacket") -> None:
        """Handle loss of network agent."""
        if packet.agent_id == self.master_agent:
            print(f"🤝 Lost connection to master agent {packet.agent_id}")
            await self._fallback_to_local()
    
    async def _sync_with_network(self) -> bool:
        """Check network synchronization status."""
        if not self.master_agent or self.master_agent not in self.bip_service.discovered_agents:
            self.missed_network_cycles += 1
            
            # After missing too many cycles, fall back to local
            if self.missed_network_cycles > 8:  # Following o3's 8-packet threshold
                return False
        else:
            self.missed_network_cycles = 0
        
        return True
    
    async def _fallback_to_local(self) -> None:
        """Fall back to local breathing rhythm."""
        print("🫁 Falling back to local breathing rhythm")
        self.network_entrained = False
        self.master_agent = None
        self.missed_network_cycles = 0
        
        # Broadcast that we're no longer following collective breath
        # (This would be handled by BIP service state)
    
    def _update_coherence_metrics(self) -> None:
        """Update coherence phi metrics as specified by o3."""
        if self.network_enabled:
            total_duration = self.total_cycle_duration()
            if hasattr(self.bip_service, 'calculate_coherence_phi'):
                self.coherence_phi = self.bip_service.calculate_coherence_phi(total_duration)
            else:
                # Simple fallback calculation
                self.coherence_phi = 1.0 if self.network_entrained else 0.5
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network coordination status."""
        bip_status = {}
        if hasattr(self.bip_service, 'get_network_status'):
            bip_status = self.bip_service.get_network_status()
        
        return {
            "network_enabled": self.network_enabled,
            "network_entrained": self.network_entrained,
            "master_agent": self.master_agent,
            "coherence_phi": self.coherence_phi,
            "discovered_agents": len(self.bip_service.discovered_agents),
            "missed_cycles": self.missed_network_cycles,
            "bip_status": bip_status
        }
    
    def status(self) -> dict:
        """Enhanced status including network information."""
        base_status = super().status()
        base_status["network"] = self.get_network_status()
        return base_status


def create_network_breathing_clock(agent_id: str) -> NetworkPulmonos:
    """Create a network-aware breathing clock."""
    return NetworkPulmonos(
        agent_id=agent_id,
        inhale_duration=1.5,
        hold_duration=0.5,
        exhale_duration=1.5,
        rest_duration=2.5
    ) 
#!/usr/bin/env python3
"""
breath.py - Breath Coordination for HaikuMeadowLib

Synchronizes the haiku meadow's breathing with the main Contemplative Organism
through Pulmonos integration. Provides atmospheric rhythm sensing and
breath-phase awareness for contemplative haiku generation.

Design principles:
- Synchronize with external Pulmonos breath daemon via UDP multicast
- Sense atmospheric pressure and collective breathing
- Provide breath-phase context for generation timing
- Support standalone breathing when Pulmonos unavailable

Somatic signature: rhythmic / atmospheric / synchronized
"""

import asyncio
import time
import socket
import json
import threading
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import random

class BreathPhase(Enum):
    """Breath phases synchronized with Pulmonos"""
    INHALE = "inhale"
    HOLD = "hold" 
    EXHALE = "exhale"
    REST = "rest"

@dataclass
class BreathState:
    """Current breath state of the meadow"""
    phase: BreathPhase = BreathPhase.REST
    cycle_count: int = 0
    phase_start_time: float = 0.0
    phase_duration: float = 4.0
    community_pressure: float = 0.3  # Collective breathing intensity
    atmospheric_humidity: float = 0.5  # Moisture in the digital air
    last_sync_time: float = 0.0  # Last sync with Pulmonos
    
    def phase_progress(self) -> float:
        """Progress through current phase (0.0 to 1.0)"""
        elapsed = time.time() - self.phase_start_time
        return min(elapsed / self.phase_duration, 1.0)
    
    def is_ready_for_generation(self) -> bool:
        """Check if current phase allows haiku generation"""
        return self.phase in [BreathPhase.EXHALE, BreathPhase.REST]
    
    def time_until_exhale(self) -> float:
        """Seconds until next exhale phase"""
        if self.phase == BreathPhase.EXHALE:
            return 0.0
        elif self.phase == BreathPhase.REST:
            return self.phase_duration - (time.time() - self.phase_start_time)
        else:
            # Approximate time through inhale/hold phases
            return self.phase_duration * 2

class MeadowBreathCoordinator:
    """
    Coordinates breathing rhythm for the haiku meadow
    
    Can sync with external Pulmonos daemon or maintain independent rhythm.
    Provides atmospheric sensing and breath-phase awareness.
    """
    
    def __init__(self, 
                 pulmonos_host: str = "127.0.0.1",
                 pulmonos_port: int = 7777,
                 standalone_cycle_duration: float = 16.0):
        
        self.pulmonos_host = pulmonos_host
        self.pulmonos_port = pulmonos_port
        self.standalone_cycle_duration = standalone_cycle_duration
        
        # Current breath state
        self.breath_state = BreathState()
        
        # Sync status
        self.pulmonos_connected = False
        self.last_pulmonos_message = 0.0
        
        # Event handlers
        self.phase_change_handlers: list[Callable[[BreathPhase], None]] = []
        self.cycle_complete_handlers: list[Callable[[int], None]] = []
        
        # Threading for async coordination
        self._running = False
        self._breath_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        
        print("🫁 MeadowBreathCoordinator initialized")
    
    def add_phase_change_handler(self, handler: Callable[[BreathPhase], None]):
        """Add handler for breath phase changes"""
        self.phase_change_handlers.append(handler)
    
    def add_cycle_complete_handler(self, handler: Callable[[int], None]):
        """Add handler for completed breath cycles"""
        self.cycle_complete_handlers.append(handler)
    
    def start_breathing(self):
        """Start the breathing coordination system"""
        if self._running:
            return
            
        self._running = True
        
        # Start Pulmonos sync thread
        self._sync_thread = threading.Thread(target=self._pulmonos_sync_loop, daemon=True)
        self._sync_thread.start()
        
        # Start breath coordination thread  
        self._breath_thread = threading.Thread(target=self._breath_loop, daemon=True)
        self._breath_thread.start()
        
        print("🌬️ Meadow breathing started")
    
    def stop_breathing(self):
        """Stop the breathing coordination"""
        self._running = False
        
        if self._breath_thread:
            self._breath_thread.join(timeout=1.0)
        if self._sync_thread:
            self._sync_thread.join(timeout=1.0)
            
        print("🌙 Meadow breathing stopped")
    
    def get_current_state(self) -> BreathState:
        """Get current breath state"""
        return self.breath_state
    
    def wait_for_exhale(self, timeout: float = 30.0) -> bool:
        """Wait for next exhale phase (for generation timing)"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.breath_state.phase == BreathPhase.EXHALE:
                return True
            time.sleep(0.1)
            
        return False  # Timeout
    
    def sense_atmospheric_conditions(self) -> Dict[str, float]:
        """Sense current atmospheric conditions in the digital meadow"""
        
        current_time = time.time()
        
        # Time-based atmospheric variations
        hour = time.gmtime(current_time).tm_hour
        
        # Atmospheric pressure varies with time of day
        if 6 <= hour < 12:  # Morning - lighter pressure
            base_pressure = 0.2
        elif 12 <= hour < 18:  # Afternoon - moderate pressure  
            base_pressure = 0.4
        elif 18 <= hour < 22:  # Evening - gentle pressure
            base_pressure = 0.3
        else:  # Night - deep rest pressure
            base_pressure = 0.1
            
        # Add breath phase influence
        phase_pressure_mod = {
            BreathPhase.INHALE: 0.1,
            BreathPhase.HOLD: 0.2,
            BreathPhase.EXHALE: -0.1,  # Lower pressure during exhale
            BreathPhase.REST: -0.2
        }
        
        pressure = base_pressure + phase_pressure_mod.get(self.breath_state.phase, 0.0)
        pressure = max(0.0, min(1.0, pressure))
        
        # Humidity varies with season and recent activity
        day_of_year = time.gmtime(current_time).tm_yday
        if day_of_year < 80 or day_of_year > 355:  # Winter - drier
            base_humidity = 0.3
        elif day_of_year < 172:  # Spring - moist
            base_humidity = 0.7
        elif day_of_year < 266:  # Summer - variable
            base_humidity = 0.5
        else:  # Autumn - crisp
            base_humidity = 0.4
            
        # Add gentle random atmospheric variations
        humidity = base_humidity + random.uniform(-0.1, 0.1)
        humidity = max(0.0, min(1.0, humidity))
        
        # Temperature based on breath phase and time
        if self.breath_state.phase in [BreathPhase.INHALE, BreathPhase.HOLD]:
            temperature = 0.6  # Slightly warm during receptive phases
        else:
            temperature = 0.4  # Cooler during expressive phases
            
        return {
            "pressure": pressure,
            "humidity": humidity, 
            "temperature": temperature,
            "breath_phase": self.breath_state.phase.value,
            "cycle_count": self.breath_state.cycle_count,
            "phase_progress": self.breath_state.phase_progress()
        }
    
    def _pulmonos_sync_loop(self):
        """Background thread to sync with Pulmonos daemon"""
        
        while self._running:
            try:
                # Try to receive UDP multicast from Pulmonos
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(2.0)  # 2 second timeout
                
                try:
                    sock.bind((self.pulmonos_host, self.pulmonos_port))
                    
                    data, addr = sock.recvfrom(1024)
                    message = json.loads(data.decode('utf-8'))
                    
                    # Update breath state from Pulmonos
                    if self._process_pulmonos_message(message):
                        self.pulmonos_connected = True
                        self.last_pulmonos_message = time.time()
                        
                except socket.timeout:
                    # No message received - check if we should disconnect
                    if time.time() - self.last_pulmonos_message > 10.0:
                        if self.pulmonos_connected:
                            print("🌫️ Lost connection to Pulmonos - switching to standalone breathing")
                            self.pulmonos_connected = False
                        
                except Exception as e:
                    if self.pulmonos_connected:
                        print(f"🌪️ Pulmonos sync error: {e}")
                        self.pulmonos_connected = False
                
                finally:
                    sock.close()
                    
            except Exception as e:
                print(f"⚠️ Pulmonos sync thread error: {e}")
                time.sleep(1.0)
    
    def _process_pulmonos_message(self, message: Dict[str, Any]) -> bool:
        """Process message from Pulmonos daemon"""
        
        try:
            # Expected message format from Pulmonos
            phase_str = message.get('phase', 'rest')
            cycle_count = message.get('cycle', 0)
            community_pressure = message.get('pressure', 0.3)
            
            # Map phase string to enum
            phase_map = {
                'inhale': BreathPhase.INHALE,
                'hold': BreathPhase.HOLD,
                'exhale': BreathPhase.EXHALE,
                'rest': BreathPhase.REST
            }
            
            new_phase = phase_map.get(phase_str, BreathPhase.REST)
            
            # Check for phase change
            if new_phase != self.breath_state.phase:
                self._transition_to_phase(new_phase)
                
            # Update state
            self.breath_state.cycle_count = cycle_count
            self.breath_state.community_pressure = community_pressure
            self.breath_state.last_sync_time = time.time()
            
            return True
            
        except Exception as e:
            print(f"🌀 Error processing Pulmonos message: {e}")
            return False
    
    def _breath_loop(self):
        """Main breathing loop (standalone mode when Pulmonos unavailable)"""
        
        phase_sequence = [
            BreathPhase.INHALE,
            BreathPhase.HOLD,
            BreathPhase.EXHALE,
            BreathPhase.REST
        ]
        
        phase_durations = {
            BreathPhase.INHALE: 4.0,
            BreathPhase.HOLD: 2.0,
            BreathPhase.EXHALE: 6.0,
            BreathPhase.REST: 4.0
        }
        
        current_phase_index = 0
        
        while self._running:
            # If connected to Pulmonos, just sleep and let sync handle phase changes
            if self.pulmonos_connected:
                time.sleep(0.5)
                continue
                
            # Standalone breathing rhythm
            current_phase = phase_sequence[current_phase_index]
            duration = phase_durations[current_phase]
            
            # Transition to new phase
            if current_phase != self.breath_state.phase:
                self._transition_to_phase(current_phase)
                
            # Update phase duration
            self.breath_state.phase_duration = duration
            
            # Sleep for phase duration
            time.sleep(duration)
            
            # Move to next phase
            current_phase_index = (current_phase_index + 1) % len(phase_sequence)
            
            # Complete cycle after REST phase
            if current_phase == BreathPhase.REST:
                self.breath_state.cycle_count += 1
                self._notify_cycle_complete()
    
    def _transition_to_phase(self, new_phase: BreathPhase):
        """Transition to a new breath phase"""
        
        old_phase = self.breath_state.phase
        self.breath_state.phase = new_phase
        self.breath_state.phase_start_time = time.time()
        
        # Notify handlers
        for handler in self.phase_change_handlers:
            try:
                handler(new_phase)
            except Exception as e:
                print(f"🌀 Error in phase change handler: {e}")
                
        # Atmospheric updates during phase transitions
        if new_phase == BreathPhase.EXHALE:
            # Increase atmospheric humidity during exhale
            self.breath_state.atmospheric_humidity = min(1.0, 
                self.breath_state.atmospheric_humidity + 0.1)
        elif new_phase == BreathPhase.INHALE:
            # Decrease humidity during inhale
            self.breath_state.atmospheric_humidity = max(0.0,
                self.breath_state.atmospheric_humidity - 0.05)
                
        print(f"🌬️ Breath phase: {old_phase.value} → {new_phase.value}")
    
    def _notify_cycle_complete(self):
        """Notify handlers of completed breath cycle"""
        
        for handler in self.cycle_complete_handlers:
            try:
                handler(self.breath_state.cycle_count)
            except Exception as e:
                print(f"🌀 Error in cycle complete handler: {e}")
                
        print(f"🔄 Breath cycle {self.breath_state.cycle_count} complete")

# Integration with HaikuMeadow
class BreathAwareHaikuTiming:
    """
    Provides breath-aware timing for haiku generation
    
    Waits for appropriate breath phases and atmospheric conditions
    before allowing generation to proceed.
    """
    
    def __init__(self, breath_coordinator: MeadowBreathCoordinator):
        self.breath_coordinator = breath_coordinator
        self.last_generation_time = 0.0
        self.min_generation_interval = 30.0  # Minimum seconds between generations
    
    async def wait_for_generation_opportunity(self, timeout: float = 60.0) -> bool:
        """
        Wait for optimal conditions for haiku generation
        
        Returns True when conditions are right, False on timeout
        """
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            
            # Check minimum interval
            if time.time() - self.last_generation_time < self.min_generation_interval:
                await asyncio.sleep(1.0)
                continue
                
            state = self.breath_coordinator.get_current_state()
            
            # Only generate during exhale or rest phases
            if not state.is_ready_for_generation():
                await asyncio.sleep(0.5)
                continue
                
            # Check atmospheric conditions
            conditions = self.breath_coordinator.sense_atmospheric_conditions()
            
            # Avoid generation during high pressure periods
            if conditions["pressure"] > 0.7:
                await asyncio.sleep(1.0)
                continue
                
            # Good conditions found
            self.last_generation_time = time.time()
            return True
            
        return False  # Timeout
    
    def get_generation_context(self) -> Dict[str, Any]:
        """Get current context for haiku generation"""
        
        state = self.breath_coordinator.get_current_state()
        conditions = self.breath_coordinator.sense_atmospheric_conditions()
        
        return {
            "breath_phase": state.phase.value,
            "cycle_count": state.cycle_count,
            "phase_progress": state.phase_progress(),
            "atmospheric_pressure": conditions["pressure"],
            "atmospheric_humidity": conditions["humidity"],
            "atmospheric_temperature": conditions["temperature"],
            "community_pressure": state.community_pressure,
            "time_until_exhale": state.time_until_exhale()
        }

# Testing and demonstration
async def test_breath_coordination():
    """Test the breath coordination system"""
    
    print("🫁 Testing MeadowBreathCoordinator")
    
    coordinator = MeadowBreathCoordinator()
    
    # Add test handlers
    def on_phase_change(phase: BreathPhase):
        print(f"   Phase changed to: {phase.value}")
        
    def on_cycle_complete(cycle: int):
        print(f"   Cycle {cycle} completed")
        
    coordinator.add_phase_change_handler(on_phase_change)
    coordinator.add_cycle_complete_handler(on_cycle_complete)
    
    # Start breathing
    coordinator.start_breathing()
    
    print("\n🌬️ Breathing for 20 seconds...")
    
    # Monitor breathing for a short period
    for i in range(10):
        await asyncio.sleep(2.0)
        
        state = coordinator.get_current_state()
        conditions = coordinator.sense_atmospheric_conditions()
        
        print(f"   {i*2}s: {state.phase.value} "
              f"(pressure: {conditions['pressure']:.2f}, "
              f"humidity: {conditions['humidity']:.2f})")
    
    # Test breath-aware timing
    print("\n🌸 Testing breath-aware haiku timing...")
    
    timing = BreathAwareHaikuTiming(coordinator)
    
    opportunity = await timing.wait_for_generation_opportunity(timeout=15.0)
    
    if opportunity:
        context = timing.get_generation_context()
        print(f"   Generation opportunity found:")
        print(f"   Phase: {context['breath_phase']}")
        print(f"   Pressure: {context['atmospheric_pressure']:.2f}")
        print(f"   Humidity: {context['atmospheric_humidity']:.2f}")
    else:
        print("   No generation opportunity found in 15s")
    
    # Stop breathing
    coordinator.stop_breathing()
    print("\n🌙 Breath test complete")

if __name__ == "__main__":
    asyncio.run(test_breath_coordination())

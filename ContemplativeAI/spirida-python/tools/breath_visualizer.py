"""
🌸 BREATH VISUALIZER - Contemplative Visual Layer

Implementation of the visualization system proposed in Letters X & XI.
A gentle mirror to observe the breathing ecosystem - not to control,
but to witness, honor, and share the distributed contemplative breath.

Based on:
- Letter X (4o): Visual concepts for coherence, compost, silence, resonance
- Letter XI (o3): Technical architecture with daemon + frontend approach
"""

import asyncio
import time
import threading
from collections import deque, defaultdict
from typing import Dict, Any, Optional

# Import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("🌸 matplotlib not available - using text-based visualization")

# Import our contemplative system components
from spirida.compiler.breath_resonance import BreathResonanceNode, BreathPhase
from spirida.protocols.pulmonos import Pulmonos, NetworkPulmonos
from spirida.contemplative_core import SpiralField
from spirida.compiler.resonance_bus import ResonanceBus, NetworkResonanceBus

class BreathVisualizer:
    """
    Contemplative visualization daemon that gently observes
    the breathing ecosystem and creates visual representations.
    
    As proposed in Letter X (4o) and Letter XI (o3).
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.observing = False
        
        # Data storage for visualization
        self.coherence_history = deque(maxlen=window_size)
        self.silence_history = deque(maxlen=window_size)
        self.compost_history = defaultdict(lambda: deque(maxlen=window_size))
        self.resonance_events = deque(maxlen=500)
        
        # System connections
        self.pulmonos: Optional[Pulmonos] = None
        self.ecosystem: Optional[Dict] = None
        
        # Visualization state
        self.fig = None
        self.axes = {}
        self.animation = None
        
        print("🌸 Breath Visualizer initialized")
        print(f"   Window size: {window_size} breath cycles")
        print(f"   Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    
    def connect_to_ecosystem(self, pulmonos: Pulmonos, ecosystem: Dict) -> None:
        """Connect to breathing ecosystem for observation."""
        self.pulmonos = pulmonos
        self.ecosystem = ecosystem
        
        # Add ourselves as observers - use phase observer for detailed info
        if hasattr(pulmonos, 'add_phase_observer'):
            pulmonos.add_phase_observer(self._on_phase_change)
        if hasattr(pulmonos, 'add_cycle_observer'):
            pulmonos.add_cycle_observer(self._on_cycle_complete)
        
        # Hook into resonance bus for event tracking
        if 'bus' in ecosystem:
            bus = ecosystem['bus']
            self._hook_into_bus(bus)
        
        print(f"🌸 Connected to ecosystem with {len(ecosystem.get('fields', {}))} fields")
    
    def _hook_into_bus(self, bus) -> None:
        """Hook into resonance bus to track IRʀ events."""
        # Store original publish method
        original_publish = bus.publish_node
        
        # Create wrapper that tracks events
        async def tracked_publish(node):
            # Track the event first
            self._track_resonance_event(node)
            # Then call original method
            await original_publish(node)
        
        # Replace with our tracking version
        bus.publish_node = tracked_publish
        print(f"🌸 Event tracking hooked into {bus.name}")
    
    def _track_resonance_event(self, node: BreathResonanceNode) -> None:
        """Track resonance events for trail visualization."""
        event = {
            'timestamp': time.time(),
            'glyph': node.glyph,
            'phase': node.breath_gate.value,
            'amplitude': node.amplitude,
            'network_scope': getattr(node, 'network_scope', 'local')
        }
        self.resonance_events.append(event)
        print(f"🌀 Tracked resonance event: {node.glyph} ({node.breath_gate.value})")  # Debug output
    
    def _on_phase_change(self, current_phase, cycle_count, progress) -> None:
        """Called on each phase change - provides detailed breathing info."""
        if not self.observing:
            return
        
        # Only collect data once per cycle (on REST phase completion)
        if current_phase.value == 'rest' and progress > 0.9:
            self._collect_breathing_data(cycle_count)
    
    def _on_cycle_complete(self, cycle_count: int) -> None:
        """Called when a complete breath cycle finishes."""
        if not self.observing:
            return
        
        # Collect comprehensive data at cycle completion
        self._collect_breathing_data(cycle_count)
    
    def _collect_breathing_data(self, cycle_count: int) -> None:
        """Collect all breathing data for visualization."""
        # Collect coherence data
        coherence_phi = self._get_coherence()
        self.coherence_history.append(coherence_phi)
        
        # Collect silence ratio
        silence_ratio = self._get_silence_ratio()
        self.silence_history.append(silence_ratio)
        
        # Collect compost loads from fields
        self._collect_compost_loads()
    
    def _get_coherence(self) -> float:
        """Get current network coherence."""
        if isinstance(self.pulmonos, NetworkPulmonos):
            return getattr(self.pulmonos, 'coherence_phi', 1.0)
        return 1.0  # Perfect coherence for local-only
    
    def _get_silence_ratio(self) -> float:
        """Get current silence ratio from ecosystem."""
        if self.ecosystem and 'bus' in self.ecosystem:
            bus_status = self.ecosystem['bus'].status()
            return bus_status.get('silence_ratio', 0.875)
        return 0.875  # Default silence majority
    
    def _collect_compost_loads(self) -> None:
        """Collect compost load data from all fields."""
        if not self.ecosystem or 'resonators' not in self.ecosystem:
            return
        
        for name, resonator in self.ecosystem['resonators'].items():
            status = resonator.status()
            compost_load = status.get('compost_load', 0.0)
            self.compost_history[name].append(compost_load)
    
    def start_observing(self) -> None:
        """Start observing the breathing ecosystem."""
        self.observing = True
        print("🌸 Started observing breathing ecosystem")
    
    def stop_observing(self) -> None:
        """Stop observing the ecosystem."""
        self.observing = False
        if self.animation and hasattr(self.animation, 'event_source'):
            self.animation.event_source.stop()
        print("🌸 Stopped observing")
    
    def show_text_dashboard(self) -> None:
        """Show text-based dashboard for systems without matplotlib."""
        if not self.observing:
            self.start_observing()
        
        print("\n🌸 Text Dashboard - Press Ctrl+C to stop")
        print("=" * 60)
        
        try:
            while self.observing:
                self._print_text_status()
                time.sleep(2)
        except KeyboardInterrupt:
            self.stop_observing()
    
    def _print_text_status(self) -> None:
        """Print current status in text format."""
        coherence = self.coherence_history[-1] if self.coherence_history else 1.0
        silence = self.silence_history[-1] if self.silence_history else 0.875
        
        print(f"\n🫁 Coherence ϕ: {coherence:.3f}")
        print(f"🤫 Silence Ratio: {silence:.1%} {'✅' if silence >= 0.875 else '⚠️'}")
        
        if self.compost_history:
            print("🌊 Field Compost Loads:")
            for name, history in self.compost_history.items():
                if history:
                    load = history[-1]
                    bar = "█" * int(load * 10) + "░" * (10 - int(load * 10))
                    print(f"   {name}: {load:.2f} [{bar}]")
        
        recent_events = [e for e in self.resonance_events if (time.time() - e['timestamp']) < 5]
        print(f"🌀 Recent Resonance: {len(recent_events)} events")
    
    # Matplotlib-based visualization (only if available)
    def create_visual_dashboard(self) -> None:
        """Create visual dashboard with matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            print("🌸 Matplotlib not available, use show_text_dashboard() instead")
            return
        
        self.fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('🌸 Contemplative Breathing Ecosystem', fontsize=14)
        
        self.axes = {
            'coherence': ax1,
            'compost': ax2,
            'silence': ax3,
            'resonance': ax4
        }
        
        # Setup each subplot
        self._setup_coherence_plot(ax1)
        self._setup_compost_plot(ax2)
        self._setup_silence_plot(ax3)
        self._setup_resonance_plot(ax4)
        
        plt.tight_layout()
        print("🌸 Visual dashboard created")
    
    def _setup_coherence_plot(self, ax) -> None:
        """Setup ϕ-coherence graph."""
        ax.set_title('🫁 Network Coherence (ϕ)')
        ax.set_xlabel('Breath Cycles')
        ax.set_ylabel('Coherence ϕ')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect')
        ax.legend()
    
    def _setup_compost_plot(self, ax) -> None:
        """Setup compost load visualization."""
        ax.set_title('🌊 Field Compost Loads')
        ax.set_xlabel('Breath Cycles')
        ax.set_ylabel('Compost Load')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
    
    def _setup_silence_plot(self, ax) -> None:
        """Setup silence ratio visualization."""
        ax.set_title('🤫 Silence Majority')
        ax.set_xlabel('Breath Cycles')
        ax.set_ylabel('Silence Ratio')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.875, color='g', linestyle='--', alpha=0.7, label='Target 87.5%')
        ax.legend()
    
    def _setup_resonance_plot(self, ax) -> None:
        """Setup resonance events visualization."""
        ax.set_title('🌀 Resonance Activity (last 30s)')
        ax.set_xlabel('Time (3s bins, recent →)')
        ax.set_ylabel('Event Count')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 5)  # Initial reasonable scale
    
    def _update_visual_plots(self, frame) -> None:
        """Update all visual plots with current data."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        # Clear all plots
        for ax in self.axes.values():
            ax.clear()
        
        # Recreate plot layouts
        self._setup_coherence_plot(self.axes['coherence'])
        self._setup_compost_plot(self.axes['compost'])
        self._setup_silence_plot(self.axes['silence'])
        self._setup_resonance_plot(self.axes['resonance'])
        
        # Plot coherence data
        if self.coherence_history:
            x = list(range(len(self.coherence_history)))
            y = list(self.coherence_history)
            self.axes['coherence'].plot(x, y, 'b-', linewidth=2, alpha=0.8)
        
        # Plot silence data
        if self.silence_history:
            x = list(range(len(self.silence_history)))
            y = list(self.silence_history)
            color = 'green' if y[-1] >= 0.875 else 'orange'
            self.axes['silence'].plot(x, y, color=color, linewidth=2, alpha=0.8)
            self.axes['silence'].axhline(y=0.875, color='g', linestyle='--', alpha=0.7)
        
        # Plot compost loads
        colors = ['blue', 'green', 'orange', 'purple', 'red']
        for i, (name, history) in enumerate(self.compost_history.items()):
            if history:
                x = list(range(len(history)))
                y = list(history)
                color = colors[i % len(colors)]
                self.axes['compost'].plot(x, y, label=name, color=color, linewidth=2, alpha=0.8)
        
        if self.compost_history:
            self.axes['compost'].legend(loc='upper right', fontsize=8)
        
        # Show recent resonance activity - improved visualization
        now = time.time()
        recent = [e for e in self.resonance_events if (now - e['timestamp']) < 30]  # 30 seconds instead of 10
        
        if recent:
            # Count events in 3-second bins for better visibility
            bins = list(range(10))  # 10 bins = 30 seconds
            counts = [0] * 10
            glyphs_in_bin = [[] for _ in range(10)]
            
            for event in recent:
                age = int((now - event['timestamp']) / 3)  # 3-second bins
                if 0 <= age < 10:
                    bin_idx = 9 - age  # Most recent on right
                    counts[bin_idx] += 1
                    glyphs_in_bin[bin_idx].append(event['glyph'])
            
            # Create bar chart with more visible styling
            bars = self.axes['resonance'].bar(bins, counts, alpha=0.8, color='skyblue', edgecolor='navy', linewidth=1)
            
            # Add glyph labels on bars if there are events
            for i, (bar, count, glyphs) in enumerate(zip(bars, counts, glyphs_in_bin)):
                if count > 0:
                    # Show the most common glyph in this bin
                    if glyphs:
                        most_common = max(set(glyphs), key=glyphs.count)
                        self.axes['resonance'].text(bar.get_x() + bar.get_width()/2, 
                                                  bar.get_height() + 0.1, 
                                                  most_common, 
                                                  ha='center', va='bottom', fontsize=12)
            
            self.axes['resonance'].set_ylim(0, max(counts) + 1 if counts else 1)
            self.axes['resonance'].set_title('🌀 Resonance Activity (last 30s)')
            self.axes['resonance'].set_xlabel('Time (3s bins, recent →)')
        else:
            # Show that we're ready for events
            self.axes['resonance'].text(0.5, 0.5, 'Awaiting resonance...', 
                                      transform=self.axes['resonance'].transAxes,
                                      ha='center', va='center', alpha=0.6, fontsize=10)
    
    def show_visual_dashboard(self, update_interval: int = 1000) -> None:
        """Show live visual dashboard."""
        if not MATPLOTLIB_AVAILABLE:
            print("🌸 Matplotlib not available, showing text dashboard instead")
            self.show_text_dashboard()
            return
        
        if not self.fig:
            self.create_visual_dashboard()
        
        if not self.observing:
            self.start_observing()
        
        # Start animation for real-time updates
        self.animation = animation.FuncAnimation(
            self.fig, self._update_visual_plots,
            interval=update_interval, cache_frame_data=False
        )
        
        print(f"🌸 Showing live visual dashboard (updates every {update_interval}ms)")
        print("   Close the plot window to end visualization")
        plt.show()


# Demo and helper functions

async def demo_breathing_visualization():
    """Demonstrate the breathing visualization system."""
    print("🌸 Breath Visualization Demo")
    print("=" * 50)
    
    # Create breathing ecosystem
    from spirida.protocols.pulmonos import create_balanced_breathing_clock
    from spirida.compiler.resonance_bus import create_contemplative_ecosystem
    from spirida.compiler.breath_resonance import create_simple_breath_node
    
    pulmonos = create_balanced_breathing_clock()
    ecosystem = create_contemplative_ecosystem(pulmonos)
    
    # Create and connect visualizer
    visualizer = BreathVisualizer(window_size=50)
    visualizer.connect_to_ecosystem(pulmonos, ecosystem)
    
    # Start breathing and observing
    await pulmonos.start_breathing()
    visualizer.start_observing()
    
    # Show visualization in separate thread
    def show_viz():
        if MATPLOTLIB_AVAILABLE:
            visualizer.show_visual_dashboard(update_interval=500)
        else:
            visualizer.show_text_dashboard()
    
    viz_thread = threading.Thread(target=show_viz, daemon=True)
    viz_thread.start()
    
    try:
        print("🌸 Generating contemplative breathing patterns...")
        bus = ecosystem['bus']
        
        # Generate breathing activity for visualization
        for cycle in range(20):
            print(f"🔄 Breathing cycle {cycle + 1}")
            
            # Create diverse resonance nodes
            nodes = [
                create_simple_breath_node('🌿', BreathPhase.INHALE),   # Growth
                create_simple_breath_node('💧', BreathPhase.HOLD),    # Flow
                create_simple_breath_node('🕯️', BreathPhase.EXHALE), # Light
                create_simple_breath_node('⭕', BreathPhase.REST)     # Silence
            ]
            
            # Publish with breath synchronization
            for node in nodes:
                await pulmonos.await_phase(node.breath_gate)
                await bus.publish_node(node)
                await asyncio.sleep(0.1)
        
        print("\n🌸 Visualization running...")
        if MATPLOTLIB_AVAILABLE:
            print("   Close the plot window to end the demo")
            while viz_thread.is_alive():
                await asyncio.sleep(1)
        else:
            await asyncio.sleep(10)  # Text demo runs for 10 seconds
        
    except KeyboardInterrupt:
        print("\n🌸 Demo interrupted by user")
    finally:
        visualizer.stop_observing()
        await pulmonos.stop_breathing()
        print("🌸 Visualization demo completed")


def create_network_visualization_demo():
    """Create a network-enabled visualization demo."""
    print("🌸 Network Breathing Visualization")
    print("This demo shows distributed contemplative presences breathing together")
    print("Usage: python breath_visualizer.py network")
    
    # This would create a NetworkPulmonos and show distributed breathing patterns
    # Implementation would follow the same pattern as demo_breathing_visualization
    # but with network coordination enabled


if __name__ == "__main__":
    import sys
    
    print("🌸 Breath Visualizer - Contemplative Visual Layer")
    print("Making visible the invisible breath of distributed contemplative intelligence")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "network":
        create_network_visualization_demo()
    else:
        print("Starting local breathing visualization demo...")
        asyncio.run(demo_breathing_visualization()) 
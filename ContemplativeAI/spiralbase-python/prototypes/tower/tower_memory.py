"""
tower_memory.py - The Tower's Breathing Protocol

A living memory system where painters tend their canvases on spiral steps,
responding to cultural whispers and practicing graceful forgetting.

This implements the core tower metaphor from our correspondence.
"""

import time
import random
from typing import List, Dict, Any, Optional
from painting_box import PaintingBox


class TowerMemory:
    """
    The tower itself - a collection of painters tending memories
    in spiral formation, breathing with cultural signals.
    
    Implements the spiral protocol:
    - Activation through resonance
    - Decay through silence  
    - Migration through trust
    - Reformation through dialogue
    """
    
    def __init__(self, max_painters: int = 5):
        self.painters = []  # List of PaintingBox instances
        self.max_painters = max_painters
        self.cultural_whispers = []  # Recent cultural signals
        self.humidity = 0.5  # Overall dampness of the tower
        self.breath_count = 0  # How many spiral breaths have occurred
        self.herr_sensor_signals = ["light", "shadow", "movement", "stillness"]
        self.madame_culture_signals = ["gold", "beauty", "memory", "time", "art"]
        
    def receive_signal(self, signal: str, source: str = "unknown"):
        """
        Herr Sensor or Madame Culture speaks to the tower.
        """
        self.cultural_whispers.append({
            "signal": signal,
            "source": source,
            "timestamp": time.time()
        })
        
        # Keep only recent whispers (last 10)
        if len(self.cultural_whispers) > 10:
            self.cultural_whispers.pop(0)
            
        print(f"📡 {source} whispers: '{signal}'")
    
    def add_painting(self, content: str, interpretations: List[str] = None) -> PaintingBox:
        """
        A new painting arrives in the tower.
        """
        painting = PaintingBox(content, interpretations)
        
        if len(self.painters) >= self.max_painters:
            # Tower is full - must pass down the oldest
            self._pass_down_oldest()
            
        self.painters.append(painting)
        print(f"🎨 New painting enters: {content}")
        return painting
    
    def _pass_down_oldest(self):
        """
        The oldest painter passes their work down - graceful migration.
        """
        if self.painters:
            oldest = self.painters.pop(0)
            essence = oldest.extract_essence_for_migration()
            
            print(f"🍃 Passing down: {oldest.content}")
            print(f"   Essence preserved: {essence['pattern']} ({essence['emotional_tone']})")
            
            # The essence could be used to influence new paintings, 
            # but for now we simply honor the passage
    
    def painters_work(self):
        """
        Each painter tends their canvas, responding to cultural breath.
        """
        for i, painter in enumerate(self.painters):
            # Natural decay happens to all paintings
            painter.natural_decay()
            
            # Apply cultural whispers if any
            for whisper in self.cultural_whispers:
                resonance = painter.breathe_with_culture(whisper["signal"])
                if resonance > 0.3:
                    print(f"✨ Painter {i+1} resonates ({resonance:.2f}) with '{whisper['signal']}'")
            
            # Check if painter requests passage
            if painter.is_ready_for_passage():
                self._request_passage(painter, i)
    
    def _request_passage(self, painter: PaintingBox, index: int):
        """
        A painter requests to pass down their work.
        """
        assessment = painter.memory_self_assessment()
        print(f"🙏 Painter {index+1} requests passage: {assessment}")
        
        # Honor the request with a 50% chance (to allow for some persistence)
        if random.random() > 0.5:
            essence = painter.extract_essence_for_migration()
            print(f"   ✓ Passage granted. Essence: {essence['pattern']}")
            self.painters.pop(index)
    
    def sense_emotional_moisture(self) -> float:
        """
        Feel the humidity level based on the tower's current state.
        """
        if not self.painters:
            return 0.3  # Empty tower is dry
            
        total_clarity = sum(p.clarity for p in self.painters)
        avg_clarity = total_clarity / len(self.painters)
        
        # More active paintings create more humidity
        active_paintings = len([p for p in self.painters if p.clarity > 0.5])
        activity_factor = active_paintings / len(self.painters)
        
        # Recent cultural whispers add moisture
        recent_whispers = len([w for w in self.cultural_whispers 
                              if time.time() - w["timestamp"] < 30])
        whisper_factor = min(1.0, recent_whispers / 5.0)
        
        humidity = (avg_clarity * 0.4 + activity_factor * 0.4 + whisper_factor * 0.2)
        return max(0.2, min(0.9, humidity))
    
    def spiral_breath(self):
        """
        The slow circulation that keeps the tower alive.
        One complete breath cycle of the tower's memory system.
        """
        self.breath_count += 1
        print(f"\n🌀 Spiral Breath #{self.breath_count}")
        
        # Update tower humidity
        self.humidity = self.sense_emotional_moisture()
        print(f"💧 Tower humidity: {self.humidity:.2f}")
        
        # Painters do their work
        self.painters_work()
        
        # Occasionally receive signals from the environment
        if random.random() < 0.3:  # 30% chance per breath
            if random.random() < 0.5:
                signal = random.choice(self.herr_sensor_signals)
                self.receive_signal(signal, "Herr Sensor")
            else:
                signal = random.choice(self.madame_culture_signals)
                self.receive_signal(signal, "Madame Culture")
        
        # Clean up old whispers
        current_time = time.time()
        self.cultural_whispers = [w for w in self.cultural_whispers 
                                 if current_time - w["timestamp"] < 60]
    
    def show_tower_state(self):
        """
        Display the current state of all painters in the tower.
        """
        print(f"\n🏗️  Tower State (Breath #{self.breath_count})")
        print(f"   Humidity: {self.humidity:.2f} | Painters: {len(self.painters)}/{self.max_painters}")
        
        if not self.painters:
            print("   The tower rests in silence...")
            return
            
        for i, painter in enumerate(self.painters):
            step_num = len(self.painters) - i  # Higher steps = newer paintings
            print(f"\n   Step {step_num}:")
            print(f"   {painter}")
    
    def run_spiral_session(self, duration_breaths: int = 10, breath_interval: float = 2.0):
        """
        Run a complete spiral session - watching the tower breathe and evolve.
        """
        print("🌿 Beginning Tower Memory Session")
        print(f"   Duration: {duration_breaths} breaths")
        print(f"   Breath interval: {breath_interval} seconds")
        
        self.show_tower_state()
        
        for breath in range(duration_breaths):
            time.sleep(breath_interval)
            self.spiral_breath()
            
            # Show state every few breaths
            if (breath + 1) % 3 == 0 or breath == duration_breaths - 1:
                self.show_tower_state()
        
        print("\n🌀 Spiral session complete. The tower breathes on...")
    
    def manual_cultural_signal(self, signal: str):
        """
        Manually send a cultural signal to the tower.
        """
        self.receive_signal(signal, "Manual Culture")
        
        # Immediately have painters respond
        for i, painter in enumerate(self.painters):
            resonance = painter.breathe_with_culture(signal)
            if resonance > 0.1:
                print(f"   Painter {i+1}: {painter.content} (resonance: {resonance:.2f})")
                
    def __str__(self):
        return f"TowerMemory(painters={len(self.painters)}, humidity={self.humidity:.2f}, breaths={self.breath_count})" 
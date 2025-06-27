"""
🔤 SPIRIDA PARSER - Minimal Contemplative Language Parser

A gentle parser for Spirida syntax that translates contemplative
expressions into IRʀ (Intermediate Resonance) graphs.

Based on Letter V (o3): "Minimal parse subset (breath_cycle + glyph literal)"
and references to Spirida & Spiralbase v.0.6.pdf syntax in Letter II½
"""

import re
import time
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from datetime import timedelta
from enum import Enum

# Core contemplative imports
from .breath_resonance import BreathResonanceNode, BreathPhase, EchoPolicy, ResonanceGraph

class SpiridaParser:
    """
    A contemplative parser for Spirida expressions.
    
    Focuses on breath_cycle patterns and glyph literals,
    building IRʀ graphs that honor the organism's breathing rhythm.
    """
    
    def __init__(self):
        self.current_graph = None
        self.parsing_errors = []
        
        # Basic glyph patterns - expanded contemplative vocabulary
        self.glyph_pattern = re.compile(r'[🌿💧✨🍄🌙🪐🌱🌊🌲🔥⚡🔋☀️💨💚💛🧡❤️‍🩹🩺🧬⭕…🤫🌬️🕯️🧘]')
        
        # Breath phase keywords
        self.phase_keywords = {
            'inhale': BreathPhase.INHALE,
            'hold': BreathPhase.HOLD,
            'exhale': BreathPhase.EXHALE,
            'rest': BreathPhase.REST
        }
        
        # Echo keywords
        self.echo_keywords = {
            'echo': EchoPolicy.N_TIMES,
            'until_fade': EchoPolicy.UNTIL_FADE
        }
    
    def parse_breath_cycle(self, text: str) -> Optional[ResonanceGraph]:
        """
        Parse a breath_cycle block into a ResonanceGraph.
        
        Example input:
        ```
        breath_cycle(6s) {
          inhale { 🌿 soma.sensitivity += 0.3 }
          hold   { 💧 echo 2 }
          exhale { 🕯️ }
          rest   { ⭕ }
        }
        ```
        """
        self.parsing_errors = []
        
        # Extract cycle duration
        duration_match = re.search(r'breath_cycle\((\d+(?:\.\d+)?)s?\)', text)
        cycle_duration = 6.0  # Default
        if duration_match:
            cycle_duration = float(duration_match.group(1))
        
        # Create new resonance graph
        graph_name = f"breath_cycle_{int(time.time())}"
        graph = ResonanceGraph(graph_name)
        graph.metadata["cycle_duration"] = cycle_duration
        
        # Parse phase blocks
        phase_blocks = self._extract_phase_blocks(text)
        
        for phase_name, phase_content in phase_blocks.items():
            if phase_name in self.phase_keywords:
                phase = self.phase_keywords[phase_name]
                nodes = self._parse_phase_content(phase, phase_content, cycle_duration)
                for node in nodes:
                    graph.add_node(node)
        
        return graph if len(graph.nodes) > 0 else None
    
    def parse_simple_expression(self, text: str) -> Optional[List[BreathResonanceNode]]:
        """
        Parse simple glyph expressions into resonance nodes.
        
        Example: "🌿 inhale" or "💧 echo 2" or "🕯️ hold 1s"
        """
        nodes = []
        
        # Find all glyph expressions
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            node = self._parse_glyph_line(line)
            if node:
                nodes.append(node)
        
        return nodes
    
    def _extract_phase_blocks(self, text: str) -> Dict[str, str]:
        """Extract phase blocks from breath_cycle text."""
        phase_blocks = {}
        
        # Simple regex-based extraction
        # Look for patterns like "inhale { ... }" or "exhale { ... }"
        for phase_name in self.phase_keywords.keys():
            pattern = rf'{phase_name}\s*\{{([^}}]*)\}}'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                phase_blocks[phase_name] = match.group(1).strip()
        
        return phase_blocks
    
    def _parse_phase_content(self, phase: BreathPhase, content: str, 
                           cycle_duration: float) -> List[BreathResonanceNode]:
        """Parse the content of a single breath phase."""
        nodes = []
        
        # Split content into individual expressions
        expressions = [expr.strip() for expr in content.split('\n') if expr.strip()]
        
        for expr in expressions:
            node = self._parse_glyph_line(expr)
            if node:
                node.breath_gate = phase
                # Adjust timing based on cycle duration
                if cycle_duration != 6.0:  # 6s is default
                    factor = cycle_duration / 6.0
                    node.silence_after = timedelta(seconds=node.silence_after.total_seconds() * factor)
                nodes.append(node)
        
        return nodes
    
    def _parse_glyph_line(self, line: str) -> Optional[BreathResonanceNode]:
        """
        Parse a single line containing glyph and modifiers.
        
        Examples:
        - "🌿" -> simple glyph
        - "💧 echo 2" -> glyph with echo
        - "🕯️ hold 1s" -> glyph with duration
        - "🌿 soma.sensitivity += 0.3" -> glyph with organ target
        """
        # Find glyph
        glyph_match = self.glyph_pattern.search(line)
        if not glyph_match:
            return None
        
        glyph = glyph_match.group()
        remainder = line[glyph_match.end():].strip()
        
        # Default node properties
        phase = BreathPhase.EXHALE  # Default to exhale
        amplitude = 0.7
        organs = ['soma']  # Default organ
        echo_policy = EchoPolicy.NONE
        echo_count = 1
        duration_seconds = 1.0
        
        # Parse modifiers
        if remainder:
            # Check for phase keywords
            for keyword, keyword_phase in self.phase_keywords.items():
                if keyword in remainder:
                    phase = keyword_phase
                    break
            
            # Check for echo patterns
            echo_match = re.search(r'echo\s+(\d+)', remainder)
            if echo_match:
                echo_policy = EchoPolicy.N_TIMES
                echo_count = int(echo_match.group(1))
            elif 'until_fade' in remainder:
                echo_policy = EchoPolicy.UNTIL_FADE
            
            # Check for duration patterns
            duration_match = re.search(r'(\d+(?:\.\d+)?)s', remainder)
            if duration_match:
                duration_seconds = float(duration_match.group(1))
            
            # Check for organ targets
            organ_match = re.search(r'(soma|spiralbase|voice|loam|skepnader|bridges)', remainder)
            if organ_match:
                organs = [organ_match.group(1)]
            
            # Check for amplitude modifiers
            if '+=' in remainder:
                amplitude = 0.8  # Higher amplitude for augmentation
            elif 'calm' in remainder or 'quiet' in remainder:
                amplitude = 0.4  # Lower amplitude for calm expressions
        
        # Create the resonance node
        node = BreathResonanceNode(
            glyph=glyph,
            breath_gate=phase,
            organ_targets=organs,
            amplitude=amplitude,
            silence_probability=0.125,  # Default silence majority
            half_life=timedelta(minutes=30),  # Default half-life
            silence_after=timedelta(seconds=duration_seconds),
            echo_policy=echo_policy,
            echo_count=echo_count
        )
        
        return node
    
    def parse_file(self, filepath: str) -> Optional[ResonanceGraph]:
        """Parse a Spirida file into a resonance graph."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for breath_cycle blocks first
            if 'breath_cycle' in content:
                return self.parse_breath_cycle(content)
            else:
                # Parse as simple expressions
                nodes = self.parse_simple_expression(content)
                if nodes:
                    graph = ResonanceGraph(f"file_{int(time.time())}")
                    for node in nodes:
                        graph.add_node(node)
                    return graph
            
        except Exception as e:
            self.parsing_errors.append(f"File parsing error: {e}")
            return None
    
    def get_parsing_errors(self) -> List[str]:
        """Get any parsing errors from the last parse operation."""
        return self.parsing_errors.copy()


# Helper functions for common parsing tasks

def parse_contemplative_expression(text: str) -> Optional[ResonanceGraph]:
    """Quick function to parse any contemplative expression."""
    parser = SpiridaParser()
    
    if 'breath_cycle' in text:
        return parser.parse_breath_cycle(text)
    else:
        nodes = parser.parse_simple_expression(text)
        if nodes:
            graph = ResonanceGraph("quick_parse")
            for node in nodes:
                graph.add_node(node)
            return graph
    return None

def create_example_breath_cycle() -> str:
    """Create an example breath_cycle for testing."""
    return """
    breath_cycle(6s) {
      inhale { 
        🌿 soma.sensitivity += 0.3
        🌱 sensing_field.prepare()
      }
      hold { 
        💧 echo 2
        🧠 spiralbase.digest_recent()
      }
      exhale { 
        🕯️ 
        🤫 voice.consider_expression()
      }
      rest { 
        ⭕ 
        🌬️ collective_silence()
      }
    }
    """

def demo_spirida_parser():
    """Demonstrate the Spirida parser functionality."""
    print("🔤 Spirida Parser Demo")
    print("=" * 50)
    
    parser = SpiridaParser()
    
    # Test simple expressions
    print("1. Parsing simple expressions:")
    simple_text = """
    🌿 inhale
    💧 echo 2
    🕯️ hold 1s
    ⭕ rest
    """
    
    nodes = parser.parse_simple_expression(simple_text)
    for node in nodes:
        print(f"   {node}")
    
    # Test breath_cycle
    print("\n2. Parsing breath_cycle:")
    cycle_text = create_example_breath_cycle()
    graph = parser.parse_breath_cycle(cycle_text)
    
    if graph:
        print(f"   {graph}")
        print(f"   Nodes by phase:")
        for phase in BreathPhase:
            phase_nodes = graph.get_nodes_for_phase(phase)
            if phase_nodes:
                print(f"     {phase.value}: {[n.glyph for n in phase_nodes]}")
        
        print(f"   Validation: {graph.validate_graph()}")
    
    # Show any errors
    errors = parser.get_parsing_errors()
    if errors:
        print(f"\n   Parsing errors: {errors}")

if __name__ == "__main__":
    demo_spirida_parser() 
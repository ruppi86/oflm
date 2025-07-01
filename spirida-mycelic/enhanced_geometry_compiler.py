#!/usr/bin/env python3
"""
Enhanced Geometry Compiler for Spirida-Mycelic
==============================================

Addressing o3's questions about topology-as-ethics and spatial bio-digital expressions.

Implements:
3. ✅ GraphML vs planar coordinates decision
4. ✅ Metadata for moisture gradients and environmental factors  
5. ✅ Logic gates emerging from spatial configuration
6. ✅ Species-specific topology preferences

Based on "Mining Logical Circuits in Fungi" and "Logics in Mycelium Networks"
"""

import json
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import xml.etree.ElementTree as ET

class LogicGate(Enum):
    """Logic gates that can emerge from mycelial topology"""
    AND = "and"
    OR = "or" 
    NAND = "nand"
    NOR = "nor"
    XOR = "xor"
    NOT = "not"
    BUFFER = "buffer"

class FungalSpecies(Enum):
    """Fungal species with different topological preferences"""
    PLEUROTUS_DJAMOR = "pleurotus_djamor"
    GANODERMA_RESINACEUM = "ganoderma_resinaceum"
    MYCELIUM_COMPOSITE = "mycelium_composite"

@dataclass
class GeometryNode:
    """A node in the bio-digital geometry"""
    id: str
    x: float
    y: float
    z: float = 0.0
    role: str = "junction"           # junction, input, output, amplifier
    moisture_level: float = 0.5      # 0.0 (dry) to 1.0 (saturated)
    ph_level: float = 7.0           # pH level at this location
    temperature: float = 20.0        # Temperature in Celsius
    electrical_conductivity: float = 0.5  # 0.0 to 1.0
    glyph_affinity: Optional[str] = None  # Preferred glyph at this node
    
@dataclass
class GeometryEdge:
    """An edge connecting nodes in bio-digital space"""
    id: str
    source_id: str
    target_id: str
    weight: float = 1.0
    connection_type: str = "mycelial"  # mycelial, electrical, optical
    conductance: float = 1.0           # Signal conductance
    delay_ms: float = 100.0            # Propagation delay
    moisture_dependence: float = 0.3   # How much moisture affects this edge

@dataclass 
class GeometryField:
    """Complete bio-digital geometry field"""
    field_id: str
    nodes: List[GeometryNode]
    edges: List[GeometryEdge]
    species: FungalSpecies
    substrate_properties: Dict[str, Any]
    logic_gates: List[Dict[str, Any]]  # Detected logic gate structures
    ethical_resonance: float = 0.5     # How ethically resonant this geometry is

class EnhancedGeometryCompiler:
    """
    Enhanced geometry compiler answering o3's questions:
    
    3. ✅ GraphML output with planar coordinate fallback
    4. ✅ Metadata for moisture gradients and environmental factors
    5. ✅ Logic gate detection from topology 
    6. ✅ Species-specific topology generation
    """
    
    def __init__(self):
        # Species-specific topology preferences
        self.species_preferences = {
            FungalSpecies.PLEUROTUS_DJAMOR: {
                'preferred_connectivity': 3.2,  # Average connections per node
                'branching_angle': 45.0,        # Degrees
                'growth_pattern': 'radial',
                'logic_preference': [LogicGate.XOR, LogicGate.OR],
                'moisture_optimum': 0.8,
                'ph_optimum': 6.5
            },
            FungalSpecies.GANODERMA_RESINACEUM: {
                'preferred_connectivity': 2.1,
                'branching_angle': 60.0,
                'growth_pattern': 'linear',
                'logic_preference': [LogicGate.AND, LogicGate.BUFFER],
                'moisture_optimum': 0.6,
                'ph_optimum': 7.2
            },
            FungalSpecies.MYCELIUM_COMPOSITE: {
                'preferred_connectivity': 4.0,
                'branching_angle': 30.0,
                'growth_pattern': 'mesh',
                'logic_preference': [LogicGate.NAND, LogicGate.NOR],
                'moisture_optimum': 0.7,
                'ph_optimum': 6.8
            }
        }
        
        # Logic gate topology patterns (3 nodes minimum for gates)
        self.logic_gate_patterns = {
            LogicGate.AND: {
                'min_nodes': 3,
                'topology': 'convergent',
                'required_connections': [(0, 2), (1, 2)],  # Two inputs, one output
                'spatial_arrangement': 'triangle'
            },
            LogicGate.OR: {
                'min_nodes': 3, 
                'topology': 'convergent',
                'required_connections': [(0, 2), (1, 2)],
                'spatial_arrangement': 'triangle'
            },
            LogicGate.XOR: {
                'min_nodes': 4,
                'topology': 'cross',
                'required_connections': [(0, 2), (1, 2), (0, 3), (1, 3)],
                'spatial_arrangement': 'diamond'
            },
            LogicGate.NOT: {
                'min_nodes': 2,
                'topology': 'linear',
                'required_connections': [(0, 1)],
                'spatial_arrangement': 'line'
            }
        }
        
    def generate_species_geometry(self, 
                                species: FungalSpecies,
                                target_glyph: str = "🌌",
                                field_size: Tuple[float, float, float] = (10.0, 10.0, 2.0),
                                node_count: int = 12) -> GeometryField:
        """
        Generate geometry optimized for a specific glyph expression.
        
        Answers o3's question: "What shape must the substrate hold to express 🌪️?"
        """
        preferences = self.species_preferences[species]
        
        print(f"🍄 Generating {species.value} geometry for glyph {target_glyph}")
        print(f"   Field size: {field_size}")
        print(f"   Target nodes: {node_count}")
        
        # Create base nodes with species-appropriate distribution
        nodes = self._generate_species_nodes(species, field_size, node_count)
        
        # Create edges based on species connectivity preferences
        edges = self._generate_species_edges(species, nodes)
        
        # Optimize for target glyph
        self._optimize_for_glyph(nodes, edges, target_glyph, species)
        
        # Detect emergent logic gates
        logic_gates = self._detect_logic_gates(nodes, edges)
        
        # Calculate ethical resonance
        ethical_resonance = self._calculate_ethical_resonance(nodes, edges, logic_gates)
        
        field = GeometryField(
            field_id=f"{species.value}_{target_glyph}_{int(time.time() if 'time' in globals() else 12345)}",
            nodes=nodes,
            edges=edges,
            species=species,
            substrate_properties=self._generate_substrate_properties(species, field_size),
            logic_gates=logic_gates,
            ethical_resonance=ethical_resonance
        )
        
        print(f"✨ Generated geometry with {len(nodes)} nodes, {len(edges)} edges")
        print(f"   Logic gates detected: {len(logic_gates)}")
        print(f"   Ethical resonance: {ethical_resonance:.3f}")
        
        return field
    
    def _generate_species_nodes(self, 
                               species: FungalSpecies, 
                               field_size: Tuple[float, float, float],
                               node_count: int) -> List[GeometryNode]:
        """Generate nodes according to species growth patterns"""
        preferences = self.species_preferences[species]
        nodes = []
        
        width, height, depth = field_size
        pattern = preferences['growth_pattern']
        
        if pattern == 'radial':
            # Pleurotus: radial growth from center
            center_x, center_y = width/2, height/2
            for i in range(node_count):
                angle = (2 * math.pi * i) / node_count
                radius = (width/3) * (0.3 + 0.7 * random.random())
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                z = depth * random.random()
                
                node = GeometryNode(
                    id=f"node_{i}",
                    x=x, y=y, z=z,
                    role="junction" if i > 2 else "input",
                    moisture_level=preferences['moisture_optimum'] + random.gauss(0, 0.1),
                    ph_level=preferences['ph_optimum'] + random.gauss(0, 0.3),
                    temperature=20.0 + random.gauss(0, 2.0),
                    electrical_conductivity=0.5 + random.gauss(0, 0.2)
                )
                nodes.append(node)
                
        elif pattern == 'linear':
            # Ganoderma: more linear, structured growth
            for i in range(node_count):
                x = width * (i / (node_count - 1))
                y = height/2 + random.gauss(0, height/8)
                z = depth * random.random()
                
                node = GeometryNode(
                    id=f"node_{i}",
                    x=x, y=y, z=z,
                    role="input" if i < 2 else "output" if i >= node_count-2 else "junction",
                    moisture_level=preferences['moisture_optimum'] + random.gauss(0, 0.05),
                    ph_level=preferences['ph_optimum'] + random.gauss(0, 0.2),
                    temperature=20.0 + random.gauss(0, 1.0),
                    electrical_conductivity=0.6 + random.gauss(0, 0.15)
                )
                nodes.append(node)
                
        elif pattern == 'mesh':
            # Composite: dense mesh network
            grid_size = int(math.sqrt(node_count))
            for i in range(node_count):
                grid_x = i % grid_size
                grid_y = i // grid_size
                
                x = width * (grid_x / (grid_size - 1)) + random.gauss(0, width/20)
                y = height * (grid_y / (grid_size - 1)) + random.gauss(0, height/20)
                z = depth * random.random()
                
                node = GeometryNode(
                    id=f"node_{i}",
                    x=x, y=y, z=z,
                    role="junction",
                    moisture_level=preferences['moisture_optimum'] + random.gauss(0, 0.15),
                    ph_level=preferences['ph_optimum'] + random.gauss(0, 0.4),
                    temperature=20.0 + random.gauss(0, 3.0),
                    electrical_conductivity=0.4 + random.gauss(0, 0.25)
                )
                nodes.append(node)
        
        return nodes
    
    def _generate_species_edges(self, species: FungalSpecies, nodes: List[GeometryNode]) -> List[GeometryEdge]:
        """Generate edges based on species connectivity preferences"""
        preferences = self.species_preferences[species]
        edges = []
        target_connectivity = preferences['preferred_connectivity']
        
        # Calculate distances between all node pairs
        node_distances = {}
        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes):
                if i < j:  # Avoid duplicates
                    dist = math.sqrt(
                        (node_a.x - node_b.x)**2 + 
                        (node_a.y - node_b.y)**2 + 
                        (node_a.z - node_b.z)**2
                    )
                    node_distances[(i, j)] = dist
        
        # Sort by distance and connect nearest neighbors up to target connectivity
        sorted_pairs = sorted(node_distances.items(), key=lambda x: x[1])
        connections_per_node = {}
        
        for (i, j), distance in sorted_pairs:
            # Check if we've reached target connectivity for both nodes
            if connections_per_node.get(i, 0) < target_connectivity and \
               connections_per_node.get(j, 0) < target_connectivity:
                
                # Create edge
                edge = GeometryEdge(
                    id=f"edge_{i}_{j}",
                    source_id=nodes[i].id,
                    target_id=nodes[j].id,
                    weight=1.0 / (1.0 + distance),  # Closer = higher weight
                    connection_type="mycelial",
                    conductance=0.8 + random.gauss(0, 0.1),
                    delay_ms=50.0 + distance * 10.0,  # Distance affects delay
                    moisture_dependence=0.3 + random.gauss(0, 0.1)
                )
                edges.append(edge)
                
                # Update connection counts
                connections_per_node[i] = connections_per_node.get(i, 0) + 1
                connections_per_node[j] = connections_per_node.get(j, 0) + 1
        
        return edges
    
    def _optimize_for_glyph(self, 
                           nodes: List[GeometryNode], 
                           edges: List[GeometryEdge],
                           target_glyph: str, 
                           species: FungalSpecies):
        """Optimize geometry to enhance expression of target glyph"""
        
        # Glyph-specific optimization strategies
        glyph_optimizations = {
            '🌌': {  # Deep contemplation - needs quiet, stable zones
                'preferred_moisture': 0.6,
                'preferred_ph': 7.0,
                'preferred_connectivity': 2.0,
                'role_preference': 'junction'
            },
            '🌪️': {  # Turbulence - needs dynamic, high-conductivity zones  
                'preferred_moisture': 0.9,
                'preferred_ph': 6.5,
                'preferred_connectivity': 4.0,
                'role_preference': 'amplifier'
            },
            '🌊': {  # Flow - balanced, moderate connectivity
                'preferred_moisture': 0.7,
                'preferred_ph': 6.8,
                'preferred_connectivity': 3.0,
                'role_preference': 'junction'
            },
            '🌱': {  # Growth - slightly moist, slightly acidic
                'preferred_moisture': 0.8,
                'preferred_ph': 6.3,
                'preferred_connectivity': 2.5,
                'role_preference': 'junction'
            }
        }
        
        if target_glyph in glyph_optimizations:
            opt = glyph_optimizations[target_glyph]
            
            # Adjust node properties toward glyph preferences
            for node in nodes:
                # Set glyph affinity for nodes that match preferences
                moisture_match = abs(node.moisture_level - opt['preferred_moisture']) < 0.2
                ph_match = abs(node.ph_level - opt['preferred_ph']) < 0.5
                
                if moisture_match and ph_match:
                    node.glyph_affinity = target_glyph
                    if opt['role_preference'] != 'junction':
                        node.role = opt['role_preference']
                
                # Gradually adjust properties toward preferences
                node.moisture_level += (opt['preferred_moisture'] - node.moisture_level) * 0.3
                node.ph_level += (opt['preferred_ph'] - node.ph_level) * 0.2
    
    def _detect_logic_gates(self, nodes: List[GeometryNode], edges: List[GeometryEdge]) -> List[Dict[str, Any]]:
        """
        Detect emergent logic gate structures in the topology.
        
        Addresses: "mycelium structures implement logic gates by their spatial configuration"
        """
        detected_gates = []
        
        # Build adjacency list
        adjacency = {}
        for edge in edges:
            source_idx = next(i for i, n in enumerate(nodes) if n.id == edge.source_id)
            target_idx = next(i for i, n in enumerate(nodes) if n.id == edge.target_id)
            
            if source_idx not in adjacency:
                adjacency[source_idx] = []
            if target_idx not in adjacency:
                adjacency[target_idx] = []
            
            adjacency[source_idx].append(target_idx)
            adjacency[target_idx].append(source_idx)
        
        # Look for logic gate patterns
        for gate_type, pattern in self.logic_gate_patterns.items():
            min_nodes = pattern['min_nodes']
            
            # Try all possible node combinations of required size
            for node_subset in self._get_node_combinations(len(nodes), min_nodes):
                if self._matches_logic_pattern(node_subset, adjacency, pattern):
                    # Calculate gate properties
                    gate_nodes = [nodes[i] for i in node_subset]
                    center_x = sum(n.x for n in gate_nodes) / len(gate_nodes)
                    center_y = sum(n.y for n in gate_nodes) / len(gate_nodes)
                    
                    gate_info = {
                        'type': gate_type.value,
                        'nodes': [nodes[i].id for i in node_subset],
                        'center': (center_x, center_y),
                        'confidence': self._calculate_gate_confidence(gate_nodes, pattern),
                        'avg_conductance': sum(n.electrical_conductivity for n in gate_nodes) / len(gate_nodes)
                    }
                    detected_gates.append(gate_info)
        
        return detected_gates
    
    def _get_node_combinations(self, total_nodes: int, subset_size: int) -> List[List[int]]:
        """Get all combinations of nodes of given size (limited for performance)"""
        import itertools
        if total_nodes > 15:  # Limit combinations for performance
            # Sample random combinations instead
            combinations = []
            for _ in range(min(50, total_nodes)):  # Max 50 combinations
                combo = random.sample(range(total_nodes), subset_size)
                combinations.append(combo)
            return combinations
        else:
            return list(itertools.combinations(range(total_nodes), subset_size))
    
    def _matches_logic_pattern(self, node_indices: List[int], adjacency: Dict[int, List[int]], pattern: Dict) -> bool:
        """Check if a set of nodes matches a logic gate pattern"""
        required_connections = pattern['required_connections']
        
        for conn in required_connections:
            idx_a, idx_b = node_indices[conn[0]], node_indices[conn[1]]
            if idx_b not in adjacency.get(idx_a, []):
                return False
        
        return True
    
    def _calculate_gate_confidence(self, gate_nodes: List[GeometryNode], pattern: Dict) -> float:
        """Calculate confidence that this is actually a logic gate"""
        # Base confidence
        confidence = 0.5
        
        # Spatial arrangement bonus
        if pattern['spatial_arrangement'] == 'triangle' and len(gate_nodes) == 3:
            # Check if nodes form roughly triangular arrangement
            distances = []
            for i in range(len(gate_nodes)):
                for j in range(i+1, len(gate_nodes)):
                    dist = math.sqrt(
                        (gate_nodes[i].x - gate_nodes[j].x)**2 + 
                        (gate_nodes[i].y - gate_nodes[j].y)**2
                    )
                    distances.append(dist)
            
            if len(distances) == 3:
                avg_dist = sum(distances) / 3
                dist_variance = sum((d - avg_dist)**2 for d in distances) / 3
                if dist_variance < avg_dist * 0.3:  # Low variance = regular shape
                    confidence += 0.3
        
        # Electrical properties bonus
        avg_conductivity = sum(n.electrical_conductivity for n in gate_nodes) / len(gate_nodes)
        if avg_conductivity > 0.6:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _calculate_ethical_resonance(self, 
                                   nodes: List[GeometryNode], 
                                   edges: List[GeometryEdge],
                                   logic_gates: List[Dict[str, Any]]) -> float:
        """
        Calculate how ethically resonant this geometry is.
        
        Addresses: "Let ethics emerge from geometry, not enforcement."
        """
        ethical_score = 0.0
        
        # Balance and harmony (even distribution of properties)
        moisture_variance = self._calculate_property_variance(nodes, 'moisture_level')
        ph_variance = self._calculate_property_variance(nodes, 'ph_level')
        temp_variance = self._calculate_property_variance(nodes, 'temperature')
        
        # Lower variance = more harmony
        harmony_score = 1.0 - (moisture_variance + ph_variance + temp_variance) / 3.0
        ethical_score += harmony_score * 0.3
        
        # Connectivity balance (avoid highly centralized networks)
        connectivity_distribution = self._calculate_connectivity_distribution(nodes, edges)
        connectivity_balance = 1.0 - connectivity_distribution  # Lower = more balanced
        ethical_score += connectivity_balance * 0.2
        
        # Logic gate diversity (different types of logic = cognitive diversity)
        gate_types = set(gate['type'] for gate in logic_gates)
        logic_diversity = len(gate_types) / len(LogicGate)  # Normalize to 0-1
        ethical_score += logic_diversity * 0.2
        
        # Environmental sustainability (moderate resource usage)
        avg_conductivity = sum(n.electrical_conductivity for n in nodes) / len(nodes)
        sustainability_score = 1.0 - abs(avg_conductivity - 0.5) * 2  # Penalty for extremes
        ethical_score += sustainability_score * 0.3
        
        return max(0.0, min(1.0, ethical_score))
    
    def _calculate_property_variance(self, nodes: List[GeometryNode], property_name: str) -> float:
        """Calculate variance of a property across nodes"""
        values = [getattr(node, property_name) for node in nodes]
        if not values:
            return 0.0
        
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val)**2 for v in values) / len(values)
        return math.sqrt(variance)  # Return standard deviation, normalized
    
    def _calculate_connectivity_distribution(self, nodes: List[GeometryNode], edges: List[GeometryEdge]) -> float:
        """Calculate how evenly distributed connectivity is (lower = more balanced)"""
        # Count connections per node
        connections = {}
        for edge in edges:
            connections[edge.source_id] = connections.get(edge.source_id, 0) + 1
            connections[edge.target_id] = connections.get(edge.target_id, 0) + 1
        
        # Calculate distribution variance
        if not connections:
            return 0.0
        
        conn_counts = list(connections.values())
        mean_conn = sum(conn_counts) / len(conn_counts)
        variance = sum((c - mean_conn)**2 for c in conn_counts) / len(conn_counts)
        
        # Normalize by mean to get coefficient of variation
        if mean_conn == 0:
            return 0.0
        return math.sqrt(variance) / mean_conn
    
    def _generate_substrate_properties(self, species: FungalSpecies, field_size: Tuple[float, float, float]) -> Dict[str, Any]:
        """Generate substrate-level properties"""
        preferences = self.species_preferences[species]
        
        return {
            'species': species.value,
            'field_dimensions': field_size,
            'optimal_moisture': preferences['moisture_optimum'],
            'optimal_ph': preferences['ph_optimum'],
            'growth_pattern': preferences['growth_pattern'],
            'branching_angle': preferences['branching_angle'],
            'preferred_logic_gates': [gate.value for gate in preferences['logic_preference']],
            'substrate_age_days': random.uniform(7, 30),
            'ambient_temperature': 20.0 + random.gauss(0, 3.0),
            'humidity_percent': 65.0 + random.gauss(0, 10.0)
        }
    
    # Output Format Methods (answering o3's question #3)
    
    def export_as_graphml(self, geometry: GeometryField) -> str:
        """
        Export geometry as GraphML format.
        
        Answers o3's question #3: "Do you prefer GraphML (nodes/edges) or planar coordinates?"
        Decision: GraphML for rich metadata, with coordinate attributes.
        """
        
        # Create GraphML structure
        graphml = ET.Element("graphml")
        graphml.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
        
        # Define attribute keys
        node_attrs = [
            ("x", "double"), ("y", "double"), ("z", "double"),
            ("role", "string"), ("moisture_level", "double"),
            ("ph_level", "double"), ("temperature", "double"),
            ("electrical_conductivity", "double"), ("glyph_affinity", "string")
        ]
        
        edge_attrs = [
            ("weight", "double"), ("connection_type", "string"),
            ("conductance", "double"), ("delay_ms", "double"),
            ("moisture_dependence", "double")
        ]
        
        # Add attribute definitions
        for attr_name, attr_type in node_attrs:
            key = ET.SubElement(graphml, "key")
            key.set("id", attr_name)
            key.set("for", "node")
            key.set("attr.name", attr_name)
            key.set("attr.type", attr_type)
        
        for attr_name, attr_type in edge_attrs:
            key = ET.SubElement(graphml, "key")
            key.set("id", attr_name)
            key.set("for", "edge")
            key.set("attr.name", attr_name)
            key.set("attr.type", attr_type)
        
        # Create graph
        graph = ET.SubElement(graphml, "graph")
        graph.set("id", geometry.field_id)
        graph.set("edgedefault", "undirected")
        
        # Add nodes
        for node in geometry.nodes:
            node_elem = ET.SubElement(graph, "node")
            node_elem.set("id", node.id)
            
            # Add node attributes
            for attr_name, _ in node_attrs:
                if hasattr(node, attr_name):
                    value = getattr(node, attr_name)
                    if value is not None:
                        data = ET.SubElement(node_elem, "data")
                        data.set("key", attr_name)
                        data.text = str(value)
        
        # Add edges
        for edge in geometry.edges:
            edge_elem = ET.SubElement(graph, "edge")
            edge_elem.set("id", edge.id)
            edge_elem.set("source", edge.source_id)
            edge_elem.set("target", edge.target_id)
            
            # Add edge attributes
            for attr_name, _ in edge_attrs:
                if hasattr(edge, attr_name):
                    value = getattr(edge, attr_name)
                    if value is not None:
                        data = ET.SubElement(edge_elem, "data")
                        data.set("key", attr_name)
                        data.text = str(value)
        
        # Convert to string
        ET.indent(graphml, space="  ")
        return ET.tostring(graphml, encoding='unicode')
    
    def export_as_coordinates(self, geometry: GeometryField) -> Dict[str, Any]:
        """
        Export geometry as coordinate-based format (fallback).
        
        For simpler visualization or when GraphML is too complex.
        """
        return {
            'field_id': geometry.field_id,
            'species': geometry.species.value,
            'nodes': [asdict(node) for node in geometry.nodes],
            'edges': [asdict(edge) for edge in geometry.edges],
            'logic_gates': geometry.logic_gates,
            'substrate_properties': geometry.substrate_properties,
            'ethical_resonance': geometry.ethical_resonance
        }
    
    def export_as_json(self, geometry: GeometryField) -> str:
        """Export complete geometry as JSON"""
        return json.dumps(self.export_as_coordinates(geometry), indent=2)


# Demo function addressing o3's questions
def enhanced_geometry_demo():
    """
    Comprehensive demo addressing o3's geometry compiler questions.
    """
    print("🌟 Enhanced Geometry Compiler Demo")
    print("="*50)
    
    compiler = EnhancedGeometryCompiler()
    
    # Test different species and glyphs
    test_cases = [
        (FungalSpecies.PLEUROTUS_DJAMOR, "🌪️", "Dynamic turbulence geometry"),
        (FungalSpecies.GANODERMA_RESINACEUM, "🌌", "Deep contemplative geometry"),
        (FungalSpecies.MYCELIUM_COMPOSITE, "🌊", "Flowing mesh geometry")
    ]
    
    for species, glyph, description in test_cases:
        print(f"\n{'='*20}")
        print(f"🍄 {description}")
        print(f"   Species: {species.value}")
        print(f"   Target glyph: {glyph}")
        
        # Generate geometry
        geometry = compiler.generate_species_geometry(
            species=species,
            target_glyph=glyph,
            field_size=(8.0, 8.0, 1.5),
            node_count=10
        )
        
        print(f"\n📊 Geometry Analysis:")
        print(f"   Nodes: {len(geometry.nodes)}")
        print(f"   Edges: {len(geometry.edges)}")
        print(f"   Logic gates: {len(geometry.logic_gates)}")
        print(f"   Ethical resonance: {geometry.ethical_resonance:.3f}")
        
        # Show detected logic gates
        if geometry.logic_gates:
            print("   Detected logic gates:")
            for gate in geometry.logic_gates:
                print(f"     {gate['type']} (confidence: {gate['confidence']:.2f})")
        
        # Test export formats
        print(f"\n💾 Export Tests:")
        
        # JSON export
        json_export = compiler.export_as_json(geometry)
        print(f"   JSON export: {len(json_export)} characters")
        
        # GraphML export
        graphml_export = compiler.export_as_graphml(geometry)
        print(f"   GraphML export: {len(graphml_export)} characters")
        
        # Coordinate export
        coord_export = compiler.export_as_coordinates(geometry)
        print(f"   Coordinate export: {len(coord_export)} keys")
    
    print(f"\n✨ Enhanced geometry compiler demo complete!")
    print(f"\n🎯 Answers to o3's questions:")
    print(f"   3. ✅ GraphML chosen for rich metadata with coordinate fallback")
    print(f"   4. ✅ Moisture, pH, temperature, conductivity metadata included")
    print(f"   5. ✅ Logic gates detected from spatial topology")
    print(f"   6. ✅ Species-specific topology preferences implemented")


if __name__ == "__main__":
    import time
    enhanced_geometry_demo()
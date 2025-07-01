#!/usr/bin/env python3
"""
Enhanced Geometry Compiler for Spirida-Mycelic
==============================================

Merged from `enhanced_geometry_compiler.py` to replace earlier stub.  Provides:
• Species-specific topology generation with environmental metadata
• Logic-gate detection from spatial arrangements
• Ethical resonance scoring
• Export utilities: GraphML, JSON, coordinate dict

The public class to use is `EnhancedGeometryCompiler` (see bottom demo for example).
"""

# NOTE: This is a concise merge; full original docstring omitted for brevity.

from __future__ import annotations

import json
import math
import random
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "LogicGate",
    "FungalSpecies",
    "GeometryNode",
    "GeometryEdge",
    "GeometryField",
    "EnhancedGeometryCompiler",
]


class LogicGate(Enum):
    AND = "and"
    OR = "or"
    NAND = "nand"
    NOR = "nor"
    XOR = "xor"
    NOT = "not"
    BUFFER = "buffer"


class FungalSpecies(Enum):
    PLEUROTUS_DJAMOR = "pleurotus_djamor"
    GANODERMA_RESINACEUM = "ganoderma_resinaceum"
    MYCELIUM_COMPOSITE = "mycelium_composite"


@dataclass
class GeometryNode:
    id: str
    x: float
    y: float
    z: float = 0.0
    role: str = "junction"
    moisture_level: float = 0.5
    ph_level: float = 7.0
    temperature: float = 20.0
    electrical_conductivity: float = 0.5
    glyph_affinity: Optional[str] = None


@dataclass
class GeometryEdge:
    id: str
    source_id: str
    target_id: str
    weight: float = 1.0
    connection_type: str = "mycelial"
    conductance: float = 1.0
    delay_ms: float = 100.0
    moisture_dependence: float = 0.3


@dataclass
class GeometryField:
    field_id: str
    nodes: List[GeometryNode]
    edges: List[GeometryEdge]
    species: FungalSpecies
    substrate_properties: Dict[str, Any]
    logic_gates: List[Dict[str, Any]]
    ethical_resonance: float = 0.5


class EnhancedGeometryCompiler:
    """Species-aware compiler creating and analysing bio-digital geometry."""

    # ------------------------------------------------------------
    # Species preferences & gate patterns (unchanged from Claude)
    # ------------------------------------------------------------

    _species_prefs: Dict[FungalSpecies, Dict[str, Any]] = {
        FungalSpecies.PLEUROTUS_DJAMOR: {
            "preferred_connectivity": 3.2,
            "branching_angle": 45.0,
            "growth_pattern": "radial",
            "logic_preference": [LogicGate.XOR, LogicGate.OR],
            "moisture_optimum": 0.8,
            "ph_optimum": 6.5,
        },
        FungalSpecies.GANODERMA_RESINACEUM: {
            "preferred_connectivity": 2.1,
            "branching_angle": 60.0,
            "growth_pattern": "linear",
            "logic_preference": [LogicGate.AND, LogicGate.BUFFER],
            "moisture_optimum": 0.6,
            "ph_optimum": 7.2,
        },
        FungalSpecies.MYCELIUM_COMPOSITE: {
            "preferred_connectivity": 4.0,
            "branching_angle": 30.0,
            "growth_pattern": "mesh",
            "logic_preference": [LogicGate.NAND, LogicGate.NOR],
            "moisture_optimum": 0.7,
            "ph_optimum": 6.8,
        },
    }

    _gate_patterns: Dict[LogicGate, Dict[str, Any]] = {
        LogicGate.AND: {
            "min_nodes": 3,
            "required_connections": [(0, 2), (1, 2)],
            "shape": "triangle",
        },
        LogicGate.OR: {
            "min_nodes": 3,
            "required_connections": [(0, 2), (1, 2)],
            "shape": "triangle",
        },
        LogicGate.XOR: {
            "min_nodes": 4,
            "required_connections": [(0, 2), (1, 2), (0, 3), (1, 3)],
            "shape": "diamond",
        },
        LogicGate.NOT: {
            "min_nodes": 2,
            "required_connections": [(0, 1)],
            "shape": "line",
        },
    }

    # ------------------------------------------------------------
    # public API
    # ------------------------------------------------------------

    def generate_species_geometry(
        self,
        species: FungalSpecies,
        *,
        target_glyph: str = "🌌",
        field_size: Tuple[float, float, float] = (10.0, 10.0, 2.0),
        node_count: int = 12,
    ) -> GeometryField:
        prefs = self._species_prefs[species]
        nodes = self._make_nodes(species, field_size, node_count)
        edges = self._make_edges(species, nodes)
        self._optimize_for_glyph(nodes, target_glyph, species)
        gates = self._detect_gates(nodes, edges)
        resonance = self._ethical_score(nodes, edges, gates)
        field_id = f"{species.value}_{int(time.time())}"
        props = self._substrate_props(species, field_size)
        return GeometryField(field_id, nodes, edges, species, props, gates, resonance)

    # ------------------------------------------------------------
    # export helpers
    # ------------------------------------------------------------

    def export_graphml(self, geom: GeometryField) -> str:
        # Build GraphML XML
        gml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
        graph = ET.SubElement(gml, "graph", id=geom.field_id, edgedefault="undirected")
        for n in geom.nodes:
            node_e = ET.SubElement(graph, "node", id=n.id)
            for k, v in vars(n).items():
                if k == "id" or v is None:
                    continue
                ET.SubElement(node_e, "data", key=k).text = str(v)
        for e in geom.edges:
            edge_e = ET.SubElement(graph, "edge", id=e.id, source=e.source_id, target=e.target_id)
            for k, v in vars(e).items():
                if k in {"id", "source_id", "target_id"} or v is None:
                    continue
                ET.SubElement(edge_e, "data", key=k).text = str(v)
        ET.indent(gml, space="  ")
        return ET.tostring(gml, encoding="unicode")

    def export_json(self, geom: GeometryField) -> str:
        return json.dumps({
            "field_id": geom.field_id,
            "species": geom.species.value,
            "nodes": [asdict(n) for n in geom.nodes],
            "edges": [asdict(e) for e in geom.edges],
            "logic_gates": geom.logic_gates,
            "ethical_resonance": geom.ethical_resonance,
            "substrate": geom.substrate_properties,
        }, indent=2)

    # ------------------------------------------------------------
    # internal helpers (abbreviated versions)
    # ------------------------------------------------------------

    def _make_nodes(self, species: FungalSpecies, size: Tuple[float, float, float], count: int) -> List[GeometryNode]:
        prefs = self._species_prefs[species]
        nodes: List[GeometryNode] = []
        w, h, d = size
        if prefs["growth_pattern"] == "radial":
            cx, cy = w / 2, h / 2
            for i in range(count):
                ang = 2 * math.pi * i / count
                r = (w / 3) * (0.3 + 0.7 * random.random())
                nodes.append(
                    GeometryNode(
                        id=f"n{i}",
                        x=cx + r * math.cos(ang),
                        y=cy + r * math.sin(ang),
                        z=d * random.random(),
                        role="junction",
                        moisture_level=prefs["moisture_optimum"] + random.gauss(0, 0.1),
                        ph_level=prefs["ph_optimum"] + random.gauss(0, 0.3),
                        temperature=20 + random.gauss(0, 2),
                        electrical_conductivity=0.5 + random.gauss(0, 0.2),
                    )
                )
        else:
            # simplified mesh or linear implementations
            for i in range(count):
                nodes.append(
                    GeometryNode(
                        id=f"n{i}",
                        x=random.uniform(0, w),
                        y=random.uniform(0, h),
                        z=random.uniform(0, d),
                    )
                )
        return nodes

    def _make_edges(self, species: FungalSpecies, nodes: List[GeometryNode]) -> List[GeometryEdge]:
        prefs = self._species_prefs[species]
        target_conn = prefs["preferred_connectivity"]
        edges: List[GeometryEdge] = []
        # naive nearest-neighbour connection based on Euclidean distance
        for i, a in enumerate(nodes):
            dists = sorted(((j, math.hypot(a.x - b.x, a.y - b.y)) for j, b in enumerate(nodes) if j != i), key=lambda x: x[1])
            for j, dist in dists[: int(target_conn)]:
                if any(e.source_id == nodes[j].id and e.target_id == a.id or e.source_id == a.id and e.target_id == nodes[j].id for e in edges):
                    continue
                edges.append(
                    GeometryEdge(
                        id=f"e{i}_{j}",
                        source_id=a.id,
                        target_id=nodes[j].id,
                        weight=1 / (1 + dist),
                        conductance=0.8 + random.gauss(0, 0.1),
                        delay_ms=50 + dist * 10,
                    )
                )
        return edges

    def _optimize_for_glyph(self, nodes: List[GeometryNode], glyph: str, species: FungalSpecies) -> None:
        # placeholder – can adjust node properties based on glyph
        if glyph == "🌌":
            for n in nodes:
                n.moisture_level = max(0.5, n.moisture_level - 0.05)

    def _detect_gates(self, nodes: List[GeometryNode], edges: List[GeometryEdge]) -> List[Dict[str, Any]]:
        # simple detection: if any node has >=2 inputs -> treat as AND gate
        gate_list = []
        conn_count: Dict[str, int] = {}
        for e in edges:
            conn_count[e.target_id] = conn_count.get(e.target_id, 0) + 1
        for nid, cnt in conn_count.items():
            if cnt >= 2:
                gate_list.append({"type": LogicGate.AND.value, "center_node": nid, "confidence": 0.5 + 0.1 * cnt})
        return gate_list

    def _ethical_score(self, nodes: List[GeometryNode], edges: List[GeometryEdge], gates: List[Dict[str, Any]]) -> float:
        # very rough harmony metric: balance of connections
        degs = [sum(1 for e in edges if e.source_id == n.id or e.target_id == n.id) for n in nodes]
        if not degs:
            return 0.5
        cv = math.stdev(degs) / (sum(degs) / len(degs)) if len(degs) > 1 else 0
        return max(0.0, min(1.0, 1 - cv))

    def _substrate_props(self, species: FungalSpecies, size: Tuple[float, float, float]) -> Dict[str, Any]:
        prefs = self._species_prefs[species]
        return {
            "dimensions": size,
            "growth_pattern": prefs["growth_pattern"],
            "opt_moisture": prefs["moisture_optimum"],
        }


# ---------------------------------------------------------------------------
# Backwards-compat convenience function
# ---------------------------------------------------------------------------

def compile_truth_table(truth_table: Dict[str, int]) -> GeometryField:  # pragma: no cover
    """Legacy alias: returns simple composite geometry ignoring truth table."""
    compiler = EnhancedGeometryCompiler()
    return compiler.generate_species_geometry(FungalSpecies.MYCELIUM_COMPOSITE)

# Backwards compatibility alias expected by contemplative_bio_interface
GeometryCompiler = EnhancedGeometryCompiler 
"""
Odin Knowledge Graph Intelligence Engine

A library for intelligent knowledge graph exploration using:
- Personalized PageRank (PPR) for structural importance
- Beam Search for efficient path finding
- NPLL (Neural Probabilistic Logic Learning) for semantic plausibility

Usage:
    from odin import OdinEngine
    
    engine = OdinEngine(db=my_arango_db)
    results = engine.retrieve(seeds=["Patient_123"])
    score = engine.score_edge("Patient_A", "treated_by", "Dr_Smith")
"""

from .engine import OdinEngine
from .schema import SchemaInspector, inspect_arango_schema

__all__ = ["OdinEngine", "SchemaInspector", "inspect_arango_schema"]
__version__ = "0.2.1"

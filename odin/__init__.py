"""
Odin Knowledge Graph Intelligence Engine

A library for intelligent knowledge graph exploration using:
- Personalized PageRank (PPR) for structural importance
- Beam Search for efficient path finding
- NPLL (Neural Probabilistic Logic Learning) for semantic plausibility

Usage:
    from odin import OdinEngine
    from odin.backends.arango import ArangoBackend, ArangoGraphConfig

    graph = ArangoGraphConfig(
        node_collection="MyEntities",
        edge_collection="MyRelationships",
        relation_field="predicate",
    )
    engine = OdinEngine(ArangoBackend(my_arango_db, graph))
    results = engine.retrieve(seeds=["Patient_123"])
    score = engine.score_edge("Patient_A", "treated_by", "Dr_Smith")
"""

def __getattr__(name):
    # Keep backend contracts importable before engine/NPLL initialization.
    if name == "OdinEngine":
        from .engine import OdinEngine
        return OdinEngine
    if name == "inspect_schema":
        from .schema import inspect_schema
        return inspect_schema
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OdinEngine", "inspect_schema"]
__version__ = "0.3.0"

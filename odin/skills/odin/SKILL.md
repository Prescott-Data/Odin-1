---
name: odin
description: Use when writing or reviewing Python that uses the Odin graph-intelligence engine (the `odin-engine` package, `from odin import OdinEngine`). Covers correct API usage, the exact `retrieve()` result shape, edge/motif/triage structures, adapters, and the mistakes AI agents most often make with Odin.
---

# Odin

Odin is a Python library for guided knowledge-graph retrieval. Given seed entities it returns ranked, scored paths through a graph using Personalized PageRank, beam search, and learned edge-plausibility scoring (NPLL). It is the compass, not the explorer: it returns structured evidence, the calling agent interprets it.

Package: `odin-engine`. Import name: `odin`.

This source checkout documents the unreleased backend API. Install it with
`pip install -e ".[arango]"` for ArangoDB or `pip install -e .` for core only.
Published 0.3.0 does not provide these backend interfaces.

## Setup

```python
from arango import ArangoClient
from odin import OdinEngine
from odin.backends.arango import ArangoBackend, ArangoGraphConfig

client = ArangoClient(hosts="http://localhost:8529")
db = client.db("my_graph", username="root", password="")

# OdinEngine takes a graph backend. ArangoBackend keeps credentials in the
# already-connected database handle and requires an explicit graph mapping.
graph = ArangoGraphConfig(
    node_collection="CaseRecords",
    edge_collection="EvidenceLinks",
    relation_field="predicate",
)
backend = ArangoBackend(db, graph)
engine = OdinEngine(
    backend,
    community_id="global",   # scope; "global" explores the whole graph
    cache_size=5000,
    auto_train=True,          # trains NPLL from the graph on first run, persists weights in ArangoDB
    community_mode="none",    # "none" = global, "mapping" = scoped to community_id
)
```

## Core API (exact signatures)

```python
engine.retrieve(seeds, max_paths=50, hop_limit=3, beam_width=64) -> dict
engine.score_edge(src, rel, dst) -> float           # learned edge plausibility; not proof or calibrated source truth
engine.find_anchors(seeds, topn=20) -> list[tuple[str, float]]   # (node_id, ppr_score)
engine.get_neighbors(node_id) -> dict                # {"node", "neighbors", "degree"}
engine.retrain_model() -> bool
engine.has_npll                                      # property: True if NPLL model active
engine.get_status() -> dict
```

`seeds` are entity IDs (strings). In ArangoDB they use the configured
`collection/key` form, e.g. `"CaseRecords/claim_123"`.

## The `retrieve()` result shape (READ THIS BEFORE USING RESULTS)

```python
{
    "topk_ppr": [...],                  # anchor nodes by PageRank
    "paths": [
        {
            "id": "path_0",
            "score": 0.94,
            "edges": [                  # a path is described by its EDGES
                {"u": "entity/A", "v": "entity/B", "relation": "billed_by",
                 "u_label": "...", "v_label": "...", "confidence": 0.89,
                 "created_at": "...", "provenance": {...}}
            ]
        }
    ],
    "evidence_strength": 0.0,           # 0..1
    "community_relevance": 0.0,         # 0..1
    "insight_score": 0.0,               # 0..1 overall quality
    "aggregates": {
        "motifs": [
            {"pattern": "billed_by->flagged_in", "edge_count": 24,
             "path_count": 12, "avg_edge_conf": 0.88, "median_recency_days": 5.0}
        ],
        "relation_share": {"billed_by": {"count": 42, "share": 0.42}},
        "snippet_anchors": [...],
        "summary": {"total_paths": ..., "provenance": ..., "recency": ...,
                    "label_coverage": ..., "motif_density": ..., "low_support": false}
    },
    "triage": {"score": 87, "components": {...}, "dominant_relation": {...}},  # score is 0-100
    "ics": {...},
    "used_budget": {...},
    "trace": {"timings_ms": {"total": 412}, ...}
}
```

Deriving the node sequence of a path (there is no `nodes` field):

```python
for p in result["paths"]:
    edges = p["edges"]
    nodes = [edges[0]["u"], *(e["v"] for e in edges)] if edges else []
    print(f"[{p['score']:.2f}]", " -> ".join(str(n) for n in nodes))
```

Gate agent work on the triage score:

```python
if result["triage"]["score"] >= 70:
    agent.investigate(result["paths"])
```

## Common mistakes to AVOID

- **Do NOT use `path["nodes"]`.** Paths have no `nodes` key. Use `path["edges"]` (each edge has `u`, `v`, `relation`) and derive nodes as shown above.
- **Motif fields are `edge_count` and `path_count`, not `count`.** The `pattern` is a relation sequence like `"billed_by->flagged_in"`, not a node chain.
- **`relation_share[rel]` is `{"count": int, "share": float}`**, not a bare float.
- **`score_edge` argument order is `(src, rel, dst)`** (source, relation, destination).
- **`beam_width` default is 64** (not 10).
- **`OdinEngine` takes a `GraphBackend`**, not a connection string or raw database handle. Build `ArangoBackend(db, graph)` from the connected ArangoDB handle and required `ArangoGraphConfig`.
- **Check `engine.has_npll`** before relying on fine-grained plausibility; if False, training was explicitly disabled. Training failures raise; failed retraining preserves the active model.

## Other backends (adapters)

`OdinEngine` works with any backend that supplies the complete `GraphAccessor`
surface. `ArangoBackend` supplies both retrieval and NPLL training. A
retrieval-only backend must be passed with `auto_train=False`; a backend without
training capabilities raises `BackendCapabilityError` if training is requested.
`JanusGraphAccessor` ships for direct orchestrator composition:

Install its optional driver from this checkout: `pip install -e ".[gremlin]"`.

```python
from retrieval.orchestrator import RetrievalOrchestrator, OrchestratorParams
from retrieval.confidence import ConstantConfidence
from retrieval.adapters_janus import JanusGraphAccessor

orch = RetrievalOrchestrator(accessor=JanusGraphAccessor(graph),
                             edge_confidence=ConstantConfidence(0.8))
result = orch.retrieve(seeds=["v1"],
                       params=OrchestratorParams(community_id="global",
                                                 max_paths=50, hop_limit=3, beam_width=64))
```

NPLL bootstrap is backend-neutral. ArangoBackend currently supplies the training
source and model store. Custom backends may implement both; retrieval-only
backends require `auto_train=False`. Direct orchestrator users supply confidence.

## Evidence mappings and training

Arango optional fields are opt-in. Raw assertions live once under
`provenance.assertion`; canonical `created_at` comes only from the configured
timestamp field. Application documents and vectors are returned unchanged;
model-input filtering belongs to the consuming tool layer.
Bridge/affinity mappings require explicit entity, strength, community, endpoint,
and score fields. Algorithm filters require explicit field mappings and values.
Use full Arango document IDs in bridge and membership records. Community signals
are optional retrieval features and have no role in model storage. Configured
membership lookup requires one distinct community per entity after algorithm
filtering; multiple distinct memberships raise `BackendConfigurationError` during
scoring. Multi-membership scoring is deferred.

Global training artifacts are shared across retrieval communities. Neural scorers
learn from known facts and uniform corruptions of both endpoints; E-M holdout
facts are excluded from supervision. Reload loads learned tensor state from an
integrity-checked model blob without training. Evidence outputs preserve vectors.
Older numeric artifact versions and stale config schemas retrain. Newer versions
raise `NewerModelVersionError` and must never be replaced by an older reader.
Damaged blobs raise; runtime/device differences do not invalidate weights. Rule mining uses bounded sampled motifs
and confidence thresholds; grounding reuses observed joins and sampled contrasts.

## Schema introspection

```python
from odin import inspect_schema

schema = inspect_schema(backend)
inspect_schema(backend, output_file="schema.json")
```

## Reference

Full documentation: https://odin.developers.prescottdata.io. The Reference section (`/reference/`) has the authoritative API and result schema.

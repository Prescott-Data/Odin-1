---
icon: material/graph-outline
---

# Using Neo4j (Experimental)

Odin's retrieval layer — [PPR](../concepts/ppr.md), [beam search](../concepts/beam-search.md), and [path scoring](../concepts/scoring.md) — can run against a Neo4j labeled property graph through the experimental `Neo4jGraphAccessor`. This page shows how to wire it up and, just as importantly, where the boundary is.

!!! warning "Experimental scope"
    This adapter covers the **retrieval layer only**. `OdinEngine`'s
    auto-bootstrap (NPLL training and weight persistence) and
    `SchemaInspector` still require ArangoDB. Opening that boundary is the
    next step on the roadmap. To score edges over Neo4j today, provide your
    own `EdgeConfidenceProvider` (or a constant confidence).

## Install

The Neo4j driver is an optional extra, so existing installs are unaffected:

```bash
pip install "odin-engine[neo4j]"
```

## Wire up the accessor

```python
from neo4j import GraphDatabase
from retrieval.adapters_neo4j import Neo4jGraphAccessor
from retrieval.cache import CachedGraphAccessor
from retrieval.confidence import ConstantConfidence
from retrieval.orchestrator import RetrievalOrchestrator, OrchestratorParams

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "..."))

accessor = Neo4jGraphAccessor(
    driver,
    database="neo4j",          # optional Neo4j database name
    node_label="Entity",       # optional: scope matches to one label
    id_property="id",          # node property holding the stable node ID
    weight_property="weight",  # relationship property for edge weight (default 1.0)
    community_property="communityId",  # used by nodes(community_id)
)

# Always cache in production — PPR hammers iter_out otherwise.
cached = CachedGraphAccessor(accessor, cache_size=5000)

orchestrator = RetrievalOrchestrator(
    accessor=cached,
    edge_confidence=ConstantConfidence(0.8),
)

result = orchestrator.retrieve(
    seeds=["Patient_123"],
    params=OrchestratorParams(community_id="global", max_paths=50),
)
```

## Graph mapping

| Odin concept | Neo4j source |
|--------------|--------------|
| Node ID | node property (`id_property`, default `id`) |
| Relation | relationship type (`type(r)`) |
| Edge weight | relationship property (`weight_property`, default `weight`, fallback `1.0`) |
| Community | node property (`community_property`) — used only by `nodes(community_id)` |

All queries are parameterized Cypher; node IDs are never interpolated into query text.

## What works and what doesn't

| Capability | Status |
|------------|--------|
| PPR anchors and beam-search retrieval | ✅ works via `RetrievalOrchestrator` |
| Path and insight scoring | ✅ works (bring your own confidence provider) |
| `CachedGraphAccessor` wrapping | ✅ works |
| NPLL auto-training via `OdinEngine` | ❌ requires ArangoDB |
| `SchemaInspector` | ❌ ArangoDB-specific |
| Edge provenance / content search helpers | ❌ ArangoDB accessor extras |

## Conformance

The adapter is covered by a shared conformance suite (`tests/unit/test_adapter_conformance.py`) that every `GraphAccessor` implementation must pass. If you write your own backend adapter, register it there and the contract tests come for free.

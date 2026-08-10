---
icon: material/graph
---

# Data Model

Odin explores a **directed, labeled knowledge graph**: entities connected by typed relationships. This page describes what Odin expects from your data and the terms used throughout the docs.

---

## Entities and relationships

| Term | Meaning | Example |
|------|---------|---------|
| **Entity** (node) | A thing in your domain, addressed by an ID | `entity/claim_123`, `provider/456` |
| **Relationship** (edge) | A typed, directed connection between two entities | `claim_123 —billed_by→ provider_456` |
| **Relation type** | The label on an edge | `billed_by`, `has_diagnosis`, `treated_by` |
| **Path** | An ordered sequence of entities joined by edges | `A → B → C` |
| **Seed** | An entity you start exploration from | `["entity/claim_123"]` |

Node IDs are opaque strings. In ArangoDB they follow the `collection/key` convention (for example `ExtractedEntities/claim_123`), and Odin passes them through unchanged.

---

## Edges carry weight and confidence

Each edge may carry metadata that Odin uses during scoring:

| Field | Used for |
|-------|----------|
| `_from` / `_to` | Direction of the relationship |
| relation / type | The relation label |
| `weight` | Optional structural weight |
| `confidence` | Optional pre-existing confidence (falls back to `weight`) |
| `created_at` | Optional recency signal in aggregation |
| `provenance` | Optional source/document reference |

If confidence and weight are absent, Odin defaults them sensibly (see [Aggregation](aggregation.md)).

---

## Communities

A **community** is a named scope for exploration. It lets you partition a large graph so retrieval stays relevant to one tenant, dataset, or domain.

- `community_mode="none"` — global exploration across the whole graph (the default).
- `community_mode="mapping"` — scope queries to a specific `community_id`.

```python
# Global
engine = OdinEngine(db, community_id="global", community_mode="none")

# Scoped to one community
engine = OdinEngine(db, community_id="medicare_claims", community_mode="mapping")
```

Communities are also the natural unit for NPLL: the model learns the edge patterns of the community it is trained on.

---

## What Odin does *not* require

- **No fixed schema.** Odin discovers collections and fields at runtime — see [Schema Introspection](../guides/schema-introspection.md).
- **No pre-computed embeddings.** NPLL trains directly from your graph's edge structure.
- **No query language on your side.** You pass entity IDs; Odin handles traversal.

---

## Backends

The reference backend is **ArangoDB** (`retrieval/adapters_arango.py`). The accessor layer is an interface, so additional backends (Neo4j, Neptune, Gremlin-compatible stores) can be added by implementing the same node/edge access methods. Contributions of new adapters are welcome — see [Contributing](https://github.com/Prescott-Data/Odin-1/blob/main/CONTRIBUTING.md).

---

## Next

- [Personalized PageRank](ppr.md) — how Odin ranks node importance
- [Connecting ArangoDB](../guides/arangodb.md) — practical connection setup

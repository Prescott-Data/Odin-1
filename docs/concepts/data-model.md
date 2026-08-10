---
icon: material/graph
---

# Data Model

Everything Odin does operates on a **directed, labeled knowledge graph**: entities joined by typed relationships. Before the pipeline pages make sense, it helps to pin down what Odin expects from that graph and the handful of terms used throughout these docs.

## The vocabulary

An **entity** is a thing in your domain, addressed by an ID; a **relationship** is a typed, directed edge between two of them. The label on that edge is its **relation type**. String a few edges together and you have a **path**, and the entity you start a path from is a **seed**.

| Term | Meaning | Example |
|------|---------|---------|
| **Entity** (node) | A thing in your domain, addressed by an ID | `entity/claim_123`, `provider/456` |
| **Relationship** (edge) | A typed, directed connection between two entities | `claim_123 →billed_by→ provider_456` |
| **Relation type** | The label on an edge | `billed_by`, `has_diagnosis`, `treated_by` |
| **Path** | An ordered sequence of entities joined by edges | `A → B → C` |
| **Seed** | An entity you start exploration from | `["entity/claim_123"]` |

Node IDs are opaque strings. In ArangoDB they follow the `collection/key` convention (`ExtractedEntities/claim_123`, say), and Odin passes them straight through without interpreting them.

## What edges can carry

Odin works with a bare `_from`/`_to`/relation edge, but it makes use of richer metadata when it is present. A `weight` gives an edge structural importance; a `confidence` supplies a pre-existing plausibility (and falls back to `weight` when absent); `created_at` feeds the recency signal, and `provenance` links the edge back to a source document.

| Field | Used for |
|-------|----------|
| `_from` / `_to` | Direction of the relationship |
| relation / type | The relation label |
| `weight` | Optional structural weight |
| `confidence` | Optional pre-existing confidence (falls back to `weight`) |
| `created_at` | Optional recency signal in aggregation |
| `provenance` | Optional source/document reference |

None of the optional fields are required (Odin defaults them sensibly), but the more of them your edges carry, the more the [aggregation](aggregation.md) and [triage](scoring.md) signals have to work with. Richer edges simply produce more trustworthy scores.

## Communities scope the graph

A **community** is a named scope for exploration. On a large multi-tenant graph, it keeps a retrieval, and the model behind it, focused on one tenant, dataset, or domain instead of the whole thing. You choose the behavior with `community_mode`:

```python
# Global exploration across the whole graph (default)
engine = OdinEngine(db, community_id="global", community_mode="none")

# Scoped to a single community
engine = OdinEngine(db, community_id="medicare_claims", community_mode="mapping")
```

Communities are also the natural unit for [NPLL](npll.md): a model learns the edge patterns of the community it was trained on, so scoping and semantics line up.

## What Odin does *not* need from you

Notably, there is a lot you do **not** have to prepare. There is no fixed schema to declare, because Odin discovers collections and fields at runtime via [Schema Introspection](../guides/schema-introspection.md). There are no pre-computed embeddings, because NPLL trains straight from the graph's edge structure. And there is no query language to write on your side: you hand Odin entity IDs and it handles the traversal.

The reference backend is **ArangoDB**, but the accessor is an interface, so new backends like Neo4j, Neptune, or other Gremlin-compatible stores can be added by implementing the same node/edge access methods, and [contributions](https://github.com/Prescott-Data/Odin-1/blob/main/CONTRIBUTING.md) of adapters are welcome.

---

With the graph model in hand, the natural next step is how Odin ranks importance within it, [Personalized PageRank](ppr.md), or the practical side of wiring up a database in [Connecting ArangoDB](../guides/arangodb.md).

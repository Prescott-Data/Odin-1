---
icon: material/play
---

# Your First Retrieval

This guide runs the full Odin pipeline once and then walks through every field of the result, so you know exactly what you are getting back and where to reach for each piece of information.

## Run a retrieval

```python
from arango import ArangoClient
from odin import OdinEngine

client = ArangoClient(hosts="http://localhost:8529")
db = client.db("my_graph", username="root", password="")

engine = OdinEngine(db=db, community_id="global")

result = engine.retrieve(
    seeds=["entity/claim_123"],
    max_paths=50,
    hop_limit=3,
    beam_width=64,
)
```

That single call runs PPR, then beam search, then NPLL scoring, then aggregation, and returns everything as one dictionary. Four parameters shape it:

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `seeds` | `list[str]` | required | Entity IDs to start from |
| `max_paths` | `int` | `50` | Maximum paths to return |
| `hop_limit` | `int` | `3` | Maximum path length |
| `beam_width` | `int` | `64` | Paths kept per hop during [beam search](../concepts/beam-search.md) |

How they interact is covered in [Tuning Retrieval](tuning.md); for now the defaults are fine.

## What comes back

The result dictionary has ten keys, and it helps to see them grouped by purpose:

```python
result.keys()
# dict_keys(['topk_ppr', 'paths', 'evidence_strength',
#            'community_relevance', 'insight_score', 'aggregates',
#            'triage', 'ics', 'used_budget', 'trace'])
```

**The paths** are the primary output, returned best-first. Each carries a combined `score` and a list of `edges`; there is no separate `nodes` field, so the node sequence is derived from the edges:

```python
for p in result["paths"][:5]:
    edges = p["edges"]
    nodes = [edges[0]["u"], *(e["v"] for e in edges)] if edges else []
    print(f"[{p['score']:.2f}]", " -> ".join(str(n) for n in nodes))
```

**The triage score** is the single number most agent loops gate on, along with the breakdown that produced it (see [Triage & Insight Scoring](../concepts/scoring.md)):

```python
result["triage"]["score"]         # 0-100 prioritization number
result["triage"]["components"]    # per-component breakdown
```

**The aggregates** turn those paths into patterns, and **`topk_ppr`** exposes the anchors the walk started from:

```python
result["topk_ppr"]                      # structurally important anchor nodes
result["aggregates"]["motifs"]          # recurring patterns
result["aggregates"]["relation_share"]  # relation-type breakdown
result["aggregates"]["summary"]         # provenance, recency, coverage, ...
```

**The quality signals** are there when you want finer-grained observability than the triage score alone:

```python
result["insight_score"]         # 0.0-1.0 overall quality
result["evidence_strength"]     # 0.0-1.0
result["community_relevance"]   # 0.0-1.0
result["ics"]                   # decomposition of insight_score
```

**The trace and budget** are for operability, safe to log or ignore:

```python
result["used_budget"]   # how much exploration budget was consumed
result["trace"]         # ppr/beam traces, params, timings_ms
```

Every field is documented exhaustively in the [Result Schema](../reference/result-schema.md).

## Putting it together

In practice a first look at a retrieval reads like this:

```python
result = engine.retrieve(seeds=["entity/claim_123"], max_paths=25)

print(f"Triage: {result['triage']['score']}/100   "
      f"Insight: {result['insight_score']:.2f}   "
      f"Paths: {len(result['paths'])}")

print("\nTop motifs:")
for m in result["aggregates"]["motifs"][:5]:
    print("  ", m)

print("\nTop paths:")
for p in result["paths"][:5]:
    edges = p["edges"]
    nodes = [edges[0]["u"], *(e["v"] for e in edges)] if edges else []
    print(f"  [{p['score']:.2f}]", " -> ".join(str(n) for n in nodes))
```

## Next

From here, [Scoring Edges](edge-scoring.md) covers the single-edge counterpart to retrieval, [Tuning Retrieval](tuning.md) explains the parameters, and [AI Agent Integration](agent-integration.md) wires the result into an agent loop.

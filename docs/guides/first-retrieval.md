---
icon: material/play
---

# Your First Retrieval

This guide runs the full Odin pipeline and walks through **every field** of the result so you know exactly what you are getting back.

---

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

`retrieve()` runs **PPR → beam search → NPLL scoring → aggregation** and returns a single dictionary.

---

## Parameters

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `seeds` | `list[str]` | — | Entity IDs to start from |
| `max_paths` | `int` | `50` | Maximum paths to return |
| `hop_limit` | `int` | `3` | Maximum path length |
| `beam_width` | `int` | `64` | Paths kept per hop during [beam search](../concepts/beam-search.md) |

See [Tuning Retrieval](tuning.md) for how these interact.

---

## The result at a glance

```python
result.keys()
# dict_keys(['topk_ppr', 'paths', 'evidence_strength',
#            'community_relevance', 'insight_score', 'aggregates',
#            'triage', 'ics', 'used_budget', 'trace'])
```

### Ranked paths

```python
for path in result["paths"][:5]:
    nodes = " → ".join(str(n) for n in path["nodes"])
    print(f"[{path['score']:.2f}] {nodes}")
```

Each path has `nodes`, `edges`, and a combined `score`. Paths are returned best-first.

### Triage score

```python
result["triage"]["score"]         # 0–100 prioritization number
result["triage"]["components"]    # per-component breakdown
```

This is usually the number you gate on. See [Triage & Insight Scoring](../concepts/scoring.md).

### Anchors (PPR)

```python
result["topk_ppr"]   # the structurally important nodes used as anchors
```

### Aggregates

```python
result["aggregates"]["motifs"]          # recurring patterns
result["aggregates"]["relation_share"]  # relation-type breakdown
result["aggregates"]["summary"]         # provenance, recency, coverage, ...
```

### Quality signals

```python
result["insight_score"]         # 0.0–1.0 overall quality
result["evidence_strength"]     # 0.0–1.0
result["community_relevance"]   # 0.0–1.0
result["ics"]                   # decomposition of insight_score
```

### Observability

```python
result["used_budget"]   # how much exploration budget was consumed
result["trace"]         # ppr/beam traces, params, timings_ms
```

The full, authoritative shape is documented in the [Result Schema](../reference/result-schema.md).

---

## A complete example

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
    print(f"  [{p['score']:.2f}]", " → ".join(str(n) for n in p["nodes"]))
```

---

## Next

- [Scoring Edges](edge-scoring.md)
- [Tuning Retrieval](tuning.md)
- [AI Agent Integration](agent-integration.md)

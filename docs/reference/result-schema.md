---
icon: material/code-json
---

# Result Schema

`engine.retrieve(...)` returns a single dictionary. This page documents every top-level field.

```python
result = engine.retrieve(seeds=["entity/claim_123"])
result.keys()
# dict_keys(['topk_ppr', 'paths', 'evidence_strength', 'community_relevance',
#            'insight_score', 'aggregates', 'triage', 'ics',
#            'used_budget', 'trace'])
```

---

## Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `topk_ppr` | `list` | Structurally important nodes (PPR anchors) used to seed exploration |
| `paths` | `list[dict]` | Ranked, scored paths (the primary output) |
| `insight_score` | `float` | Overall retrieval quality, `0.0-1.0` |
| `evidence_strength` | `float` | Strength of supporting evidence, `0.0-1.0` |
| `community_relevance` | `float` | Relevance to the community scope, `0.0-1.0` |
| `aggregates` | `dict` | Motifs, relation shares, and summary statistics |
| `triage` | `dict` | The `0-100` prioritization score and its components |
| `ics` | `dict` | Decomposition of `insight_score` |
| `used_budget` | `dict` | Exploration budget consumed |
| `trace` | `dict` | PPR/beam traces, params, and timings for observability |

---

## `paths`

Each path is best-first ranked:

```python
{
    "nodes": ["entity/A", "entity/B", "entity/C"],
    "edges": [
        {"relation": "billed_by", "score": 0.89},
        ...
    ],
    "score": 0.94,
}
```

| Field | Meaning |
|-------|---------|
| `nodes` | Ordered entity IDs along the path |
| `edges` | The relations connecting the nodes, each with a plausibility `score` |
| `score` | Combined path score (structural + semantic) |

```python
for p in result["paths"][:5]:
    print(f"[{p['score']:.2f}]", " -> ".join(str(n) for n in p["nodes"]))
```

---

## `aggregates`

```python
{
    "motifs":          [{"pattern": "A→B→C", "count": 12}, ...],
    "relation_share":  {"billed_by": 0.42, "has_diagnosis": 0.31, ...},
    "snippet_anchors": [...],
    "summary": {
        "total_paths": 47,
        "unique_motifs": 8,
        "unique_relations": 5,
        "total_edges": 120,
        "provenance": 0.82,
        "recency": 0.74,
        "label_coverage": 0.91,
        "motif_density": 0.36,
        "has_baseline": false,
        "low_support": false,
    },
}
```

See [Motifs & Aggregation](../concepts/aggregation.md) for what each summary field means.

---

## `triage`

```python
{
    "score": 87,                       # 0-100
    "components": {
        "provenance": 0.82,
        "recency": 0.74,
        "surprise": 0.61,
        "motif_density": 0.36,
        "controllability": 1.0,
        "label_coverage": 0.91,
        "penalty": 0.0,
        "low_support": false,
    },
    "dominant_relation": {...},
}
```

The scoring formula and guards are documented in [Triage & Insight Scoring](../concepts/scoring.md).

---

## `trace`

Observability data, safe to log or drop:

```python
{
    "ppr":  {...},                     # PPR trace
    "beam": {...},                     # beam-search trace
    "params": {...},                   # the effective parameters
    "timings_ms": {"total": 412, ...}, # per-stage timing
    "linker_cfg": {...},
}
```

```python
result["trace"]["timings_ms"]["total"]   # end-to-end latency in ms
```

---

## Empty results

When there is insufficient support (for example, seeds with no reachable neighbors), Odin returns a well-formed result with empty `paths`, `insight_score` of `0.0`, and `summary["low_support"] == True`. Always check `len(result["paths"])` before iterating.

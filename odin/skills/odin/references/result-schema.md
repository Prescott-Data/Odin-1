# retrieve() result schema

`engine.retrieve(...)` returns one dictionary with these top-level keys: `topk_ppr`, `paths`, `evidence_strength`, `community_relevance`, `insight_score`, `aggregates`, `triage`, `ics`, `used_budget`, `trace`.

Authoritative page: https://odin.developers.prescottdata.io/reference/result-schema/

| Field | Type | Meaning |
| --- | --- | --- |
| `topk_ppr` | list | Structurally important nodes (PPR anchors) |
| `paths` | list of dict | Ranked, scored paths. The primary output |
| `insight_score` | float | Overall retrieval quality, 0.0 to 1.0 |
| `evidence_strength` | float | 0.0 to 1.0 |
| `community_relevance` | float | 0.0 to 1.0 |
| `aggregates` | dict | Motifs, relation shares, summary statistics |
| `triage` | dict | A 0 to 100 prioritization score and its components |
| `ics` | dict | Decomposition of `insight_score`. Read its keys at runtime |
| `used_budget` | dict | Exploration budget consumed |
| `trace` | dict | PPR and beam traces, effective params, `timings_ms` |

## `paths`

```python
{
    "id": "path_0",
    "score": 0.94,
    "edges": [
        {
            "u": "CaseRecords/claim_123", "v": "CaseRecords/provider_456",
            "relation": "billed_by",
            "u_label": "Claim 123", "v_label": "Provider 456",
            "confidence": 0.89,
            "created_at": "2026-01-04T12:00:00Z",
            "provenance": {...},
        },
    ],
}
```

- A path's `id` is currently `None`; the `"path_0"` in the example is illustrative.
- The documented node sequence is the first edge's `u` followed by every edge's `v`. Paths also carry internal keys such as `nodes` and `decomp`; they are not part of the documented schema, so do not depend on them.
- `u_label`, `v_label`, `created_at` and `provenance` can be absent or empty. `created_at` is set only when the graph mapping configures a timestamp field (`edge_timestamp_field`).
- `confidence` is learned plausibility (or a constant when `auto_train=False`), not a calibrated probability.
- When the Arango backend supplies the raw stored edge, it is under `provenance["assertion"]`.

## `aggregates`

```python
{
    "motifs": [{"pattern": "billed_by->flagged_in", "edge_count": 24,
                "path_count": 12, "avg_edge_conf": 0.88, "median_recency_days": 5.0}],
    "relation_share": {"billed_by": {"count": 42, "share": 0.42}},
    "snippet_anchors": [...],
    "summary": {"total_paths": 47, "unique_motifs": 8, "unique_relations": 5,
                "total_edges": 120, "provenance": 0.82, "recency": 0.74,
                "label_coverage": 0.91, "motif_density": 0.36,
                "has_baseline": False, "low_support": False},
}
```

A motif `pattern` is a relation sequence, not a chain of nodes.

## `triage`

```python
{"score": 87, "components": {"provenance": 0.82, "recency": 0.74, "surprise": 0.61,
                             "motif_density": 0.36, "controllability": 1.0,
                             "label_coverage": 0.91, "penalty": 0.0, "low_support": False},
 "dominant_relation": {...}}
```

`score` is 0 to 100. A penalty of 15.0 means label coverage was below 0.8. Treat the score as a ranking signal and let the application choose any cutoff.

## Empty results

With no reachable evidence Odin returns a well-formed result: empty `paths`, `insight_score` of 0.0 and `summary["low_support"]` true. Common causes are seeds that do not exist (use full `collection/key` IDs), seeds with no neighbors within `hop_limit`, or a community scope that excludes the relevant subgraph. Check `engine.get_neighbors(seed)["degree"]`. An empty result is not an exception; failures raise (see `errors.md`).

## `trace`

`result["trace"]["params"]` is a dataclass, so convert it with `dataclasses.asdict` (or drop `trace`) before `json.dumps`.

```python
result["trace"]["timings_ms"]["total"]   # end-to-end milliseconds
```

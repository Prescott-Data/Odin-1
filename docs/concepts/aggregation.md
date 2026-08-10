---
icon: material/chart-donut
---

# Motifs & Aggregation

After [beam search](beam-search.md) and [NPLL scoring](npll.md) produce a set of surviving paths, Odin **aggregates** them into higher-level structure: recurring motifs, relation shares, and the summary statistics that feed the [triage score](scoring.md).

---

## Motifs

A **motif** is a recurring pattern across the returned paths — for example, the shape `Claim → billed_by → Provider → flagged_in → Audit` appearing many times. Motifs turn a list of individual paths into *patterns*, which is usually what an agent actually cares about.

```python
motifs = result["aggregates"]["motifs"]
for m in motifs:
    print(m)   # e.g. {"pattern": "A→B→C", "count": 12}
```

Frequent motifs act as lightweight anomaly and pattern detection: if the same suspicious chain shows up 47 times, that is signal.

---

## Relation shares

`relation_share` reports how the returned edges break down by relation type — a quick fingerprint of *what kind* of connections dominate this retrieval. The most prominent one is surfaced as the `dominant_relation`.

---

## The summary block

Aggregation computes a `summary` of quality signals over the returned paths. These are the inputs to triage:

| Field | Meaning |
|-------|---------|
| `total_paths` | Number of paths returned |
| `unique_motifs` | Distinct motif shapes |
| `unique_relations` | Distinct relation types |
| `total_edges` | Edges across all paths |
| `provenance` | Fraction of edges with a source/provenance reference |
| `recency` | Recency-weighted freshness of the edges |
| `label_coverage` | Fraction of nodes/edges that are properly labeled |
| `motif_density` | How concentrated the results are into repeated motifs |
| `has_baseline` | Whether a baseline comparison was available |
| `low_support` | True when there is too little evidence to be confident |

`provenance` and `recency` reward well-sourced, fresh evidence; `label_coverage` guards against unlabeled noise; `low_support` flags thin results.

---

## Baselines and surprise

When a baseline set of paths is available, aggregation compares the current retrieval against it to compute **surprise** — how much the dominant pattern deviates from the expected/prior distribution. Surprise is a major component of the triage score: a result that is *unexpected* is often the most worth an agent's attention.

---

## The full aggregates object

```python
result["aggregates"] = {
    "motifs":          [ {"pattern": "...", "count": N}, ... ],
    "relation_share":  { "billed_by": 0.42, "has_diagnosis": 0.31, ... },
    "snippet_anchors": [ ... ],
    "summary":         { "total_paths": ..., "provenance": ..., ... },
}
```

See the [Result Schema](../reference/result-schema.md) for the complete return shape.

---

## Next

- [Triage & Insight Scoring](scoring.md) — how these signals collapse into one number

---
icon: material/chart-donut
---

# Motifs & Aggregation

Beam search hands back a list of surviving paths, but a list of paths is rarely what you actually want; patterns are. Aggregation is the final stage of retrieval: it folds the paths into recurring **motifs**, a breakdown of **relation shares**, and a **summary** of quality signals that the [triage score](scoring.md) is computed from.

## Motifs turn paths into patterns

A motif is a shape that recurs across the returned paths, for instance `Claim → billed_by → Provider → flagged_in → Audit` showing up again and again. That repetition is usually the real finding, so Odin counts it for you:

```python
for m in result["aggregates"]["motifs"]:
    print(m)   # e.g. {"pattern": "A→B→C", "count": 12}
```

This doubles as lightweight anomaly detection. One suspicious chain is a coincidence; the same chain appearing 47 times is signal. Alongside the motifs, `relation_share` reports how the edges break down by relation type, a quick fingerprint of *what kind* of connections dominate a retrieval, and the strongest one is surfaced separately as `dominant_relation`.

## The summary is the raw material for scoring

Aggregation also computes a `summary` of quality signals over the paths. These are not just diagnostics; they are the exact inputs the triage score consumes:

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

Read together they tell a story about *trustworthiness*: `provenance` and `recency` reward well-sourced, fresh evidence, `label_coverage` guards against unlabeled noise, and `low_support` flags results too thin to stand behind.

## Surprise comes from a baseline

When a baseline set of paths is available, aggregation compares the current retrieval against it to compute **surprise**: how far the dominant pattern deviates from what the prior distribution led you to expect. Surprise carries real weight in the triage score, because an *unexpected* result is often the one most worth an agent's attention.

The whole thing arrives under a single key:

```python
result["aggregates"] = {
    "motifs":          [ {"pattern": "...", "count": N}, ... ],
    "relation_share":  { "billed_by": 0.42, "has_diagnosis": 0.31, ... },
    "snippet_anchors": [ ... ],
    "summary":         { "total_paths": ..., "provenance": ..., ... },
}
```

---

These summary signals do not stay separate for long. [Triage & Insight Scoring](scoring.md) is where they collapse into one number, and the [Result Schema](../reference/result-schema.md) documents every field in full.

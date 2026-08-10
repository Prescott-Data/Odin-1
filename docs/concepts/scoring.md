---
icon: material/numeric
---

# Triage & Insight Scoring

A single retrieval can come back with dozens of paths and a handful of motifs. An agent cannot act on all of that; it needs one number that says *how much attention this deserves*. That number is the **triage score**, an integer from 0 to 100, and it is the value most agent loops gate on.

## How the score is built

The score is a weighted blend of five components drawn from the [aggregation summary](aggregation.md), each on a `0-1` scale:

```text
score = 25·provenance + 25·recency + 25·surprise + 15·motif_density + 10·controllability
```

| Component | Weight | Rewards |
|-----------|--------|---------|
| `provenance` | 25 | Well-sourced edges with references |
| `recency` | 25 | Fresh, recently-updated evidence |
| `surprise` | 25 | Deviation from the baseline / prior |
| `motif_density` | 15 | Concentration into repeated patterns |
| `controllability` | 10 | How actionable the finding is |

The weighting is deliberate: provenance, recency, and surprise dominate because a finding that is well-sourced, current, and *unexpected* is the kind worth waking an analyst for. With everything maxed the score reaches 100.

## Two guards keep it honest

Raw weighting alone would let flimsy results look impressive, so two guards pull them back down. If `label_coverage` falls below `0.8`, `motif_density` is capped at `0.3` and a flat **15 points** are subtracted, because poorly-labeled data cannot earn full pattern credit. And if `low_support` is set because there simply is not enough evidence, the whole score is cut by **40%**. Together they ensure a confident-looking number is actually backed by confident-looking data.

## A score is never a black box

Every triage result carries the breakdown that produced it, so you can always see *why* a number came out the way it did:

```python
triage = result["triage"]
triage["score"]        # e.g. 87
triage["components"]   # each clamped input, the penalty applied, and the flags
triage["dominant_relation"]
```

```python
{
  "provenance": 0.82,
  "recency": 0.74,
  "surprise": 0.61,
  "motif_density": 0.30,     # capped by the label-coverage guard
  "controllability": 1.0,
  "label_coverage": 0.71,
  "penalty": 15.0,           # the guard fired here
  "low_support": false
}
```

When a score comes back lower than you expected, this is the first thing to read: a `penalty` of 15 or a low `provenance` usually explains it immediately.

## Triage vs. the other signals

Triage is the headline, but Odin reports finer-grained signals alongside it. The `insight_score` is an overall `0-1` quality measure, and `ics` decomposes it into the `evidence_strength` and `community_relevance` that make it up:

| Field | Range | Use |
|-------|-------|-----|
| `triage["score"]` | 0-100 | Prioritization: "should I look at this?" |
| `insight_score` | 0.0-1.0 | Overall retrieval quality |
| `evidence_strength` | 0.0-1.0 | Strength of the supporting evidence |
| `community_relevance` | 0.0-1.0 | Relevance to the community scope |
| `ics` | object | Decomposition of `insight_score` |

The rule of thumb is simple: **gate on the triage score**, and reach for the others when you need observability rather than a decision.

```python
result = engine.retrieve(seeds=[...])
if result["triage"]["score"] >= 70:
    agent.investigate(result["paths"])
else:
    agent.skip(reason="low triage")
```

---

That gate is the seam between Odin and the agent: [AI Agent Integration](../guides/agent-integration.md) builds it out into a full loop, and the [Result Schema](../reference/result-schema.md) documents every field these scores live in.

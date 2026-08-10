---
icon: material/numeric
---

# Triage & Insight Scoring

A retrieval can return dozens of paths and motifs. To make that actionable, Odin collapses it into a single **triage score** from 0–100 — the number an agent uses to decide *how much attention this result deserves*.

---

## The triage score

`result["triage"]["score"]` is an integer in `[0, 100]`, computed from five weighted components:

$$
\text{score} = 25\,p + 25\,r + 25\,s + 15\,m + 10\,c
$$

| Symbol | Component | Weight | Rewards |
|--------|-----------|--------|---------|
| $p$ | `provenance` | 25 | Well-sourced edges with references |
| $r$ | `recency` | 25 | Fresh, recently-updated evidence |
| $s$ | `surprise` | 25 | Deviation from the baseline / prior |
| $m$ | `motif_density` | 15 | Concentration into repeated patterns |
| $c$ | `controllability` | 10 | How actionable the finding is |

All components are in `[0, 1]`, so the maximum possible score is 100.

---

## Guards

Two guards keep the score honest:

- **Low label coverage** — if `label_coverage < 0.8`, `motif_density` is capped at `0.3` and **15 points** are subtracted. Poorly-labeled data cannot earn full pattern credit.
- **Low support** — if `low_support` is set (too little evidence), the score is reduced by **40%**. Thin results are not allowed to look confident.

```python
triage = result["triage"]
print(triage["score"])        # e.g. 87
print(triage["components"])   # per-component breakdown, penalties, flags
print(triage["dominant_relation"])
```

---

## Reading the components

The `components` dict returns each clamped input, the applied `penalty`, and the `low_support` flag — so a score is never a black box. If a score is surprisingly low, inspect the components to see whether provenance, recency, or a guard pulled it down.

```python
{
  "provenance": 0.82,
  "recency": 0.74,
  "surprise": 0.61,
  "motif_density": 0.30,     # possibly capped by the label-coverage guard
  "controllability": 1.0,
  "label_coverage": 0.71,
  "penalty": 15.0,           # label-coverage guard fired
  "low_support": false
}
```

---

## Insight score vs. triage score

Odin also reports an `insight_score` (a `[0, 1]` float) alongside `evidence_strength` and `community_relevance`. The `ics` field decomposes the insight score into these contributing parts.

| Field | Range | Use |
|-------|-------|-----|
| `triage["score"]` | 0–100 | Prioritization for agents — "should I look at this?" |
| `insight_score` | 0.0–1.0 | Overall retrieval quality signal |
| `evidence_strength` | 0.0–1.0 | Strength of the supporting evidence |
| `community_relevance` | 0.0–1.0 | How relevant the result is to the community scope |
| `ics` | object | Decomposition of `insight_score` |

For most agent loops, the **triage score is the number to gate on**; the others are available when you need finer-grained observability.

---

## Using it in an agent

```python
result = engine.retrieve(seeds=[...])
if result["triage"]["score"] >= 70:
    agent.investigate(result["paths"])
else:
    agent.skip(reason="low triage")
```

See [AI Agent Integration](../guides/agent-integration.md) for a complete loop.

---

## Next

- [Result Schema](../reference/result-schema.md) — the exact return shape
- [Tuning Retrieval](../guides/tuning.md) — how parameters affect the scores

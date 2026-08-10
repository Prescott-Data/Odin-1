---
icon: material/shield-check
---

# NPLL Edge Scoring

NPLL — **Neural Probabilistic Logic Learning** — is Odin's **semantic** signal. It scores how plausible a given edge is, learned directly from the structure of your graph.

---

## What it does

Given a candidate edge `(head, relation, tail)`, NPLL returns a probability between `0.0` and `1.0`:

```python
score = engine.score_edge("entity/patient_001", "treated_by", "entity/doctor_smith")
# 0.0  → impossible / contradicts learned patterns
# 1.0  → highly plausible / matches learned patterns
```

During retrieval this signal prunes semantically invalid paths — the ones naive traversal happily follows because the edge simply *exists* in the data, even when it makes no domain sense.

---

## Why "learned from your graph"

NPLL does not ship with hand-written rules about medicine, finance, or supply chains. Instead it **learns the plausibility patterns of your specific graph**: which relation types connect which kinds of entities, and in which direction. A relationship that is common and consistent in your data scores high; a rare or contradictory one scores low.

This makes the same engine work across domains without domain-specific configuration.

---

## Self-managing lifecycle

You never train NPLL by hand. On first initialization, `OdinEngine` bootstraps the model:

1. **First run** — extract edge patterns from the graph and train the model (~2–5 minutes).
2. **Persist** — save the learned weights into an ArangoDB collection.
3. **Subsequent runs** — load the weights and rebuild the model in ~30 seconds.

There is no separate ML pipeline, no `.pt` files to ship, and no DevOps overhead. See [Model Lifecycle](../guides/npll-lifecycle.md) for the details and how to force a retrain.

```python
# Force a retrain after major graph changes
engine.retrain_model()
```

---

## Graceful fallback

If a model cannot be trained or loaded (for example, an empty or brand-new graph), Odin does **not** fail. It falls back to a constant edge-confidence so retrieval keeps working, and reports the mode:

```python
engine.has_npll        # True if the NPLL model is active
engine.get_status()    # {'intelligence_mode': 'NPLL' | 'Constant', ...}
```

In constant mode you still get PPR-driven structural exploration; you just lose the semantic pruning until a model is available.

---

## Where the signal is used

| Use | How |
|-----|-----|
| Pruning during [beam search](beam-search.md) | Implausible extensions are dropped early |
| Path scoring | Contributes to each path's final score |
| Direct edge validation | `score_edge()` in your own agent logic |

---

## Next

- [Model Lifecycle](../guides/npll-lifecycle.md) — training, persistence, and retraining
- [Scoring Edges](../guides/edge-scoring.md) — using `score_edge()` in agent loops
- [Triage & Insight Scoring](scoring.md) — how edge scores roll up into path scores

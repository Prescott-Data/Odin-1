---
icon: material/shield-check
---

# NPLL Edge Scoring

[PPR](ppr.md) finds important nodes and [beam search](beam-search.md) reaches them, but neither knows whether an edge *makes sense*. That is NPLL's job. **Neural Probabilistic Logic Learning** is Odin's semantic signal: it scores how plausible any given edge is, using patterns learned directly from your graph.

Given a candidate edge `(head, relation, tail)`, it returns a probability from `0.0` to `1.0`:

```python
score = engine.score_edge("entity/patient_001", "treated_by", "entity/doctor_smith")
# 0.0  → impossible / contradicts learned patterns
# 1.0  → highly plausible / matches learned patterns
```

Naive traversal will happily follow an edge just because it *exists* in the data, even when it is nonsense for the domain. NPLL is what lets Odin down-weight and prune those paths instead.

## Learned from your graph, not from rules

NPLL ships with no hand-written knowledge about medicine, finance, or supply chains. It learns the plausibility patterns of *your* graph: which relation types connect which kinds of entities, and in which direction. A relationship that is common and consistent in your data scores high; a rare or contradictory one scores low. The upshot is that the same engine works across wildly different domains with no domain-specific configuration: point it at a claims graph and it learns claims; point it at a supply chain and it learns that instead.

## You never train it by hand

The model is self-managing. The first time you construct an `OdinEngine` against a graph, it bootstraps itself, extracting edge patterns and training the model (2-5 minutes), then persisting the learned weights into an ArangoDB collection. Every run after that just loads those weights and rebuilds in about 30 seconds. There is no separate ML pipeline, no `.pt` files to ship, and no DevOps overhead; when the graph's structure changes materially you simply ask for a retrain:

```python
engine.retrain_model()   # re-learn after major graph changes
```

The full story (persistence, per-community models, and when to retrain) is in [Model Lifecycle](../guides/npll-lifecycle.md).

## What happens when there is no model

A model cannot always train: an empty or brand-new graph has nothing to learn from. Rather than fail, Odin falls back to a constant edge-confidence so retrieval keeps working, and it tells you which mode you are in:

```python
engine.has_npll        # True when the NPLL model is active
engine.get_status()    # {'intelligence_mode': 'NPLL' | 'Constant', ...}
```

In constant mode you keep PPR-driven structural exploration; you only lose the semantic pruning until a model becomes available. It is always worth checking the mode before you lean on fine-grained plausibility.

---

The same signal shows up in three places: it prunes implausible extensions mid-[beam-search](beam-search.md), it feeds each path's final [score](scoring.md), and it is exposed directly as `score_edge()` for your own agent logic, covered in [Scoring Edges](../guides/edge-scoring.md).

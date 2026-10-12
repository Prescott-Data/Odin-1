---
icon: material/shield-check
---

# NPLL Edge Scoring

[PPR](ppr.md) finds important nodes and [beam search](beam-search.md) reaches them, but neither knows whether an edge *makes sense*. That is NPLL's job. **Neural Probabilistic Logic Learning** is Odin's semantic signal: it scores how plausible any given edge is, using patterns learned directly from your graph.

Given a candidate edge `(head, relation, tail)`, it returns a learned plausibility score from `0.0` to `1.0`. It is not a calibrated probability that the edge is true:

```python
score = engine.score_edge("CaseRecords/patient_001", "treated_by", "CaseRecords/doctor_smith")
# Lower  → matches learned patterns less well
# Higher → matches learned patterns better
```

Naive traversal will happily follow an edge just because it *exists* in the data, even when it is nonsense for the domain. NPLL is what lets Odin down-weight and prune those paths instead.

## Learned from your graph, not from rules

NPLL ships with no hand-written knowledge about medicine, finance, or supply chains. It learns the plausibility patterns of *your* graph: which relation types connect which kinds of entities, and in which direction. A relationship that is common and consistent in your data tends to score higher, and a rare or contradictory one lower, but that is learned behavior, not a guarantee. The upshot is that the same engine works across wildly different domains with no domain-specific configuration: point it at a claims graph and it learns claims; point it at a supply chain and it learns that instead.

## You never train it by hand

The model is self-managing. The first time you construct an `OdinEngine` against a graph, it bootstraps itself, extracting edge patterns and training the model (cost depends on graph size and hardware), then persisting the learned weights into an ArangoDB collection. Subsequent startups reuse compatible artifacts when the training snapshot is unchanged, loading the weights and rebuilding the architecture without training. A changed snapshot or changed model semantics triggers automatic retraining. There is no separate ML pipeline, no `.pt` files to ship, and no DevOps overhead; to force a retrain yourself, for example after a major graph change:

```python
engine.retrain_model()   # re-learn after major graph changes
```

The full story (persistence, global training with scoped retrieval, and when to retrain) is in [Model Lifecycle](../guides/npll-lifecycle.md).

## What happens when there is no model

A model cannot always train: an empty graph has nothing to learn from. Requested
training that fails raises an error. To use structural retrieval without a
model, explicitly construct the engine with `auto_train=False`:

```python
engine.has_npll        # True when the NPLL model is active
engine.get_status()    # {'intelligence_mode': 'NPLL' | 'Constant', ...}
```

In constant mode you keep PPR-driven structural exploration; you only lose the semantic pruning until a model becomes available. It is always worth checking the mode before you lean on fine-grained plausibility.

---

The same signal shows up in three places: it prunes implausible extensions mid-[beam-search](beam-search.md), it feeds each path's final [score](scoring.md), and it is exposed directly as `score_edge()` for your own agent logic, covered in [Scoring Edges](../guides/edge-scoring.md).

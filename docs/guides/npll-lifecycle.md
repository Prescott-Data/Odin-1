---
icon: material/refresh
---

# Model Lifecycle

Odin's [NPLL](../concepts/npll.md) model is **self-managing**: it trains itself from your graph, persists to the database, and reloads automatically. This guide explains the lifecycle and how to control it.

---

## The three phases

```
first run ──▶ train (2–5 min) ──▶ persist weights ──▶ ArangoDB
                                                          │
later runs ◀── rebuild (~30 s) ◀── load weights ◀────────┘
```

1. **First run** — when you construct an `OdinEngine` and no model exists, `KnowledgeBootstrapper` extracts edge patterns from the graph and trains the NPLL model (typically 2–5 minutes).
2. **Persist** — the learned weights are stored in an ArangoDB collection. There are no `.pt` files to manage.
3. **Subsequent runs** — the weights are loaded from the database and the model is rebuilt in about 30 seconds.

---

## Controlling training at startup

```python
# Default: train automatically if no model exists
engine = OdinEngine(db, auto_train=True)

# Disable training — use a constant-confidence fallback instead
engine = OdinEngine(db, auto_train=False)
```

With `auto_train=False`, Odin skips NPLL entirely and uses a constant edge-confidence. You still get PPR-driven structural exploration, but not semantic pruning.

---

## Checking the mode

```python
engine.has_npll          # True if the NPLL model is active
engine.get_status()
# {'community_id': 'global', 'npll_loaded': True,
#  'intelligence_mode': 'NPLL', 'cache_size': 5000}
```

If `intelligence_mode` is `Constant`, either training was disabled, or it could not complete (for example, an empty graph).

---

## Forcing a retrain

After **significant** graph changes — new relation types, a large data load, or a structural shift — retrain so the model reflects current patterns:

```python
ok = engine.retrain_model()   # returns True on success
```

`retrain_model()` retrains from scratch, persists the new weights, and rebuilds the engine's confidence and orchestrator to use them.

!!! tip "When to retrain"
    You do **not** need to retrain for ordinary incremental writes. Retrain when the *shape* of the graph changes — new kinds of entities or relationships — not merely when a few documents are added.

---

## Per-community models

Because NPLL learns the patterns of the community it trains on, a model is tied to its `community_id`. If you serve multiple communities, each gets its own model, trained the first time you initialize an engine for that community.

```python
claims = OdinEngine(db, community_id="claims", community_mode="mapping")
supply = OdinEngine(db, community_id="supply", community_mode="mapping")
# Each trains/loads its own NPLL model on first use.
```

---

## Failure handling

NPLL initialization is defensive: if training or loading raises, Odin logs the error and falls back to constant confidence rather than failing the whole engine. Inspect logs under the `odin` logger to see what happened, and confirm the mode with `get_status()`.

---

## Next

- [NPLL Edge Scoring](../concepts/npll.md)
- [Production Deployment](production.md)

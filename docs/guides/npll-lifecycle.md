---
icon: material/refresh
---

# Model Lifecycle

The [NPLL](../concepts/npll.md) model is self-managing: it trains itself from your graph, persists to the database, and reloads on its own. You rarely have to think about it, but when you do, this is how it works and how to steer it.

## The three phases

```
first run ──▶ train (2-5 min) ──▶ persist weights ──▶ ArangoDB
                                                          │
later runs ◀── rebuild (~30 s) ◀── load weights ◀────────┘
```

The **first run** does the heavy lifting. When you construct an `OdinEngine` and no model exists yet, `KnowledgeBootstrapper` extracts the graph's edge patterns and trains the model, usually in two to five minutes. Those learned weights are then **persisted** into an ArangoDB collection, so there are no `.pt` files to ship or mount. On every **subsequent run**, Odin loads the weights and rebuilds the model in about thirty seconds.

## Controlling training at startup

By default the engine trains automatically when it finds no model. You can turn that off:

```python
engine = OdinEngine(db, auto_train=True)    # default: train if none exists
engine = OdinEngine(db, auto_train=False)   # skip NPLL, use constant confidence
```

With `auto_train=False` Odin skips NPLL entirely and falls back to a constant edge-confidence. You keep PPR-driven structural exploration, but you lose semantic pruning until a model is available.

Whichever path you take, you can always confirm which mode you ended up in:

```python
engine.has_npll          # True if the NPLL model is active
engine.get_status()
# {'community_id': 'global', 'npll_loaded': True,
#  'intelligence_mode': 'NPLL', 'npll_source': 'trained',
#  'npll_converged': True, 'npll_trained_at': '2026-09-08T12:00:00Z',
#  'cache_size': 5000}
```

An `intelligence_mode` of `Constant` means training was either disabled or could not complete, which most often happens on an empty graph.

## Trusting the model: convergence diagnostics

A model that finished training is not automatically a model you should trust. NPLL trains with an E-M loop, and if that loop hits its iteration budget before the ELBO and rule weights stabilize, the resulting edge confidences can look plausible while being poorly calibrated.

Every training run therefore produces a **training report** that is persisted next to the weights and survives cached-weight reloads:

```python
report = engine.training_report
if report and not report.converged:
    print(f"Not converged after {report.total_em_iterations} E-M iterations")
    print(f"ELBO trajectory: {report.elbo_history}")
    print(f"Weight deltas:   {report.rule_weight_delta_history}")
```

The report carries the complete evidence for the convergence decision:

| Field | Meaning |
|-------|---------|
| `converged` | Whether the E-M loop met the convergence criteria |
| `convergence_epoch` | Epoch at which convergence was declared (or `None`) |
| `elbo_history` | ELBO after every E-M iteration, unabridged |
| `rule_weight_delta_history` | Max-abs rule-weight change per iteration |
| `final_elbo` / `best_elbo` | Last and best ELBO seen |
| `convergence_criteria` | The `elbo_rel_tol`, `weight_abs_tol`, and patience used |
| `trained_at` | UTC timestamp of the run |

Convergence is declared when the relative ELBO change **and** the rule-weight change stay under their tolerances for `convergence_patience` consecutive iterations.

If the active model did not converge, the engine logs a warning at initialization — including when the weights were loaded from cache, since the cache remembers how its training run went. The usual fix is `engine.retrain_model()`, or a look at whether the graph has enough facts to learn from.

`engine.training_report` is `None` when auto-train is disabled, training failed, or the weights were saved by a version that predates reports.

## Retraining

When the graph changes materially (new relation types, a large data load, a structural shift), retrain so the model reflects current patterns:

```python
ok = engine.retrain_model()   # returns True on success
```

`retrain_model()` trains from scratch, persists the new weights, and rebuilds the engine's confidence and orchestrator to use them.

!!! tip "When to retrain"
    Ordinary incremental writes do not need a retrain. Retrain when the *shape* of the graph changes, meaning new kinds of entities or relationships, not merely a few new documents.

## Per-community models

Because NPLL learns the patterns of the community it trains on, each model is tied to its `community_id`. If you serve several communities, each one gets its own model, trained the first time you initialize an engine for it:

```python
claims = OdinEngine(db, community_id="claims", community_mode="mapping")
supply = OdinEngine(db, community_id="supply", community_mode="mapping")
# Each trains or loads its own NPLL model on first use.
```

## When things go wrong

Initialization is deliberately defensive. If training or loading raises, Odin logs the error and falls back to constant confidence rather than taking the whole engine down. When a model does not come up as expected, check the logs under the `odin` logger and confirm the mode with `get_status()`.

## Next

The signal this model produces is explained in [NPLL Edge Scoring](../concepts/npll.md), and warming a model before serving traffic is covered in [Production Deployment](production.md).

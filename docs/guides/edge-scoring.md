---
icon: material/shield-check
---

# Scoring Edges

Beyond whole-path retrieval, Odin can score a **single** relationship. This is the [NPLL](../concepts/npll.md) plausibility signal exposed directly, for use inside your own agent logic.

---

## The method

```python
score = engine.score_edge(src, rel, dst)
```

| Argument | Meaning |
|----------|---------|
| `src` | Source entity ID |
| `rel` | Relation type |
| `dst` | Destination entity ID |

Returns a `float` in `[0.0, 1.0]`:

- `0.0` — impossible / contradicts learned patterns
- `1.0` — highly plausible / matches learned patterns

```python
engine.score_edge("entity/patient_001", "treated_by", "entity/doctor_smith")
# 0.91
engine.score_edge("entity/patient_001", "diagnosed_by", "entity/aspirin")
# 0.03   ← implausible: a patient is not diagnosed_by a medication
```

---

## Using it in a decision loop

The typical pattern is a plausibility gate before acting on a candidate relationship:

```python
def maybe_investigate(agent, src, rel, dst):
    score = engine.score_edge(src, rel, dst)
    if score > 0.7:
        agent.investigate_further(src, rel, dst)
    elif score < 0.2:
        agent.discard(src, rel, dst, reason="implausible edge")
    else:
        agent.request_human_review(src, rel, dst, confidence=score)
```

---

## Validating a hypothesis

`score_edge()` is a fast way to check a relationship an agent *proposes* but has not confirmed in the graph — for example, validating an LLM's guess before writing it back:

```python
proposed = ("entity/fund_A", "managed_by", "entity/sanctioned_entity")
if engine.score_edge(*proposed) > 0.5:
    compliance_agent.flag_for_review(proposed)
```

---

## Performance

Edge scores are cached (see [Caching](../concepts/caching.md)), so repeatedly scoring the same triple within a session is effectively free after the first call — roughly **~5 ms per edge** cached.

---

## When NPLL is unavailable

If the engine is running in constant-confidence mode (`engine.has_npll == False`), `score_edge()` returns the constant fallback rather than a learned score. Check the mode before relying on fine-grained plausibility:

```python
if engine.has_npll:
    score = engine.score_edge(src, rel, dst)
else:
    # No trained model yet — see the Model Lifecycle guide
    ...
```

See [Model Lifecycle](npll-lifecycle.md) to ensure a model is trained.

---

## Next

- [NPLL Edge Scoring](../concepts/npll.md) — the concept behind the score
- [AI Agent Integration](agent-integration.md) — full agent loops

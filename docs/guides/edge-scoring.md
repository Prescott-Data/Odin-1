---
icon: material/shield-check
---

# Scoring Edges

Retrieval walks whole paths, but sometimes you only need a score for one relationship. `score_edge()` gives you exactly that: the [NPLL](../concepts/npll.md) plausibility signal Odin uses internally, exposed so you can drop it straight into your own agent logic.

## Calling it

You pass a source, a relation, and a destination, and get back a learned plausibility score:

```python
score = engine.score_edge(src, rel, dst)
```

| Argument | Meaning |
|----------|---------|
| `src` | Source entity ID |
| `rel` | Relation type |
| `dst` | Destination entity ID |

The result is a `float` between `0.0` and `1.0`. Larger values mean the triple better matches patterns learned from the configured graph. It is not a calibrated probability that a statement is true: a low score does not disprove an existing fact, and a high score does not verify a proposed fact. The difference is easiest to see with two edges from the same entity:

```python
engine.score_edge("entity/patient_001", "treated_by", "entity/doctor_smith")
# 0.91   plausible: patients are treated by doctors
engine.score_edge("entity/patient_001", "diagnosed_by", "entity/aspirin")
# 0.03   implausible: a patient is not diagnosed_by a medication
```

## Gating a decision

The typical use is a plausibility gate in front of an action. Use scores to prioritize investigation alongside source evidence. Choose thresholds on labeled validation data from your domain; the thresholds below are illustrative:

```python
def maybe_investigate(agent, src, rel, dst):
    score = engine.score_edge(src, rel, dst)
    if score > 0.7:
        agent.investigate_further(src, rel, dst)
    elif score < 0.2:
        agent.request_human_review(src, rel, dst, reason="low learned plausibility")
    else:
        agent.request_human_review(src, rel, dst, confidence=score)
```

The same call is also the a way to assess the plausibility of a relationship an agent *proposes* but has not confirmed in the graph, such as checking an LLM's guess before you write it back:

```python
proposed = ("entity/fund_A", "managed_by", "entity/sanctioned_entity")
if engine.score_edge(*proposed) > 0.5:
    compliance_agent.flag_for_review(proposed)
```

## Checking a trained model

Compare scores for observed edges with seeded, unobserved head and tail
corruptions. Keep the exact triples and scores so unusually low observed scores
and high contrast scores can be inspected. An unobserved triple is not necessarily
false, and observed training facts do not establish generalization. Use genuinely
held-out labeled examples for accuracy and calibration evaluation.

Retrieval exposes the same value as `edge["confidence"]` in every path's
`edges` list. The path's `score` combines this with other retrieval signals;
it is not the edge confidence. After loading a saved artifact, verify that
`engine.score_edge(src, rel, dst)` reproduces the saved model's scores.

## Cost and availability

Edge scores are cached (see [Caching](../concepts/caching.md)), so scoring the same triple repeatedly within a session avoids repeated model evaluation after the first lookup; measure latency on your own data and hardware. One thing to check before you lean on fine-grained plausibility: if the engine is in constant-confidence mode, `score_edge()` returns the constant fallback rather than a learned score.

```python
if engine.has_npll:
    score = engine.score_edge(src, rel, dst)
else:
    ...   # no trained model yet; see the Model Lifecycle guide
```

If you find yourself in that mode, [Model Lifecycle](npll-lifecycle.md) explains how to get a model trained.

## Next

The concept behind the score is [NPLL Edge Scoring](../concepts/npll.md), and [AI Agent Integration](agent-integration.md) shows `score_edge()` inside complete agent loops.

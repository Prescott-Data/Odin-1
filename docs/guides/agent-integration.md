---
icon: material/robot
---

# AI Agent Integration

Odin is built to sit inside an AI agent's reasoning loop. It supplies **structured, scored evidence**; the agent supplies the language reasoning. This guide shows the integration patterns.

---

## The division of labor

| Odin (the compass) | The agent (the explorer) |
|--------------------|--------------------------|
| Finds important nodes ([PPR](../concepts/ppr.md)) | Decides what to investigate |
| Explores paths ([beam search](../concepts/beam-search.md)) | Interprets what paths mean |
| Scores plausibility ([NPLL](../concepts/npll.md)) | Writes conclusions in natural language |
| Prioritizes ([triage](../concepts/scoring.md)) | Takes actions / escalates |

Odin never hallucinates a relationship — every path it returns exists in your graph.

---

## Pattern 1 — Retrieve, then reason

The core loop: retrieve scored evidence, gate on the triage score, and hand the surviving paths to the LLM.

```python
def investigate(engine, agent, seeds):
    result = engine.retrieve(seeds=seeds, max_paths=50, hop_limit=3)

    if result["triage"]["score"] < 60:
        return agent.note("Low-signal region; skipping.")

    evidence = format_paths(result["paths"][:10])
    motifs = result["aggregates"]["motifs"][:5]
    return agent.reason(
        prompt="Analyze these graph findings and explain the key risk.",
        evidence=evidence,
        motifs=motifs,
    )
```

---

## Pattern 2 — Validate the agent's hypotheses

When an LLM proposes a relationship, check it against the graph before acting with [`score_edge()`](edge-scoring.md):

```python
hypothesis = agent.propose_relationship()   # (src, rel, dst)
if engine.score_edge(*hypothesis) > 0.7:
    agent.act_on(hypothesis)
else:
    agent.reconsider(hypothesis, reason="not supported by the graph")
```

This closes the loop between an agent's free-form reasoning and verifiable graph structure.

---

## Pattern 3 — Give the agent graph context

Use [schema introspection](schema-introspection.md) to prime the agent so it can request the right seeds or write valid queries:

```python
from odin import inspect_arango_schema

inspect_arango_schema(db, output_file="schema.json")
agent.load_context("schema.json")   # now the agent knows the collections/fields
```

---

## Pattern 4 — Human-in-the-loop on the margin

The triage score gives you a clean three-way gate — act, skip, or escalate:

```python
score = result["triage"]["score"]
if score >= 75:
    agent.act(result)
elif score <= 40:
    agent.skip(result)
else:
    human_review.enqueue(result)   # uncertain — ask a person
```

---

## Formatting evidence for an LLM

Keep the payload compact and readable — paths as arrows, plus the motifs and the score:

```python
def format_paths(paths):
    lines = []
    for p in paths:
        chain = " -> ".join(str(n) for n in p["nodes"])
        lines.append(f"[{p['score']:.2f}] {chain}")
    return "\n".join(lines)
```

You can also let the agent pull the raw context itself: Odin returns everything as plain Python data, so it serializes cleanly to JSON for tool calls.

---

## Reference architecture

```
        seeds
          │
          ▼
   ┌──────────────┐   scored paths + triage    ┌──────────────┐
   │    Odin      │ ─────────────────────────▶ │    Agent     │
   │ (evidence)   │ ◀───────────────────────── │  (reasoning) │
   └──────────────┘   score_edge(hypothesis)   └──────────────┘
```

For the full technical write-up, see the [Agent Integration Guide](https://github.com/Prescott-Data/Odin-1/blob/main/whitepaper/AGENT_INTEGRATION_GUIDE.md) in the repository.

---

## Next

- [Scoring Edges](edge-scoring.md)
- [Examples](../examples/index.md)

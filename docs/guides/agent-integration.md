---
icon: material/robot
---

# AI Agent Integration

Odin was built to sit inside an AI agent's reasoning loop. The division is clean: Odin supplies structured, scored evidence, and the agent supplies the language reasoning on top of it. Keeping that boundary sharp is what makes the whole thing reliable — because Odin only ever ranks relationships that already exist in your graph, there is no hallucinated evidence for the agent to reason from.

| Odin (the compass) | The agent (the explorer) |
|--------------------|--------------------------|
| Finds important nodes ([PPR](../concepts/ppr.md)) | Decides what to investigate |
| Explores paths ([beam search](../concepts/beam-search.md)) | Interprets what paths mean |
| Scores plausibility ([NPLL](../concepts/npll.md)) | Writes conclusions in natural language |
| Prioritizes ([triage](../concepts/scoring.md)) | Takes actions / escalates |

In practice this comes together as a few recurring patterns. Most integrations use two or three of them at once.

## Retrieve, then reason

The core loop is the one you will reach for most: retrieve scored evidence, gate on the triage score so the agent never burns tokens on a low-signal region, and hand the survivors to the LLM.

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

## Validate the agent's hypotheses

Reasoning runs the other way too. When the LLM *proposes* a relationship, check it against the graph before acting on it with [`score_edge()`](edge-scoring.md) — this closes the loop between free-form generation and verifiable structure:

```python
hypothesis = agent.propose_relationship()   # (src, rel, dst)
if engine.score_edge(*hypothesis) > 0.7:
    agent.act_on(hypothesis)
else:
    agent.reconsider(hypothesis, reason="not supported by the graph")
```

## Give the agent graph context

Both of those work better when the agent knows the shape of the graph in the first place. [Schema introspection](schema-introspection.md) primes it so it can request sensible seeds or write valid queries:

```python
from odin import inspect_arango_schema

inspect_arango_schema(db, output_file="schema.json")
agent.load_context("schema.json")   # now the agent knows the collections/fields
```

## Escalate on the margin

Finally, the triage score doubles as a clean three-way gate — a natural place to bring a human in only when it is actually warranted:

```python
score = result["triage"]["score"]
if score >= 75:
    agent.act(result)          # strong signal — proceed
elif score <= 40:
    agent.skip(result)         # weak signal — drop it
else:
    human_review.enqueue(result)   # uncertain — ask a person
```

## Handing evidence to the LLM

Across all of these, keep the payload you give the model compact and readable — paths as arrows, plus the motifs and the score:

```python
def format_paths(paths):
    lines = []
    for p in paths:
        chain = " -> ".join(str(n) for n in p["nodes"])
        lines.append(f"[{p['score']:.2f}] {chain}")
    return "\n".join(lines)
```

You can also let the agent pull context itself: Odin returns everything as plain Python data, so it serializes to JSON cleanly for tool calls. Put together, the patterns form a simple two-way contract — Odin sends scored evidence downstream, the agent sends hypotheses back for validation:

```
        seeds
          │
          ▼
   ┌──────────────┐   scored paths + triage    ┌──────────────┐
   │    Odin      │ ─────────────────────────▶ │    Agent     │
   │ (evidence)   │ ◀───────────────────────── │  (reasoning) │
   └──────────────┘   score_edge(hypothesis)   └──────────────┘
```

---

For the full technical write-up, see the [Agent Integration Guide](https://github.com/Prescott-Data/Odin-1/blob/main/whitepaper/AGENT_INTEGRATION_GUIDE.md) in the repository, or see the patterns applied end-to-end in the [Examples](../examples/index.md).

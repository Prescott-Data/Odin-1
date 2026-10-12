---
icon: material/robot
---

# AI Agent Integration

Odin was built to sit inside an AI agent's reasoning loop. The division is clean: Odin supplies structured, scored evidence, and the agent supplies the language reasoning on top of it. Keeping that boundary sharp is what makes the whole thing reliable: Odin only ever ranks relationships that exist in your graph, so every edge it returns can be traced to a stored record. Its scores are learned plausibility, not proof, so the agent still has to treat them as ranking signals and cite the sources.

| Odin (the compass) | The agent (the explorer) |
|--------------------|--------------------------|
| Finds important nodes ([PPR](../concepts/ppr.md)) | Decides what to investigate |
| Explores paths ([beam search](../concepts/beam-search.md)) | Interprets what paths mean |
| Scores plausibility ([NPLL](../concepts/npll.md)) | Writes conclusions in natural language |
| Prioritizes ([triage](../concepts/scoring.md)) | Takes actions / escalates |

In practice this comes together as a few recurring patterns. Most integrations use two or three of them at once.

## Retrieve, then reason

The core loop is the one you will reach for most: retrieve scored evidence, skip regions your application considers low-signal, and hand the rest to the LLM. The cutoff is your application's decision, so pass it in rather than hard-coding it. Odin returns an empty result when there is no reachable evidence and raises an exception on failure; handle the two differently.

```python
def investigate(engine, agent, seeds, min_triage):
    # Exceptions (backend, training, configuration) propagate: a failure is not "no evidence".
    result = engine.retrieve(seeds=seeds, max_paths=50, hop_limit=3)

    if not result["paths"]:
        return agent.note("No reachable evidence for these seeds.")
    if result["triage"]["score"] < min_triage:
        return agent.note("Below the configured triage threshold; skipping.")

    evidence = format_paths(result["paths"])
    return agent.reason(
        prompt="Analyze these graph findings and explain the key risk.",
        evidence=evidence,
        motifs=result["aggregates"]["motifs"],
    )
```

If the context window forces you to send fewer paths, cut deliberately and say so in the prompt, for example "showing 10 of 47 returned paths". Do not slice the list silently.

## Validate the agent's hypotheses

Reasoning runs the other way too. When the LLM *proposes* a relationship, compare it with what the graph has learned using [`score_edge()`](edge-scoring.md). The score says how well the proposal matches learned patterns; it does not confirm the claim is true, so use it to prioritize, and verify against source records before acting:

```python
hypothesis = agent.propose_relationship()   # (src, rel, dst)
if engine.score_edge(*hypothesis) >= min_plausibility:   # your application's cutoff
    agent.verify_against_sources(hypothesis)
else:
    agent.reconsider(hypothesis, reason="weak match with learned graph patterns")
```

## Give the agent graph context

Both of those work better when the agent knows the shape of the graph in the first place. [Schema introspection](schema-introspection.md) primes it so it can request sensible seeds or write valid queries:

```python
from odin import inspect_schema

inspect_schema(backend, output_file="schema.json")
agent.load_context("schema.json")   # now the agent knows the collections/fields
```

## Escalate on the margin

Finally, the triage score can drive a three-way routing decision, a natural place to bring a human in only when it is warranted. Both cutoffs come from your application's configuration:

```python
score = result["triage"]["score"]
if score >= act_at:
    agent.act(result)              # strong signal, proceed
elif score <= skip_at:
    agent.skip(result)             # weak signal, record that it was skipped
else:
    human_review.enqueue(result)   # uncertain, ask a person
```

## Handing evidence to the LLM

Across all of these, keep the payload readable, but do not drop what makes it evidence: every edge with its relation label and direction, its confidence, and its provenance:

```python
def format_paths(paths):
    lines = []
    for p in paths:
        hops = []
        for e in p["edges"]:
            hops.append(
                f"{e['u']} --{e['relation']}--> {e['v']}"
                f" (confidence {e['confidence']:.2f}, source {e.get('provenance')})"
            )
        lines.append(f"[{p['score']:.2f}] " + "; ".join(hops))
    return "\n".join(lines)
```

`provenance` can hold the raw stored edge, which may include large fields such as embedding vectors. Filter those before sending text to a model, and state which fields you left out.

You can also let the agent pull context itself: Odin returns everything as plain Python data, so it serializes to JSON cleanly for tool calls. Put together, the patterns form a simple two-way contract: Odin sends scored evidence downstream, the agent sends hypotheses back for comparison:

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

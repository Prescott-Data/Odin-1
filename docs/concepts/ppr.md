---
icon: material/star-four-points
---

# Personalized PageRank

Personalized PageRank (PPR) is Odin's **structural** signal. It answers a single question: *relative to my seed entities, which nodes in the graph matter most?*

---

## Why PPR

Classic PageRank ranks every node by global importance. That is the wrong question for guided exploration — you do not care about globally famous nodes, you care about nodes important **to your seeds**.

Personalized PageRank biases the random walk toward a set of seed nodes. The result is a per-seed importance distribution: nodes that are well-connected *to the things you care about* score highly, and distant or irrelevant nodes fade out.

This is what lets Odin **reduce the search space before it ever starts traversing** — beam search then only expands from high-PPR anchors instead of the whole graph.

---

## How Odin uses it

Inside the pipeline, PPR runs first and produces **anchors** — the top nodes to explore from. You can also call it directly:

```python
anchors = engine.find_anchors(seeds=["community/insurance_claims"], topn=20)
for node_id, ppr_score in anchors:
    print(f"{ppr_score:.4f}  {node_id}")
```

`find_anchors()` returns a list of `(node_id, ppr_score)` tuples sorted by importance. See [Finding Anchors](../guides/anchors.md) for practical usage.

---

## Intuition

Think of PPR as releasing a large number of walkers at your seed nodes. Each walker follows edges at random, and with some probability *teleports* back to a seed. After many steps, the fraction of time spent at each node is its PPR score.

- Nodes tightly connected to seeds accumulate high scores.
- Bottleneck nodes that many paths pass through score highly — often the interesting connectors.
- Nodes far from every seed score near zero and are skipped.

---

## Parameters that matter

PPR importance is shaped by the seeds you provide and the number of anchors you keep:

| Knob | Effect |
|------|--------|
| `seeds` | The personalization vector — what importance is measured *relative to* |
| `topn` | How many anchors to return (via `find_anchors`) |
| `community_id` / `community_mode` | Restricts the walk to a scope |

More seeds broaden the notion of "important"; fewer, more specific seeds produce sharper, more targeted anchors.

---

## Caching

PPR is computed over the cached graph accessor, so repeated retrievals against a warm cache are fast. See [Caching](caching.md).

---

## Next

- [Beam Search](beam-search.md) — how anchors become multi-hop paths
- [Finding Anchors](../guides/anchors.md) — the practical guide

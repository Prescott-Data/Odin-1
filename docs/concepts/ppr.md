---
icon: material/star-four-points
---

# Personalized PageRank

Personalized PageRank (PPR) is the first stage of every retrieval and Odin's **structural** signal. It answers one question before any traversal happens: *relative to my seed entities, which nodes in the graph actually matter?*

Getting that answer first is what makes the rest of the pipeline affordable. Instead of exploring the whole graph, [beam search](beam-search.md) only ever expands outward from the handful of nodes PPR marks as important.

## Why *personalized*

Classic PageRank ranks every node by global importance — the graph's celebrities. That is the wrong question here. You do not care that a node is famous across the whole graph; you care that it is important *to the things you are investigating*.

Personalized PageRank fixes this by biasing the random walk toward your seed set. The result is a per-query importance distribution: nodes well-connected to your seeds rise to the top, and everything distant or irrelevant fades toward zero. Change the seeds and the entire ranking shifts, because importance is always measured *relative to* what you asked about.

## How it behaves

The intuition is a crowd of walkers. Release many of them at your seed nodes; each one follows edges at random but, every so often, teleports back to a seed. After enough steps, the share of time spent at each node is its PPR score. Three things fall out of this:

- Nodes tightly connected to your seeds accumulate high scores.
- Bottleneck nodes that many paths funnel through also score highly — often the most interesting connectors in an investigation.
- Nodes far from every seed collect almost no walker time and are effectively skipped.

## Using it directly

Inside `retrieve()` these scores become the **anchors** the pipeline explores from, but you can also ask for them on their own — a fast way to orient yourself in an unfamiliar neighborhood:

```python
anchors = engine.find_anchors(seeds=["community/insurance_claims"], topn=20)
for node_id, ppr_score in anchors:
    print(f"{ppr_score:.4f}  {node_id}")
```

`find_anchors()` returns `(node_id, ppr_score)` tuples sorted by importance. The two levers that shape the result are your **seeds** — the personalization vector that decides what "important" is measured against — and **`topn`**, how many anchors you keep. Broad seeds give a wide sense of importance; a few specific seeds produce sharp, targeted anchors. If you are working within a partitioned graph, `community_id` and `community_mode` confine the walk to that scope.

Because PPR runs over the [cached graph accessor](caching.md), repeated retrievals against a warm cache stay fast even though a walk touches many nodes.

---

Once PPR has named the anchors, the pipeline hands them to [Beam Search](beam-search.md) to grow multi-hop paths. For the practical, task-oriented version of everything above, see [Finding Anchors](../guides/anchors.md).

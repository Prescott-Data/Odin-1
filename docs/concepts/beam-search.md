---
icon: material/magnify-scan
---

# Beam Search

Beam search is Odin's **exploration** strategy. It walks outward from PPR anchors to build multi-hop paths — without the exponential blow-up of naive traversal.

---

## The problem it solves

A 3-hop exploration from a single node in a densely connected graph can generate **100K+ paths**, the vast majority of which are noise. Breadth-first and depth-first search both explode; random walks never converge.

Beam search keeps exploration bounded by only ever carrying forward the **top-K partial paths** at each hop.

---

## How it works

At each hop:

1. Take the current set of partial paths (the *beam*).
2. Expand each one by its outgoing edges.
3. Score every extended path (structural + semantic signals).
4. Keep only the best `beam_width` paths.
5. Repeat until `hop_limit` is reached or paths terminate.

```
hop 0:  [seed]
hop 1:  keep top-K of  seed→{a,b,c,d,...}
hop 2:  keep top-K of  (top-K)→{...}
hop 3:  keep top-K of  (top-K)→{...}
```

Because the beam is fixed-width, work per hop is bounded — turning a combinatorial explosion into a linear-in-hops search.

---

## The two knobs

| Parameter | Default | Effect |
|-----------|---------|--------|
| `beam_width` | `64` | How many partial paths survive each hop. Higher = broader, slower, more thorough. |
| `hop_limit` | `3` | Maximum path length. Higher = deeper chains, more compute. |
| `max_paths` | `50` | How many final paths to return. |

Widening the beam trades compute for recall; deepening `hop_limit` finds longer causal chains at the cost of latency. See [Tuning Retrieval](../guides/tuning.md) for guidance.

---

## Why scoring matters mid-search

Beam search is only as good as the signal it prunes with. Odin scores each candidate extension using both structural importance and **[NPLL edge plausibility](npll.md)** — so an implausible edge (for example `Patient → diagnosed_by → Medication`) is pruned early rather than expanded into thousands of downstream nonsense paths.

This tight coupling of search and semantics is what keeps Odin's output high-signal.

---

## Budgets

Exploration runs under a budget so a single retrieval cannot run away. The budget usage is reported back in the result under `used_budget`, useful for observability and tuning.

---

## Next

- [NPLL Edge Scoring](npll.md) — the plausibility signal that prunes the beam
- [Motifs & Aggregation](aggregation.md) — what happens to the surviving paths

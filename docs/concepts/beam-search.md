---
icon: material/magnify-scan
---

# Beam Search

Once [PPR](ppr.md) has named the anchors worth exploring, beam search is what actually walks the graph. It builds multi-hop paths outward from those anchors while refusing to let the search explode — which, in a real knowledge graph, it very much wants to do.

## The problem it solves

A three-hop exploration from a single node in a densely connected graph can generate **100,000+ paths**, almost all of them noise. Breadth-first and depth-first search both drown in that combinatorial growth; a random walk never converges on anything. None of them give an agent a short, ranked list of things to look at.

Beam search sidesteps the explosion with one rule: at every hop, only the **top-K partial paths** survive. Everything else is discarded before it can spawn descendants.

## How it works

Each hop repeats the same four steps. Start from the current set of partial paths — the *beam* — and expand each one along its outgoing edges. Score every extended path using both structural importance and semantic plausibility. Keep only the best `beam_width` of them, and carry that set into the next hop. Repeat until paths reach `hop_limit` or naturally terminate.

```
hop 0:  [seed]
hop 1:  keep top-K of  seed → {a, b, c, d, …}
hop 2:  keep top-K of  (top-K) → {…}
hop 3:  keep top-K of  (top-K) → {…}
```

Because the beam is a fixed width, the work per hop is bounded no matter how dense the graph is. That turns a combinatorial explosion into a search that grows only linearly with depth.

## Why the scoring signal is the whole game

Beam search is only ever as good as the signal it prunes with — a wide beam over a bad score just keeps more junk. This is why Odin scores each candidate extension with [NPLL edge plausibility](npll.md) *during* the walk, not after it. An implausible edge such as `Patient → diagnosed_by → Medication` gets pruned at the hop it appears, instead of being expanded into thousands of downstream nonsense paths. Coupling the search to semantics this tightly is what keeps the final output high-signal.

## The knobs

Two parameters shape how the beam behaves, and a third caps the output:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `beam_width` | `64` | Partial paths kept per hop. Wider trades compute for recall. |
| `hop_limit` | `3` | Maximum path length. Deeper finds longer causal chains, at more cost. |
| `max_paths` | `50` | How many finished paths to return. |

Latency tracks roughly with `beam_width × hop_limit`, so the two interact — to go deeper without paying for it, narrow the beam. [Tuning Retrieval](../guides/tuning.md) works through the trade-offs. The whole walk also runs under a budget so a single call can never run away; how much of it was consumed comes back in the result as `used_budget`, which is handy for both observability and tuning.

---

The paths that survive the beam are still just a list. [Motifs & Aggregation](aggregation.md) is what turns them into patterns and scores, and the [NPLL](npll.md) signal that did the pruning is worth understanding in its own right.

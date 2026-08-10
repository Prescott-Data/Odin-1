---
icon: material/tune
---

# Tuning Retrieval

Odin's retrieval behavior is governed by a few parameters. This guide explains what each one does, how they interact, and how to trade recall for latency.

---

## The parameters

| Parameter | Where | Default | Trades |
|-----------|-------|---------|--------|
| `beam_width` | `retrieve()` | `64` | Recall ↔ latency |
| `hop_limit` | `retrieve()` | `3` | Depth ↔ latency |
| `max_paths` | `retrieve()` | `50` | Output size |
| `cache_size` | `OdinEngine()` | `5000` | Memory ↔ database round-trips |
| `community_mode` | `OdinEngine()` | `"none"` | Scope of exploration |

---

## Beam width

`beam_width` is how many partial paths survive each hop of [beam search](../concepts/beam-search.md).

- **Wider** (128, 256) → higher recall, finds more of the relevant paths, more compute.
- **Narrower** (16, 32) → faster, sharper, may miss relevant-but-lower-scored paths.

Start at the default `64`; widen only if you observe that relevant paths are being missed.

---

## Hop limit

`hop_limit` is the maximum path length.

| `hop_limit` | Use for |
|-------------|---------|
| 2 | Direct relationships and immediate context |
| 3 (default) | Most retrieval — good depth without explosion |
| 4–5 | Deep causal chains, e.g. multi-tier supply chains |

Each additional hop increases cost roughly with the beam width, so raise `hop_limit` deliberately.

---

## How they interact

Latency is driven primarily by `beam_width × hop_limit`. Two rules of thumb:

- To go **deeper**, consider narrowing the beam so total work stays bounded.
- To go **broader**, keep hops shallow and widen the beam.

```python
# Deep but focused
engine.retrieve(seeds, hop_limit=5, beam_width=24)

# Shallow but thorough
engine.retrieve(seeds, hop_limit=2, beam_width=128)
```

---

## Seeds matter most

The single biggest lever is **seed quality**. Specific, relevant seeds produce sharp, high-triage results; broad seeds dilute the [PPR](../concepts/ppr.md) signal. When results feel noisy, tighten the seeds — or run an [anchor-then-retrieve](anchors.md#anchor-then-retrieve-pattern) pass first.

---

## Reading the budget

Every result reports how much exploration budget it consumed and where time went:

```python
result["used_budget"]              # budget consumed
result["trace"]["timings_ms"]      # per-stage timings, including 'total'
```

Use these to see whether PPR, beam search, or scoring dominates your latency before changing parameters.

---

## A tuning workflow

1. Start at defaults (`beam_width=64`, `hop_limit=3`).
2. Check `triage["score"]` and whether expected paths appear.
3. If recall is low → widen the beam or improve seeds.
4. If latency is high → inspect `timings_ms`, then narrow the beam or reduce hops.
5. If memory is high → lower `cache_size` (accepting more DB round-trips).

---

## Next

- [Caching](../concepts/caching.md)
- [Production Deployment](production.md)

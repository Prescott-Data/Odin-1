---
icon: material/tune
---

# Tuning Retrieval

Odin's defaults are chosen to work well out of the box, and most of the time you should leave them alone. When you do need to tune, only a few parameters matter — and understanding what each one trades away is more useful than any recommended value.

| Parameter | Where | Default | Trades |
|-----------|-------|---------|--------|
| `beam_width` | `retrieve()` | `64` | Recall ↔ latency |
| `hop_limit` | `retrieve()` | `3` | Depth ↔ latency |
| `max_paths` | `retrieve()` | `50` | Output size |
| `cache_size` | `OdinEngine()` | `5000` | Memory ↔ database round-trips |
| `community_mode` | `OdinEngine()` | `"none"` | Scope of exploration |

## Width and depth

The two dials that shape a search are `beam_width` and `hop_limit`. `beam_width` controls how many partial paths survive each hop of [beam search](../concepts/beam-search.md): widen it to 128 or 256 for higher recall when you suspect relevant paths are being missed, or narrow it to 16–32 for faster, sharper results. Start at the default `64` and only move once you have a reason to.

`hop_limit` sets how far a path can reach. Two hops is enough for direct relationships and immediate context; the default of three covers most retrieval with good depth and no explosion; four or five is for genuinely deep chains, like tracing a multi-tier supply chain. Because each extra hop multiplies work by roughly the beam width, raise it deliberately rather than by default.

That multiplication is the key to using them together — latency tracks with `beam_width × hop_limit`, so the two are not independent. To go deeper without paying for it, narrow the beam; to go broader, keep hops shallow and widen it:

```python
engine.retrieve(seeds, hop_limit=5, beam_width=24)    # deep but focused
engine.retrieve(seeds, hop_limit=2, beam_width=128)   # shallow but thorough
```

## The lever most people miss

Before touching any of that, look at your **seeds** — they are the single biggest influence on quality. Specific, relevant seeds sharpen the [PPR](../concepts/ppr.md) signal and produce high-triage results; broad or generic seeds dilute it and everything downstream feels noisy. When a retrieval disappoints, tighten the seeds first, or run an [anchor-then-retrieve](anchors.md#anchor-then-retrieve-pattern) pass to discover better ones. Parameter tuning cannot rescue a bad starting point.

## Tune with evidence, not guesswork

Every result tells you where its time and budget went, so you never have to guess which stage to adjust:

```python
result["used_budget"]              # exploration budget consumed
result["trace"]["timings_ms"]      # per-stage timings, including 'total'
```

That turns tuning into a short loop. Start at the defaults and check whether `triage["score"]` is reasonable and the paths you expected showed up. If recall is low, widen the beam or improve the seeds. If latency is high, read `timings_ms` to find the culprit, then narrow the beam or cut a hop. If memory is the problem, lower `cache_size` and accept a few more database round-trips. Change one thing, re-measure, repeat.

---

The memory side of that loop is covered in [Caching](../concepts/caching.md), and [Production Deployment](production.md) puts these knobs in the context of a real serving setup.

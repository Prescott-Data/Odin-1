---
icon: material/cached
---

# Caching

Graph exploration is repetitive by nature — PPR and beam search revisit the same nodes and edges constantly. To keep that from turning into a flood of database round-trips, Odin wraps every graph read in an LRU cache, so warm data is served from memory.

## Two layers, one goal

There are actually two caches working together. The **graph accessor** cache sits over ArangoDB: when you build an engine, the raw accessor is wrapped in a `CachedGraphAccessor`, and every node fetch and neighbor lookup during a walk goes through it.

```python
engine = OdinEngine(db, cache_size=5000)   # LRU size for graph reads
```

Above that, the **NPLL confidence** layer caches `score_edge` results internally. Because a beam search evaluates the same `(head, relation, tail)` triples over and over, this makes edge scoring effectively free after the first lookup — roughly 5 ms per edge, cached.

| Layer | What it caches | Controlled by |
|-------|----------------|---------------|
| Graph accessor | Node and edge reads from the database | `cache_size` (default `5000`) |
| NPLL confidence | Edge plausibility scores | internal (`cache_size=10000`) |

## Why it matters for latency

The payoff is concrete. On a warm cache Odin sees **>80% hit rates** after a short warm-up, which is a large part of why a 50-path, 3-hop retrieval lands in the typical **300–800 ms** range. The exception is the first retrieval after startup: a cold cache is slower while the working set fills, which is why warming an engine before it serves traffic matters in [production](../guides/production.md).

## Sizing the cache

`cache_size` is the main lever, and the right value depends on your graph and host. On a large, dense graph, raise it to keep more of the working set resident. On a memory-constrained host, lower it and accept more round-trips. For many short, unrelated queries, a small cache is fine because there is little locality to exploit anyway. Plan for roughly **500 MB–2 GB** of memory depending on cache size and graph density.

One caveat on freshness: the cache holds reads for the lifetime of the engine's working set. After a **significant** data change, construct a fresh engine — and if the edge patterns themselves shifted, [retrain the NPLL model](../guides/npll-lifecycle.md) — so stale structure does not linger.

---

Cache size is one dial among a few; [Tuning Retrieval](../guides/tuning.md) covers how it interacts with beam width and hops, and [Production Deployment](../guides/production.md) puts it in the context of real workloads.

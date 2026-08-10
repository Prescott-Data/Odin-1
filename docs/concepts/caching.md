---
icon: material/cached
---

# Caching

Graph exploration touches the same nodes and edges repeatedly. Odin wraps every graph read in an **LRU cache** so retrieval stays fast on warm data.

---

## Where caching happens

When you construct an `OdinEngine`, the raw ArangoDB accessor is wrapped in a `CachedGraphAccessor`:

```python
engine = OdinEngine(db, cache_size=5000)   # LRU size for graph access
```

Every node fetch and neighbor lookup during PPR and beam search goes through this cache. The `cache_size` parameter controls how many entries are retained.

---

## Two layers of caching

| Layer | What it caches | Controlled by |
|-------|----------------|---------------|
| Graph accessor | Node and edge reads from the database | `cache_size` (default `5000`) |
| NPLL confidence | Edge plausibility scores | internal (`cache_size=10000`) |

The NPLL confidence layer caches `score_edge` results, so repeatedly evaluating the same `(head, relation, tail)` during a search is effectively free after the first call.

---

## Performance impact

On a warm cache, Odin reports **>80% hit rates** after a short warm-up, contributing to the typical **300–800 ms** end-to-end retrieval latency for 50 paths at a 3-hop limit. Cold caches (the first retrieval after startup) are slower while the working set is populated.

---

## Tuning the cache

| Situation | Recommendation |
|-----------|----------------|
| Large, dense graph | Increase `cache_size` to keep more of the working set resident |
| Memory-constrained host | Decrease `cache_size`; expect more database round-trips |
| Many short, unrelated queries | A smaller cache is fine — locality is low anyway |

Memory usage scales with cache size and graph density — plan for roughly **500 MB–2 GB** depending on both.

---

## Freshness

The cache holds graph reads for the lifetime of the engine's working set. After **significant** data changes you should generally construct a fresh engine (and, if edge patterns changed materially, [retrain the NPLL model](../guides/npll-lifecycle.md)) so stale structure does not linger.

---

## Next

- [Tuning Retrieval](../guides/tuning.md) — beam width, hops, and budgets
- [Production Deployment](../guides/production.md) — sizing for real workloads

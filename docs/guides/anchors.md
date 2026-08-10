---
icon: material/star-four-points
---

# Finding Anchors

Anchors are the structurally important nodes relative to your seeds, found with [Personalized PageRank](../concepts/ppr.md). They tell an agent *where to look* before any traversal happens.

---

## The method

```python
anchors = engine.find_anchors(seeds, topn=20)
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `seeds` | — | Seed entities for personalized PageRank |
| `topn` | `20` | How many top nodes to return |

Returns a list of `(node_id, ppr_score)` tuples, sorted by importance (highest first).

```python
anchors = engine.find_anchors(seeds=["community/insurance_claims"], topn=20)
for node_id, ppr_score in anchors[:10]:
    print(f"{ppr_score:.4f}  {node_id}")
```

---

## When to use anchors

| Use case | Why anchors help |
|----------|------------------|
| **Targeted retrieval** | Feed high-PPR anchors back in as seeds for a focused second pass |
| **Graph orientation** | Show an agent the neighborhood's key players before deciding what to explore |
| **Budget control** | Explore only from the top anchors instead of the entire frontier |

---

## Anchor-then-retrieve pattern

A common two-step flow: discover the important nodes, then explore from the strongest ones.

```python
# 1. Find the important nodes around a broad seed
anchors = engine.find_anchors(seeds=["community/insurance_claims"], topn=10)

# 2. Retrieve paths from the strongest anchors
strong = [node_id for node_id, score in anchors[:3]]
result = engine.retrieve(seeds=strong, max_paths=50, hop_limit=3)
```

This narrows a broad starting point down to the parts of the graph that actually matter before spending exploration budget.

---

## Choosing `topn`

| `topn` | Effect |
|--------|--------|
| Small (5–10) | Sharp focus on the few most important nodes |
| Medium (20) | Balanced neighborhood view (default) |
| Large (50+) | Broad map of the region; more compute downstream |

---

## Inspecting a node's neighborhood

To look at one node in detail rather than rank many, use `get_neighbors()`:

```python
info = engine.get_neighbors("entity/provider_456")
print(info["degree"])
for n in info["neighbors"][:10]:
    print(n["direction"], n["rel"], n["id"], n["weight"])
```

---

## Next

- [Personalized PageRank](../concepts/ppr.md) — the concept behind anchors
- [Your First Retrieval](first-retrieval.md)

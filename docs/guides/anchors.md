---
icon: material/star-four-points
---

# Finding Anchors

Anchors are the structurally important nodes relative to your seeds, found with [Personalized PageRank](../concepts/ppr.md). They answer *where to look* before any traversal happens, which makes them useful both as a standalone orientation tool and as a way to sharpen a retrieval.

## Calling it

Give `find_anchors()` your seeds and how many nodes you want back:

```python
anchors = engine.find_anchors(seeds, topn=20)
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `seeds` | required | Seed entities for personalized PageRank |
| `topn` | `20` | How many top nodes to return |

It returns `(node_id, ppr_score)` tuples sorted by importance, highest first:

```python
anchors = engine.find_anchors(seeds=["community/insurance_claims"], topn=20)
for node_id, ppr_score in anchors[:10]:
    print(f"{ppr_score:.4f}  {node_id}")
```

`topn` is the main dial. A small value (5 to 10) gives sharp focus on the few most important nodes; the default of 20 is a balanced neighborhood view; a large value (50 or more) maps a wider region at the cost of more work downstream.

## Anchor-then-retrieve

The most valuable pattern is a two-step flow: use anchors to discover the important nodes around a broad seed, then retrieve paths from only the strongest of them.

```python
# 1. Find the important nodes around a broad seed
anchors = engine.find_anchors(seeds=["community/insurance_claims"], topn=10)

# 2. Retrieve paths from the strongest anchors
strong = [node_id for node_id, score in anchors[:3]]
result = engine.retrieve(seeds=strong, max_paths=50, hop_limit=3)
```

This narrows a vague starting point down to the parts of the graph that actually matter before you spend any exploration budget, which is often the difference between a noisy retrieval and a sharp one. The same idea shows up in [Tuning Retrieval](tuning.md) as the single biggest lever on quality.

## Inspecting one node

When you want to study a single node rather than rank many, `get_neighbors()` returns its immediate neighborhood with relation types and directions:

```python
info = engine.get_neighbors("entity/provider_456")
print(info["degree"])
for n in info["neighbors"][:10]:
    print(n["direction"], n["rel"], n["id"], n["weight"])
```

## Next

The concept behind all of this is [Personalized PageRank](../concepts/ppr.md), and [Your First Retrieval](first-retrieval.md) shows where anchors fit in the full pipeline.

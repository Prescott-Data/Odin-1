---
icon: material/engine
---

# OdinEngine API

`OdinEngine` is the main entry point. Construct it with a connected ArangoDB database, then call its methods.

```python
from odin import OdinEngine
```

---

## Constructor

```python
OdinEngine(
    db,
    community_id: str = "global",
    cache_size: int = 5000,
    auto_train: bool = True,
    community_mode: str = "none",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | `StandardDatabase` | — | A connected `python-arango` database instance |
| `community_id` | `str` | `"global"` | Scope to explore within |
| `cache_size` | `int` | `5000` | LRU size for the graph accessor |
| `auto_train` | `bool` | `True` | Train NPLL if no model exists |
| `community_mode` | `str` | `"none"` | `"none"` = global, `"mapping"` = community-scoped |

On first construction against a graph, NPLL trains (2–5 min) unless `auto_train=False`. See [Model Lifecycle](../guides/npll-lifecycle.md).

---

## `retrieve`

```python
retrieve(
    seeds: list[str],
    max_paths: int = 50,
    hop_limit: int = 3,
    beam_width: int = 64,
) -> dict
```

Runs the full pipeline (PPR → beam search → NPLL scoring → aggregation) and returns scored paths.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seeds` | `list[str]` | — | Starting entity IDs |
| `max_paths` | `int` | `50` | Maximum paths to return |
| `hop_limit` | `int` | `3` | Maximum path length |
| `beam_width` | `int` | `64` | Paths kept per hop |

**Returns:** a `dict` — see the [Result Schema](result-schema.md).

```python
result = engine.retrieve(seeds=["entity/claim_123"], max_paths=50)
result["triage"]["score"]     # 0–100
result["paths"]               # ranked paths
```

---

## `score_edge`

```python
score_edge(src: str, rel: str, dst: str) -> float
```

Returns the NPLL plausibility of a single edge, from `0.0` (impossible) to `1.0` (highly plausible).

```python
engine.score_edge("entity/patient_001", "treated_by", "entity/doctor_smith")
# 0.91
```

See [Scoring Edges](../guides/edge-scoring.md).

---

## `find_anchors`

```python
find_anchors(seeds: list[str], topn: int = 20) -> list[tuple[str, float]]
```

Returns the top-N nodes by Personalized PageRank relative to `seeds`, as `(node_id, ppr_score)` tuples sorted by importance.

```python
for node_id, ppr in engine.find_anchors(["community/claims"], topn=20):
    print(ppr, node_id)
```

See [Finding Anchors](../guides/anchors.md).

---

## `get_neighbors`

```python
get_neighbors(node_id: str) -> dict
```

Returns a node and its neighbors with relation types and directions.

```python
info = engine.get_neighbors("entity/provider_456")
info["degree"]        # total neighbor count
info["neighbors"]     # list of {"id", "rel", "weight", "direction"}
```

Each neighbor's `direction` is `"out"` or `"in"`.

---

## `retrain_model`

```python
retrain_model() -> bool
```

Forces a full NPLL retrain, persists the new weights, and rebuilds the engine's scoring. Returns `True` on success. Use after **structural** graph changes — see [Model Lifecycle](../guides/npll-lifecycle.md).

---

## `has_npll`

```python
has_npll -> bool     # property
```

`True` when a trained NPLL model is active; `False` in constant-confidence fallback.

---

## `get_status`

```python
get_status() -> dict
```

Returns a small status dictionary:

```python
{
    "community_id": "global",
    "npll_loaded": True,
    "intelligence_mode": "NPLL",   # or "Constant"
    "cache_size": 5000,
}
```

---

## Method summary

| Method | Returns | Purpose |
|--------|---------|---------|
| `retrieve(...)` | `dict` | Full pipeline: ranked, scored paths |
| `score_edge(src, rel, dst)` | `float` | Plausibility of one edge |
| `find_anchors(seeds, topn)` | `list[tuple]` | Top PPR nodes |
| `get_neighbors(node_id)` | `dict` | A node's neighborhood |
| `retrain_model()` | `bool` | Force NPLL retrain |
| `has_npll` | `bool` | Whether NPLL is active |
| `get_status()` | `dict` | Engine status |

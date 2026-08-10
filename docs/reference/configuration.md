---
icon: material/cog
---

# Configuration

All configuration is passed as arguments — there are no global settings or environment variables that Odin reads on its own. This page collects every parameter in one place.

---

## Engine construction

`OdinEngine(...)` — see the [OdinEngine API](engine.md#constructor).

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `db` | `StandardDatabase` | — | Connected `python-arango` handle |
| `community_id` | `str` | `"global"` | Exploration scope |
| `cache_size` | `int` | `5000` | Graph-accessor LRU size — see [Caching](../concepts/caching.md) |
| `auto_train` | `bool` | `True` | Train NPLL on first run if no model exists |
| `community_mode` | `str` | `"none"` | `"none"` = global, `"mapping"` = scoped |

---

## Retrieval

`retrieve(...)` — see [Tuning Retrieval](../guides/tuning.md).

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `seeds` | `list[str]` | — | Where exploration starts |
| `max_paths` | `int` | `50` | Output size |
| `hop_limit` | `int` | `3` | Maximum path length |
| `beam_width` | `int` | `64` | Paths kept per hop |

---

## Anchors

`find_anchors(...)`

| Parameter | Type | Default |
|-----------|------|---------|
| `seeds` | `list[str]` | — |
| `topn` | `int` | `20` |

---

## Schema inspection

`SchemaInspector(...)`

| Parameter | Type | Default |
|-----------|------|---------|
| `db` | `StandardDatabase` | — |
| `max_sample_docs` | `int` | `5` |

`inspect_arango_schema(db, output_file="schema.json")`.

---

## Database connection (yours)

Odin does not manage the connection — you construct it with `python-arango`. In production, source these from the environment:

| Variable (suggested) | Used for |
|----------------------|----------|
| `ARANGO_HOSTS` | ArangoDB URL(s) |
| `ARANGO_DB` | Database name |
| `ARANGO_USER` | Username |
| `ARANGO_PASSWORD` | Password (from a secrets manager) |

```python
import os
from arango import ArangoClient

db = ArangoClient(hosts=os.environ["ARANGO_HOSTS"]).db(
    os.environ["ARANGO_DB"],
    username=os.environ["ARANGO_USER"],
    password=os.environ["ARANGO_PASSWORD"],
)
```

See [Connecting ArangoDB](../guides/arangodb.md).

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.9 |
| ArangoDB | ≥ 3.10 |
| PyTorch | ≥ 2.0 |

Runtime dependencies (`torch`, `python-arango`, `numpy`, `scipy`, `networkx`, `scikit-learn`, `gremlinpython`) install automatically with `pip install odin-engine`.

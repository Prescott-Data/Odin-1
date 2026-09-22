---
icon: material/cog
---

# Configuration

All configuration is passed as arguments; Odin reads no global settings or environment variables of its own. This page collects every parameter in one place.

---

## Engine construction

`OdinEngine(...)`. See the [OdinEngine API](engine.md#constructor).

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `backend` | `GraphBackend` | required | Retrieval backend; NPLL training needs source and store capabilities |
| `community_id` | `str` | `"global"` | Exploration scope |
| `cache_size` | `int` | `5000` | Graph-accessor LRU size (see [Caching](../concepts/caching.md)) |
| `auto_train` | `bool` | `True` | Train NPLL on first run if no model exists |
| `community_mode` | `str` | `"none"` | `"none"` = global, `"mapping"` = scoped |

---

## Retrieval

`retrieve(...)`. See [Tuning Retrieval](../guides/tuning.md).

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `seeds` | `list[str]` | required | Where exploration starts |
| `max_paths` | `int` | `50` | Output size |
| `hop_limit` | `int` | `3` | Maximum path length |
| `beam_width` | `int` | `64` | Paths kept per hop |

---

## Anchors

`find_anchors(...)`

| Parameter | Type | Default |
|-----------|------|---------|
| `seeds` | `list[str]` | required |
| `topn` | `int` | `20` |

---

## Schema inspection

`inspect_schema(backend, ...)`

| Parameter | Type | Default |
|-----------|------|---------|
| `backend` | `GraphBackend` | required |
| `refresh` | `bool` | `False` |

`inspect_schema(backend, output_file="schema.json")`.

---

## Database connection (yours)

Odin does not manage connections. For ArangoDB, construct the connection with
`python-arango` and wrap it in `ArangoBackend`. Other backends use their own
connection mechanisms. In production, source these Arango settings from the environment:

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
| ArangoDB (Arango backend only) | ≥ 3.10 |
| PyTorch | ≥ 2.0 |

For this unreleased source checkout, install core dependencies with
`pip install -e .`. Use `pip install -e ".[arango]"` or
`pip install -e ".[gremlin]"` for the corresponding driver.
See [Backend migration](../guides/backend-migration.md).

## NPLL dataset configuration

`get_config("OdinTriples")` in `npll.utils.config` selects the explicit
production graph configuration. Its 32-dimensional entity/relation embeddings,
64-dimensional rules, and 64-unit scorer are sized for production graph
lifecycle operation. Bootstrap separately sets its trainer budget to 10 epochs
and up to 5 E-M iterations per epoch.
Unknown names, including the retired `ArangoDB_Triples`, raise `ValueError`.

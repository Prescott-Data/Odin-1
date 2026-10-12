---
icon: material/table-search
---

# Schema Inspection API

`inspect_schema` delegates schema discovery to the configured backend. A
backend that does not implement schema inspection raises `BackendCapabilityError`.

```python
from odin import inspect_schema
```

## `inspect_schema`

```python
inspect_schema(
    backend,
    *,
    refresh: bool = False,
    output_file: str | None = None,
) -> dict
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `GraphBackend` | required | Backend that supplies schema inspection |
| `refresh` | `bool` | `False` | Request a refreshed schema map |
| `output_file` | `str \| None` | `None` | Optional JSON destination for the complete map |

```python
schema = inspect_schema(backend, output_file="schema.json")
```

The schema-map shape is backend-defined. ArangoDB currently reports its
database name, document collections, edge collections, sampled fields, and
edge endpoint collections. Other backends must report only the information
they can inspect accurately.

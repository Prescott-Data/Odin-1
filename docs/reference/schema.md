---
icon: material/table-search
---

# SchemaInspector API

`SchemaInspector` discovers your ArangoDB structure at runtime. See the [Schema Introspection guide](../guides/schema-introspection.md) for usage patterns.

```python
from odin import SchemaInspector, inspect_arango_schema
```

---

## Constructor

```python
SchemaInspector(db, max_sample_docs: int = 5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | `StandardDatabase` | — | A connected `python-arango` database |
| `max_sample_docs` | `int` | `5` | Documents sampled per collection to discover fields |

---

## `get_schema_map`

```python
get_schema_map(refresh: bool = False) -> dict
```

Returns the full schema map. Cached after the first call; pass `refresh=True` to rebuild.

```python
schema = inspector.get_schema_map()
schema["database_name"]   # str
schema["collections"]     # list of document collections
schema["edges"]           # list of edge collections
```

**Shape:**

```python
{
    "database_name": "my_graph",
    "collections": [
        {"name": "ExtractedEntities", "type": "document",
         "count": 12345, "fields": ["_key", "name", "type", ...]},
        ...
    ],
    "edges": [
        {"name": "ExtractedRelationships", "count": 45678,
         "from_collections": ["ExtractedEntities"],
         "to_collections": ["ExtractedEntities"],
         "fields": ["_from", "_to", "relation", ...]},
        ...
    ],
}
```

System collections (names starting with `_`) are skipped.

---

## `get_collection_info`

```python
get_collection_info(collection_name: str) -> dict | None
```

Returns the schema entry for a document **or** edge collection, or `None` if not found.

```python
info = inspector.get_collection_info("ExtractedEntities")
info["fields"]    # discovered field names
```

---

## `get_edge_info`

```python
get_edge_info(edge_collection: str) -> dict | None
```

Returns the schema entry for an edge collection, including its `from_collections` and `to_collections`.

```python
edge = inspector.get_edge_info("ExtractedRelationships")
edge["from_collections"]   # e.g. ["ExtractedEntities"]
edge["to_collections"]     # e.g. ["ExtractedEntities"]
```

---

## `inspect_arango_schema`

```python
inspect_arango_schema(db, output_file: str = "schema.json") -> dict
```

Convenience one-call helper: builds the schema map and optionally writes it to `output_file` as JSON.

```python
from odin import inspect_arango_schema

inspect_arango_schema(db, output_file="schema.json")
```

Use the exported file to prime an agent, generate documentation, or validate structures across environments.

---

## Data classes

The inspector returns plain dictionaries, backed internally by these dataclasses:

| Dataclass | Fields |
|-----------|--------|
| `CollectionSchema` | `name`, `type`, `count`, `fields` |
| `EdgeSchema` | `name`, `count`, `from_collections`, `to_collections`, `fields` |
| `SchemaMap` | `database_name`, `collections`, `edges` |

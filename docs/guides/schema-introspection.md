---
icon: material/table-search
---

# Schema Introspection

`SchemaInspector` discovers your ArangoDB structure **at runtime** — every collection, its fields, and how edge collections connect. It is designed to give AI agents the context they need to write valid queries, and to auto-generate documentation.

---

## Why introspection

Agents that write their own AQL need to know what collections and fields exist. Rather than hard-coding a schema, `SchemaInspector` reads it live, so it stays correct as your graph evolves.

---

## Basic usage

```python
from arango import ArangoClient
from odin import SchemaInspector

client = ArangoClient(hosts="http://localhost:8529")
db = client.db("my_graph", username="root", password="")

inspector = SchemaInspector(db)
schema = inspector.get_schema_map()

print(f"Database: {schema['database_name']}")
print(f"Collections: {len(schema['collections'])}")
print(f"Edge collections: {len(schema['edges'])}")
```

`get_schema_map()` returns a dictionary with `database_name`, `collections`, and `edges`. It is cached after the first call; pass `refresh=True` to rebuild it.

---

## Collection details

```python
for col in schema["collections"]:
    print(f"{col['name']}: {col['count']} docs")
    print(f"  fields: {', '.join(col['fields'][:5])}")
```

Each document collection reports its `name`, `type`, `count`, and discovered `fields`.

```python
info = inspector.get_collection_info("ExtractedEntities")
print(info["fields"])
```

---

## Edge relationships

Edge collections additionally report which collections they connect:

```python
edge = inspector.get_edge_info("ExtractedRelationships")
print("From:", edge["from_collections"])
print("To:  ", edge["to_collections"])
print("Fields:", edge["fields"])
```

This `_from`/`_to` mapping is exactly what an agent needs to construct a valid traversal.

---

## Save schema for agents or docs

```python
from odin import inspect_arango_schema

inspect_arango_schema(db, output_file="schema.json")
```

Use the exported file to:

- **Prime an agent** with graph context in its system prompt.
- **Auto-generate** database schema documentation.
- **Validate** that collection structures match across environments.
- **Track** schema evolution over time in version control.

---

## Sampling behavior

Field discovery samples a small number of documents per collection (`max_sample_docs`, default `5`). Increase it when documents are highly heterogeneous:

```python
inspector = SchemaInspector(db, max_sample_docs=25)
```

System collections (names starting with `_`) are skipped automatically.

---

## Next

- [SchemaInspector API](../reference/schema.md) — full method reference
- [AI Agent Integration](agent-integration.md)

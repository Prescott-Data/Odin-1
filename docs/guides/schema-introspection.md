---
icon: material/table-search
---

# Schema Introspection

`SchemaInspector` discovers your ArangoDB structure at runtime: every collection, its fields, and how edge collections connect. Agents that write their own AQL need that structure, and hard-coding it goes stale the moment the graph changes. Reading it live keeps it correct, which is what makes introspection useful both for priming agents and for generating documentation.

## Basic usage

Construct an inspector over a connected database and ask for the schema map:

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

`get_schema_map()` returns a dictionary with `database_name`, `collections`, and `edges`. It is cached after the first call, so pass `refresh=True` when you want to rebuild it.

## Reading collections and edges

Each document collection reports its `name`, `type`, `count`, and the `fields` discovered by sampling:

```python
for col in schema["collections"]:
    print(f"{col['name']}: {col['count']} docs")
    print(f"  fields: {', '.join(col['fields'][:5])}")

info = inspector.get_collection_info("ExtractedEntities")
print(info["fields"])
```

Edge collections additionally report which collections they connect, which is exactly the `_from`/`_to` mapping an agent needs to construct a valid traversal:

```python
edge = inspector.get_edge_info("ExtractedRelationships")
print("From:", edge["from_collections"])
print("To:  ", edge["to_collections"])
print("Fields:", edge["fields"])
```

## Exporting the schema

For agents and documentation, export the whole map to a file in one call:

```python
from odin import inspect_arango_schema

inspect_arango_schema(db, output_file="schema.json")
```

That exported file has several uses: priming an agent with graph context in its system prompt, auto-generating database schema documentation, validating that collection structures match across environments, and tracking schema evolution over time in version control.

## Sampling behavior

Field discovery samples a small number of documents per collection, set by `max_sample_docs` (default `5`). Raise it when your documents are highly heterogeneous and a small sample would miss fields:

```python
inspector = SchemaInspector(db, max_sample_docs=25)
```

System collections, whose names start with `_`, are skipped automatically.

## Next

The full method surface is in the [SchemaInspector API](../reference/schema.md), and [AI Agent Integration](agent-integration.md) shows introspection used to give an agent graph context.

---
icon: material/table-search
---

# Schema Introspection

Schema introspection is a backend capability. It lets agents inspect the graph
structure they are about to query without assuming an ArangoDB data model.

## Basic usage

Construct a backend, then ask Odin for its complete schema map:

```python
from odin import inspect_schema
from retrieval.backends.arango import ArangoBackend, ArangoGraphConfig

graph = ArangoGraphConfig(
    node_collection="CaseRecords",
    edge_collection="EvidenceLinks",
    relation_field="predicate",
)
backend = ArangoBackend(db, graph)
schema = inspect_schema(backend)
```

The returned map is backend-defined. ArangoDB reports document collections,
edge collections, sampled fields, and edge endpoint collections. A backend
that cannot inspect its schema raises `BackendCapabilityError` rather than
inventing incomplete metadata.

## Exporting the schema

For agents and documentation, write the complete map to JSON:

```python
inspect_schema(backend, output_file="schema.json")
agent.load_context("schema.json")
```

The exported file can prime an agent with graph context, generate database
documentation, validate structures across environments, and track schema
evolution in version control.

## Refreshing cached inspection

When a backend caches inspection results, request a new map after a schema
change:

```python
schema = inspect_schema(backend, refresh=True)
```

## Next

The full method surface is in the [Schema Inspection API](../reference/schema.md),
and [AI Agent Integration](agent-integration.md) shows introspection used to
give an agent graph context.

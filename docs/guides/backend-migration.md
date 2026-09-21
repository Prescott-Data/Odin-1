---
icon: material/swap-horizontal
---

# Migrating to explicit backends

These breaking changes are **unreleased** and describe this source checkout.
Use the checkout containing the backend migration; the published 0.3.0 package
does not provide this API. A package release and version bump are separate steps.

## Install the backend driver

From the repository root:

```bash
pip install -e ".[arango]"
```

The core package no longer requires `python-arango` or `gremlinpython`.
Use `.[gremlin]` for the JanusGraph accessor, or `.[dev]` for development tools
and both drivers. Importing `odin` and using an in-memory backend do not require
either database driver.

## Wrap your existing Arango connection

Replace `OdinEngine(db)` with an explicit backend and graph mapping:

```python
from arango import ArangoClient
from odin import OdinEngine, inspect_schema
from retrieval.backends.arango import ArangoBackend, ArangoGraphConfig

client = ArangoClient(hosts="http://localhost:8529")
db = client.db("my_graph", username="root", password="your-password")
graph = ArangoGraphConfig(
    node_collection="CaseRecords",
    edge_collection="EvidenceLinks",
    relation_field="predicate",
    entity_type_field="record_type",
)
backend = ArangoBackend(db, graph)
engine = OdinEngine(backend, community_id="global", community_mode="none")

# The seed must exist in your populated graph.
result = engine.retrieve(seeds=["CaseRecords/your_entity"])
schema = inspect_schema(backend)
```

Connections and credentials remain caller-owned. The mapping is required: Odin
never assumes collection or field names.

| `ArangoGraphConfig` field | Arango requirement |
| --- | --- |
| `node_collection` | Document collection whose `_id` values are seed and node identities. |
| `edge_collection` | Edge collection whose `_from` and `_to` point to those node `_id` values. |
| `relation_field` | Every trainable edge has a non-empty string predicate in this field. |
| `entity_type_field` | Optional node field; each non-null value must be a non-empty string. Odin trains it as `has_type`. |
| `membership_*` fields | Optional, all-or-nothing mapping used only with `community_mode="mapping"`. |
| `provenance_edge_collection` | Optional edge collection used for retrieval provenance. |

Arango `_id`, `_from`, and `_to` values are opaque canonical identities; Odin
does not need a particular `_key`, collection prefix, label, or node schema.
Missing endpoints are excluded from the training snapshot. An invalid relation
or configured type value stops training clearly instead of being coerced or
dropped. See [Connecting ArangoDB](arangodb.md) for a complete example.

## Update imports and direct component calls

| Previous API | Current API |
| --- | --- |
| `OdinEngine(db)` | `OdinEngine(ArangoBackend(db, graph))` |
| `from odin import SchemaInspector, inspect_arango_schema` | `from odin import inspect_schema` |
| `SchemaInspector(db).get_schema_map()` | `inspect_schema(backend)` |
| `inspect_arango_schema(db, output_file="schema.json")` | `inspect_schema(backend, output_file="schema.json")` |
| `from retrieval import ArangoCommunityAccessor, GlobalGraphAccessor` | Import from `retrieval.adapters_arango` |
| `from retrieval.adapters import JanusGraphAccessor` | Import from `retrieval.adapters_janus` |
| `from retrieval import JanusGraphAccessor` | Import from `retrieval.adapters_janus` |
| `from retrieval import ArangoWriter` | Import from `retrieval.writers.arango_writer` |
| `from retrieval import JanusGraphWriter` | Import from `retrieval.writers.janus_writer` |
| `get_config("ArangoDB_Triples")` | `get_config("OdinTriples")` |

The writer imports also move out of `retrieval.writers`. Unknown configuration
names raise `ValueError`; `OdinTriples` preserves the previously effective
training settings under an explicit name. Old imports and raw database handles
are not supported through compatibility aliases.

Direct bootstrap users now inject capabilities:

```python
from npll import KnowledgeBootstrapper

bootstrap = KnowledgeBootstrapper(
    backend.triple_source(),
    backend.model_store("global", "none"),
)
ready = bootstrap.ensure_model_ready()
```

## Expect a new model artifact on first startup

Arango training now uses full document IDs and exact relation labels, including
case and spaces. The fingerprint covers the extracted triples and entity types,
so changing endpoints or types invalidates cached weights even when counts stay
constant. Stored rules that differ from current generated rules trigger retraining.

Artifacts live in `OdinModels`, namespaced by database, the complete graph
mapping, community ID, and community mode. Old unscoped artifacts remain
untouched and are not reused. Changing any mapping creates a distinct model
namespace. Plan for training on the first startup after migration.
Training still reads the global graph; community settings scope retrieval and
artifact storage, not the training data.

Saves use revision-checked atomic replacement. Concurrent trainers can receive
`ModelConflictError`; the engine does not silently overwrite the winning model.
Artifacts preserve all relation names, rules, and training-report histories.
The existing weights-only design does not persist embeddings or scoring-network
parameters, so reloading does not guarantee identical neural scores.

## Choose training behavior explicitly

`auto_train=True` requires a training source and model store. Missing capabilities
raise `BackendCapabilityError`. Training failures and an initialization result
without a model raise `npll.TrainingError`. Failed retraining preserves the active
serving state and raises; it does not return `False` or substitute constant scores.

For structural retrieval without training, choose:

```python
engine = OdinEngine(backend, auto_train=False)
```

This uses constant confidence alongside PPR and beam search. It does not load a
cached NPLL model. For error handling, see [Troubleshooting](../reference/troubleshooting.md).

## Available capabilities

| Implementation | Engine integration | Training and persistence | Schema inspection |
| --- | --- | --- | --- |
| `ArangoBackend` | Supplied backend | Supported | Supported |
| `JanusGraphAccessor` | Direct orchestrator or caller-written backend | Not supplied | Not supplied |
| `KGCommunityAccessor` | Direct orchestrator or caller-written backend | Not supplied | Not supplied |
| Custom backend | Implements `GraphBackend` | Optional source/store capabilities | Optional introspector |

Neo4j engine support and cross-database parity validation are not included in
this checkout. See [Adapters](../concepts/adapters.md) for custom retrieval
integration and [Backend contracts](../development/backend-contracts.md) for
implementing training and persistence.

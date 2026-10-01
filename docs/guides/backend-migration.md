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
| `edge_weight_field` | Optional numeric edge field. Without it, every traversed edge has structural weight `1.0`; Odin never assumes `weight`. |
| `edge_timestamp_field` / `edge_valid_from_field` / `edge_valid_to_field` / `edge_status_field` | Optional edge metadata fields. Omit a field to disable that metadata signal; Odin never assumes timestamp or status names. |
| `edge_provenance_fields` | Optional tuple of edge fields preserved as retrieval provenance. Odin never assumes a provenance-field name. |
| `community_property_field` | Required only with `community_mode="property"`; no property field is assumed. |
| `membership_*` fields | Optional, all-or-nothing mapping used only with `community_mode="mapping"`. |
| `provenance_edge_collection` | Optional edge collection used for retrieval provenance. |
| `bridge_collection` / `affinity_collection` / `community_algorithm` | Optional, all three together enable bridge and affinity scoring. With membership, they also enable the global cross-community accessor. |

Arango `_id`, `_from`, and `_to` values are opaque canonical identities; Odin
does not need a particular `_key`, collection prefix, label, or node schema.
Missing endpoints are excluded from the training snapshot. An invalid relation
or configured type value stops training clearly instead of being coerced or
dropped. See [Connecting ArangoDB](arangodb.md) for a complete example.
Without the complete bridge, affinity, and algorithm configuration, ordinary
retrieval performs no bridge or affinity query. Without the complete membership,
bridge, affinity, and algorithm configuration, `backend.global_accessor()`
returns `None` rather than probing assumed collection names.

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
names raise `ValueError`; `OdinTriples` provides an explicit compact production
profile. Old imports and raw database handles are not supported through
compatibility aliases.

Direct bootstrap users now inject capabilities:

```python
from npll import KnowledgeBootstrapper

bootstrap = KnowledgeBootstrapper(
    backend.triple_source(),
    backend.model_store("global", "none"),
)
ready = bootstrap.ensure_model_ready()
```

Direct `ArangoCommunityAccessor` construction is also explicit: pass the node
collection, edge collection, and relation property. Its standalone analytics
and content helpers require every collection and queried field as arguments.
They return complete Arango documents (and provenance edges where applicable),
not a Prescott-shaped subset. Prefer `backend.accessor(...)` for normal engine
retrieval.

## Expect a new model artifact on first startup

Arango training now uses full document IDs and exact relation labels, including
case and spaces. The fingerprint covers the extracted triples and entity types,
so changing endpoints or types invalidates cached weights even when counts stay
constant. Stored rules that differ from current generated rules trigger retraining.

Artifacts live in `OdinModels`, namespaced by database and training collection/field
mappings. Retrieval communities share the globally trained model. Changing bridge,
affinity, timestamp, or retrieval scope settings does not duplicate it.

Saves use revision-checked atomic replacement. On a startup race, the losing worker
loads the winner only after validating its snapshot, rules, configuration, and
scorer recipe. Forced retraining conflicts still raise `ModelConflictError`.

Artifact version `6.0` records the complete rule report and neural scorer training
recipe, runtime version, example count, and loss history. Scorer parameters and
internal embedding vectors are reconstructed by deterministic training replay;
they are not persisted. Reload verifies replay evidence and preserves identical
scores under the recorded runtime and configuration. Older artifacts fail the
version boundary clearly and must be replaced through a deliberate migration.

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

### Training artifact identity

Arango training reads the global graph. Its model namespace includes the database,
node collection, edge collection, relation field, and optional entity type field.
Retrieval community IDs, scope modes, bridge mappings, and timestamp mappings do
not create separate copies or trigger retraining. The exact triple snapshot hash
invalidates a model when training evidence changes.

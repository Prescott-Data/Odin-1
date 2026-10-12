# Migrating from 0.3.0

Version 0.3.0 constructed the engine from a raw ArangoDB handle. 0.4.0 requires an explicit backend and an explicit graph mapping. There are no compatibility aliases: old imports and raw handles fail.

Authoritative guide: https://odin.developers.prescottdata.io/guides/backend-migration/

Install the driver as an extra: `pip install "odin-engine[arango]"` (`[gremlin]` for the JanusGraph accessor). The core package no longer requires `python-arango` or `gremlinpython`.

## Constructing the engine

```python
# 0.3.0
engine = OdinEngine(db)

# 0.4.0
from odin.backends.arango import ArangoBackend, ArangoGraphConfig
graph = ArangoGraphConfig(node_collection="CaseRecords", edge_collection="EvidenceLinks",
                          relation_field="predicate")
engine = OdinEngine(ArangoBackend(db, graph))
```

Odin no longer assumes collection names or field names. Map yours (see `arango-config.md`).

## API changes

| Before | Now |
| --- | --- |
| `OdinEngine(db)` | `OdinEngine(ArangoBackend(db, graph))` |
| `from odin import SchemaInspector, inspect_arango_schema` | `from odin import inspect_schema` |
| `SchemaInspector(db).get_schema_map()` | `inspect_schema(backend)` |
| `inspect_arango_schema(db, output_file="schema.json")` | `inspect_schema(backend, output_file="schema.json")` |
| `from retrieval import ArangoCommunityAccessor, GlobalGraphAccessor` | `from retrieval.adapters_arango import ...` |
| `from retrieval.adapters import JanusGraphAccessor` or `from retrieval import JanusGraphAccessor` | `from retrieval.adapters_janus import JanusGraphAccessor` |
| `from retrieval import ArangoWriter` | `from retrieval.writers.arango_writer import ArangoWriter` |
| `from retrieval import JanusGraphWriter` | `from retrieval.writers.janus_writer import JanusGraphWriter` |
| `get_config("ArangoDB_Triples")` | `get_config("OdinTriples")` |
| `engine.global_accessor` | `backend.global_accessor()` |

Direct bootstrap users inject capabilities: `KnowledgeBootstrapper(backend.triple_source(), backend.model_store())`.

## Behavior changes

- Training and retraining failures raise (`npll.TrainingError`) instead of returning `False` or switching to constant scores. Pass `auto_train=False` only for deliberate structural retrieval.
- Training identities are full document IDs and exact relation labels. Expect a new model artifact on first startup; old weights are not reused.
- Standalone helpers return complete documents, including vector fields. Filter model input in the consuming application.
- `GlobalGraphAccessor` lost its old keyword constructor, `min_affinity_threshold` and `score_community_crossing`.

## JanusGraph

`JanusGraphAccessor` is a retrieval accessor only. It has no engine backend, training or schema inspection. Compose it with `RetrievalOrchestrator` directly.

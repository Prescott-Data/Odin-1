---
icon: material/database
---

# Connecting ArangoDB

Odin's reference backend is [ArangoDB](https://www.arangodb.com/). This guide covers connecting to it, doing so safely in production, and scoping exploration to a community.

## Connecting

Install the ArangoDB Python driver with the `arango` extra. It imports as
`arango`; connect, select a database, and wrap the connected handle in
`ArangoBackend`. From the unreleased source checkout:

```bash
pip install -e ".[arango]"
```

```python
from arango import ArangoClient
from odin import OdinEngine
from retrieval.backends.arango import ArangoBackend, ArangoGraphConfig

client = ArangoClient(hosts="http://localhost:8529")
db = client.db("my_graph", username="root", password="")

graph = ArangoGraphConfig(
    node_collection="CaseRecords",
    edge_collection="EvidenceLinks",
    relation_field="predicate",
    entity_type_field="record_type",
)
backend = ArangoBackend(db, graph)
engine = OdinEngine(backend)
```

The important detail is that `OdinEngine` takes a backend, while `ArangoBackend`
uses an already-connected `StandardDatabase` object and never manages credentials
itself. That keeps secrets in your connection code and out of Odin entirely.

## Map your graph data

`ArangoGraphConfig` is required and makes the backend portable across your
existing Arango schemas. Odin reads node IDs from `_id`, edge endpoints from
`_from` and `_to`, and relation labels from the field you configure. It does
not require collection names such as `ExtractedEntities` or a node label field.

For the configuration above, an edge looks like:

```json
{
  "_from": "CaseRecords/claim_1042",
  "_to": "CaseRecords/person_73",
  "predicate": "submitted_by"
}
```

`predicate` must be a non-empty string on every edge included in training.
`record_type` is optional, but when configured its non-null node values must be
non-empty strings. Odin adds these as `has_type` training triples. Optional
retrieval metadata such as `weight`, `created_at`, and provenance can remain in
your own fields; it is not required to train or retrieve.

## Production connections

Outside local development, never disable authentication. Use a dedicated, least-privilege user and pull secrets from the environment rather than the source:

```python
import os
from arango import ArangoClient

client = ArangoClient(hosts=os.environ["ARANGO_HOSTS"])   # e.g. https://db.internal:8529
db = client.db(
    os.environ["ARANGO_DB"],
    username=os.environ["ARANGO_USER"],
    password=os.environ["ARANGO_PASSWORD"],
)
```

!!! warning "Secrets"
    Do not hard-code passwords in source or notebooks. Load them from environment variables or a secrets manager. Odin never logs your credentials.

## A local database with Docker

For development, Docker gives you an instance in one command:

```bash
docker run -d --name arango -p 8529:8529 \
  -e ARANGO_NO_AUTH=1 arangodb:3.12
```

`ARANGO_NO_AUTH` is fine locally but never in production. For an authenticated local instance, set a root password instead:

```bash
docker run -d --name arango -p 8529:8529 \
  -e ARANGO_ROOT_PASSWORD=change-me arangodb:3.12
```

## Scoping to a community

A [community](../concepts/data-model.md#communities-scope-the-graph) restricts exploration to a named subset of the graph. Reach for `community_mode="mapping"` when you have partitioned a large multi-tenant graph and want scoped retrieval. Training remains global for the Arango backend:

```python
# Global exploration (default); `graph` is the explicit mapping above.
backend = ArangoBackend(db, graph)
engine = OdinEngine(backend, community_id="global", community_mode="none")

# Scoped to one partition. Add all three membership fields to `graph`.
backend = ArangoBackend(db, graph)
engine = OdinEngine(backend, community_id="medicare_claims", community_mode="mapping")
```

Mapping mode requires `membership_collection`, `membership_entity_field`, and
`membership_community_field` together. Without that complete mapping Odin
raises `BackendConfigurationError`; it never guesses a membership schema.

## Verifying it worked

`get_status()` confirms the connection and which intelligence mode you are in:

```python
print(engine.get_status())
# {'community_id': 'global', 'npll_loaded': True,
#  'intelligence_mode': 'NPLL', 'cache_size': 5000}
```

An `intelligence_mode` of `Constant` means the NPLL model has not trained yet, which [Model Lifecycle](npll-lifecycle.md) explains how to resolve.

## Other backends

The accessor layer is an interface (`retrieval/adapters.py`). Odin ships adapters for ArangoDB and JanusGraph, and you can implement the same contract for other stores. See [Adapters](../concepts/adapters.md) for the full picture.

## Next

With a connection in place, run [Your First Retrieval](first-retrieval.md), or let an agent discover the graph's structure with [Schema Introspection](schema-introspection.md).

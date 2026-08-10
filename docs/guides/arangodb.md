---
icon: material/database
---

# Connecting ArangoDB

Odin's reference backend is [ArangoDB](https://www.arangodb.com/). This guide covers connecting to it, doing so safely in production, and scoping exploration to a community.

## Connecting

The ArangoDB Python driver (`python-arango`) is installed automatically with `odin-engine` and imports as `arango`. You connect, select a database, and hand the resulting object to the engine:

```python
from arango import ArangoClient
from odin import OdinEngine

client = ArangoClient(hosts="http://localhost:8529")
db = client.db("my_graph", username="root", password="")

engine = OdinEngine(db=db)
```

The important detail is that `OdinEngine` takes an already-connected `StandardDatabase` object and never manages credentials itself. That keeps secrets in your connection code and out of Odin entirely.

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

A [community](../concepts/data-model.md#communities-scope-the-graph) restricts exploration to a named subset of the graph. Reach for `community_mode="mapping"` when you have partitioned a large multi-tenant graph and want both retrieval and the NPLL model focused on one partition:

```python
# Global exploration (default)
engine = OdinEngine(db, community_id="global", community_mode="none")

# Scoped to one partition
engine = OdinEngine(db, community_id="medicare_claims", community_mode="mapping")
```

## Verifying it worked

`get_status()` confirms the connection and which intelligence mode you are in:

```python
print(engine.get_status())
# {'community_id': 'global', 'npll_loaded': True,
#  'intelligence_mode': 'NPLL', 'cache_size': 5000}
```

An `intelligence_mode` of `Constant` means the NPLL model has not trained yet, which [Model Lifecycle](npll-lifecycle.md) explains how to resolve.

## Other backends

The accessor layer is an interface (`retrieval/adapters.py`), so backends beyond ArangoDB can be added by implementing the same node and edge access contract. If you build one, [contributions](https://github.com/Prescott-Data/Odin-1/blob/main/CONTRIBUTING.md) are welcome.

## Next

With a connection in place, run [Your First Retrieval](first-retrieval.md), or let an agent discover the graph's structure with [Schema Introspection](schema-introspection.md).

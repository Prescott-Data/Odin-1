---
icon: material/database
---

# Connecting ArangoDB

Odin's reference backend is [ArangoDB](https://www.arangodb.com/). This guide covers connecting, choosing a database, and scoping exploration to a community.

---

## Install the driver

The ArangoDB Python driver (`python-arango`) is installed automatically with `odin-engine`. You import it as `arango`:

```python
from arango import ArangoClient
```

---

## Basic connection

```python
from arango import ArangoClient
from odin import OdinEngine

client = ArangoClient(hosts="http://localhost:8529")
db = client.db("my_graph", username="root", password="")

engine = OdinEngine(db=db)
```

`OdinEngine` takes an already-connected `StandardDatabase` object — it never manages credentials itself. That keeps secrets in your connection code and out of Odin.

---

## Authenticated / production connections

Never disable authentication outside local development. Use a dedicated, least-privilege user and pull secrets from the environment:

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

---

## Local database with Docker

```bash
docker run -d --name arango -p 8529:8529 \
  -e ARANGO_NO_AUTH=1 arangodb:3.12
```

For an authenticated local instance, use `ARANGO_ROOT_PASSWORD` instead of `ARANGO_NO_AUTH`:

```bash
docker run -d --name arango -p 8529:8529 \
  -e ARANGO_ROOT_PASSWORD=change-me arangodb:3.12
```

---

## Scoping to a community

A [community](../concepts/data-model.md#communities-scope-the-graph) restricts exploration to a named subset of the graph:

```python
# Global exploration (default)
engine = OdinEngine(db, community_id="global", community_mode="none")

# Scoped exploration
engine = OdinEngine(db, community_id="medicare_claims", community_mode="mapping")
```

Use `community_mode="mapping"` when you have partitioned a large multi-tenant graph and want retrieval — and the NPLL model — focused on one partition.

---

## Verifying the connection

```python
print(engine.get_status())
# {'community_id': 'global', 'npll_loaded': True,
#  'intelligence_mode': 'NPLL', 'cache_size': 5000}
```

If `intelligence_mode` is `Constant`, the NPLL model has not trained yet — see [Model Lifecycle](npll-lifecycle.md).

---

## Other backends

The accessor layer is an interface (`retrieval/adapters.py`). Backends beyond ArangoDB can be added by implementing the same node/edge access contract. See [Contributing](https://github.com/Prescott-Data/Odin-1/blob/main/CONTRIBUTING.md).

---

## Next

- [Your First Retrieval](first-retrieval.md)
- [Schema Introspection](schema-introspection.md)

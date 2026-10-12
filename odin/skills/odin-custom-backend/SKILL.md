---
name: odin-custom-backend
description: Use when implementing an Odin backend for a graph database other than ArangoDB (for example Neo4j), or a retrieval-only in-memory backend for tests. Covers the GraphBackend, accessor, training and model-store contracts of the odin-engine explicit-backend API (0.4.0 and later) and how the engine validates them. For using an existing backend, use the odin skill instead.
---

# Writing an Odin backend

An Odin backend is a plain object that gives `OdinEngine` an accessor for retrieval and, optionally, a training source and model store. There is no base class to inherit.

## Check the installed library first

```bash
python -c "from odin.backends import GraphBackend, TripleSource, ModelStore"
```

If that fails, the installed Odin predates the explicit-backend API (0.4.0). Do not use these examples. If the files named below are missing, use https://odin.developers.prescottdata.io/development/backend-contracts/ (index for agents: https://odin.developers.prescottdata.io/llms.txt).

## Choose a scope

| Goal | Implement | Construct with |
| --- | --- | --- |
| Retrieval only | `accessor(community_id, community_mode)` | `OdinEngine(backend, auto_train=False)` |
| Retrieval plus learned scoring | the above, `triple_source()` and `model_store()` | `OdinEngine(backend)` |
| Schema inspection | `schema_inspector()` | `inspect_schema(backend)` |

Start with retrieval only and prove it with a test before adding training. Read `references/protocols.md` for every signature and rule.

## Minimal retrieval-only backend

```python
from odin import OdinEngine

EDGES = [("A", "linked_to", "B"), ("B", "linked_to", "C")]
NODES = sorted({n for u, _, v in EDGES for n in (u, v)})


class Accessor:
    def iter_out(self, node):
        for u, rel, v in EDGES:
            if u == node:
                yield v, rel, 1.0

    def iter_in(self, node):
        for u, rel, v in EDGES:
            if v == node:
                yield u, rel, 1.0

    def iter_out_edges(self, node):
        for i, (u, rel, v) in enumerate(EDGES):
            if u == node:
                yield {"_id": f"e{i}", "u": u, "rel": rel, "v": v, "weight": 1.0}

    def nodes(self, community_id):
        return list(NODES)

    def degree(self, node):
        return sum(1 for u, _, _ in EDGES if u == node)

    def get_node(self, node_id, fields=None):
        return {"id": node_id} if node_id in NODES else {}

    def community_seed_norm(self, community_id, seeds):
        return seeds


class Backend:
    def accessor(self, community_id, community_mode):
        if community_mode != "none":
            raise ValueError("this backend supports global retrieval only")
        return Accessor()


engine = OdinEngine(Backend(), auto_train=False)
result = engine.retrieve(seeds=["A"])
```

## Rules that cause the most mistakes

- All seven accessor methods are required, including `iter_out_edges`. The engine checks at construction and raises `BackendConfigurationError` naming the missing ones. A method inherited unchanged from the `GraphAccessor` protocol counts as missing, so implement `community_seed_norm` explicitly even when it returns the seeds unchanged.
- Node IDs from the accessor and entity IDs in training triples must be the same exact strings. Relation labels are exact and case-sensitive.
- `iter_out` and `iter_out_edges` must describe the same edges. Each edge record needs `_id`, `u`, `rel`, `v` and `weight`.
- A retrieval-only backend must be built with `auto_train=False`. With training on and no `triple_source` or `model_store`, the engine raises `BackendCapabilityError`.
- Failures must raise the documented exceptions (`BackendIOError`, `ModelConflictError`, ...). Never return an empty result or a missing model to hide a failure. `ModelStore.load` returns `None` only when the artifact is absent.
- Do not add Arango-specific names, collections or fields to a backend for another database.

## Test it

Before claiming it works, run it against `OdinEngine` without a live database, as in the example, and check that a bad accessor (missing method) raises `BackendConfigurationError` and that `retrieve()` returns edges with the relation labels you stored.

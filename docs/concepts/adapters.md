---
icon: material/power-plug
---

# Adapters

Everything above the graph, PPR, beam search, aggregation, works against an abstract view of your data called a **graph accessor**. An adapter is a concrete implementation of that view for a particular backend. ArangoDB is the reference adapter and the one `OdinEngine` wires up for you, but the interface is small and deliberately backend-agnostic, so the community can connect Odin to other knowledge graphs.

## The accessor interface

An adapter implements the `GraphAccessor` protocol from `retrieval.adapters`. It is only a handful of methods:

```python
from typing import Iterable, Tuple, List
from retrieval.adapters import GraphAccessor, NodeId, RelId

class MyAccessor(GraphAccessor):

    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Yield (neighbor, relation, weight) for each outgoing edge."""
        ...

    def nodes(self, community_id: str) -> Iterable[NodeId]:
        """All node IDs in this community."""
        ...

    def degree(self, node: NodeId) -> int:
        """Out-degree of a node."""
        ...

    def community_seed_norm(self, community_id: str, seeds: List[NodeId]) -> List[NodeId]:
        """Optional: map external seed IDs to internal ones. Default: passthrough."""
        return seeds
```

If your backend can traverse outgoing edges and enumerate nodes, it can drive Odin. Everything the engine needs, PPR, beam search, and aggregation, is expressed in terms of these methods.

## Adapters that ship with Odin

| Adapter | Backend | Notes |
|---------|---------|-------|
| `ArangoCommunityAccessor` | ArangoDB | The reference adapter. Rich edges (weights, timestamps, provenance), community mapping, and the backend `OdinEngine` uses. |
| `JanusGraphAccessor` | JanusGraph | Runs over a Gremlin traversal source; validated against JanusGraph 1.x with the GraphSON v3 serializer. |
| `KGCommunityAccessor` | In-memory `KnowledgeGraph` | For a pre-sliced community held in memory. |
| `OverlayAccessor` | Any accessor | Wraps a base accessor and adds soft overlay edges on top. |

Because JanusGraph speaks Gremlin, `JanusGraphAccessor` is also a starting point for other Gremlin-compatible stores such as Amazon Neptune — those targets are untested, so validate before relying on them.

## Using a non-Arango adapter

`OdinEngine` is a convenience wrapper that constructs the ArangoDB adapter from a `db` handle. To drive Odin with a different backend, compose the retrieval orchestrator directly with your accessor:

```python
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.driver.serializer import GraphSONSerializersV3d0
from gremlin_python.process.anonymous_traversal import traversal

from retrieval.orchestrator import RetrievalOrchestrator, OrchestratorParams
from retrieval.confidence import ConstantConfidence
from retrieval.adapters import JanusGraphAccessor

# GraphSON v3 serializer: newer gremlinpython clients default to GraphBinary,
# which JanusGraph 1.x (TinkerPop 3.7) servers reject.
conn = DriverRemoteConnection(
    "ws://localhost:8182/gremlin", "g",
    message_serializer=GraphSONSerializersV3d0(),
)
g = traversal().with_remote(conn)

accessor = JanusGraphAccessor(g)

orchestrator = RetrievalOrchestrator(
    accessor=accessor,
    edge_confidence=ConstantConfidence(0.8),  # see note on NPLL below
)

result = orchestrator.retrieve(
    seeds=["v_123"],
    params=OrchestratorParams(
        community_id="global",
        max_paths=50,
        hop_limit=3,
        beam_width=64,
    ),
)
```

The result has the same shape documented in the [Result Schema](../reference/result-schema.md).

!!! note "NPLL and non-Arango backends"
    Odin's learned [NPLL](npll.md) model is bootstrapped from an ArangoDB graph, so it is tied to the ArangoDB adapter today. On other backends, pass an edge-confidence function explicitly, `ConstantConfidence(...)` to treat all edges as equally plausible, or your own implementation of the confidence interface. You still get the full PPR and beam-search machinery; you supply the semantic signal.

## Caching your adapter

Wrap any accessor in `CachedGraphAccessor` to get the same LRU caching `OdinEngine` uses (see [Caching](caching.md)):

```python
from retrieval.cache import CachedGraphAccessor

accessor = CachedGraphAccessor(JanusGraphAccessor(g), cache_size=5000)
```

## Writing your own

To add a backend, implement the four `GraphAccessor` methods for it. `iter_out` is the workhorse: given a node, return its outgoing `(neighbor, relation, weight)` tuples. The shipped `JanusGraphAccessor` and `ArangoCommunityAccessor` are the best worked references. New adapters, Neo4j, Neptune, or anything else, are welcome as [contributions](https://github.com/Prescott-Data/Odin-1/blob/main/CONTRIBUTING.md).

## Next

- [Data Model](data-model.md) covers what Odin expects from the graph itself.
- [Connecting ArangoDB](../guides/arangodb.md) is the practical setup for the reference backend.

from __future__ import annotations
from typing import Iterable, Tuple, Protocol, List, Dict, Optional
from gremlin_python.process.graph_traversal import __

NodeId = str
RelId = str


class GraphAccessor(Protocol):
    """
    Minimal graph view inside a single community/subgraph.
    Implement these methods for your KG.
    """

    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Yield (neighbor, relation, weight)."""
        ...

    def community_seed_norm(self, community_id: str, seeds: List[NodeId]) -> List[NodeId]:
        """Optional mapping from external IDs to internal; default passthrough."""
        return seeds

    def nodes(self, community_id: str) -> Iterable[NodeId]:
        """All node IDs in this community."""
        ...

    def degree(self, node: NodeId) -> int:
        """Fast out-degree if available; else len(list(iter_out(node)))."""
        ...


class KGCommunityAccessor:
    """
    Example adapter to a KnowledgeGraph with a pre-sliced community.
    Provide a set of allowed node IDs (entity names) for the community.
    """

    def __init__(self, kg, community_id: str, community_nodes: Optional[set[str]] = None):
        self.kg = kg
        self.community_id = community_id
        self.allowed = community_nodes or set(n.name for n in kg.entities)

    def iter_out(self, node: NodeId):
        ent = self.kg.get_entity(node)
        if ent is None:
            return
        for triple in self.kg.get_facts_by_head(ent):
            v = triple.tail.name
            if v in self.allowed:
                yield v, triple.relation.name, 1.0

    def nodes(self, community_id: str):
        return list(self.allowed)

    def degree(self, node: NodeId) -> int:
        ent = self.kg.get_entity(node)
        if ent is None:
            return 0
        return sum(1 for t in self.kg.get_facts_by_head(ent) if t.tail.name in self.allowed)


class OverlayAccessor:
    """
    Wrap a base GraphAccessor with soft overlay edges.
    Overlay edges are tuples (u, rel, v, weight) scoped to a community.
    """

    def __init__(self, base: GraphAccessor, community_id: str):
        self.base = base
        self.cid = community_id
        self._overlay: dict[NodeId, list[tuple[NodeId, str, float]]] = {}

    def add_edge(self, u: NodeId, rel: str, v: NodeId, weight: float = 1.0):
        self._overlay.setdefault(u, []).append((v, rel, weight))

    def iter_out(self, node: NodeId):
        # Base edges
        for v, r, w in self.base.iter_out(node):
            yield v, r, w
        # Overlay edges
        for v, r, w in self._overlay.get(node, []):
            yield v, r, w

    def community_seed_norm(self, community_id: str, seeds: list[NodeId]) -> list[NodeId]:
        return getattr(self.base, 'community_seed_norm', lambda cid, s: s)(community_id, seeds)

    def nodes(self, community_id: str):
        return getattr(self.base, 'nodes', lambda cid: [])(community_id)

    def degree(self, node: NodeId) -> int:
        base_deg = getattr(self.base, 'degree', lambda n: 0)(node)
        return base_deg + len(self._overlay.get(node, []))


class JanusGraphAccessor(GraphAccessor):
    """
    GraphAccessor implementation for JanusGraph (experimental).

    Takes a Gremlin GraphTraversalSource, e.g.:

        from gremlin_python.process.anonymous_traversal import traversal
        from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
        g = traversal().with_remote(DriverRemoteConnection("ws://localhost:8182/gremlin", "g"))
        accessor = JanusGraphAccessor(g)

    Uses project() so results arrive as plain value dicts rather than
    driver element objects. Edge weight is read from a configurable edge
    property (default: "weight", falling back to 1.0).
    """

    def __init__(self, g, community_id_property: str = "communityId",
                 weight_property: str = "weight"):
        self.g = g
        self.community_id_property = community_id_property
        self.weight_property = weight_property

    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Yield (neighbor, relation, weight) for outgoing edges."""
        rows = (
            self.g.V(node).outE()
            .project('neighbor', 'rel', 'weight')
            .by(__.inV().id_())
            .by(__.label())
            .by(__.coalesce(__.values(self.weight_property), __.constant(1.0)))
            .toList()
        )
        for row in rows:
            yield row['neighbor'], row['rel'], float(row['weight'])

    def iter_in(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Yield (neighbor, relation, weight) for incoming edges."""
        rows = (
            self.g.V(node).inE()
            .project('neighbor', 'rel', 'weight')
            .by(__.outV().id_())
            .by(__.label())
            .by(__.coalesce(__.values(self.weight_property), __.constant(1.0)))
            .toList()
        )
        for row in rows:
            yield row['neighbor'], row['rel'], float(row['weight'])

    def nodes(self, community_id: Optional[str] = None) -> Iterable[NodeId]:
        """Yield node IDs, scoped to a community when one is given."""
        if community_id:
            rows = self.g.V().has(self.community_id_property, community_id).id_().toList()
        else:
            rows = self.g.V().id_().toList()
        for node_id in rows:
            yield node_id

    def degree(self, node: NodeId) -> int:
        """Out-degree of a node."""
        return int(self.g.V(node).outE().count().next())

    def in_degree(self, node: NodeId) -> int:
        """In-degree of a node."""
        return int(self.g.V(node).inE().count().next())

    def community_seed_norm(self, community_id: str, seeds: List[NodeId]) -> List[NodeId]:
        """Passthrough: Gremlin vertex IDs are used as-is."""
        return seeds

    def get_node(self, node_id: NodeId, fields: Optional[List[str]] = None) -> Dict[str, str]:
        """Return a node's properties as plain values (first value per key)."""
        rows = self.g.V(node_id).value_map().toList()
        if not rows:
            return {}
        props = {}
        for key, value in rows[0].items():
            props[key] = value[0] if isinstance(value, list) and value else value
        if fields:
            props = {k: v for k, v in props.items() if k in fields}
        return props

from __future__ import annotations
from typing import Iterable, Tuple, Protocol, List, Dict, Optional
from gremlin_python.process.graph_traversal import __
from gremlin_python.structure.graph import Graph

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
    GraphAccessor implementation for JanusGraph.
    Assumes connection to a JanusGraph instance via a Gremlin client.
    """
    def __init__(self, graph: Graph, community_id_property: str = "communityId"):
        self.g = graph.traversal()
        self.community_id_property = community_id_property

    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """
        Yields (neighbor, relation, weight) for outgoing edges from a node.
        Example Gremlin query: g.V(node).outE().as_('e').inV().as_('v').select('e', 'v')
        """
        traversal = self.g.V(node).outE().as_('e').inV().as_('v').select('e', 'v').toList()
        for edge, neighbor in traversal:
            # Note: You might need to adjust how you access IDs and properties based on your graph schema
            neighbor_id = neighbor.id
            rel_id = edge.label
            weight = edge.properties.get('weight', 1.0)
            yield neighbor_id, rel_id, weight

    def nodes(self, community_id: str) -> Iterable[NodeId]:
        """
        Yields all node IDs in a given community.
        Example Gremlin query: g.V().has('communityId', community_id).id()
        """
        traversal = self.g.V().has(self.community_id_property, community_id).id().toList()
        for node_id in traversal:
            yield node_id

    def degree(self, node: NodeId) -> int:
        """
        Returns the out-degree of a node.
        Example Gremlin query: g.V(node).outE().count()
        """
        return self.g.V(node).outE().count().next()

    def in_degree(self, node: NodeId) -> int:
        """
        Returns the in-degree of a node.
        Example Gremlin query: g.V(node).inE().count()
        """
        return self.g.V(node).inE().count().next()


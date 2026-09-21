from __future__ import annotations
from typing import Iterable, Tuple, Protocol, List, Dict, Optional

NodeId = str
RelId = str


class GraphAccessor(Protocol):
    """
    Minimal graph view inside a single community/subgraph.
    Implement every method for your KG.
    """

    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Yield (neighbor, relation, weight)."""
        ...

    def iter_in(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Yield (neighbor, relation, weight) for incoming edges."""
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

    def get_node(self, node_id: NodeId, fields: Optional[List[str]] = None) -> Dict[str, object]:
        """Return node properties, or an empty dict when the node is absent."""
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

    def iter_in(self, node: NodeId):
        ent = self.kg.get_entity(node)
        if ent is None:
            return
        for triple in self.kg.get_facts_by_tail(ent):
            u = triple.head.name
            if u in self.allowed:
                yield u, triple.relation.name, 1.0

    def nodes(self, community_id: str):
        return list(self.allowed)

    def community_seed_norm(self, community_id: str, seeds: List[NodeId]) -> List[NodeId]:
        return seeds

    def degree(self, node: NodeId) -> int:
        ent = self.kg.get_entity(node)
        if ent is None:
            return 0
        return sum(1 for t in self.kg.get_facts_by_head(ent) if t.tail.name in self.allowed)

    def get_node(self, node_id: NodeId, fields: Optional[List[str]] = None) -> Dict[str, object]:
        ent = self.kg.get_entity(node_id)
        if ent is None or ent.name not in self.allowed:
            return {}
        properties: Dict[str, object] = {"id": ent.name}
        if fields is not None:
            return {key: value for key, value in properties.items() if key in fields}
        return properties


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

    def iter_in(self, node: NodeId):
        for u, relation, weight in self.base.iter_in(node):
            yield u, relation, weight
        for u, edges in self._overlay.items():
            for v, relation, weight in edges:
                if v == node:
                    yield u, relation, weight

    def community_seed_norm(self, community_id: str, seeds: list[NodeId]) -> list[NodeId]:
        return getattr(self.base, 'community_seed_norm', lambda cid, s: s)(community_id, seeds)

    def nodes(self, community_id: str):
        return getattr(self.base, 'nodes', lambda cid: [])(community_id)

    def degree(self, node: NodeId) -> int:
        base_deg = getattr(self.base, 'degree', lambda n: 0)(node)
        return base_deg + len(self._overlay.get(node, []))

    def get_node(self, node_id: NodeId, fields: Optional[List[str]] = None) -> Dict[str, object]:
        return self.base.get_node(node_id, fields)

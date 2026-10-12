"""JanusGraph graph accessor backed by the optional Gremlin driver."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from gremlin_python.process.graph_traversal import __

from .adapters import GraphAccessor, NodeId, RelId


class JanusGraphAccessor(GraphAccessor):
    """GraphAccessor implementation for a Gremlin traversal source."""

    def __init__(
        self,
        graph,
        community_id_property: str = "communityId",
        weight_property: str = "weight",
    ):
        self.graph = graph
        self.community_id_property = community_id_property
        self.weight_property = weight_property

    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        rows = (
            self.graph.V(node).outE()
            .project("neighbor", "rel", "weight")
            .by(__.inV().id_())
            .by(__.label())
            .by(__.coalesce(__.values(self.weight_property), __.constant(1.0)))
            .toList()
        )
        for row in rows:
            yield row["neighbor"], row["rel"], float(row["weight"])

    def iter_in(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        rows = (
            self.graph.V(node).inE()
            .project("neighbor", "rel", "weight")
            .by(__.outV().id_())
            .by(__.label())
            .by(__.coalesce(__.values(self.weight_property), __.constant(1.0)))
            .toList()
        )
        for row in rows:
            yield row["neighbor"], row["rel"], float(row["weight"])

    def iter_out_edges(self, node: NodeId):
        rows = (self.graph.V(node).outE()
                .project("id", "neighbor", "rel", "weight", "properties")
                .by(__.id_()).by(__.inV().id_()).by(__.label())
                .by(__.coalesce(__.values(self.weight_property), __.constant(1.0)))
                .by(__.value_map()).toList())
        for row in rows:
            yield {"_id": str(row["id"]), "u": node, "v": row["neighbor"],
                   "rel": row["rel"], "weight": float(row["weight"]),
                   "provenance": {"assertion": row["properties"]}}

    def nodes(self, community_id: Optional[str] = None) -> Iterable[NodeId]:
        if community_id:
            rows = (
                self.graph.V()
                .has(self.community_id_property, community_id)
                .id_()
                .toList()
            )
        else:
            rows = self.graph.V().id_().toList()
        yield from rows

    def degree(self, node: NodeId) -> int:
        return int(self.graph.V(node).outE().count().next())

    def in_degree(self, node: NodeId) -> int:
        return int(self.graph.V(node).inE().count().next())

    def community_seed_norm(
        self, community_id: str, seeds: List[NodeId]
    ) -> List[NodeId]:
        return seeds

    def get_node(
        self, node_id: NodeId, fields: Optional[List[str]] = None
    ) -> Dict[str, object]:
        rows = self.graph.V(node_id).value_map().toList()
        if not rows:
            return {}
        properties = {
            key: value[0] if isinstance(value, list) and value else value
            for key, value in rows[0].items()
        }
        if fields is not None:
            return {key: value for key, value in properties.items() if key in fields}
        return properties

"""
Experimental Neo4j adapter for the Odin retrieval layer.

Implements the GraphAccessor protocol over the official `neo4j` driver so the
retrieval pipeline (PPR + beam search + scoring) can run against a Neo4j
labeled property graph. Wrap it in CachedGraphAccessor for production use,
exactly like the ArangoDB accessor.

Boundary: this covers the retrieval layer only. OdinEngine auto-bootstrap,
NPLL training persistence, and schema introspection still require ArangoDB.

The accessor takes an already-connected driver (duck-typed), so this module
imports without the `neo4j` package installed. Install it with:

    pip install odin-engine[neo4j]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from retrieval.adapters import GraphAccessor, NodeId, RelId

logger = logging.getLogger(__name__)


class Neo4jGraphAccessor(GraphAccessor):
    """
    GraphAccessor implementation for Neo4j (experimental).

    Nodes are identified by a configurable node property (default: ``id``).
    The relationship type is used as the relation ID, and edge weight is read
    from a configurable relationship property (default: ``weight``, falling
    back to 1.0 when absent).

    Args:
        driver: A connected ``neo4j.Driver`` (or compatible object exposing
            ``execute_query``).
        database: Optional Neo4j database name.
        node_label: Optional label to scope all node matches (e.g. "Entity").
        id_property: Node property holding the stable node ID.
        weight_property: Relationship property holding the edge weight.
        community_property: Node property used by ``nodes(community_id)`` to
            scope enumeration to a community. Ignored when ``nodes`` is called
            with no community or the property is None.
    """

    def __init__(
        self,
        driver: Any,
        database: Optional[str] = None,
        node_label: Optional[str] = None,
        id_property: str = "id",
        weight_property: str = "weight",
        community_property: Optional[str] = "communityId",
    ):
        self.driver = driver
        self.database = database
        self.node_label = node_label
        self.id_property = id_property
        self.weight_property = weight_property
        self.community_property = community_property

    # -- query helpers -----------------------------------------------------

    def _label_fragment(self) -> str:
        return f":`{self.node_label}`" if self.node_label else ""

    def _run(self, query: str, **params) -> List[Any]:
        result = self.driver.execute_query(query, params, database_=self.database)
        # neo4j.EagerResult is a (records, summary, keys) namedtuple
        return list(result.records)

    # -- GraphAccessor protocol --------------------------------------------

    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Yield (neighbor, relation, weight) for outgoing edges."""
        query = (
            f"MATCH (u{self._label_fragment()} {{`{self.id_property}`: $node}})"
            f"-[r]->(v{self._label_fragment()}) "
            f"RETURN v.`{self.id_property}` AS neighbor, type(r) AS rel, "
            f"coalesce(r.`{self.weight_property}`, 1.0) AS weight"
        )
        for record in self._run(query, node=node):
            if record["neighbor"] is None:
                continue
            yield record["neighbor"], record["rel"], float(record["weight"])

    def iter_in(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Yield (neighbor, relation, weight) for incoming edges."""
        query = (
            f"MATCH (u{self._label_fragment()} {{`{self.id_property}`: $node}})"
            f"<-[r]-(v{self._label_fragment()}) "
            f"RETURN v.`{self.id_property}` AS neighbor, type(r) AS rel, "
            f"coalesce(r.`{self.weight_property}`, 1.0) AS weight"
        )
        for record in self._run(query, node=node):
            if record["neighbor"] is None:
                continue
            yield record["neighbor"], record["rel"], float(record["weight"])

    def nodes(self, community_id: Optional[str] = None) -> Iterable[NodeId]:
        """Yield node IDs, scoped to a community when configured and given."""
        if community_id and self.community_property:
            query = (
                f"MATCH (n{self._label_fragment()}) "
                f"WHERE n.`{self.community_property}` = $cid "
                f"RETURN n.`{self.id_property}` AS id"
            )
            records = self._run(query, cid=community_id)
        else:
            query = (
                f"MATCH (n{self._label_fragment()}) "
                f"RETURN n.`{self.id_property}` AS id"
            )
            records = self._run(query)
        for record in records:
            if record["id"] is not None:
                yield record["id"]

    def degree(self, node: NodeId) -> int:
        """Out-degree of a node."""
        query = (
            f"MATCH (u{self._label_fragment()} {{`{self.id_property}`: $node}})"
            f"-[r]->() RETURN count(r) AS degree"
        )
        records = self._run(query, node=node)
        return int(records[0]["degree"]) if records else 0

    def community_seed_norm(self, community_id: str, seeds: List[NodeId]) -> List[NodeId]:
        """Passthrough: Neo4j node IDs are used as-is."""
        return seeds

    def get_node(self, node_id: NodeId, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return a node's properties (optionally restricted to fields)."""
        query = (
            f"MATCH (n{self._label_fragment()} {{`{self.id_property}`: $node}}) "
            f"RETURN properties(n) AS props LIMIT 1"
        )
        records = self._run(query, node=node_id)
        if not records:
            return {}
        props = dict(records[0]["props"])
        if fields:
            props = {k: v for k, v in props.items() if k in fields}
        return props

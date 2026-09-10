"""
Adapter conformance tests: shared contract every GraphAccessor implementation
must satisfy, run against the in-memory accessors and the experimental
Neo4jGraphAccessor (with a fake driver — no live database).
"""

from collections import namedtuple

import pytest

from retrieval.adapters import KGCommunityAccessor, OverlayAccessor
from retrieval.adapters_neo4j import Neo4jGraphAccessor


# --- Fake Neo4j driver ----------------------------------------------------

EagerResult = namedtuple("EagerResult", ["records", "summary", "keys"])

EDGES = [
    # (src, rel, dst, weight)
    ("A", "KNOWS", "B", 0.9),
    ("A", "WORKS_AT", "C", None),  # no weight property -> default 1.0
    ("B", "KNOWS", "C", 0.5),
]

NODE_PROPS = {
    "A": {"id": "A", "communityId": "c1", "kind": "person"},
    "B": {"id": "B", "communityId": "c1", "kind": "person"},
    "C": {"id": "C", "communityId": "c2", "kind": "org"},
}


class _FakeDriver:
    """Answers the exact Cypher shapes Neo4jGraphAccessor issues."""

    def __init__(self, edges=EDGES, node_props=NODE_PROPS):
        self.edges = edges
        self.node_props = node_props
        self.queries = []

    def execute_query(self, query, params, database_=None):
        self.queries.append((query, dict(params)))
        if "-[r]->" in query and "count(r)" in query:
            node = params["node"]
            count = sum(1 for s, _, _, _ in self.edges if s == node)
            return EagerResult([{"degree": count}], None, ["degree"])
        if "-[r]->" in query:
            node = params["node"]
            records = [
                {"neighbor": d, "rel": r, "weight": w if w is not None else 1.0}
                for s, r, d, w in self.edges
                if s == node
            ]
            return EagerResult(records, None, ["neighbor", "rel", "weight"])
        if "<-[r]-" in query:
            node = params["node"]
            records = [
                {"neighbor": s, "rel": r, "weight": w if w is not None else 1.0}
                for s, r, d, w in self.edges
                if d == node
            ]
            return EagerResult(records, None, ["neighbor", "rel", "weight"])
        if "properties(n)" in query:
            props = self.node_props.get(params["node"])
            records = [{"props": props}] if props else []
            return EagerResult(records, None, ["props"])
        if "WHERE n." in query:
            cid = params["cid"]
            records = [
                {"id": nid}
                for nid, props in self.node_props.items()
                if props.get("communityId") == cid
            ]
            return EagerResult(records, None, ["id"])
        # all nodes
        records = [{"id": nid} for nid in self.node_props]
        return EagerResult(records, None, ["id"])


# --- In-memory KG fixture for the generic accessors -----------------------


def _tiny_kg():
    from npll.core import load_knowledge_graph_from_triples

    return load_knowledge_graph_from_triples(
        [("A", "knows", "B"), ("A", "works_at", "C"), ("B", "knows", "C")],
        "ConformanceKG",
    )


def make_kg_accessor():
    return KGCommunityAccessor(_tiny_kg(), community_id="c1")


def make_overlay_accessor():
    return OverlayAccessor(make_kg_accessor(), community_id="c1")


def make_neo4j_accessor():
    return Neo4jGraphAccessor(_FakeDriver())


ACCESSOR_FACTORIES = {
    "kg": make_kg_accessor,
    "overlay": make_overlay_accessor,
    "neo4j": make_neo4j_accessor,
}


@pytest.fixture(params=sorted(ACCESSOR_FACTORIES))
def accessor(request):
    return ACCESSOR_FACTORIES[request.param]()


# --- Shared contract -------------------------------------------------------


class TestGraphAccessorContract:
    def test_iter_out_yields_neighbor_rel_weight_triples(self, accessor):
        for neighbor, rel, weight in accessor.iter_out("A"):
            assert isinstance(neighbor, str)
            assert isinstance(rel, str)
            assert isinstance(weight, float)
            assert weight > 0

    def test_iter_out_of_unknown_node_is_empty(self, accessor):
        assert list(accessor.iter_out("does-not-exist")) == []

    def test_degree_matches_iter_out(self, accessor):
        for node in ("A", "B", "C"):
            assert accessor.degree(node) == len(list(accessor.iter_out(node)))

    def test_community_seed_norm_defaults_to_passthrough(self, accessor):
        seeds = ["A", "B"]
        norm = getattr(accessor, "community_seed_norm", lambda cid, s: s)
        assert norm("c1", seeds) == seeds

    def test_nodes_returns_ids(self, accessor):
        node_ids = list(accessor.nodes("c1"))
        assert node_ids
        assert all(isinstance(n, str) for n in node_ids)


# --- Neo4j-specific behavior ----------------------------------------------


class TestNeo4jGraphAccessor:
    def test_iter_out_reads_weight_with_default(self):
        accessor = make_neo4j_accessor()
        edges = {(v, r): w for v, r, w in accessor.iter_out("A")}
        assert edges[("B", "KNOWS")] == pytest.approx(0.9)
        assert edges[("C", "WORKS_AT")] == pytest.approx(1.0)

    def test_iter_in_reverses_direction(self):
        accessor = make_neo4j_accessor()
        incoming = list(accessor.iter_in("C"))
        assert ("A", "WORKS_AT", 1.0) in incoming
        assert ("B", "KNOWS", 0.5) in incoming

    def test_nodes_scoped_by_community_property(self):
        accessor = make_neo4j_accessor()
        assert sorted(accessor.nodes("c1")) == ["A", "B"]
        assert sorted(accessor.nodes()) == ["A", "B", "C"]

    def test_nodes_unscoped_when_community_property_disabled(self):
        accessor = Neo4jGraphAccessor(_FakeDriver(), community_property=None)
        assert sorted(accessor.nodes("c1")) == ["A", "B", "C"]

    def test_get_node_returns_properties(self):
        accessor = make_neo4j_accessor()
        assert accessor.get_node("A") == NODE_PROPS["A"]
        assert accessor.get_node("A", fields=["kind"]) == {"kind": "person"}
        assert accessor.get_node("missing") == {}

    def test_node_label_scopes_queries(self):
        driver = _FakeDriver()
        accessor = Neo4jGraphAccessor(driver, node_label="Entity")
        list(accessor.iter_out("A"))
        query, _ = driver.queries[-1]
        assert ":`Entity`" in query

    def test_queries_are_parameterized_not_interpolated(self):
        driver = _FakeDriver()
        accessor = Neo4jGraphAccessor(driver)
        node = "A'}) MATCH (x) DETACH DELETE x //"
        list(accessor.iter_out(node))
        query, params = driver.queries[-1]
        assert node not in query
        assert params["node"] == node

    def test_works_behind_cached_accessor(self):
        from retrieval.cache import CachedGraphAccessor

        cached = CachedGraphAccessor(make_neo4j_accessor(), cache_size=10)
        first = list(cached.iter_out("A"))
        second = list(cached.iter_out("A"))
        assert first == second
        assert len(first) == 2

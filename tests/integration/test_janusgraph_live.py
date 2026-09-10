"""
Live JanusGraph integration test for JanusGraphAccessor (issue #6).

Requires a running JanusGraph/Gremlin Server; skipped unless JANUSGRAPH_URL
is set, e.g.:

    docker run -d --name janusgraph-test -p 8182:8182 janusgraph/janusgraph:latest
    JANUSGRAPH_URL=ws://localhost:8182/gremlin pytest tests/integration/test_janusgraph_live.py -v
"""

import os

import pytest

JANUSGRAPH_URL = os.environ.get("JANUSGRAPH_URL")

pytestmark = pytest.mark.skipif(
    not JANUSGRAPH_URL, reason="JANUSGRAPH_URL not set; live JanusGraph required"
)


@pytest.fixture(scope="module")
def g():
    from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
    from gremlin_python.driver.serializer import GraphSONSerializersV3d0
    from gremlin_python.process.anonymous_traversal import traversal

    # GraphSON v3: gremlinpython 3.8 defaults to GraphBinary, which JanusGraph
    # 1.x (TinkerPop 3.7) servers reject with serialization KeyErrors.
    conn = DriverRemoteConnection(
        JANUSGRAPH_URL, "g", message_serializer=GraphSONSerializersV3d0()
    )
    g = traversal().with_remote(conn)
    yield g
    conn.close()


@pytest.fixture(scope="module")
def seeded_graph(g):
    """Seed A -knows(0.9)-> B -knows(0.5)-> C, A -works_at-> C (no weight).

    Uses to_list()/next() terminals instead of iterate(): gremlinpython 3.8's
    iterate() emits a discard() step that TinkerPop 3.7 servers (JanusGraph
    1.x) do not implement.
    """
    g.V().has("testTag", "odin-adapter-test").drop().to_list()
    a = (g.add_v("entity").property("nodeId", "A").property("testTag", "odin-adapter-test")
         .property("communityId", "c1").next())
    b = (g.add_v("entity").property("nodeId", "B").property("testTag", "odin-adapter-test")
         .property("communityId", "c1").next())
    c = (g.add_v("entity").property("nodeId", "C").property("testTag", "odin-adapter-test")
         .property("communityId", "c2").next())
    g.V(a).add_e("knows").to(b).property("weight", 0.9).next()
    g.V(b).add_e("knows").to(c).property("weight", 0.5).next()
    g.V(a).add_e("works_at").to(c).next()
    ids = {"A": a.id, "B": b.id, "C": c.id}
    yield ids
    g.V().has("testTag", "odin-adapter-test").drop().to_list()


@pytest.fixture
def accessor(g):
    from retrieval.adapters import JanusGraphAccessor

    return JanusGraphAccessor(g)


class TestJanusGraphLive:
    def test_iter_out(self, accessor, seeded_graph):
        edges = list(accessor.iter_out(seeded_graph["A"]))
        by_rel = {rel: (nid, w) for nid, rel, w in edges}
        assert by_rel["knows"] == (seeded_graph["B"], pytest.approx(0.9))
        assert by_rel["works_at"] == (seeded_graph["C"], pytest.approx(1.0))

    def test_iter_in(self, accessor, seeded_graph):
        incoming = list(accessor.iter_in(seeded_graph["C"]))
        rels = {rel for _, rel, _ in incoming}
        assert rels == {"knows", "works_at"}

    def test_degree(self, accessor, seeded_graph):
        assert accessor.degree(seeded_graph["A"]) == 2
        assert accessor.in_degree(seeded_graph["C"]) == 2

    def test_get_node(self, accessor, seeded_graph):
        props = accessor.get_node(seeded_graph["A"])
        assert props["nodeId"] == "A"
        assert props["communityId"] == "c1"

    def test_retrieval_pipeline_end_to_end(self, accessor, seeded_graph):
        """The whole point: PPR + beam + scoring over live JanusGraph."""
        from retrieval.cache import CachedGraphAccessor
        from retrieval.confidence import ConstantConfidence
        from retrieval.orchestrator import OrchestratorParams, RetrievalOrchestrator

        cached = CachedGraphAccessor(accessor, cache_size=100)
        orchestrator = RetrievalOrchestrator(
            accessor=cached, edge_confidence=ConstantConfidence(0.8)
        )
        result = orchestrator.retrieve(
            seeds=[seeded_graph["A"]],
            params=OrchestratorParams(community_id="c1", max_paths=10, hop_limit=2),
        )
        assert result["paths"], "expected at least one path from seed A"

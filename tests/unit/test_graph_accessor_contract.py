"""Regression tests for the complete GraphAccessor surface."""

from npll.core.knowledge_graph import KnowledgeGraph
from retrieval.adapters import KGCommunityAccessor, OverlayAccessor
from retrieval.adapters_arango import ArangoCommunityAccessor, GlobalGraphAccessor
from tests.utils.backend_fakes import FakeArango


def _knowledge_graph():
    graph = KnowledgeGraph()
    graph.add_known_fact("A", "linked_to", "B")
    return graph


def test_in_memory_accessor_implements_inbound_nodes_and_seed_contract():
    accessor = KGCommunityAccessor(_knowledge_graph(), "global", {"A", "B"})

    assert list(accessor.iter_in("B")) == [("A", "linked_to", 1.0)]
    assert accessor.get_node("A") == {"id": "A"}
    assert accessor.get_node("A", fields=["missing"]) == {}
    assert accessor.get_node("missing") == {}
    assert accessor.community_seed_norm("global", ["A"]) == ["A"]


def test_overlay_exposes_inbound_overlay_edges_and_node_properties():
    base = KGCommunityAccessor(_knowledge_graph(), "global", {"A", "B"})
    accessor = OverlayAccessor(base, "global")
    accessor.add_edge("B", "annotates", "A", 0.6)

    assert ("B", "annotates", 0.6) in list(accessor.iter_in("A"))
    assert accessor.get_node("A") == {"id": "A"}


def test_global_accessor_declares_the_complete_graph_accessor_surface():
    accessor = GlobalGraphAccessor(db=object())

    for method in (
        "iter_out", "iter_in", "nodes", "degree", "get_node", "community_seed_norm",
    ):
        assert callable(getattr(accessor, method))


def test_arango_accessors_return_an_empty_dict_for_absent_nodes():
    db = FakeArango()
    queries = []

    def absent_node(query, **kwargs):
        queries.append(query)
        return iter([None])

    db.aql.execute = absent_node
    accessors = (
        ArangoCommunityAccessor(db, community_id="global", community_mode="none"),
        GlobalGraphAccessor(db),
    )

    for accessor in accessors:
        assert accessor.get_node("ExtractedEntities/missing") == {}
        assert accessor.get_node("ExtractedEntities/missing", fields=["type"]) == {}

    assert all("FILTER d != null" in query for query in queries[1::2])

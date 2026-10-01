"""Regression tests for the complete GraphAccessor surface."""

import pytest

from npll.core.knowledge_graph import KnowledgeGraph
from retrieval.adapters import KGCommunityAccessor, OverlayAccessor
from retrieval.adapters_arango import ArangoCommunityAccessor, GlobalGraphAccessor
from tests.utils.backend_fakes import FakeArango


ARANGO_ACCESSOR_KWARGS = {
    "nodes_collection": "Records",
    "edges_collection": "Links",
    "relation_property": "predicate",
}
GLOBAL_ACCESSOR_KWARGS = {
    **ARANGO_ACCESSOR_KWARGS,
    "algorithm": "leiden",
    "bridge_entity_field": "record_id",
    "bridge_strength_field": "strength",
    "bridge_community_field": "group_id",
    "bridge_algorithm_field": "method",
    "affinity_from_field": "left",
    "affinity_to_field": "right",
    "affinity_score_field": "score",
    "affinity_algorithm_field": "method",
    "bridge_collection": "Bridges",
    "affinity_collection": "Affinities",
    "membership_collection": "Memberships",
    "membership_entity_field": "record_id",
    "membership_community_field": "group_id",
}


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
    accessor = GlobalGraphAccessor(db=object(), **GLOBAL_ACCESSOR_KWARGS)

    for method in (
        "iter_out", "iter_in", "nodes", "degree", "get_node", "community_seed_norm",
    ):
        assert callable(getattr(accessor, method))


def test_direct_arango_accessors_require_explicit_schema_mappings():
    with pytest.raises(TypeError):
        ArangoCommunityAccessor(FakeArango(), community_id="global")
    with pytest.raises(TypeError):
        GlobalGraphAccessor(FakeArango())


def test_static_arango_helpers_require_their_collection_mappings():
    with pytest.raises(TypeError):
        ArangoCommunityAccessor.get_top_n_entities_by_degree(FakeArango())
    with pytest.raises(TypeError):
        ArangoCommunityAccessor.get_document_content(FakeArango(), "Sources/a")
    with pytest.raises(TypeError):
        ArangoCommunityAccessor.search_content(FakeArango(), "claim")


def test_static_arango_helpers_query_only_supplied_schema_fields():
    db = FakeArango()
    queries = []

    def capture_query(aql, bind_vars=None):
        queries.append((aql, bind_vars or {}))
        return []

    db.aql.execute = capture_query

    ArangoCommunityAccessor.get_top_entities_in_community(
        db,
        "group-a",
        membership_collection="Memberships",
        membership_entity_field="record_id",
        membership_community_field="group_id",
        edges_collection="Links",
    )
    ArangoCommunityAccessor.get_recent_entities(
        db,
        "2026-01-01T00:00:00Z",
        nodes_collection="Records",
        created_at_property="entered_at",
        updated_at_property="changed_at",
    )
    ArangoCommunityAccessor.search_entities(
        db,
        "claim",
        nodes_collection="Records",
        search_fields=["title", "details"],
    )
    ArangoCommunityAccessor.get_document_content(
        db,
        "EvidenceText/one",
        text_collection="EvidenceText",
        table_collection="EvidenceTables",
        image_collection="EvidenceImages",
        document_collection="EvidenceDocuments",
    )
    ArangoCommunityAccessor.get_entity_sources(
        db,
        "Records/one",
        extracted_from_collection="RecordSources",
    )
    ArangoCommunityAccessor.search_content(
        db,
        "claim",
        text_collection="EvidenceText",
        table_collection="EvidenceTables",
        image_collection="EvidenceImages",
        text_search_fields=["body"],
        table_search_fields=["heading"],
        image_search_fields=["transcript"],
    )

    generated_aql = "\n".join(aql for aql, _ in queries)
    generated_bind_values = [bind for _, bind in queries]
    assert "ExtractedEntities" not in generated_aql
    assert "ExtractedRelationships" not in generated_aql
    assert "TextBlocks" not in generated_aql
    assert "EXTRACTED_FROM" not in generated_aql
    assert ".name" not in generated_aql
    assert ".text" not in generated_aql
    assert any(bind.get("search_fields") == ["title", "details"] for bind in generated_bind_values)
    assert any(bind.get("text_search_fields") == ["body"] for bind in generated_bind_values)


def test_arango_accessors_return_an_empty_dict_for_absent_nodes():
    db = FakeArango()
    queries = []

    def absent_node(query, **kwargs):
        queries.append(query)
        # AQL FILTER removes the row; RETURN DOCUMENT alone yields null.
        return iter([] if "FILTER d != null" in query else [None])

    db.aql.execute = absent_node
    accessors = (
        ArangoCommunityAccessor(
            db,
            community_id="global",
            community_mode="none",
            **ARANGO_ACCESSOR_KWARGS,
        ),
        GlobalGraphAccessor(db, **GLOBAL_ACCESSOR_KWARGS),
    )

    for accessor in accessors:
        assert accessor.community_seed_norm("global", ["Records/A"]) == ["Records/A"]
        assert accessor.get_node("Records/missing") == {}
        assert accessor.get_node("Records/missing", fields=["kind"]) == {}

    assert all("FILTER d != null" in query for query in queries[1::2])

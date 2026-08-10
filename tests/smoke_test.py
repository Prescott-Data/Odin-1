from retrieval import RetrievalOrchestrator, OrchestratorParams, KGCommunityAccessor, ConstantConfidence
from npll.core.knowledge_graph import KnowledgeGraph


def build_tiny_kg():
    kg = KnowledgeGraph()
    kg.add_known_fact("A", "rel", "B")
    kg.add_known_fact("B", "rel", "C")
    kg.add_known_fact("C", "rel", "D")
    return kg


def test_smoke():
    kg = build_tiny_kg()
    A = KGCommunityAccessor(kg, community_id="c1", community_nodes={e.name for e in kg.entities})
    orc = RetrievalOrchestrator(accessor=A, edge_confidence=ConstantConfidence(0.8))
    res = orc.retrieve(seeds=["A"], params=OrchestratorParams(community_id="c1"))
    assert "insight_score" in res
    assert isinstance(res["topk_ppr"], list)

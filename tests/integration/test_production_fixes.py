"""
Integration tests for production fixes.
Tests the complete retrieval pipeline with all new features.
"""

import pytest
from retrieval import (
    RetrievalOrchestrator,
    OrchestratorParams,
    KGCommunityAccessor,
    CachedGraphAccessor,
    NPLLConfidence,
    ConstantConfidence,
)
from npll.core.knowledge_graph import KnowledgeGraph


def build_test_kg():
    """Create a small test knowledge graph."""
    kg = KnowledgeGraph()
    # Create a small graph with hub structure
    kg.add_known_fact("Hub", "connects_to", "A")
    kg.add_known_fact("Hub", "connects_to", "B")
    kg.add_known_fact("Hub", "links_to", "C")
    kg.add_known_fact("A", "relates_to", "B")
    kg.add_known_fact("B", "rel", "C")
    kg.add_known_fact("C", "rel", "D")
    return kg


class TestCachedAccessorIntegration:
    """Integration tests with CachedGraphAccessor."""
    
    def test_retrieval_with_cache(self):
        """Test that CachedGraphAccessor works in full retrieval."""
        from retrieval import (
            KGCommunityAccessor,
            CachedGraphAccessor,
            RetrievalOrchestrator,
            OrchestratorParams,
            ConstantConfidence,
        )
        from npll.core.knowledge_graph import KnowledgeGraph
        
        # Build small KG
        kg = KnowledgeGraph()
        for i in range(10):
            kg.add_known_fact(f"A{i}", "rel", f"B{i}")
            kg.add_known_fact(f"B{i}", "rel", f"C{i}")
        
        # Base accessor
        base = KGCommunityAccessor(
            kg, 
            community_id="test", 
            community_nodes={e.name for e in kg.entities}
        )
        
        # Wrap with cache
        from retrieval.cache import CachedGraphAccessor
        cached = CachedGraphAccessor(base, cache_size=100)
        
        # Run retrieval
        from retrieval import RetrievalOrchestrator, OrchestratorParams, ConstantConfidence
        orc = RetrievalOrchestrator(accessor=cached, edge_confidence=ConstantConfidence(0.8))
        result = orc.retrieve(
            seeds=["A0"],
            params=OrchestratorParams(community_id="test", hop_limit=2)
        )
        
        # Verify cache was used
        stats = cached.cache_stats()
        assert stats["hits"] > 0  # Should have some cache hits during PPR
        assert stats["hit_rate"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

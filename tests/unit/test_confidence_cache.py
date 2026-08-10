"""
Unit tests for NPLLConfidence LRU cache.
CRITICAL: Memory leak prevention for 24/7 production runs.
"""

import pytest
import torch
from retrieval.confidence import NPLLConfidence


class MockNPLLModel:
    """Mock NPLL model for testing."""
    def __init__(self):
        self.scoring_module = self
        self.call_count = 0
    
    def eval(self):
        pass
    
    def forward_with_names(self, heads, rels, tails):
        """Return mock scores."""
        self.call_count += 1
        return torch.tensor([0.8] * len(heads))
    
    def probability_transform(self, scores, apply_temperature=True, group_ids=None):
        """Return mock probabilities."""
        return scores


class TestNPLLConfidenceCache:
    """Test suite for NPLLConfidence LRU cache."""
    
    def test_cache_prevents_redundant_inference(self):
        """Verify cache prevents redundant NPLL inference calls."""
        model = MockNPLLModel()
        confidence = NPLLConfidence(model, cache_size=1000)
        
        # First call - cache miss
        score1 = confidence.confidence("A", "rel", "B")
        assert model.call_count == 1
        assert score1 == pytest.approx(0.8, rel=1e-5)
        
        # Second call - cache hit
        score2 = confidence.confidence("A", "rel", "B")
        assert model.call_count == 1  # No new inference
        assert score2 == pytest.approx(0.8, rel=1e-5)
    
    def test_batch_caching(self):
        """Verify batch scoring with caching."""
        model = MockNPLLModel()
        confidence = NPLLConfidence(model, cache_size=1000)
        
        edges = [
            ("A", "rel", "B"),
            ("B", "rel", "C"),
            ("C", "rel", "D"),
        ]
        
        # First batch - all cache misses
        scores1 = confidence.confidence_batch(edges)
        assert model.call_count == 1
        assert len(scores1) == 3
        
        # Second batch with overlap - partial cache hits
        edges2 = [
            ("A", "rel", "B"),  # Cached
            ("D", "rel", "E"),  # New
        ]
        scores2 = confidence.confidence_batch(edges2)
        assert model.call_count == 2  # Only 1 new inference
        assert len(scores2) == 2
    
    def test_lru_eviction_at_capacity(self):
        """Verify LRU eviction when cache reaches capacity."""
        model = MockNPLLModel()
        confidence = NPLLConfidence(model, cache_size=3)  # Small cache
        
        # Fill cache
        confidence.confidence("A", "rel", "B")  # 1
        confidence.confidence("C", "rel", "D")  # 2
        confidence.confidence("E", "rel", "F")  # 3
        assert model.call_count == 3
        
        # Cache is full. Add one more - should evict oldest (A,rel,B)
        confidence.confidence("G", "rel", "H")  # 4
        assert model.call_count == 4
        
        # Access A,rel,B again - should be cache miss (evicted)
        confidence.confidence("A", "rel", "B")
        assert model.call_count == 5
    
    def test_cache_stats(self):
        """Verify cache statistics."""
        model = MockNPLLModel()
        confidence = NPLLConfidence(model, cache_size=100)
        
        stats = confidence.cache_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 100
        assert stats["utilization"] == 0
        
        # Add 10 entries
        for i in range(10):
            confidence.confidence(f"A{i}", "rel", f"B{i}")
        
        stats = confidence.cache_stats()
        assert stats["size"] == 10
        assert stats["utilization"] == 0.1
    
    def test_clear_cache(self):
        """Verify cache clearing."""
        model = MockNPLLModel()
        confidence = NPLLConfidence(model, cache_size=100)
        
        confidence.confidence("A", "rel", "B")
        assert model.call_count == 1
        
        # Should be cached
        confidence.confidence("A", "rel", "B")
        assert model.call_count == 1
        
        # Clear cache
        confidence.clear_cache()
        assert confidence.cache_stats()["size"] == 0
        
        # Should be cache miss now
        confidence.confidence("A", "rel", "B")
        assert model.call_count == 2
    
    def test_memory_bounded_growth(self):
        """
        CRITICAL: Verify cache doesn't grow unbounded.
        This test simulates 24/7 operation.
        """
        model = MockNPLLModel()
        cache_size = 1000
        confidence = NPLLConfidence(model, cache_size=cache_size)
        
        # Simulate 10K unique inferences (10x cache size)
        for i in range(10000):
            confidence.confidence(f"A{i}", "rel", f"B{i}")
        
        # Cache should NOT exceed max size
        stats = confidence.cache_stats()
        assert stats["size"] <= cache_size
        assert stats["size"] == cache_size  # Should be at capacity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

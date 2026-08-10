"""
Unit tests for CachedGraphAccessor.
CRITICAL: This cache prevents network hammering in production.
"""

import pytest
from retrieval.cache import CachedGraphAccessor
from retrieval.adapters import GraphAccessor


class MockAccessor:
    """Mock accessor that tracks calls."""
    def __init__(self):
        self.out_calls = 0
        self.in_calls = 0
        
    def iter_out(self, node):
        self.out_calls += 1
        # Return mock neighbors
        return iter([
            (f"{node}_neighbor_1", "rel1", 0.9),
            (f"{node}_neighbor_2", "rel2", 0.8),
        ])
    
    def iter_in(self, node):
        self.in_calls += 1
        return iter([
            (f"{node}_parent_1", "rel1", 0.9),
        ])
    
    def nodes(self, community_id=None):
        return iter(["A", "B", "C"])


class TestCachedGraphAccessor:
    """Test suite for CachedGraphAccessor."""
    
    def test_cache_hit_reduces_calls(self):
        """Verify cache prevents redundant accessor calls."""
        base = MockAccessor()
        cached = CachedGraphAccessor(base, cache_size=100)
        
        # First call - cache miss
        list(cached.iter_out("A"))
        assert base.out_calls == 1
        
        # Second call - cache hit
        list(cached.iter_out("A"))
        assert base.out_calls == 1  # Should still be 1 (cached)
        
        # Third call - cache hit
        list(cached.iter_out("A"))
        assert base.out_calls == 1  # Still cached
    
    def test_cache_miss_for_different_nodes(self):
        """Different nodes should cause cache misses."""
        base = MockAccessor()
        cached = CachedGraphAccessor(base, cache_size=100)
        
        list(cached.iter_out("A"))
        list(cached.iter_out("B"))
        list(cached.iter_out("C"))
        
        assert base.out_calls == 3  # 3 different nodes
    
    def test_lru_eviction(self):
        """Verify LRU eviction when cache is full."""
        base = MockAccessor()
        cached = CachedGraphAccessor(base, cache_size=2)  # Small cache
        
        # Fill cache
        list(cached.iter_out("A"))  # 1st - in cache
        list(cached.iter_out("B"))  # 2nd - in cache
        
        # Access A again (moves to end of LRU)
        list(cached.iter_out("A"))  # Still cached
        
        # Add C - should evict B (least recently used)
        list(cached.iter_out("C"))  # 3rd call
        
        assert base.out_calls == 3
        
        # Access B again - should be cache miss (was evicted)
        list(cached.iter_out("B"))  # 4th call - evicts A
        
        assert base.out_calls == 4
        
        # Now cache has {C, B}
        # C should be cached, A should not
        list(cached.iter_out("C"))  # Cached
        assert base.out_calls == 4  # No new call
        
        list(cached.iter_out("A"))  # Cache miss - was evicted in previous step
        assert base.out_calls == 5
    
    def test_cache_stats(self):
        """Verify cache statistics are accurate."""
        base = MockAccessor()
        cached = CachedGraphAccessor(base, cache_size=10)
        
        # Initial state
        stats = cached.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0
        
        # Cache miss
        list(cached.iter_out("A"))
        stats = cached.cache_stats()
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0
        
        # Cache hit
        list(cached.iter_out("A"))
        stats = cached.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        
        # Another hit
        list(cached.iter_out("A"))
        stats = cached.cache_stats()
        assert stats["hits"] == 2
        assert stats["hit_rate"] == 2/3
    
    def test_separate_in_out_caches(self):
        """Verify inbound and outbound have separate caches."""
        base = MockAccessor()
        cached = CachedGraphAccessor(base, cache_size=10)
        
        list(cached.iter_out("A"))
        list(cached.iter_in("A"))
        
        assert base.out_calls == 1
        assert base.in_calls == 1
        
        # Both should be cached now
        list(cached.iter_out("A"))
        list(cached.iter_in("A"))
        
        assert base.out_calls == 1
        assert base.in_calls == 1
    
    def test_clear_cache(self):
        """Verify cache clearing works."""
        base = MockAccessor()
        cached = CachedGraphAccessor(base, cache_size=10)
        
        list(cached.iter_out("A"))
        assert base.out_calls == 1
        
        # Should be cached
        list(cached.iter_out("A"))
        assert base.out_calls == 1
        
        # Clear cache
        cached.clear_cache()
        
        # Should be cache miss now
        list(cached.iter_out("A"))
        assert base.out_calls == 2
    
    def test_warm_cache(self):
        """Verify cache warming works."""
        base = MockAccessor()
        cached = CachedGraphAccessor(base, cache_size=10)
        
        # Warm cache for nodes
        cached.warm_cache(["A", "B", "C"], direction="out")
        assert base.out_calls == 3
        
        # All should be cached now
        list(cached.iter_out("A"))
        list(cached.iter_out("B"))
        list(cached.iter_out("C"))
        assert base.out_calls == 3  # No additional calls


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

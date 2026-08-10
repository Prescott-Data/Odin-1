"""
Performance and regression tests for production Odin.
CRITICAL: These tests catch performance regressions and memory leaks.
"""

import pytest
import time
import psutil
import os
from retrieval.cache import CachedGraphAccessor
from retrieval.confidence import NPLLConfidence
from retrieval.adapters import GraphAccessor


class MockAccessor:
    """Mock accessor with configurable latency."""
    def __init__(self, latency_ms=0):
        self.latency_ms = latency_ms
    
    def iter_out(self, node):
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000)
        return iter([
            (f"{node}_n1", "rel", 0.9),
            (f"{node}_n2", "rel", 0.8),
        ])
    
    def iter_in(self, node):
        return iter([])
    
    def nodes(self, community_id=None):
        return iter([f"node_{i}" for i in range(100)])


class MockNPLLModel:
    """Mock NPLL model."""
    def __init__(self):
        self.scoring_module = self
    
    def eval(self):
        pass
    
    def forward_with_names(self, heads, rels, tails):
        import torch
        return torch.tensor([0.8] * len(heads))
    
    def probability_transform(self, scores, apply_temperature=True, group_ids=None):
        return scores


@pytest.mark.performance
class TestCachePerformance:
    """Performance tests for CachedGraphAccessor."""
    
    def test_cache_speedup(self):
        """
        Verify cache provides significant speedup.
        Should be at least 10x faster with cache for repeated access.
        """
        # Slow accessor (10ms per call)
        slow_accessor = MockAccessor(latency_ms=10)
        cached = CachedGraphAccessor(slow_accessor, cache_size=100)
        
        # First pass - fill cache (slow)
        start = time.time()
        for _ in range(100):
            list(cached.iter_out("A"))
        uncached_time = time.time() - start
        
        # Clear and re-run without cache
        slow_accessor_direct = MockAccessor(latency_ms=10)
        start = time.time()
        for _ in range(100):
            list(slow_accessor_direct.iter_out("A"))
        direct_time = time.time() - start
        
        # Cached should be much faster (only 1 real call)
        speedup = direct_time / uncached_time
        print(f"\nSpeedup: {speedup:.1f}x")
        assert speedup > 10, f"Cache speedup too low: {speedup:.1f}x (expected >10x)"
    
    def test_cache_hit_rate_on_ppr_pattern(self):
        """
        Simulate PPR access pattern (repeated access to same nodes).
        Should achieve >80% hit rate.
        """
        accessor = MockAccessor()
        cached = CachedGraphAccessor(accessor, cache_size=100)
        
        # Simulate PPR: repeatedly access same subset of nodes
        nodes = [f"node_{i}" for i in range(20)]
        for _ in range(10):  # 10 iterations
            for node in nodes:
                list(cached.iter_out(node))
        
        stats = cached.cache_stats()
        hit_rate = stats["hit_rate"]
        print(f"\nHit rate: {hit_rate:.2%}")
        
        # After first pass, should have >80% hit rate
        assert hit_rate > 0.8, f"Cache hit rate too low: {hit_rate:.2%}"


@pytest.mark.performance
class TestMemoryLeaks:
    """Memory leak tests - CRITICAL for 24/7 operation."""
    
    def test_npll_confidence_no_memory_leak(self):
        """
        CRITICAL: Verify NPLLConfidence doesn't leak memory.
        Simulates 24-hour operation with 100K inferences.
        """
        import torch
        
        model = MockNPLLModel()
        confidence = NPLLConfidence(model, cache_size=1000)
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate heavy usage: 100K unique inferences
        for i in range(100_000):
            confidence.confidence(f"A{i}", "rel", f"B{i}")
            
            # Check memory every 10K
            if i % 10_000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                print(f"\nAfter {i} inferences: {memory_growth:.1f}MB growth")
        
        # Final memory check
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        print(f"\nTotal memory growth: {total_growth:.1f}MB")
        
        # Memory should be bounded (not growing linearly)
        # With 1K cache, expect <50MB growth
        assert total_growth < 50, f"Memory leak detected: {total_growth:.1f}MB growth"
        
        # Verify cache size is bounded
        stats = confidence.cache_stats()
        assert stats["size"] <= 1000
    
    def test_cache_accessor_no_memory_leak(self):
        """Verify CachedGraphAccessor doesn't leak memory."""
        accessor = MockAccessor()
        cached = CachedGraphAccessor(accessor, cache_size=1000)
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Access 100K unique nodes (100x cache size)
        for i in range(100_000):
            list(cached.iter_out(f"node_{i}"))
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        print(f"\nMemory growth: {memory_growth:.1f}MB")
        
        # Cache should be bounded at 1000 entries
        stats = cached.cache_stats()
        assert stats["out_cache_size"] <= 1000
        
        # Memory growth should be bounded
        assert memory_growth < 50


@pytest.mark.benchmark
class TestLatencyBenchmarks:
    """Latency benchmarks - set baselines for monitoring."""
    
    def test_cached_retrieval_latency(self):
        """
        Benchmark retrieval latency with cache.
        This sets a baseline for monitoring regressions.
        """
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
        for i in range(50):
            kg.add_known_fact(f"A{i}", "rel", f"B{i}")
            kg.add_known_fact(f"B{i}", "rel", f"C{i}")
        
        base = KGCommunityAccessor(
            kg, 
            community_id="test",
            community_nodes={e.name for e in kg.entities}
        )
        cached = CachedGraphAccessor(base, cache_size=1000)
        
        orc = RetrievalOrchestrator(
            accessor=cached,
            edge_confidence=ConstantConfidence(0.8)
        )
        
        # Benchmark
        start = time.time()
        result = orc.retrieve(
            seeds=["A0"],
            params=OrchestratorParams(community_id="test", hop_limit=2)
        )
        latency = (time.time() - start) * 1000  # ms
        
        print(f"\nRetrieval latency: {latency:.1f}ms")
        
        # Set baseline: should complete in <1000ms for small graph
        assert latency < 1000, f"Latency too high: {latency:.1f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])

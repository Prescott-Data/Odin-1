"""
Caching layer for GraphAccessor to prevent network hammering during PPR.

PRODUCTION FIX: PPR's push algorithm repeatedly calls accessor.iter_out(u) for
the same nodes, causing excessive network traffic. CachedGraphAccessor wraps any
accessor and caches neighbor lookups.
"""

from __future__ import annotations
from typing import Iterable, Optional, List, Tuple
from collections import OrderedDict

from .adapters import GraphAccessor, NodeId, RelId


class CachedGraphAccessor:
    """
    Wraps a GraphAccessor with LRU caching for neighbor queries.
    
    Critical for production: Prevents "network hammer" issue where PPR
    makes repeated calls to iter_out() for the same nodes, each hitting
    the database/network.
    
    Usage:
        base_accessor = ArangoCommunityAccessor(db, community_id="insurance")
        cached_accessor = CachedGraphAccessor(base_accessor, cache_size=5000)
        
        # Now PPR won't hammer the network
        orchestrator = RetrievalOrchestrator(accessor=cached_accessor, ...)
    """
    
    def __init__(self, base: GraphAccessor, cache_size: int = 5000):
        """
        Args:
            base: The underlying GraphAccessor to wrap
            cache_size: Maximum number of nodes to cache (default: 5000)
        """
        self.base = base
        self.cache_size = cache_size
        
        # LRU caches for outbound and inbound neighbors
        self._out_cache: OrderedDict[NodeId, List[Tuple[NodeId, RelId, float]]] = OrderedDict()
        self._in_cache: OrderedDict[NodeId, List[Tuple[NodeId, RelId, float]]] = OrderedDict()
        
        # Stats for monitoring
        self._hits = 0
        self._misses = 0
    
    def iter_out(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Get outbound neighbors with caching."""
        if node in self._out_cache:
            # Cache hit - move to end (LRU)
            self._out_cache.move_to_end(node)
            self._hits += 1
            return iter(self._out_cache[node])
        
        # Cache miss - fetch from base accessor
        self._misses += 1
        neighbors = list(self.base.iter_out(node))
        
        # Store in cache with LRU eviction
        if len(self._out_cache) >= self.cache_size:
            self._out_cache.popitem(last=False)  # Remove oldest
        self._out_cache[node] = neighbors
        
        return iter(neighbors)
    
    def iter_in(self, node: NodeId) -> Iterable[Tuple[NodeId, RelId, float]]:
        """Get inbound neighbors with caching."""
        if node in self._in_cache:
            # Cache hit - move to end (LRU)
            self._in_cache.move_to_end(node)
            self._hits += 1
            return iter(self._in_cache[node])
        
        # Cache miss - fetch from base accessor
        self._misses += 1
        neighbors = list(self.base.iter_in(node))
        
        # Store in cache with LRU eviction
        if len(self._in_cache) >= self.cache_size:
            self._in_cache.popitem(last=False)  # Remove oldest
        self._in_cache[node] = neighbors
        
        return iter(neighbors)
    
    def nodes(self, community_id: Optional[str] = None) -> Iterable[NodeId]:
        """Pass through to base accessor (no caching)."""
        return self.base.nodes(community_id)
    
    def get_node(self, node_id: NodeId, fields: Optional[List[str]] = None) -> dict:
        """Pass through to base accessor (node lookups are typically one-off)."""
        return self.base.get_node(node_id, fields)
    
    def degree(self, node: NodeId) -> int:
        """Pass through to base accessor."""
        return self.base.degree(node)
    
    def community_seed_norm(self, community_id: str, seeds: List[str]) -> List[str]:
        """Pass through to base accessor."""
        return self.base.community_seed_norm(community_id, seeds)
    
    def clear_cache(self):
        """Clear all caches. Useful for memory management or testing."""
        self._out_cache.clear()
        self._in_cache.clear()
        self._hits = 0
        self._misses = 0
    
    def cache_stats(self) -> dict:
        """
        Return cache statistics for monitoring.
        
        Returns:
            Dict with hit rate, sizes, and utilization metrics
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total,
            "hit_rate": hit_rate,
            "out_cache_size": len(self._out_cache),
            "in_cache_size": len(self._in_cache),
            "max_cache_size": self.cache_size,
            "out_cache_utilization": len(self._out_cache) / self.cache_size,
            "in_cache_utilization": len(self._in_cache) / self.cache_size,
        }
    
    def warm_cache(self, nodes: List[NodeId], direction: str = "out"):
        """
        Pre-populate cache for a list of nodes.
        Useful for batch operations where you know which nodes will be accessed.
        
        Args:
            nodes: List of node IDs to pre-fetch
            direction: "out" or "in" for outbound/inbound neighbors
        """
        if direction == "out":
            for node in nodes:
                if node not in self._out_cache:
                    neighbors = list(self.base.iter_out(node))
                    if len(self._out_cache) >= self.cache_size:
                        self._out_cache.popitem(last=False)
                    self._out_cache[node] = neighbors
        elif direction == "in":
            for node in nodes:
                if node not in self._in_cache:
                    neighbors = list(self.base.iter_in(node))
                    if len(self._in_cache) >= self.cache_size:
                        self._in_cache.popitem(last=False)
                    self._in_cache[node] = neighbors
    
    # Delegate all other methods to base accessor
    def __getattr__(self, name):
        """Delegate unknown methods to the base accessor."""
        return getattr(self.base, name)

from __future__ import annotations
from typing import Protocol, Tuple, Optional, Callable, List
from collections import OrderedDict
from math import isfinite
import torch

NodeId = str
RelId = str


class EdgeConfidenceProvider(Protocol):
    def confidence(self, u: NodeId, rel: RelId, v: NodeId) -> float: ...
    def confidence_batch(self, edges: List[Tuple[NodeId, RelId, NodeId]]) -> List[float]: ...


class ConstantConfidence:
    def __init__(self, value: float = 0.8):
        self.value = max(1e-6, min(1.0, value))

    def confidence(self, u, rel, v) -> float:
        return self.value
    
    def confidence_batch(self, edges: List[Tuple[NodeId, RelId, NodeId]]) -> List[float]:
        return [self.value] * len(edges)


class NPLLConfidence:
    """
    Wraps an NPLL model for retrieval-time scoring with LRU caching.
    Uses the model's scoring module for batched probabilities when available.
    
    PRODUCTION FIX: Uses bounded LRU cache to prevent memory leaks in long-running processes.
    """

    def __init__(self, npll_model, cache_size: int = 10000):
        """
        Args:
            npll_model: Trained NPLL model for scoring triples
            cache_size: Maximum number of cached confidence scores (default: 10K)
        """
        self.model = npll_model
        self.cache_size = cache_size
        self._cache: OrderedDict[Tuple[str, str, str], float] = OrderedDict()

    def confidence(self, u: NodeId, rel: RelId, v: NodeId) -> float:
        return self.confidence_batch([(u, rel, v)])[0]

    def confidence_batch(self, edges: List[Tuple[NodeId, RelId, NodeId]]) -> List[float]:
        todo = [(u, r, v) for (u, r, v) in edges if (u, r, v) not in self._cache]
        if todo:
            heads, rels, tails = zip(*todo)
            self.model.eval()
            with torch.no_grad():
                scores = self.model.scoring_module.forward_with_names(list(heads), list(rels), list(tails))
                # Don't apply per-group temperature scaling (requires group_ids we don't have)
                probs = self.model.probability_transform(scores, apply_temperature=False)
            for (u, r, v), p in zip(todo, probs.tolist()):
                confidence = max(1e-6, float(p)) if isfinite(p) else 1e-6
                
                # LRU eviction: remove oldest if at capacity
                if len(self._cache) >= self.cache_size:
                    self._cache.popitem(last=False)  # Remove oldest (FIFO)
                
                self._cache[(u, r, v)] = confidence
        
        # Move accessed items to end (LRU behavior)
        result = []
        for edge in edges:
            conf = self._cache[edge]
            # Move to end to mark as recently used
            self._cache.move_to_end(edge)
            result.append(conf)
        
        return result

    def clear_cache(self):
        """Clear the confidence cache. Useful for testing or memory management."""
        self._cache.clear()

    def cache_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        return {
            "size": len(self._cache),
            "max_size": self.cache_size,
            "utilization": len(self._cache) / self.cache_size if self.cache_size > 0 else 0,
        }



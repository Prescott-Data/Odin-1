"""
Odin Benchmarks: Academic validation against standard KG datasets.

This module provides:
- Standard dataset loaders (FB15k-237, WN18RR)
- KG completion metrics (MRR, Hits@K)
- Benchmark runners for NPLL evaluation
- Ablation study tools
"""

from .metrics import mrr, hits_at_k, evaluate_rankings
from .datasets import load_fb15k237, load_wn18rr, BenchmarkDataset

__all__ = [
    "mrr", "hits_at_k", "evaluate_rankings",
    "load_fb15k237", "load_wn18rr", "BenchmarkDataset"
]

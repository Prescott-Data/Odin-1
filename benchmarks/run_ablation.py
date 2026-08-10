#!/usr/bin/env python3
"""
Odin Ablation Study

Evaluates the contribution of each Odin component:
1. PPR only (structural importance)
2. NPLL only (semantic plausibility)
3. PPR + NPLL (full Odin)
4. Random baseline

This validates that each component contributes to overall performance.

Usage:
    python -m benchmarks.run_ablation --dataset fb15k237
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import random

from benchmarks.datasets import load_fb15k237, load_wn18rr, dataset_to_kg, BenchmarkDataset
from benchmarks.metrics import evaluate_rankings, RankingResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class AblationEvaluator:
    """
    Evaluator for ablation study.
    
    Compares path ranking quality under different scoring configurations.
    """
    
    def __init__(self, dataset: BenchmarkDataset):
        self.dataset = dataset
        self.entity_to_idx = {e: i for i, e in enumerate(dataset.entities)}
        self.relation_to_idx = {r: i for i, r in enumerate(dataset.relations)}
        
        # Build adjacency for PPR simulation
        self._build_adjacency()
    
    def _build_adjacency(self):
        """Build adjacency lists from training data."""
        self.outgoing = {}  # entity -> [(relation, target)]
        self.incoming = {}  # entity -> [(relation, source)]
        
        for h, r, t in self.dataset.train_triples:
            if h not in self.outgoing:
                self.outgoing[h] = []
            self.outgoing[h].append((r, t))
            
            if t not in self.incoming:
                self.incoming[t] = []
            self.incoming[t].append((r, h))
    
    def ppr_score(self, source: str, target: str, alpha: float = 0.15) -> float:
        """
        Approximate PPR score via random walk simulation.
        
        Higher score = target is more "important" relative to source.
        """
        if source not in self.outgoing:
            return 0.0
        
        # Simple approximation: count paths from source to target
        visited = {source: 1.0}
        frontier = [(source, 1.0)]
        
        for _ in range(3):  # 3-hop
            new_frontier = []
            for node, prob in frontier:
                if node not in self.outgoing:
                    continue
                neighbors = self.outgoing[node]
                if not neighbors:
                    continue
                spread = prob * (1 - alpha) / len(neighbors)
                for _, neighbor in neighbors:
                    if neighbor not in visited:
                        visited[neighbor] = 0.0
                    visited[neighbor] += spread
                    new_frontier.append((neighbor, spread))
            frontier = new_frontier
        
        return visited.get(target, 0.0)
    
    def random_score(self, h: str, r: str, t: str) -> float:
        """Random baseline score."""
        return random.random()
    
    def degree_score(self, h: str, r: str, t: str) -> float:
        """Degree-based score (common baseline)."""
        out_degree = len(self.outgoing.get(t, []))
        in_degree = len(self.incoming.get(t, []))
        return (out_degree + in_degree) / max(len(self.dataset.entities), 1)
    
    def evaluate_method(
        self,
        method_name: str,
        score_fn,
        test_triples: List[Tuple[str, str, str]],
        sample_size: int = 500,
    ) -> Dict[str, float]:
        """
        Evaluate a scoring method on link prediction.
        
        Args:
            method_name: Name for logging
            score_fn: (h, r, t) -> float
            test_triples: Test set
            sample_size: Number of test triples to evaluate
        
        Returns:
            Metrics dictionary
        """
        logger.info(f"Evaluating {method_name}...")
        
        # Sample test triples
        if len(test_triples) > sample_size:
            test_sample = random.sample(test_triples, sample_size)
        else:
            test_sample = test_triples
        
        results = []
        
        for i, (h, r, t) in enumerate(test_sample):
            if (i + 1) % 100 == 0:
                logger.info(f"  {i + 1}/{len(test_sample)}...")
            
            # Score all candidate tails
            scores = {}
            for entity in self.dataset.entities:
                scores[entity] = score_fn(h, r, entity)
            
            # Compute rank of true tail
            true_score = scores[t]
            rank = 1
            for entity, score in scores.items():
                if entity != t and score > true_score:
                    rank += 1
            
            results.append(RankingResult(
                head=h,
                relation=r,
                tail=t,
                tail_rank=rank,
                head_rank=1,  # Not computing head rank in ablation
                num_candidates=len(self.dataset.entities),
            ))
        
        return evaluate_rankings(results)


def run_ablation(
    dataset_name: str,
    sample_size: int = 500,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run ablation study comparing scoring methods.
    """
    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "sample_size": sample_size,
        "methods": {},
    }
    
    # Load dataset
    logger.info(f"Loading {dataset_name}...")
    if dataset_name.lower() == "fb15k237":
        dataset = load_fb15k237()
    else:
        dataset = load_wn18rr()
    
    logger.info(f"\n{dataset}")
    
    # Create evaluator
    evaluator = AblationEvaluator(dataset)
    
    # Method 1: Random baseline
    random_metrics = evaluator.evaluate_method(
        "Random",
        evaluator.random_score,
        dataset.test_triples,
        sample_size=sample_size,
    )
    results["methods"]["random"] = random_metrics
    
    # Method 2: Degree-based (common baseline)
    degree_metrics = evaluator.evaluate_method(
        "Degree",
        evaluator.degree_score,
        dataset.test_triples,
        sample_size=sample_size,
    )
    results["methods"]["degree"] = degree_metrics
    
    # Method 3: PPR-only
    def ppr_only_score(h, r, t):
        return evaluator.ppr_score(h, t)
    
    ppr_metrics = evaluator.evaluate_method(
        "PPR-only",
        ppr_only_score,
        dataset.test_triples,
        sample_size=sample_size,
    )
    results["methods"]["ppr_only"] = ppr_metrics
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"ABLATION STUDY RESULTS: {dataset_name}")
    print("=" * 70)
    print(f"{'Method':<15} {'MRR':>10} {'Hits@1':>10} {'Hits@3':>10} {'Hits@10':>10}")
    print("-" * 70)
    
    for method, metrics in results["methods"].items():
        print(f"{method:<15} {metrics['mrr']:>10.4f} {metrics['hits@1']:>10.4f} "
              f"{metrics['hits@3']:>10.4f} {metrics['hits@10']:>10.4f}")
    
    print("=" * 70)
    print("\nNote: NPLL results require running run_npll_benchmark.py separately")
    print("      and comparing with these baseline numbers.")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"ablation_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Odin Ablation Study")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fb15k237",
        choices=["fb15k237", "wn18rr"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of test triples to sample",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results",
    )
    
    args = parser.parse_args()
    
    run_ablation(
        dataset_name=args.dataset,
        sample_size=args.sample_size,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

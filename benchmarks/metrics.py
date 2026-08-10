"""
Standard Knowledge Graph Completion Metrics

Implements metrics used in academic KG completion papers:
- Mean Reciprocal Rank (MRR)
- Hits@K (K=1, 3, 10)

These metrics evaluate link prediction quality:
Given (h, r, ?), rank all entities by predicted score.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class RankingResult:
    """Result of ranking evaluation for a single triple."""
    head: str
    relation: str
    tail: str
    tail_rank: int  # Rank of true tail among all candidates
    head_rank: int  # Rank of true head among all candidates (for inverse)
    num_candidates: int


def mrr(ranks: List[int]) -> float:
    """
    Mean Reciprocal Rank
    
    MRR = (1/|Q|) * Σ (1/rank_i)
    
    Args:
        ranks: List of ranks (1-indexed, where 1 is best)
    
    Returns:
        MRR score in [0, 1]
    """
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in ranks]))


def hits_at_k(ranks: List[int], k: int) -> float:
    """
    Hits@K - proportion of ranks <= K
    
    Args:
        ranks: List of ranks (1-indexed)
        k: Cutoff threshold
    
    Returns:
        Hits@K score in [0, 1]
    """
    if not ranks:
        return 0.0
    return float(np.mean([1 if r <= k else 0 for r in ranks]))


def evaluate_rankings(results: List[RankingResult]) -> Dict[str, float]:
    """
    Compute all standard metrics from ranking results.
    
    Args:
        results: List of RankingResult objects
    
    Returns:
        Dictionary with MRR, Hits@1, Hits@3, Hits@10
    """
    if not results:
        return {
            "mrr": 0.0,
            "hits@1": 0.0,
            "hits@3": 0.0,
            "hits@10": 0.0,
            "num_queries": 0
        }
    
    # Use tail ranks (standard protocol: predict tail given head+relation)
    tail_ranks = [r.tail_rank for r in results]
    
    return {
        "mrr": mrr(tail_ranks),
        "hits@1": hits_at_k(tail_ranks, 1),
        "hits@3": hits_at_k(tail_ranks, 3),
        "hits@10": hits_at_k(tail_ranks, 10),
        "num_queries": len(results),
        "mean_rank": float(np.mean(tail_ranks)),
        "median_rank": float(np.median(tail_ranks)),
    }


def filtered_rank(
    true_entity: str,
    scores: Dict[str, float],
    filter_entities: set,
) -> int:
    """
    Compute filtered rank (standard protocol).
    
    Filtered ranking removes other valid answers from consideration,
    so we only penalize for ranking random entities above the true one.
    
    Args:
        true_entity: The correct answer
        scores: Dict mapping entity -> score
        filter_entities: Other valid entities to filter out
    
    Returns:
        Filtered rank (1-indexed)
    """
    true_score = scores.get(true_entity, float('-inf'))
    
    rank = 1
    for entity, score in scores.items():
        if entity == true_entity:
            continue
        if entity in filter_entities:
            continue  # Filter out other valid answers
        if score > true_score:
            rank += 1
    
    return rank


class LinkPredictionEvaluator:
    """
    Evaluator for link prediction task.
    
    Standard protocol:
    1. For each test triple (h, r, t):
       a. Corrupt tail: score all (h, r, e) for e in entities
       b. Corrupt head: score all (e, r, t) for e in entities
    2. Filter out other valid triples (filtered setting)
    3. Report filtered MRR, Hits@1/3/10
    """
    
    def __init__(
        self,
        all_entities: List[str],
        train_triples: set,
        valid_triples: set = None,
    ):
        """
        Args:
            all_entities: List of all entity IDs
            train_triples: Set of (h, r, t) tuples from training
            valid_triples: Set of (h, r, t) tuples from validation (optional)
        """
        self.all_entities = all_entities
        self.train_triples = train_triples
        self.valid_triples = valid_triples or set()
        self.known_triples = train_triples | self.valid_triples
        
        # Build index for filtering
        self._build_filter_index()
    
    def _build_filter_index(self):
        """Build indices for efficient filtering."""
        # For (h, r, ?): which tails are valid?
        self.hr_to_tails = {}
        # For (?, r, t): which heads are valid?
        self.rt_to_heads = {}
        
        for h, r, t in self.known_triples:
            key_hr = (h, r)
            key_rt = (r, t)
            
            if key_hr not in self.hr_to_tails:
                self.hr_to_tails[key_hr] = set()
            self.hr_to_tails[key_hr].add(t)
            
            if key_rt not in self.rt_to_heads:
                self.rt_to_heads[key_rt] = set()
            self.rt_to_heads[key_rt].add(h)
    
    def evaluate_triple(
        self,
        head: str,
        relation: str,
        tail: str,
        score_fn,
    ) -> RankingResult:
        """
        Evaluate a single test triple.
        
        Args:
            head: Head entity
            relation: Relation type
            tail: True tail entity
            score_fn: Function (h, r, t) -> float
        
        Returns:
            RankingResult with filtered ranks
        """
        # Score all possible tails
        tail_scores = {}
        for entity in self.all_entities:
            tail_scores[entity] = score_fn(head, relation, entity)
        
        # Get filter set (other valid tails for this h,r pair)
        filter_tails = self.hr_to_tails.get((head, relation), set()) - {tail}
        
        # Compute filtered rank
        tail_rank = filtered_rank(tail, tail_scores, filter_tails)
        
        # Score all possible heads (for bidirectional evaluation)
        head_scores = {}
        for entity in self.all_entities:
            head_scores[entity] = score_fn(entity, relation, tail)
        
        # Get filter set for heads
        filter_heads = self.rt_to_heads.get((relation, tail), set()) - {head}
        head_rank = filtered_rank(head, head_scores, filter_heads)
        
        return RankingResult(
            head=head,
            relation=relation,
            tail=tail,
            tail_rank=tail_rank,
            head_rank=head_rank,
            num_candidates=len(self.all_entities),
        )
    
    def evaluate_batch(
        self,
        test_triples: List[Tuple[str, str, str]],
        score_fn,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate on a batch of test triples.
        
        Args:
            test_triples: List of (h, r, t) tuples
            score_fn: Function (h, r, t) -> float
            verbose: Print progress
        
        Returns:
            Dictionary with all metrics
        """
        results = []
        
        for i, (h, r, t) in enumerate(test_triples):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Evaluated {i + 1}/{len(test_triples)} triples...")
            
            result = self.evaluate_triple(h, r, t, score_fn)
            results.append(result)
        
        return evaluate_rankings(results)


# Convenience function for quick evaluation
def quick_evaluate(
    test_triples: List[Tuple[str, str, str]],
    score_fn,
    all_entities: List[str],
    train_triples: set,
) -> Dict[str, float]:
    """
    Quick evaluation helper.
    
    Args:
        test_triples: Test set
        score_fn: Scoring function
        all_entities: All entities in KG
        train_triples: Training triples for filtering
    
    Returns:
        Metrics dictionary
    """
    evaluator = LinkPredictionEvaluator(all_entities, train_triples)
    return evaluator.evaluate_batch(test_triples, score_fn)

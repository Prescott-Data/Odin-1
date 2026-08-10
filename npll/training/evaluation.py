"""
Evaluation Metrics for NPLL Implementation
Implements MRR, Hit@K, and other knowledge graph evaluation metrics from the paper
"""

import torch
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from collections import defaultdict
import time

from ..core import KnowledgeGraph, Triple, Entity, Relation
from ..npll_model import NPLLModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for knowledge graph completion
    """
    # Link prediction metrics (standard)
    mrr: float  # Mean Reciprocal Rank
    hit_at_1: float  # Hit@1
    hit_at_3: float  # Hit@3
    hit_at_10: float  # Hit@10
    
    # Extended metrics
    mean_rank: float  # Mean rank
    median_rank: float  # Median rank
    
    # Rule quality metrics
    rule_precision: Optional[float] = None
    rule_recall: Optional[float] = None
    rule_f1: Optional[float] = None
    
    # Confidence calibration
    calibration_error: Optional[float] = None
    confidence_correlation: Optional[float] = None
    
    # Efficiency metrics
    evaluation_time: float = 0.0
    predictions_per_second: float = 0.0
    
    def __str__(self) -> str:
        return (f"Evaluation Metrics:\\n"
                f"  MRR: {self.mrr:.4f}\\n"
                f"  Hit@1: {self.hit_at_1:.4f}\\n"
                f"  Hit@3: {self.hit_at_3:.4f}\\n"
                f"  Hit@10: {self.hit_at_10:.4f}\\n"
                f"  Mean Rank: {self.mean_rank:.2f}\\n"
                f"  Median Rank: {self.median_rank:.2f}")


class KnowledgeGraphEvaluator:
    """
    Base evaluator for knowledge graph tasks
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, filter_known: bool = True):
        """
        Initialize evaluator
        
        Args:
            knowledge_graph: Knowledge graph for evaluation
            filter_known: Whether to filter out known facts during ranking
        """
        self.kg = knowledge_graph
        self.filter_known = filter_known
        
        # Create sets for efficient lookup
        self.known_fact_set = set()
        for fact in self.kg.known_facts:
            self.known_fact_set.add((fact.head.name, fact.relation.name, fact.tail.name))
    
    def create_negative_samples(self, 
                               test_triple: Triple, 
                               corruption_mode: str = "both") -> List[Triple]:
        """
        Create negative samples by corrupting head or tail entities
        
        Args:
            test_triple: Triple to create negatives for
            corruption_mode: "head", "tail", or "both"
            
        Returns:
            List of negative triples
        """
        negatives = []
        
        if corruption_mode in ["head", "both"]:
            # Corrupt head entity
            for entity in self.kg.entities:
                if entity != test_triple.head:
                    negative = Triple(
                        head=entity,
                        relation=test_triple.relation,
                        tail=test_triple.tail
                    )
                    negatives.append(negative)
        
        if corruption_mode in ["tail", "both"]:
            # Corrupt tail entity
            for entity in self.kg.entities:
                if entity != test_triple.tail:
                    negative = Triple(
                        head=test_triple.head,
                        relation=test_triple.relation,
                        tail=entity
                    )
                    negatives.append(negative)
        
        return negatives
    
    def filter_candidates(self, candidates: List[Triple]) -> List[Triple]:
        """Filter out known facts from candidates if filtering is enabled"""
        if not self.filter_known:
            return candidates
        
        filtered = []
        for candidate in candidates:
            triple_tuple = (candidate.head.name, candidate.relation.name, candidate.tail.name)
            if triple_tuple not in self.known_fact_set:
                filtered.append(candidate)
        
        return filtered


class LinkPredictionEvaluator(KnowledgeGraphEvaluator):
    """
    Evaluator for link prediction task
    Implements standard knowledge graph completion metrics
    """
    
    def evaluate_link_prediction(self, 
                                model: NPLLModel,
                                test_triples: Optional[List[Triple]] = None,
                                top_k: List[int] = [1, 3, 10],
                                corruption_mode: str = "both") -> Dict[str, float]:
        """
        Evaluate link prediction performance
        
        Args:
            model: Trained NPLL model
            test_triples: Test triples (uses unknown facts if None)
            top_k: List of K values for Hit@K computation
            corruption_mode: "head", "tail", or "both"
            
        Returns:
            Dictionary with evaluation metrics
        """
        if test_triples is None:
            test_triples = list(self.kg.unknown_facts)
        
        if not test_triples:
            logger.warning("No test triples available for evaluation")
            return {}
        
        logger.info(f"Evaluating link prediction on {len(test_triples)} test triples")
        
        start_time = time.time()
        ranks = []
        
        for i, test_triple in enumerate(test_triples):
            if i % 100 == 0:
                logger.debug(f"Evaluating triple {i}/{len(test_triples)}")
            
            # Create candidates (test triple + negatives)
            candidates = [test_triple] + self.create_negative_samples(test_triple, corruption_mode)
            
            # Filter known facts
            candidates = self.filter_candidates(candidates)
            
            if not candidates:
                continue
            
            # Get model predictions
            try:
                predictions = model.forward(candidates)
                scores = predictions['probabilities'].cpu().numpy()
                
                # Find rank of test triple (first candidate)
                test_score = scores[0]
                rank = 1 + np.sum(scores > test_score)  # Rank starts from 1
                ranks.append(rank)
                
            except Exception as e:
                logger.warning(f"Error evaluating triple {test_triple}: {e}")
                continue
        
        if not ranks:
            logger.error("No valid ranks computed")
            return {}
        
        evaluation_time = time.time() - start_time
        
        # Compute metrics
        ranks = np.array(ranks, dtype=float)
        
        metrics = {
            'mrr': float(np.mean(1.0 / ranks)),
            'mean_rank': float(np.mean(ranks)),
            'median_rank': float(np.median(ranks)),
            'evaluation_time': evaluation_time,
            'predictions_per_second': len(test_triples) / evaluation_time if evaluation_time > 0 else 0.0
        }
        
        # Compute Hit@K metrics
        for k in top_k:
            hit_at_k = np.mean(ranks <= k)
            metrics[f'hit@{k}'] = float(hit_at_k)
        
        logger.info(f"Link prediction evaluation completed: MRR={metrics['mrr']:.4f}")
        
        return metrics
    
    def evaluate_entity_ranking(self, 
                               model: NPLLModel,
                               query_relations: List[str],
                               top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Evaluate entity ranking for specific relations
        
        Args:
            model: Trained NPLL model
            query_relations: Relations to evaluate
            top_k: Number of top entities to return
            
        Returns:
            Dictionary mapping relations to ranked entity lists
        """
        results = {}
        
        for relation_name in query_relations:
            relation = self.kg.get_relation(relation_name)
            if relation is None:
                continue
            
            # For each head entity, rank all possible tail entities
            entity_scores = defaultdict(list)
            
            for head_entity in self.kg.entities:
                candidates = []
                for tail_entity in self.kg.entities:
                    if head_entity != tail_entity:
                        candidate = Triple(head=head_entity, relation=relation, tail=tail_entity)
                        candidates.append(candidate)
                
                # Get predictions
                if candidates:
                    predictions = model.forward(candidates)
                    scores = predictions['probabilities'].cpu().numpy()
                    
                    for candidate, score in zip(candidates, scores):
                        entity_scores[candidate.tail.name].append(score)
            
            # Average scores and rank
            avg_scores = [(entity, np.mean(scores)) 
                         for entity, scores in entity_scores.items()]
            avg_scores.sort(key=lambda x: x[1], reverse=True)
            
            results[relation_name] = avg_scores[:top_k]
        
        return results


class RuleQualityEvaluator(KnowledgeGraphEvaluator):
    """
    Evaluator for logical rule quality
    """
    
    def evaluate_rule_quality(self, model: NPLLModel) -> Dict[str, float]:
        """
        Evaluate quality of learned logical rules
        
        Args:
            model: Trained NPLL model
            
        Returns:
            Dictionary with rule quality metrics
        """
        if not model.is_initialized or model.mln is None:
            return {}
        
        rule_confidences = model.get_rule_confidences()
        if not rule_confidences:
            return {}
        
        # Compute rule statistics
        confidences = list(rule_confidences.values())
        
        metrics = {
            'avg_rule_confidence': float(np.mean(confidences)),
            'std_rule_confidence': float(np.std(confidences)),
            'min_rule_confidence': float(np.min(confidences)),
            'max_rule_confidence': float(np.max(confidences)),
            'num_high_confidence_rules': int(np.sum(np.array(confidences) > 0.8)),
            'num_low_confidence_rules': int(np.sum(np.array(confidences) < 0.2))
        }
        
        return metrics
    
    def evaluate_rule_coverage(self, model: NPLLModel) -> Dict[str, float]:
        """
        Evaluate how well rules cover the known facts
        
        Args:
            model: Trained NPLL model
            
        Returns:
            Dictionary with coverage metrics
        """
        if not model.is_initialized or model.mln is None:
            return {}
        
        # This would require more complex analysis of ground rules
        # For now, return basic statistics
        
        total_ground_rules = len(model.mln.ground_rules) if model.mln.ground_rules else 0
        total_facts = len(self.kg.known_facts) + len(self.kg.unknown_facts)
        
        coverage_ratio = total_ground_rules / total_facts if total_facts > 0 else 0.0
        
        return {
            'total_ground_rules': total_ground_rules,
            'total_facts': total_facts,
            'coverage_ratio': coverage_ratio
        }


class ConfidenceCalibrationEvaluator(KnowledgeGraphEvaluator):
    """
    Evaluator for prediction confidence calibration
    """
    
    def evaluate_calibration(self, 
                           model: NPLLModel,
                           test_triples: List[Triple],
                           num_bins: int = 10) -> Dict[str, float]:
        """
        Evaluate confidence calibration using reliability diagrams
        
        Args:
            model: Trained NPLL model
            test_triples: Test triples with ground truth
            num_bins: Number of confidence bins
            
        Returns:
            Dictionary with calibration metrics
        """
        # Get model predictions
        predictions = model.forward(test_triples)
        confidences = predictions['probabilities'].cpu().numpy()
        
        # For this example, assume all test triples are positive
        # In practice, you'd need ground truth labels
        ground_truth = np.ones(len(test_triples))  # Placeholder
        
        # Compute Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = ground_truth[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'expected_calibration_error': float(ece),
            'avg_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences))
        }


def create_evaluator(knowledge_graph: KnowledgeGraph, 
                    evaluation_type: str = "link_prediction") -> KnowledgeGraphEvaluator:
    """
    Factory function to create appropriate evaluator
    
    Args:
        knowledge_graph: Knowledge graph for evaluation
        evaluation_type: Type of evaluator to create
        
    Returns:
        Configured evaluator
    """
    if evaluation_type == "link_prediction":
        return LinkPredictionEvaluator(knowledge_graph)
    elif evaluation_type == "rule_quality":
        return RuleQualityEvaluator(knowledge_graph)
    elif evaluation_type == "confidence_calibration":
        return ConfidenceCalibrationEvaluator(knowledge_graph)
    else:
        return LinkPredictionEvaluator(knowledge_graph)  # Default


def comprehensive_evaluation(model: NPLLModel,
                            knowledge_graph: KnowledgeGraph,
                            test_triples: Optional[List[Triple]] = None) -> EvaluationMetrics:
    """
    Run comprehensive evaluation of NPLL model
    
    Args:
        model: Trained NPLL model
        knowledge_graph: Knowledge graph for evaluation
        test_triples: Optional test triples
        
    Returns:
        Comprehensive evaluation metrics
    """
    start_time = time.time()
    
    # Link prediction evaluation
    link_evaluator = LinkPredictionEvaluator(knowledge_graph)
    link_metrics = link_evaluator.evaluate_link_prediction(model, test_triples)
    
    # Rule quality evaluation
    rule_evaluator = RuleQualityEvaluator(knowledge_graph)
    rule_metrics = rule_evaluator.evaluate_rule_quality(model)
    
    evaluation_time = time.time() - start_time
    
    # Create comprehensive metrics object
    metrics = EvaluationMetrics(
        mrr=link_metrics.get('mrr', 0.0),
        hit_at_1=link_metrics.get('hit@1', 0.0),
        hit_at_3=link_metrics.get('hit@3', 0.0),
        hit_at_10=link_metrics.get('hit@10', 0.0),
        mean_rank=link_metrics.get('mean_rank', 0.0),
        median_rank=link_metrics.get('median_rank', 0.0),
        rule_precision=rule_metrics.get('avg_rule_confidence'),
        evaluation_time=evaluation_time,
        predictions_per_second=link_metrics.get('predictions_per_second', 0.0)
    )
    
    return metrics


# Example usage function
def example_evaluation():
    """
    Example showing comprehensive evaluation with sample data
    """
    from ..core import load_knowledge_graph_from_triples
    from ..core.logical_rules import RuleGenerator
    from ..npll_model import create_npll_model
    from ..utils import get_config
    
    # Create sample data
    sample_triples = [
        ('Alice', 'friendOf', 'Bob'),
        ('Bob', 'worksAt', 'Company'),
        ('Charlie', 'friendOf', 'Alice'),
        ('Alice', 'livesIn', 'NYC')
    ]
    
    kg = load_knowledge_graph_from_triples(sample_triples, "Eval Test")
    
    # Generate rules
    rule_gen = RuleGenerator(kg)
    rules = rule_gen.generate_simple_rules(min_support=1)
    
    # Create and train model (simplified)
    config = get_config("ArangoDB_Triples")
    model = create_npll_model(config)
    model.initialize(kg, rules)
    
    # Add unknown facts for evaluation
    kg.add_unknown_fact('Charlie', 'worksAt', 'Company')
    kg.add_unknown_fact('Bob', 'livesIn', 'NYC')
    
    # Quick training
    model.train_epoch()
    
    # Comprehensive evaluation
    metrics = comprehensive_evaluation(model, kg)
    print(f"Evaluation Results: {metrics}")
    
    # Specific evaluations
    evaluator = create_evaluator(kg)
    link_metrics = evaluator.evaluate_link_prediction(model)
    print(f"Link Prediction: {link_metrics}")
    
    return metrics


if __name__ == "__main__":
    example_evaluation()
"""
Batch processing utilities for NPLL ground rules
Handles efficient batching and sampling of ground rules for MLN computations
"""

import torch
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Iterator, Any
from collections import defaultdict
import random
import logging
from dataclasses import dataclass

from ..core import Triple, LogicalRule, GroundRule, KnowledgeGraph
from ..utils.config import NPLLConfig

logger = logging.getLogger(__name__)


@dataclass
class GroundRuleBatch:
    """
    Batch of ground rules for efficient processing
    
    Contains ground rules and associated metadata for batch operations
    """
    ground_rules: List[GroundRule]
    rule_indices: torch.Tensor  # Which logical rule each ground rule belongs to
    fact_indices: Dict[Triple, int]  # Mapping from facts to batch indices
    batch_facts: List[Triple]  # All unique facts in this batch
    batch_size: int
    
    def __post_init__(self):
        """Validate batch consistency"""
        assert len(self.ground_rules) == self.batch_size, \
            f"Inconsistent batch size: {len(self.ground_rules)} vs {self.batch_size}"
        
        assert len(self.rule_indices) == self.batch_size, \
            f"Rule indices length mismatch: {len(self.rule_indices)} vs {self.batch_size}"
    
    def get_fact_truth_matrix(self, fact_assignments: Dict[Triple, bool]) -> torch.Tensor:
        """
        Create truth value matrix for facts in this batch
        
        Returns:
            Tensor of shape [batch_size, max_facts_per_rule] with truth values
        """
        max_facts = max(len(gr.get_all_facts()) for gr in self.ground_rules) if self.ground_rules else 0
        
        if max_facts == 0:
            return torch.zeros(self.batch_size, 0, dtype=torch.bool)
        
        truth_matrix = torch.zeros(self.batch_size, max_facts, dtype=torch.bool)
        
        for i, ground_rule in enumerate(self.ground_rules):
            facts = ground_rule.get_all_facts()
            for j, fact in enumerate(facts):
                if j < max_facts:
                    truth_matrix[i, j] = fact_assignments.get(fact, False)
        
        return truth_matrix
    
    def evaluate_ground_rules(self, fact_assignments: Dict[Triple, bool]) -> torch.Tensor:
        """
        Evaluate all ground rules in batch
        
        Returns:
            Boolean tensor indicating which ground rules are satisfied
        """
        satisfaction = torch.zeros(self.batch_size, dtype=torch.bool)
        
        for i, ground_rule in enumerate(self.ground_rules):
            # Check body satisfaction
            body_satisfied = all(
                fact_assignments.get(fact, False)
                for fact in ground_rule.body_facts
            )
            
            if not body_satisfied:
                # Body false -> rule vacuously true
                satisfaction[i] = True
            else:
                # Body true -> check head
                head_satisfied = fact_assignments.get(ground_rule.head_fact, False)
                satisfaction[i] = head_satisfied
        
        return satisfaction


class GroundRuleSampler:
    """
    Samples ground rules for efficient MLN training and inference
    
    Paper Section 4.2: "this paper randomly samples batches of ground rules to form datasets,
    wherein the ground rules are approximately independent of each batch"
    """
    
    def __init__(self, config: NPLLConfig, random_seed: Optional[int] = None):
        self.config = config
        self.batch_size = config.batch_size
        self.max_ground_rules = config.max_ground_rules
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
    
    def sample_ground_rules(self, all_ground_rules: List[GroundRule], 
                          num_batches: int = 1,
                          sampling_strategy: str = "uniform") -> List[GroundRuleBatch]:
        """
        Sample batches of ground rules
        
        Args:
            all_ground_rules: All available ground rules
            num_batches: Number of batches to create
            sampling_strategy: 'uniform', 'weighted', or 'stratified'
            
        Returns:
            List of GroundRuleBatch objects
        """
        if not all_ground_rules:
            return []
        
        total_rules = len(all_ground_rules)
        rules_per_batch = min(self.batch_size, total_rules // num_batches) if num_batches > 1 else min(self.batch_size, total_rules)
        
        batches = []
        
        for batch_idx in range(num_batches):
            if sampling_strategy == "uniform":
                sampled_rules = self._uniform_sampling(all_ground_rules, rules_per_batch)
            elif sampling_strategy == "weighted":
                sampled_rules = self._weighted_sampling(all_ground_rules, rules_per_batch)
            elif sampling_strategy == "stratified":
                sampled_rules = self._stratified_sampling(all_ground_rules, rules_per_batch)
            else:
                sampled_rules = self._uniform_sampling(all_ground_rules, rules_per_batch)
            
            if sampled_rules:
                batch = self._create_batch_from_rules(sampled_rules)
                batches.append(batch)
        
        logger.debug(f"Created {len(batches)} ground rule batches with avg size {rules_per_batch}")
        return batches
    
    def _uniform_sampling(self, ground_rules: List[GroundRule], 
                         sample_size: int) -> List[GroundRule]:
        """Uniform random sampling of ground rules"""
        if sample_size >= len(ground_rules):
            return ground_rules.copy()
        
        return random.sample(ground_rules, sample_size)
    
    def _weighted_sampling(self, ground_rules: List[GroundRule], 
                          sample_size: int) -> List[GroundRule]:
        """
        Weighted sampling based on rule confidence/support
        Higher confidence rules are more likely to be sampled
        """
        if sample_size >= len(ground_rules):
            return ground_rules.copy()
        
        # Use parent rule confidence as weight
        weights = [gr.parent_rule.confidence for gr in ground_rules]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Sample with replacement
        sampled_indices = np.random.choice(
            len(ground_rules), 
            size=sample_size, 
            p=weights, 
            replace=False if sample_size <= len(ground_rules) else True
        )
        
        return [ground_rules[i] for i in sampled_indices]
    
    def _stratified_sampling(self, ground_rules: List[GroundRule], 
                           sample_size: int) -> List[GroundRule]:
        """
        Stratified sampling ensuring representation from different rule types
        """
        if sample_size >= len(ground_rules):
            return ground_rules.copy()
        
        # Group by parent rule type
        rule_type_groups = defaultdict(list)
        for gr in ground_rules:
            rule_type_groups[gr.parent_rule.rule_type].append(gr)
        
        # Sample proportionally from each group
        sampled_rules = []
        remaining_samples = sample_size
        
        for rule_type, type_rules in rule_type_groups.items():
            # Proportional allocation
            group_sample_size = min(
                len(type_rules),
                max(1, int(remaining_samples * len(type_rules) / len(ground_rules)))
            )
            
            if group_sample_size > 0:
                group_sample = random.sample(type_rules, group_sample_size)
                sampled_rules.extend(group_sample)
                remaining_samples -= group_sample_size
        
        # If we need more samples, fill randomly
        if remaining_samples > 0 and len(sampled_rules) < sample_size:
            remaining_rules = [gr for gr in ground_rules if gr not in sampled_rules]
            if remaining_rules:
                additional_samples = min(remaining_samples, len(remaining_rules))
                additional_rules = random.sample(remaining_rules, additional_samples)
                sampled_rules.extend(additional_rules)
        
        return sampled_rules[:sample_size]
    
    def _create_batch_from_rules(self, ground_rules: List[GroundRule]) -> GroundRuleBatch:
        """Create GroundRuleBatch from list of ground rules"""
        if not ground_rules:
            return GroundRuleBatch(
                ground_rules=[],
                rule_indices=torch.tensor([]),
                fact_indices={},
                batch_facts=[],
                batch_size=0
            )
        
        # Extract rule indices (assuming rules are indexed by their position in logical_rules list)
        rule_indices = []
        unique_facts = set()
        
        # Build parent rule ID to index mapping (this should be provided by MLN)
        rule_id_to_idx = {}
        for i, gr in enumerate(ground_rules):
            if gr.parent_rule.rule_id not in rule_id_to_idx:
                rule_id_to_idx[gr.parent_rule.rule_id] = len(rule_id_to_idx)
            
            rule_indices.append(rule_id_to_idx[gr.parent_rule.rule_id])
            
            # Collect all unique facts
            unique_facts.update(gr.get_all_facts())
        
        # Create fact indexing
        batch_facts = list(unique_facts)
        fact_indices = {fact: i for i, fact in enumerate(batch_facts)}
        
        return GroundRuleBatch(
            ground_rules=ground_rules,
            rule_indices=torch.tensor(rule_indices, dtype=torch.long),
            fact_indices=fact_indices,
            batch_facts=batch_facts,
            batch_size=len(ground_rules)
        )
    
    def create_batches_for_training(self, ground_rules: List[GroundRule], 
                                  shuffle: bool = True) -> List[GroundRuleBatch]:
        """
        Create batches specifically for training
        
        Args:
            ground_rules: All ground rules to batch
            shuffle: Whether to shuffle before batching
            
        Returns:
            List of training batches
        """
        if not ground_rules:
            return []
        
        # Shuffle if requested
        rules_to_batch = ground_rules.copy()
        if shuffle:
            random.shuffle(rules_to_batch)
        
        # Create sequential batches
        batches = []
        for i in range(0, len(rules_to_batch), self.batch_size):
            batch_rules = rules_to_batch[i:i + self.batch_size]
            batch = self._create_batch_from_rules(batch_rules)
            batches.append(batch)
        
        return batches


class FactBatchProcessor:
    """
    Processes facts in batches for efficient scoring and probability computation
    """
    
    def __init__(self, config: NPLLConfig):
        self.config = config
        self.batch_size = config.batch_size
    
    def create_fact_batches(self, facts: List[Triple], 
                          batch_size: Optional[int] = None) -> List[List[Triple]]:
        """Create batches of facts for processing"""
        batch_size = batch_size or self.batch_size
        
        batches = []
        for i in range(0, len(facts), batch_size):
            batch = facts[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def process_fact_batches(self, fact_batches: List[List[Triple]], 
                           processor_func) -> List[Any]:
        """Process batches using provided function"""
        results = []
        
        for batch in fact_batches:
            batch_result = processor_func(batch)
            results.append(batch_result)
        
        return results


class MemoryEfficientBatcher:
    """
    Memory-efficient batching for large-scale ground rule processing
    Uses generators to avoid loading all data into memory
    """
    
    def __init__(self, config: NPLLConfig):
        self.config = config
        self.batch_size = config.batch_size
    
    def create_ground_rule_iterator(self, ground_rules: List[GroundRule], 
                                  shuffle: bool = True) -> Iterator[GroundRuleBatch]:
        """
        Create iterator over ground rule batches for memory efficiency
        
        Args:
            ground_rules: All ground rules
            shuffle: Whether to shuffle order
            
        Yields:
            GroundRuleBatch objects
        """
        if shuffle:
            indices = list(range(len(ground_rules)))
            random.shuffle(indices)
        else:
            indices = list(range(len(ground_rules)))
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_rules = [ground_rules[idx] for idx in batch_indices]
            
            # Create batch
            batch = self._create_efficient_batch(batch_rules)
            yield batch
    
    def _create_efficient_batch(self, ground_rules: List[GroundRule]) -> GroundRuleBatch:
        """Create batch with minimal memory overhead"""
        if not ground_rules:
            return GroundRuleBatch([], torch.tensor([]), {}, [], 0)
        
        # Efficient fact collection using sets
        all_facts = set()
        rule_indices = []
        
        # Single pass to collect facts and rule indices
        for i, gr in enumerate(ground_rules):
            all_facts.update(gr.get_all_facts())
            # Use hash of rule_id as index for efficiency
            rule_indices.append(hash(gr.parent_rule.rule_id) % 1000)
        
        batch_facts = list(all_facts)
        fact_indices = {fact: i for i, fact in enumerate(batch_facts)}
        
        return GroundRuleBatch(
            ground_rules=ground_rules,
            rule_indices=torch.tensor(rule_indices, dtype=torch.long),
            fact_indices=fact_indices,
            batch_facts=batch_facts,
            batch_size=len(ground_rules)
        )


class AdaptiveBatcher:
    """
    Adaptive batching that adjusts batch size based on memory usage and performance
    """
    
    def __init__(self, config: NPLLConfig, initial_batch_size: Optional[int] = None):
        self.config = config
        self.current_batch_size = initial_batch_size or config.batch_size
        self.min_batch_size = max(1, config.batch_size // 4)
        self.max_batch_size = config.batch_size * 2
        
        # Performance tracking
        self.performance_history = []
        self.memory_usage_history = []
    
    def adapt_batch_size(self, processing_time: float, memory_usage: float, 
                        target_time: float = 1.0):
        """
        Adapt batch size based on performance metrics
        
        Args:
            processing_time: Time taken to process current batch
            target_time: Target processing time per batch
            memory_usage: Memory usage for current batch
        """
        self.performance_history.append(processing_time)
        self.memory_usage_history.append(memory_usage)
        
        # Keep only recent history
        max_history = 10
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
            self.memory_usage_history = self.memory_usage_history[-max_history:]
        
        # Adjust based on performance
        if processing_time > target_time * 1.5:
            # Too slow, decrease batch size
            new_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
        elif processing_time < target_time * 0.5:
            # Too fast, increase batch size
            new_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
        else:
            # Good performance, keep current size
            new_batch_size = self.current_batch_size
        
        if new_batch_size != self.current_batch_size:
            logger.debug(f"Adapted batch size from {self.current_batch_size} to {new_batch_size}")
            self.current_batch_size = new_batch_size
    
    def get_current_batch_size(self) -> int:
        """Get current adaptive batch size"""
        return self.current_batch_size


def create_ground_rule_sampler(config: NPLLConfig, seed: Optional[int] = None) -> GroundRuleSampler:
    """Factory function to create ground rule sampler"""
    return GroundRuleSampler(config, seed)


def verify_batch_utils():
    """Verify batch utility implementations"""
    from ..utils.config import default_config
    from ..core import Entity, Relation, load_knowledge_graph_from_triples
    from ..core.logical_rules import Variable, Atom, RuleType
    
    # Create test data
    test_triples = [
        ("A", "r1", "B"),
        ("B", "r2", "C"),
        ("A", "r3", "C")
    ]
    
    kg = load_knowledge_graph_from_triples(test_triples)
    
    # Create test rule and ground rules
    r1, r2, r3 = Relation("r1"), Relation("r2"), Relation("r3")
    x, y, z = Variable('x'), Variable('y'), Variable('z')
    
    test_rule = LogicalRule(
        rule_id="test_rule",
        body=[Atom(r1, (x, y)), Atom(r2, (y, z))],
        head=Atom(r3, (x, z)),
        rule_type=RuleType.TRANSITIVITY
    )
    
    ground_rules = test_rule.generate_ground_rules(kg, max_groundings=10)
    
    # Test sampler
    sampler = GroundRuleSampler(default_config, seed=42)
    batches = sampler.sample_ground_rules(ground_rules, num_batches=2)
    
    assert len(batches) <= 2, "Should create at most 2 batches"
    
    for batch in batches:
        assert batch.batch_size == len(batch.ground_rules), "Batch size consistency"
        assert len(batch.rule_indices) == batch.batch_size, "Rule indices length"
    
    # Test memory-efficient batcher
    efficient_batcher = MemoryEfficientBatcher(default_config)
    batch_iterator = efficient_batcher.create_ground_rule_iterator(ground_rules)
    
    batches_from_iterator = list(batch_iterator)
    assert len(batches_from_iterator) > 0, "Should create batches from iterator"
    
    logger.info("Batch utilities verified successfully")
    
    return True
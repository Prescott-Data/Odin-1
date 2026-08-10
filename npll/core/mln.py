"""
Markov Logic Network (MLN) implementation for NPLL
"""

import torch
import torch.nn as nn
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
import logging
from dataclasses import dataclass

from .knowledge_graph import KnowledgeGraph, Triple
from .logical_rules import LogicalRule, GroundRule
from ..utils.config import NPLLConfig
from ..utils.math_utils import log_sum_exp, partition_function_approximation, compute_mln_probability

logger = logging.getLogger(__name__)


@dataclass
class MLNState:

    fact_assignments: Dict[Triple, bool]  # Truth values for all facts
    known_facts: Set[Triple]  # Known facts F
    unknown_facts: Set[Triple]  # Unknown facts U
    
    def __post_init__(self):
        """Validate MLN state"""
        all_facts = set(self.fact_assignments.keys())
        expected_facts = self.known_facts | self.unknown_facts
        
        if all_facts != expected_facts:
            missing = expected_facts - all_facts
            extra = all_facts - expected_facts
            logger.warning(f"MLN state inconsistency. Missing: {len(missing)}, Extra: {len(extra)}")
    
    def evaluate_ground_rule(self, ground_rule: GroundRule) -> bool:

        # Check if all body facts are true
        body_satisfied = all(
            self.fact_assignments.get(fact, False)
            for fact in ground_rule.body_facts
        )
        
        # If body is false, rule is vacuously true
        if not body_satisfied:
            return True
        
        # If body is true, check if head is true
        head_satisfied = self.fact_assignments.get(ground_rule.head_fact, False)
        return head_satisfied
    
    def count_satisfied_ground_rules(self, ground_rules: List[GroundRule]) -> int:
        """Count number of ground rules satisfied in this state"""
        return sum(1 for gr in ground_rules if self.evaluate_ground_rule(gr))


class MarkovLogicNetwork(nn.Module):
    
    def __init__(self, config: NPLLConfig):
        super().__init__()
        self.config = config
        
        # Core MLN components
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.logical_rules: List[LogicalRule] = []
        self.ground_rules: List[GroundRule] = []
        
        # Rule weights ω (learnable parameters)
        self.rule_weights: Optional[nn.Parameter] = None
        
        # Ground rule organization
        self.rule_to_ground_rules: Dict[str, List[GroundRule]] = defaultdict(list)
        self.ground_rule_facts: Set[Triple] = set()
        # Inverted index for fast lookup: fact -> ground rules containing it
        self.fact_to_groundrules: Dict[Triple, List[GroundRule]] = defaultdict(list)
        
        # Caching for efficiency
        self._partition_function_cache: Dict[str, torch.Tensor] = {}
        self._ground_rule_counts_cache: Optional[torch.Tensor] = None
    
    def add_knowledge_graph(self, kg: KnowledgeGraph):
        """Add knowledge graph to MLN"""
        self.knowledge_graph = kg
        logger.info(f"Added knowledge graph with {len(kg.known_facts)} known facts")
    
    def add_logical_rules(self, rules: List[LogicalRule]):
        self.logical_rules.extend(rules)
        
        # Initialize or expand rule weights
        if self.rule_weights is None:
            # Initialize rule weights to small random values
            initial_weights = torch.randn(len(rules)) * 0.1
            self.rule_weights = nn.Parameter(initial_weights, requires_grad=True)
        else:
            # Expand existing weights
            old_weights = self.rule_weights.data
            new_weights = torch.randn(len(rules)) * 0.1
            expanded_weights = torch.cat([old_weights, new_weights])
            self.rule_weights = nn.Parameter(expanded_weights, requires_grad=True)
        
        # Generate ground rules for new logical rules
        if self.knowledge_graph is not None:
            self._generate_ground_rules(rules)
        
        logger.info(f"Added {len(rules)} logical rules. Total: {len(self.logical_rules)}")
    
    def _generate_ground_rules(self, rules: List[LogicalRule]):
        """Generate ground rules from logical rules using knowledge graph"""
        new_ground_rules = []
        
        for rule in rules:
            # Generate ground rules for this logical rule
            ground_rules = rule.generate_ground_rules(
                self.knowledge_graph, 
                max_groundings=self.config.max_ground_rules
            )
            
            # Add to collections
            new_ground_rules.extend(ground_rules)
            self.rule_to_ground_rules[rule.rule_id].extend(ground_rules)
            
            # Collect all facts involved in ground rules
            for gr in ground_rules:
                self.ground_rule_facts.update(gr.get_all_facts())
                # Build inverted index for each fact
                for f in gr.get_all_facts():
                    self.fact_to_groundrules[f].append(gr)
        
        self.ground_rules.extend(new_ground_rules)
        logger.info(f"Generated {len(new_ground_rules)} ground rules. Total: {len(self.ground_rules)}")
    
    def compute_ground_rule_counts(self, fact_assignments: Dict[Triple, bool]) -> torch.Tensor:
        """
        Compute N(F,U) - number of satisfied ground rules for each logical rule
        """
        if not self.logical_rules:
            return torch.tensor([])
        
        rule_counts = torch.zeros(len(self.logical_rules))
        
        for rule_idx, rule in enumerate(self.logical_rules):
            ground_rules = self.rule_to_ground_rules[rule.rule_id]
            satisfied_count = 0
            
            for ground_rule in ground_rules:
                # Check if this ground rule is satisfied
                all_facts = ground_rule.get_all_facts()
                
                # Check if all required facts are true (for body) and conclusion follows
                body_satisfied = all(
                    fact_assignments.get(fact, False)
                    for fact in ground_rule.body_facts
                )
                
                if body_satisfied:
                    # If body is true, rule is satisfied if head is also true
                    head_satisfied = fact_assignments.get(ground_rule.head_fact, False)
                    if head_satisfied:
                        satisfied_count += 1
                else:
                    # If body is false, rule is vacuously satisfied
                    satisfied_count += 1
            
            rule_counts[rule_idx] = satisfied_count
        
        return rule_counts
    
    def compute_partition_function(self, sample_states: Optional[List[Dict[Triple, bool]]] = None,
                                 use_approximation: bool = True) -> torch.Tensor:
        """
        Compute MLN partition function Z(ω) from Equation 2
        """
        if self.rule_weights is None or len(self.logical_rules) == 0:
            return torch.tensor(0.0)
        
        # Use caching if available
        cache_key = str(self.rule_weights.data.tolist())
        if cache_key in self._partition_function_cache:
            return self._partition_function_cache[cache_key]
        
        if use_approximation or sample_states is not None:
            # Sampling-based approximation
            if sample_states is None:
                sample_states = self._generate_sample_states(num_samples=1000)
            
            # Compute counts for each sample state
            all_counts = []
            for state_assignment in sample_states:
                counts = self.compute_ground_rule_counts(state_assignment)
                all_counts.append(counts)
            
            if all_counts:
                counts_tensor = torch.stack(all_counts)  # [num_samples, num_rules]
                log_partition = partition_function_approximation(
                    self.rule_weights, counts_tensor, use_log_domain=True
                )
            else:
                log_partition = torch.tensor(0.0)
        else:
            # Exact computation (intractable for large graphs)
            logger.warning("Exact partition function computation is intractable for large graphs")
            log_partition = self._compute_exact_partition_function()
        
        # Cache result
        self._partition_function_cache[cache_key] = log_partition
        
        return log_partition
    
    def compute_joint_probability(self, fact_assignments: Dict[Triple, bool],
                                log_partition: Optional[torch.Tensor] = None,
                                detach_weights: bool = False) -> torch.Tensor:
        """
        Compute joint probability P(F,U|ω) from Equation 1

        """
        if self.rule_weights is None:
            return torch.tensor(0.0)
        
        # Compute ground rule counts N(F,U)
        counts = self.compute_ground_rule_counts(fact_assignments)
        
        # Compute log partition function if not provided
        if log_partition is None:
            log_partition = self.compute_partition_function()
        
        # Compute log probability using utility function
        weights_to_use = self.rule_weights.detach() if detach_weights else self.rule_weights
        log_prob = compute_mln_probability(
            weights_to_use, counts.unsqueeze(0), log_partition
        )
        
        return log_prob.squeeze(0)
    
    def _generate_sample_states(self, num_samples: int = 1000) -> List[Dict[Triple, bool]]:
        """
        Generate sample states for partition function approximation
        """
        sample_states = []
        
        if self.knowledge_graph is None:
            return sample_states
        
        # Get all facts that appear in ground rules
        all_facts = list(self.ground_rule_facts)
        
        if not all_facts:
            return sample_states
        
        # Generate random assignments
        for _ in range(num_samples):
            # Start with known facts as true
            assignment = {}
            
            # Set known facts to true
            for fact in self.knowledge_graph.known_facts:
                assignment[fact] = True
            
            # Randomly assign unknown facts
            unknown_facts_in_rules = [f for f in all_facts if f not in assignment]
            for fact in unknown_facts_in_rules:
                # Assign random truth value (could be made smarter)
                assignment[fact] = torch.rand(1).item() > 0.5
            
            sample_states.append(assignment)
        
        return sample_states
    
    def _compute_exact_partition_function(self) -> torch.Tensor:
        """
        Compute exact partition function 
        """
        if not self.ground_rule_facts:
            return torch.tensor(0.0)
        
        all_facts = list(self.ground_rule_facts)
        num_facts = len(all_facts)
        
        if num_facts > 20:  # Arbitrary limit to prevent memory explosion
            logger.error(f"Too many facts ({num_facts}) for exact partition function computation")
            return self.compute_partition_function(use_approximation=True)
        
        # Enumerate all possible truth assignments
        total_log_prob = []
        
        for i in range(2 ** num_facts):
            # Generate truth assignment from binary representation
            assignment = {}
            for j, fact in enumerate(all_facts):
                assignment[fact] = bool((i >> j) & 1)
            
            # Compute counts for this assignment
            counts = self.compute_ground_rule_counts(assignment)
            
            # Compute potential
            log_potential = torch.sum(self.rule_weights * counts)
            total_log_prob.append(log_potential)
        
        # Compute log-sum-exp
        if total_log_prob:
            log_partition = log_sum_exp(torch.stack(total_log_prob))
        else:
            log_partition = torch.tensor(0.0)
        
        return log_partition
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about the MLN"""
        stats = {
            'num_logical_rules': len(self.logical_rules),
            'num_ground_rules': len(self.ground_rules),
            'num_facts_in_ground_rules': len(self.ground_rule_facts),
            'rule_weights': self.rule_weights.data.tolist() if self.rule_weights is not None else []
        }
        
        # Per-rule statistics
        rule_stats = []
        for i, rule in enumerate(self.logical_rules):
            ground_rules = self.rule_to_ground_rules[rule.rule_id]
            rule_stat = {
                'rule_id': rule.rule_id,
                'rule_type': rule.rule_type.value,
                'num_ground_rules': len(ground_rules),
                'weight': self.rule_weights[i].item() if self.rule_weights is not None else 0.0,
                'learned_confidence': torch.sigmoid(self.rule_weights[i]).item() if self.rule_weights is not None else None,
                'support': rule.support
            }
            rule_stats.append(rule_stat)
        
        stats['rule_details'] = rule_stats
        
        return stats
    
    def forward(self, fact_assignments_batch: List[Dict[Triple, bool]]) -> torch.Tensor:
        """
        Forward pass for batch of fact assignments
        """
        if not fact_assignments_batch:
            return torch.tensor([])
        
        # Compute partition function once
        log_partition = self.compute_partition_function()
        
        # Compute probabilities for each assignment
        log_probs = []
        for assignment in fact_assignments_batch:
            log_prob = self.compute_joint_probability(assignment, log_partition)
            log_probs.append(log_prob)
        
        return torch.stack(log_probs) if log_probs else torch.tensor([])
    
    def sample_from_distribution(self, num_samples: int = 100) -> List[Dict[Triple, bool]]:
        """
        Sample fact assignments from MLN distribution using Gibbs sampling
        """
        if not self.ground_rule_facts:
            return []
        
        samples = []
        all_facts = list(self.ground_rule_facts)
        
        # Initialize with random assignment
        current_assignment = {fact: torch.rand(1).item() > 0.5 for fact in all_facts}
        
        # Set known facts to true (they don't change)
        if self.knowledge_graph:
            for fact in self.knowledge_graph.known_facts:
                current_assignment[fact] = True
        
        # Gibbs sampling
        for _ in range(num_samples):
            # Sample each unknown fact given others
            for fact in all_facts:
                if self.knowledge_graph and fact in self.knowledge_graph.known_facts:
                    continue  # Skip known facts
                
                # Compute conditional probability P(fact=True | others)
                prob_true = self._compute_conditional_probability(fact, current_assignment)
                
                # Sample from Bernoulli distribution
                current_assignment[fact] = torch.rand(1).item() < prob_true
            
            # Store sample
            samples.append(current_assignment.copy())
        
        return samples
    
    def _compute_conditional_probability(self, target_fact: Triple, 
                                       current_assignment: Dict[Triple, bool]) -> float:
        """
        Compute P(target_fact=True | other_facts) using local MLN structure
        """
        # Create two assignments: one with target_fact=True, one with False
        assignment_true = current_assignment.copy()
        assignment_false = current_assignment.copy()
        assignment_true[target_fact] = True
        assignment_false[target_fact] = False
        
        # Compute unnormalized probabilities
        log_prob_true = self.compute_joint_probability(assignment_true)
        log_prob_false = self.compute_joint_probability(assignment_false)
        
        # Normalize using log-sum-exp
        log_probs = torch.stack([log_prob_false, log_prob_true])
        normalized_probs = torch.softmax(log_probs, dim=0)
        
        return normalized_probs[1].item()  # Return P(target_fact=True)


def create_mln_from_kg_and_rules(kg: KnowledgeGraph, rules: List[LogicalRule], 
                                config: NPLLConfig) -> MarkovLogicNetwork:
    """
    Factory function to create MLN from knowledge graph and logical rules
    """
    mln = MarkovLogicNetwork(config)
    mln.add_knowledge_graph(kg)
    mln.add_logical_rules(rules)
    
    logger.info(f"Created MLN with {len(rules)} rules and {len(mln.ground_rules)} ground rules")
    
    return mln


def verify_mln_implementation():
    """Verify MLN implementation with small test case"""
    from ..utils.config import default_config
    from .knowledge_graph import Entity, Relation, load_knowledge_graph_from_triples
    from .logical_rules import Variable, Atom, RuleType
    
    # Create test knowledge graph
    test_triples = [
        ("Tom", "plays", "basketball"),
        ("Tom", "friend", "John"),  
        ("John", "plays", "soccer")
    ]
    
    kg = load_knowledge_graph_from_triples(test_triples, "TestKG")
    
    # Create test rule: plays(x, y) ∧ friend(x, z) ⇒ plays(z, y)
    plays_rel = Relation("plays")
    friend_rel = Relation("friend")
    
    x, y, z = Variable('x'), Variable('y'), Variable('z')
    
    body_atoms = [
        Atom(plays_rel, (x, y)),
        Atom(friend_rel, (x, z))
    ]
    head_atom = Atom(plays_rel, (z, y))
    
    test_rule = LogicalRule(
        rule_id="test_transitivity",
        body=body_atoms,
        head=head_atom,
        rule_type=RuleType.TRANSITIVITY,
        confidence=0.8
    )
    
    # Create MLN
    mln = create_mln_from_kg_and_rules(kg, [test_rule], default_config)
    
    # Verify MLN properties
    assert len(mln.logical_rules) == 1, "Should have 1 logical rule"
    assert len(mln.ground_rules) > 0, "Should have generated ground rules"
    assert mln.rule_weights is not None, "Should have initialized rule weights"
    
    # Test probability computation
    test_assignment = {fact: True for fact in kg.known_facts}
    log_prob = mln.compute_joint_probability(test_assignment)
    
    assert torch.isfinite(log_prob), "Joint probability should be finite"
    
    logger.info("MLN implementation verified successfully")
    
    return True
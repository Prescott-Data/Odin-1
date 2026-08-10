import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Set, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict

from ..core import Triple, LogicalRule, GroundRule, KnowledgeGraph
from ..core.mln import MarkovLogicNetwork
from ..scoring import NPLLScoringModule
from ..utils.config import NPLLConfig
from ..utils.math_utils import (
    safe_log, safe_sigmoid, bernoulli_log_prob, bernoulli_entropy,
    kl_divergence_bernoulli
)
from ..utils.batch_utils import GroundRuleBatch, GroundRuleSampler
from .elbo import ELBOComputer, VariationalInference

logger = logging.getLogger(__name__)


@dataclass
class EStepResult:
    """
    Result of E-step computation containing all relevant outputs
    """
    approximate_posterior_probs: torch.Tensor  # Q(U) probabilities
    fact_probabilities: Dict[Triple, float]  # Individual fact probabilities
    ground_rule_expectations: torch.Tensor  # Expected ground rule counts
    entropy: torch.Tensor  # Total entropy of Q(U)
    elbo_value: torch.Tensor  # Current ELBO value
    convergence_info: Dict[str, Any]  # Convergence diagnostics
    iteration_count: int  # Number of iterations used
    
    def __str__(self) -> str:
        return (f"E-step Result:\n"
                f"  Unknown facts: {len(self.approximate_posterior_probs)}\n"
                f"  Mean probability: {self.approximate_posterior_probs.mean().item():.4f}\n"
                f"  Entropy: {self.entropy.item():.4f}\n"
                f"  ELBO: {self.elbo_value.item():.4f}\n"
                f"  Iterations: {self.iteration_count}")


class MeanFieldApproximation(nn.Module):
    """
    Mean-field approximation for approximate posterior Q(U)
    """
    
    def __init__(self, config: NPLLConfig):
        super().__init__()
        self.config = config
        self.max_iterations = config.mean_field_iterations
        self.convergence_threshold = config.convergence_threshold
    
    def initialize_fact_probabilities(self, unknown_facts: List[Triple],
                                    scoring_module: NPLLScoringModule) -> torch.Tensor:
        """
        Initialize fact probabilities using scoring module
        """
        if not unknown_facts:
            return torch.tensor([])
        
        # Get initial scores from scoring module
        with torch.no_grad():
            initial_scores = scoring_module.forward(unknown_facts)
            # Transform to probabilities using sigmoid
            initial_probs = safe_sigmoid(initial_scores)
        
        return initial_probs
    
    def compute_q_u_distribution(self, fact_probs: torch.Tensor,
                                ground_rule_structure: Optional[List[List[int]]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute Q(U) distribution components
        """
        if len(fact_probs) == 0:
            return {
                'fact_probs': torch.tensor([]),
                'log_probs': torch.tensor([]),
                'entropy': torch.tensor(0.0)
            }
        
        # Clamp probabilities for numerical stability
        fact_probs_clamped = torch.clamp(fact_probs, min=1e-8, max=1.0 - 1e-8)
        
        # Compute log probabilities
        log_probs = safe_log(fact_probs_clamped)
        log_neg_probs = safe_log(1 - fact_probs_clamped)
        
        # Compute entropy of individual facts
        fact_entropies = bernoulli_entropy(fact_probs_clamped)
        total_entropy = torch.sum(fact_entropies)
        
        result = {
            'fact_probs': fact_probs_clamped,
            'log_probs': log_probs,
            'log_neg_probs': log_neg_probs,
            'fact_entropies': fact_entropies,
            'total_entropy': total_entropy
        }
        
        # If ground rule structure provided, compute ground rule probabilities
        if ground_rule_structure is not None:
            ground_rule_probs = self._compute_ground_rule_probabilities(
                fact_probs_clamped, ground_rule_structure
            )
            result.update(ground_rule_probs)
        
        return result
    
    def _compute_ground_rule_probabilities(self, fact_probs: torch.Tensor,
                                         ground_rule_structure: List[List[int]]) -> Dict[str, torch.Tensor]:
        """
        Compute probabilities for ground rules under mean-field approximation
        
        For ground rule with facts [i, j, k]: P(rule) = ∏ p_i * p_j * p_k
        """
        ground_rule_probs = []
        ground_rule_log_probs = []
        
        for fact_indices in ground_rule_structure:
            if fact_indices:
                # Get probabilities for facts in this ground rule
                rule_fact_probs = fact_probs[fact_indices]
                
                # Product probability (independence assumption)
                rule_prob = torch.prod(rule_fact_probs)
                rule_log_prob = torch.sum(safe_log(rule_fact_probs))
                
                ground_rule_probs.append(rule_prob)
                ground_rule_log_probs.append(rule_log_prob)
            else:
                ground_rule_probs.append(torch.tensor(0.0))
                ground_rule_log_probs.append(torch.tensor(float('-inf')))
        
        return {
            'ground_rule_probs': torch.stack(ground_rule_probs) if ground_rule_probs else torch.tensor([]),
            'ground_rule_log_probs': torch.stack(ground_rule_log_probs) if ground_rule_log_probs else torch.tensor([])
        }


class EStepOptimizer(nn.Module):
    """
    Optimizes the E-step objective function
    """
    
    def __init__(self, config: NPLLConfig):
        super().__init__()
        self.config = config
        self.mean_field = MeanFieldApproximation(config)
        self.elbo_computer = ELBOComputer(config)
        self.variational_inference = VariationalInference(config)
        
        # Optimization parameters
        self.max_iterations = config.mean_field_iterations
        self.convergence_threshold = config.convergence_threshold
    
    def optimize_approximate_posterior(self,
                                     mln: MarkovLogicNetwork,
                                     scoring_module: NPLLScoringModule,
                                     known_facts: List[Triple],
                                     unknown_facts: List[Triple],
                                     ground_rule_batches: Optional[List[GroundRuleBatch]] = None) -> EStepResult:
        """
        Main E-step optimization procedure
        """
        if not unknown_facts:
            # No unknown facts to optimize
            return EStepResult(
                approximate_posterior_probs=torch.tensor([]),
                fact_probabilities={},
                ground_rule_expectations=torch.tensor([]),
                entropy=torch.tensor(0.0),
                elbo_value=torch.tensor(0.0),
                convergence_info={'converged': True, 'reason': 'no_unknown_facts'},
                iteration_count=0
            )
        
        logger.debug(f"Starting E-step optimization for {len(unknown_facts)} unknown facts")
        
        # Initialize fact probabilities using scoring module
        current_probs = self.mean_field.initialize_fact_probabilities(unknown_facts, scoring_module)
        
        # Run a single VI optimization; it internally iterates and computes ELBO history
        updated_result = self.variational_inference.optimize_approximate_posterior(
            mln, known_facts, unknown_facts, current_probs.detach()
        )
        current_probs = updated_result['optimized_probs'].detach()
        elbo_history = updated_result['elbo_history']
        converged = updated_result['converged']
        iteration = updated_result['iterations']
        
        # Compute final Q(U) distribution
        final_q_dist = self.mean_field.compute_q_u_distribution(current_probs)
        
        # Compute ground rule expectations
        ground_rule_expectations = self._compute_ground_rule_expectations(
            mln, current_probs, unknown_facts
        )
        
        # Create fact probability dictionary
        fact_prob_dict = {fact: current_probs[i].item() 
                         for i, fact in enumerate(unknown_facts)}
        
        # Final ELBO computation (detached to avoid gradient issues)
        with torch.no_grad():
            final_elbo_components = self.elbo_computer.compute_elbo(
                mln, known_facts, unknown_facts, current_probs.detach()
            )
        
        convergence_info = {
            'converged': converged,
            'final_change': 0.0,
            'elbo_history': elbo_history,
            'reason': 'converged' if converged else 'max_iterations'
        }
        
        result = EStepResult(
            approximate_posterior_probs=current_probs,
            fact_probabilities=fact_prob_dict,
            ground_rule_expectations=ground_rule_expectations,
            entropy=final_q_dist['total_entropy'],
            elbo_value=final_elbo_components.elbo,
            convergence_info=convergence_info,
            iteration_count=iteration + 1
        )
        
        logger.debug(f"E-step completed: {result}")
        
        return result
    
    def _compute_ground_rule_expectations(self,
                                        mln: MarkovLogicNetwork,
                                        fact_probs: torch.Tensor,
                                        unknown_facts: List[Triple]) -> torch.Tensor:
        """
        Compute expected ground rule counts under Q(U)
        """
        if not mln.logical_rules or len(fact_probs) == 0:
            return torch.tensor([])
        
        # Create fact index mapping
        fact_to_idx = {fact: i for i, fact in enumerate(unknown_facts)}
        
        # Compute expectations for each logical rule
        expected_counts = torch.zeros(len(mln.logical_rules))
        
        for rule_idx, rule in enumerate(mln.logical_rules):
            ground_rules = mln.rule_to_ground_rules[rule.rule_id]
            rule_expectation = 0.0
            
            for ground_rule in ground_rules:
                # Compute expected satisfaction for this ground rule
                ground_rule_factors = []
                
                # Body facts
                for fact in ground_rule.body_facts:
                    if fact in fact_to_idx:
                        # Unknown fact - use probability
                        fact_idx = fact_to_idx[fact]
                        ground_rule_factors.append(fact_probs[fact_idx])
                    else:
                        # Known fact - assume probability 1
                        ground_rule_factors.append(torch.tensor(1.0))
                
                # Head fact
                if ground_rule.head_fact in fact_to_idx:
                    fact_idx = fact_to_idx[ground_rule.head_fact]
                    ground_rule_factors.append(fact_probs[fact_idx])
                else:
                    ground_rule_factors.append(torch.tensor(1.0))
                
                # Expected satisfaction is product of all factors
                if ground_rule_factors:
                    ground_rule_expectation = torch.prod(torch.stack(ground_rule_factors))
                    rule_expectation += ground_rule_expectation.item()
            
            expected_counts[rule_idx] = rule_expectation
        
        return expected_counts


class EStepRunner:
    """
    High-level runner for E-step computations
    Handles batching, parallelization, and result aggregation
    """
    
    def __init__(self, config: NPLLConfig):
        self.config = config
        self.e_step_optimizer = EStepOptimizer(config)
        self.ground_rule_sampler = GroundRuleSampler(config)
    
    def run_e_step(self,
                   mln: MarkovLogicNetwork,
                   scoring_module: NPLLScoringModule,
                   kg: KnowledgeGraph) -> EStepResult:
        known_facts = list(kg.known_facts)
        unknown_facts = list(kg.unknown_facts)
        
        if not unknown_facts:
            logger.info("No unknown facts for E-step")
            return EStepResult(
                approximate_posterior_probs=torch.tensor([]),
                fact_probabilities={},
                ground_rule_expectations=torch.tensor([]),
                entropy=torch.tensor(0.0),
                elbo_value=torch.tensor(0.0),
                convergence_info={'converged': True, 'reason': 'no_unknown_facts'},
                iteration_count=0
            )
        
        logger.info(f"Running E-step for {len(known_facts)} known and {len(unknown_facts)} unknown facts")
        
        # Create ground rule batches for efficient processing
        ground_rule_batches = None
        if len(mln.ground_rules) > self.config.batch_size:
            ground_rule_batches = self.ground_rule_sampler.sample_ground_rules(
                mln.ground_rules,
                num_batches=max(1, len(mln.ground_rules) // self.config.batch_size),
                sampling_strategy="uniform"
            )
        
        # Run optimization
        result = self.e_step_optimizer.optimize_approximate_posterior(
            mln, scoring_module, known_facts, unknown_facts, ground_rule_batches
        )
        
        logger.info(f"E-step completed: ELBO={result.elbo_value.item():.4f}, "
                   f"Entropy={result.entropy.item():.4f}, Iterations={result.iteration_count}")
        
        return result
    
    def run_e_step_with_constraints(self,
                                   mln: MarkovLogicNetwork,
                                   scoring_module: NPLLScoringModule,
                                   kg: KnowledgeGraph,
                                   fact_constraints: Optional[Dict[Triple, Tuple[float, float]]] = None) -> EStepResult:

        # Standard E-step
        result = self.run_e_step(mln, scoring_module, kg)
        
        # Apply constraints if provided
        if fact_constraints and len(result.approximate_posterior_probs) > 0:
            constrained_probs = result.approximate_posterior_probs.clone()
            unknown_facts = list(kg.unknown_facts)
            
            for i, fact in enumerate(unknown_facts):
                if fact in fact_constraints:
                    min_prob, max_prob = fact_constraints[fact]
                    constrained_probs[i] = torch.clamp(constrained_probs[i], min_prob, max_prob)
            
            # Update result with constrained probabilities
            result.approximate_posterior_probs = constrained_probs
            result.fact_probabilities = {
                fact: constrained_probs[i].item() 
                for i, fact in enumerate(unknown_facts)
            }
        
        return result


def create_e_step_runner(config: NPLLConfig) -> EStepRunner:
    """Factory function to create E-step runner"""
    return EStepRunner(config)


def verify_e_step_implementation():
    """Verify E-step implementation"""
    from ..utils.config import default_config
    from ..core import load_knowledge_graph_from_triples
    from ..core.mln import create_mln_from_kg_and_rules
    from ..core.logical_rules import Variable, Atom, RuleType, LogicalRule
    from ..scoring import create_scoring_module
    
    # Create test data
    test_triples = [
        ("A", "r1", "B"),
        ("B", "r2", "C"),
    ]
    
    kg = load_knowledge_graph_from_triples(test_triples)
    
    # Add unknown facts
    kg.add_unknown_fact("A", "r3", "C")
    
    # Create test rule
    from ..core import Relation
    r1, r2, r3 = Relation("r1"), Relation("r2"), Relation("r3")
    x, y, z = Variable('x'), Variable('y'), Variable('z')
    
    test_rule = LogicalRule(
        rule_id="test_e_step_rule",
        body=[Atom(r1, (x, y)), Atom(r2, (y, z))],
        head=Atom(r3, (x, z)),
        rule_type=RuleType.TRANSITIVITY
    )
    
    # Create MLN and scoring module
    mln = create_mln_from_kg_and_rules(kg, [test_rule], default_config)
    scoring_module = create_scoring_module(default_config, kg)
    
    # Test E-step
    e_step_runner = EStepRunner(default_config)
    result = e_step_runner.run_e_step(mln, scoring_module, kg)
    
    # Verify results
    assert len(result.approximate_posterior_probs) == len(kg.unknown_facts), \
        "Should have probabilities for all unknown facts"
    
    assert torch.all(result.approximate_posterior_probs >= 0), "Probabilities should be non-negative"
    assert torch.all(result.approximate_posterior_probs <= 1), "Probabilities should be <= 1"
    
    assert torch.isfinite(result.elbo_value), "ELBO should be finite"
    assert torch.isfinite(result.entropy), "Entropy should be finite"
    
    logger.info("E-step implementation verified successfully")
    
    return True
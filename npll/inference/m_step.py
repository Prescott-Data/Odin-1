
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from ..core import Triple, LogicalRule, GroundRule, KnowledgeGraph
from ..core.mln import MarkovLogicNetwork
from ..utils.config import NPLLConfig
from ..utils.math_utils import (
    safe_log, gradient_clipping, compute_markov_blanket_prob,
    log_sum_exp, partition_function_approximation
)
from ..utils.batch_utils import GroundRuleBatch, GroundRuleSampler
from .e_step import EStepResult

logger = logging.getLogger(__name__)


@dataclass
class MStepResult:
    """
    Result of M-step computation containing optimization details
    """
    updated_rule_weights: torch.Tensor  # New rule weights ω
    weight_changes: torch.Tensor  # Changes in weights from previous iteration
    gradient_norms: torch.Tensor  # Gradient norms for each rule
    pseudo_likelihood: torch.Tensor  # Final pseudo-likelihood value
    optimization_history: List[float]  # History of objective values
    convergence_info: Dict[str, Any]  # Convergence diagnostics
    iteration_count: int  # Number of optimization iterations
    
    def __str__(self) -> str:
        return (f"M-step Result:\n"
                f"  Rule weights: {self.updated_rule_weights.tolist()}\n"
                f"  Max weight change: {torch.max(torch.abs(self.weight_changes)).item():.6f}\n"
                f"  Pseudo-likelihood: {self.pseudo_likelihood.item():.4f}\n"
                f"  Iterations: {self.iteration_count}")


class PseudoLikelihoodComputer:
    """
    Computes pseudo-log-likelihood objective for M-step optimization

    """
    
    def __init__(self, config: NPLLConfig):
        self.config = config
        self.markov_blanket_size = config.markov_blanket_size
    
    def compute_pseudo_likelihood(self,
                                mln: MarkovLogicNetwork,
                                fact_probabilities: Dict[Triple, float],
                                ground_rule_batches: Optional[List[GroundRuleBatch]] = None) -> torch.Tensor:
        """
        Compute pseudo-log-likelihood objective
        """
        if not fact_probabilities or mln.rule_weights is None:
            return torch.tensor(0.0)

        likelihood_parts = []
        
        # Compute pseudo-likelihood for each fact
        for fact, q_prob in fact_probabilities.items():
            # Get Markov blanket for this fact
            markov_blanket = self._get_markov_blanket(fact, mln)
            
            # Compute P(fact=True|ω, Markov Blanket) as a tensor
            p_true = self._compute_conditional_probability(
                fact, markov_blanket, mln, fact_probabilities
            )
            # Full cross-entropy against Q: q*log p + (1-q)*log(1-p)
            device = mln.rule_weights.device if mln.rule_weights is not None else 'cpu'
            q = torch.tensor(q_prob, device=device, requires_grad=False)
            contribution = q * safe_log(p_true) + (1.0 - q) * safe_log(1.0 - p_true)
            likelihood_parts.append(contribution)
        
        # Sum all parts at the end to create the final tensor
        if likelihood_parts:
            total_pseudo_likelihood = torch.sum(torch.stack(likelihood_parts))
        else:
            total_pseudo_likelihood = torch.tensor(0.0)
            
        return total_pseudo_likelihood
    
    def _get_markov_blanket(self, target_fact: Triple, mln: MarkovLogicNetwork) -> Set[Triple]:
        """
        Get Markov blanket for a fact - all facts that appear in the same ground rules
      
        """
        markov_blanket = set()
        
        # Find all ground rules containing the target fact
        for ground_rule in mln.ground_rules:
            rule_facts = ground_rule.get_all_facts()
            if target_fact in rule_facts:
                # Add all other facts in this ground rule to Markov blanket
                markov_blanket.update(fact for fact in rule_facts if fact != target_fact)
                
                # Limit size to prevent computational explosion
                if len(markov_blanket) >= self.markov_blanket_size:
                    break
        
        return markov_blanket
    
    def _compute_conditional_probability(self,
                                       target_fact: Triple,
                                       markov_blanket: Set[Triple],
                                       mln: MarkovLogicNetwork,
                                       fact_probabilities: Dict[Triple, float]) -> torch.Tensor:
        """Expected feature-difference log-odds approximation; no hard 0/1 blanket."""
        device = mln.rule_weights.device if mln.rule_weights is not None else 'cpu'
        if mln.rule_weights is None:
            return torch.tensor(0.5, device=device)

        delta = torch.zeros((), device=device, requires_grad=True)
        rule_id_to_idx = {rule.rule_id: i for i, rule in enumerate(mln.logical_rules)}
        relevant_ground_rules = getattr(mln, 'fact_to_groundrules', {}).get(target_fact, [])
        if not relevant_ground_rules:
            relevant_ground_rules = [gr for gr in mln.ground_rules if target_fact in gr.get_all_facts()]

        def q_true(f: Triple) -> torch.Tensor:
            if f in fact_probabilities:
                return torch.tensor(fact_probabilities[f], device=device)
            if mln.knowledge_graph and f in mln.knowledge_graph.known_facts:
                return torch.tensor(1.0, device=device)
            return torch.tensor(0.5, device=device)

        for gr in relevant_ground_rules:
            w = mln.rule_weights[rule_id_to_idx[gr.parent_rule.rule_id]]
            if gr.body_facts:
                p_body_true = torch.stack([q_true(f) for f in gr.body_facts]).prod()
            else:
                p_body_true = torch.tensor(1.0, device=device)

            p_head_true = q_true(gr.head_fact)
            p_head_false = 1.0 - p_head_true

            if gr.head_fact == target_fact:
                delta = delta + w * p_body_true
            elif target_fact in gr.body_facts:
                other_body = [f for f in gr.body_facts if f != target_fact]
                p_other_body_true = (torch.stack([q_true(f) for f in other_body]).prod()
                                     if other_body else torch.tensor(1.0, device=device))
                delta = delta - w * (p_other_body_true * p_head_false)
            else:
                pass

        delta = torch.clamp(delta, -40.0, 40.0)
        return torch.sigmoid(delta)
    
    def _compute_local_potential(self,
                               target_fact: Triple,
                               assignment: Dict[Triple, bool],
                               mln: MarkovLogicNetwork) -> torch.Tensor:
        """
        Compute local potential for assignment involving target fact

        """
        # Start with a zero tensor that is on the same device as the weights
        local_potential = torch.tensor(0.0, device=mln.rule_weights.device)
        
        if mln.rule_weights is None:
            return local_potential
        
        # Find all ground rules containing the target fact
        relevant_ground_rules = []
        for ground_rule in mln.ground_rules:
            if target_fact in ground_rule.get_all_facts():
                relevant_ground_rules.append(ground_rule)
        
        # Create a mapping from rule_id to its index in the weights tensor
        rule_id_to_idx = {rule.rule_id: i for i, rule in enumerate(mln.logical_rules)}
        
        for ground_rule in relevant_ground_rules:
            # Check if ground rule is satisfied
            body_satisfied = all(
                assignment.get(fact, False)
                for fact in ground_rule.body_facts
            )
            
            rule_satisfied = False
            if not body_satisfied:
                # Body false -> rule vacuously true
                rule_satisfied = True
            else:
                # Body true -> check head
                rule_satisfied = assignment.get(ground_rule.head_fact, False)

            # Add contribution to potential
            if rule_satisfied:
                rule_idx = rule_id_to_idx.get(ground_rule.parent_rule.rule_id)
                if rule_idx is not None:
                    # Add the weight *tensor* to the potential, preserving the graph
                    local_potential = local_potential + mln.rule_weights[rule_idx]
        
        return local_potential


class GradientComputer:
    """
    Computes gradients for rule weight optimization

    """
    
    def __init__(self, config: NPLLConfig):
        self.config = config
        self.pseudo_likelihood_computer = PseudoLikelihoodComputer(config)
    
    def compute_rule_weight_gradients(self,
                                    mln: MarkovLogicNetwork,
                                    fact_probabilities: Dict[Triple, float]) -> torch.Tensor:
        """
        Compute gradients of pseudo-likelihood with respect to rule weights
        """
        if mln.rule_weights is None or not fact_probabilities:
            return torch.zeros(len(mln.logical_rules))
        
        # Enable gradient computation
        if mln.rule_weights is not None:
            mln.rule_weights.requires_grad_(True)
        
        # Compute pseudo-likelihood
        pseudo_likelihood = self.pseudo_likelihood_computer.compute_pseudo_likelihood(
            mln, fact_probabilities
        )
        
        # Compute gradients
        pseudo_likelihood.backward()
        
        if mln.rule_weights.grad is not None:
            gradients = mln.rule_weights.grad.clone()
        else:
            gradients = torch.zeros_like(mln.rule_weights)
        
        return gradients
    
    def compute_finite_difference_gradients(self,
                                          mln: MarkovLogicNetwork,
                                          fact_probabilities: Dict[Triple, float],
                                          epsilon: float = 1e-5) -> torch.Tensor:
        """
        Compute gradients using finite differences (for verification/debugging)
        """
        if mln.rule_weights is None:
            return torch.zeros(len(mln.logical_rules))
        
        gradients = torch.zeros_like(mln.rule_weights)
        original_weights = mln.rule_weights.data.clone()
        
        for i in range(len(mln.rule_weights)):
            # Forward difference
            mln.rule_weights.data[i] += epsilon
            pseudo_likelihood_plus = self.pseudo_likelihood_computer.compute_pseudo_likelihood(
                mln, fact_probabilities
            )
            
            # Backward difference  
            mln.rule_weights.data[i] -= 2 * epsilon
            pseudo_likelihood_minus = self.pseudo_likelihood_computer.compute_pseudo_likelihood(
                mln, fact_probabilities
            )
            
            # Compute gradient
            gradients[i] = (pseudo_likelihood_plus - pseudo_likelihood_minus) / (2 * epsilon)
            
            # Restore original weight
            mln.rule_weights.data[i] = original_weights[i]
        
        return gradients


class MStepOptimizer:
    """
    Main M-step optimizer that updates rule weights ω
   
    """
    
    def __init__(self, config: NPLLConfig):
        self.config = config
        self.pseudo_likelihood_computer = PseudoLikelihoodComputer(config)
        self.gradient_computer = GradientComputer(config)
        
        # Optimization parameters
        self.learning_rate = config.learning_rate
        self.max_iterations = 100  # M-step specific iterations
        self.convergence_threshold = config.convergence_threshold
        self.grad_clip_norm = config.grad_clip_norm
    
    def optimize_rule_weights(self,
                            mln: MarkovLogicNetwork,
                            e_step_result: EStepResult) -> MStepResult:
        """
        Main M-step optimization procedure

        """
        if mln.rule_weights is None:
            logger.warning("No rule weights to optimize in M-step")
            return MStepResult(
                updated_rule_weights=torch.tensor([]),
                weight_changes=torch.tensor([]),
                gradient_norms=torch.tensor([]),
                pseudo_likelihood=torch.tensor(0.0),
                optimization_history=[],
                convergence_info={'converged': True, 'reason': 'no_weights'},
                iteration_count=0
            )
        
        logger.debug(f"Starting M-step optimization for {len(mln.rule_weights)} rule weights")
        
        # Store initial weights
        initial_weights = mln.rule_weights.data.clone()
        
        # Setup optimizer
        optimizer = optim.Adam([mln.rule_weights], lr=self.learning_rate)
        
        # Optimization history
        objective_history = []
        gradient_history = []
        
        converged = False
        iteration = 0
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Ensure gradients are enabled for rule weights
            if mln.rule_weights is not None:
                mln.rule_weights.requires_grad_(True)
            
            # Compute pseudo-likelihood objective
            pseudo_likelihood = self.pseudo_likelihood_computer.compute_pseudo_likelihood(
                mln, e_step_result.fact_probabilities
            )
            
            # We want to maximize pseudo-likelihood, so minimize negative
            loss = -pseudo_likelihood
            
            # Backward pass
            loss.backward()
            
            # Store objective value (detached to prevent gradient issues)
            objective_history.append(pseudo_likelihood.detach().item())
            
            # Clip gradients
            if mln.rule_weights.grad is not None:
                grad_norm = gradient_clipping([mln.rule_weights], self.grad_clip_norm)
                gradient_history.append(grad_norm)
            else:
                gradient_history.append(0.0)
            
            # Optimization step
            optimizer.step()
            
            # Check convergence
            if iteration > 0:
                objective_change = abs(objective_history[-1] - objective_history[-2])
                if objective_change < self.convergence_threshold:
                    converged = True
                    logger.debug(f"M-step converged at iteration {iteration}")
                    break
        
        # Compute final metrics
        final_weights = mln.rule_weights.data.clone()
        weight_changes = final_weights - initial_weights
        
        final_pseudo_likelihood = self.pseudo_likelihood_computer.compute_pseudo_likelihood(
            mln, e_step_result.fact_probabilities
        )
        
        convergence_info = {
            'converged': converged,
            'final_objective_change': abs(objective_history[-1] - objective_history[-2]) if len(objective_history) > 1 else 0.0,
            'objective_history': objective_history,
            'gradient_history': gradient_history,
            'reason': 'converged' if converged else 'max_iterations'
        }
        
        result = MStepResult(
            updated_rule_weights=final_weights,
            weight_changes=weight_changes,
            gradient_norms=torch.tensor(gradient_history),
            pseudo_likelihood=final_pseudo_likelihood,
            optimization_history=objective_history,
            convergence_info=convergence_info,
            iteration_count=iteration + 1
        )
        
        logger.debug(f"M-step completed: {result}")
        
        return result
    
    def optimize_with_regularization(self,
                                   mln: MarkovLogicNetwork,
                                   e_step_result: EStepResult,
                                   l1_weight: float = 0.01,
                                   l2_weight: float = 0.01) -> MStepResult:
        """
        M-step optimization with L1 and L2 regularization on rule weights
        
        Helps prevent overfitting and encourages sparse rule sets
        """
        if mln.rule_weights is None:
            return self.optimize_rule_weights(mln, e_step_result)
        
        initial_weights = mln.rule_weights.data.clone()
        optimizer = optim.Adam([mln.rule_weights], lr=self.learning_rate)
        
        objective_history = []
        converged = False
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Primary pseudo-likelihood objective
            pseudo_likelihood = self.pseudo_likelihood_computer.compute_pseudo_likelihood(
                mln, e_step_result.fact_probabilities
            )
            
            # Regularization terms
            l1_reg = torch.sum(torch.abs(mln.rule_weights))
            l2_reg = torch.sum(mln.rule_weights ** 2)
            
            # Total objective (maximize pseudo-likelihood, minimize regularization)
            total_objective = pseudo_likelihood - l1_weight * l1_reg - l2_weight * l2_reg
            loss = -total_objective
            
            # Optimization
            loss.backward()
            gradient_clipping([mln.rule_weights], self.grad_clip_norm)
            optimizer.step()
            
            objective_history.append(total_objective.item())
            
            # Convergence check
            if iteration > 0:
                if abs(objective_history[-1] - objective_history[-2]) < self.convergence_threshold:
                    converged = True
                    break
        
        final_weights = mln.rule_weights.data.clone()
        weight_changes = final_weights - initial_weights
        
        return MStepResult(
            updated_rule_weights=final_weights,
            weight_changes=weight_changes,
            gradient_norms=torch.tensor([0.0]),  # Placeholder
            pseudo_likelihood=pseudo_likelihood,
            optimization_history=objective_history,
            convergence_info={'converged': converged},
            iteration_count=len(objective_history)
        )


class MStepRunner:
    """
    High-level runner for M-step computations
    Handles different optimization strategies and result aggregation
    """
    
    def __init__(self, config: NPLLConfig):
        self.config = config
        self.m_step_optimizer = MStepOptimizer(config)
    
    def run_m_step(self,
                   mln: MarkovLogicNetwork,
                   e_step_result: EStepResult,
                   optimization_strategy: str = "adam") -> MStepResult:
        """
        Run complete M-step for rule weight optimization

        """
        logger.info(f"Running M-step with {optimization_strategy} optimization")
        
        if optimization_strategy == "regularized":
            result = self.m_step_optimizer.optimize_with_regularization(
                mln, e_step_result, l1_weight=0.01, l2_weight=0.01
            )
        else:
            # Standard optimization
            result = self.m_step_optimizer.optimize_rule_weights(mln, e_step_result)
        
        logger.info(f"M-step completed: Pseudo-likelihood={result.pseudo_likelihood.item():.4f}, "
                   f"Max weight change={torch.max(torch.abs(result.weight_changes)).item():.6f}")
        
        return result
    
    def run_m_step_with_constraints(self,
                                   mln: MarkovLogicNetwork,
                                   e_step_result: EStepResult,
                                   weight_bounds: Optional[List[Tuple[float, float]]] = None) -> MStepResult:
        """
        Run M-step with constraints on rule weights
        
        """
        # Standard M-step
        result = self.run_m_step(mln, e_step_result)
        
        # Apply weight constraints
        if weight_bounds and mln.rule_weights is not None:
            with torch.no_grad():
                for i, (min_weight, max_weight) in enumerate(weight_bounds):
                    if i < len(mln.rule_weights):
                        mln.rule_weights[i] = torch.clamp(
                            mln.rule_weights[i], min_weight, max_weight
                        )
                
                # Update result with constrained weights
                result.updated_rule_weights = mln.rule_weights.data.clone()
        
        return result


def create_m_step_runner(config: NPLLConfig) -> MStepRunner:
    """Factory function to create M-step runner"""
    return MStepRunner(config)


def verify_m_step_implementation():
    """Verify M-step implementation"""
    from ..utils.config import default_config
    from ..core import load_knowledge_graph_from_triples
    from ..core.mln import create_mln_from_kg_and_rules
    from ..core.logical_rules import Variable, Atom, RuleType, LogicalRule
    from ..scoring import create_scoring_module
    from .e_step import create_e_step_runner
    
    # Create test data
    test_triples = [
        ("A", "r1", "B"),
        ("B", "r2", "C"),
    ]
    
    kg = load_knowledge_graph_from_triples(test_triples)
    kg.add_unknown_fact("A", "r3", "C")
    
    # Create test rule
    from ..core import Relation
    r1, r2, r3 = Relation("r1"), Relation("r2"), Relation("r3")
    x, y, z = Variable('x'), Variable('y'), Variable('z')
    
    test_rule = LogicalRule(
        rule_id="test_m_step_rule",
        body=[Atom(r1, (x, y)), Atom(r2, (y, z))],
        head=Atom(r3, (x, z)),
        rule_type=RuleType.TRANSITIVITY
    )
    
    # Create MLN and scoring module
    mln = create_mln_from_kg_and_rules(kg, [test_rule], default_config)
    scoring_module = create_scoring_module(default_config, kg)
    
    # Run E-step to get Q(U)
    e_step_runner = create_e_step_runner(default_config)
    e_step_result = e_step_runner.run_e_step(mln, scoring_module, kg)
    
    # Test M-step
    m_step_runner = MStepRunner(default_config)
    m_step_result = m_step_runner.run_m_step(mln, e_step_result)
    
    # Verify results
    assert len(m_step_result.updated_rule_weights) == len(mln.logical_rules), \
        "Should have weights for all rules"
    
    assert torch.all(torch.isfinite(m_step_result.updated_rule_weights)), \
        "All weights should be finite"
    
    assert torch.isfinite(m_step_result.pseudo_likelihood), \
        "Pseudo-likelihood should be finite"
    
    logger.info("M-step implementation verified successfully")
    
    return True
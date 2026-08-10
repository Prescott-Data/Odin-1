"""
ELBO (Evidence Lower Bound) computation for NPLL
Exact implementation of Equations 3-5 from the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from ..core import Triple, LogicalRule, GroundRule, KnowledgeGraph
from ..core.mln import MarkovLogicNetwork
from ..utils.config import NPLLConfig
from ..utils.math_utils import (
    safe_log, bernoulli_log_prob, kl_divergence_bernoulli, 
    log_sum_exp, compute_elbo_loss
)
from ..utils.batch_utils import GroundRuleBatch, GroundRuleSampler

logger = logging.getLogger(__name__)


@dataclass
class ELBOComponents:
    """
    Components of ELBO computation for analysis and debugging
    """
    joint_term: torch.Tensor  # Σ_U Q(U) log P(F,U|ω)
    entropy_term: torch.Tensor  # -Σ_U Q(U) log Q(U)
    elbo: torch.Tensor  # Total ELBO
    kl_divergence: torch.Tensor  # KL(Q||P) if available
    num_samples: int  # Number of samples used
    
    def __str__(self) -> str:
        return (f"ELBO Components:\n"
                f"  Joint term: {self.joint_term.item():.6f}\n"
                f"  Entropy term: {self.entropy_term.item():.6f}\n" 
                f"  Total ELBO: {self.elbo.item():.6f}\n"
                f"  Samples: {self.num_samples}")


class ELBOComputer(nn.Module):
    """
    Computes Evidence Lower Bound (ELBO) for NPLL training
    
    Paper Section 4: The optimization objective becomes maximizing the ELBO value
    
    Implements exact computation of Equations 3-5:
    - Equation 3: log P(F|ω) = log[P(F,U|ω)/Q(U)] - log[P(U|F,ω)/Q(U)]
    - Equation 4: log P(F|ω) = ELBO + KL(q||p)
    - Equation 5: ELBO = Σ_U Q(U) log P(F,U|ω) - Σ_U Q(U) log Q(U)
    """
    
    def __init__(self, config: NPLLConfig):
        super().__init__()
        self.config = config
        self.elbo_weight = config.elbo_weight
        self.kl_weight = config.kl_weight
    
    def compute_elbo(self, 
                    mln: MarkovLogicNetwork,
                    known_facts: List[Triple],
                    unknown_facts: List[Triple],
                    approximate_posterior_probs: torch.Tensor,
                    ground_rule_batches: Optional[List[GroundRuleBatch]] = None) -> ELBOComponents:
        """
        Compute ELBO following Equation 5
        """
        if len(unknown_facts) == 0:
            # No unknown facts, ELBO is just log P(F|ω)
            log_known_prob = self._compute_known_facts_probability(mln, known_facts)
            zero = self._scalar_like(0.0, log_known_prob)
            return ELBOComponents(
                joint_term=log_known_prob,
                entropy_term=zero,
                elbo=log_known_prob,
                kl_divergence=zero,
                num_samples=0
            )
        
        # Sample possible worlds for ELBO computation
        sampled_worlds = self._sample_possible_worlds(
            known_facts, unknown_facts, approximate_posterior_probs
        )
        
        if not sampled_worlds:
            logger.warning("No sampled worlds for ELBO computation")
            zero = self._scalar_like(0.0, approximate_posterior_probs)
            return ELBOComponents(
                joint_term=zero,
                entropy_term=zero,
                elbo=zero,
                kl_divergence=zero,
                num_samples=0
            )
        
        # Compute joint term: E_Q[log P(F,U|ω)] via MC average (do NOT weight by Q again)
        joint_term = self._compute_joint_term(mln, sampled_worlds, propagate_grads=False)
        
        # Compute entropy term: -Σ_U Q(U) log Q(U)
        entropy_term = self._compute_entropy_term(approximate_posterior_probs)
        
        # Total ELBO
        elbo = joint_term + entropy_term
        
        # Optional: compute KL divergence if true posterior is available
        kl_div = torch.tensor(0.0)          
        return ELBOComponents(
            joint_term=joint_term,
            entropy_term=entropy_term,
            elbo=elbo,
            kl_divergence=kl_div,
            num_samples=len(sampled_worlds)
        )
    
    def _compute_joint_term(self, 
                           mln: MarkovLogicNetwork,
                           sampled_worlds: List[Dict[Triple, bool]],
                           propagate_grads: bool = False) -> torch.Tensor:
        """
        Monte-Carlo estimate of E_Q[log P(F,U|ω)] as a simple average over worlds.
        If propagate_grads=True, allow gradients to flow to MLN weights.
        """
        if not sampled_worlds:
            ref = next(mln.parameters()).detach() if any(mln.parameters()) else None
            return self._scalar_like(0.0, ref)

        log_joint = []
        for world_assignment in sampled_worlds:
            logp = mln.compute_joint_probability(world_assignment, detach_weights=not propagate_grads)
            log_joint.append(logp)
        log_joint_tensor = torch.stack(log_joint)
        return log_joint_tensor.mean()
    
    def _compute_entropy_term(self, q_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy term: -Σ_U Q(U) log Q(U)
        
        This is the second term in Equation 5
        """
        # Clamp probabilities to avoid log(0)
        q_probs_clamped = torch.clamp(q_probs, min=1e-8, max=1.0 - 1e-8)
        
        # Bernoulli entropy: -[p*log(p) + (1-p)*log(1-p)]
        entropy = -(q_probs_clamped * safe_log(q_probs_clamped) + 
                   (1 - q_probs_clamped) * safe_log(1 - q_probs_clamped))
        
        # Sum over all unknown facts
        total_entropy = torch.sum(entropy)
        return total_entropy
    
    def _sample_possible_worlds(self, 
                               known_facts: List[Triple],
                               unknown_facts: List[Triple],
                               q_probs: torch.Tensor,
                               num_samples: int = None) -> List[Dict[Triple, bool]]:
        """Sample worlds over unknown facts only, preserving order; vectorized draws."""
        M = len(unknown_facts)
        if M == 0:
            return []
        num_samples = num_samples or min(100, 2 ** min(M, 10))
        device = q_probs.device
        probs = q_probs.unsqueeze(0).expand(num_samples, -1)
        samples = torch.bernoulli(probs).bool()
        worlds: List[Dict[Triple, bool]] = []
        for s in samples:
            w = {f: True for f in known_facts}
            for i, fact in enumerate(unknown_facts):
                w[fact] = bool(s[i].item())
            worlds.append(w)
        return worlds
    
    def _compute_world_q_probability(self, 
                                   world_assignment: Dict[Triple, bool],
                                   q_probs: torch.Tensor,
                                   unknown_facts: Optional[List[Triple]] = None) -> torch.Tensor:
        """
        Compute Q(U) probability for a specific world assignment
        
        Q(U) = ∏_{uk∈U} p_k^{uk} (1-p_k)^{1-uk}
        """
        ref = q_probs
        log_prob = self._scalar_like(0.0, ref)
        facts = unknown_facts if unknown_facts is not None else [f for f in world_assignment.keys() if f not in []]
        for i, fact in enumerate(facts):
            if i >= len(q_probs):
                break
            prob = torch.clamp(q_probs[i], min=1e-8, max=1.0 - 1e-8)
            truth_value = world_assignment[fact]
            log_prob = log_prob + (safe_log(prob) if truth_value else safe_log(1 - prob))
        return torch.exp(log_prob)
    
    def _compute_known_facts_probability(self, 
                                       mln: MarkovLogicNetwork,
                                       known_facts: List[Triple]) -> torch.Tensor:
        """Compute log P(F|ω) when no unknown facts"""
        if not known_facts:
            return torch.tensor(0.0)
        
        # Create assignment with all known facts as true
        known_assignment = {fact: True for fact in known_facts}
        
        # Compute joint probability
        log_prob = mln.compute_joint_probability(known_assignment)
        
        return log_prob
    
    def compute_elbo_gradient(self,
                            mln: MarkovLogicNetwork,
                            known_facts: List[Triple],
                            unknown_facts: List[Triple],
                            q_probs: torch.Tensor) -> torch.Tensor:
        """Compute ∂ELBO/∂ω allowing gradients to MLN weights."""
        if mln.rule_weights is not None:
            mln.rule_weights.requires_grad_(True)
            if mln.rule_weights.grad is not None:
                mln.rule_weights.grad.zero_()

        sampled_worlds = self._sample_possible_worlds(known_facts, unknown_facts, q_probs)
        joint_term = self._compute_joint_term(mln, sampled_worlds, propagate_grads=True)
        entropy_term = self._compute_entropy_term(q_probs)
        elbo = joint_term + entropy_term
        loss = -elbo
        loss.backward()

        if mln.rule_weights is not None and mln.rule_weights.grad is not None:
            gradients = mln.rule_weights.grad.clone()
        else:
            ref = next(mln.parameters()).detach() if any(mln.parameters()) else None
            zeros = torch.zeros(len(mln.logical_rules)) if ref is None else torch.zeros(len(mln.logical_rules), device=ref.device, dtype=ref.dtype)
            gradients = zeros
        return gradients


class ELBOLoss(nn.Module):
    """
    ELBO-based loss function for NPLL training
    
    Implements the loss function that maximizes ELBO (minimizes negative ELBO)
    """
    
    def __init__(self, config: NPLLConfig):
        super().__init__()
        self.config = config
        self.elbo_computer = ELBOComputer(config)
        
        # Loss weights
        self.elbo_weight = config.elbo_weight
        self.regularization_weight = 0.01
    
    def forward(self,
                mln: MarkovLogicNetwork,
                known_facts: List[Triple],
                unknown_facts: List[Triple],
                predicted_probs: torch.Tensor,
                target_probs: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss
        
        """
        # Compute ELBO components
        elbo_components = self.elbo_computer.compute_elbo(
            mln, known_facts, unknown_facts, predicted_probs
        )
        
        # Primary loss: negative ELBO (we want to maximize ELBO)
        elbo_loss = -elbo_components.elbo
        
        # Optional supervised loss if targets provided
        supervised_loss = torch.tensor(0.0)
        if target_probs is not None:
            supervised_loss = F.binary_cross_entropy(predicted_probs, target_probs)
        
        # Regularization loss on rule weights
        regularization_loss = torch.tensor(0.0)
        if mln.rule_weights is not None:
            regularization_loss = torch.sum(mln.rule_weights ** 2)
        
        # Total loss
        total_loss = (self.elbo_weight * elbo_loss + 
                      supervised_loss +
                      self.regularization_weight * regularization_loss)
        
        return {
            'total_loss': total_loss,
            'elbo_loss': elbo_loss,
            'supervised_loss': supervised_loss,
            'regularization_loss': regularization_loss,
            'elbo_components': elbo_components
        }


class VariationalInference:
    """
    Variational inference for NPLL using ELBO optimization
    """
    
    def __init__(self, config: NPLLConfig):
        self.config = config
        self.elbo_computer = ELBOComputer(config)
        self.convergence_threshold = config.convergence_threshold
        self.max_iterations = config.em_iterations
    
    def optimize_approximate_posterior(self,
                                     mln: MarkovLogicNetwork,
                                     known_facts: List[Triple],
                                     unknown_facts: List[Triple],
                                     initial_probs: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Optimize approximate posterior Q(U) to maximize ELBO
        
        This implements the variational inference component of the E-step

        """
        num_unknown = len(unknown_facts)
        if num_unknown == 0:
            return {
                'optimized_probs': torch.tensor([]),
                'elbo_history': [],
                'converged': True,
                'iterations': 0
            }
        
        # Initialize logits phi (unconstrained)
        if initial_probs is None:
            phi = torch.zeros(num_unknown)
        else:
            with torch.no_grad():
                init = torch.clamp(initial_probs, 1e-6, 1 - 1e-6)
                phi = torch.log(init / (1 - init))
        device = mln.rule_weights.device if (mln.rule_weights is not None) else torch.device('cpu')
        phi = phi.to(device)
        phi.requires_grad_(True)

        optimizer = torch.optim.Adam([phi], lr=0.01)
        elbo_history: List[float] = []
        prev_elbo = float('-inf')

        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            q_probs = torch.sigmoid(phi)
            elbo_components = self.elbo_computer.compute_elbo(
                mln, known_facts, unknown_facts, q_probs
            )
            loss = -elbo_components.elbo
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_([phi], max_norm=1.0)
            optimizer.step()
            
            # Detach to prevent gradient accumulation issues
            current_elbo = elbo_components.elbo.detach().item()
            elbo_history.append(current_elbo)
            if abs(current_elbo - prev_elbo) < self.convergence_threshold:
                logger.debug(f"Variational inference converged at iteration {iteration}")
                break
            prev_elbo = current_elbo

        return {
            'optimized_probs': torch.sigmoid(phi).detach(),
            'elbo_history': elbo_history,
            'converged': len(elbo_history) < self.max_iterations,
            'iterations': len(elbo_history),
            'final_elbo': elbo_history[-1] if elbo_history else float('-inf')
        }

    def _scalar_like(self, value: float, ref: Optional[torch.Tensor]) -> torch.Tensor:
        """Create a scalar tensor on the same device/dtype as ref (if provided)."""
        if ref is None:
            return torch.tensor(value)
        return torch.tensor(value, device=ref.device, dtype=ref.dtype)


def create_elbo_computer(config: NPLLConfig) -> ELBOComputer:
    """Factory function to create ELBO computer"""
    return ELBOComputer(config)


def verify_elbo_implementation():
    """Verify ELBO computation implementation"""
    from ..utils.config import default_config
    from ..core import load_knowledge_graph_from_triples
    from ..core.mln import create_mln_from_kg_and_rules
    from ..core.logical_rules import Variable, Atom, RuleType, LogicalRule
    
    # Create test data
    test_triples = [
        ("A", "r1", "B"),
        ("B", "r2", "C")
    ]
    
    kg = load_knowledge_graph_from_triples(test_triples)
    
    # Create test rule
    from ..core import Relation
    r1, r2, r3 = Relation("r1"), Relation("r2"), Relation("r3") 
    x, y, z = Variable('x'), Variable('y'), Variable('z')
    
    test_rule = LogicalRule(
        rule_id="test_elbo_rule",
        body=[Atom(r1, (x, y)), Atom(r2, (y, z))],
        head=Atom(r3, (x, z)),
        rule_type=RuleType.TRANSITIVITY
    )
    
    # Create MLN
    mln = create_mln_from_kg_and_rules(kg, [test_rule], default_config)
    
    # Test ELBO computation
    elbo_computer = ELBOComputer(default_config)
    
    known_facts = list(kg.known_facts)
    unknown_facts = [list(kg.known_facts)[0]]  # Treat one known fact as unknown for testing
    q_probs = torch.tensor([0.8])  # High probability for the "unknown" fact
    
    elbo_components = elbo_computer.compute_elbo(
        mln, known_facts[:-1], unknown_facts, q_probs
    )
    
    # Verify components
    assert torch.isfinite(elbo_components.elbo), "ELBO should be finite"
    assert torch.isfinite(elbo_components.joint_term), "Joint term should be finite"
    assert torch.isfinite(elbo_components.entropy_term), "Entropy term should be finite"
    
    # Test variational inference
    vi = VariationalInference(default_config)
    vi_result = vi.optimize_approximate_posterior(mln, known_facts[:-1], unknown_facts)
    
    assert len(vi_result['optimized_probs']) == len(unknown_facts), "Should optimize all unknown facts"
    assert len(vi_result['elbo_history']) > 0, "Should have ELBO history"
    
    logger.info("ELBO implementation verified successfully")
    
    return True
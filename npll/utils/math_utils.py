"""
Mathematical utilities for NPLL implementation
Implements key mathematical functions from the paper
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import math
import logging

logger = logging.getLogger(__name__)


def log_sum_exp(values: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Numerically stable log-sum-exp operation
    Used for partition function computation in MLN (Equation 2)
    
    Args:
        values: Input tensor
        dim: Dimension to sum over
        keepdim: Whether to keep dimension
        
    Returns:
        log(sum(exp(values))) computed stably
    """
    max_val, _ = values.max(dim=dim, keepdim=True)
    
    # Handle case where all values are -inf
    max_val = torch.where(torch.isfinite(max_val), max_val, torch.zeros_like(max_val))
    
    # Compute log-sum-exp
    result = max_val + torch.log(torch.sum(torch.exp(values - max_val), dim=dim, keepdim=True))
    
    if not keepdim:
        result = result.squeeze(dim)
    
    return result


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Safe logarithm that clamps input to avoid log(0)
    Used throughout NPLL for probability computations
    """
    return torch.log(torch.clamp(x, min=eps))


def safe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable sigmoid function
    Used in scoring module probability transformation (Section 4.1)
    """
    return torch.sigmoid(torch.clamp(x, min=-50, max=50))


def partition_function_approximation(rule_weights: torch.Tensor,
                                   ground_rule_counts: torch.Tensor,
                                   use_log_domain: bool = True) -> torch.Tensor:
    """
    Approximate MLN partition function Z(ω) from Equation 2
    
    Paper Equation 2: Z(ω) = Σ_{F,U} ∏_{r∈R} exp(ωr * N(F,U))
    
    For large knowledge graphs, exact computation is intractable,
    so we use sampling-based approximation
    
    Args:
        rule_weights: Tensor of shape [num_rules] - rule weights ω
        ground_rule_counts: Tensor of shape [num_samples, num_rules] - N(F,U) values
        use_log_domain: Whether to compute in log domain for stability
        
    Returns:
        Approximation of partition function
    """
    if use_log_domain:
        # Compute in log domain: log(Z) = log_sum_exp(Σ_r ωr * N(F,U))
        log_potentials = torch.sum(rule_weights.unsqueeze(0) * ground_rule_counts, dim=1)
        log_partition = log_sum_exp(log_potentials, dim=0)
        return log_partition
    else:
        # Direct computation (less stable)
        potentials = torch.exp(torch.sum(rule_weights.unsqueeze(0) * ground_rule_counts, dim=1))
        return torch.sum(potentials)


def compute_mln_probability(rule_weights: torch.Tensor,
                           ground_rule_counts: torch.Tensor,
                           log_partition: torch.Tensor) -> torch.Tensor:
    """
    Compute MLN probability P(F,U|ω) from Equation 1
    
    Paper Equation 1: P(F,U|ω) = (1/Z(ω)) * ∏_{r∈R} exp(ωr * N(F,U))
    
    Args:
        rule_weights: Tensor of shape [num_rules] - ω values
        ground_rule_counts: Tensor of shape [batch_size, num_rules] - N(F,U) values  
        log_partition: Log partition function log(Z(ω))
        
    Returns:
        Log probabilities of shape [batch_size]
    """
    # Compute log potential: Σ_r ωr * N(F,U)
    log_potentials = torch.sum(rule_weights.unsqueeze(0) * ground_rule_counts, dim=1)
    
    # Subtract log partition function
    log_probabilities = log_potentials - log_partition
    
    return log_probabilities


def compute_elbo_loss(predicted_probs: torch.Tensor,
                     approximate_posterior: torch.Tensor,
                     rule_weights: torch.Tensor,
                     ground_rule_counts: torch.Tensor,
                     log_partition: torch.Tensor) -> torch.Tensor:
    """
    Compute ELBO loss from Equation 5
    
    Paper Equation 5: ELBO = Σ_U Q(U) log P(F,U|ω) - Σ_U Q(U) log Q(U)
    
    Args:
        predicted_probs: Predicted fact probabilities from scoring module
        approximate_posterior: Q(U) distribution
        rule_weights: MLN rule weights ω
        ground_rule_counts: Ground rule satisfaction counts N(F,U)
        log_partition: Log partition function
        
    Returns:
        Negative ELBO loss (to minimize)
    """
    # First term: Σ_U Q(U) log P(F,U|ω)
    log_joint_probs = compute_mln_probability(rule_weights, ground_rule_counts, log_partition)
    joint_term = torch.sum(approximate_posterior * log_joint_probs)
    
    # Second term: -Σ_U Q(U) log Q(U) (entropy)
    entropy_term = -torch.sum(approximate_posterior * safe_log(approximate_posterior))
    
    elbo = joint_term + entropy_term
    
    # Return negative ELBO as loss (we want to maximize ELBO)
    return -elbo


def bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of Bernoulli distribution
    Used in E-step for Q(U) entropy computation
    
    H(p) = -p*log(p) - (1-p)*log(1-p)
    """
    return -(p * safe_log(p) + (1 - p) * safe_log(1 - p))


def bernoulli_log_prob(value: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """
    Compute log probability of Bernoulli distribution
    Used for fact probability computations in E-step
    
    log P(x=value) = value*log(p) + (1-value)*log(1-p)
    """
    return value * safe_log(prob) + (1 - value) * safe_log(1 - prob)


def compute_markov_blanket_prob(fact_prob: torch.Tensor,
                               neighbor_probs: torch.Tensor,
                               rule_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute probability of fact given Markov blanket
    Used in M-step pseudo-likelihood computation (Equation 13)
    
    Args:
        fact_prob: Probability of target fact
        neighbor_probs: Probabilities of facts in Markov blanket
        rule_weights: Weights of rules involving this fact
        
    Returns:
        P(uk | Markov Blanket)
    """
    # Simplified computation - in practice this involves complex inference
    # over the local Markov network structure
    
    # Compute local potential based on neighboring facts and rule weights
    local_potential = torch.sum(rule_weights * neighbor_probs)
    
    # Normalize using sigmoid
    return torch.sigmoid(local_potential + torch.logit(fact_prob))


def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply temperature scaling for confidence calibration
    Used in paper for calibrating confidence estimates
    
    Args:
        logits: Input logits
        temperature: Temperature parameter (1.0 = no scaling)
        
    Returns:
        Temperature-scaled probabilities
    """
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=-1)


def kl_divergence_bernoulli(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between two Bernoulli distributions
    KL(p||q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    
    Used for measuring distance between true and approximate posteriors
    """
    eps = 1e-8
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    
    kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    return kl


def gradient_clipping(parameters: List[torch.nn.Parameter], 
                     max_norm: float = 1.0) -> float:
    """
    Clip gradients to prevent exploding gradients
    Returns the total norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return total_norm.item()


def compute_metrics(predictions: torch.Tensor, 
                   targets: torch.Tensor,
                   k_values: List[int] = [1, 3, 10]) -> Dict[str, float]:
    """
    Compute evaluation metrics as specified in paper Section 5.2
    
    Args:
        predictions: Predicted scores/ranks [batch_size, num_entities]
        targets: Target entity indices [batch_size]
        k_values: Values of k for Hit@k computation
        
    Returns:
        Dictionary with MRR and Hit@k metrics
    """
    batch_size = predictions.size(0)
    
    # Get ranks of target entities
    sorted_indices = torch.argsort(predictions, dim=1, descending=True)
    ranks = torch.zeros(batch_size, dtype=torch.float)
    
    for i in range(batch_size):
        target_idx = targets[i]
        rank = (sorted_indices[i] == target_idx).nonzero(as_tuple=True)[0][0] + 1
        ranks[i] = rank.float()
    
    # Compute MRR
    mrr = torch.mean(1.0 / ranks).item()
    
    # Compute Hit@k
    metrics = {'MRR': mrr}
    for k in k_values:
        hit_at_k = torch.mean((ranks <= k).float()).item()
        metrics[f'Hit@{k}'] = hit_at_k
    
    return metrics


def moving_average(values: List[float], window_size: int = 10) -> float:
    """Compute moving average of values"""
    if len(values) < window_size:
        return sum(values) / len(values) if values else 0.0
    return sum(values[-window_size:]) / window_size


def cosine_similarity(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute cosine similarity between vectors"""
    return F.cosine_similarity(x, y, dim=dim)


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance between vectors"""
    return torch.norm(x - y, p=2, dim=-1)


def sample_negative_triples(positive_triples: torch.Tensor,
                          num_entities: int,
                          corruption_mode: str = 'both') -> torch.Tensor:
    """
    Sample negative triples for training by corrupting positive ones
    
    Args:
        positive_triples: Tensor of shape [batch_size, 3] (head, relation, tail)
        num_entities: Total number of entities
        corruption_mode: 'head', 'tail', or 'both'
        
    Returns:
        Negative triples tensor
    """
    batch_size = positive_triples.size(0)
    negative_triples = positive_triples.clone()
    
    for i in range(batch_size):
        if corruption_mode == 'head' or (corruption_mode == 'both' and i % 2 == 0):
            # Corrupt head entity
            negative_triples[i, 0] = torch.randint(0, num_entities, (1,))
        else:
            # Corrupt tail entity
            negative_triples[i, 2] = torch.randint(0, num_entities, (1,))
    
    return negative_triples


class NumericalStabilizer:
    """Utility class for numerical stability in computations"""
    
    @staticmethod
    def stabilize_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Numerically stable softmax"""
        x_max, _ = torch.max(x, dim=dim, keepdim=True)
        exp_x = torch.exp(x - x_max)
        return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    
    @staticmethod
    def stabilize_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Numerically stable log-softmax"""
        return F.log_softmax(x, dim=dim)
    
    @staticmethod
    def clamp_probabilities(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Clamp probabilities to valid range"""
        return torch.clamp(p, min=eps, max=1.0 - eps)


# Constants for numerical stability
EPS = 1e-8
LOG_EPS = math.log(EPS)
LARGE_NUMBER = 1e8
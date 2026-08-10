
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import logging
import numpy as np

from ..utils.math_utils import safe_sigmoid, safe_log, bernoulli_log_prob, bernoulli_entropy
from ..core import Triple

logger = logging.getLogger(__name__)


class ProbabilityTransform(nn.Module):
    """
    Temperature scaling with optional per-group (e.g., per-relation) temperatures.
    Guarantees T>0 via softplus on an unconstrained log-T parameter.
    """

    def __init__(self, temperature: float = 1.0, per_group: bool = False, num_groups: int = 1):
        super().__init__()
        self.per_group = per_group
        init = torch.log(torch.expm1(torch.tensor(float(temperature))))
        if per_group:
            assert num_groups >= 1
            self.log_t = nn.Parameter(init.repeat(int(num_groups)))
        else:
            self.log_t = nn.Parameter(init.unsqueeze(0))

    def _temperature(self, device, group_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        T = F.softplus(self.log_t.to(device)) + 1e-6
        if self.per_group:
            if group_ids is None:
                raise ValueError("group_ids is required when per_group=True")
            return T.index_select(0, group_ids)
        return T[0]

    def ensure_num_groups(self, num_groups: int):
        if not self.per_group:
            return
        cur = self.log_t.numel()
        if num_groups <= cur:
            return
        with torch.no_grad():
            new = torch.empty(num_groups, device=self.log_t.device, dtype=self.log_t.dtype)
            new[:cur] = self.log_t.data
            init = torch.log(torch.expm1(torch.tensor(1.0, device=new.device, dtype=new.dtype)))
            new[cur:] = init
        self.log_t = nn.Parameter(new)

    def forward(self, scores: torch.Tensor,
                apply_temperature: bool = True,
                group_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        scaled = scores if not apply_temperature else scores / self._temperature(scores.device, group_ids)
        return torch.sigmoid(scaled)

    def log_probs(self, scores: torch.Tensor,
                  apply_temperature: bool = True,
                  group_ids: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled = scores if not apply_temperature else scores / self._temperature(scores.device, group_ids)
        return F.logsigmoid(scaled), F.logsigmoid(-scaled)

    # Backward-compat alias
    def get_log_probabilities(self, scores: torch.Tensor, apply_temperature: bool = True,
                              group_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        log_p, _ = self.log_probs(scores, apply_temperature, group_ids)
        return log_p

    @torch.no_grad()
    def calibrate_temperature(self,
                              logits: torch.Tensor,  # raw scores before sigmoid
                              labels: torch.Tensor,  # {0,1}
                              max_iter: int = 100,
                              weight: Optional[torch.Tensor] = None,
                              group_ids: Optional[torch.LongTensor] = None) -> float:
        """
        Temperature scaling on held-out logits using BCEWithLogitsLoss.
        Optimizes log-T only.
        """
        logits = logits.detach()
        labels = labels.float().detach()
        if weight is not None:
            weight = weight.to(logits.device)

        optimizer = torch.optim.LBFGS([self.log_t], max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            # Always divide by T (scalar for global, vector for per-group)
            if self.per_group:
                T = self._temperature(logits.device, group_ids)
            else:
                T = self._temperature(logits.device, None)
            scaled = logits / T
            loss = F.binary_cross_entropy_with_logits(scaled, labels, weight=weight)
            loss.backward()
            return loss

        optimizer.step(closure)
        T = (F.softplus(self.log_t) + 1e-6).detach().cpu()
        logger.info(f"Temperature calibrated; mean T={T.mean().item():.4f}")
        return T.mean().item()


class FactProbabilityComputer:
    """
    Computes probabilities for facts in the context of NPLL
    Handles both known facts F and unknown facts U
    """
    
    def __init__(self, probability_transform: ProbabilityTransform):
        self.prob_transform = probability_transform
    
    def compute_fact_probabilities(self, scores: torch.Tensor, 
                                 fact_types: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute probabilities and stable log-probabilities for facts from scores.
        """
        probabilities = self.prob_transform(scores)
        log_p, log1m_p = self.prob_transform.log_probs(scores)
        result = {
            'probabilities': probabilities,
            'log_probabilities': log_p,
            'neg_log_probabilities': log1m_p
        }
        if fact_types is not None:
            device = probabilities.device
            known_mask = torch.tensor([ft == 'known' for ft in fact_types], device=device, dtype=torch.bool)
            unknown_mask = ~known_mask
            if torch.any(known_mask):
                result['known_probabilities'] = probabilities[known_mask]
            if torch.any(unknown_mask):
                result['unknown_probabilities'] = probabilities[unknown_mask]
        return result
    
    def compute_bernoulli_parameters(self, scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Bernoulli distribution parameters for facts
        """
        probabilities = self.prob_transform(scores)
        log_p, log1m_p = self.prob_transform.log_probs(scores)
        return {
            'success_prob': probabilities,
            'failure_prob': 1 - probabilities,
            'log_success_prob': log_p,
            'log_failure_prob': log1m_p,
            'entropy': bernoulli_entropy(probabilities)
        }


class ApproximatePosteriorComputer:
    """
    Computes approximate posterior distribution Q(U) as described in paper Section 4.2
    
    """
    
    def __init__(self, probability_transform: ProbabilityTransform):
        self.prob_transform = probability_transform
        self.fact_prob_computer = FactProbabilityComputer(probability_transform)
    
    def compute_q_u_distribution(self, unknown_fact_scores: torch.Tensor,
                               ground_rule_structure: Optional[List[List[int]]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute approximate posterior distribution Q(U)
        
        """
        # Get Bernoulli parameters for unknown facts
        bernoulli_params = self.fact_prob_computer.compute_bernoulli_parameters(unknown_fact_scores)
        
        result = {
            'fact_probabilities': bernoulli_params['success_prob'],
            'fact_log_probabilities': bernoulli_params['log_success_prob'], 
            'fact_entropies': bernoulli_params['entropy']
        }
        
        # If ground rule structure provided, compute ground rule probabilities
        if ground_rule_structure is not None:
            ground_rule_probs = []
            ground_rule_log_probs = []
            
            for rule_fact_indices in ground_rule_structure:
                if rule_fact_indices:
                    # Product of fact probabilities in this ground rule
                    rule_fact_probs = bernoulli_params['success_prob'][rule_fact_indices]
                    rule_prob = torch.prod(rule_fact_probs)
                    rule_log_prob = torch.sum(bernoulli_params['log_success_prob'][rule_fact_indices])
                    
                    ground_rule_probs.append(rule_prob)
                    ground_rule_log_probs.append(rule_log_prob)
            
            if ground_rule_probs:
                result['ground_rule_probabilities'] = torch.stack(ground_rule_probs)
                result['ground_rule_log_probabilities'] = torch.stack(ground_rule_log_probs)
        
        return result
    
    def compute_expected_counts(self, fact_probabilities: torch.Tensor,
                              ground_rule_structure: List[List[int]]) -> torch.Tensor:
        """
        Compute expected counts N(F,U) for ground rules
        
        """
        expected_counts = []
        
        for rule_fact_indices in ground_rule_structure:
            if rule_fact_indices:
                # Expected count for this ground rule is product of fact probabilities
                rule_fact_probs = fact_probabilities[rule_fact_indices]
                expected_count = torch.prod(rule_fact_probs)
                expected_counts.append(expected_count)
            else:
                expected_counts.append(torch.tensor(0.0, device=fact_probabilities.device))
        
        return torch.stack(expected_counts) if expected_counts else torch.empty(0, device=fact_probabilities.device)


class ProbabilityCalibrator:
    """
    Post-hoc probability calibration using various methods
    Improves reliability of confidence estimates
    """
    
    def __init__(self, method: str = 'platt'):
        """
        Args:
            method: Calibration method ('platt', 'isotonic', 'temperature')
        """
        self.method = method
        self.calibration_function = None
        self.is_fitted = False
    
    def fit(self, predicted_probs: np.ndarray, true_labels: np.ndarray):
        """
        Fit calibration function to data
        """
        if self.method == 'platt':
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
            
            # Platt scaling using logistic regression
            self.calibration_function = LogisticRegression()
            self.calibration_function.fit(predicted_probs.reshape(-1, 1), true_labels)
            
        elif self.method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            
            self.calibration_function = IsotonicRegression(out_of_bounds='clip')
            self.calibration_function.fit(predicted_probs, true_labels)
            
        elif self.method == 'temperature':
            # Temperature scaling (implemented in ProbabilityTransform)
            pass
        
        self.is_fitted = True
        logger.info(f"Calibration function fitted using {self.method}")
    
    def transform(self, predicted_probs: np.ndarray) -> np.ndarray:
        """Apply calibration to predicted probabilities"""
        if not self.is_fitted:
            logger.warning("Calibration function not fitted, returning original probabilities")
            return predicted_probs
        
        if self.method == 'platt':
            return self.calibration_function.predict_proba(predicted_probs.reshape(-1, 1))[:, 1]
        elif self.method == 'isotonic':
            return self.calibration_function.transform(predicted_probs)
        else:
            return predicted_probs
    
    def compute_calibration_error(self, predicted_probs: np.ndarray, 
                                true_labels: np.ndarray, 
                                n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)
        
        ECE measures the difference between confidence and accuracy
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_samples = len(predicted_probs)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            prop_in_bin = in_bin.sum() / total_samples
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = predicted_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class ConfidenceEstimator:
    """
    Estimates confidence scores for logical rules using scoring module outputs
    Integrates with NPLL's probabilistic framework
    """
    
    def __init__(self, probability_transform: ProbabilityTransform):
        self.prob_transform = probability_transform
        self.posterior_computer = ApproximatePosteriorComputer(probability_transform)
    
    def estimate_rule_confidence(self, rule_satisfaction_scores: torch.Tensor,
                                supporting_evidence_scores: torch.Tensor) -> Dict[str, float]:
        """
        Estimate confidence for a logical rule based on satisfaction and evidence
        """
        # Convert scores to probabilities
        satisfaction_probs = self.prob_transform(rule_satisfaction_scores)
        evidence_probs = self.prob_transform(supporting_evidence_scores)
        
        # Compute various confidence metrics
        confidence_metrics = {
            'mean_satisfaction': satisfaction_probs.mean().item(),
            'median_satisfaction': satisfaction_probs.median().item(),
            'min_satisfaction': satisfaction_probs.min().item(),
            'max_satisfaction': satisfaction_probs.max().item(),
            'std_satisfaction': satisfaction_probs.std().item(),
            'mean_evidence': evidence_probs.mean().item(),
            'evidence_strength': evidence_probs.sum().item(),
            'num_supporting_instances': (evidence_probs > 0.5).sum().item()
        }
        
        # Combined confidence score (weighted average)
        combined_confidence = (
            0.7 * confidence_metrics['mean_satisfaction'] + 
            0.3 * confidence_metrics['mean_evidence']
        )
        confidence_metrics['combined_confidence'] = combined_confidence
        
        return confidence_metrics
    
    def compute_uncertainty_measures(self, probabilities: torch.Tensor) -> Dict[str, float]:
        """
        Compute various uncertainty measures for probability estimates
        
        Returns:
            Dictionary with uncertainty metrics
        """
        # Entropy-based uncertainty
        entropy = bernoulli_entropy(probabilities).mean().item()
        
        # Variance-based uncertainty  
        variance = (probabilities * (1 - probabilities)).mean().item()
        
        # Confidence intervals (assuming independence)
        confidence_95 = 1.96 * torch.sqrt(probabilities * (1 - probabilities))
        mean_ci_width = confidence_95.mean().item()
        
        return {
            'entropy': entropy,
            'variance': variance, 
            'mean_ci_width_95': mean_ci_width,
            'prediction_uncertainty': entropy,  # Alias for compatibility
        }


def create_probability_components(temperature: float = 1.0,
                                  per_relation: bool = False,
                                  num_relations: int = 1) -> Tuple[ProbabilityTransform, ApproximatePosteriorComputer]:
    """
    Factory function to create probability computation components.
    Supports optional per-relation temperature scaling.
    """
    prob_transform = ProbabilityTransform(temperature=temperature,
                                          per_group=per_relation,
                                          num_groups=(num_relations if per_relation else 1))
    posterior_computer = ApproximatePosteriorComputer(prob_transform)
    return prob_transform, posterior_computer


def verify_probability_computations():
    """Verify probability computation implementations"""
    # Test probability transform
    prob_transform = ProbabilityTransform(temperature=1.0)
    
    # Test scores
    test_scores = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    probs = prob_transform(test_scores, apply_temperature=False)
    
    # Verify sigmoid properties
    assert torch.all(probs >= 0) and torch.all(probs <= 1), "Probabilities not in [0,1]"
    assert abs(probs[2].item() - 0.5) < 1e-5, "sigmoid(0) should be 0.5"
    
    # Test Bernoulli computations
    fact_computer = FactProbabilityComputer(prob_transform)
    bernoulli_params = fact_computer.compute_bernoulli_parameters(test_scores)
    
    # Verify Bernoulli properties
    success_prob = bernoulli_params['success_prob']
    failure_prob = bernoulli_params['failure_prob'] 
    assert torch.allclose(success_prob + failure_prob, torch.ones_like(success_prob)), \
        "Success + failure probabilities should sum to 1"
    
    logger.info("Probability computation implementations verified successfully")
    
    return True
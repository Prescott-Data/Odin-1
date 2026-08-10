"""
Main NPLL Model - Neural Probabilistic Logic Learning
Integrates all components: MLN, Scoring Module, E-step, M-step
Exact implementation of the complete NPLL framework from the paper
"""

import torch
import torch.nn as nn
from typing import List, Dict, Set, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import time
import os

from .core import (
    KnowledgeGraph, Entity, Relation, Triple, LogicalRule, GroundRule,
    MarkovLogicNetwork, create_mln_from_kg_and_rules
)
from .scoring import (
    NPLLScoringModule, create_scoring_module, 
    ProbabilityTransform, create_probability_components
)
from .inference import (
    EStepRunner, MStepRunner, ELBOComputer, EStepResult, MStepResult,
    create_e_step_runner, create_m_step_runner
)
from .utils import NPLLConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class NPLLTrainingState:
    """
    Training state for NPLL model
    Tracks progress through E-M iterations
    """
    epoch: int
    em_iteration: int
    elbo_history: List[float]
    rule_weight_history: List[List[float]]
    convergence_info: Dict[str, Any]
    training_time: float
    best_elbo: float
    best_weights: Optional[torch.Tensor] = None
    
    def __str__(self) -> str:
        return (f"NPLL Training State:\n"
                f"  Epoch: {self.epoch}\n"
                f"  EM Iteration: {self.em_iteration}\n"
                f"  Current ELBO: {self.elbo_history[-1] if self.elbo_history else 'N/A'}\n"
                f"  Best ELBO: {self.best_elbo:.6f}\n"
                f"  Training Time: {self.training_time:.2f}s")


class NPLLModel(nn.Module):
    """
    Complete Neural Probabilistic Logic Learning Model
    
    Integrates all paper components:
    - Knowledge Graph representation (Section 3)
    - Scoring Module with bilinear function (Section 4.1, Equation 7)
    - Markov Logic Network (Sections 3-4, Equations 1-2)
    - E-M Algorithm (Sections 4.2-4.3, Equations 8-14)
    - ELBO optimization (Equations 3-5)
    """
    
    def __init__(self, config: NPLLConfig):
        super().__init__()
        self.config = config
        
        # Core components (initialized when knowledge graph is provided)
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.mln: Optional[MarkovLogicNetwork] = None
        self.scoring_module: Optional[NPLLScoringModule] = None
        
        # Inference components
        self.e_step_runner: Optional[EStepRunner] = None
        self.m_step_runner: Optional[MStepRunner] = None
        self.elbo_computer: Optional[ELBOComputer] = None
        
        # Probability transformation
        self.probability_transform: Optional[ProbabilityTransform] = None
        
        # Training state
        self.training_state: Optional[NPLLTrainingState] = None
        self.is_initialized = False
        
        # Model metadata
        self.model_version = "1.0"
        self.creation_time = time.time()
        self.calibration_version = None
    
    def initialize(self, 
                  knowledge_graph: KnowledgeGraph,
                  logical_rules: List[LogicalRule]):
        """
        Initialize NPLL model with knowledge graph and rules
        
        Args:
            knowledge_graph: Knowledge graph K = (E, L, F)
            logical_rules: List of logical rules R
        """
        logger.info("Initializing NPLL model...")
        
        # Store knowledge graph
        self.knowledge_graph = knowledge_graph
        
        # Create and initialize MLN
        self.mln = create_mln_from_kg_and_rules(knowledge_graph, logical_rules, self.config)
        
        # Create scoring module
        self.scoring_module = create_scoring_module(self.config, knowledge_graph)
        
        # Create inference components
        self.e_step_runner = create_e_step_runner(self.config)
        self.m_step_runner = create_m_step_runner(self.config)
        
        # Create ELBO computer
        from .inference import create_elbo_computer
        self.elbo_computer = create_elbo_computer(self.config)
        
        # Create probability transformation (enable per-relation groups if available)
        num_rel = None
        if self.scoring_module is not None and hasattr(self.scoring_module, 'embedding_manager'):
            emb_mgr = self.scoring_module.embedding_manager
            if hasattr(emb_mgr, 'relation_num_groups'):
                num_rel = emb_mgr.relation_num_groups
        per_relation = num_rel is not None
        prob_transform, _ = create_probability_components(
            self.config.temperature,
            per_relation=per_relation,
            num_relations=(num_rel or 1)
        )
        self.probability_transform = prob_transform
        
        # Initialize training state
        self.training_state = NPLLTrainingState(
            epoch=0,
            em_iteration=0,
            elbo_history=[],
            rule_weight_history=[],
            convergence_info={},
            training_time=0.0,
            best_elbo=float('-inf')
        )
        
        self.is_initialized = True
        
        logger.info(f"NPLL model initialized with {len(logical_rules)} rules, "
                   f"{len(knowledge_graph.entities)} entities, "
                   f"{len(knowledge_graph.relations)} relations")
    
    def forward(self, triples: List[Triple]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through NPLL model
        
        Args:
            triples: List of triples to score
            
        Returns:
            Dictionary with scores and probabilities
        """
        if not self.is_initialized:
            raise RuntimeError("NPLL model not initialized. Call initialize() first.")
        
        # Get raw scores from scoring module
        raw_scores = self.scoring_module.forward(triples)
        
        # Group IDs per relation (no vocab mutation)
        group_ids = None
        if hasattr(self.scoring_module, 'embedding_manager') and getattr(self.probability_transform, 'per_group', False):
            emb_mgr = self.scoring_module.embedding_manager
            group_ids = emb_mgr.relation_group_ids_for_triples(triples, add_if_missing=False)
            # Ensure transform capacity if table grew
            if hasattr(self.probability_transform, 'ensure_num_groups'):
                self.probability_transform.ensure_num_groups(emb_mgr.relation_num_groups)
        
        # Transform to probabilities
        probabilities = self.probability_transform(raw_scores, apply_temperature=True, group_ids=group_ids)
        
        # Get log probabilities
        log_probabilities = self.probability_transform.get_log_probabilities(raw_scores, apply_temperature=True, group_ids=group_ids)
        
        return {
            'raw_scores': raw_scores,
            'probabilities': probabilities,
            'log_probabilities': log_probabilities
        }
    
    def predict_single_triple(self, head: str, relation: str, tail: str, transient: bool = True) -> Dict[str, float]:
        """
        Predict probability for a single triple
        
        Args:
            head: Head entity name
            relation: Relation name  
            tail: Tail entity name
            transient: If True, do not mutate the underlying knowledge graph
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_initialized:
            raise RuntimeError("NPLL model not initialized")
        
        # Create triple object without mutating KG by default
        if transient:
            head_entity = Entity(head)
            relation_obj = Relation(relation)
            tail_entity = Entity(tail)
        else:
            head_entity = self.knowledge_graph.get_entity(head) or self.knowledge_graph.add_entity(head)
            relation_obj = self.knowledge_graph.get_relation(relation) or self.knowledge_graph.add_relation(relation)
            tail_entity = self.knowledge_graph.get_entity(tail) or self.knowledge_graph.add_entity(tail)
        
        triple = Triple(head=head_entity, relation=relation_obj, tail=tail_entity)
        
        # Get predictions
        self.eval()
        with torch.no_grad():
            results = self.forward([triple])
        
        return {
            'probability': results['probabilities'][0].item(),
            'log_probability': results['log_probabilities'][0].item(),
            'raw_score': results['raw_scores'][0].item()
        }
    
    def run_single_em_iteration(self) -> Dict[str, Any]:
        """
        Run a single E-M iteration
        
        Returns:
            Dictionary with iteration results
        """
        if not self.is_initialized:
            raise RuntimeError("NPLL model not initialized")
        
        iteration_start_time = time.time()
        
        logger.debug(f"Starting E-M iteration {self.training_state.em_iteration}")
        
        # E-step: Optimize Q(U)
        logger.debug("Running E-step...")
        e_step_result = self.e_step_runner.run_e_step(
            self.mln, self.scoring_module, self.knowledge_graph
        )
        
        # M-step: Optimize rule weights ω
        logger.debug("Running M-step...")
        m_step_result = self.m_step_runner.run_m_step(self.mln, e_step_result)
        
        # Update training state
        current_elbo = e_step_result.elbo_value.item()
        self.training_state.elbo_history.append(current_elbo)
        
        if self.mln.rule_weights is not None:
            current_weights = self.mln.rule_weights.data.tolist()
            self.training_state.rule_weight_history.append(current_weights)
            
            # Track best snapshot (MLN + scoring)
            if current_elbo > self.training_state.best_elbo:
                self.training_state.best_elbo = current_elbo
                self.training_state.best_weights = {
                    'mln': {k: v.clone() for k, v in self.mln.state_dict().items()},
                    'scoring': {k: v.clone() for k, v in self.scoring_module.state_dict().items()} if self.scoring_module else {},
                }
        
        self.training_state.em_iteration += 1
        iteration_time = time.time() - iteration_start_time
        self.training_state.training_time += iteration_time
        
        # Check convergence
        converged = self._check_em_convergence()
        
        iteration_result = {
            'em_iteration': self.training_state.em_iteration - 1,
            'e_step_result': e_step_result,
            'm_step_result': m_step_result,
            'elbo': current_elbo,
            'iteration_time': iteration_time,
            'converged': converged,
            'convergence_info': {
                'e_step_converged': e_step_result.convergence_info.get('converged', False),
                'm_step_converged': m_step_result.convergence_info.get('converged', False)
            }
        }
        
        logger.debug(f"E-M iteration completed: ELBO={current_elbo:.6f}, "
                    f"Time={iteration_time:.2f}s, Converged={converged}")
        
        return iteration_result
    
    def train_epoch(self, max_em_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Train for one epoch (multiple E-M iterations until convergence)
        
        Args:
            max_em_iterations: Maximum E-M iterations per epoch
            
        Returns:
            Dictionary with epoch results
        """
        if not self.is_initialized:
            raise RuntimeError("NPLL model not initialized")
        
        max_iterations = max_em_iterations or self.config.em_iterations
        epoch_start_time = time.time()
        
        logger.info(f"Starting training epoch {self.training_state.epoch}")
        
        epoch_results = []
        converged = False
        
        for em_iter in range(max_iterations):
            iteration_result = self.run_single_em_iteration()
            epoch_results.append(iteration_result)
            
            if iteration_result['converged']:
                converged = True
                logger.info(f"Converged after {em_iter + 1} E-M iterations")
                break
        
        epoch_time = time.time() - epoch_start_time
        self.training_state.epoch += 1
        
        epoch_summary = {
            'epoch': self.training_state.epoch - 1,
            'em_iterations': len(epoch_results),
            'converged': converged,
            'final_elbo': epoch_results[-1]['elbo'] if epoch_results else float('-inf'),
            'best_elbo_this_epoch': max(r['elbo'] for r in epoch_results) if epoch_results else float('-inf'),
            'epoch_time': epoch_time,
            'iteration_results': epoch_results
        }
        
        logger.info(f"Epoch {self.training_state.epoch - 1} completed: "
                   f"ELBO={epoch_summary['final_elbo']:.6f}, "
                   f"EM iterations={epoch_summary['em_iterations']}, "
                   f"Time={epoch_time:.2f}s")
        
        return epoch_summary
    
    def _check_em_convergence(self) -> bool:
        """Check if E-M algorithm has converged with patience and relative tolerance"""
        if len(self.training_state.elbo_history) < 2:
            return False
        h = self.training_state.elbo_history
        rel = abs(h[-1] - h[-2]) / (abs(h[-2]) + 1e-8)
        elbo_ok = rel < getattr(self.config, 'elbo_rel_tol', self.config.convergence_threshold)

        weight_ok = True
        if len(self.training_state.rule_weight_history) >= 2:
            current_weights = torch.tensor(self.training_state.rule_weight_history[-1])
            prev_weights = torch.tensor(self.training_state.rule_weight_history[-2])
            weight_change = torch.norm(current_weights - prev_weights).item()
            weight_ok = weight_change < getattr(self.config, 'weight_abs_tol', self.config.convergence_threshold)

        if elbo_ok and weight_ok:
            hits = self.training_state.convergence_info.get('hits', 0) + 1
            self.training_state.convergence_info['hits'] = hits
        else:
            self.training_state.convergence_info['hits'] = 0
        patience = getattr(self.config, 'convergence_patience', 3)
        return self.training_state.convergence_info.get('hits', 0) >= patience
    
    def get_rule_confidences(self) -> Dict[str, float]:
        """Get learned confidence scores for all rules"""
        if not self.is_initialized or self.mln is None or self.mln.rule_weights is None:
            return {}
        
        confidences = {}
        for i, rule in enumerate(self.mln.logical_rules):
            # Convert rule weight to confidence using sigmoid
            weight = self.mln.rule_weights[i].item()
            confidence = torch.sigmoid(torch.tensor(weight)).item()
            confidences[rule.rule_id] = confidence
        
        return confidences
    
    def predict_unknown_facts(self, top_k: int = 10) -> List[Tuple[Triple, float]]:
        """
        Predict probabilities for unknown facts
        
        Args:
            top_k: Number of top predictions to return
            
        Returns:
            List of (triple, probability) tuples sorted by probability
        """
        if not self.is_initialized:
            raise RuntimeError("NPLL model not initialized")
        
        unknown_facts = list(self.knowledge_graph.unknown_facts)
        if not unknown_facts:
            return []
        
        # Get predictions for unknown facts
        results = self.forward(unknown_facts)
        probabilities = results['probabilities']
        
        # Create (triple, probability) pairs
        predictions = list(zip(unknown_facts, probabilities.tolist()))
        
        # Sort by probability (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_k]
    
    def save_model(self, filepath: str):
        """Save complete NPLL model state with KG and rules serialization"""
        if not self.is_initialized:
            raise RuntimeError("Cannot save uninitialized model")

        rules = self.mln.logical_rules if self.mln else []
        payload = {
            'schema_version': self.model_version,
            'config': self.config.__dict__,
            'creation_time': self.creation_time,
            'training_state': self.training_state,
            'mln_state': self.mln.state_dict() if self.mln else None,
            'scoring_state': self.scoring_module.state_dict() if self.scoring_module else None,
            'prob_state': self.probability_transform.state_dict() if hasattr(self.probability_transform, 'state_dict') else None,
            'knowledge_graph': self.knowledge_graph.serialize() if self.knowledge_graph else None,
            'rules': [rule.serialize() for rule in rules],
        }
        torch.save(payload, filepath)
        logger.info(f"NPLL model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load complete NPLL model state, rebuilding KG and rules before weights.
        
        PRODUCTION FIX: Respects device configuration (cpu/cuda) instead of hardcoding cpu.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        from .core import KnowledgeGraph, LogicalRule
        
        # Respect device config: load to configured device (cpu or cuda)
        # Note: weights_only=False required for custom objects (NPLLTrainingState, etc.)
        device = getattr(self, 'device', 'cpu')
        payload = torch.load(filepath, map_location=device, weights_only=False)
        
        # Rebuild config
        self.config = NPLLConfig(**payload['config']) if isinstance(payload.get('config'), dict) else payload['config']
        # Rebuild KG and rules
        kg = KnowledgeGraph.deserialize(payload['knowledge_graph']) if payload.get('knowledge_graph') else None
        rules = [LogicalRule.deserialize(x) for x in payload.get('rules', [])]
        # Recreate components
        self.__init__(self.config)
        if kg is not None:
            self.initialize(kg, rules)
        # Load weights
        if payload.get('mln_state') and self.mln:
            self.mln.load_state_dict(payload['mln_state'])
        if payload.get('scoring_state') and self.scoring_module:
            self.scoring_module.load_state_dict(payload['scoring_state'])
        if payload.get('prob_state') and hasattr(self.probability_transform, 'load_state_dict'):
            self.probability_transform.load_state_dict(payload['prob_state'])
        # Metadata
        self.model_version = payload.get('schema_version', '1.0')
        self.creation_time = payload.get('creation_time', time.time())
        self.training_state = payload.get('training_state', self.training_state)
        logger.info(f"NPLL model loaded from {filepath}")
    
    def save_to_buffer(self, buffer):
        """Save complete NPLL model state to a BytesIO buffer (for database storage)."""
        if not self.is_initialized:
            raise RuntimeError("Cannot save uninitialized model")

        rules = self.mln.logical_rules if self.mln else []
        payload = {
            'schema_version': self.model_version,
            'config': self.config.__dict__,
            'creation_time': self.creation_time,
            'training_state': self.training_state,
            'mln_state': self.mln.state_dict() if self.mln else None,
            'scoring_state': self.scoring_module.state_dict() if self.scoring_module else None,
            'prob_state': self.probability_transform.state_dict() if hasattr(self.probability_transform, 'state_dict') else None,
            'knowledge_graph': self.knowledge_graph.serialize() if self.knowledge_graph else None,
            'rules': [rule.serialize() for rule in rules],
        }
        torch.save(payload, buffer)
        logger.info("NPLL model saved to buffer")
    
    def load_from_buffer(self, buffer):
        """Load complete NPLL model state from a BytesIO buffer (for database storage)."""
        from .core import KnowledgeGraph, LogicalRule
        
        # Respect device config
        device = getattr(self, 'device', 'cpu')
        buffer.seek(0)
        payload = torch.load(buffer, map_location=device, weights_only=False)
        
        # Rebuild config
        self.config = NPLLConfig(**payload['config']) if isinstance(payload.get('config'), dict) else payload['config']
        # Rebuild KG and rules
        kg = KnowledgeGraph.deserialize(payload['knowledge_graph']) if payload.get('knowledge_graph') else None
        rules = [LogicalRule.deserialize(x) for x in payload.get('rules', [])]
        # Recreate components
        self.__init__(self.config)
        if kg is not None:
            self.initialize(kg, rules)
        # Load weights
        if payload.get('mln_state') and self.mln:
            self.mln.load_state_dict(payload['mln_state'])
        if payload.get('scoring_state') and self.scoring_module:
            self.scoring_module.load_state_dict(payload['scoring_state'])
        if payload.get('prob_state') and hasattr(self.probability_transform, 'load_state_dict'):
            self.probability_transform.load_state_dict(payload['prob_state'])
        # Metadata
        self.model_version = payload.get('schema_version', '1.0')
        self.creation_time = payload.get('creation_time', time.time())
        self.training_state = payload.get('training_state', self.training_state)
        logger.info("NPLL model loaded from buffer")
    
    def _get_device(self) -> str:
        for p in self.parameters():
            return str(p.device)
        return 'cpu'

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary (device-safe)"""
        summary = {
            'model_version': self.model_version,
            'is_initialized': self.is_initialized,
            'config': self.config.__dict__ if self.config else {},
            'creation_time': self.creation_time,
            'calibration_version': self.calibration_version,
            'training_state': self.training_state.__dict__ if self.training_state else {},
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'device': self._get_device(),
        }
        if self.is_initialized:
            summary.update({
                'knowledge_graph_stats': self.knowledge_graph.get_statistics() if self.knowledge_graph else {},
                'mln_stats': self.mln.get_rule_statistics() if self.mln else {},
                'rule_confidences': self.get_rule_confidences(),
            })
        return summary

    # --- Utilities: checkpoints and calibration ---
    def restore_best(self) -> bool:
        """Restore the best-scoring MLN and scoring-module weights if available."""
        if not self.training_state or not getattr(self.training_state, 'best_weights', None):
            return False
        best = self.training_state.best_weights
        if 'mln' in best and self.mln:
            self.mln.load_state_dict(best['mln'])
        if 'scoring' in best and self.scoring_module:
            self.scoring_module.load_state_dict(best['scoring'])
        return True

    def calibrate_temperature_on_data(self,
                                      triples: List[Triple],
                                      labels: torch.Tensor,
                                      max_iter: int = 100,
                                      version: Optional[str] = None) -> float:
        """
        Calibrate the ProbabilityTransform temperature on a holdout set.
        Stores the learned temperature in the transform and records a calibration version.
        Returns the optimized temperature value.
        """
        if not self.is_initialized:
            raise RuntimeError("NPLL model not initialized")
        self.eval()
        with torch.no_grad():
            scores = self.scoring_module.forward(triples)

        # Extract group_ids for calibration if per_group is enabled
        group_ids = None
        if hasattr(self.scoring_module, 'embedding_manager') and getattr(self.probability_transform, 'per_group', False):
            emb_mgr = self.scoring_module.embedding_manager
            group_ids = emb_mgr.relation_group_ids_for_triples(triples, add_if_missing=False)
            if hasattr(self.probability_transform, 'ensure_num_groups'):
                self.probability_transform.ensure_num_groups(emb_mgr.relation_num_groups)

        optimized_temp = self.probability_transform.calibrate_temperature(scores, labels.float(), max_iter=max_iter, group_ids=group_ids)
        self.calibration_version = version or f"temp@{time.time():.0f}"
        return optimized_temp
    
    def __str__(self) -> str:
        if not self.is_initialized:
            return "NPLL Model (not initialized)"
        
        stats = self.knowledge_graph.get_statistics() if self.knowledge_graph else {}
        return (f"NPLL Model:\n"
                f"  Entities: {stats.get('num_entities', 0)}\n"
                f"  Relations: {stats.get('num_relations', 0)}\n"
                f"  Known Facts: {stats.get('num_known_facts', 0)}\n"
                f"  Unknown Facts: {stats.get('num_unknown_facts', 0)}\n"
                f"  Rules: {len(self.mln.logical_rules) if self.mln else 0}\n"
                f"  Training State: {self.training_state}")


def create_npll_model(config: NPLLConfig) -> NPLLModel:
    """
    Factory function to create NPLL model
    
    Args:
        config: NPLL configuration
        
    Returns:
        Uninitialized NPLL model
    """
    return NPLLModel(config)


def create_initialized_npll_model(knowledge_graph: KnowledgeGraph,
                                 logical_rules: List[LogicalRule],
                                 config: Optional[NPLLConfig] = None) -> NPLLModel:
    """
    Factory function to create and initialize NPLL model
    
    Args:
        knowledge_graph: Knowledge graph
        logical_rules: List of logical rules
        config: Optional configuration (uses default if not provided)
        
    Returns:
        Initialized NPLL model ready for training
    """
    if config is None:
        config = get_config("ArangoDB_Triples")  # Default configuration
    
    model = NPLLModel(config)
    model.initialize(knowledge_graph, logical_rules)
    
    return model
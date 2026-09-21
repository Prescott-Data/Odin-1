"""
NPLL Training Infrastructure
Complete training loop with E-M algorithm, validation, and checkpointing
"""

import torch
import torch.nn as nn
from typing import List, Dict, Set, Tuple, Optional, Any, Union
import logging
import time
import os
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from ..npll_model import NPLLModel, NPLLTrainingState
from ..core import KnowledgeGraph, LogicalRule
from ..utils import NPLLConfig
from .evaluation import EvaluationMetrics, create_evaluator

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Configuration for NPLL training process
    """
    # Training parameters
    num_epochs: int = 100
    max_em_iterations_per_epoch: int = 20
    early_stopping_patience: int = 10
    
    # Validation
    validate_every_n_epochs: int = 5
    validation_split: float = 0.1
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 10
    keep_best_checkpoint: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_metrics_every_n_iterations: int = 5
    
    # Performance
    device: str = "cpu"  # or "cuda"
    num_workers: int = 1
    
    # Optimization
    learning_rate_schedule: bool = True
    lr_decay_factor: float = 0.9
    lr_decay_patience: int = 5


@dataclass
class TrainingResult:
    """
    Result of NPLL training process
    """
    # Training progress
    total_epochs: int
    total_em_iterations: int
    final_elbo: float
    best_elbo: float
    converged: bool
    
    # Training history
    elbo_history: List[float]
    validation_metrics_history: List[Dict[str, float]]
    
    # Timing
    total_training_time: float
    average_epoch_time: float
    
    # Model state
    final_model_path: Optional[str] = None
    best_model_path: Optional[str] = None
    
    # Convergence info
    convergence_epoch: Optional[int] = None
    early_stopping_triggered: bool = False
    # Max-abs rule weight change per E-M iteration (None entries = first iteration of a run)
    rule_weight_delta_history: List[Optional[float]] = field(default_factory=list)


class NPLLTrainer:
    """
    Complete NPLL training infrastructure
    
    Manages the full training pipeline:
    - E-M algorithm execution
    - Validation and evaluation
    - Checkpointing and model saving
    - Early stopping and convergence detection
    - Learning rate scheduling
    """
    
    def __init__(self, 
                 model: NPLLModel,
                 training_config: TrainingConfig,
                 evaluator=None):
        """
        Initialize NPLL trainer
        
        Args:
            model: NPLL model to train
            training_config: Training configuration
            evaluator: Optional evaluator for validation
        """
        self.model = model
        self.config = training_config
        self.evaluator = evaluator
        
        # Setup device
        self.device = torch.device(self.config.device)
        if self.model.is_initialized:
            self.model.to(self.device)
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.training_history = {
            'epochs': [],
            'elbo_history': [],
            'rule_weight_deltas': [],
            'validation_metrics': [],
            'learning_rates': [],
            'convergence_info': []
        }
        
        # Early stopping state
        self.best_validation_score = float('-inf')
        self.epochs_without_improvement = 0
        
        # Checkpointing
        if self.config.save_checkpoints:
            self.checkpoint_dir = Path(self.config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"NPLL Trainer initialized with config: {self.config}")
    
    def _setup_logging(self):
        """Setup training logging"""
        log_level = getattr(logging, self.config.log_level.upper())
        logging.getLogger(__name__).setLevel(log_level)
    
    def train(self, 
              validation_kg: Optional[KnowledgeGraph] = None,
              validation_rules: Optional[List[LogicalRule]] = None) -> TrainingResult:
        """
        Complete training process
        
        Args:
            validation_kg: Optional validation knowledge graph
            validation_rules: Optional validation rules
            
        Returns:
            TrainingResult with comprehensive training information
        """
        if not self.model.is_initialized:
            raise RuntimeError("Model must be initialized before training")
        
        logger.info("Starting NPLL training process")
        training_start_time = time.time()
        
        # Setup validation if provided
        validation_available = validation_kg is not None and validation_rules is not None
        if validation_available and self.evaluator is None:
            self.evaluator = create_evaluator(validation_kg)
        
        # Training loop
        converged = False
        early_stopped = False
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Train one epoch
            epoch_result = self._train_epoch(epoch)
            
            # Update training history
            self._update_training_history(epoch, epoch_result)
            
            # Validation
            validation_metrics = {}
            if validation_available and epoch % self.config.validate_every_n_epochs == 0:
                validation_metrics = self._validate(validation_kg, validation_rules)
                self.training_history['validation_metrics'].append(validation_metrics)
                
                # Early stopping check
                early_stopped = self._check_early_stopping(validation_metrics)
            
            # Checkpointing
            if self.config.save_checkpoints and epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, epoch_result, validation_metrics)
            
            # Convergence check
            converged = epoch_result['converged']
            
            # Log progress
            self._log_epoch_progress(epoch, epoch_result, validation_metrics, 
                                   time.time() - epoch_start_time)
            
            # Break conditions
            if converged:
                logger.info(f"Training converged at epoch {epoch}")
                break
            
            if early_stopped:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Training completed
        total_training_time = time.time() - training_start_time
        
        # Save final model
        final_model_path = None
        if self.config.save_checkpoints:
            final_model_path = self.checkpoint_dir / "final_model.pt"
            self.model.save_model(str(final_model_path))
        
        # Create training result
        result = self._create_training_result(
            total_epochs=epoch + 1,
            total_training_time=total_training_time,
            converged=converged,
            early_stopped=early_stopped,
            final_model_path=str(final_model_path) if final_model_path else None
        )
        
        logger.info(f"Training completed: {result}")
        return result
    
    def _train_epoch(self, epoch: int) -> Dict[str, Any]:
        """Train a single epoch"""
        logger.debug(f"Training epoch {epoch}")
        
        # Train epoch with E-M iterations
        epoch_result = self.model.train_epoch(
            max_em_iterations=self.config.max_em_iterations_per_epoch
        )
        
        return epoch_result
    
    def _validate(self, validation_kg: KnowledgeGraph, 
                  validation_rules: List[LogicalRule]) -> Dict[str, float]:
        """Run validation evaluation"""
        if self.evaluator is None:
            return {}
        
        logger.debug("Running validation evaluation")
        
        # Set model to eval mode
        self.model.eval()
        
        try:
            # Run evaluation
            metrics = self.evaluator.evaluate_link_prediction(self.model, top_k=[1, 3, 10])
            
            # Add rule quality metrics if possible
            try:
                rule_metrics = self.evaluator.evaluate_rule_quality(self.model)
                metrics.update(rule_metrics)
            except Exception as e:
                logger.debug(f"Could not evaluate rule quality: {e}")
            
            return metrics
        
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return {}
        
        finally:
            # Set model back to train mode
            self.model.train()
    
    def _check_early_stopping(self, validation_metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered"""
        if not validation_metrics:
            return False
        
        # Use MRR as primary validation metric
        current_score = validation_metrics.get('mrr', float('-inf'))
        
        if current_score > self.best_validation_score:
            self.best_validation_score = current_score
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, epoch: int, epoch_result: Dict[str, Any], 
                        validation_metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state': self.model.get_model_summary(),
            'training_history': self.training_history,
            'training_config': asdict(self.config),
            'epoch_result': epoch_result,
            'validation_metrics': validation_metrics
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save model state
        model_checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch}.pt"
        self.model.save_model(str(model_checkpoint_path))
        
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def _update_training_history(self, epoch: int, epoch_result: Dict[str, Any]):
        """Update training history"""
        self.training_history['epochs'].append(epoch)
        self.training_history['elbo_history'].extend(
            [r['elbo'] for r in epoch_result['iteration_results']]
        )
        self.training_history['rule_weight_deltas'].extend(
            [r.get('rule_weight_delta') for r in epoch_result['iteration_results']]
        )
        self.training_history['convergence_info'].append({
            'epoch': epoch,
            'converged': epoch_result['converged'],
            'em_iterations': epoch_result['em_iterations'],
            'final_elbo': epoch_result['final_elbo']
        })
    
    def _log_epoch_progress(self, epoch: int, epoch_result: Dict[str, Any],
                           validation_metrics: Dict[str, float], epoch_time: float):
        """Log training progress"""
        elbo = epoch_result['final_elbo']
        em_iters = epoch_result['em_iterations']
        converged = epoch_result['converged']
        
        log_msg = (f"Epoch {epoch}: ELBO={elbo:.6f}, EM_iters={em_iters}, "
                  f"Converged={converged}, Time={epoch_time:.2f}s")
        
        if validation_metrics:
            mrr = validation_metrics.get('mrr', 0.0)
            hit1 = validation_metrics.get('hit@1', 0.0)
            log_msg += f", Val_MRR={mrr:.4f}, Val_Hit@1={hit1:.4f}"
        
        logger.info(log_msg)
    
    def _create_training_result(self, total_epochs: int, total_training_time: float,
                               converged: bool, early_stopped: bool,
                               final_model_path: Optional[str]) -> TrainingResult:
        """Create comprehensive training result"""
        
        # Get total EM iterations
        total_em_iterations = sum(
            info['em_iterations'] for info in self.training_history['convergence_info']
        )
        
        # Get final and best ELBO
        final_elbo = self.training_history['elbo_history'][-1] if self.training_history['elbo_history'] else float('-inf')
        best_elbo = max(self.training_history['elbo_history']) if self.training_history['elbo_history'] else float('-inf')
        
        # Find convergence epoch
        convergence_epoch = None
        for info in self.training_history['convergence_info']:
            if info['converged']:
                convergence_epoch = info['epoch']
                break
        
        return TrainingResult(
            total_epochs=total_epochs,
            total_em_iterations=total_em_iterations,
            final_elbo=final_elbo,
            best_elbo=best_elbo,
            converged=converged,
            elbo_history=self.training_history['elbo_history'],
            validation_metrics_history=self.training_history['validation_metrics'],
            total_training_time=total_training_time,
            average_epoch_time=total_training_time / total_epochs if total_epochs > 0 else 0.0,
            final_model_path=final_model_path,
            convergence_epoch=convergence_epoch,
            early_stopping_triggered=early_stopped,
            rule_weight_delta_history=self.training_history['rule_weight_deltas']
        )
    
    def resume_training(self, checkpoint_path: str) -> TrainingResult:
        """Resume training from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path)
        
        # Restore training history
        self.training_history = checkpoint_data['training_history']
        
        # Resume training from next epoch
        start_epoch = checkpoint_data['epoch'] + 1
        
        # Continue training
        # (This would require modifying the train method to accept start_epoch)
        logger.info(f"Training resumed from epoch {start_epoch}")
        
        return self.train()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'config': asdict(self.config),
            'model_summary': self.model.get_model_summary(),
            'training_history': self.training_history,
            'best_validation_score': self.best_validation_score,
            'epochs_without_improvement': self.epochs_without_improvement
        }


def create_trainer(model: NPLLModel, 
                  training_config: Optional[TrainingConfig] = None,
                  validation_kg: Optional[KnowledgeGraph] = None) -> NPLLTrainer:
    """
    Factory function to create NPLL trainer
    
    Args:
        model: NPLL model to train
        training_config: Optional training configuration
        validation_kg: Optional validation knowledge graph for evaluator
        
    Returns:
        Configured NPLL trainer
    """
    if training_config is None:
        training_config = TrainingConfig()
    
    # Create evaluator if validation data provided
    evaluator = None
    if validation_kg is not None:
        evaluator = create_evaluator(validation_kg)
    
    return NPLLTrainer(model, training_config, evaluator)


def train_npll_from_scratch(knowledge_graph: KnowledgeGraph,
                           logical_rules: List[LogicalRule],
                           npll_config: Optional[NPLLConfig] = None,
                           training_config: Optional[TrainingConfig] = None) -> Tuple[NPLLModel, TrainingResult]:
    """
    Complete training pipeline from scratch
    
    Args:
        knowledge_graph: Knowledge graph for training
        logical_rules: Logical rules
        npll_config: NPLL model configuration
        training_config: Training configuration
        
    Returns:
        (Trained model, Training result)
    """
    from ..npll_model import create_npll_model
    from ..utils import get_config
    
    # Create model
    if npll_config is None:
        npll_config = get_config("ArangoDB_Triples")
    
    model = create_npll_model(npll_config)
    model.initialize(knowledge_graph, logical_rules)
    
    # Create trainer
    trainer = create_trainer(model, training_config)
    
    # Train
    result = trainer.train()
    
    return model, result


# Example usage function
def example_training_pipeline():
    """
    Example showing complete training pipeline with sample data
    """
    from ..core import load_knowledge_graph_from_triples
    from ..core.logical_rules import RuleGenerator
    from ..utils import get_config
    
    # 1. Create sample data (your data adapter would provide this format)
    sample_triples = [
        ('Alice', 'friendOf', 'Bob'),
        ('Bob', 'worksAt', 'Company'),
        ('Charlie', 'friendOf', 'Alice'),
        ('Bob', 'livesIn', 'NYC'),
        ('Alice', 'livesIn', 'NYC'),
        ('Company', 'locatedIn', 'NYC')
    ]
    
    # Load knowledge graph
    kg = load_knowledge_graph_from_triples(sample_triples, "Sample KG")
    
    # Generate rules
    rule_generator = RuleGenerator(kg)
    rules = rule_generator.generate_simple_rules(min_support=1)
    rules.extend(rule_generator.generate_symmetry_rules(min_support=1))
    
    # 2. Configure training
    npll_config = get_config("ArangoDB_Triples")
    training_config = TrainingConfig(
        num_epochs=10,
        max_em_iterations_per_epoch=5,
        early_stopping_patience=3,
        validate_every_n_epochs=2
    )
    
    # 3. Train model
    model, result = train_npll_from_scratch(kg, rules, npll_config, training_config)
    
    # 4. Results
    print(f"Training completed: {result}")
    print(f"Final model: {model}")
    
    return model, result


if __name__ == "__main__":
    example_training_pipeline()
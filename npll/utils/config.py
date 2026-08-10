"""
Configuration for Neural Probabilistic Logic Learning (NPLL)
Hyperparameters and settings based on the paper specifications
"""

from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class NPLLConfig:
    """Configuration class for NPLL implementation following paper specifications"""
    
    # Entity and Relation Embedding Dimensions (Paper Section 4.1)
    entity_embedding_dim: int = 256  # d-dimensional entity embeddings
    relation_embedding_dim: int = 256  # d-dimensional relation embeddings
    rule_embedding_dim: int = 512  # k-dimensional rule embeddings
    
    # Scoring Module Parameters (Equation 7)
    # g(l, eh, et) = u^T_R f(e^T_h W_R et + V_R [eh; et] + b_R)
    scoring_hidden_dim: int = 512  # k dimension for scoring function
    scoring_activation: str = "relu"  # Non-linear activation function f
    
    # MLN Parameters (Equations 1-2)
    max_rule_length: int = 3  # Maximum atoms per rule premise
    max_ground_rules: int = 1000  # Maximum ground rules per batch
    temperature: float = 1.0  # Temperature scaling for calibration
    
    # Training Hyperparameters (Paper Section 5)
    learning_rate: float = 0.0005  # Initial learning rate from paper
    batch_size: int = 128  # Batch size for ground rule sampling
    max_epochs: int = 100  # Maximum training epochs
    patience: int = 20  # Early stopping patience
    
    # E-M Algorithm Parameters (Sections 4.2-4.3)
    em_iterations: int = 10  # Number of E-M alternations per epoch
    convergence_threshold: float = 1e-4  # Convergence criterion for E-M
    # Extended convergence controls
    elbo_rel_tol: float = 1e-4  # relative ELBO tol
    weight_abs_tol: float = 1e-4  # weight change tol
    convergence_patience: int = 3  # number of consecutive hits required
    
    # Regularization and Optimization
    dropout: float = 0.1  # Dropout rate
    weight_decay: float = 0.01  # L2 regularization
    grad_clip_norm: float = 1.0  # Gradient clipping
    
    # ELBO Optimization (Equation 5)
    elbo_weight: float = 1.0  # Weight for ELBO term
    kl_weight: float = 1.0  # Weight for KL divergence term
    
    # Mean-field Approximation (Equation 8)
    mean_field_iterations: int = 5  # Iterations for mean-field convergence
    
    # Pseudo-log-likelihood (Equation 13)
    pseudo_likelihood: bool = True  # Use pseudo-likelihood in M-step
    markov_blanket_size: int = 10  # Size of Markov blanket
    
    # Device and Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True  # Pin memory for GPU
    
    # Evaluation Settings (Paper Section 5.2)
    eval_batch_size: int = 256  # Evaluation batch size
    eval_metrics: List[str] = None  # MRR, Hit@1, Hit@3, Hit@10
    filtered_evaluation: bool = True  # Filtered setting from paper
    
    # Dataset-specific Settings (Paper datasets)
    dataset_name: str = "ArangoDB_Triples"  # Default dataset
    train_ratio: float = 1.0  # Training data ratio (for data efficiency experiments)
    
    # Logging and Checkpointing
    log_interval: int = 10  # Log every N epochs
    save_interval: int = 50  # Save model every N epochs
    checkpoint_dir: str = "checkpoints/"
    
    def __post_init__(self):
        """Initialize derived configurations"""
        if self.eval_metrics is None:
            self.eval_metrics = ["MRR", "Hit@1", "Hit@3", "Hit@10"]
        
        # Ensure scoring dimensions are consistent
        assert self.scoring_hidden_dim > 0, "Scoring hidden dimension must be positive"
        assert self.entity_embedding_dim == self.relation_embedding_dim, \
            "Entity and relation embedding dimensions must match (paper assumption)"


# Paper-specific configurations for different datasets
FB15K_237_CONFIG = NPLLConfig(
    dataset_name="FB15k-237",
    entity_embedding_dim=256,
    relation_embedding_dim=256,
    rule_embedding_dim=512,
    learning_rate=0.0005,
    max_epochs=200
)

WN18RR_CONFIG = NPLLConfig(
    dataset_name="WN18RR",
    entity_embedding_dim=256,
    relation_embedding_dim=256,
    rule_embedding_dim=512,
    learning_rate=0.0005,
    max_epochs=200
)

UMLS_CONFIG = NPLLConfig(
    dataset_name="UMLS",
    entity_embedding_dim=128,
    relation_embedding_dim=128,
    rule_embedding_dim=256,
    learning_rate=0.001,
    max_epochs=100
)

KINSHIP_CONFIG = NPLLConfig(
    dataset_name="Kinship",
    entity_embedding_dim=512,
    relation_embedding_dim=512,
    rule_embedding_dim=512,
    learning_rate=0.0005,
    max_epochs=150
)


def get_config(dataset_name: str) -> NPLLConfig:
    """Get dataset-specific configuration"""
    configs = {
        "FB15k-237": FB15K_237_CONFIG,
        "WN18RR": WN18RR_CONFIG,
        "UMLS": UMLS_CONFIG,
        "Kinship": KINSHIP_CONFIG
    }
    
    if dataset_name in configs:
        return configs[dataset_name]
    else:
        print(f"Warning: Unknown dataset {dataset_name}, using default ArangoDB_Triples config")
        return FB15K_237_CONFIG


# Export default config
default_config = FB15K_237_CONFIG
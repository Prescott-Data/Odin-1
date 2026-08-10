"""
Scoring module for NPLL implementation
Implements the bilinear scoring function and probability transformations from the paper
"""

from .embeddings import (
    EntityEmbedding, RelationEmbedding, EmbeddingManager,
    create_embedding_manager, initialize_embeddings_from_pretrained
)
from .scoring_module import (
    BilinearScoringFunction, NPLLScoringModule, BatchedScoringModule,
    create_scoring_module, verify_equation7_implementation
)
from .probability import (
    ProbabilityTransform, FactProbabilityComputer, ApproximatePosteriorComputer,
    ProbabilityCalibrator, ConfidenceEstimator, create_probability_components,
    verify_probability_computations
)

__all__ = [
    # Embeddings
    'EntityEmbedding',
    'RelationEmbedding', 
    'EmbeddingManager',
    'create_embedding_manager',
    'initialize_embeddings_from_pretrained',
    
    # Scoring Module
    'BilinearScoringFunction',
    'NPLLScoringModule',
    'BatchedScoringModule', 
    'create_scoring_module',
    'verify_equation7_implementation',
    
    # Probability Components
    'ProbabilityTransform',
    'FactProbabilityComputer',
    'ApproximatePosteriorComputer',
    'ProbabilityCalibrator',
    'ConfidenceEstimator',
    'create_probability_components',
    'verify_probability_computations'
]
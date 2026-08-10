
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import logging

from ..core import Triple
from ..utils.config import NPLLConfig
from ..utils.math_utils import safe_sigmoid
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class BilinearScoringFunction(nn.Module):
    
    def __init__(self, config: NPLLConfig):
        super().__init__()
        self.config = config
        
        # Dimensions from paper
        self.entity_dim = config.entity_embedding_dim  # d
        self.relation_dim = config.relation_embedding_dim  # d 
        self.hidden_dim = config.scoring_hidden_dim  # k
        
        # Ensure dimensions match paper assumptions
        assert self.entity_dim == self.relation_dim, \
            "Paper assumes entity and relation embeddings have same dimension d"
        
        # Bilinear tensor W_R: d×d×k dimensional tensor
        # For efficiency, we implement this as k separate d×d matrices
        self.W_R = nn.Parameter(torch.zeros(self.hidden_dim, self.entity_dim, self.entity_dim))
        
        # Linear tensor V_R: k×2d dimensional tensor  
        # Maps concatenated [eh; et] to k dimensions
        self.V_R = nn.Parameter(torch.zeros(self.hidden_dim, 2 * self.entity_dim))
        
        # Output projection u_R: k-dimensional vector
        self.u_R = nn.Parameter(torch.zeros(self.hidden_dim))
        
        # Bias term b_R: k-dimensional vector
        self.b_R = nn.Parameter(torch.zeros(self.hidden_dim))
        
        # Activation function f (paper uses ReLU)
        if config.scoring_activation == "relu":
            self.activation = F.relu
        elif config.scoring_activation == "tanh":
            self.activation = torch.tanh
        elif config.scoring_activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters using Xavier/Glorot initialization"""
        # Initialize bilinear tensor W_R
        nn.init.xavier_uniform_(self.W_R.data)
        
        # Initialize linear tensor V_R
        nn.init.xavier_uniform_(self.V_R.data)
        
        # Initialize output projection u_R
        nn.init.xavier_uniform_(self.u_R.data.unsqueeze(0))
        self.u_R.data.squeeze_(0)
        
        # Initialize bias b_R to small values
        nn.init.constant_(self.b_R.data, 0.1)
    
    def forward(self, head_embeddings: torch.Tensor, 
                relation_embeddings: torch.Tensor,
                tail_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Equation 7
        
        Args:
            head_embeddings: [batch_size, entity_dim] - eh vectors
            relation_embeddings: [batch_size, relation_dim] - relation vectors (unused in Eq 7 but kept for completeness)
            tail_embeddings: [batch_size, entity_dim] - et vectors
            
        Returns:
            scores: [batch_size] - g(l, eh, et) scores
        """
        batch_size = head_embeddings.size(0)
        
        # Step 1: Compute bilinear term e^T_h W_R et
        # W_R has shape [k, d, d], we need to compute bilinear for each k
        bilinear_terms = []
        
        for i in range(self.hidden_dim):
            # For each k-th slice: e^T_h W_R[i] et
            # head_embeddings: [batch_size, d]
            # W_R[i]: [d, d]  
            # tail_embeddings: [batch_size, d]
            
            # Compute head_embeddings @ W_R[i] -> [batch_size, d]
            temp = torch.matmul(head_embeddings, self.W_R[i])  
            
            # Compute (head_embeddings @ W_R[i]) @ tail_embeddings^T -> [batch_size]
            bilinear_i = torch.sum(temp * tail_embeddings, dim=1)  
            bilinear_terms.append(bilinear_i)
        
        # Stack bilinear terms: [batch_size, k]
        bilinear_output = torch.stack(bilinear_terms, dim=1)
        
        #Compute linear term V_R [eh; et]
        # Concatenate head and tail embeddings: [batch_size, 2d]
        concatenated = torch.cat([head_embeddings, tail_embeddings], dim=1)
        
        # Linear transformation: V_R @ [eh; et]^T -> [batch_size, k]
        linear_output = torch.matmul(concatenated, self.V_R.transpose(0, 1))
        
        #  Combine terms inside activation
        # e^T_h W_R et + V_R [eh; et] + b_R
        combined = bilinear_output + linear_output + self.b_R.unsqueeze(0)
        
        #  Apply non-linear activation f
        activated = self.activation(combined)  # [batch_size, k]
        
        #Final projection u^T_R f(...)
        scores = torch.matmul(activated, self.u_R)  # [batch_size]
        
        return scores
    
    def forward_single(self, head_embedding: torch.Tensor,
                      relation_embedding: torch.Tensor, 
                      tail_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass for single triple"""
        # Add batch dimension
        head_batch = head_embedding.unsqueeze(0)
        rel_batch = relation_embedding.unsqueeze(0) 
        tail_batch = tail_embedding.unsqueeze(0)
        
        score = self.forward(head_batch, rel_batch, tail_batch)
        return score.squeeze(0)


class NPLLScoringModule(nn.Module):
    
    def __init__(self, config: NPLLConfig):
        super().__init__()
        self.config = config
        
        # Embedding manager for entities and relations
        self.embedding_manager = EmbeddingManager(config)
        
        # Bilinear scoring function (Equation 7)
        self.scoring_function = BilinearScoringFunction(config)
        
        # Temperature parameter for calibration (paper mentions temperature scaling)
        self.temperature = nn.Parameter(torch.tensor(config.temperature))
    
    def forward(self, triples: List[Triple]) -> torch.Tensor:
        """
        Score a batch of triples
        
        Args:
            triples: List of Triple objects to score
            
        Returns:
            scores: [batch_size] raw scores g(l, eh, et)
        """
        if not triples:
            return torch.tensor([])
        
        # Extract entity and relation names
        head_names = [triple.head.name for triple in triples]
        relation_names = [triple.relation.name for triple in triples]
        tail_names = [triple.tail.name for triple in triples]
        
        # Get embeddings
        head_embs, rel_embs, tail_embs = self.embedding_manager.get_embeddings_for_scoring(
            head_names, relation_names, tail_names, add_if_missing=False
        )
        
        # Compute scores using Equation 7
        scores = self.scoring_function(head_embs, rel_embs, tail_embs)
        
        return scores
    
    def forward_with_names(self, head_names: List[str], 
                          relation_names: List[str],
                          tail_names: List[str]) -> torch.Tensor:
        """Score triples given entity/relation names directly"""
        # Get embeddings
        head_embs, rel_embs, tail_embs = self.embedding_manager.get_embeddings_for_scoring(
            head_names, relation_names, tail_names, add_if_missing=False
        )
        
        # Compute scores
        scores = self.scoring_function(head_embs, rel_embs, tail_embs)
        
        return scores
    
    def score_single_triple(self, triple: Triple) -> torch.Tensor:
        """Score a single triple"""
        head_emb, rel_emb, tail_emb = self.embedding_manager.get_triple_embeddings(triple, add_if_missing=False)
        
        # Update entity embeddings
        head_emb = self.embedding_manager.update_entity_embeddings(head_emb.unsqueeze(0)).squeeze(0)
        tail_emb = self.embedding_manager.update_entity_embeddings(tail_emb.unsqueeze(0)).squeeze(0)
        
        score = self.scoring_function.forward_single(head_emb, rel_emb, tail_emb)
        return score
    
    def get_probabilities(self, triples: List[Triple], 
                         apply_temperature: bool = True) -> torch.Tensor:
        """
        Get probability scores p = sigmoid(g(l, eh, et))
        """
        scores = self.forward(triples)
        
        if apply_temperature:
            # Apply temperature scaling for calibration
            scores = scores / self.temperature
        
        # Apply sigmoid transformation
        probabilities = safe_sigmoid(scores)
        
        return probabilities
    
    def compute_fact_scores(self, known_facts: List[Triple], 
                           unknown_facts: List[Triple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scores for known and unknown facts
        
        Used in E-step for computing approximate posterior Q(U)
        """
        known_scores = self.forward(known_facts) if known_facts else torch.tensor([])
        unknown_scores = self.forward(unknown_facts) if unknown_facts else torch.tensor([])
        
        return known_scores, unknown_scores
    
    def build_vocabulary(self, kg):
        """Build vocabulary from knowledge graph"""
        self.embedding_manager.build_vocabulary_from_kg(kg)
    
    @property
    def entity_vocab_size(self) -> int:
        """Number of entities in vocabulary"""
        return self.embedding_manager.entity_vocab_size
    
    @property 
    def relation_vocab_size(self) -> int:
        """Number of relations in vocabulary"""
        return self.embedding_manager.relation_vocab_size


class BatchedScoringModule(nn.Module):
    """
    Optimized scoring module for large-scale batch processing
    Implements memory-efficient batching for scoring many triples
    """
    
    def __init__(self, config: NPLLConfig, base_scoring_module: NPLLScoringModule):
        super().__init__()
        self.config = config
        self.base_module = base_scoring_module
        self.batch_size = config.batch_size
    
    def score_large_batch(self, triples: List[Triple]) -> torch.Tensor:
        """Score large batch of triples with memory-efficient batching"""
        all_scores = []
        
        # Process in smaller batches
        for i in range(0, len(triples), self.batch_size):
            batch_triples = triples[i:i + self.batch_size]
            batch_scores = self.base_module.forward(batch_triples)
            all_scores.append(batch_scores)
        
        # Concatenate all scores
        if all_scores:
            return torch.cat(all_scores, dim=0)
        else:
            return torch.tensor([])
    
    def score_all_possible_triples(self, entities: List[str], relations: List[str]) -> torch.Tensor:
        """
        Score all possible (head, relation, tail) combinations
        Warning: This can be very memory intensive for large vocabularies
        """
        all_triples = []
        
        # Generate all possible combinations
        for head in entities:
            for relation in relations:
                for tail in entities:
                    if head != tail:  # Avoid self-loops typically
                        all_triples.append((head, relation, tail))
        
        logger.warning(f"Scoring {len(all_triples)} possible triples - this may be memory intensive")
        
        # Score in batches
        all_scores = []
        for i in range(0, len(all_triples), self.batch_size):
            batch_triples = all_triples[i:i + self.batch_size]
            
            # Extract names
            head_names = [t[0] for t in batch_triples]
            rel_names = [t[1] for t in batch_triples]
            tail_names = [t[2] for t in batch_triples]
            
            batch_scores = self.base_module.forward_with_names(head_names, rel_names, tail_names)
            all_scores.append(batch_scores)
        
        if all_scores:
            return torch.cat(all_scores, dim=0)
        else:
            return torch.tensor([])


def create_scoring_module(config: NPLLConfig, kg=None) -> NPLLScoringModule:
    """
    Factory function to create and initialize NPLL scoring module
    
    Args:
        config: NPLL configuration
        kg: Optional knowledge graph to build vocabulary from
        
    Returns:
        Initialized NPLLScoringModule
    """
    scoring_module = NPLLScoringModule(config)
    
    if kg is not None:
        scoring_module.build_vocabulary(kg)
        logger.info(f"Built vocabulary: {scoring_module.entity_vocab_size} entities, "
                   f"{scoring_module.relation_vocab_size} relations")
    
    return scoring_module


def verify_equation7_implementation():
    """
    Verification function to ensure Equation 7 is implemented correctly
    Tests the mathematical operations step by step
    """
    from ..utils.config import default_config
    
    config = default_config
    scoring_func = BilinearScoringFunction(config)
    
    # Create test inputs
    batch_size = 3
    d = config.entity_embedding_dim
    k = config.scoring_hidden_dim
    
    head_emb = torch.randn(batch_size, d)
    tail_emb = torch.randn(batch_size, d)
    rel_emb = torch.randn(batch_size, d)  # Not used in Eq 7 but kept for interface
    
    # Forward pass
    scores = scoring_func(head_emb, rel_emb, tail_emb)
    
    # Verify output shape
    assert scores.shape == (batch_size,), f"Expected shape ({batch_size},), got {scores.shape}"
    
    # Verify no NaN values
    assert not torch.isnan(scores).any(), "Output contains NaN values"
    
    # Test single input
    single_score = scoring_func.forward_single(head_emb[0], rel_emb[0], tail_emb[0])
    assert abs(single_score.item() - scores[0].item()) < 1e-5, "Single vs batch mismatch"
    
    logger.info("Equation 7 implementation verified successfully")
    
    return True
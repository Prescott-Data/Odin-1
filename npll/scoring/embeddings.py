
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple
import logging
from collections import OrderedDict

from ..core import KnowledgeGraph, Entity, Relation, Triple
from ..utils.config import NPLLConfig

logger = logging.getLogger(__name__)


class EntityEmbedding(nn.Module):
    """
    Entity embedding layer with dynamic vocabulary expansion

    """
    
    def __init__(self, config: NPLLConfig, initial_vocab_size: int = 10000):
        super().__init__()
        self.config = config
        self.embedding_dim = config.entity_embedding_dim
        
        # Entity vocabulary mapping: entity_name -> index
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        
        # Reserve indices: 0 padding, 1 OOV; new entries from 2+
        self.padding_idx = 0
        self.oov_idx = 1
        # Embedding layer - will expand dynamically
        self.embedding = nn.Embedding(
            num_embeddings=initial_vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )
        
        # Initialize embeddings using Xavier initialization
        nn.init.xavier_uniform_(self.embedding.weight.data)
        # Set padding embedding to zero; init OOV to a valid vector
        with torch.no_grad():
            self.embedding.weight[self.padding_idx].zero_()
        # Track next available index (0 pad, 1 oov reserved)
        self._next_idx = 2
    
    def add_entity(self, entity_name: str) -> int:
        """Add entity to vocabulary and return its index"""
        if entity_name not in self.entity_to_idx:
            # Check if we need to expand embedding layer
            if self._next_idx >= self.embedding.num_embeddings:
                self._expand_embeddings()
            
            # Add entity to vocabulary
            idx = self._next_idx
            self.entity_to_idx[entity_name] = idx
            self.idx_to_entity[idx] = entity_name
            self._next_idx += 1
            
            logger.debug(f"Added entity '{entity_name}' with index {idx}")
        
        return self.entity_to_idx[entity_name]
    
    def _expand_embeddings(self, grow_by: int = 1000, on_expand=None):
        """Expand embedding layer when vocabulary grows (preserve device/dtype)."""
        old = self.embedding
        old_size = old.num_embeddings
        new_size = max(old_size * 2, self._next_idx + grow_by)
        new = nn.Embedding(
            num_embeddings=new_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
            dtype=old.weight.dtype,
            device=old.weight.device,
        )
        with torch.no_grad():
            new.weight[:old_size].copy_(old.weight)
            nn.init.xavier_uniform_(new.weight[old_size:])
            new.weight[self.padding_idx].zero_()
        self.embedding = new
        if on_expand is not None:
            on_expand(self.embedding)
        logger.info(f"Expanded entity embeddings from {old_size} to {new_size}")
    
    def get_entity_index(self, entity_name: str, add_if_missing: bool = False) -> int:
        """Get index for entity; returns OOV if missing and add_if_missing=False."""
        idx = self.entity_to_idx.get(entity_name)
        if idx is None:
            return self.add_entity(entity_name) if add_if_missing else self.oov_idx
        return idx
    
    def get_entity_name(self, idx: int) -> Optional[str]:
        """Get entity name from index"""
        return self.idx_to_entity.get(idx)
    
    def get_embedding(self, entity_name: str, add_if_missing: bool = False) -> torch.Tensor:
        """Get embedding vector for entity (device-safe)."""
        idx = self.get_entity_index(entity_name, add_if_missing=add_if_missing)
        device = self.embedding.weight.device
        return self.embedding(torch.tensor([idx], dtype=torch.long, device=device)).squeeze(0)
    
    def get_embeddings_batch(self, entity_names: List[str], add_if_missing: bool = False) -> torch.Tensor:
        """Get embedding vectors for batch of entities (device-safe)."""
        indices = [self.get_entity_index(name, add_if_missing=add_if_missing) for name in entity_names]
        device = self.embedding.weight.device
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        return self.embedding(indices_tensor)
    
    def forward(self, entity_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass for entity embeddings"""
        return self.embedding(entity_indices)
    
    @property
    def vocab_size(self) -> int:
        """Current vocabulary size"""
        return len(self.entity_to_idx)
    
    def state_dict_with_vocab(self) -> Dict:
        """Get state dict including vocabulary mappings"""
        state = super().state_dict()
        state['entity_to_idx'] = self.entity_to_idx.copy()
        state['idx_to_entity'] = self.idx_to_entity.copy()
        state['_next_idx'] = self._next_idx
        state['padding_idx'] = self.padding_idx
        state['oov_idx'] = self.oov_idx
        return state
    
    def load_state_dict_with_vocab(self, state_dict: Dict):
        """Load state dict including vocabulary mappings"""
        # Load vocabulary mappings first
        self.entity_to_idx = state_dict.pop('entity_to_idx', {})
        self.idx_to_entity = state_dict.pop('idx_to_entity', {})
        self._next_idx = state_dict.pop('_next_idx', 2)
        self.padding_idx = state_dict.pop('padding_idx', 0)
        self.oov_idx = state_dict.pop('oov_idx', 1)
        needed = max(self._next_idx, self.embedding.num_embeddings)
        if needed > self.embedding.num_embeddings:
            old = self.embedding
            new = nn.Embedding(
                num_embeddings=needed,
                embedding_dim=self.embedding_dim,
                padding_idx=self.padding_idx,
                dtype=old.weight.dtype,
                device=old.weight.device,
            )
            with torch.no_grad():
                new.weight[:old.num_embeddings].copy_(old.weight)
                nn.init.xavier_uniform_(new.weight[old.num_embeddings:])
                new.weight[self.padding_idx].zero_()
            self.embedding = new
        # Load model parameters leniently in case of size diffs
        super().load_state_dict(state_dict, strict=False)


class RelationEmbedding(nn.Module):
    """
    Relation embedding layer with dynamic vocabulary expansion
    """
    
    def __init__(self, config: NPLLConfig, initial_vocab_size: int = 1000):
        super().__init__()
        self.config = config
        self.embedding_dim = config.relation_embedding_dim
        
        # Relation vocabulary mapping
        self.relation_to_idx: Dict[str, int] = {}
        self.idx_to_relation: Dict[int, str] = {}
        
        # Embedding layer
        self.padding_idx = 0
        self.oov_idx = 1
        self.embedding = nn.Embedding(
            num_embeddings=initial_vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.embedding.weight.data)
        with torch.no_grad():
            self.embedding.weight[self.padding_idx].zero_()
        
        self._next_idx = 2
    
    def add_relation(self, relation_name: str) -> int:
        """Add relation to vocabulary and return its index"""
        if relation_name not in self.relation_to_idx:
            if self._next_idx >= self.embedding.num_embeddings:
                self._expand_embeddings()
            
            idx = self._next_idx
            self.relation_to_idx[relation_name] = idx
            self.idx_to_relation[idx] = relation_name
            self._next_idx += 1
            
            logger.debug(f"Added relation '{relation_name}' with index {idx}")
        
        return self.relation_to_idx[relation_name]
    
    def _expand_embeddings(self, grow_by: int = 100, on_expand=None):
        """Expand embedding layer when vocabulary grows (preserve device/dtype)."""
        old = self.embedding
        old_size = old.num_embeddings
        new_size = max(old_size * 2, self._next_idx + grow_by)
        new = nn.Embedding(
            num_embeddings=new_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
            dtype=old.weight.dtype,
            device=old.weight.device,
        )
        with torch.no_grad():
            new.weight[:old_size].copy_(old.weight)
            nn.init.xavier_uniform_(new.weight[old_size:])
            new.weight[self.padding_idx].zero_()
        self.embedding = new
        if on_expand is not None:
            on_expand(self.embedding)
        logger.info(f"Expanded relation embeddings from {old_size} to {new_size}")
    
    def get_relation_index(self, relation_name: str, add_if_missing: bool = False) -> int:
        """Get index for relation; returns OOV if missing and add_if_missing=False."""
        idx = self.relation_to_idx.get(relation_name)
        if idx is None:
            return self.add_relation(relation_name) if add_if_missing else self.oov_idx
        return idx
    
    def get_embedding(self, relation_name: str, add_if_missing: bool = False) -> torch.Tensor:
        """Get embedding vector for relation (device-safe)."""
        idx = self.get_relation_index(relation_name, add_if_missing=add_if_missing)
        device = self.embedding.weight.device
        return self.embedding(torch.tensor([idx], dtype=torch.long, device=device)).squeeze(0)
    
    def get_embeddings_batch(self, relation_names: List[str], add_if_missing: bool = False) -> torch.Tensor:
        """Get embedding vectors for batch of relations (device-safe)."""
        indices = [self.get_relation_index(name, add_if_missing=add_if_missing) for name in relation_names]
        device = self.embedding.weight.device
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        return self.embedding(indices_tensor)
    
    def forward(self, relation_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass for relation embeddings"""
        return self.embedding(relation_indices)
    
    @property
    def vocab_size(self) -> int:
        """Current vocabulary size"""
        return len(self.relation_to_idx)


class EmbeddingManager(nn.Module):
    """
    Manages both entity and relation embeddings
    Handles vocabulary building from knowledge graph
    """
    
    def __init__(self, config: NPLLConfig):
        super().__init__()
        self.config = config
        
        # Entity and relation embedding layers
        self.entity_embeddings = EntityEmbedding(config)
        self.relation_embeddings = RelationEmbedding(config)
        
        # Neural network for updating embeddings (paper Section 4.1)
        self.entity_update_network = nn.Sequential(
            nn.Linear(config.entity_embedding_dim, config.entity_embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.entity_embedding_dim, config.entity_embedding_dim)
        )
    
    def build_vocabulary_from_kg(self, kg: KnowledgeGraph):
        """Build vocabulary from knowledge graph entities and relations"""
        logger.info("Building vocabulary from knowledge graph...")
        
        # Add all entities
        for entity in kg.entities:
            self.entity_embeddings.add_entity(entity.name)
        
        # Add all relations
        for relation in kg.relations:
            self.relation_embeddings.add_relation(relation.name)
        
        logger.info(f"Vocabulary built: {self.entity_embeddings.vocab_size} entities, "
                   f"{self.relation_embeddings.vocab_size} relations")
    
    def get_triple_embeddings(self, triple: Triple, add_if_missing: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get embeddings for a triple (head, relation, tail)
        """
        head_emb = self.entity_embeddings.get_embedding(triple.head.name, add_if_missing=add_if_missing)
        rel_emb = self.relation_embeddings.get_embedding(triple.relation.name, add_if_missing=add_if_missing)
        tail_emb = self.entity_embeddings.get_embedding(triple.tail.name, add_if_missing=add_if_missing)
        
        return head_emb, rel_emb, tail_emb
    
    def get_triple_embeddings_batch(self, triples: List[Triple], add_if_missing: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get embeddings for batch of triples"""
        head_names = [t.head.name for t in triples]
        rel_names = [t.relation.name for t in triples]
        tail_names = [t.tail.name for t in triples]
        
        head_embs = self.entity_embeddings.get_embeddings_batch(head_names, add_if_missing=add_if_missing)
        rel_embs = self.relation_embeddings.get_embeddings_batch(rel_names, add_if_missing=add_if_missing)
        tail_embs = self.entity_embeddings.get_embeddings_batch(tail_names, add_if_missing=add_if_missing)
        
        return head_embs, rel_embs, tail_embs
    
    def update_entity_embeddings(self, entity_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Update entity embeddings using neural network

        """
        return self.entity_update_network(entity_embeddings)
    
    def get_entity_embedding(self, entity_name: str) -> torch.Tensor:
        """Get embedding for single entity"""
        return self.entity_embeddings.get_embedding(entity_name)
    
    def get_relation_embedding(self, relation_name: str) -> torch.Tensor:
        """Get embedding for single relation"""
        return self.relation_embeddings.get_embedding(relation_name)
    
    def get_embeddings_for_scoring(self, head_names: List[str], 
                                 relation_names: List[str], 
                                 tail_names: List[str],
                                 add_if_missing: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get embeddings formatted for scoring module input
    
        """
        # Get base embeddings
        head_embs = self.entity_embeddings.get_embeddings_batch(head_names, add_if_missing=add_if_missing)
        rel_embs = self.relation_embeddings.get_embeddings_batch(relation_names, add_if_missing=add_if_missing)
        tail_embs = self.entity_embeddings.get_embeddings_batch(tail_names, add_if_missing=add_if_missing)
        
        # Update entity embeddings through neural network
        head_embs = self.update_entity_embeddings(head_embs)
        tail_embs = self.update_entity_embeddings(tail_embs)
        
        return head_embs, rel_embs, tail_embs
    
    @property
    def entity_vocab_size(self) -> int:
        """Number of entities in vocabulary"""
        return self.entity_embeddings.vocab_size
    
    @property
    def relation_vocab_size(self) -> int:
        """Number of relations in vocabulary"""
        return self.relation_embeddings.vocab_size

    @property
    def relation_num_groups(self) -> int:
        """Size of relation embedding table (for per-relation temperature groups)."""
        return int(self.relation_embeddings.embedding.num_embeddings)

    def get_relation_indices_batch(self, relation_names: List[str], add_if_missing: bool = False) -> torch.Tensor:
        idxs = [self.relation_embeddings.get_relation_index(n, add_if_missing=add_if_missing) for n in relation_names]
        device = self.relation_embeddings.embedding.weight.device
        return torch.tensor(idxs, dtype=torch.long, device=device)

    def relation_group_ids_for_triples(self, triples: List[Triple], add_if_missing: bool = False) -> torch.Tensor:
        rels = [t.relation.name for t in triples]
        return self.get_relation_indices_batch(rels, add_if_missing=add_if_missing)
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary mappings to file"""
        vocab_data = {
            'entity_to_idx': self.entity_embeddings.entity_to_idx,
            'relation_to_idx': self.relation_embeddings.relation_to_idx,
            'config': self.config
        }
        torch.save(vocab_data, filepath)
        logger.info(f"Saved vocabulary to {filepath}")
    
    def load_vocabulary(self, filepath: str):
        """Load vocabulary mappings from file"""
        vocab_data = torch.load(filepath)
        
        # Load entity vocabulary
        self.entity_embeddings.entity_to_idx = vocab_data['entity_to_idx']
        self.entity_embeddings.idx_to_entity = {
            v: k for k, v in vocab_data['entity_to_idx'].items()
        }
        self.entity_embeddings._next_idx = len(vocab_data['entity_to_idx']) + 1
        
        # Load relation vocabulary
        self.relation_embeddings.relation_to_idx = vocab_data['relation_to_idx']
        self.relation_embeddings.idx_to_relation = {
            v: k for k, v in vocab_data['relation_to_idx'].items()
        }
        self.relation_embeddings._next_idx = len(vocab_data['relation_to_idx']) + 1
        
        logger.info(f"Loaded vocabulary from {filepath}")


def initialize_embeddings_from_pretrained(embedding_manager: EmbeddingManager,
                                        pretrained_entity_embeddings: Optional[Dict[str, torch.Tensor]] = None,
                                        pretrained_relation_embeddings: Optional[Dict[str, torch.Tensor]] = None):
    """
    Initialize embeddings from pretrained vectors 
    """
    if pretrained_entity_embeddings:
        logger.info(f"Initializing {len(pretrained_entity_embeddings)} entity embeddings from pretrained")
        
        with torch.no_grad():
            for entity_name, embedding_vector in pretrained_entity_embeddings.items():
                idx = embedding_manager.entity_embeddings.add_entity(entity_name)
                if embedding_vector.size(0) == embedding_manager.config.entity_embedding_dim:
                    embedding_manager.entity_embeddings.embedding.weight[idx] = embedding_vector
                else:
                    logger.warning(f"Dimension mismatch for entity {entity_name}: "
                                 f"expected {embedding_manager.config.entity_embedding_dim}, "
                                 f"got {embedding_vector.size(0)}")
    
    if pretrained_relation_embeddings:
        logger.info(f"Initializing {len(pretrained_relation_embeddings)} relation embeddings from pretrained")
        
        with torch.no_grad():
            for relation_name, embedding_vector in pretrained_relation_embeddings.items():
                idx = embedding_manager.relation_embeddings.add_relation(relation_name)
                if embedding_vector.size(0) == embedding_manager.config.relation_embedding_dim:
                    embedding_manager.relation_embeddings.embedding.weight[idx] = embedding_vector
                else:
                    logger.warning(f"Dimension mismatch for relation {relation_name}")


# Factory function for creating embedding manager
def create_embedding_manager(config: NPLLConfig, kg: Optional[KnowledgeGraph] = None) -> EmbeddingManager:
    """
    Factory function to create and initialize EmbeddingManager
    
    """
    manager = EmbeddingManager(config)
    
    if kg is not None:
        manager.build_vocabulary_from_kg(kg)
    
    return manager
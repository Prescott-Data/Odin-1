
from typing import Dict, List, Set, Tuple, Optional, Iterator, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=True)
class Entity:
    """Entity representation"""
    name: str
    entity_id: Optional[int] = None
    
    def __str__(self) -> str:
        return self.name
    
    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True, eq=True)  
class Relation:
    """Relation/Predicate representation """
    name: str
    relation_id: Optional[int] = None
    
    def __str__(self) -> str:
        return self.name
    
    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True, eq=True)
class Triple:
    """
    Triple representation: (head, relation, tail) or l(eh, et)
    """
    head: Entity
    relation: Relation  
    tail: Entity
    
    def __str__(self) -> str:
        return f"{self.relation.name}({self.head.name}, {self.tail.name})"
    
    def __hash__(self) -> int:
        return hash((self.head, self.relation, self.tail))
    
    def to_predicate_logic(self) -> str:
        """Convert to predicate logic format: relation(head, tail)"""
        return f"{self.relation.name}({self.head.name}, {self.tail.name})"
    
    def is_valid(self) -> bool:
        """Check if triple has valid components"""
        return all([
            self.head.name.strip(), 
            self.relation.name.strip(), 
            self.tail.name.strip()
        ])


class KnowledgeGraph:
   
    def __init__(self, name: str = "KnowledgeGraph"):
        self.name = name
        
        # Core components as per paper definition
        self._entities: Dict[str, Entity] = {}  # E: entity set
        self._relations: Dict[str, Relation] = {}  # L: relation set
        self._known_facts: Set[Triple] = set()  # F: known facts
        self._unknown_facts: Set[Triple] = set()  # U: unknown facts
        
        # Indexing for efficient access
        self._head_index: Dict[Entity, Set[Triple]] = defaultdict(set)
        self._tail_index: Dict[Entity, Set[Triple]] = defaultdict(set)
        self._relation_index: Dict[Relation, Set[Triple]] = defaultdict(set)
        
        # Statistics
        self._stats = {
            'num_entities': 0,
            'num_relations': 0, 
            'num_known_facts': 0,
            'num_unknown_facts': 0
        }
    
    @property
    def entities(self) -> Set[Entity]:
        """Get entity set E = {e1, e2, ..., eM}"""
        return set(self._entities.values())
    
    @property 
    def relations(self) -> Set[Relation]:
        """Get relation set L = {l1, l2, ..., lN}"""
        return set(self._relations.values())
    
    @property
    def known_facts(self) -> Set[Triple]:
        """Get known facts set F = {f1, f2, ..., fS}"""
        return self._known_facts.copy()
    
    @property
    def unknown_facts(self) -> Set[Triple]:
        """Get unknown facts set U (for inference)"""
        return self._unknown_facts.copy()
    
    @property
    def all_facts(self) -> Set[Triple]:
        """Get all facts F ∪ U"""
        return self._known_facts | self._unknown_facts
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name"""
        return self._entities.get(name)
    
    def get_relation(self, name: str) -> Optional[Relation]:
        """Get relation by name"""
        return self._relations.get(name)
    
    def add_entity(self, name: str) -> Entity:
        """Add entity to E set"""
        if name not in self._entities:
            entity_id = len(self._entities)
            entity = Entity(name=name, entity_id=entity_id)
            self._entities[name] = entity
            self._stats['num_entities'] = len(self._entities)
            logger.debug(f"Added entity: {name}")
        return self._entities[name]
    
    def add_relation(self, name: str) -> Relation:
        """Add relation to L set"""
        if name not in self._relations:
            relation_id = len(self._relations)
            relation = Relation(name=name, relation_id=relation_id)
            self._relations[name] = relation
            self._stats['num_relations'] = len(self._relations)
            logger.debug(f"Added relation: {name}")
        return self._relations[name]
    
    def add_known_fact(self, head: str, relation: str, tail: str) -> Triple:
        """
        Add known fact to F set
        """
        head_entity = self.add_entity(head)
        relation_obj = self.add_relation(relation)
        tail_entity = self.add_entity(tail)
        
        triple = Triple(head=head_entity, relation=relation_obj, tail=tail_entity)
        
        if triple.is_valid() and triple not in self._known_facts:
            self._known_facts.add(triple)
            self._update_indices(triple)
            self._stats['num_known_facts'] = len(self._known_facts)
            logger.debug(f"Added known fact: {triple}")
        
        return triple
    
    def add_unknown_fact(self, head: str, relation: str, tail: str) -> Triple:
        """Add unknown fact to U set (for inference)"""
        head_entity = self.add_entity(head)
        relation_obj = self.add_relation(relation)
        tail_entity = self.add_entity(tail)
        
        triple = Triple(head=head_entity, relation=relation_obj, tail=tail_entity)
        
        if triple.is_valid() and triple not in self._unknown_facts:
            self._unknown_facts.add(triple)
            self._update_indices(triple)
            self._stats['num_unknown_facts'] = len(self._unknown_facts)
            logger.debug(f"Added unknown fact: {triple}")
        
        return triple
    
    def _update_indices(self, triple: Triple):
        """Update internal indices for efficient querying"""
        self._head_index[triple.head].add(triple)
        self._tail_index[triple.tail].add(triple)
        self._relation_index[triple.relation].add(triple)
    
    def get_facts_by_head(self, entity: Entity) -> Set[Triple]:
        """Get all facts with given entity as head"""
        return self._head_index[entity].copy()
    
    def get_facts_by_tail(self, entity: Entity) -> Set[Triple]:
        """Get all facts with given entity as tail"""
        return self._tail_index[entity].copy()
    
    def get_facts_by_relation(self, relation: Relation) -> Set[Triple]:
        """Get all facts with given relation"""
        return self._relation_index[relation].copy()
    
    def contains_fact(self, head: str, relation: str, tail: str) -> bool:
        """Check if fact exists in known facts F"""
        head_entity = self.get_entity(head)
        relation_obj = self.get_relation(relation)
        tail_entity = self.get_entity(tail)
        
        if not all([head_entity, relation_obj, tail_entity]):
            return False
        
        triple = Triple(head=head_entity, relation=relation_obj, tail=tail_entity)
        return triple in self._known_facts
    
    def get_neighbors(self, entity: Entity, relation: Optional[Relation] = None) -> Set[Entity]:
        """Get neighboring entities connected by relation"""
        neighbors = set()
        
        # Outgoing edges (entity as head)
        for triple in self.get_facts_by_head(entity):
            if relation is None or triple.relation == relation:
                neighbors.add(triple.tail)
        
        # Incoming edges (entity as tail) 
        for triple in self.get_facts_by_tail(entity):
            if relation is None or triple.relation == relation:
                neighbors.add(triple.head)
        
        return neighbors
    
    def sample_negative_facts(self, num_samples: int, 
                            corruption_mode: str = "both") -> List[Triple]:
        """
        Generate negative facts for training
        Corruption modes: 'head', 'tail', 'both'
        """
        negative_facts = []
        entities_list = list(self.entities)
        
        for _ in range(num_samples):
            # Sample a known fact to corrupt
            known_fact = next(iter(self._known_facts))
            
            if corruption_mode == "head" or (corruption_mode == "both" and len(negative_facts) % 2 == 0):
                # Corrupt head entity
                corrupt_head = entities_list[hash(known_fact) % len(entities_list)]
                negative_triple = Triple(corrupt_head, known_fact.relation, known_fact.tail)
            else:
                # Corrupt tail entity  
                corrupt_tail = entities_list[hash(known_fact) % len(entities_list)]
                negative_triple = Triple(known_fact.head, known_fact.relation, corrupt_tail)
            
            # Ensure it's actually negative
            if negative_triple not in self._known_facts:
                negative_facts.append(negative_triple)
        
        return negative_facts
    
    def get_statistics(self) -> Dict[str, int]:
        """Get knowledge graph statistics"""
        return self._stats.copy()
    
    def __str__(self) -> str:
        return (f"KnowledgeGraph({self.name}): "
                f"{self._stats['num_entities']} entities, "
                f"{self._stats['num_relations']} relations, "
                f"{self._stats['num_known_facts']} known facts")
    
    def __repr__(self) -> str:
        return self.__str__()

    # --- Serialization helpers for robust save/load ---
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the knowledge graph to a portable dict.
        Entities/relations are referenced by name; facts are triplets of names.
        """
        return {
            'name': self.name,
            'entities': [e.name for e in self.entities],
            'relations': [r.name for r in self.relations],
            'known_facts': [
                (t.head.name, t.relation.name, t.tail.name) for t in self._known_facts
            ],
            'unknown_facts': [
                (t.head.name, t.relation.name, t.tail.name) for t in self._unknown_facts
            ],
        }

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'KnowledgeGraph':
        """
        Deserialize a knowledge graph previously produced by serialize().
        """
        kg = KnowledgeGraph(name=data.get('name', 'KnowledgeGraph'))
        # Pre-create entities and relations
        for e in data.get('entities', []):
            kg.add_entity(e)
        for r in data.get('relations', []):
            kg.add_relation(r)
        for h, r, t in data.get('known_facts', []):
            kg.add_known_fact(h, r, t)
        for h, r, t in data.get('unknown_facts', []):
            kg.add_unknown_fact(h, r, t)
        return kg


def load_knowledge_graph_from_triples(triples: List[Tuple[str, str, str]], 
                                    name: str = "LoadedKG") -> KnowledgeGraph:
    """
    Load knowledge graph from list of (head, relation, tail) tuples
    """
    kg = KnowledgeGraph(name=name)
    
    for head, relation, tail in triples:
        kg.add_known_fact(head, relation, tail)
    
    logger.info(f"Loaded knowledge graph: {kg}")
    return kg
from __future__ import annotations
from typing import List, Tuple, Dict, Set
import random
from collections import defaultdict

from npll.core.knowledge_graph import KnowledgeGraph, Entity, Relation, Triple

# Type alias for node identifiers
NodeId = str

class KGDataGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(self.seed)

    def generate_random_kg(self, num_entities: int, num_relations: int, num_facts: int) -> KnowledgeGraph:
        kg = KnowledgeGraph(name="RandomKG")

        entity_names = [f"E{i}" for i in range(num_entities)]
        relation_names = [f"R{i}" for i in range(num_relations)]

        for name in entity_names:
            kg.add_entity(name)
        for name in relation_names:
            kg.add_relation(name)

        entities = list(kg.entities)
        relations = list(kg.relations)

        for _ in range(num_facts):
            head = random.choice(entities).name
            relation = random.choice(relations).name
            tail = random.choice(entities).name
            kg.add_known_fact(head, relation, tail)
        
        return kg

    def generate_path_kg(self, length: int, entity_prefix: str = "P", relation_name: str = "connected_to") -> KnowledgeGraph:
        kg = KnowledgeGraph(name="PathKG")
        last_entity = None
        for i in range(length):
            current_entity = f"{entity_prefix}{i}"
            kg.add_entity(current_entity)
            if last_entity:
                kg.add_known_fact(last_entity, relation_name, current_entity)
            last_entity = current_entity
        return kg

    def generate_star_kg(self, num_spokes: int, center_entity: str = "Center", relation_name: str = "has_spoke") -> KnowledgeGraph:
        kg = KnowledgeGraph(name="StarKG")
        kg.add_entity(center_entity)
        for i in range(num_spokes):
            spoke_entity = f"Spoke{i}"
            kg.add_entity(spoke_entity)
            kg.add_known_fact(center_entity, relation_name, spoke_entity)
        return kg

    def generate_community_kg(self, num_communities: int, entities_per_community: int, intra_community_facts: int, inter_community_facts: int) -> Tuple[KnowledgeGraph, Dict[str, Set[NodeId]]]:
        kg = KnowledgeGraph(name="CommunityKG")
        community_nodes: Dict[str, Set[NodeId]] = defaultdict(set)
        all_entities = []
        
        # Generate entities and relations
        for c_idx in range(num_communities):
            community_id = f"C{c_idx}"
            for e_idx in range(entities_per_community):
                entity_name = f"E{c_idx}_{e_idx}"
                kg.add_entity(entity_name)
                community_nodes[community_id].add(entity_name)
                all_entities.append(entity_name)
        
        kg.add_relation("intra_rel")
        kg.add_relation("inter_rel")
        kg.add_relation("generic_rel")
        relations = list(kg.relations)

        # Intra-community facts
        for c_idx in range(num_communities):
            nodes_in_comm = list(community_nodes[f"C{c_idx}"])
            for _ in range(intra_community_facts):
                if len(nodes_in_comm) < 2: continue
                h, t = random.sample(nodes_in_comm, 2)
                r = random.choice([rel for rel in relations if rel.name == "intra_rel" or rel.name == "generic_rel"]).name
                kg.add_known_fact(h, r, t)

        # Inter-community facts
        for _ in range(inter_community_facts):
            if len(all_entities) < 2: continue
            h, t = random.sample(all_entities, 2)
            r = random.choice([rel for rel in relations if rel.name == "inter_rel" or rel.name == "generic_rel"]).name
            kg.add_known_fact(h, r, t)
            
        return kg, community_nodes
